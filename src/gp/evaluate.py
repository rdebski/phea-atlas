"""
LOSO CV, post-hoc temperature scaling, and calibration metrics
for the heteroskedastic Matérn GP.

Pipeline
--------
1. run_loso()              → raw LOSO predictions (may be slightly overconfident)
2. calibrate_temperature() → find scalar T minimising NLL on LOSO holdouts
3. apply_temperature()     → rescale std; recompute coverage / ECE
4. print_summary()         → full metrics table
5. reliability_diagram()   → calibration plot data
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .data_prep import GPDataset, load_gp_data
from .train import train_gp


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class LOSOResult:
    pair: str
    h05_true: float
    h05_miedema: float
    h05_gp: float             # h05_miedema + GP residual mean  [kJ/mol]
    h05_gp_std: float         # GP posterior std (model + noise) [kJ/mol]
    h05_gp_std_model: float   # GP posterior std (model only)    [kJ/mol]
    residual_true: float
    residual_pred: float
    residual_std: float       # used for calibration
    noise_kJmol: float
    in_ci95_raw: bool         # before temperature scaling
    in_ci95_cal: bool         # after  temperature scaling (filled by apply_temperature)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
    test_x: torch.Tensor,
    test_noise_var: torch.Tensor,
    dataset: GPDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, std_total, std_model) in original kJ/mol residual space."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_pred   = model(test_x)
        obs_pred = likelihood(f_pred, noise=test_noise_var)

    mean      = dataset.unscale_y(obs_pred.mean).numpy()
    std_total = obs_pred.stddev.numpy() * dataset.y_std
    std_model = f_pred.stddev.numpy()   * dataset.y_std
    return mean, std_total, std_model


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso(
    df: pd.DataFrame | None = None,
    n_iter: int = 300,
    lr: float = 0.05,
    verbose: bool = False,
) -> pd.DataFrame:
    """Leave-One-System-Out CV.  Returns raw (uncalibrated) LOSO DataFrame."""
    if df is None:
        df = load_gp_data()

    pairs = df["pair"].values
    n     = len(df)
    results: list[LOSOResult] = []

    print(f"LOSO CV on {n} pairs …")
    for i, pair in enumerate(pairs):
        train_idx = np.array([j for j in range(n) if j != i])
        test_idx  = np.array([i])

        ds = GPDataset(df, train_idx, test_idx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, likelihood, _ = train_gp(
                ds.train_x, ds.train_y, ds.train_noise_var,
                n_iter=n_iter, lr=lr, verbose=False,
            )

        res_mean, res_std, res_std_model = predict(
            model, likelihood, ds.test_x, ds.test_noise_var, ds
        )

        row = df.iloc[i]
        res_true = float(row["residual"])
        r = LOSOResult(
            pair             = pair,
            h05_true         = float(row["h05"]),
            h05_miedema      = float(row["h05_miedema"]),
            h05_gp           = float(row["h05_miedema"]) + float(res_mean[0]),
            h05_gp_std       = float(res_std[0]),
            h05_gp_std_model = float(res_std_model[0]),
            residual_true    = res_true,
            residual_pred    = float(res_mean[0]),
            residual_std     = float(res_std[0]),
            noise_kJmol      = float(row["noise_kJmol"]),
            in_ci95_raw      = abs(res_true - float(res_mean[0])) <= 1.96 * float(res_std[0]),
            in_ci95_cal      = False,  # filled by apply_temperature
        )
        results.append(r)

        if (i + 1) % 20 == 0 or verbose:
            mae_so_far = np.mean([abs(r.h05_true - r.h05_gp) for r in results])
            print(
                f"  [{i+1:3d}/{n}] {pair:10s} | "
                f"pred {r.h05_gp:+7.2f} | true {r.h05_true:+7.2f} | "
                f"std {r.h05_gp_std:.2f} | running MAE {mae_so_far:.3f}"
            )

    return _to_df(results)


# ---------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration)
# ---------------------------------------------------------------------------

def calibrate_temperature(loso_df: pd.DataFrame) -> float:
    """
    Find scalar T ≥ 1 that minimises the average Gaussian NLL on LOSO holdouts.

    std_calibrated = T × std_raw
    """
    z_raw = (
        (loso_df["residual_true"] - loso_df["residual_pred"]).values /
        loso_df["residual_std"].values
    )

    def neg_nll(log_T: float) -> float:
        T     = np.exp(log_T)
        z_cal = z_raw / T
        return float(-norm.logpdf(z_cal).mean() + np.log(T))

    result = minimize_scalar(neg_nll, bounds=(-1.0, 2.0), method="bounded")
    T = float(np.exp(result.x))
    return max(T, 1.0)   # calibration should only widen intervals


def apply_temperature(loso_df: pd.DataFrame, T: float) -> pd.DataFrame:
    """Return a copy of loso_df with calibrated std and CI flag."""
    df = loso_df.copy()
    df["residual_std_cal"] = df["residual_std"] * T
    df["h05_gp_std_cal"]   = df["h05_gp_std"]  * T
    df["in_ci95_cal"]      = (
        (df["residual_true"] - df["residual_pred"]).abs() <= 1.96 * df["residual_std_cal"]
    )
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _to_df(results: list[LOSOResult]) -> pd.DataFrame:
    df = pd.DataFrame([vars(r) for r in results])
    df["abs_err_miedema"] = (df["h05_true"] - df["h05_miedema"]).abs()
    df["abs_err_gp"]      = (df["h05_true"] - df["h05_gp"]).abs()
    return df


def print_summary(loso_df: pd.DataFrame, T: float | None = None) -> None:
    err_mied = loso_df["h05_true"] - loso_df["h05_miedema"]
    err_gp   = loso_df["h05_true"] - loso_df["h05_gp"]

    mae_mied  = loso_df["abs_err_miedema"].mean()
    mae_gp    = loso_df["abs_err_gp"].mean()
    rmse_mied = float(np.sqrt((err_mied ** 2).mean()))
    rmse_gp   = float(np.sqrt((err_gp   ** 2).mean()))
    r2_gp     = float(
        1 - (err_gp**2).sum() /
        ((loso_df["h05_true"] - loso_df["h05_true"].mean())**2).sum()
    )
    ci95_raw  = loso_df["in_ci95_raw"].mean() * 100

    print("\n=== LOSO Summary ===")
    print(f"  Miedema raw   MAE  : {mae_mied:.3f} kJ/mol")
    print(f"  Miedema raw   RMSE : {rmse_mied:.3f} kJ/mol")
    print(f"  GP            MAE  : {mae_gp:.3f} kJ/mol")
    print(f"  GP            RMSE : {rmse_gp:.3f} kJ/mol")
    print(f"  GP            R²   : {r2_gp:.4f}")
    print(f"  GP mean std (raw)  : {loso_df['h05_gp_std'].mean():.3f} kJ/mol")
    print(f"  95%CI (raw)        : {ci95_raw:.1f}%")

    if T is not None and "in_ci95_cal" in loso_df.columns:
        ci95_cal = loso_df["in_ci95_cal"].mean() * 100
        std_cal  = loso_df["h05_gp_std_cal"].mean() if "h05_gp_std_cal" in loso_df.columns \
                   else loso_df["h05_gp_std"].mean() * T
        print(f"  Temperature T      : {T:.4f}")
        print(f"  GP mean std (cal)  : {std_cal:.3f} kJ/mol")
        print(f"  95%CI (calibrated) : {ci95_cal:.1f}%  (target: 95%)")

    print(f"  Max |error| GP     : {loso_df['abs_err_gp'].max():.2f} kJ/mol")
    print(f"  Pairs              : {len(loso_df)}")


def reliability_diagram(
    loso_df: pd.DataFrame,
    std_col: str = "residual_std",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Coverage at each confidence level α ∈ [0.1, 0.99].

    std_col : which std column to use ('residual_std' raw or 'residual_std_cal' calibrated)
    """
    z = (
        (loso_df["residual_true"] - loso_df["residual_pred"]).values /
        loso_df[std_col].values
    )

    alphas   = np.linspace(0.1, 0.99, n_bins)
    observed = [(np.abs(z) <= norm.ppf((1 + a) / 2)).mean() for a in alphas]

    calib = pd.DataFrame({"alpha": alphas, "expected": alphas, "observed": observed})
    calib["miscalibration"] = calib["observed"] - calib["expected"]

    ece = float(np.mean(np.abs(calib["miscalibration"])))
    label = "calibrated" if "cal" in std_col else "raw"
    print(f"\nECE ({label}): {ece:.4f}")
    return calib
