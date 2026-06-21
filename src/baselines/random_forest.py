"""
Random Forest baseline with bootstrap uncertainty for h05 residual prediction.

Design
------
- Same training data as GP: 224 metallic pairs from Deffrennes (semimetals excluded)
- Same features: 15 symmetric descriptors (FEATURE_NAMES from data_prep)
- Same target: residual = h05_exp - h05_Miedema  [kJ/mol]
- Same evaluation: LOSO cross-validation

Uncertainty decomposition
--------------------------
  σ²_total = σ²_epistemic + σ²_aleatoric

  σ_epistemic  = std of individual tree predictions (bootstrap variance)
  σ_aleatoric  = per-pair noise_kJmol from Deffrennes data density
                 (same as GP FixedNoise: σ_i = 1.5 / sqrt(n_pts / 99))

This mirrors the GP total posterior std, enabling direct ECE comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.gp.data_prep import FEATURE_NAMES, load_gp_data


def run_loso(
    df: pd.DataFrame | None = None,
    n_estimators: int = 500,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Leave-One-System-Out CV for Random Forest baseline.

    Parameters
    ----------
    df           : pre-loaded GP dataset (calls load_gp_data() if None)
    n_estimators : number of trees (500 gives stable bootstrap variance estimates)
    random_state : for reproducibility
    verbose      : print progress every 20 pairs

    Returns
    -------
    DataFrame with columns mirroring gp_loso_metals.csv for direct comparison:
        pair, h05_true, h05_miedema, h05_rf, h05_rf_std,
        residual_true, residual_pred, residual_std, residual_std_epistemic,
        noise_kJmol, in_ci95, abs_err_miedema, abs_err_rf
    """
    if df is None:
        df = load_gp_data()

    X = df[FEATURE_NAMES].values       # (N, 15) — no scaling needed for RF
    y = df["residual"].values           # (N,)    — target: h05_exp - h05_mied
    noise = df["noise_kJmol"].values    # (N,)    — aleatoric σ per pair

    N = len(df)
    rows: list[dict] = []

    print(f"RF LOSO CV on {N} pairs  (n_estimators={n_estimators}) …")

    for i in range(N):
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X[train_mask], y[train_mask])

        # Bootstrap uncertainty: std across individual tree predictions
        x_test = X[[i]]
        tree_preds = np.array([t.predict(x_test)[0] for t in rf.estimators_])

        pred_mean        = float(tree_preds.mean())
        std_epistemic    = float(tree_preds.std())
        noise_i          = float(noise[i])
        std_total        = float(np.sqrt(std_epistemic ** 2 + noise_i ** 2))

        row = df.iloc[i]
        res_true = float(row["residual"])

        rows.append({
            "pair":                   row["pair"],
            "h05_true":               float(row["h05"]),
            "h05_miedema":            float(row["h05_miedema"]),
            "h05_rf":                 float(row["h05_miedema"]) + pred_mean,
            "h05_rf_std":             std_total,
            "residual_true":          res_true,
            "residual_pred":          pred_mean,
            "residual_std":           std_total,
            "residual_std_epistemic": std_epistemic,
            "noise_kJmol":            noise_i,
            "in_ci95":                abs(res_true - pred_mean) <= 1.96 * std_total,
        })

        if verbose and ((i + 1) % 20 == 0 or i == N - 1):
            mae_so_far = np.mean([abs(r["h05_true"] - r["h05_rf"]) for r in rows])
            print(
                f"  [{i+1:3d}/{N}] {row['pair']:10s} | "
                f"pred {rows[-1]['h05_rf']:+7.2f} | "
                f"true {rows[-1]['h05_true']:+7.2f} | "
                f"σ_ep {std_epistemic:.2f}  σ_tot {std_total:.2f} | "
                f"running MAE {mae_so_far:.3f}"
            )

    out = pd.DataFrame(rows)
    out["abs_err_miedema"] = (out["h05_true"] - out["h05_miedema"]).abs()
    out["abs_err_rf"]      = (out["h05_true"] - out["h05_rf"]).abs()
    return out


def compute_metrics(loso_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute MAE, RMSE, R², 95% CI coverage, ECE for RF LOSO results.

    ECE uses the same reliability-diagram approach as evaluate.py:
        z_i = (residual_true_i - residual_pred_i) / residual_std_i
        ECE = mean_α |observed_coverage(α) - α|  for α ∈ [0.1, 0.99]
    """
    from scipy.stats import norm

    err    = loso_df["h05_true"] - loso_df["h05_rf"]
    mae    = float(loso_df["abs_err_rf"].mean())
    rmse   = float(np.sqrt((err ** 2).mean()))
    r2     = float(
        1 - (err ** 2).sum() /
        ((loso_df["h05_true"] - loso_df["h05_true"].mean()) ** 2).sum()
    )
    ci95   = float(loso_df["in_ci95"].mean() * 100)

    z = (
        (loso_df["residual_true"] - loso_df["residual_pred"]).values /
        loso_df["residual_std"].values
    )
    alphas   = np.linspace(0.1, 0.99, 10)
    observed = [(np.abs(z) <= norm.ppf((1 + a) / 2)).mean() for a in alphas]
    ece      = float(np.mean(np.abs(np.array(observed) - alphas)))

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "CI95": ci95, "ECE": ece}


def print_summary(loso_df: pd.DataFrame, gp_csv: str | None = None) -> None:
    """Print RF metrics and, if gp_csv provided, a side-by-side comparison."""
    from pathlib import Path

    metrics = compute_metrics(loso_df)

    print("\n=== RF LOSO Summary ===")
    print(f"  Miedema MAE        : {loso_df['abs_err_miedema'].mean():.3f} kJ/mol")
    print(f"  RF      MAE        : {metrics['MAE']:.3f} kJ/mol")
    print(f"  RF      RMSE       : {metrics['RMSE']:.3f} kJ/mol")
    print(f"  RF      R²         : {metrics['R2']:.4f}")
    print(f"  RF mean σ_total    : {loso_df['h05_rf_std'].mean():.3f} kJ/mol")
    print(f"  RF mean σ_epistemic: {loso_df['residual_std_epistemic'].mean():.3f} kJ/mol")
    print(f"  RF mean σ_noise    : {loso_df['noise_kJmol'].mean():.3f} kJ/mol")
    print(f"  95% CI coverage    : {metrics['CI95']:.1f}%  (target: 95%)")
    print(f"  ECE                : {metrics['ECE']:.4f}")

    if gp_csv and Path(gp_csv).exists():
        gp = pd.read_csv(gp_csv)
        gp_err  = gp["h05_true"] - gp["h05_gp"]
        gp_mae  = float(gp["abs_err_gp"].mean())
        gp_rmse = float(np.sqrt((gp_err ** 2).mean()))
        gp_r2   = float(
            1 - (gp_err ** 2).sum() /
            ((gp["h05_true"] - gp["h05_true"].mean()) ** 2).sum()
        )
        gp_ci95 = float(gp["in_ci95_cal"].mean() * 100) \
                  if "in_ci95_cal" in gp.columns else float(gp["in_ci95_raw"].mean() * 100)

        from scipy.stats import norm as sp_norm
        # Use raw (uncalibrated) GP std for ECE — matches CLAUDE.md value and is
        # fair to RF which has no post-hoc temperature scaling.
        z_gp    = (gp["residual_true"] - gp["residual_pred"]).values / gp["residual_std"].values
        alphas  = np.linspace(0.1, 0.99, 10)
        obs_gp  = [(np.abs(z_gp) <= sp_norm.ppf((1 + a) / 2)).mean() for a in alphas]
        gp_ece  = float(np.mean(np.abs(np.array(obs_gp) - alphas)))

        print("\n=== Comparison: Miedema baseline | RF | GP ===")
        print(f"  {'Metric':20s} {'Miedema':>10} {'RF':>10} {'GP':>10}")
        print(f"  {'-'*52}")
        mied_mae  = float(loso_df["abs_err_miedema"].mean())
        mied_rmse = float(np.sqrt(((gp["h05_true"] - gp["h05_miedema"]) ** 2).mean()))
        print(f"  {'MAE  (kJ/mol)':20s} {mied_mae:>10.3f} {metrics['MAE']:>10.3f} {gp_mae:>10.3f}")
        print(f"  {'RMSE (kJ/mol)':20s} {mied_rmse:>10.3f} {metrics['RMSE']:>10.3f} {gp_rmse:>10.3f}")
        print(f"  {'R²':20s} {'—':>10} {metrics['R2']:>10.4f} {gp_r2:>10.4f}")
        print(f"  {'95% CI coverage':20s} {'—':>10} {metrics['CI95']:>10.1f} {gp_ci95:>10.1f}")
        print(f"  {'ECE':20s} {'—':>10} {metrics['ECE']:>10.4f} {gp_ece:>10.4f}")
