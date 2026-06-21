"""
Deep Ensemble baseline (Lakshminarayanan et al. 2017).

Method:
  - Train M independent MLPs, each from a different random seed.
  - Each network predicts the residual mean; aleatoric noise comes from the
    per-pair Deffrennes proxy (same σ as GP and BNN baselines).
  - Epistemic uncertainty = std of ensemble predictions.
  - Total uncertainty:  σ_total = sqrt(σ_epistemic² + σ_noise²)

Architecture: MLP 15 → 64 → 64 → 1  (ReLU, same as BNN for fair comparison)
Ensemble size: M = 5 (default; Lakshminarayanan et al. show diminishing returns beyond 5)

Evaluation: LOSO CV, same metrics as RF, BNN, and GP.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.gp.data_prep import FEATURE_NAMES, load_gp_data


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Deterministic MLP: 15 → 64 → 64 → 1."""

    def __init__(self, n_features: int = 15, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _nll_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    noise_var: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL with fixed per-point heteroskedastic noise."""
    return 0.5 * ((target - pred) ** 2 / noise_var + torch.log(2 * math.pi * noise_var)).mean()


def train_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    noise_var_train: torch.Tensor,
    n_features: int = 15,
    hidden: int = 64,
    lr: float = 0.01,
    n_epochs: int = 1500,
    random_state: int = 42,
) -> MLP:
    torch.manual_seed(random_state)
    model     = MLP(n_features, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = _nll_loss(model(x_train), y_train, noise_var_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_ensemble(
    models: list[MLP],
    x: torch.Tensor,
    noise_std: float,
) -> tuple[float, float, float]:
    """
    Returns (pred_mean, std_total, std_epistemic) in residual space [kJ/mol].

    Law of total variance:
        σ²_epistemic = Var[μ_i]  (variance across ensemble means)
        σ²_total     = σ²_epistemic + σ²_noise
    """
    preds = torch.stack([m(x) for m in models], dim=0)   # (M,)
    mean           = float(preds.mean())
    std_epistemic  = float(preds.std())
    std_total      = math.sqrt(std_epistemic ** 2 + noise_std ** 2)
    return mean, std_total, std_epistemic


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso(
    df: pd.DataFrame | None = None,
    n_members: int = 5,
    hidden: int = 64,
    lr: float = 0.01,
    n_epochs: int = 1500,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-One-System-Out CV for Deep Ensemble baseline."""
    if df is None:
        df = load_gp_data()

    X_raw = df[FEATURE_NAMES].values.astype(np.float32)
    y_raw = df["residual"].values.astype(np.float32)
    noise = df["noise_kJmol"].values.astype(np.float32)

    N    = len(df)
    rows: list[dict] = []

    print(
        f"Deep Ensemble LOSO CV on {N} pairs "
        f"(M={n_members}, hidden={hidden}, epochs={n_epochs}) …"
    )

    for i in range(N):
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False

        scaler     = StandardScaler()
        X_train_np = scaler.fit_transform(X_raw[train_mask])
        X_test_np  = scaler.transform(X_raw[[i]])

        y_train_np  = y_raw[train_mask]
        noise_train = noise[train_mask]
        noise_i     = float(noise[i])

        x_tr  = torch.tensor(X_train_np,        dtype=torch.float32)
        y_tr  = torch.tensor(y_train_np,         dtype=torch.float32)
        nv_tr = torch.tensor(noise_train ** 2,   dtype=torch.float32)
        x_te  = torch.tensor(X_test_np,          dtype=torch.float32)

        models = [
            train_mlp(
                x_tr, y_tr, nv_tr,
                n_features=len(FEATURE_NAMES),
                hidden=hidden,
                lr=lr,
                n_epochs=n_epochs,
                random_state=random_state + i * n_members + m,
            )
            for m in range(n_members)
        ]

        pred_mean, std_total, std_epistemic = predict_ensemble(models, x_te, noise_i)

        row      = df.iloc[i]
        res_true = float(row["residual"])

        rows.append({
            "pair":                   row["pair"],
            "h05_true":               float(row["h05"]),
            "h05_miedema":            float(row["h05_miedema"]),
            "h05_de":                 float(row["h05_miedema"]) + pred_mean,
            "h05_de_std":             std_total,
            "residual_true":          res_true,
            "residual_pred":          pred_mean,
            "residual_std":           std_total,
            "residual_std_epistemic": std_epistemic,
            "noise_kJmol":            noise_i,
            "in_ci95":                abs(res_true - pred_mean) <= 1.96 * std_total,
        })

        if verbose and ((i + 1) % 20 == 0 or i == N - 1):
            mae_so_far = np.mean([abs(r["h05_true"] - r["h05_de"]) for r in rows])
            print(
                f"  [{i+1:3d}/{N}] {row['pair']:10s} | "
                f"pred {rows[-1]['h05_de']:+7.2f} | "
                f"true {rows[-1]['h05_true']:+7.2f} | "
                f"σ_ep {std_epistemic:.2f}  σ_tot {std_total:.2f} | "
                f"running MAE {mae_so_far:.3f}"
            )

    out = pd.DataFrame(rows)
    out["abs_err_miedema"] = (out["h05_true"] - out["h05_miedema"]).abs()
    out["abs_err_de"]      = (out["h05_true"] - out["h05_de"]).abs()
    return out


# ---------------------------------------------------------------------------
# Metrics & reporting
# ---------------------------------------------------------------------------

def compute_metrics(loso_df: pd.DataFrame) -> dict[str, float]:
    from scipy.stats import norm

    err  = loso_df["h05_true"] - loso_df["h05_de"]
    mae  = float(loso_df["abs_err_de"].mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    r2   = float(
        1 - (err ** 2).sum() /
        ((loso_df["h05_true"] - loso_df["h05_true"].mean()) ** 2).sum()
    )
    ci95 = float(loso_df["in_ci95"].mean() * 100)

    z        = (loso_df["residual_true"] - loso_df["residual_pred"]).values / \
               loso_df["residual_std"].values
    alphas   = np.linspace(0.1, 0.99, 10)
    observed = [(np.abs(z) <= norm.ppf((1 + a) / 2)).mean() for a in alphas]
    ece      = float(np.mean(np.abs(np.array(observed) - alphas)))

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "CI95": ci95, "ECE": ece}


def print_summary(
    loso_df: pd.DataFrame,
    gp_csv: str | None = None,
    rf_csv: str | None = None,
    bnn_csv: str | None = None,
) -> None:
    from pathlib import Path
    from scipy.stats import norm

    m = compute_metrics(loso_df)

    print("\n=== Deep Ensemble LOSO Summary ===")
    print(f"  Miedema MAE         : {loso_df['abs_err_miedema'].mean():.3f} kJ/mol")
    print(f"  DE      MAE         : {m['MAE']:.3f} kJ/mol")
    print(f"  DE      RMSE        : {m['RMSE']:.3f} kJ/mol")
    print(f"  DE      R²          : {m['R2']:.4f}")
    print(f"  DE mean σ_total     : {loso_df['h05_de_std'].mean():.3f} kJ/mol")
    print(f"  DE mean σ_epistemic : {loso_df['residual_std_epistemic'].mean():.3f} kJ/mol")
    print(f"  DE mean σ_noise     : {loso_df['noise_kJmol'].mean():.3f} kJ/mol")
    print(f"  95% CI coverage     : {m['CI95']:.1f}%  (target: 95%)")
    print(f"  ECE                 : {m['ECE']:.4f}")

    if not (gp_csv and rf_csv and bnn_csv):
        return
    if not all(Path(p).exists() for p in [gp_csv, rf_csv, bnn_csv]):
        return

    gp  = pd.read_csv(gp_csv)
    rf  = pd.read_csv(rf_csv)
    bnn = pd.read_csv(bnn_csv)

    def _m(df_in, pred_col, std_col, err_col):
        err  = df_in["h05_true"] - df_in[pred_col]
        mae  = float(df_in[err_col].mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        r2   = float(1 - (err**2).sum() /
                     ((df_in["h05_true"] - df_in["h05_true"].mean())**2).sum())
        ci   = (df_in["residual_true"] - df_in["residual_pred"]).abs() <= \
               1.96 * df_in[std_col]
        z    = (df_in["residual_true"] - df_in["residual_pred"]).values / \
               df_in[std_col].values
        obs  = [(np.abs(z) <= norm.ppf((1+a)/2)).mean()
                for a in np.linspace(0.1, 0.99, 10)]
        ece  = float(np.mean(np.abs(np.array(obs) - np.linspace(0.1, 0.99, 10))))
        return mae, rmse, r2, float(ci.mean()*100), ece

    gp_m  = _m(gp,  "h05_gp",  "residual_std",     "abs_err_gp")
    rf_m  = _m(rf,  "h05_rf",  "residual_std",     "abs_err_rf")
    bnn_m = _m(bnn, "h05_bnn", "residual_std",     "abs_err_bnn")
    mied_mae  = float(loso_df["abs_err_miedema"].mean())
    mied_rmse = float(np.sqrt(((gp["h05_true"] - gp["h05_miedema"])**2).mean()))

    print("\n=== Comparison: Miedema | RF | BNN | DE | GP ===")
    print(f"  {'Metric':22s} {'Miedema':>9} {'RF':>9} {'BNN':>9} {'DE':>9} {'GP':>9}")
    print(f"  {'-'*70}")
    print(f"  {'MAE  (kJ/mol)':22s} {mied_mae:>9.3f} {rf_m[0]:>9.3f} {bnn_m[0]:>9.3f} {m['MAE']:>9.3f} {gp_m[0]:>9.3f}")
    print(f"  {'RMSE (kJ/mol)':22s} {mied_rmse:>9.3f} {rf_m[1]:>9.3f} {bnn_m[1]:>9.3f} {m['RMSE']:>9.3f} {gp_m[1]:>9.3f}")
    print(f"  {'R²':22s} {'—':>9} {rf_m[2]:>9.4f} {bnn_m[2]:>9.4f} {m['R2']:>9.4f} {gp_m[2]:>9.4f}")
    print(f"  {'95% CI coverage':22s} {'—':>9} {rf_m[3]:>9.1f} {bnn_m[3]:>9.1f} {m['CI95']:>9.1f} {gp_m[3]:>9.1f}")
    print(f"  {'ECE':22s} {'—':>9} {rf_m[4]:>9.4f} {bnn_m[4]:>9.4f} {m['ECE']:>9.4f} {gp_m[4]:>9.4f}")
