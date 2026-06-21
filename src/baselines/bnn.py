"""
Bayesian Neural Network baseline with mean-field Variational Inference.

Method: Bayes by Backprop (Blundell et al. 2015)
  - Weight prior:        N(0, prior_std²) on all weights and biases
  - Posterior approx.:  factored Gaussian q(w) = N(μ_w, softplus(ρ_w)²)
  - Training objective: ELBO = E_q[log p(y|f)] − KL[q(w) ‖ p(w)] / N_train
  - KL warm-up:         KL weight linearly increases 0→1 over first 200 epochs
                        (prevents posterior collapse on small datasets)

Architecture: MLP 15 → 64 → 64 → 1  (ReLU activations)

Uncertainty decomposition (same as RF baseline):
  σ²_total = σ²_epistemic + σ²_aleatoric
  σ_epistemic = std of predictions over MC weight samples
  σ_aleatoric = per-pair noise_kJmol from Deffrennes data density

Evaluation: LOSO CV, same metrics as RF and GP (MAE, RMSE, R², ECE, 95% CI).
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
# Bayesian layers
# ---------------------------------------------------------------------------

def _softplus(rho: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(rho))


def _kl_gaussian(mu: torch.Tensor, std: torch.Tensor, prior_std: float) -> torch.Tensor:
    """KL[N(μ, σ²) ‖ N(0, prior_std²)], summed over all elements."""
    var       = std ** 2
    prior_var = prior_std ** 2
    return 0.5 * (var / prior_var + mu ** 2 / prior_var - 1.0
                  + math.log(prior_var) - torch.log(var)).sum()


class BayesianLinear(nn.Module):
    """Linear layer with Gaussian variational posterior (mean-field)."""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.prior_std = prior_std

        # Variational parameters
        self.weight_mu  = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -4.0))
        self.bias_mu    = nn.Parameter(torch.zeros(out_features))
        self.bias_rho   = nn.Parameter(torch.full((out_features,), -4.0))

        # Xavier init for means (better starting point)
        nn.init.xavier_normal_(self.weight_mu)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w_std = _softplus(self.weight_rho)
            b_std = _softplus(self.bias_rho)
            weight = self.weight_mu + w_std * torch.randn_like(w_std)
            bias   = self.bias_mu   + b_std * torch.randn_like(b_std)
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return nn.functional.linear(x, weight, bias)

    def kl(self) -> torch.Tensor:
        return (
            _kl_gaussian(self.weight_mu, _softplus(self.weight_rho), self.prior_std)
            + _kl_gaussian(self.bias_mu, _softplus(self.bias_rho),   self.prior_std)
        )


class BNN(nn.Module):
    """2-hidden-layer Bayesian MLP: 15 → 64 → 64 → 1."""

    def __init__(self, n_features: int = 15, hidden: int = 64, prior_std: float = 1.0):
        super().__init__()
        self.fc1 = BayesianLinear(n_features, hidden, prior_std)
        self.fc2 = BayesianLinear(hidden,     hidden, prior_std)
        self.fc3 = BayesianLinear(hidden,     1,      prior_std)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        x = torch.relu(self.fc1(x, sample))
        x = torch.relu(self.fc2(x, sample))
        return self.fc3(x, sample).squeeze(-1)

    def kl(self) -> torch.Tensor:
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _elbo_loss(
    model: BNN,
    x: torch.Tensor,
    y: torch.Tensor,
    noise_var: torch.Tensor,
    n_data: int,
    kl_weight: float = 1.0,
    n_mc: int = 5,
) -> torch.Tensor:
    """
    ELBO = E_q[log p(y|f)] − kl_weight · KL[q ‖ p] / n_data

    Gaussian log-likelihood with per-point heteroskedastic noise.
    """
    log_liks = []
    for _ in range(n_mc):
        pred     = model(x, sample=True)
        log_liks.append(
            -0.5 * ((y - pred) ** 2 / noise_var + torch.log(2 * math.pi * noise_var))
        )
    avg_log_lik = torch.stack(log_liks, dim=0).mean(dim=0).sum()
    kl          = model.kl()
    return -(avg_log_lik - kl_weight * kl / n_data)


def train_bnn(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    noise_var_train: torch.Tensor,
    n_features: int = 15,
    hidden: int = 64,
    prior_std: float = 1.0,
    lr: float = 0.01,
    n_epochs: int = 1500,
    kl_warmup: int = 200,
    n_mc: int = 5,
    random_state: int = 42,
) -> BNN:
    """Train BNN with ELBO objective and KL warm-up."""
    torch.manual_seed(random_state)
    model     = BNN(n_features, hidden, prior_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_data    = len(y_train)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        kl_weight = min(1.0, epoch / max(kl_warmup, 1))
        loss      = _elbo_loss(model, x_train, y_train, noise_var_train,
                               n_data, kl_weight, n_mc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_bnn(
    model: BNN,
    x: torch.Tensor,
    noise_std: float,
    n_samples: int = 500,
) -> tuple[float, float, float]:
    """
    Returns (pred_mean, std_total, std_epistemic) in residual space [kJ/mol].

    std_total = sqrt(std_epistemic² + noise_std²)
    """
    model.eval()
    preds = torch.stack([model(x, sample=True) for _ in range(n_samples)], dim=0)
    mean         = float(preds.mean())
    std_epistemic = float(preds.std())
    std_total    = math.sqrt(std_epistemic ** 2 + noise_std ** 2)
    return mean, std_total, std_epistemic


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso(
    df: pd.DataFrame | None = None,
    hidden: int = 64,
    prior_std: float = 1.0,
    lr: float = 0.01,
    n_epochs: int = 1500,
    kl_warmup: int = 200,
    n_mc_train: int = 5,
    n_mc_pred: int = 500,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Leave-One-System-Out CV for BNN baseline.

    Returns DataFrame with same column schema as rf_loso_metals.csv.
    """
    if df is None:
        df = load_gp_data()

    X_raw  = df[FEATURE_NAMES].values.astype(np.float32)
    y_raw  = df["residual"].values.astype(np.float32)
    noise  = df["noise_kJmol"].values.astype(np.float32)

    N = len(df)
    rows: list[dict] = []

    print(f"BNN LOSO CV on {N} pairs "
          f"(hidden={hidden}, prior_std={prior_std}, "
          f"epochs={n_epochs}, kl_warmup={kl_warmup}) …")

    for i in range(N):
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False

        # StandardScaler fit on training fold only
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_raw[train_mask])
        X_test_np  = scaler.transform(X_raw[[i]])

        y_train_np    = y_raw[train_mask]
        noise_train   = noise[train_mask]
        noise_i       = float(noise[i])

        x_tr  = torch.tensor(X_train_np, dtype=torch.float32)
        y_tr  = torch.tensor(y_train_np,  dtype=torch.float32)
        nv_tr = torch.tensor(noise_train ** 2, dtype=torch.float32)
        x_te  = torch.tensor(X_test_np,   dtype=torch.float32)

        model = train_bnn(
            x_tr, y_tr, nv_tr,
            n_features=len(FEATURE_NAMES),
            hidden=hidden,
            prior_std=prior_std,
            lr=lr,
            n_epochs=n_epochs,
            kl_warmup=kl_warmup,
            n_mc=n_mc_train,
            random_state=random_state + i,
        )

        pred_mean, std_total, std_epistemic = predict_bnn(
            model, x_te, noise_i, n_samples=n_mc_pred
        )

        row     = df.iloc[i]
        res_true = float(row["residual"])

        rows.append({
            "pair":                   row["pair"],
            "h05_true":               float(row["h05"]),
            "h05_miedema":            float(row["h05_miedema"]),
            "h05_bnn":                float(row["h05_miedema"]) + pred_mean,
            "h05_bnn_std":            std_total,
            "residual_true":          res_true,
            "residual_pred":          pred_mean,
            "residual_std":           std_total,
            "residual_std_epistemic": std_epistemic,
            "noise_kJmol":            noise_i,
            "in_ci95":                abs(res_true - pred_mean) <= 1.96 * std_total,
        })

        if verbose and ((i + 1) % 20 == 0 or i == N - 1):
            mae_so_far = np.mean([abs(r["h05_true"] - r["h05_bnn"]) for r in rows])
            print(
                f"  [{i+1:3d}/{N}] {row['pair']:10s} | "
                f"pred {rows[-1]['h05_bnn']:+7.2f} | "
                f"true {rows[-1]['h05_true']:+7.2f} | "
                f"σ_ep {std_epistemic:.2f}  σ_tot {std_total:.2f} | "
                f"running MAE {mae_so_far:.3f}"
            )

    out = pd.DataFrame(rows)
    out["abs_err_miedema"] = (out["h05_true"] - out["h05_miedema"]).abs()
    out["abs_err_bnn"]     = (out["h05_true"] - out["h05_bnn"]).abs()
    return out


# ---------------------------------------------------------------------------
# Metrics & reporting
# ---------------------------------------------------------------------------

def compute_metrics(loso_df: pd.DataFrame) -> dict[str, float]:
    """MAE, RMSE, R², 95% CI coverage, ECE — identical formula to RF baseline."""
    from scipy.stats import norm

    err  = loso_df["h05_true"] - loso_df["h05_bnn"]
    mae  = float(loso_df["abs_err_bnn"].mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    r2   = float(
        1 - (err ** 2).sum() /
        ((loso_df["h05_true"] - loso_df["h05_true"].mean()) ** 2).sum()
    )
    ci95 = float(loso_df["in_ci95"].mean() * 100)

    z = (loso_df["residual_true"] - loso_df["residual_pred"]).values / \
        loso_df["residual_std"].values
    alphas   = np.linspace(0.1, 0.99, 10)
    observed = [(np.abs(z) <= norm.ppf((1 + a) / 2)).mean() for a in alphas]
    ece      = float(np.mean(np.abs(np.array(observed) - alphas)))

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "CI95": ci95, "ECE": ece}


def print_summary(
    loso_df: pd.DataFrame,
    gp_csv: str | None = None,
    rf_csv: str | None = None,
) -> None:
    """Print BNN metrics and optional side-by-side comparison with RF and GP."""
    from pathlib import Path
    from scipy.stats import norm

    m = compute_metrics(loso_df)

    print("\n=== BNN LOSO Summary ===")
    print(f"  Miedema MAE         : {loso_df['abs_err_miedema'].mean():.3f} kJ/mol")
    print(f"  BNN     MAE         : {m['MAE']:.3f} kJ/mol")
    print(f"  BNN     RMSE        : {m['RMSE']:.3f} kJ/mol")
    print(f"  BNN     R²          : {m['R2']:.4f}")
    print(f"  BNN mean σ_total    : {loso_df['h05_bnn_std'].mean():.3f} kJ/mol")
    print(f"  BNN mean σ_epistemic: {loso_df['residual_std_epistemic'].mean():.3f} kJ/mol")
    print(f"  BNN mean σ_noise    : {loso_df['noise_kJmol'].mean():.3f} kJ/mol")
    print(f"  95% CI coverage     : {m['CI95']:.1f}%  (target: 95%)")
    print(f"  ECE                 : {m['ECE']:.4f}")

    # Full comparison table
    if gp_csv and Path(gp_csv).exists() and rf_csv and Path(rf_csv).exists():
        gp = pd.read_csv(gp_csv)
        rf = pd.read_csv(rf_csv)

        def _metrics(df_in, pred_col, std_col):
            err  = df_in["h05_true"] - df_in[pred_col]
            mae  = float(df_in[f"abs_err_{pred_col.split('_')[1]}"].mean()) \
                   if f"abs_err_{pred_col.split('_')[1]}" in df_in.columns \
                   else float(err.abs().mean())
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

        gp_mae, gp_rmse, gp_r2, gp_ci95, gp_ece = _metrics(gp, "h05_gp", "residual_std")
        rf_mae, rf_rmse, rf_r2, rf_ci95, rf_ece  = _metrics(rf, "h05_rf", "residual_std")
        mied_mae  = float(loso_df["abs_err_miedema"].mean())
        mied_rmse = float(np.sqrt(((gp["h05_true"] - gp["h05_miedema"])**2).mean()))

        print("\n=== Comparison: Miedema | RF | BNN | GP ===")
        print(f"  {'Metric':22s} {'Miedema':>9} {'RF':>9} {'BNN':>9} {'GP':>9}")
        print(f"  {'-'*60}")
        print(f"  {'MAE  (kJ/mol)':22s} {mied_mae:>9.3f} {rf_mae:>9.3f} {m['MAE']:>9.3f} {gp_mae:>9.3f}")
        print(f"  {'RMSE (kJ/mol)':22s} {mied_rmse:>9.3f} {rf_rmse:>9.3f} {m['RMSE']:>9.3f} {gp_rmse:>9.3f}")
        print(f"  {'R²':22s} {'—':>9} {rf_r2:>9.4f} {m['R2']:>9.4f} {gp_r2:>9.4f}")
        print(f"  {'95% CI coverage':22s} {'—':>9} {rf_ci95:>9.1f} {m['CI95']:>9.1f} {gp_ci95:>9.1f}")
        print(f"  {'ECE':22s} {'—':>9} {rf_ece:>9.4f} {m['ECE']:>9.4f} {gp_ece:>9.4f}")
