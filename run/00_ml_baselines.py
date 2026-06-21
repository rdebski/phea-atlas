"""
ML baseline comparison (paper Table, §4.1): GP vs Random Forest, BNN, Deep Ensemble.

Reproduces the Leave-One-System-Out (LOSO) benchmark over the 222 metallic binary
pairs that justifies the GP choice (lowest MAE, best-calibrated ECE).

Outputs (out/data/)
-------------------
  gp_loso_metals.csv    — GP LOSO predictions and uncertainties (reference)
  rf_loso_metals.csv    — Random Forest LOSO
  bnn_loso_metals.csv   — Bayesian NN (VI/ELBO) LOSO
  de_loso_metals.csv    — Deep Ensemble (M=5) LOSO

NOTE: compute-heavy (BNN/DE retrain per held-out pair). Expect several minutes.
The headline P(HEA) pipeline (run/01..08) does NOT depend on this script — it is
the methodological comparison only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.gp.evaluate import run_loso as gp_run_loso
from src.baselines.random_forest import run_loso as rf_run_loso
from src.baselines.bnn import run_loso as bnn_run_loso
from src.baselines.deep_ensemble import run_loso as de_run_loso, print_summary

OUT = ROOT / "out" / "data"
GP_CSV  = OUT / "gp_loso_metals.csv"
RF_CSV  = OUT / "rf_loso_metals.csv"
BNN_CSV = OUT / "bnn_loso_metals.csv"
DE_CSV  = OUT / "de_loso_metals.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== GP (ours) LOSO ===")
    gp_run_loso(verbose=True).to_csv(GP_CSV, index=False, float_format="%.6f")

    print("\n=== Random Forest LOSO ===")
    rf_run_loso(n_estimators=500, verbose=True).to_csv(RF_CSV, index=False, float_format="%.6f")

    print("\n=== BNN (VI/ELBO) LOSO ===")
    bnn_run_loso(
        hidden=64, prior_std=1.0, lr=0.01, n_epochs=1500,
        kl_warmup=200, n_mc_train=5, n_mc_pred=500, verbose=True,
    ).to_csv(BNN_CSV, index=False, float_format="%.6f")

    print("\n=== Deep Ensemble (M=5) LOSO ===")
    de_df = de_run_loso(n_members=5, hidden=64, lr=0.01, n_epochs=1500, verbose=True)
    de_df.to_csv(DE_CSV, index=False, float_format="%.6f")

    print("\n=== §4.1 comparison table ===")
    print_summary(de_df, gp_csv=str(GP_CSV), rf_csv=str(RF_CSV), bnn_csv=str(BNN_CSV))


if __name__ == "__main__":
    main()
