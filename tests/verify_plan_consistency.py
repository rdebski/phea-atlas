"""
Cross-check every quantitative claim in paper/kbs-in-a-nutshell.md against the
actual output files.  Run standalone:  uv run python tests/verify_plan_consistency.py

Prints PASS/FAIL per claim; exits non-zero if any FAIL.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "out" / "data"
RES = ROOT / "out" / "data"   # ML-baseline LOSO CSVs (produced by run/00_ml_baselines.py)

fails = []


def check(label, got, expect, tol):
    ok = (got is not None) and abs(got - expect) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:46s} plan={expect:<10} got={got}")
    if not ok:
        fails.append(label)


def check_eq(label, got, expect):
    ok = got == expect
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:46s} plan={expect!s:<10} got={got}")
    if not ok:
        fails.append(label)


def metrics_from_loso(csv, pred_col):
    df = pd.read_csv(csv)
    err = (df["h05_" + pred_col] - df["h05_true"]).values
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((df["h05_true"] - df["h05_true"].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return mae, rmse, r2, len(df)


print("=" * 78)
print("§4.1  GP + ML baselines (LOSO, 222 pairs)")
print("=" * 78)
if not (RES / "gp_loso_metals.csv").exists():
    print("  [skipped] LOSO baseline CSVs not found in out/data/.")
    print("  Run `python run/00_ml_baselines.py` first to check §4.1 (compute-heavy).")
else:
    gp_mae, gp_rmse, gp_r2, n_gp = metrics_from_loso(RES / "gp_loso_metals.csv", "gp")
    check_eq("GP training pairs", n_gp, 222)
    mied_mae = float(pd.read_csv(RES / "gp_loso_metals.csv")["abs_err_miedema"].mean())
    check("Miedema baseline MAE", mied_mae, 5.836, 0.02)
    check("GP MAE", gp_mae, 4.041, 0.02)
    check("GP RMSE", gp_rmse, 6.434, 0.05)
    check("GP R2", gp_r2, 0.861, 0.01)
    # Reported ECE=0.028 is the RAW (native GP) ECE — post-hoc T-scaling makes it
    # WORSE (0.047).  So 92.8% raw coverage is the consistent partner of ECE=0.028,
    # not the 94.1% calibrated coverage.
    cov = float(pd.read_csv(RES / "gp_loso_metals.csv")["in_ci95_raw"].mean())
    check("GP 95% CI coverage (raw; matches ECE=0.028)", cov * 100, 92.8, 0.3)

    rf_mae, _, rf_r2, _ = metrics_from_loso(RES / "rf_loso_metals.csv", "rf")
    check("RF MAE", rf_mae, 4.557, 0.05)
    check("RF R2", rf_r2, 0.823, 0.01)
    bnn_mae, _, bnn_r2, _ = metrics_from_loso(RES / "bnn_loso_metals.csv", "bnn")
    check("BNN MAE", bnn_mae, 4.642, 0.05)
    check("BNN R2", bnn_r2, 0.828, 0.01)
    de_mae, _, de_r2, _ = metrics_from_loso(RES / "de_loso_metals.csv", "de")
    check("Deep Ensemble MAE", de_mae, 4.879, 0.05)
    check("Deep Ensemble R2", de_r2, 0.817, 0.01)

print("=" * 78)
print("§3.2 / §4.2  Calibration layer & dataset")
print("=" * 78)
m = json.loads((OUT / "calibration_metrics.json").read_text())
check_eq("n compositions", m["n_compositions"], 433)
check_eq("n HEA", m["n_hea_1"], 248)
check_eq("n non-HEA", m["n_hea_0"], 185)
check("AUC-ROC", m["loocv_raw"]["auc_roc"], 0.732, 0.002)
check("ECE (T-scaled)", m["loocv_temperature_scaled"]["ece"], 0.041, 0.003)
check("ECE raw", m["loocv_raw"]["ece"], 0.181, 0.005)
check("Temperature T (LOO)", m["loocv_temperature_scaled"]["temperature"], 3.08, 0.05)
check("pen_hi coef (~0)", m["coefficients"]["pen_hi"], 0.0, 0.02)

st = json.loads((OUT / "exp_data_stats.json").read_text())
check_eq("conflicting labels", st["n_conflict"], 77)

print("=" * 78)
print("§0/§1.3/§4.3  Decision value (pure Miedema headline + GP ablation)")
print("=" * 78)
dv = json.loads((OUT / "decision_value.json").read_text())
check("base rate", dv["base_rate"], 0.5727, 0.001)
s = dv["strategies"]["P(HEA)"]
check_eq("P(HEA) HEA@20", s["hea@20"], 20)
check_eq("P(HEA) HEA@50", s["hea@50"], 45)
check_eq("P(HEA) HEA@100", s["hea@100"], 86)
check("P(HEA) EF@50", s["ef@50"], 1.571, 0.01)
mo = dv["strategies"]["Miedema+Omega"]
check_eq("Miedema+Ω HEA@50 (pure h05)", mo["hea@50"], 40)
check("Miedema+Ω EF@50 (pure h05)", mo["ef@50"], 1.40, 0.01)
check("Miedema+Ω Prec@20 (pure h05)", mo["prec@20"], 0.75, 0.01)
wc = dv["strategies"]["Miedema window-centre"]
check_eq("Miedema window HEA@50 (pure h05)", wc["hea@50"], 41)
check("Miedema window Prec@20 (pure h05)", wc["prec@20"], 0.85, 0.01)
cj = dv["strategies"]["Miedema window AND Omega"]
check_eq("Conjunctive window∧Ω HEA@50 (= window leg)", cj["hea@50"], 40)
check("Conjunctive window∧Ω EF@50", cj["ef@50"], 1.40, 0.01)
hs = dv["strategies"]["Screen accept-set (random within)"]
check_eq("Honest unordered screen: accept-set size", hs["accept_set_size"], 296)
check("Honest unordered screen EF@50 (random within accept)", hs["ef@50"], 1.11, 0.01)
ab = dv["ablation_gp_mu_dH"]["Omega rule (GP mu_dH)"]
check_eq("Ablation Ω HEA@50 (GP μ_ΔH)", ab["hea@50"], 43)
check("Ablation Ω EF@50 (GP μ_ΔH)", ab["ef@50"], 1.50, 0.01)

print("=" * 78)
print("§4.4  Probabilistic atlas")
print("=" * 78)
atlas = pd.read_csv(OUT / "atlas_phea.csv")
check_eq("atlas total subsets", len(atlas), math.comb(44, 5))
valid = atlas.dropna(subset=["mean_P"])
check_eq("subsets with valid P", len(valid), 1_077_940)
check_eq("all-OOR subsets", int((atlas["n_valid"] == 0).sum()), 8068)
# §4.4 denominator = ALL subsets (95.2%); CLAUDE.md uses valid only (95.9%)
n_low = int((valid["mean_P"] < 0.1).sum())
check("share mean_P<0.1 / all subsets (%)", n_low / len(atlas) * 100, 95.2, 0.1)
check("share mean_P<0.1 / valid (%)", n_low / len(valid) * 100, 95.9, 0.1)
check_eq("subsets mean_P > 0.3", int((valid["mean_P"] > 0.3).sum()), 99)
check_eq("subsets mean_P > 0.5", int((valid["mean_P"] > 0.5).sum()), 0)
top = valid.nlargest(1, "mean_P").iloc[0]
check_eq("top subset", set(top["subset_key"].split("-")), {"Co", "Cr", "Cu", "Fe", "Ni"})
check("top subset mean_P", top["mean_P"], 0.467, 0.005)
check("top subset max_P", top["max_P"], 0.935, 0.005)

print("\n" + "=" * 78)
if fails:
    print(f"RESULT: {len(fails)} FAIL — {fails}")
    raise SystemExit(1)
print("RESULT: ALL CLAIMS CONSISTENT ✓")
