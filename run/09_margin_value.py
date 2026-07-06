"""
Decision value of the calibrated margin.

Addresses the "modest gains" concern on the axis a bare ranking metric misses: the
system does not just rank candidates, it states *how confident* each ranking is,
and that confidence is empirically trustworthy.  Two panels, both on the n=433
held-out (LOO-CV) calibrated predictions -- no retraining:

  (a) Selective prediction / risk-coverage.  Acting only on the most confident
      decisions (largest calibrated margin |P-0.5|) raises accuracy monotonically
      -- the "gradient of evidence" made operational.  The epistemic GP channel
      sigma_dH is shown as a control: on this pre-filtered labelled set it does
      NOT separate outcomes (its value is atlas-coverage, not benchmark accuracy),
      so we do not overclaim it.

  (b) Reliability of the calibrated P(HEA): binned predicted probability vs
      observed HEA frequency, with Wilson 95% intervals.  A stated P reads as an
      expected success rate -- the property that licenses budget allocation.

Inputs  : out/data/calibration_predictions.csv  (P_HEA_loocv_cal, is_hea)
          out/data/calibration_features.csv      (sigma_dH)  [control curve only]
Outputs : out/figures/margin_value.{pdf,png}
          out/data/margin_value.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_DATA = ROOT / "out" / "data"
OUT_FIGS = ROOT / "out" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load held-out predictions (+ sigma for the control curve)
# ---------------------------------------------------------------------------
pred = pd.read_csv(OUT_DATA / "calibration_predictions.csv").set_index("comp_key")
feat = pd.read_csv(OUT_DATA / "calibration_features.csv").set_index("comp_key")

y = pred["is_hea"].values.astype(int)
P = pred["P_HEA_loocv_cal"].values          # raw calibrated LOO probability (NOT deployment-zeroed)
sigma = feat.loc[pred.index, "sigma_dH"].values
n = len(y)
base = y.mean()
correct = (P >= 0.5).astype(int) == y       # the P>=0.5 decision, per composition
overall_acc = correct.mean()
print(f"n={n}  base_rate={base:.3f}  overall accuracy(P>=0.5)={overall_acc:.3f}")

# ---------------------------------------------------------------------------
# (a) Selective prediction: accuracy on the most-confident coverage fraction
# ---------------------------------------------------------------------------
def selective_curve(rank_key: np.ndarray, coverages: np.ndarray) -> np.ndarray:
    """rank_key: larger = more confident / retained first.  Returns accuracy per coverage."""
    order = np.argsort(-rank_key, kind="stable")
    accs = []
    for c in coverages:
        m = max(1, int(round(n * c)))
        accs.append(correct[order[:m]].mean())
    return np.array(accs)

cov = np.linspace(0.2, 1.0, 33)
acc_margin = selective_curve(np.abs(P - 0.5), cov)      # hero: calibrated margin
acc_sigma  = selective_curve(-sigma, cov)               # control: low sigma = "confident"

# ---------------------------------------------------------------------------
# (b) Reliability: 10 equal-width bins, Wilson 95% CI on observed frequency
# ---------------------------------------------------------------------------
def wilson(k: int, m: int, z: float = 1.96):
    if m == 0:
        return (0.0, 0.0)
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return (max(0.0, c - h), min(1.0, c + h))

edges = np.linspace(0, 1, 11)
bins = []
ece = 0.0
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (P >= lo) & (P < hi) if hi < 1.0 else (P >= lo) & (P <= hi)
    m = int(sel.sum())
    if m == 0:
        continue
    meanP = float(P[sel].mean())
    k = int(y[sel].sum())
    emp = k / m
    lo_ci, hi_ci = wilson(k, m)
    ece += (m / n) * abs(emp - meanP)
    bins.append(dict(meanP=meanP, emp=emp, n=m, k=k, ci_lo=lo_ci, ci_hi=hi_ci))
print(f"reliability ECE (10 equal-width bins) = {ece:.3f}")

# ---------------------------------------------------------------------------
# Save numbers
# ---------------------------------------------------------------------------
margin_json = {
    "n": n, "base_rate": round(float(base), 4),
    "overall_accuracy_P>=0.5": round(float(overall_acc), 4),
    "selective_prediction": {
        "note": "accuracy on the most-confident coverage fraction; margin=|P-0.5|, control=low sigma_dH",
        "coverage": [round(float(c), 3) for c in cov],
        "accuracy_margin": [round(float(a), 4) for a in acc_margin],
        "accuracy_sigma_control": [round(float(a), 4) for a in acc_sigma],
    },
    "reliability_bins": bins,
    "ece_equal_width_10bin": round(float(ece), 4),
}
with open(OUT_DATA / "margin_value.json", "w") as fh:
    json.dump(margin_json, fh, indent=2)
print(f"Saved: {OUT_DATA / 'margin_value.json'}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
C_MARGIN, C_SIGMA, C_REF = "#2166ac", "#999999", "0.55"
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

# ---- (a) selective prediction ----
axA.plot(cov * 100, acc_margin, "-o", color=C_MARGIN, ms=3.5, lw=1.8,
         label=r"rank by calibrated margin $|P-0.5|$")
axA.plot(cov * 100, acc_sigma, "--s", color=C_SIGMA, ms=3, lw=1.4, mfc="white",
         label=r"rank by GP uncertainty $\sigma_{\Delta H}$ (control)")
axA.axhline(overall_acc, color=C_REF, ls=":", lw=1.3,
            label=f"no selection (all {n}): {overall_acc:.2f}")
axA.set_xlabel("Coverage: most-confident % of decisions acted on", fontsize=11)
axA.set_ylabel(r"Accuracy of the $P\!\geq\!0.5$ decision", fontsize=11)
axA.set_title("(a) The calibrated margin carries decision value", fontsize=11)
axA.set_xlim(100, 20)                         # high coverage (left) -> selective (right)
axA.grid(alpha=0.25, lw=0.6)
axA.legend(fontsize=8.5, loc="upper left")
axA.annotate("act only on\nconfident calls", xy=(40, acc_margin[cov.searchsorted(0.40)]),
             xytext=(58, 0.83), fontsize=8.5, color=C_MARGIN,
             arrowprops=dict(arrowstyle="->", color=C_MARGIN, lw=1.0))

# ---- (b) reliability ----
mp = np.array([b["meanP"] for b in bins])
em = np.array([b["emp"] for b in bins])
lo = np.array([b["ci_lo"] for b in bins])
hi = np.array([b["ci_hi"] for b in bins])
cnts = np.array([b["n"] for b in bins])
axB.plot([0, 1], [0, 1], ls="--", color=C_REF, lw=1.3, label="perfect calibration")
axB.errorbar(mp, em, yerr=[em - lo, hi - em], fmt="o", color=C_MARGIN, ms=6,
             capsize=3, lw=1.2, label="observed (Wilson 95% CI)")
for x, yv, c in zip(mp, em, cnts):
    axB.annotate(f"n={c}", (x, yv), textcoords="offset points", xytext=(6, -9),
                 fontsize=7, color="0.35")
axB.set_xlabel(r"Predicted $P(\mathrm{HEA})$", fontsize=11)
axB.set_ylabel("Observed HEA frequency", fontsize=11)
axB.set_title(f"(b) A stated probability is a trustworthy magnitude (ECE={ece:.3f})", fontsize=11)
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_aspect("equal")
axB.grid(alpha=0.25, lw=0.6)
axB.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(OUT_FIGS / f"margin_value.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIGS / 'margin_value.pdf'} (+ .png)")
print("\nDone.")
