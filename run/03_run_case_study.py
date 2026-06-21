"""
Case study: Al-Cu-Fe-Ni-Ti — P(HEA) for equimolar and non-equimolar compositions.

Part 1 — Equimolar subsets (k = 3, 4, 5)
  All C(5,k) combinations of {Al, Cu, Fe, Ni, Ti}.
  Note: k=3 (ΔS=1.10R) and k=4 (ΔS=1.39R) are below the HEA entropy threshold
  (ΔS ≥ 1.5R); included for completeness but flagged.

Part 2 — Non-equimolar k=5
  All compositions with x_i ∈ {0.05, 0.10, ..., 0.35}, Σ=1  → 1,451 compositions.

Outputs
-------
  out/data/case_study_equimolar.csv
  out/data/case_study_noneq.csv
  out/figures/case_study_phea_dist.pdf/png
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_DATA = ROOT / "out" / "data"
OUT_FIGS = ROOT / "out" / "figures"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

ELEMENTS = ["Al", "Cu", "Fe", "Ni", "Ti"]

# ---------------------------------------------------------------------------
# Reuse composition generator from old pipeline (no screening logic imported)
# ---------------------------------------------------------------------------
from src.gp.pipeline import generate_compositions

# ---------------------------------------------------------------------------
# Load predictor
# ---------------------------------------------------------------------------
print("Loading HEA predictor...")
from src.phea.predict import HEAPredictor
pred = HEAPredictor.load()
print("Done.\n")

# ===========================================================================
# Part 1 — Equimolar subsets
# ===========================================================================
print("=" * 55)
print("PART 1 — Equimolar subsets (k=3,4,5)")
print("=" * 55)

rows_eq = []
for k in [3, 4, 5]:
    for combo in combinations(ELEMENTS, k):
        comp = {e: 1.0 / k for e in combo}
        result = pred.predict_full(comp)
        ds_R = -sum((1/k) * math.log(1/k) for _ in range(k))
        rows_eq.append({
            "elements":    "+".join(sorted(combo)),
            "k":           k,
            "dS_over_R":   round(ds_R, 4),
            "hea_entropy": ds_R >= 1.5,
            **{f"x_{e}": round(comp.get(e, 0.0), 4) for e in ELEMENTS},
            "P_HEA":       round(result["P_HEA"], 4),
            "mu_dH":       round(result["mu_dH"], 3),
            "sigma_dH":    round(result["sigma_dH"], 3),
            "delta":       round(result["delta"], 4),
            "T_m":         round(result["T_m"], 1),
        })

df_eq = pd.DataFrame(rows_eq).sort_values("P_HEA", ascending=False)

print(f"\n{'Elements':<25s}  {'k':>2s}  {'ΔS/R':>5s}  {'HEA?':>5s}  "
      f"{'P(HEA)':>7s}  {'μ_ΔH':>7s}  {'σ_ΔH':>6s}  {'δ':>6s}")
print("-" * 75)
for _, row in df_eq.iterrows():
    hea_flag = "✓" if row["hea_entropy"] else "✗"
    print(f"{row['elements']:<25s}  {row['k']:>2d}  {row['dS_over_R']:>5.3f}  "
          f"{hea_flag:>5s}  {row['P_HEA']:>7.3f}  {row['mu_dH']:>7.2f}  "
          f"{row['sigma_dH']:>6.3f}  {row['delta']:>6.4f}")

eq_path = OUT_DATA / "case_study_equimolar.csv"
df_eq.to_csv(eq_path, index=False)
print(f"\nSaved: {eq_path}")

# ===========================================================================
# Part 2 — Non-equimolar k=5
# ===========================================================================
print("\n" + "=" * 55)
print("PART 2 — Non-equimolar Al-Cu-Fe-Ni-Ti (k=5)")
print("=" * 55)

fracs = generate_compositions(n_elem=5, x_min=0.05, x_max=0.35, step=0.05)
print(f"Compositions generated: {len(fracs)}")

pred.precompute_pairs(ELEMENTS)
feat_arr = pred.feat.compute_array(ELEMENTS, fracs)
p_hea    = pred.predict_array_filtered(ELEMENTS, fracs)

from src.phea.features import FEATURE_NAMES as FN, DH_DEPLOY_LO, DH_DEPLOY_HI
feat_df = pd.DataFrame(feat_arr, columns=FN)

oor_mask = (feat_arr[:, 0] < DH_DEPLOY_LO) | (feat_arr[:, 0] > DH_DEPLOY_HI)
n_oor    = int(oor_mask.sum())

df_noneq = pd.DataFrame(fracs, columns=[f"x_{e}" for e in ELEMENTS])
df_noneq["P_HEA"]     = p_hea
df_noneq["mu_dH"]     = feat_df["mu_dH"]
df_noneq["sigma_dH"]  = feat_df["sigma_dH"]
df_noneq["delta"]     = feat_df["delta"]
df_noneq["dS_over_R"] = feat_df["dS_R"]
df_noneq["T_m"]       = feat_df["T_m"]
df_noneq["oor"]       = oor_mask

df_noneq = df_noneq.sort_values("P_HEA", ascending=False).reset_index(drop=True)

p_valid = p_hea[~oor_mask]
print(f"\nDeployment range: mu_dH ∈ [{DH_DEPLOY_LO}, {DH_DEPLOY_HI}] kJ/mol")
print(f"  In-range : {len(p_valid)} compositions")
print(f"  OOR (P=0): {n_oor} compositions")
print(f"\nP(HEA) statistics (all {len(p_hea)} compositions, OOR set to 0):")
print(f"  Mean   : {p_hea.mean():.3f}")
print(f"  Median : {np.median(p_hea):.3f}")
print(f"  Max    : {p_hea.max():.3f}")
print(f"  > 0.5  : {(p_hea > 0.5).sum()} ({100*(p_hea>0.5).mean():.1f}%)")
print(f"  > 0.7  : {(p_hea > 0.7).sum()} ({100*(p_hea>0.7).mean():.1f}%)")

print(f"\nTop 10 compositions by P(HEA):")
hdr = (f"{'xAl':>5s} {'xCu':>5s} {'xFe':>5s} {'xNi':>5s} {'xTi':>5s}  "
       f"{'P(HEA)':>7s}  {'μ_ΔH':>7s}  {'σ_ΔH':>6s}  {'δ':>6s}")
print(hdr)
print("-" * 55)
for _, row in df_noneq.head(10).iterrows():
    print(f"{row['x_Al']:>5.2f} {row['x_Cu']:>5.2f} {row['x_Fe']:>5.2f} "
          f"{row['x_Ni']:>5.2f} {row['x_Ti']:>5.2f}  "
          f"{row['P_HEA']:>7.3f}  {row['mu_dH']:>7.2f}  "
          f"{row['sigma_dH']:>6.3f}  {row['delta']:>6.4f}")

noneq_path = OUT_DATA / "case_study_noneq.csv"
df_noneq.to_csv(noneq_path, index=False)
print(f"\nSaved: {noneq_path}")

# ===========================================================================
# Figure — P(HEA) distribution
# ===========================================================================
# Style notes:
#   * delta (Hume-Rothery size mismatch) is computed once per composition; there
#     is no averaging here.  The colour scale is therefore driven by the actual
#     data range (robust 2-98 percentile clip) instead of a hardcoded window,
#     so the full perceptually-uniform colormap is used (the old vmin=0.04,
#     vmax=0.12 wasted >50% of the colour range — delta only spans ~0.044-0.077).
#   * viridis (sequential, perceptually uniform) replaces the diverging RdYlBu_r:
#     delta is a one-sided "lower is better" quantity, not a diverging one, and
#     this keeps the figure consistent with the atlas maps (Fig. 5).
delta_col = FN.index("delta")
delta_vals = feat_arr[:, delta_col]
d_lo, d_hi = np.percentile(delta_vals, [2, 98])

plt.rcParams.update({"axes.axisbelow": True})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

# Left: histogram non-equimolar
ax = axes[0]
ax.hist(p_hea, bins=25, color="#3a7bbf", alpha=0.9, edgecolor="white", linewidth=0.6)
ax.axvline(0.5, color="#d7191c", lw=1.6, ls="--", label="P = 0.5")
ax.axvline(np.median(p_hea), color="#1a9641", lw=1.6, ls=":",
           label=f"Median = {np.median(p_hea):.2f}")
ax.set_xlabel("P(HEA)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Non-equimolar Al-Cu-Fe-Ni-Ti  (1,451 compositions)", fontsize=11)
ax.grid(axis="y", color="0.85", lw=0.6)
ax.legend(fontsize=9, framealpha=0.9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# Right: scatter mu_dH vs P(HEA), coloured by delta.
# delta is shown in percent (the conventional Hume-Rothery unit, e.g. the
# delta<=6.5% rule) so the colourbar ticks are compact 2-3 digit numbers
# (5.5, 6.0, ...) instead of 0.0550-style 5-character labels.
ax = axes[1]
sc = ax.scatter(feat_arr[:, 0], p_hea, c=delta_vals * 100.0, cmap="viridis",
                s=14, alpha=0.75, vmin=d_lo * 100.0, vmax=d_hi * 100.0,
                linewidths=0.2, edgecolors="white")
cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("δ  [%]  (size mismatch)", fontsize=11)
cb.ax.tick_params(labelsize=9)
ax.axhline(0.5, color="#d7191c", lw=1.2, ls="--", alpha=0.7)
ax.set_xlabel("μ$_{\\Delta H}$  [kJ/mol]", fontsize=12)
ax.set_ylabel("P(HEA)", fontsize=12)
ax.set_title("P(HEA) vs. mixing enthalpy", fontsize=11)
ax.grid(color="0.9", lw=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# Mark equimolar k=5 — a single diamond with white halo so it reads over the
# dense cloud, labelled with a leader line going left and slightly down (~190
# deg) into the empty region above the lower-mu arcs (no legend, no overlap).
eq5 = df_eq[df_eq["k"] == 5].iloc[0]
ax.scatter(eq5["mu_dH"], eq5["P_HEA"], s=120, marker="*", color="#440154",
           edgecolors="white", linewidths=0.9, zorder=7)
ax.annotate(f"Equimolar (P = {eq5['P_HEA']:.2f})",
            xy=(eq5["mu_dH"], eq5["P_HEA"]), xytext=(-112, -18),
            textcoords="offset points", ha="left", va="top",
            fontsize=9, zorder=6,
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.9,
                            shrinkA=2, shrinkB=6))

plt.tight_layout()
for ext in ("pdf", "png"):
    fp = OUT_FIGS / f"case_study_phea_dist.{ext}"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_FIGS / 'case_study_phea_dist.pdf'}")
print(f"  delta colour range (p2-p98): [{d_lo:.4f}, {d_hi:.4f}]  "
      f"(data span {delta_vals.min():.4f}-{delta_vals.max():.4f})")
plt.close()

print("\nDone.")
