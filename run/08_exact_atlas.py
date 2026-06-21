"""
Canonical HEA atlas with EXACT GP-covariance propagation of σ_ΔH.

This is the canonical atlas generator (the exact-σ atlas; it also writes the
independence-approximation atlas used in the appendix comparison).  σ_ΔH is
propagated through the FULL GP posterior
covariance — the off-diagonal covariance the ARD kernel induces between binary
pairs that share an element / lie close in descriptor space:

    σ²_ΔH = wᵀ K w  = Σ_{p,q} w_p w_q Cov(h_p, h_q)      [exact]
    σ²_ΔH ≈ Σ_p w_p² Var(h_p)                            [independence, legacy]

where w_p = 4·c_i·c_j for pair p=(i,j) and K is the GP posterior covariance of
the binary h05 predictions.  "Exact" here means exact propagation of GP
uncertainty *under the pairwise (Muggianu-type) ΔH approximation* — it does not
change the ΔH model itself.

Compute-once trick
------------------
The C(44,2)=946 binary pairs are shared across the C(44,5)=1,086,008 subsets, and
Cov(h_p, h_q) is a property of the pair-of-pairs (subset-independent).  We compute
the full 946×946 posterior covariance ONCE (predict_joint over the pool, ~0.1 s);
each subset slices the 10×10 block of its C(5,2)=10 pairs and evaluates wᵀKw via
(W@Kb)·W.  No GP forward pass inside the loop.

Outputs
-------
  out/data/atlas_phea.csv                  — CANONICAL atlas (exact σ primary,
                                             + *_indep columns for the appendix)
  out/data/atlas_phea_independence.csv     — backup of the previous (legacy) atlas
  out/data/independence_atlas_comparison.json — exact-vs-indep divergence summary
  out/data/atlas_frequency.json            — top-1000 element/pair frequencies
                                             (exact and indep, for the manuscript)
  paper/atlas_phea_heatmap.pdf, atlas_sigma_heatmap.pdf, atlas_group_panel.pdf
  paper/atlas_exact_vs_indep.pdf           — appendix scatter + |ΔP| histogram
"""

from __future__ import annotations

import json
import time
from collections import Counter
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

POOL = sorted([
    "Ag", "Al", "Au", "B",  "Ba", "Bi", "Ca", "Cd", "Ce", "Co",
    "Cr", "Cu", "Eu", "Fe", "Ga", "Gd", "Ge", "Hg", "In", "Ir",
    "Li", "Mg", "Mn", "Na", "Nb", "Ni", "Pb", "Pd", "Pt", "Rh",
    "Sb", "Sc", "Se", "Si", "Sn", "Sr", "Tb", "Te", "Ti", "Tl",
    "Y",  "Yb", "Zn", "Zr",
])
N_SUBSETS = 1_086_008  # C(44, 5)

# ---------------------------------------------------------------------------
# Load predictor + composition grid
# ---------------------------------------------------------------------------
print("Loading HEA predictor...")
from src.phea.predict import HEAPredictor
from src.phea.features import DH_DEPLOY_LO, DH_DEPLOY_HI
from src.gp.pipeline import generate_compositions

pred = HEAPredictor.load()

print("Pre-computing GP pair cache (for independence σ via compute_array)...")
t0 = time.time()
pred.precompute_pool()
print(f"  cache: {len(pred.feat._pair_cache)} pairs in {time.time()-t0:.1f}s")

print("  (full 946×946 GP posterior covariance precomputed inside precompute_pool)")

FRACS = generate_compositions(n_elem=5, x_min=0.05, x_max=0.35, step=0.05)
N_COMP = len(FRACS)
EQ_IDX = int(np.where(np.all(np.abs(FRACS - 0.2) < 1e-9, axis=1))[0][0])
print(f"Compositions per subset: {N_COMP}  (equimolar index {EQ_IDX})")

# ---------------------------------------------------------------------------
# Atlas loop
# ---------------------------------------------------------------------------
print(f"Canonical exact atlas: {N_SUBSETS:,} subsets × {N_COMP} comps")
print(f"Deployment range: mu_dH ∈ [{DH_DEPLOY_LO}, {DH_DEPLOY_HI}] kJ/mol\n")

rows = []
t_start = time.time()
report_every = 50_000

for i, combo in enumerate(combinations(POOL, 5)):
    elems = list(combo)   # already sorted (POOL is sorted)

    feat_exact = pred.feat.compute_array(elems, FRACS, exact=True)   # canonical
    feat_indep = pred.feat.compute_array(elems, FRACS, exact=False)  # diagonal (appendix)
    mu_dH = feat_exact[:, 0]
    valid = (mu_dH >= DH_DEPLOY_LO) & (mu_dH <= DH_DEPLOY_HI)
    n_valid = int(valid.sum())
    n_oor = N_COMP - n_valid

    row = {
        "subset_key": "-".join(elems),
        "el1": elems[0], "el2": elems[1], "el3": elems[2],
        "el4": elems[3], "el5": elems[4],
    }

    if n_valid == 0:
        row.update({
            "mean_P": np.nan, "max_P": np.nan, "eq_P": np.nan, "frac_05": 0.0,
            "mean_sigma_dH": np.nan, "n_valid": 0, "n_oor": n_oor,
            "best_comp": "", "best_mu_dH": np.nan,
            "mean_P_indep": np.nan, "max_P_indep": np.nan, "eq_P_indep": np.nan,
            "mean_sigma_indep": np.nan,
        })
        rows.append(row)
        continue

    sig_exact = feat_exact[:, 1]
    p_ex = pred._features_to_phea(feat_exact)            # exact, all comps
    p_in = pred._features_to_phea(feat_indep)            # independence, all comps

    p_ex_v = p_ex[valid]
    best_local = int(p_ex_v.argmax())
    best_fracs = FRACS[valid][best_local]
    eq_ok = bool(valid[EQ_IDX])

    row.update({
        "mean_P":        round(float(p_ex_v.mean()), 5),
        "max_P":         round(float(p_ex_v.max()), 5),
        "eq_P":          round(float(p_ex[EQ_IDX]), 5) if eq_ok else np.nan,
        "frac_05":       round(float((p_ex_v > 0.5).mean()), 5),
        "mean_sigma_dH": round(float(sig_exact[valid].mean()), 4),
        "n_valid":       n_valid,
        "n_oor":         n_oor,
        "best_comp":     "|".join(f"{e}:{x:.2f}" for e, x in zip(elems, best_fracs)),
        "best_mu_dH":    round(float(mu_dH[valid][best_local]), 3),
        "mean_P_indep":     round(float(p_in[valid].mean()), 5),
        "max_P_indep":      round(float(p_in[valid].max()), 5),
        "eq_P_indep":       round(float(p_in[EQ_IDX]), 5) if eq_ok else np.nan,
        "mean_sigma_indep": round(float(feat_indep[valid, 1].mean()), 4),
    })
    rows.append(row)

    if (i + 1) % report_every == 0:
        el = time.time() - t_start
        rate = (i + 1) / el
        print(f"  {i+1:>10,} / {N_SUBSETS:,}  ({100*(i+1)/N_SUBSETS:.1f}%)  "
              f"elapsed {el/60:.1f}m  ETA {(N_SUBSETS-i-1)/rate/60:.1f}m")

elapsed = time.time() - t_start
print(f"\nLoop done in {elapsed/60:.1f} min")

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Write the canonical decision atlas (exact σ) — 15 decision-support columns only.
# The diagonal-approximation (independence) values go to a separate appendix file
# so the user-facing atlas stays clean.
# ---------------------------------------------------------------------------
CANONICAL_COLS = [
    "subset_key", "el1", "el2", "el3", "el4", "el5",
    "mean_P", "max_P", "eq_P", "frac_05", "mean_sigma_dH",
    "n_valid", "n_oor", "best_comp", "best_mu_dH",
]
INDEP_COLS = [
    "subset_key", "el1", "el2", "el3", "el4", "el5",
    "mean_P_indep", "max_P_indep", "eq_P_indep", "mean_sigma_indep",
]
atlas_path = OUT_DATA / "atlas_phea.csv"
df[CANONICAL_COLS].to_csv(atlas_path, index=False)
print(f"Saved CANONICAL atlas: {atlas_path}  ({len(df):,} rows, exact σ)")

# Independence-approximation atlas (appendix comparison only)
indep_path = OUT_DATA / "atlas_phea_independence.csv"
df[INDEP_COLS].to_csv(indep_path, index=False)
print(f"Saved independence-approx atlas (appendix): {indep_path}")

# ---------------------------------------------------------------------------
# Exact-vs-independence divergence summary (appendix)
# ---------------------------------------------------------------------------
v = df.dropna(subset=["mean_P"]).copy()
v["dP_mean"] = v["mean_P"] - v["mean_P_indep"]
absdP = v["dP_mean"].abs()
sig_ratio = (v["mean_sigma_dH"] / v["mean_sigma_indep"].replace(0, np.nan))

comparison = {
    "n_subsets_total": int(len(df)),
    "n_subsets_valid": int(len(v)),
    "runtime_min": round(elapsed / 60, 2),
    "note": "exact = wᵀKw (canonical); indep = diagonal approximation (legacy).",
    "mean_P": {
        "abs_dP_mean": float(absdP.mean()),
        "abs_dP_median": float(absdP.median()),
        "abs_dP_p95": float(absdP.quantile(0.95)),
        "abs_dP_p99": float(absdP.quantile(0.99)),
        "abs_dP_max": float(absdP.max()),
        "frac_dP_lt_0.01": float((absdP < 0.01).mean()),
        "frac_dP_lt_0.02": float((absdP < 0.02).mean()),
        "frac_dP_lt_0.05": float((absdP < 0.05).mean()),
        "n_exact_higher_gt_0.01": int((v["dP_mean"] > 0.01).sum()),
        "n_exact_lower_gt_0.01": int((v["dP_mean"] < -0.01).sum()),
        "max_dP_signed_pos": float(v["dP_mean"].max()),
        "max_dP_signed_neg": float(v["dP_mean"].min()),
        "pearson": float(v["mean_P"].corr(v["mean_P_indep"])),
        "spearman": float(v["mean_P"].corr(v["mean_P_indep"], method="spearman")),
    },
    "sigma_ratio_exact_over_indep": {
        "median": float(sig_ratio.median()),
        "p95": float(sig_ratio.quantile(0.95)),
        "max": float(sig_ratio.max()),
    },
    "ranking_topN_overlap": {},
    "worst_subsets_by_abs_dP_mean": (
        v.loc[absdP.nlargest(15).index,
              ["subset_key", "mean_P_indep", "mean_P", "mean_sigma_indep", "mean_sigma_dH"]]
        .rename(columns={"mean_P": "mean_P_exact", "mean_sigma_dH": "mean_sigma_exact"})
        .to_dict("records")
    ),
}
for Ntop in (100, 1000):
    a = set(v.nlargest(Ntop, "mean_P_indep")["subset_key"])
    b = set(v.nlargest(Ntop, "mean_P")["subset_key"])
    comparison["ranking_topN_overlap"][str(Ntop)] = {
        "overlap": len(a & b), "of": Ntop, "jaccard": len(a & b) / len(a | b),
    }
(OUT_DATA / "independence_atlas_comparison.json").write_text(json.dumps(comparison, indent=2))
print(f"Saved: {OUT_DATA / 'independence_atlas_comparison.json'}")

# ---------------------------------------------------------------------------
# Top-1000 element / pair frequencies (manuscript paragraph), exact and indep
# ---------------------------------------------------------------------------
def freqs(col, Ntop=1000):
    top = v.nlargest(Ntop, col)
    el = Counter()
    pr = Counter()
    for key in top["subset_key"]:
        els = key.split("-")
        el.update(els)
        pr.update("-".join(p) for p in combinations(sorted(els), 2))
    return {"elements": dict(el.most_common()),
            "pairs": dict(pr.most_common(25))}

freq = {"top_n": 1000,
        "exact": freqs("mean_P"),
        "independence": freqs("mean_P_indep")}
(OUT_DATA / "atlas_frequency.json").write_text(json.dumps(freq, indent=2))
print(f"Saved: {OUT_DATA / 'atlas_frequency.json'}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n=== Canonical (exact) atlas summary ===")
print(f"  valid subsets: {len(v):,}")
for lo, hi in [(0,.1),(.1,.3),(.3,.5),(.5,1.01)]:
    n = ((v["mean_P"] >= lo) & (v["mean_P"] < hi)).sum()
    print(f"    mean_P [{lo:.1f},{hi:.1f}): {n:>9,} ({100*n/len(v):.1f}%)")
print("\n  Top-8 by mean_P (exact):")
print(v.nlargest(8, "mean_P")[["subset_key","mean_P","mean_P_indep","max_P","eq_P"]].to_string(index=False))
print("\n=== Exact vs independence ===")
print(f"  |ΔP| median/p95/max: {absdP.median():.4f}/{absdP.quantile(.95):.4f}/{absdP.max():.4f}")
print(f"  exact higher by >0.01: {(v['dP_mean']>0.01).sum()}   lower by >0.01: {(v['dP_mean']<-0.01).sum()}")
print(f"  Spearman: {comparison['mean_P']['spearman']:.4f}")
for Ntop in (100, 1000):
    o = comparison["ranking_topN_overlap"][str(Ntop)]
    print(f"  top-{Ntop} overlap: {o['overlap']}/{Ntop}")
print("\n  Top-1000 element frequency (exact):")
print("   ", ", ".join(f"{e}{n}" for e, n in list(freq["exact"]["elements"].items())[:12]))

# ---------------------------------------------------------------------------
# Regenerate element-pair maps (Fig. 5/7) from the canonical exact atlas
# ---------------------------------------------------------------------------
print("\nRegenerating atlas maps from exact atlas...")
import importlib.util
_spec = importlib.util.spec_from_file_location("atlas_maps", ROOT / "run" / "04_atlas_maps.py")
_atlas_maps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_atlas_maps)
_atlas_maps.make_atlas_maps(df)

# ---------------------------------------------------------------------------
# Appendix figure: exact vs independence
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
hb = ax1.hexbin(v["mean_P_indep"], v["mean_P"], gridsize=80, bins="log",
                cmap="viridis", mincnt=1)
m = float(v["mean_P_indep"].max())
ax1.plot([0, m], [0, m], "r--", lw=1, label="y = x")
ax1.set_xlabel("mean P(HEA) — independence (diagonal)")
ax1.set_ylabel("mean P(HEA) — exact (wᵀKw)")
ax1.set_title("Per-subset P(HEA): approximation vs exact")
ax1.legend(loc="upper left", fontsize=9)
fig.colorbar(hb, ax=ax1, label="log₁₀ count")
ax2.hist(absdP, bins=np.linspace(0, max(absdP.max(), 1e-3), 80), color="#2166ac", alpha=0.85)
ax2.axvline(absdP.quantile(0.95), color="#d7191c", ls="--", lw=1.2,
            label=f"95th pct = {absdP.quantile(0.95):.3f}")
ax2.set_xlabel("|ΔP_mean| = |exact − independence|")
ax2.set_ylabel("Number of subsets"); ax2.set_yscale("log")
ax2.set_title("Divergence of the decision variable")
ax2.legend(fontsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(OUT_FIGS / f"atlas_exact_vs_indep.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure: {OUT_FIGS / 'atlas_exact_vs_indep.pdf'}")
print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")
print("Done.")
