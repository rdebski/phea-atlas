"""
Decision-value analysis: does ranking by calibrated P(HEA) retrieve confirmed
HEAs more efficiently than deterministic enthalpy screening?

Evaluation is retrospective on the n=433 experimental compositions, using the
LOO-CV P(HEA) predictions as the ranking signal (out/data/calibration_predictions.csv)
and the experimental is_hea label as ground truth.

Strategies compared (all rank the SAME 433 compositions; higher score = synthesise first):
  1. P(HEA) ranking        — descending LOO-CV calibrated P(HEA), with the same
                             deployment-range safeguard as the atlas applied
                             (μ_ΔH outside [-50,+8] → P=0).  The headline is
                             invariant to both bounds over their plausible ranges
                             (see deployment_sensitivity in the JSON output).
  2. Ω≥1.1 rule            — descending Ω = T_m·ΔS / |ΔH_Miedema|  (Yang & Zhang 2012)
  3. enthalpy-window rule  — margin to the acceptance window [−15,+5]; ascending
                             |ΔH_Miedema − (−5)| (≡ proximity to the window midpoint −5,
                             since min(ΔH+15, 5−ΔH) = 10 − |ΔH+5|)
  4. window ∧ Ω (conjunct.)— actual practice: accept iff BOTH hold, rank survivors by Ω.
                             Coincides with the window leg here (every in-window comp.
                             also satisfies Ω≥1.1) → 40/50 at B=50. Reported, not plotted.
  4b. screen accept-set     — HONEST deterministic value: the screen returns an UNORDERED
      (random within)        accept set, so E[HEA@B]=B·precision(accept). 296 pass, prec
                             0.635 → EF≈1.11. The Ω/window rankings are charitable upper
                             bounds on this. Key "Screen accept-set (random within)".
  (JSON keys retain the legacy names "Miedema+Omega" / "Miedema window-centre"; the
   conjunctive baseline is keyed "Miedema window AND Omega".)
  5. Random                — base rate (HEA fraction of the database)
  (ablation: the deterministic screen — conjunctive window∧Ω, its two single-rule
   legs, and the best-case envelope — re-evaluated on GP μ_ΔH instead of raw Miedema h05)

Metrics at budget B:
  HEA@B   = number of confirmed HEAs among the top-B ranked
  Prec@B  = HEA@B / B
  EF@B    = Prec@B / base_rate     (enrichment factor)
  AUDC    = area under the discovery curve (mean fractional recall over B=1..n)

BASELINE DESIGN — two complementary comparisons.

  HEADLINE (the paper's claim "deterministic screening → calibrated probabilistic ranking"):
    the Ω and enthalpy-window rules are applied to the *pure* Miedema pairwise ΔH
    (h05_miedema) — the enthalpy a practitioner actually screens with today (no GP,
    no experimental binary data).  This measures the end-to-end value of the whole
    framework over current practice.

  ABLATION (isolates the calibration layer):
    the SAME deterministic screen (conjunctive window∧Ω, and its best-case envelope)
    applied to the *GP posterior* μ_ΔH — the enthalpy the P(HEA) layer itself consumes.
    Here every strategy sees identical (GP) thermodynamics and differs only in the
    decision rule, so the remaining gap is attributable to the calibration layer
    (LogReg + δ/ΔS/T_m), not the enthalpy estimate.  P(HEA) still leads (45/50 vs
    43/50 at B=50), confirming the gain is not merely from better enthalpies.

Outputs
-------
  out/data/decision_value.json   — full table of HEA@B / Prec@B / EF@B / AUDC,
                                   deployment_sensitivity (upper/lower bound sweep),
                                   and recovery_ranks (Mechanism: Co-Cr-Cu-Fe-Ni-family
                                   rank lifts under each ranking)
  out/figures/decision_value.pdf/png
"""
from __future__ import annotations

import json
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

R_GAS = 8.314  # J/(mol·K)
BUDGETS = [20, 50, 100]

# ---------------------------------------------------------------------------
# Load predictions + experimental compositions
# ---------------------------------------------------------------------------
print("Loading predictions and experimental data...")
pred_df = pd.read_csv(OUT_DATA / "calibration_predictions.csv")
exp_df = pd.read_csv(OUT_DATA / "exp_data_clean.csv")
exp_df["composition"] = exp_df["composition"].apply(json.loads)

# Align by comp_key (predictions preserve exp ordering, but join to be safe)
exp_df = exp_df.set_index("comp_key")
pred_df = pred_df.set_index("comp_key")
df = pred_df.join(exp_df[["composition"]], how="left")
assert df["composition"].notna().all(), "comp_key mismatch between files"

y = df["is_hea"].values.astype(int)
base_rate = float(y.mean())
n = len(y)
print(f"  n={n}  base_rate={base_rate:.4f}  ({y.sum()} HEA / {n})")

# ---------------------------------------------------------------------------
# Miedema features (pure Miedema ΔH, Ω, window centre distance)
# ---------------------------------------------------------------------------
print("Loading GP/Miedema model for baseline ΔH...")
from src.gp.predict import GPPredictor

gp = GPPredictor.load(str(ROOT / "models" / "gp_full_model.pt"))
ep = gp.ep  # element properties indexed by symbol


def miedema_dH(comp: dict[str, float]) -> float:
    """Pairwise Miedema mixing enthalpy [kJ/mol]: 4 Σ_{i<j} x_i x_j h05(i,j)."""
    el = list(comp)
    return sum(
        4.0 * comp[a] * comp[b] * gp.miedema_model.h_mix_fn(a, b, 0.5)
        for a, b in combinations(el, 2)
    )


# T_m and ΔS come from the composition (shared by both enthalpy variants).
feat_df = pd.read_csv(OUT_DATA / "calibration_features.csv").set_index("comp_key")
Tm_arr = feat_df.loc[df.index, "T_m"].values
dSR_arr = feat_df.loc[df.index, "dS_R"].values
dS_J = dSR_arr * R_GAS                                  # J/(mol·K)

# HEADLINE: deterministic rules on the PURE Miedema pairwise ΔH (current practice).
dH_pure = np.array([miedema_dH(c) for c in df["composition"]])
omega_pure = Tm_arr * dS_J / np.abs(dH_pure * 1000.0)
dist_centre_pure = np.abs(dH_pure - (-5.0))

# CONJUNCTIVE screen (actual deterministic practice): accept iff in window AND Ω≥1.1.
# Ranked: gate-passers first (by Ω desc), non-passers below (by window proximity), so
# the discovery curve is defined over all n compositions. On this set the gate coincides
# with the window leg alone (every in-window composition also satisfies Ω≥1.1).
pass_gate_pure = (dH_pure >= -15.0) & (dH_pure <= 5.0) & (omega_pure >= 1.1)
score_conj_pure = np.where(pass_gate_pure, omega_pure, -1e6 - dist_centre_pure)

# ABLATION: same deterministic screen on the GP posterior μ_ΔH (isolates the
# calibration layer).  The standard screen is the conjunctive window ∧ Ω gate
# (and, as the charitable upper bound, the best-case envelope of the three rules)
# — exactly the screen used for the headline, but now fed the GP enthalpy.
mu_gp = feat_df.loc[df.index, "mu_dH"].values
omega_gp = Tm_arr * dS_J / np.abs(mu_gp * 1000.0)
dist_centre_gp = np.abs(mu_gp - (-5.0))
pass_gate_gp = (mu_gp >= -15.0) & (mu_gp <= 5.0) & (omega_gp >= 1.1)
score_conj_gp = np.where(pass_gate_gp, omega_gp, -1e6 - dist_centre_gp)

# Deployment-range safeguard, applied CONSISTENTLY with the 1.08M-subset atlas
# (predict_array_filtered): compositions with GP μ_ΔH outside [DH_DEPLOY_LO,
# DH_DEPLOY_HI] receive P(HEA)=0.  The headline is invariant to this (see the
# deployment_sensitivity block below); we apply it so the retrospective ranking
# uses exactly the same model output as the atlas.
from src.phea.features import DH_DEPLOY_LO, DH_DEPLOY_HI

p_hea_raw = df["P_HEA_loocv_cal"].values
in_deploy = (mu_gp >= DH_DEPLOY_LO) & (mu_gp <= DH_DEPLOY_HI)
p_hea = np.where(in_deploy, p_hea_raw, 0.0)

# ---------------------------------------------------------------------------
# Rankings (score; higher = synthesise first)
# ---------------------------------------------------------------------------
rankings = {
    "P(HEA)":                 p_hea,
    "Miedema+Omega":          omega_pure,
    "Miedema window-centre":  -dist_centre_pure,
}

# Conjunctive baseline (window ∧ Ω) — reported in the table/JSON but NOT plotted.
rankings_extra = {
    "Miedema window AND Omega": score_conj_pure,
}

# Ablation rankings using GP μ_ΔH (reported separately, not plotted).  The
# headline reference is the conjunctive window ∧ Ω screen; the two single-rule
# legs are retained for transparency (Ω is redundant — every in-window comp also
# satisfies Ω≥1.1 — so window∧Ω coincides with the Ω leg at B=50).
rankings_ablation = {
    "window AND Omega (GP mu_dH)":   score_conj_gp,
    "Omega rule (GP mu_dH)":         omega_gp,
    "Window-centre rule (GP mu_dH)": -dist_centre_gp,
}


def discovery_curve(score: np.ndarray) -> np.ndarray:
    """Cumulative HEA count along the ranking induced by `score` (desc)."""
    order = np.argsort(-score, kind="stable")
    return np.cumsum(y[order])


def audc(score: np.ndarray) -> float:
    """Area under the (fractional-recall vs budget) discovery curve."""
    cum = discovery_curve(score)
    total = y.sum()
    return float(np.mean(cum / total))


# ---------------------------------------------------------------------------
# Build results table
# ---------------------------------------------------------------------------
results = {"base_rate": base_rate, "n": n, "n_hea": int(y.sum()), "strategies": {}}

print("\n" + "=" * 78)
print(f"{'Strategy':24s}" + "".join(f"  Prec@{B:<3d} HEA@{B:<3d} EF@{B:<4d}" for B in BUDGETS) + "  AUDC")
print("=" * 78)

for name, score in {**rankings, **rankings_extra}.items():
    cum = discovery_curve(score)
    row = {"audc": round(audc(score), 4)}
    cells = []
    for B in BUDGETS:
        hea = int(cum[B - 1])
        prec = hea / B
        ef = prec / base_rate
        row[f"hea@{B}"] = hea
        row[f"prec@{B}"] = round(prec, 4)
        row[f"ef@{B}"] = round(ef, 4)
        cells.append(f"  {prec:6.2f}  {hea:3d}/{B:<3d} {ef:6.3f}")
    results["strategies"][name] = row
    print(f"{name:24s}" + "".join(cells) + f"  {row['audc']:.3f}")

# Best-case deterministic envelope: at each budget the BEST of the three standard
# rankings (Ω, enthalpy-window, window∧Ω) on the pure Miedema ΔH. A charitable upper
# bound that grants the deterministic screen its most favourable ordering, so the
# head-to-head comparison cannot be accused of weakening the baseline.
cum_det_env = np.maximum.reduce([
    discovery_curve(omega_pure),          # Ω≥1.1 rule
    discovery_curve(-dist_centre_pure),   # enthalpy-window rule
    discovery_curve(score_conj_pure),     # window ∧ Ω (actual practice)
])
env_row = {"audc": round(float(np.mean(cum_det_env / y.sum())), 4)}
ecells = []
for B in BUDGETS:
    hea = int(cum_det_env[B - 1]); prec = hea / B; ef = prec / base_rate
    env_row[f"hea@{B}"] = hea; env_row[f"prec@{B}"] = round(prec, 4); env_row[f"ef@{B}"] = round(ef, 4)
    ecells.append(f"  {prec:6.2f}  {hea:3d}/{B:<3d} {ef:6.3f}")
results["strategies"]["Best-case deterministic (envelope)"] = env_row
print(f"{'Best-case det. (envelope)':24s}" + "".join(ecells) + f"  {env_row['audc']:.3f}")

# Honest deterministic screen: a pass/fail screen returns an UNORDERED accept set, so when
# |accept| > B the practitioner cannot prioritise -> expected retrieval = random WITHIN the
# accept set (E[HEA@B] = B * accept-set precision). This is the deterministic decision value
# WITHOUT any imputed ranking; the Ω/window rankings above are charitable upper bounds on it.
Na, Ha = int(pass_gate_pure.sum()), int(y[pass_gate_pure].sum())
Nr, Hr = n - Na, int(y.sum() - Ha)
kk = np.arange(1, n + 1)
honest_cum = np.where(kk <= Na, kk * Ha / Na, Ha + (kk - Na) * Hr / max(Nr, 1))
honest_row = {"audc": round(float(np.mean(honest_cum / y.sum())), 4),
              "accept_set_size": Na, "accept_set_precision": round(Ha / Na, 4)}
hcells = []
for B in BUDGETS:
    e = float(honest_cum[B - 1]); prec = e / B; ef = prec / base_rate
    honest_row[f"hea@{B}"] = round(e, 1)
    honest_row[f"prec@{B}"] = round(prec, 4)
    honest_row[f"ef@{B}"] = round(ef, 4)
    hcells.append(f"  {prec:6.2f}  {e:5.1f}   {ef:6.3f}")
results["strategies"]["Screen accept-set (random within)"] = honest_row
print(f"{'Screen accept-set (random)':24s}" + "".join(hcells) + f"  {honest_row['audc']:.3f}")

# Random + oracle reference curves
oracle_cum = np.minimum(np.arange(1, n + 1), y.sum())
random_audc = float(np.mean((np.arange(1, n + 1) * base_rate) / y.sum()))
results["strategies"]["Random"] = {
    "audc": round(random_audc, 4),
    **{f"prec@{B}": round(base_rate, 4) for B in BUDGETS},
    **{f"hea@{B}": round(base_rate * B, 1) for B in BUDGETS},
    **{f"ef@{B}": 1.0 for B in BUDGETS},
}
results["strategies"]["Oracle"] = {"audc": round(float(np.mean(oracle_cum / y.sum())), 4)}
print(f"{'Random':24s}" + "".join(f"  {base_rate:6.2f}  {base_rate*B:5.1f}   {1.0:6.3f}" for B in BUDGETS))

# Ablation: same deterministic rules on the GP posterior μ_ΔH (isolates calibration)
print("\n--- Ablation: deterministic rules on GP μ_ΔH (isolates the calibration layer) ---")
results["ablation_gp_mu_dH"] = {}
for name, score in rankings_ablation.items():
    cum = discovery_curve(score)
    row = {"audc": round(audc(score), 4)}
    cells = []
    for B in BUDGETS:
        hea = int(cum[B - 1]); prec = hea / B; ef = prec / base_rate
        row[f"hea@{B}"] = hea; row[f"prec@{B}"] = round(prec, 4); row[f"ef@{B}"] = round(ef, 4)
        cells.append(f"  {prec:6.2f}  {hea:3d}/{B:<3d} {ef:6.3f}")
    results["ablation_gp_mu_dH"][name] = row
    print(f"{name:34s}" + "".join(cells) + f"  {row['audc']:.3f}")

# Best-case deterministic envelope on the GP μ_ΔH — the exact counterpart of the
# headline envelope, but fed the GP enthalpy.  This is the screen the ablation
# headline number quotes (parallel to "Best-case deterministic (envelope)" above).
cum_det_env_gp = np.maximum.reduce([
    discovery_curve(score_conj_gp),       # window ∧ Ω (standard screen)
    discovery_curve(omega_gp),            # Ω≥1.1 leg
    discovery_curve(-dist_centre_gp),     # enthalpy-window leg
])
env_gp_row = {"audc": round(float(np.mean(cum_det_env_gp / y.sum())), 4)}
egcells = []
for B in BUDGETS:
    hea = int(cum_det_env_gp[B - 1]); prec = hea / B; ef = prec / base_rate
    env_gp_row[f"hea@{B}"] = hea; env_gp_row[f"prec@{B}"] = round(prec, 4)
    env_gp_row[f"ef@{B}"] = round(ef, 4)
    egcells.append(f"  {prec:6.2f}  {hea:3d}/{B:<3d} {ef:6.3f}")
results["ablation_gp_mu_dH"]["Best-case deterministic (GP mu_dH, envelope)"] = env_gp_row
print(f"{'Best-case det. (GP mu, envelope)':34s}" + "".join(egcells) + f"  {env_gp_row['audc']:.3f}")

# ---------------------------------------------------------------------------
# Deployment-range sensitivity — the headline must not hinge on the exact bounds.
# Re-rank P(HEA) under alternative upper/lower μ_ΔH cut-offs (the other bound held
# at its deployment value).  EF@B for B<=100 is invariant across the whole plausible
# literature range (lower -50..-15, upper +8..+12); only AUDC drifts, always in the
# conservative direction (tightening zeros confirmed HEAs and sinks them).
# ---------------------------------------------------------------------------
def _ranking_metrics(score: np.ndarray) -> dict:
    cum = discovery_curve(score)
    out = {"audc": round(audc(score), 4)}
    for B in BUDGETS:
        hea = int(cum[B - 1])
        out[f"hea@{B}"] = hea
        out[f"ef@{B}"] = round((hea / B) / base_rate, 4)
    return out


sens = {"upper": {}, "lower": {}}
for T in (6.0, 8.0, 10.0, 12.0):
    score = np.where((mu_gp >= DH_DEPLOY_LO) & (mu_gp <= T), p_hea_raw, 0.0)
    sens["upper"][f"+{T:g}"] = _ranking_metrics(score)
for T in (-50.0, -40.0, -30.0, -25.0, -20.0, -15.0):
    score = np.where((mu_gp >= T) & (mu_gp <= DH_DEPLOY_HI), p_hea_raw, 0.0)
    sens["lower"][f"{T:g}"] = _ranking_metrics(score)
results["deployment_sensitivity"] = sens

print("\n--- Deployment-range sensitivity (EF@B invariant; AUDC drifts conservatively) ---")
for side, label in (("upper", "upper bound"), ("lower", "lower bound")):
    print(f"  {label}:")
    for k, v in sens[side].items():
        print(f"    {k:>5s}  HEA@50={v['hea@50']:>2d}/50  EF@20={v['ef@20']:.3f}  "
              f"EF@50={v['ef@50']:.3f}  EF@100={v['ef@100']:.3f}  AUDC={v['audc']:.4f}")

# ---------------------------------------------------------------------------
# Recovery ranks — the "Mechanism" result of the paper.
# The borderline Co-Cr-Cu-Fe-Ni-family HEAs whose GP μ_ΔH sits just above the
# enthalpy window ([+5.8,+7.0]) are exactly what the Ω / window rules deprioritise.
# We record each one's rank (1 = synthesise first) under the three rankings already
# computed above so the paper's "rank 102 / 297 → 6" and the lift ranges are
# reproducible — these positions are not otherwise persisted.
# ---------------------------------------------------------------------------
def _ranks_from_score(score: np.ndarray) -> np.ndarray:
    """Rank each composition (1 = highest score = synthesise first); stable order."""
    order = np.argsort(-score, kind="stable")
    r = np.empty(n, dtype=int)
    r[order] = np.arange(1, n + 1)
    return r


r_phea = _ranks_from_score(p_hea)
r_omega = _ranks_from_score(omega_pure)
r_window = _ranks_from_score(-dist_centre_pure)

FAMILY = {"Co", "Cr", "Cu", "Fe", "Ni"}
MU_LO, MU_HI = 5.8, 7.0                         # paper's recovered-band window
TOL = 0.05                                       # include μ that round to the bounds
recov = []
for i, (ck, comp) in enumerate(zip(df.index.values, df["composition"])):
    if set(comp) == FAMILY and (MU_LO - TOL) <= mu_gp[i] <= (MU_HI + TOL):
        recov.append({
            "comp_key": ck,
            "mu_dH": round(float(mu_gp[i]), 3),
            "dH_pure_miedema": round(float(dH_pure[i]), 3),
            "P_HEA": round(float(p_hea[i]), 3),
            "is_hea": int(y[i]),
            "rank_P_HEA": int(r_phea[i]),
            "rank_omega": int(r_omega[i]),
            "rank_window": int(r_window[i]),
            "lift_over_omega": int(r_omega[i] - r_phea[i]),
            "lift_over_window": int(r_window[i] - r_phea[i]),
        })
recov.sort(key=lambda d: d["mu_dH"])

if recov:
    p_vals = [c["P_HEA"] for c in recov]
    lo_om = [c["lift_over_omega"] for c in recov]
    lo_win = [c["lift_over_window"] for c in recov]
    example = recov[0]                            # the lowest-μ (≈+5.8) Cu-rich case
    results["recovery_ranks"] = {
        "description": ("Ranks (1 = synthesise first) of the Co-Cr-Cu-Fe-Ni-family "
                        "compositions P(HEA) recovers that the deterministic rules miss "
                        "(GP mu_dH in [+5.8,+7.0], just above the enthalpy window). "
                        "Reproduces the paper's Mechanism paragraph."),
        "family_elements": sorted(FAMILY),
        "mu_dH_window": [MU_LO, MU_HI],
        "compositions": recov,
        "summary": {
            "n": len(recov),
            "P_HEA_min": min(p_vals), "P_HEA_max": max(p_vals),
            "lift_over_omega_min": min(lo_om), "lift_over_omega_max": max(lo_om),
            "lift_over_window_min": min(lo_win), "lift_over_window_max": max(lo_win),
            "highlighted_example": {
                "comp_key": example["comp_key"], "mu_dH": example["mu_dH"],
                "rank_P_HEA": example["rank_P_HEA"],
                "rank_omega": example["rank_omega"],
                "rank_window": example["rank_window"],
            },
        },
    }
    print("\n--- Recovery ranks (Mechanism): Co-Cr-Cu-Fe-Ni family, mu_dH in [+5.8,+7.0] ---")
    for c in recov:
        print(f"  mu={c['mu_dH']:5.2f}  P={c['P_HEA']:.3f}  rank P(HEA)={c['rank_P_HEA']:3d}  "
              f"Omega={c['rank_omega']:3d}  window={c['rank_window']:3d}  "
              f"(lift +{c['lift_over_omega']}/+{c['lift_over_window']})")
else:
    print("\n[warn] no Co-Cr-Cu-Fe-Ni recovery-band compositions found; recovery_ranks omitted")

with open(OUT_DATA / "decision_value.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nSaved: {OUT_DATA / 'decision_value.json'}")

# ---------------------------------------------------------------------------
# Figure: discovery curves — TWO approaches (no baseline zoo).
#
#   * calibrated P(HEA) ranking ............ the contribution (hero).
#   * best-case deterministic screen ....... at each B the BEST of the three
#       standard rankings {Ω, enthalpy-window, window∧Ω} on the pure Miedema ΔH.
#       This is a charitable UPPER bound: we grant the deterministic screen its
#       most favourable ordering, so the comparison cannot be accused of
#       strawmanning the baseline.  P(HEA) still leads at every budget.
#   * deterministic screen, accept set ..... the HONEST lower bound: a pass/fail
#       screen returns an UNORDERED accept set, so E[HEA@B] = B·precision.
#   * Oracle ............................... reference ceiling.
#   (The random-selection reference is omitted: over the real candidate space it is not a
#    ~50% line and shows it as one is misleading; EF is still defined relative to it.)
#
# JSON/table output above is unchanged; this only restyles the figure.
# ---------------------------------------------------------------------------
B_MAX = 150
B_axis = np.arange(1, n + 1)
B_scatter = np.arange(5, B_MAX + 1, 5)   # 5, 10, ..., 150

cum_phea = discovery_curve(p_hea)
# cum_det_env (best-case deterministic envelope) computed above with the table.

C_PHEA, C_DET, C_HON = "#2166ac", "#e08214", "#b2182b"

fig, ax = plt.subplots(figsize=(7, 5))

# reference + honest lines (background)
ax.plot([0, B_MAX], [0, B_MAX], "-", color="0.6", lw=1.4, label="Oracle", zorder=1)
ax.plot(B_axis, honest_cum, "--", color=C_HON, lw=1.6, zorder=2,
        label="Det. screen (accept set, random within)")

# the two head-to-head ranked strategies (scatter every 5)
ax.scatter(B_scatter, cum_phea[B_scatter - 1], color=C_PHEA, marker="o",
           s=32, zorder=4, label=r"$P$(HEA) ranking")
ax.scatter(B_scatter, cum_det_env[B_scatter - 1], color=C_DET, marker="s",
           s=32, zorder=3, label="Best-case deterministic screen")

for B in BUDGETS:
    ax.axvline(B, color="grey", lw=0.6, ls=":")

# ---- Summary table (lower-right): the two approaches at each budget ---------
def _hea(cum, B):
    return int(cum[B - 1])

tbl = ax.table(
    cellText=[
        ["P(HEA)",
         f"{_hea(cum_phea, 20)}/20",
         f"{_hea(cum_phea, 50)}/50",
         f"{_hea(cum_phea, 100)}/100"],
        ["Best-case det.",
         f"{_hea(cum_det_env, 20)}/20",
         f"{_hea(cum_det_env, 50)}/50",
         f"{_hea(cum_det_env, 100)}/100"],
    ],
    colLabels=["", "B=20", "B=50", "B=100"],
    loc="lower right",
    bbox=[0.36, 0.02, 0.62, 0.19],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
# Give the label column extra width so "Best-case det." fits; share the rest equally.
_label_w, _ncol = 0.34, 4
for (r, c), cell in tbl.get_celld().items():
    cell.set_width(_label_w if c == 0 else (1.0 - _label_w) / (_ncol - 1))

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("0.55")
    cell.set_linewidth(0.6)
    cell.set_facecolor("white")

    edges = set()
    if row == 0:  edges |= {"T", "B"}   # header: top outer + line below header
    if row == 2:  edges.add("B")         # bottom outer
    if col == 0:  edges |= {"L", "R"}   # label col: left outer + right separator
    if col == 3:  edges.add("R")         # right outer
    cell.visible_edges = "".join(sorted(edges)) if edges else "open"

    if row == 0:
        cell.set_facecolor("#f0f0f0")
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_ha("center")
    elif col == 0:
        cell.get_text().set_ha("left")
    else:
        cell.get_text().set_ha("center")

ax.set_xlabel("Experimental budget $B$ (compositions synthesised)", fontsize=12)
ax.set_ylabel("Confirmed HEAs discovered", fontsize=12)
ax.set_title("Discovery curves: calibrated $P$(HEA) vs. best-case deterministic screening",
             fontsize=10.5)
ax.set_xlim(0, B_MAX)
ax.set_ylim(0, B_MAX)
ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(OUT_FIGS / f"decision_value.{ext}", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIGS / 'decision_value.pdf'}")
print("\nDone.")
