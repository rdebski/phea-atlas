"""
07_landscape.py — P(HEA) vs propagated enthalpy mu_dH, the "decision landscape".

Visualises how the calibrated P(HEA) relates to the propagated Miedema-window
position mu_dH on the n=433 labelled set (leave-one-out predictions). Confirmed
HEAs and non-HEAs are plotted separately; the enthalpy acceptance window
[-15, +5] kJ/mol is shaded. The figure makes one point visible without any
threshold or pass/fail partition: a band of confirmed HEAs sits just ABOVE the
window's upper edge (mu_dH up to +11.4 kJ/mol) yet still receives a high
P(HEA) -- exactly the borderline-endothermic alloys (Co-Cr-Cu-Fe-Ni family) that
a deterministic window deprioritises and the calibration layer recovers.

Visual enhancements:
  - Marker size  ∝  sigma_dH (GP uncertainty; global scale)
  - Marker alpha ∝  log(n_reports) (label reliability)
  - Right strip  :  KDE of P(HEA) by class, shared y-axis
  - Inset zoom   :  recovered band (mu_dH > +5 kJ/mol) in lower-left corner

NB: this is a continuous landscape, NOT a confusion matrix. P(HEA) is never
thresholded; the window shading is illustrative of the deterministic rule only.

Inputs : out/data/calibration_features.csv   (mu_dH, sigma_dH, n_reports)
         out/data/calibration_predictions.csv (LOO calibrated P(HEA), is_hea)
Output : paper/landscape_phea_mu.{pdf,png}
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic, gaussian_kde

ROOT = Path(__file__).parent.parent
OUT_DATA = ROOT / "out" / "data"
OUT_FIGS = ROOT / "out" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

WIN_LO, WIN_HI = -15.0, 5.0
DEPLOY_HI = 8.0
C_HEA = "#2166ac"
C_NON = "#b2182b"

# ---------------------------------------------------------------------------
feat = pd.read_csv(OUT_DATA / "calibration_features.csv")[
    ["comp_key", "mu_dH", "sigma_dH", "is_hea", "n_reports"]]
pred = pd.read_csv(OUT_DATA / "calibration_predictions.csv")[
    ["comp_key", "P_HEA_loocv_cal"]]
d = feat.merge(pred, on="comp_key", validate="one_to_one")

hea = d[d.is_hea == 1]
non = d[d.is_hea == 0]

recov = hea[hea.mu_dH > WIN_HI]
print(f"n={len(d)}  HEA={len(hea)}  non-HEA={len(non)}")
print(f"confirmed HEAs above window upper edge (+{WIN_HI:g}): {len(recov)} "
      f"(mu up to {recov.mu_dH.max():.1f}, P range "
      f"{recov.P_HEA_loocv_cal.min():.2f}–{recov.P_HEA_loocv_cal.max():.2f})")
print(f"sigma_dH: {d.sigma_dH.min():.2f} – {d.sigma_dH.max():.2f} kJ/mol")
print(f"n_reports: {d.n_reports.min()} – {d.n_reports.max()}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_sig_lo = d.sigma_dH.min()
_sig_hi = d.sigma_dH.max()
_nr_max = d.n_reports.max()
S_MIN, S_MAX = 12, 90
ALPHA_LO, ALPHA_HI = 0.35, 0.95


def sigma_to_s(sigma: pd.Series) -> np.ndarray:
    """Marker area proportional to GP uncertainty (global scale)."""
    return S_MIN + (S_MAX - S_MIN) * (sigma - _sig_lo) / (_sig_hi - _sig_lo)


def rgba_array(hex_color: str, n_reports: pd.Series) -> np.ndarray:
    """Per-point RGBA: alpha scales logarithmically with n_reports."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    alphas = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * np.log1p(n_reports) / np.log1p(_nr_max)
    out = np.ones((len(n_reports), 4))
    out[:, :3] = rgb
    out[:, 3] = np.array(alphas)
    return out


# ---------------------------------------------------------------------------
# Layout: main scatter [4] + KDE strip [1]
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(9.5, 5.0), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.04)
ax = fig.add_subplot(gs[0])
ax_kde = fig.add_subplot(gs[1])

# ---- Enthalpy window (shaded, both borders dashed) ----
ax.axvspan(WIN_LO, WIN_HI, color="0.87", alpha=0.35, zorder=0)
ax.axvline(WIN_LO, color="0.45", lw=1.0, ls="--", zorder=1)
ax.axvline(WIN_HI, color="0.45", lw=1.0, ls="--", zorder=1)
ax.text(-5.0, 0.035, "enthalpy\nwindow", ha="center", va="bottom",
        fontsize=8.5, color="0.4", linespacing=1.3, zorder=1,
        transform=ax.get_xaxis_transform())

# Deployment cap (mu_dH > +8 -> P=0) is an atlas-only safeguard; it is NOT drawn here
# because this panel shows leave-one-out P(HEA), in which the confirmed HEAs above +8 keep
# their (non-zero) probabilities. Drawing a threshold line would contradict the figure's
# point that P(HEA) is continuous and never thresholded.

# ---- Scatter: size = sigma_dH, alpha = n_reports ----
ax.scatter(non.mu_dH, non.P_HEA_loocv_cal,
           s=sigma_to_s(non.sigma_dH),
           c=rgba_array(C_NON, non.n_reports),
           marker="x", lw=1.0, zorder=2)
ax.scatter(hea.mu_dH, hea.P_HEA_loocv_cal,
           s=sigma_to_s(hea.sigma_dH),
           facecolors="none",
           edgecolors=rgba_array(C_HEA, hea.n_reports),
           lw=1.1, zorder=3)

# ---- Binned-mean trend lines (10 bins, ≥4 pts/bin) ----
BINS = np.linspace(-50, 12, 10)
for subset, c in [(hea, C_HEA), (non, C_NON)]:
    means, edges, _ = binned_statistic(
        subset.mu_dH, subset.P_HEA_loocv_cal, statistic="mean", bins=BINS)
    counts, _, _ = binned_statistic(
        subset.mu_dH, subset.P_HEA_loocv_cal, statistic="count", bins=BINS)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= 4
    ax.plot(centers[mask], means[mask], color=c, lw=2.2, alpha=0.6,
            zorder=4, solid_capstyle="round")

# ---- Annotation: axes fraction anchor, arc curves LEFT (rad > 0) ----
ax.annotate(
    "confirmed HEAs\nabove the window,\nrecovered at\nhigh $P$(HEA)",
    xy=(7.2, 0.87),
    xytext=(0.84, 0.30),
    xycoords="data",
    textcoords="axes fraction",
    fontsize=8.5, color=C_HEA, ha="center", va="center",
    arrowprops=dict(arrowstyle="->", color=C_HEA, lw=0.9,
                    connectionstyle="arc3,rad=0.22"),
)

# ---- Legend (manual handles, fixed size/alpha) ----
legend_handles = [
    Line2D([0], [0], marker="o", color="none", markeredgecolor=C_HEA,
           markersize=7, markeredgewidth=1.1, label="experimental HEA"),
    Line2D([0], [0], marker="x", color="none", markeredgecolor=C_NON,
           markersize=7, markeredgewidth=1.0, label="experimental non-HEA"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper left", framealpha=0.9)

ax.set_xlabel(r"propagated enthalpy $\mu_{\Delta H}$ (kJ mol$^{-1}$)", fontsize=12)
ax.set_ylabel(r"$P(\mathrm{HEA})$  (leave-one-out)", fontsize=12)
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(d.mu_dH.min() - 2, d.mu_dH.max() + 3)
ax.grid(True, lw=0.4, color="0.9", zorder=0)

# ---------------------------------------------------------------------------
# Dashed rectangle marking the recovered band (confirmed HEAs above the window)
# ---------------------------------------------------------------------------
ZOOM_X = (3.5, 13.5)
ZOOM_Y = (0.42, 1.04)

ax.add_patch(Rectangle(
    (ZOOM_X[0], ZOOM_Y[0]),
    ZOOM_X[1] - ZOOM_X[0],
    ZOOM_Y[1] - ZOOM_Y[0],
    linewidth=1.0, edgecolor=C_HEA, facecolor="none", linestyle="--", zorder=5,
))

# ---------------------------------------------------------------------------
# KDE strip (right panel)
# ---------------------------------------------------------------------------
p_grid = np.linspace(0.0, 1.0, 300)
for subset, c in [(hea, C_HEA), (non, C_NON)]:
    kde = gaussian_kde(subset.P_HEA_loocv_cal, bw_method=0.15)
    dens = kde(p_grid)
    ax_kde.plot(dens, p_grid, color=c, lw=1.8)
    ax_kde.fill_betweenx(p_grid, 0, dens, alpha=0.15, color=c)

ax_kde.set_ylim(-0.02, 1.05)
ax_kde.set_xlabel("density", fontsize=9, labelpad=3)
ax_kde.yaxis.set_visible(False)
ax_kde.tick_params(axis="x", labelsize=7.5)
ax_kde.grid(True, lw=0.3, color="0.9", axis="y")
ax_kde.set_title(r"$P$(HEA)" + "\ndistribution", fontsize=8.5, pad=4, color="0.4")
ax_kde.spines[["top", "right", "left"]].set_visible(False)

# ---------------------------------------------------------------------------
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIGS / f"landscape_phea_mu.{ext}", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_FIGS / 'landscape_phea_mu.pdf'}")
