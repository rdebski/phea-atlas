"""
Atlas element-pair maps (paper Figure 5) — standalone figure regeneration.

Reads the pre-computed atlas (out/data/atlas_phea.csv) and the GP model; does
NOT recompute the 10-min atlas. Produces three figures:

  atlas_phea_heatmap.pdf   44x44 best-case P(HEA) per element pair
  atlas_sigma_heatmap.pdf  44x44 native GP uncertainty sigma_GP(A,B)
  atlas_group_panel.pdf    6x6  group-level executive summary (best P + mean sigma)

Design notes (why this differs from the old version)
-----------------------------------------------------
* Statistic: per pair (A,B) we report max over the subsets containing both of
  `mean_P` (best achievable subset-level P), NOT the mean over all containing
  subsets. The mean is doubly diluted (mean_P is itself a 1451-composition mean)
  and collapses the dynamic range to 0.004-0.071 while surfacing a misleading
  Ag-dominated ranking. The max recovers the known late-3d-TM chemistry
  (Co-Cr-Cu-Fe-Ni, max(mean_P)=0.467) and matches the top-subset table.
* Uncertainty panel: native per-pair GP sigma_GP(A,B). The subset-averaged
  sigma washes out the OOD signal (Bi looks average); the native pair sigma is
  the sharp, defensible quantity.
* Elements are ordered by chemical group (not alphabetically) so the late-TM
  block reads as one bright square. Sequential, perceptually-uniform colormaps
  (viridis / cividis), scales matched to the data range, larger fonts.
"""

from __future__ import annotations

import sys
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

# ---------------------------------------------------------------------------
# Chemical-group decomposition of the 44-element pool.
#   Bi is a post-transition (poor) metal -> Post-TM, NOT a metalloid.
#   "Metalloids" replaces the mislabelled "Semimetals" bucket
#   (B, Si, Ge, Sb metalloids + Se, Te chalcogens), consistent with the
#   CLAUDE.md rule "Bi and metalloids, not semimetals".
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "Alkali / AE": ["Li", "Na", "Mg", "Ca", "Sr", "Ba"],
    "Early TM":    ["Sc", "Ti", "Y", "Zr", "Nb"],
    "Late TM":     ["Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                    "Rh", "Pd", "Ag", "Cd", "Ir", "Pt", "Au", "Hg"],
    "Lanthanides": ["Ce", "Eu", "Gd", "Tb", "Yb"],
    "Post-TM":     ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi"],
    "Metalloids":  ["B", "Si", "Ge", "Sb", "Se", "Te"],
}
ORDER = [e for g in GROUPS.values() for e in g]
assert len(ORDER) == 44 and len(set(ORDER)) == 44, "group decomposition must cover the 44-element pool exactly"
IDX = {e: i for i, e in enumerate(ORDER)}
N = len(ORDER)

GROUP_SIZES = [len(v) for v in GROUPS.values()]
GROUP_NAMES = list(GROUPS.keys())
# short labels for the crowded 44x44 margins (full names used in the group panel)
GROUP_ABBR = {
    "Alkali / AE": "Alk/AE", "Early TM": "Early TM", "Late TM": "Late TM",
    "Lanthanides": "Lanth.", "Post-TM": "Post-TM", "Metalloids": "Metalloid",
}
GROUP_STARTS = np.cumsum([0] + GROUP_SIZES[:-1])
GROUP_ENDS = np.cumsum(GROUP_SIZES)
GROUP_MIDS = (GROUP_STARTS + GROUP_ENDS - 1) / 2.0

# fonts (44x44 maps render full-width in the appendix, so these can be large)
FS_ELEM = 11
FS_GROUP = 15
FS_LABEL = 16
FS_TITLE = 17
FS_CBAR = 13


# ---------------------------------------------------------------------------
# Pair statistics
# ---------------------------------------------------------------------------
def pair_best_P(atlas: pd.DataFrame) -> np.ndarray:
    """44x44 matrix: per pair (A,B), max of mean_P over subsets containing both."""
    best: dict[tuple[str, str], float] = {}
    elcols = ["el1", "el2", "el3", "el4", "el5"]
    for els, mp in zip(atlas[elcols].to_numpy(), atlas["mean_P"].to_numpy()):
        if np.isnan(mp):
            continue
        for a, b in combinations(sorted(els), 2):
            k = (a, b)
            if mp > best.get(k, -1.0):
                best[k] = mp
    M = np.full((N, N), np.nan)
    for (a, b), v in best.items():
        i, j = IDX[a], IDX[b]
        M[i, j] = M[j, i] = v
    return M


def pair_sigma_gp() -> np.ndarray:
    """44x44 matrix of native GP total sigma_GP(A,B) [kJ/mol]."""
    from src.gp.predict import GPPredictor
    gp = GPPredictor.load(str(ROOT / "models" / "gp_full_model.pt"))
    res = gp.predict_all_pairs(ORDER, use_model_std=False)
    M = np.full((N, N), np.nan)
    for (a, b), (_, s, _) in res.items():
        i, j = IDX[a], IDX[b]
        M[i, j] = M[j, i] = s
    return M


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _decorate_groups(ax, *, label_axes: bool = True) -> None:
    """Group separator lines + bold group names along the top and left margins."""
    for s in GROUP_STARTS[1:]:
        ax.axhline(s - 0.5, color="white", lw=1.6)
        ax.axvline(s - 0.5, color="white", lw=1.6)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(ORDER, rotation=90, fontsize=FS_ELEM)
    ax.set_yticklabels(ORDER, fontsize=FS_ELEM)
    ax.tick_params(length=0)
    if label_axes:
        for mid, name in zip(GROUP_MIDS, GROUP_NAMES):
            # Top group labels sit just above the matrix (y=-1.2): far enough below
            # the title (pad=40) to stop the two colliding, close enough to read as
            # column headers.
            ax.text(mid, -1.2, GROUP_ABBR[name], ha="center", va="bottom",
                    fontsize=FS_GROUP, fontweight="bold", rotation=0)
            ax.text(-3.0, mid, GROUP_ABBR[name], ha="right", va="center",
                    fontsize=FS_GROUP, fontweight="bold", rotation=90)


def heatmap(M, *, cmap, vmin, vmax, cbar_label, title, outname) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 10))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_CBAR)
    _decorate_groups(ax)
    ax.set_title(title, fontsize=FS_TITLE, pad=40)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIGS / f"{outname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_FIGS / (outname + '.pdf')}")


def group_panel(bestP, sigma) -> None:
    """6x6 executive summary: best achievable P (left) and mean sigma_GP (right)."""
    gP = np.full((len(GROUPS), len(GROUPS)), np.nan)
    gS = np.full((len(GROUPS), len(GROUPS)), np.nan)
    for gi, ni in enumerate(GROUP_NAMES):
        for gj, nj in enumerate(GROUP_NAMES):
            ii = [IDX[e] for e in GROUPS[ni]]
            jj = [IDX[e] for e in GROUPS[nj]]
            blockP = bestP[np.ix_(ii, jj)]
            blockS = sigma[np.ix_(ii, jj)]
            gP[gi, gj] = np.nanmax(blockP) if np.isfinite(blockP).any() else np.nan
            gS[gi, gj] = np.nanmean(blockS) if np.isfinite(blockS).any() else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    specs = [
        (gP, "viridis", 0.0, np.nanmax(bestP), "highest subset-mean $P(\\mathrm{HEA})$",
         "Highest subset-mean $P(\\mathrm{HEA})$ by group pair", "%.2f"),
        (gS, "cividis", np.nanmin(sigma), np.nanmax(sigma), "mean $\\sigma_{\\mathrm{GP}}$  [kJ/mol]",
         "Mean GP uncertainty by group pair", "%.1f"),
    ]
    for ax, (G, cmap, vmin, vmax, clab, title, fmt) in zip(axes, specs):
        im = ax.imshow(G, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(clab, fontsize=13)
        cb.ax.tick_params(labelsize=11)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_yticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUP_NAMES, rotation=35, ha="right", fontsize=12.5)
        ax.set_yticklabels(GROUP_NAMES, fontsize=12.5)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=14, pad=8)
        # annotate cells, with contrast-aware text colour
        norm = (G - vmin) / (vmax - vmin)
        for i in range(len(GROUPS)):
            for j in range(len(GROUPS)):
                if not np.isfinite(G[i, j]):
                    continue
                tc = "white" if (cmap == "viridis" and norm[i, j] < 0.55) or \
                                (cmap == "cividis" and norm[i, j] < 0.5) else "black"
                ax.text(j, i, fmt % G[i, j], ha="center", va="center",
                        fontsize=11.5, color=tc)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIGS / f"atlas_group_panel.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_FIGS / 'atlas_group_panel.pdf'}")


def make_atlas_maps(atlas: pd.DataFrame) -> None:
    """Entry point reused by run/08_exact_atlas.py and the standalone __main__."""
    valid = atlas[atlas["n_valid"] > 0]
    bestP = pair_best_P(valid)
    sigma = pair_sigma_gp()

    pmax = float(np.nanmax(bestP))
    print(f"  best-P range: [{np.nanmin(bestP):.3f}, {pmax:.3f}]")
    print(f"  sigma_GP range: [{np.nanmin(sigma):.2f}, {np.nanmax(sigma):.2f}] kJ/mol")

    heatmap(
        bestP, cmap="viridis", vmin=0.0, vmax=pmax,
        cbar_label="Highest subset-mean $P(\\mathrm{HEA})$ for pairs $(A,B)$",
        title="Highest subset-mean $P(\\mathrm{HEA})$ per element pair",
        outname="atlas_phea_heatmap",
    )
    heatmap(
        sigma, cmap="cividis",
        vmin=float(np.nanmin(sigma)), vmax=float(np.nanmax(sigma)),
        cbar_label="GP uncertainty $\\sigma_{\\mathrm{GP}}(A,B)$  [kJ/mol]",
        title="Native GP uncertainty per element pair",
        outname="atlas_sigma_heatmap",
    )
    group_panel(bestP, sigma)


if __name__ == "__main__":
    csv = OUT_DATA / "atlas_phea.csv"
    if not csv.exists():
        sys.exit(f"atlas not found: {csv} (run run/08_exact_atlas.py first)")
    print(f"Loading atlas from {csv} ...")
    make_atlas_maps(pd.read_csv(csv))
