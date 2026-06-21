"""
Prepare experimental HEA dataset for calibration layer training.

Input:  data/database_of_HEAs.csv
Output: out/data/exp_data_clean.csv   — unique k=5 compositions with labels
        out/data/exp_data_stats.json  — filter counts and label stats

HEA definition (Opcja B — broad solid solution):
  is_hea = 1 if the alloy forms a solid solution without intermetallic phases:
    • Type of solution == "Single Solid Solution"  (FCC, BCC, HCP)
    • Type of solution == "Mixed Solution" AND Phase contains only solid
      solution phases (BCC+FCC, FCC+HCP, BCC+HCP, BCC+FCC+HCP) with no
      intermetallic / ordered / amorphous components detected.

  is_hea = 0 for Intermetallic, Amorphous, High Entropy Intermetallic,
  and Mixed Solution entries where Phase contains B2, L12, Laves, martensite,
  sigma, tetragonal, "not specified", or other non-SS markers.

Composition filters:
  1. Experimental entries only
  2. Well-formed Alloy strings (no slash / parentheses / unicode separators)
  3. k = 5 elements exactly
  4. All elements in 44-element GP pool
  5. x_i >= 0.05 for every element
  6. ΔS_mix >= 1.5R

Output columns (per unique comp_key):
  comp_key, elements, composition, n_reports, n_positive, p_label,
  is_hea, conflict, structure_type, dS_over_R, min_frac, alloy_names
"""

import json
import math
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_IN = ROOT / "data" / "database_of_HEAs.csv"
OUT_DIR = ROOT / "out" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV   = OUT_DIR / "exp_data_clean.csv"
OUT_STATS = OUT_DIR / "exp_data_stats.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POOL = {
    "Ag", "Al", "Au", "B",  "Ba", "Bi", "Ca", "Cd", "Ce", "Co",
    "Cr", "Cu", "Eu", "Fe", "Ga", "Gd", "Ge", "Hg", "In", "Ir",
    "Li", "Mg", "Mn", "Na", "Nb", "Ni", "Pb", "Pd", "Pt", "Rh",
    "Sb", "Sc", "Se", "Si", "Sn", "Sr", "Tb", "Te", "Ti", "Tl",
    "Y",  "Yb", "Zn", "Zr",
}
X_MIN  = 0.05
DS_MIN = 1.5

# ---------------------------------------------------------------------------
# Phase parsing
# ---------------------------------------------------------------------------

# Markers that indicate non-solid-solution components
_INTERMETALLIC_PAT = re.compile(
    r"\b("
    r"b2|l12|l21|laves|sigma|kappa|omega|mu\b|"
    r"intermetallic|martensite|amorphous|glass|"
    r"tetragonal|orthorhombic|monoclinic|"
    r"c14|c15|"
    r"precipitat|oxide|carbide|nitride|silicide|boride|"
    r"spinel|perovskite|antifluorite|eutectic|"
    r"not specified|not explicitly|varies|unknown"
    r")",
    re.IGNORECASE,
)

_FCC_PAT = re.compile(r"\b(fcc|a1|face.?cent(?:er|re)d)\b", re.IGNORECASE)
_BCC_PAT = re.compile(r"\b(bcc|a2|body.?cent(?:er|re)d)\b", re.IGNORECASE)
_HCP_PAT = re.compile(r"\b(hcp|a3|hexagonal)\b", re.IGNORECASE)


def classify_phase(phase_str: str, sol_type: str) -> tuple[bool, str]:
    """
    Returns (is_pure_ss, structure_type).

    is_pure_ss : True if only solid-solution phases detected.
    structure_type : canonical label e.g. "FCC", "BCC+FCC", "BCC+FCC+HCP",
                     "mixed_with_IM" (solid SS + intermetallic),
                     "intermetallic", "amorphous", "unknown".
    """
    # Single Solid Solution entries — trust the database label
    if sol_type == "Single Solid Solution":
        phases = []
        p = str(phase_str)
        if _FCC_PAT.search(p): phases.append("FCC")
        if _BCC_PAT.search(p): phases.append("BCC")
        if _HCP_PAT.search(p): phases.append("HCP")
        struct = "+".join(sorted(phases)) if phases else "SS_unknown"
        return True, struct

    if sol_type == "Intermetallic":
        return False, "intermetallic"

    if sol_type == "Amorphous":
        return False, "amorphous"

    if sol_type == "High Entropy Intermetallic":
        return False, "intermetallic"

    # Mixed Solution — parse Phase string
    p = str(phase_str) if not pd.isna(phase_str) else ""

    has_intermetallic = bool(_INTERMETALLIC_PAT.search(p))
    has_fcc = bool(_FCC_PAT.search(p))
    has_bcc = bool(_BCC_PAT.search(p))
    has_hcp = bool(_HCP_PAT.search(p))
    has_ss  = has_fcc or has_bcc or has_hcp

    if has_intermetallic and has_ss:
        return False, "mixed_with_IM"
    if has_intermetallic and not has_ss:
        return False, "intermetallic"
    if not has_ss:
        return False, "unknown"

    # Pure solid solution combination
    phases = []
    if has_fcc: phases.append("FCC")
    if has_bcc: phases.append("BCC")
    if has_hcp: phases.append("HCP")
    return True, "+".join(sorted(phases))


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

def is_experimental(val) -> bool:
    if pd.isna(val):
        return False
    return "experimental" in str(val).lower()


def parse_composition(alloy_str: str) -> dict[str, float] | None:
    s = str(alloy_str).strip()
    if any(c in s for c in ("(", ")", "[", "]", "/", "·")):
        return None
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", s)
    if not tokens:
        return None
    raw: dict[str, float] = {}
    for elem, val_str in tokens:
        val = float(val_str) if val_str else 1.0
        raw[elem] = raw.get(elem, 0.0) + val
    total = sum(raw.values())
    if total == 0:
        return None
    return dict(sorted({e: v / total for e, v in raw.items()}.items()))


def make_comp_key(comp: dict[str, float]) -> str:
    return "|".join(f"{e}:{v:.4f}" for e, v in comp.items())


def delta_s_over_R(comp: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in comp.values() if v > 0)


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(DATA_IN)
n0 = len(df_raw)

df = df_raw[df_raw["Experimental or theoretical"].apply(is_experimental)].copy()
n_exp = len(df)

df = df.dropna(subset=["Alloy"])
df["composition"] = df["Alloy"].apply(parse_composition)
df = df[df["composition"].notna()].copy()
n_parsed = len(df)

df["n_elem"] = df["composition"].apply(len)
df = df[df["n_elem"] == 5].copy()
n_k5 = len(df)

df["elements"] = df["composition"].apply(lambda c: tuple(c.keys()))
df["in_pool"]  = df["elements"].apply(lambda els: all(e in POOL for e in els))
df = df[df["in_pool"]].copy()
n_pool = len(df)

df["min_frac"]  = df["composition"].apply(lambda c: min(c.values()))
df = df[df["min_frac"] >= X_MIN].copy()
n_xmin = len(df)

df["dS_over_R"] = df["composition"].apply(delta_s_over_R)
df = df[df["dS_over_R"] >= DS_MIN].copy()
n_ds = len(df)

# ---------------------------------------------------------------------------
# Labels — Opcja B
# ---------------------------------------------------------------------------
phase_class = df.apply(
    lambda row: classify_phase(row["Phase"], row["Type of solution"]),
    axis=1,
)
df["is_pure_ss"]     = phase_class.apply(lambda t: t[0])
df["structure_type"] = phase_class.apply(lambda t: t[1])
df["is_hea_raw"]     = df["is_pure_ss"].astype(int)
df["comp_key"]       = df["composition"].apply(make_comp_key)

# ---------------------------------------------------------------------------
# Aggregate duplicate comp_keys
# ---------------------------------------------------------------------------

def most_common(series):
    vc = series.value_counts()
    return vc.index[0] if len(vc) else "unknown"


agg = (
    df.groupby("comp_key")
    .agg(
        n_reports   = ("is_hea_raw", "count"),
        n_positive  = ("is_hea_raw", "sum"),
        elements    = ("elements", "first"),
        composition = ("composition", "first"),
        dS_over_R   = ("dS_over_R", "first"),
        min_frac    = ("min_frac", "first"),
        alloy_names = ("Alloy", lambda x: sorted(x.unique().tolist())),
    )
    .reset_index()
)

# structure_type: most common type among positive reports;
# fall back to most common among all reports if no positives.
def agg_structure(grp):
    pos = grp[grp["is_hea_raw"] == 1]["structure_type"]
    if len(pos):
        return most_common(pos)
    return most_common(grp["structure_type"])

agg["structure_type"] = df.groupby("comp_key").apply(agg_structure).values

agg["p_label"]  = agg["n_positive"] / agg["n_reports"]
agg["is_hea"]   = (agg["p_label"] >= 0.5).astype(int)
agg["conflict"] = (agg["n_positive"] > 0) & (agg["n_positive"] < agg["n_reports"])

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
stats = {
    "filter_steps": {
        "raw":                n0,
        "after_experimental": n_exp,
        "after_parse":        n_parsed,
        "after_k5":           n_k5,
        "after_pool":         n_pool,
        "after_xmin_0.05":    n_xmin,
        "after_dS_1.5R":      n_ds,
    },
    "unique_compositions": int(len(agg)),
    "n_hea_1":          int((agg["is_hea"] == 1).sum()),
    "n_hea_0":          int((agg["is_hea"] == 0).sum()),
    "n_conflict":       int(agg["conflict"].sum()),
    "n_single_report":  int((agg["n_reports"] == 1).sum()),
    "n_multi_report":   int((agg["n_reports"] > 1).sum()),
}

print("=" * 55)
print("FILTER PIPELINE")
print("=" * 55)
for step, n in stats["filter_steps"].items():
    print(f"  {step:<25s}: {n:>6d}")

print()
print("=" * 55)
print("FINAL DATASET")
print("=" * 55)
print(f"  Unique compositions : {stats['unique_compositions']}")
print(f"  is_hea = 1          : {stats['n_hea_1']}")
print(f"  is_hea = 0          : {stats['n_hea_0']}")
print(f"  Conflicting labels  : {stats['n_conflict']}")
print(f"  Single-report       : {stats['n_single_report']}")
print(f"  Multi-report        : {stats['n_multi_report']}")

print()
print("structure_type distribution (is_hea=1):")
pos = agg[agg["is_hea"] == 1]
print(pos["structure_type"].value_counts().to_string())

print()
print("structure_type distribution (is_hea=0):")
neg = agg[agg["is_hea"] == 0]
print(neg["structure_type"].value_counts().head(10).to_string())

print()
print("p_label distribution:")
cuts = pd.cut(
    agg["p_label"],
    bins=[-0.01, 0.001, 0.25, 0.50, 0.75, 0.999, 1.001],
    labels=["0.0", "0.01–0.25", "0.26–0.50", "0.51–0.75", "0.76–0.99", "1.0"],
)
print(cuts.value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = agg.copy()
out["composition"] = out["composition"].apply(json.dumps)
out["elements"]    = out["elements"].apply(list)
out["alloy_names"] = out["alloy_names"].apply(json.dumps)

cols = [
    "comp_key", "elements", "composition", "n_reports", "n_positive",
    "p_label", "is_hea", "conflict", "structure_type",
    "dS_over_R", "min_frac", "alloy_names",
]
out[cols].to_csv(OUT_CSV, index=False)

with open(OUT_STATS, "w") as fh:
    json.dump(stats, fh, indent=2)

print()
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_STATS}")
