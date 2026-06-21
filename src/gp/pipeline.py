"""
4-phase HEA screening pipeline: deterministic (D) vs. probabilistic (S).

Phases
------
0  Composition generation  — equimolar C(N,k) combinations OR non-equimolar grid
1  Geometric filter        — δ (size mismatch), Λ = ΔS/δ² (entropy-size ratio)
2  Deterministic filter    — Miedema ΔH_mix, Ω = T_m·ΔS/|ΔH_mix| → set D
3  Probabilistic filter    — GP posterior P(H_mix ∈ A∩B) → set S

    A = { ΔH_mix ∈ [H_lo, H_hi] }
    B = { Ω ≥ Ω_min }  ≡  { |ΔH_mix| ≤ c },   c = T_m·ΔS / (Ω_min·1000)
    A∩B = { ΔH_mix ∈ [max(H_lo, −c),  min(H_hi, c)] }

    prob_pass = Φ((hi−μ)/σ) − Φ((lo−μ)/σ)   [fully analytic, no Monte Carlo]

Core result: D∩S, D\\S (Miedema false alarms), S\\D (GP-only candidates).

Non-equimolar mode (screen_nonequimolar)
----------------------------------------
For a fixed N-element set, generates all compositions on a regular grid with
x_i ∈ [x_min, x_max] and Σx_i = 1.  GP predictions are computed ONCE for each
of the C(N,2) binary pairs and then propagated analytically to all compositions:

  ΔH_mix  = Σ_{i<j} 4·xi·xj · h05(i,j)                   [R-K, L0 only]
  σ²_ΔH   = Σ_{i<j} (4·xi·xj)² · σ²_GP(i,j)              [propagation]
  prob_pass = Φ((hi−μ)/σ) − Φ((lo−μ)/σ)                   [analytic A∩B]
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parent.parent.parent

R_GAS = 8.314  # J/(mol·K)

# HEA screening thresholds (defaults; all overridable in screen_pool)
DH_WINDOW   = (-15.0, +5.0)   # kJ/mol
OMEGA_MIN   = 1.1              # Yang & Zhang 2012
DELTA_MAX   = 0.065            # dimensionless (= 6.5%)
LAMBDA_MIN  = 0.24             # Singh et al. 2014 (Λ = ΔS/R / δ²)
EPSILON     = 0.75             # probability threshold for set S


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AlloyResult:
    elements: tuple[str, ...]
    k: int                          # number of components
    x_eq: float                     # equimolar fraction = 1/k

    # Phase 1 — geometric
    delta: float                    # size mismatch δ (dimensionless)
    lambda_param: float             # Λ = ΔS/R / δ²
    ds_mix_J: float                 # ΔS_mix [J/(mol·K)]
    pass_phase1: bool

    # Phase 2 — deterministic (Miedema)
    dh_miedema: float               # ΔH_mix Miedema [kJ/mol]
    t_m_mean: float                 # mean melting temperature [K]
    omega_miedema: float            # Ω (Miedema ΔH)
    pass_phase2: bool               # ∈ D

    # Phase 3 — probabilistic (GP)
    dh_gp_mean: float               # GP mean ΔH_mix [kJ/mol]
    dh_gp_std: float                # GP std  ΔH_mix [kJ/mol]
    prob_dh: float                  # P(ΔH_mix ∈ window)  — diagnostic
    prob_omega: float               # P(Ω ≥ omega_min)    — diagnostic
    prob_pass: float                # P(A∩B) — analytic combined criterion
    pass_phase3: bool               # ∈ S

    # Convenience
    in_D_not_S: bool = field(init=False)
    in_S_not_D: bool = field(init=False)
    in_D_and_S: bool = field(init=False)

    def __post_init__(self):
        self.in_D_not_S = self.pass_phase2 and not self.pass_phase3
        self.in_S_not_D = self.pass_phase3 and not self.pass_phase2
        self.in_D_and_S = self.pass_phase2 and self.pass_phase3

    @property
    def label(self) -> str:
        return "-".join(self.elements)


# ---------------------------------------------------------------------------
# Phase 1: geometric parameters
# ---------------------------------------------------------------------------

def _geometric_params(
    elements: tuple[str, ...],
    ep: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Compute δ, Λ, ΔS_mix for equimolar alloy.

    δ       = sqrt(Σ x_i · (1 - r_i / r̄)²)           [dimensionless]
    ΔS_mix  = -R · Σ x_i · ln(x_i) = R·ln(k)          [J/(mol·K)]
    Λ       = (ΔS_mix/R) / δ²   = ln(k) / δ²           [dimensionless]

    where δ is in decimal form (0.065 = 6.5%).
    """
    k   = len(elements)
    x   = 1.0 / k
    r   = np.array([float(ep.loc[e, "r_atomic"]) for e in elements])
    r_bar = r.mean()

    delta     = float(np.sqrt(x * np.sum((1.0 - r / r_bar) ** 2)))
    ds_mix    = R_GAS * np.log(k)
    lambda_p  = np.log(k) / (delta ** 2) if delta > 0 else np.inf

    return delta, lambda_p, ds_mix


# ---------------------------------------------------------------------------
# Phase 2: deterministic thermodynamic filter
# ---------------------------------------------------------------------------

def _deterministic_thermo(
    elements: tuple[str, ...],
    miedema_model,
    ep: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Compute ΔH_mix (Miedema), T_m (linear), Ω for equimolar alloy.

    ΔH_mix = (4/k²) · Σ_{i<j} h05_miedema(i,j)

    Ω = T_m · ΔS_mix / |ΔH_mix|   [Yang & Zhang 2012]
    Returns (dh_miedema, t_m_mean, omega).
    """
    k = len(elements)
    x = 1.0 / k

    dh_sum = 0.0
    for i, a in enumerate(elements):
        for b in elements[i + 1 :]:
            a_c, b_c = (a, b) if a < b else (b, a)
            dh_sum += miedema_model.h_mix_fn(a_c, b_c, 0.5)

    dh = (4.0 / k ** 2) * dh_sum

    t_m = float(np.mean([float(ep.loc[e, "T_melt"]) for e in elements]))
    ds  = R_GAS * np.log(k)  # J/(mol·K)

    # Ω = T_m [K] · ΔS_mix [J/(mol·K)] / |ΔH_mix [J/mol]|
    # ΔH_mix is in kJ/mol → convert to J/mol
    omega = (t_m * ds / abs(dh * 1000.0)) if abs(dh) > 1e-6 else np.inf

    return dh, t_m, omega


def _passes_phase2(
    dh: float,
    omega: float,
    dh_window: tuple[float, float],
    omega_min: float,
) -> bool:
    return dh_window[0] <= dh <= dh_window[1] and omega >= omega_min


# ---------------------------------------------------------------------------
# Phase 3: probabilistic filter
# ---------------------------------------------------------------------------

def _probabilistic_thermo(
    elements: tuple[str, ...],
    predictor,
    ep: pd.DataFrame,
    dh_window: tuple[float, float],
    omega_min: float,
) -> tuple[float, float, float, float, float]:
    """
    Compute GP posterior ΔH_mix distribution and combined analytic probability.

    ΔH_mix ~ N(μ_H, σ_H²) under independence assumption across pairs.

    Criteria:
        A = { ΔH_mix ∈ [H_lo, H_hi] }
        B = { Ω ≥ Ω_min } ≡ { |ΔH_mix| ≤ c },  c = T_m·ΔS / (Ω_min·1000)
        A∩B = { ΔH_mix ∈ [max(H_lo, −c),  min(H_hi, c)] }

    Returns (dh_mean, dh_std, prob_dh, prob_omega, prob_pass).
    prob_dh and prob_omega are kept as diagnostics; prob_pass = P(A∩B).
    """
    k = len(elements)
    t_m = float(np.mean([float(ep.loc[e, "T_melt"]) for e in elements]))
    ds  = R_GAS * np.log(k)

    # Collect GP predictions for all C(k,2) pairs
    w       = 4.0 / k ** 2          # weight for each binary contribution
    mu_sum  = 0.0
    var_sum = 0.0

    for i, a in enumerate(elements):
        for b in elements[i + 1 :]:
            a_c, b_c = (a, b) if a < b else (b, a)
            h05_mean, h05_std, _ = predictor.predict(a_c, b_c, use_model_std=False)
            mu_sum  += h05_mean
            var_sum += h05_std ** 2

    dh_mean = w * mu_sum
    dh_std  = w * np.sqrt(var_sum)   # propagated under independence

    sig = dh_std if dh_std >= 1e-8 else 1e-8

    # P(ΔH_mix ∈ [H_lo, H_hi]) — diagnostic
    lo, hi = dh_window
    prob_dh = float(norm.cdf(hi, dh_mean, sig) - norm.cdf(lo, dh_mean, sig))

    # P(Ω ≥ Ω_min) ≡ P(|ΔH_mix| ≤ c) — diagnostic
    c = t_m * ds / (omega_min * 1000.0)   # kJ/mol
    prob_omega = float(norm.cdf(c, dh_mean, sig) - norm.cdf(-c, dh_mean, sig))

    # P(A∩B) — combined analytic criterion (exact for Gaussian ΔH_mix)
    lo_ab = max(dh_window[0], -c)
    hi_ab = min(dh_window[1],  c)
    prob_pass = float(norm.cdf(hi_ab, dh_mean, sig) - norm.cdf(lo_ab, dh_mean, sig))

    return dh_mean, dh_std, prob_dh, prob_omega, prob_pass


# ---------------------------------------------------------------------------
# Main screening function
# ---------------------------------------------------------------------------

def screen_pool(
    element_pool: Sequence[str],
    predictor,
    k_range: range = range(3, 6),
    dh_window: tuple[float, float] = DH_WINDOW,
    omega_min: float = OMEGA_MIN,
    delta_max: float = DELTA_MAX,
    lambda_min: float = LAMBDA_MIN,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """
    Screen all equimolar k-component alloys from element_pool.

    Parameters
    ----------
    element_pool : list of element symbols
    predictor    : GPPredictor (trained)
    k_range      : sizes to enumerate (default 3, 4, 5)
    dh_window    : (lo, hi) kJ/mol for ΔH_mix criterion
    omega_min    : Ω threshold (≥ means entropy favours mixing)
    delta_max    : maximum size mismatch δ (dimensionless, e.g. 0.065)
    lambda_min   : minimum Λ = ln(k)/δ² (dimensionless)
    epsilon      : probability threshold for set S membership

    Returns
    -------
    pd.DataFrame with one row per alloy, columns matching AlloyResult fields.
    """
    from src.data_preparation.miedema import MiedemaModel
    import pandas as pd

    mp_raw = pd.read_csv(ROOT / "data/miedema_params.csv")
    ep_raw = pd.read_csv(ROOT / "data/periodic_table/element_properties.csv")
    ep = ep_raw.set_index("elem")
    miedema = MiedemaModel(mp_raw)

    records: list[AlloyResult] = []

    for k in k_range:
        combos = list(itertools.combinations(sorted(element_pool), k))
        print(f"  k={k}: {len(combos)} combinations")

        for combo in combos:
            elements = combo   # already sorted alphabetically

            # Phase 1
            delta, lambda_p, ds_mix = _geometric_params(elements, ep)
            pass1 = (delta <= delta_max) and (lambda_p >= lambda_min)

            # Phase 2
            dh_mied, t_m, omega_mied = _deterministic_thermo(elements, miedema, ep)
            pass2 = _passes_phase2(dh_mied, omega_mied, dh_window, omega_min)

            # Phase 3 (always run, even if Phase 1/2 failed — informative)
            dh_gp_mean, dh_gp_std, prob_dh, prob_omega, prob_pass = _probabilistic_thermo(
                elements, predictor, ep, dh_window, omega_min
            )
            pass3 = prob_pass >= epsilon

            rec = AlloyResult(
                elements       = elements,
                k              = k,
                x_eq           = 1.0 / k,
                delta          = delta,
                lambda_param   = lambda_p,
                ds_mix_J       = ds_mix,
                pass_phase1    = pass1,
                dh_miedema     = dh_mied,
                t_m_mean       = t_m,
                omega_miedema  = omega_mied,
                pass_phase2    = pass2,
                dh_gp_mean     = dh_gp_mean,
                dh_gp_std      = dh_gp_std,
                prob_dh        = prob_dh,
                prob_omega     = prob_omega,
                prob_pass      = prob_pass,
                pass_phase3    = pass3,
            )
            records.append(rec)

    return _to_dataframe(records)


def _to_dataframe(records: list[AlloyResult]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "alloy":          r.label,
                "elements":       r.elements,
                "k":              r.k,
                "delta_pct":      r.delta * 100,
                "lambda":         r.lambda_param,
                "ds_mix_J":       r.ds_mix_J,
                "pass_phase1":    r.pass_phase1,
                "dh_miedema":     r.dh_miedema,
                "t_m_K":          r.t_m_mean,
                "omega_miedema":  r.omega_miedema,
                "pass_phase2":    r.pass_phase2,       # ∈ D
                "dh_gp_mean":     r.dh_gp_mean,
                "dh_gp_std":      r.dh_gp_std,
                "prob_dh":        r.prob_dh,
                "prob_omega":     r.prob_omega,
                "prob_pass":      r.prob_pass,
                "pass_phase3":    r.pass_phase3,       # ∈ S
                "in_D_and_S":     r.in_D_and_S,
                "in_D_not_S":     r.in_D_not_S,       # false alarms
                "in_S_not_D":     r.in_S_not_D,       # missed candidates
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, epsilon: float = EPSILON) -> None:
    """Print Phase 2 vs. Phase 3 comparison table."""
    D = df[df["pass_phase2"]]
    S = df[df["pass_phase3"]]
    both  = df[df["in_D_and_S"]]
    d_not_s = df[df["in_D_not_S"]]
    s_not_d = df[df["in_S_not_D"]]

    print(f"\n{'='*60}")
    print(f"  HEA Screening Summary  (ε = {epsilon:.2f})")
    print(f"{'='*60}")
    print(f"  Total alloys screened : {len(df)}")
    print(f"  Pass Phase 1 (geom.)  : {df['pass_phase1'].sum()}")
    print(f"  Set D  (Miedema)      : {len(D)}")
    print(f"  Set S  (GP, ε≥{epsilon}) : {len(S)}")
    print(f"  D ∩ S  (agreed)       : {len(both)}")
    print(f"  D \\ S  (false alarms) : {len(d_not_s)}")
    print(f"  S \\ D  (missed cand.) : {len(s_not_d)}")

    if len(d_not_s) > 0:
        print(f"\n  D\\S — Miedema says pass, GP uncertain:")
        for _, row in d_not_s.iterrows():
            print(f"    {row['alloy']:30s}  "
                  f"ΔH_mied={row['dh_miedema']:+6.1f}  "
                  f"ΔH_gp={row['dh_gp_mean']:+6.1f}±{row['dh_gp_std']:.1f}  "
                  f"P={row['prob_pass']:.2f}")

    if len(s_not_d) > 0:
        print(f"\n  S\\D — GP says pass, Miedema misses:")
        for _, row in s_not_d.iterrows():
            print(f"    {row['alloy']:30s}  "
                  f"ΔH_mied={row['dh_miedema']:+6.1f}  "
                  f"ΔH_gp={row['dh_gp_mean']:+6.1f}±{row['dh_gp_std']:.1f}  "
                  f"P={row['prob_pass']:.2f}")

    if len(both) > 0:
        print(f"\n  D∩S — both agree (pass):")
        for _, row in both.iterrows():
            print(f"    {row['alloy']:30s}  "
                  f"ΔH_mied={row['dh_miedema']:+6.1f}  "
                  f"ΔH_gp={row['dh_gp_mean']:+6.1f}±{row['dh_gp_std']:.1f}  "
                  f"P={row['prob_pass']:.2f}")
    print()


# ===========================================================================
# Non-equimolar screening (fixed element set, composition grid)
# ===========================================================================

def generate_compositions(
    n_elem: int = 5,
    x_min: float = 0.05,
    x_max: float = 0.35,
    step: float = 0.05,
) -> np.ndarray:
    """
    All compositions (x1,...,xN) on a regular grid with:
      x_i ∈ {x_min, x_min+step, ..., x_max}  and  Σx_i = 1.

    Returns
    -------
    np.ndarray of shape (M, n_elem), each row sums to 1.
    """
    import itertools

    # Represent fractions as integers to avoid float comparison issues
    scale   = round(1.0 / step)
    n_min   = round(x_min / step)
    n_max   = round(x_max / step)
    target  = scale

    grid = range(n_min, n_max + 1)
    rows = [
        tuple(v / scale for v in combo)
        for combo in itertools.product(grid, repeat=n_elem)
        if sum(combo) == target
    ]
    return np.array(rows)


def screen_nonequimolar(
    elements: list[str],
    predictor,
    x_min: float = 0.05,
    x_max: float = 0.35,
    step: float = 0.05,
    dh_window: tuple[float, float] = DH_WINDOW,
    omega_min: float = OMEGA_MIN,
    delta_max: float = DELTA_MAX,
    lambda_min: float = LAMBDA_MIN,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """
    Screen all non-equimolar compositions of a fixed N-element alloy.

    GP is called ONCE per binary pair (C(N,2) calls total), then all
    composition-dependent quantities are computed in vectorised NumPy.

    Parameters
    ----------
    elements  : ordered list of N element symbols
    predictor : GPPredictor (trained full-data GP)

    Returns
    -------
    DataFrame with one row per composition.
    """
    from src.data_preparation.miedema import MiedemaModel
    from scipy.stats import norm as sp_norm

    mp_raw = pd.read_csv(ROOT / "data/miedema_params.csv")
    ep_raw = pd.read_csv(ROOT / "data/periodic_table/element_properties.csv")
    ep = ep_raw.set_index("elem")
    miedema = MiedemaModel(mp_raw)

    elements = list(elements)
    n = len(elements)

    # ------------------------------------------------------------------
    # Step 1: precompute binary pair properties (only C(n,2) GP calls)
    # ------------------------------------------------------------------
    pairs = []           # list of (i, j) index pairs
    h05_mied = []        # Miedema h05 for each pair
    h05_gp_m  = []       # GP mean h05
    h05_gp_s  = []       # GP epistemic std h05

    for i in range(n):
        for j in range(i + 1, n):
            a, b = elements[i], elements[j]
            if a > b:
                a, b = b, a
            pairs.append((i, j))
            h05_mied.append(miedema.h_mix_fn(a, b, 0.5))
            mean, std, _ = predictor.predict(a, b, use_model_std=False)
            h05_gp_m.append(mean)
            h05_gp_s.append(std)

    h05_mied = np.array(h05_mied)
    h05_gp_m  = np.array(h05_gp_m)
    h05_gp_s  = np.array(h05_gp_s)

    # Element properties for geometric + Ω calculations
    r  = np.array([float(ep.loc[e, "r_atomic"]) for e in elements])
    tm = np.array([float(ep.loc[e, "T_melt"])   for e in elements])

    # ------------------------------------------------------------------
    # Step 2: generate all compositions (M × n matrix)
    # ------------------------------------------------------------------
    X = generate_compositions(n, x_min, x_max, step)   # (M, n)
    M = len(X)
    print(f"  {M} compositions generated for {'-'.join(elements)}")

    # ------------------------------------------------------------------
    # Step 3: vectorised physical properties
    # ------------------------------------------------------------------

    # Weights for each pair: w_{ij} = 4·xi·xj,  shape (M, n_pairs)
    W = np.column_stack([4.0 * X[:, i] * X[:, j] for i, j in pairs])  # (M, P)

    # Miedema ΔH_mix
    dh_mied_vec = W @ h05_mied            # (M,)

    # GP ΔH_mix mean and std
    dh_gp_mean_vec = W @ h05_gp_m         # (M,)
    dh_gp_var_vec  = (W ** 2) @ (h05_gp_s ** 2)  # (M,)
    dh_gp_std_vec  = np.sqrt(dh_gp_var_vec)       # (M,)

    # Thermodynamic properties per composition
    r_bar   = X @ r                                          # (M,) mean radius
    ds_J    = -R_GAS * np.sum(                               # (M,)
        np.where(X > 0, X * np.log(np.where(X > 0, X, 1.0)), 0.0),
        axis=1,
    )
    t_m_vec = X @ tm                                         # (M,)

    # δ = sqrt(Σ xi*(1 - ri/r_bar)²),  shape (M,)
    r_ratio   = r[np.newaxis, :] / r_bar[:, np.newaxis]     # (M, n)
    delta_vec = np.sqrt(np.sum(X * (1.0 - r_ratio) ** 2, axis=1))

    # Λ = (ΔS/R) / δ²  (using ΔS_mix/R = -Σ xi*ln(xi))
    ds_over_R  = ds_J / R_GAS                               # (M,)
    lambda_vec = np.where(
        delta_vec > 1e-10,
        ds_over_R / delta_vec ** 2,
        np.inf,
    )

    # Ω (Miedema) = T_m * ΔS / |ΔH_mix [J/mol]|
    abs_dh_mied_J = np.maximum(np.abs(dh_mied_vec) * 1000.0, 1e-6)
    omega_mied_vec = t_m_vec * ds_J / abs_dh_mied_J         # (M,)

    # ------------------------------------------------------------------
    # Step 4: Phase filters
    # ------------------------------------------------------------------

    pass1 = (delta_vec <= delta_max) & (lambda_vec >= lambda_min)
    pass2 = (
        (dh_mied_vec >= dh_window[0]) & (dh_mied_vec <= dh_window[1])
        & (omega_mied_vec >= omega_min)
    )

    sig_safe = np.where(dh_gp_std_vec > 1e-8, dh_gp_std_vec, 1e-8)

    # P(ΔH_mix ∈ window) — diagnostic
    prob_dh = (
        sp_norm.cdf(dh_window[1], dh_gp_mean_vec, sig_safe)
        - sp_norm.cdf(dh_window[0], dh_gp_mean_vec, sig_safe)
    )

    # P(Ω ≥ omega_min) ≡ P(|ΔH_mix| ≤ c) — diagnostic
    c_vec = t_m_vec * ds_J / (omega_min * 1000.0)           # (M,)  kJ/mol
    prob_omega = (
        sp_norm.cdf( c_vec, dh_gp_mean_vec, sig_safe)
        - sp_norm.cdf(-c_vec, dh_gp_mean_vec, sig_safe)
    )

    # P(A∩B) — combined analytic criterion
    lo_vec = np.maximum(dh_window[0], -c_vec)
    hi_vec = np.minimum(dh_window[1],  c_vec)
    prob_pass = (
        sp_norm.cdf(hi_vec, dh_gp_mean_vec, sig_safe)
        - sp_norm.cdf(lo_vec, dh_gp_mean_vec, sig_safe)
    )

    pass3      = prob_pass >= epsilon
    in_D_and_S = pass2 & pass3
    in_D_not_S = pass2 & ~pass3
    in_S_not_D = ~pass2 & pass3

    # ------------------------------------------------------------------
    # Step 5: build DataFrame
    # ------------------------------------------------------------------
    # Composition columns: x_Al, x_Cu, ...
    x_cols = {f"x_{e}": X[:, i] for i, e in enumerate(elements)}

    df = pd.DataFrame(
        {
            **x_cols,
            "delta_pct":     delta_vec * 100,
            "lambda":        lambda_vec,
            "ds_mix_J":      ds_J,
            "pass_phase1":   pass1,
            "dh_miedema":    dh_mied_vec,
            "t_m_K":         t_m_vec,
            "omega_miedema": omega_mied_vec,
            "pass_phase2":   pass2,
            "dh_gp_mean":    dh_gp_mean_vec,
            "dh_gp_std":     dh_gp_std_vec,
            "prob_dh":       prob_dh,
            "prob_omega":    prob_omega,
            "prob_pass":     prob_pass,
            "pass_phase3":   pass3,
            "in_D_and_S":    in_D_and_S,
            "in_D_not_S":    in_D_not_S,
            "in_S_not_D":    in_S_not_D,
        }
    )
    return df


def screen_pool_nonequimolar(
    element_pool: list[str],
    predictor,
    k: int = 5,
    x_min: float = 0.05,
    x_max: float = 0.35,
    step: float = 0.05,
    dh_window: tuple[float, float] = DH_WINDOW,
    omega_min: float = OMEGA_MIN,
    delta_max: float = DELTA_MAX,
    lambda_min: float = LAMBDA_MIN,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """
    Screen all k-component alloys from element_pool with non-equimolar compositions.

    Generates C(N,k) element subsets and for each runs the full non-equimolar
    grid screening.  GP binary-pair predictions are precomputed once for all
    C(N,2) unique pairs in the pool.

    Returns
    -------
    Combined DataFrame with an extra 'subset' column (hyphen-joined element names)
    and zero-filled composition columns for elements absent from a given subset.
    """
    import itertools as _itertools
    from src.data_preparation.miedema import MiedemaModel
    from scipy.stats import norm as sp_norm

    pool = sorted(element_pool)
    n_pool = len(pool)
    subsets = list(_itertools.combinations(pool, k))
    print(f"  Pool: {pool}")
    print(f"  C({n_pool},{k}) = {len(subsets)} subsets")

    mp_raw = pd.read_csv(ROOT / "data/miedema_params.csv")
    ep_raw = pd.read_csv(ROOT / "data/periodic_table/element_properties.csv")
    ep = ep_raw.set_index("elem")
    miedema = MiedemaModel(mp_raw)
    r_all  = {e: float(ep.loc[e, "r_atomic"]) for e in pool}
    tm_all = {e: float(ep.loc[e, "T_melt"])   for e in pool}

    # Precompute GP predictions for all C(N,2) unique pairs in pool
    print(f"  Precomputing GP for {n_pool*(n_pool-1)//2} unique binary pairs …")
    pair_cache: dict[tuple[str,str], tuple[float,float,float]] = {}
    for a, b in _itertools.combinations(pool, 2):
        a_c, b_c = (a, b) if a < b else (b, a)
        mean, std, mied = predictor.predict(a_c, b_c, use_model_std=False)
        pair_cache[(a_c, b_c)] = (mean, std, mied)

    # Compositions grid (same for every subset)
    X = generate_compositions(k, x_min, x_max, step)  # (M, k)
    M = len(X)

    all_frames: list[pd.DataFrame] = []

    for subset in subsets:
        elems = list(subset)          # already sorted (pool is sorted)
        pairs = []
        h05_mied_arr = []
        h05_gp_m_arr = []
        h05_gp_s_arr = []

        for i in range(k):
            for j in range(i + 1, k):
                a, b = elems[i], elems[j]
                a_c, b_c = (a, b) if a < b else (b, a)
                pairs.append((i, j))
                gp_mean, gp_std, mied_val = pair_cache[(a_c, b_c)]
                h05_mied_arr.append(mied_val)
                h05_gp_m_arr.append(gp_mean)
                h05_gp_s_arr.append(gp_std)

        h05_mied_arr = np.array(h05_mied_arr)
        h05_gp_m_arr = np.array(h05_gp_m_arr)
        h05_gp_s_arr = np.array(h05_gp_s_arr)

        r  = np.array([r_all[e]  for e in elems])
        tm = np.array([tm_all[e] for e in elems])

        # Pair weight matrices  (M, P)
        W = np.column_stack([4.0 * X[:, i] * X[:, j] for i, j in pairs])

        dh_mied_vec     = W @ h05_mied_arr
        dh_gp_mean_vec  = W @ h05_gp_m_arr
        dh_gp_std_vec   = np.sqrt((W ** 2) @ (h05_gp_s_arr ** 2))

        r_bar   = X @ r
        ds_J    = -R_GAS * np.sum(
            np.where(X > 0, X * np.log(np.where(X > 0, X, 1.0)), 0.0), axis=1
        )
        t_m_vec = X @ tm
        r_ratio = r[np.newaxis, :] / r_bar[:, np.newaxis]
        delta_vec = np.sqrt(np.sum(X * (1.0 - r_ratio) ** 2, axis=1))
        lambda_vec = np.where(
            delta_vec > 1e-10, (ds_J / R_GAS) / delta_vec ** 2, np.inf
        )
        abs_dh_mied_J  = np.maximum(np.abs(dh_mied_vec) * 1000.0, 1e-6)
        omega_mied_vec = t_m_vec * ds_J / abs_dh_mied_J

        pass1 = (delta_vec <= delta_max) & (lambda_vec >= lambda_min)
        pass2 = (
            (dh_mied_vec >= dh_window[0]) & (dh_mied_vec <= dh_window[1])
            & (omega_mied_vec >= omega_min)
        )

        sig_safe = np.where(dh_gp_std_vec > 1e-8, dh_gp_std_vec, 1e-8)

        # P(ΔH_mix ∈ window) — diagnostic
        prob_dh = (
            sp_norm.cdf(dh_window[1], dh_gp_mean_vec, sig_safe)
            - sp_norm.cdf(dh_window[0], dh_gp_mean_vec, sig_safe)
        )

        # P(Ω ≥ omega_min) ≡ P(|ΔH_mix| ≤ c) — diagnostic
        c_vec = t_m_vec * ds_J / (omega_min * 1000.0)       # (M,)  kJ/mol
        prob_omega = (
            sp_norm.cdf( c_vec, dh_gp_mean_vec, sig_safe)
            - sp_norm.cdf(-c_vec, dh_gp_mean_vec, sig_safe)
        )

        # P(A∩B) — combined analytic criterion
        lo_vec = np.maximum(dh_window[0], -c_vec)
        hi_vec = np.minimum(dh_window[1],  c_vec)
        prob_pass = (
            sp_norm.cdf(hi_vec, dh_gp_mean_vec, sig_safe)
            - sp_norm.cdf(lo_vec, dh_gp_mean_vec, sig_safe)
        )

        pass3      = prob_pass >= epsilon
        in_D_and_S = pass2 & pass3
        in_D_not_S = pass2 & ~pass3
        in_S_not_D = ~pass2 & pass3

        x_cols = {f"x_{e}": X[:, i] for i, e in enumerate(elems)}
        # Pad absent pool elements with 0
        for e in pool:
            if f"x_{e}" not in x_cols:
                x_cols[f"x_{e}"] = np.zeros(M)

        frame = pd.DataFrame(
            {
                "subset":        "-".join(elems),
                **x_cols,
                "delta_pct":     delta_vec * 100,
                "lambda":        lambda_vec,
                "ds_mix_J":      ds_J,
                "pass_phase1":   pass1,
                "dh_miedema":    dh_mied_vec,
                "t_m_K":         t_m_vec,
                "omega_miedema": omega_mied_vec,
                "pass_phase2":   pass2,
                "dh_gp_mean":    dh_gp_mean_vec,
                "dh_gp_std":     dh_gp_std_vec,
                "prob_dh":       prob_dh,
                "prob_omega":    prob_omega,
                "prob_pass":     prob_pass,
                "pass_phase3":   pass3,
                "in_D_and_S":    in_D_and_S,
                "in_D_not_S":    in_D_not_S,
                "in_S_not_D":    in_S_not_D,
            }
        )
        all_frames.append(frame)

    combined = pd.concat(all_frames, ignore_index=True)
    # Ensure consistent column order: pool elements first
    x_order = [f"x_{e}" for e in pool]
    other_cols = [c for c in combined.columns if c not in x_order and c != "subset"]
    return combined[["subset"] + x_order + other_cols]
