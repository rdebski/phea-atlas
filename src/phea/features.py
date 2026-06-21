"""
Multicomponent thermophysical features for the P(HEA) model.

Base features (from GP and element properties):
  μ_ΔH   = 4 · Σ_{i<j} x_i · x_j · μ_GP(i,j)              [kJ/mol]
  σ²_ΔH  = 16 · Σ_{i<j} x_i² · x_j² · σ²_GP(i,j)          [independence assumption]
  δ      = sqrt( Σ x_i · (1 - r_i / r̄)² )                  [Hume-Rothery size mismatch]
  dS_R   = -Σ x_i · ln(x_i)                                  [mixing entropy / R]
  T_m    = Σ x_i · T_{m,i}                                   [K]

Soft-boundary penalty features (Opcja B):
  pen_lo    = max(0, -15 − μ_ΔH)          distance below Miedema window  [kJ/mol]
  pen_hi    = max(0,  μ_ΔH − 5)           distance above Miedema window  [kJ/mol]
  pen_omega = max(0, |μ_ΔH| − c)          Ω<1.1 violation, c=T_m·ΔS_J/1100 [kJ/mol]

pen_lo and pen_hi replace the hard [-15, +5] window with linear penalties that
grow continuously outside the Miedema range.  pen_omega replaces the hard Ω≥1.1
criterion.  Inside the window (pen_lo=pen_hi=0), mu_dH still discriminates between
compositions.  All three are zero for alloys satisfying both criteria.
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

FEATURE_NAMES = ["mu_dH", "sigma_dH", "pen_lo", "pen_hi", "pen_omega",
                 "delta", "dS_R", "T_m"]

# 44-element pool (alphabetical) — used for full-covariance precompute.
_DEFAULT_POOL = [
    "Ag", "Al", "Au", "B",  "Ba", "Bi", "Ca", "Cd", "Ce", "Co",
    "Cr", "Cu", "Eu", "Fe", "Ga", "Gd", "Ge", "Hg", "In", "Ir",
    "Li", "Mg", "Mn", "Na", "Nb", "Ni", "Pb", "Pd", "Pt", "Rh",
    "Sb", "Sc", "Se", "Si", "Sn", "Sr", "Tb", "Te", "Ti", "Tl",
    "Y",  "Yb", "Zn", "Zr",
]

# Miedema window and Ω threshold used for penalty features
_DH_LO    = -15.0   # kJ/mol
_DH_HI    =   5.0   # kJ/mol
_OMEGA    =   1.1
_R_GAS    =   8.314  # J/(mol·K)

# Deployment range: mu_dH interval within which predictions are reliable.
# Upper bound +8 kJ/mol: above this the model extrapolates monotonically
# (training data has no endothermic non-HEA examples to constrain pen_hi).
# Lower bound -50 kJ/mol: covers training min (-46.5) with a small buffer.
DH_DEPLOY_LO = -50.0   # kJ/mol
DH_DEPLOY_HI =   8.0   # kJ/mol


def _penalties(mu_dH: float, T_m: float, dS_R: float) -> tuple[float, float, float]:
    """Compute soft-boundary penalties for a single composition."""
    pen_lo = max(0.0, _DH_LO - mu_dH)          # below -15 kJ/mol
    pen_hi = max(0.0, mu_dH - _DH_HI)           # above  +5 kJ/mol
    c = T_m * dS_R * _R_GAS / (_OMEGA * 1000.0) # kJ/mol: Ω=1.1 critical |ΔH|
    pen_omega = max(0.0, abs(mu_dH) - c)
    return pen_lo, pen_hi, pen_omega


class MulticomponentFeatures:
    """
    Computes eight thermophysical features for any k-component alloy.

    Parameters
    ----------
    gp_predictor : GPPredictor
        Trained GP model (src.gp.predict.GPPredictor).
    ep_path : path-like, optional
        Path to element_properties.csv.  Defaults to the project data directory.
    """

    def __init__(self, gp_predictor, ep_path: str | Path | None = None):
        self.gp = gp_predictor
        ep_path = ep_path or ROOT / "data" / "periodic_table" / "element_properties.csv"
        self._ep = pd.read_csv(ep_path).set_index("elem")
        self._pair_cache: dict[tuple[str, str], tuple[float, float]] = {}
        # Full GP posterior covariance over the pool pairs, for EXACT propagation
        # (σ²_ΔH = wᵀKw). Populated lazily by precompute_pool_covariance / on demand.
        self._K_pool = None
        self._pool_index: dict[tuple[str, str], int] | None = None
        self._pool_elems_set: set[str] = set()

    # ------------------------------------------------------------------
    # Pair cache
    # ------------------------------------------------------------------

    def _get_pair(self, a: str, b: str) -> tuple[float, float]:
        """Return (mu_GP, sigma_GP) for a binary pair, cached."""
        key = (min(a, b), max(a, b))
        if key not in self._pair_cache:
            mu, sigma, _ = self.gp.predict(key[0], key[1], use_model_std=False)
            self._pair_cache[key] = (float(mu), float(sigma))
        return self._pair_cache[key]

    def precompute_pairs(self, elements: Sequence[str]) -> None:
        """
        Pre-populate the cache for all C(N,2) pairs in *elements*.

        Call this once before processing many compositions sharing the same
        element set (e.g., all 1,451 non-equimolar compositions of a subset).
        """
        for a, b in combinations(sorted(elements), 2):
            self._get_pair(a, b)

    def precompute_pool(self, pool: Sequence[str] | None = None) -> None:
        """
        Pre-populate the cache for ALL C(44,2) pairs in the default element pool.
        Takes ~2 s on first call; subsequent calls are instant.
        """
        if pool is None:
            pool = list(_DEFAULT_POOL)
        self.precompute_pairs(pool)
        self.precompute_pool_covariance(pool)

    # ------------------------------------------------------------------
    # Exact uncertainty propagation: σ²_ΔH = wᵀ K w  (full GP covariance)
    # ------------------------------------------------------------------

    def precompute_pool_covariance(self, pool: Sequence[str] | None = None) -> None:
        """
        Pre-compute the full GP posterior covariance over all C(|pool|,2) pairs,
        once, so atlas-scale exact σ_ΔH reduces to slicing a small block per subset.
        """
        pool = sorted(pool or _DEFAULT_POOL)
        _, K, _ = self.gp.predict_joint(pool, use_model_std=False)
        self._K_pool = K
        self._pool_elems_set = set(pool)
        self._pool_index = {p: i for i, p in enumerate(combinations(pool, 2))}

    def _cov_block(self, elems_sorted: tuple[str, ...]) -> np.ndarray:
        """
        GP posterior covariance over the C(k,2) pairs of *elems_sorted*
        (pairs ordered as combinations(elems_sorted, 2)).  Uses the pre-computed
        pool covariance when it covers the elements, else a direct joint call.
        """
        if self._K_pool is not None and set(elems_sorted) <= self._pool_elems_set:
            idx = [self._pool_index[(elems_sorted[a], elems_sorted[b])]
                   for a, b in combinations(range(len(elems_sorted)), 2)]
            return self._K_pool[np.ix_(idx, idx)]
        _, K, _ = self.gp.predict_joint(list(elems_sorted), use_model_std=False)
        return K

    def _exact_var(self, elements: Sequence[str], fracs_matrix: np.ndarray) -> np.ndarray:
        """
        Exact σ²_ΔH = wᵀKw for N compositions (fracs_matrix rows), w_p = 4 c_i c_j.
        Handles arbitrary *elements* order by sorting to match the covariance block.
        """
        els = list(elements)
        order = sorted(range(len(els)), key=lambda i: els[i])
        els_sorted = tuple(els[i] for i in order)
        x = fracs_matrix[:, order]
        Kb = self._cov_block(els_sorted)
        pairs = list(combinations(range(len(els_sorted)), 2))
        Wt = 4.0 * np.stack([x[:, a] * x[:, b] for a, b in pairs], axis=1)  # (N, P)
        var = np.einsum("ni,nj,ij->n", Wt, Wt, Kb, optimize=True)
        return np.maximum(var, 0.0)

    # ------------------------------------------------------------------
    # Single composition
    # ------------------------------------------------------------------

    def compute(self, composition: dict[str, float], exact: bool = True) -> dict[str, float]:
        """
        Compute all eight features for a single composition.

        Parameters
        ----------
        composition : {element: mole_fraction}, fractions must sum to ~1.
        exact       : if True (default, canonical), σ_ΔH = √(wᵀKw) propagates the
                      full GP posterior covariance; if False, the diagonal
                      (independence) approximation σ²_ΔH = 16 Σ c_i²c_j²σ²_GP(i,j).

        Returns
        -------
        dict with keys matching FEATURE_NAMES:
          mu_dH, sigma_dH, pen_lo, pen_hi, pen_omega, delta, dS_R, T_m
        """
        elems = list(composition.keys())
        fracs = np.array([composition[e] for e in elems])

        # GP-derived thermodynamic features
        mu_dH  = 0.0
        var_dH = 0.0
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                mu_ij, sigma_ij = self._get_pair(elems[i], elems[j])
                coeff    = fracs[i] * fracs[j]
                mu_dH   += 4.0 * coeff * mu_ij
                var_dH  += 16.0 * coeff**2 * sigma_ij**2

        if exact:
            var_dH = float(self._exact_var(elems, fracs.reshape(1, -1))[0])

        # Thermophysical features from element properties table
        r  = np.array([float(self._ep.loc[e, "r_atomic"]) for e in elems])
        Tm = np.array([float(self._ep.loc[e, "T_melt"])   for e in elems])

        r_bar  = float(fracs @ r)
        delta  = float(np.sqrt(np.sum(fracs * (1.0 - r / r_bar) ** 2)))
        T_m    = float(fracs @ Tm)
        dS_R   = float(-np.sum(fracs * np.log(np.maximum(fracs, 1e-15))))

        pen_lo, pen_hi, pen_omega = _penalties(mu_dH, T_m, dS_R)

        return {
            "mu_dH":     mu_dH,
            "sigma_dH":  math.sqrt(var_dH),
            "pen_lo":    pen_lo,
            "pen_hi":    pen_hi,
            "pen_omega": pen_omega,
            "delta":     delta,
            "dS_R":      dS_R,
            "T_m":       T_m,
        }

    # ------------------------------------------------------------------
    # Batch API (list of composition dicts)
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        compositions: Sequence[dict[str, float]],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Compute features for a list of composition dicts.

        Returns pd.DataFrame with columns = FEATURE_NAMES.
        """
        rows = []
        for i, comp in enumerate(compositions):
            rows.append(self.compute(comp))
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(compositions)}")
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Array API (atlas-scale, vectorised over compositions)
    # ------------------------------------------------------------------

    def compute_array(
        self,
        elements: Sequence[str],
        fracs_matrix: np.ndarray,
        exact: bool = True,
    ) -> np.ndarray:
        """
        Vectorised feature computation for N compositions sharing the same
        element set.  Designed for atlas-scale loops.

        Parameters
        ----------
        elements     : sequence of k element symbols (any order)
        fracs_matrix : ndarray (N, k), each row sums to 1.
        exact        : if True (default, canonical), σ_ΔH propagates the full GP
                       posterior covariance (wᵀKw); if False, the diagonal
                       (independence) approximation.

        Returns
        -------
        ndarray (N, 8) — columns: mu_dH, sigma_dH, pen_lo, pen_hi,
                                   pen_omega, delta, dS_R, T_m
        """
        elems = list(elements)
        k = len(elems)
        N = fracs_matrix.shape[0]

        # Precompute pair (mu, sigma) arrays — shape (k, k)
        mu_mat    = np.zeros((k, k))
        sigma_mat = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                mu_ij, sigma_ij = self._get_pair(elems[i], elems[j])
                mu_mat[i, j] = mu_mat[j, i] = mu_ij
                sigma_mat[i, j] = sigma_mat[j, i] = sigma_ij

        # Element property arrays
        r  = np.array([float(self._ep.loc[e, "r_atomic"]) for e in elems])
        Tm = np.array([float(self._ep.loc[e, "T_melt"])   for e in elems])

        # Vectorised over N compositions ---------------------------------

        # mu_dH: 4 · Σ_{i<j} x_i · x_j · mu_ij
        # = 2 · Σ_i Σ_j x_i · x_j · mu_ij  (upper triangle counted twice by outer)
        # but we use the outer product trick:
        #   x_outer[n, i, j] = x[n, i] * x[n, j]
        x = fracs_matrix                               # (N, k)
        x_outer = x[:, :, None] * x[:, None, :]       # (N, k, k)

        mu_dH = 2.0 * np.einsum("nij,ij->n", x_outer, mu_mat)
        # Correction: the full pairwise formula is 4*Σ_{i<j}, and the outer
        # product gives 2*Σ_{i<j} (upper + lower triangle, diagonal=0).
        # So factor is 4/2 = 2 applied to the full matrix sum, which equals
        # 4 * Σ_{i<j}  ✓

        if exact:
            var_dH = self._exact_var(elems, fracs_matrix)
        else:
            var_dH = 8.0 * np.einsum("nij,ij->n", x_outer**2, sigma_mat**2)
            # 16 * Σ_{i<j} xi²·xj²·σ² = 16/2 * full_sum(xi²·xj²·σ²) = 8 * full_sum
            # full matrix (i≠j) = 2 * upper triangle, diagonal=0

        sigma_dH = np.sqrt(var_dH)

        r_bar  = x @ r                                  # (N,)
        delta  = np.sqrt(np.sum(x * (1.0 - r[None, :] / r_bar[:, None])**2, axis=1))
        T_m    = x @ Tm                                 # (N,)
        dS_R   = -np.sum(x * np.log(np.maximum(x, 1e-15)), axis=1)

        # Soft-boundary penalties — vectorised
        pen_lo    = np.maximum(0.0, _DH_LO - mu_dH)
        pen_hi    = np.maximum(0.0, mu_dH - _DH_HI)
        c         = T_m * dS_R * _R_GAS / (_OMEGA * 1000.0)
        pen_omega = np.maximum(0.0, np.abs(mu_dH) - c)

        return np.column_stack(
            [mu_dH, sigma_dH, pen_lo, pen_hi, pen_omega, delta, dS_R, T_m]
        )
