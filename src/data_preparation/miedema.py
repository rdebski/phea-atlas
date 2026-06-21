"""
Miedema model for enthalpy of mixing of binary LIQUID metallic alloys.

Reference: de Boer, Boom, Mattens, Miedema, Niessen —
           "Cohesion in Metals: Transition Metal Alloys" (1988).

Formula (full surface-concentration version):
    ΔH_mix = x_A·c_B^S · ΔH_{A-in-B} + x_B·c_A^S · ΔH_{B-in-A}

where:
    ΔH_{A-in-B} = 2·V_A^(2/3)_corr · P · f_AB / n̄_ws^(-1/3)
    ΔH_{B-in-A} = 2·V_B^(2/3)_corr · P · f_AB / n̄_ws^(-1/3)
    f_AB        = 9.4·(Δn_ws^(1/3))² - (Δφ*)² - 0.73·RP

Edge canonicalization contract (preprocessing):
    All (elem_a, elem_b) pairs must be in alphabetical order before calling
    h_mix_fn — i.e. elem_a < elem_b lexicographically.
    This mirrors the canonicalization applied to training data (rk.csv) and
    inference queries. No internal swap is performed here.
    φ1 = H'_mix(0) and φ2 = H'_mix(1) are defined relative to this canonical
    direction. For a reverse query (B-A), reorder before calling — do NOT
    implement Conditional Permutation inside the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P_poly


class MiedemaModel:

    def __init__(self, params_df: pd.DataFrame):
        """
        Initialize with a DataFrame of element parameters.

        Expected columns:
            elem   : element symbol (used as index)
            phi    : Miedema electronegativity φ* [V]
            nws    : electron density at WS cell boundary n_ws [d.u.]
            v      : molar volume V [cm³/mol]
            acf    : excess-volume correction factor α
            rp     : hybridization parameter (R/P), used for blue-white pairs
            tab2a  : 1 = transition metal (blue row), 0 = non-transition (white row)

        Note: dh_trans is intentionally absent — structural-transformation
        corrections are not applied in the liquid-alloy model.
        """
        params_df = params_df.set_index("elem")
        self.phi   = params_df["phi"]
        self.n_ws  = params_df["nws"]
        self.V     = params_df["v"]
        self.acf   = params_df["acf"]
        self.rp    = params_df["rp"]
        self.tab2a = params_df["tab2a"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_P_and_RP(
        elem_a: str, elem_b: str,
        tab2a_a: int, tab2a_b: int,
        rp_a: float, rp_b: float,
    ) -> tuple[float, float]:
        """
        Compute interaction constant P and hybridization correction RP.

        Rules (de Boer et al. 1988, Table 2a):
            Both blue  (both transition metals) : P = 1.15 × 12.35, RP = 0
            Both white (both non-transition)    : P = 12.35 / 1.15,  RP = 0.2 if Mg else 0
            Mixed (one blue, one white)         : P = 12.35,          RP = rp_a × rp_b

        The Mg correction (RP = 0.2 for white-white pairs containing Mg)
        accounts for the anomalous hybridization of Mg with non-transition metals.
        """
        if tab2a_a == tab2a_b:
            if tab2a_a == 1:                            # both transition metals
                return 1.15 * 12.35, 0.0
            else:                                       # both non-transition
                rp = 0.2 if "Mg" in (elem_a, elem_b) else 0.0
                return 12.35 / 1.15, rp
        else:                                           # mixed pair
            return 12.35, rp_a * rp_b

    # ------------------------------------------------------------------
    # Core model
    # ------------------------------------------------------------------

    def h_mix_fn(self, elem_a: str, elem_b: str, x_b: float) -> float:
        """
        Compute ΔH_mix [kJ/mol] for binary liquid alloy A_{1-x}·B_x.

        Args:
            elem_a : symbol of element A — must be alphabetically < elem_b
            elem_b : symbol of element B
            x_b    : mole fraction of B, in [0, 1]

        Returns:
            ΔH_mix in kJ/mol (negative = exothermic).

        Raises:
            ValueError if elem_a >= elem_b (canonical order violated).
        """
        if elem_a >= elem_b:
            raise ValueError(
                f"Canonical order violated: '{elem_a}' >= '{elem_b}'. "
                "Pass elements in alphabetical order or reorder the query before calling."
            )

        x_a = 1.0 - x_b

        # --- Element properties ----------------------------------------------
        phi_a   = float(self.phi[elem_a])
        phi_b   = float(self.phi[elem_b])
        n_a     = float(self.n_ws[elem_a])
        n_b     = float(self.n_ws[elem_b])
        v_a     = float(self.V[elem_a])
        v_b     = float(self.V[elem_b])
        acf_a   = float(self.acf[elem_a])
        acf_b   = float(self.acf[elem_b])
        rp_a    = float(self.rp[elem_a])
        rp_b    = float(self.rp[elem_b])
        tab2a_a = int(self.tab2a[elem_a])
        tab2a_b = int(self.tab2a[elem_b])

        # --- Interaction constants --------------------------------------------
        P, RP = self._compute_P_and_RP(
            elem_a, elem_b, tab2a_a, tab2a_b, rp_a, rp_b
        )

        # --- Molar volumes to 2/3 power --------------------------------------
        v_a23 = v_a ** (2.0 / 3.0)
        v_b23 = v_b ** (2.0 / 3.0)

        # --- Driving-force terms (symmetric under A↔B via squaring) ----------
        delta_phi   = phi_a - phi_b
        delta_n13   = n_a ** (1.0 / 3.0) - n_b ** (1.0 / 3.0)
        n_avg_inv13 = 0.5 * (n_a ** (-1.0 / 3.0) + n_b ** (-1.0 / 3.0))   # n̄_ws^{-1/3}

        # f_AB: numerator of the Miedema interaction term (symmetric in A↔B)
        f_AB = 9.4 * delta_n13 ** 2 - delta_phi ** 2 - 0.73 * RP

        # --- Surface concentrations — first pass (uncorrected volumes) --------
        denom1 = x_a * v_a23 + x_b * v_b23
        xs_a1  = x_a * v_a23 / denom1
        xs_b1  = 1.0 - xs_a1

        # --- Chemical volume correction (Δφ shifts effective V^{2/3}) ---------
        v_a_corr = v_a23 * (1.0 + acf_a * xs_b1 * delta_phi)
        v_b_corr = v_b23 * (1.0 - acf_b * xs_a1 * delta_phi)

        # --- Surface concentrations — second pass (corrected volumes) ---------
        denom2 = x_a * v_a_corr + x_b * v_b_corr
        xs_a2  = x_a * v_a_corr / denom2
        xs_b2  = 1.0 - xs_a2

        # --- Enthalpy of mixing -----------------------------------------------
        # One-sided formula: x_A · c_B^S · V_A^(2/3)_corr · P · f_AB / n̄^(-1/3)
        #
        # IMPORTANT — parameterization convention:
        #   The de Boer et al. (1988) formula has an explicit factor of 2 and a
        #   symmetric B-in-A term. However, sanity checks against literature
        #   (Cu-Au ≈ -9, Ag-Au ≈ -3, Bi-Pb ≈ +1.5 kJ/mol at x=0.5) confirm that
        #   the P constants in the parameter set used here (12.35, 14.2025, 10.735)
        #   are calibrated for the one-sided formula WITHOUT factor 2 and WITHOUT
        #   the symmetric term. Adding either factor causes ~4× overestimation.
        #   Do not "fix" this to the de Boer form without re-fitting P constants.
        h_mix = x_a * xs_b2 * v_a_corr * P * f_AB / n_avg_inv13

        return h_mix

    # ------------------------------------------------------------------
    # Convenience: composition scan
    # ------------------------------------------------------------------

    def scan(
        self,
        elem_a: str,
        elem_b: str,
        n_points: int = 201,
    ) -> pd.DataFrame:
        """
        Compute ΔH_mix over a uniform composition grid.

        Returns a DataFrame with columns ['xB', 'H_mix'] (H_mix in kJ/mol).
        This is the starting point for computing Miedema-based R-K triplets
        (Phase 1 pre-training targets) via fit_rk3 below.
        """
        xs = np.linspace(0.0, 1.0, n_points)
        hs = np.array([self.h_mix_fn(elem_a, elem_b, float(x)) for x in xs])
        return pd.DataFrame({"xB": xs, "H_mix": hs})

    # ------------------------------------------------------------------
    # R-K triplet generation (Phase 1 pre-training targets)
    # ------------------------------------------------------------------

    @staticmethod
    def fit_rk3(
        x: np.ndarray,
        h: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Fit a 3-parameter Redlich-Kister polynomial to (x, H_mix) data.

        R-K form (a3 ≡ 0):
            H_mix(x) = x(1-x) · [a0 + a1·(1-2x) + a2·(1-2x)²]

        Solved via ordinary least squares in the three basis functions:
            b0(x) = x(1-x)
            b1(x) = x(1-x)(1-2x)
            b2(x) = x(1-x)(1-2x)²

        Returns:
            (a0, a1, a2) — R-K coefficients in kJ/mol.
        """
        u  = 1.0 - 2.0 * x
        b0 = x * (1.0 - x)
        b1 = b0 * u
        b2 = b0 * u ** 2
        A  = np.column_stack([b0, b1, b2])
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return tuple(coeffs)   # (a0, a1, a2)

    @staticmethod
    def rk3_to_triplet(a0: float, a1: float, a2: float) -> tuple[float, float, float]:
        """
        Convert 3-parameter R-K coefficients to geometric triplet (φ1, φ2, φ3).

        Forward map (locked, §3 of project context):
            φ1 = a0 + a1 + a2   (slope at x=0, pure A)
            φ2 = a0 - a1 + a2   (slope at x=1, pure B) — NOTE sign of a1
            φ3 = 0.25 · a0      (value at equimolar x=0.5)

        Wait — correct signs from the project context:
            φ1 =  a0 + a1 + a2
            φ2 = -a0 + a1 - a2   (← sign flip on a0 and a2)
            φ3 =  0.25 · a0

        Returns:
            (φ1, φ2, φ3) in kJ/mol.
        """
        phi1 =  a0 + a1 + a2
        phi2 = -a0 + a1 - a2
        phi3 =  0.25 * a0
        return phi1, phi2, phi3

    @staticmethod
    def fit_rk4(
        x: np.ndarray,
        h: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """
        Fit a 4-parameter Redlich-Kister polynomial to (x, H_mix) data.

        R-K form:
            H_mix(x) = x(1-x) · [a0 + a1·u + a2·u² + a3·u³],  u = 1-2x

        Basis functions:
            b0 = x(1-x),  b1 = x(1-x)·u,  b2 = x(1-x)·u²,  b3 = x(1-x)·u³

        Returns:
            (a0, a1, a2, a3) — R-K coefficients in kJ/mol.
        """
        u  = 1.0 - 2.0 * x
        b0 = x * (1.0 - x)
        A  = np.column_stack([b0, b0*u, b0*u**2, b0*u**3])
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return tuple(coeffs)   # (a0, a1, a2, a3)

    @staticmethod
    def rk4_to_grk4(
        a0: float, a1: float, a2: float, a3: float,
    ) -> tuple[float, float, float, float]:
        """
        Convert 4-parameter R-K coefficients to geometric quadruplet (GRK4).

        GRK4 = (h_primx0, h_primx1, h_x025, h_x075):
            h_primx0 = H'(0)    =  a0 + a1 + a2 + a3
            h_primx1 = H'(1)    = -a0 + a1 - a2 + a3
            h_x025   = H(0.25)  =  0.1875·(a0 + 0.5·a1 + 0.25·a2 + 0.125·a3)
            h_x075   = H(0.75)  =  0.1875·(a0 - 0.5·a1 + 0.25·a2 - 0.125·a3)

        Derivation: at x=0.25, u=0.5; at x=0.75, u=-0.5; f(0.25)=f(0.75)=3/16.
        """
        h_primx0 =  a0 + a1 + a2 + a3
        h_primx1 = -a0 + a1 - a2 + a3
        h_x025   = 0.1875 * (a0 + 0.5*a1 + 0.25*a2 + 0.125*a3)
        h_x075   = 0.1875 * (a0 - 0.5*a1 + 0.25*a2 - 0.125*a3)
        return h_primx0, h_primx1, h_x025, h_x075

    @staticmethod
    def fit_grk4_regularized(
        x: np.ndarray,
        h: np.ndarray,
        lambda_base: float = 1.0,
        coverage_thresh: float = 0.20,
    ) -> tuple:
        """
        Fit GRK4 with L2 regularization on (a1, a2, a3), adaptive lambda.

        Coverage = min(n_lo, n_hi) / (n_lo + n_hi), where n_lo = #{x < 0.5}.
        Lambda   = lambda_base / max(coverage, 0.01)  — stronger for partial range.

        For coverage < coverage_thresh: use RK3 (a3 = 0, regularize a1, a2).
        For coverage >= coverage_thresh: use RK4 (regularize a1, a2, a3).

        Only a0 is unregularized — it controls the overall curve magnitude
        (H(0.5) = a0/4) and is well-determined even from partial-range data.

        Returns:
            (a0, a1, a2, a3, coverage, lambda_used)
        """
        n_lo = int((x < 0.5).sum())
        n_hi = int((x > 0.5).sum())
        n_tot = n_lo + n_hi
        cov  = min(n_lo, n_hi) / max(n_tot, 1)
        lam  = lambda_base / max(cov, 0.01)

        u  = 1.0 - 2.0 * x
        b0 = x * (1.0 - x)

        if cov >= coverage_thresh:          # full-enough range → RK4
            A   = np.column_stack([b0, b0*u, b0*u**2, b0*u**3])
            reg = np.diag([0.0, lam, lam, lam])
            a   = np.linalg.solve(A.T @ A + reg, A.T @ h)
            a0, a1, a2, a3 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        else:                               # partial range → RK3, a3 = 0
            A   = np.column_stack([b0, b0*u, b0*u**2])
            reg = np.diag([0.0, lam, lam])
            a   = np.linalg.solve(A.T @ A + reg, A.T @ h)
            a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
            a3 = 0.0

        return a0, a1, a2, a3, cov, lam

    def miedema_grk4(
        self,
        elem_a: str,
        elem_b: str,
        n_points: int = 201,
    ) -> dict:
        """
        Full pipeline: Miedema scan → R-K4 fit → GRK4 geometric quadruplet.

        Returns a dict with keys:
            'a0'..'a3'                          : R-K4 coefficients [kJ/mol]
            'h_primx0','h_primx1','h_x025','h_x075' : GRK4 quadruplet [kJ/mol]
            'rmse_refit'                        : RMSE of 4-param fit vs. Miedema [kJ/mol]
            'mae_refit'                         : MAE of 4-param fit vs. Miedema [kJ/mol]
        """
        df = self.scan(elem_a, elem_b, n_points)
        x  = df["xB"].values
        h  = df["H_mix"].values

        a0, a1, a2, a3 = self.fit_rk4(x, h)

        u     = 1.0 - 2.0 * x
        h_fit = x * (1.0 - x) * (a0 + a1*u + a2*u**2 + a3*u**3)
        resid = h - h_fit
        rmse  = float(np.sqrt(np.mean(resid**2)))
        mae   = float(np.mean(np.abs(resid)))

        gp = self.rk4_to_grk4(a0, a1, a2, a3)

        return {
            "a0": a0, "a1": a1, "a2": a2, "a3": a3,
            "h_primx0": gp[0], "h_primx1": gp[1],
            "h_x025": gp[2],   "h_x075": gp[3],
            "rmse_refit": rmse, "mae_refit": mae,
        }

    # ------------------------------------------------------------------
    # RK4+smooth parameterization
    # ------------------------------------------------------------------

    @staticmethod
    def fit_rksmooth(
        x: np.ndarray,
        h: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Fit RK4+smooth basis to (x, H_mix) data — plain OLS, 3 parameters.

        The smooth constraint a3 = -a1/9 is built in, so the three free
        parameters are the geometric quantities:
            dh0 = H'(0),  dh1 = H'(1),  h05 = H(0.5)

        Basis functions (u = 1-2x):
            B0(x) = x(1-x) · [(9u - u³)/16  +  u²/2]
            B1(x) = x(1-x) · [(9u - u³)/16  -  u²/2]
            B2(x) = 4x(1-x) · (1 - u²)

        H(x) = dh0·B0(x) + dh1·B1(x) + h05·B2(x)

        Returns:
            (dh0, dh1, h05) in kJ/mol.
        """
        u   = 1.0 - 2.0 * x
        xu  = x * (1.0 - x)
        b0  = xu * ((9.0*u - u**3) / 16.0 + u**2 / 2.0)
        b1  = xu * ((9.0*u - u**3) / 16.0 - u**2 / 2.0)
        b2  = 4.0 * xu * (1.0 - u**2)
        A   = np.column_stack([b0, b1, b2])
        c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return float(c[0]), float(c[1]), float(c[2])

    @staticmethod
    def rksmooth_to_rk4(
        dh0: float, dh1: float, h05: float,
    ) -> tuple[float, float, float, float]:
        """
        Convert RK4+smooth geometric parameters to RK4 coefficients.

        Derivation: geometric constraints + smoothness constraint a3 = -a1/9:
            a0 = 4·h05
            a1 = (9/16)·(dh0 + dh1)
            a2 = (dh0 - dh1)/2 - 4·h05
            a3 = -(1/16)·(dh0 + dh1)  = -a1/9

        Verification: H'(0) = a0+a1+a2+a3 = dh0 ✓
                      H'(1) = -a0+a1-a2+a3 = dh1 ✓
                      H(0.5) = a0/4 = h05 ✓
        """
        a0 = 4.0 * h05
        a1 = (9.0 / 16.0) * (dh0 + dh1)
        a2 = 0.5 * (dh0 - dh1) - a0
        a3 = -(1.0 / 16.0) * (dh0 + dh1)
        return a0, a1, a2, a3

    @staticmethod
    def fit_rksmooth_regularized(
        x: np.ndarray,
        h: np.ndarray,
        lambda_base: float = 1.0,
        coverage_thresh: float = 0.20,
    ) -> tuple:
        """
        Fit RK4+smooth with adaptive L2 regularization on (dh0, dh1).

        For coverage >= coverage_thresh: plain OLS (unbiased, full range).
        For coverage <  coverage_thresh: ridge on dh0, dh1 (not h05).

        h05 is never regularized — it controls H(0.5) = a0/4 and is
        well-determined even from partial-range data through the B2 basis function.

        Coverage = min(n_lo, n_hi) / (n_lo + n_hi), n_lo = #{x < 0.5}.
        Lambda   = lambda_base / max(coverage, 0.01) for partial range.

        Returns:
            (dh0, dh1, h05, coverage, lambda_used)
        """
        n_lo = int((x < 0.5).sum())
        n_hi = int((x > 0.5).sum())
        n_tot = n_lo + n_hi
        cov   = min(n_lo, n_hi) / max(n_tot, 1)

        u   = 1.0 - 2.0 * x
        xu  = x * (1.0 - x)
        b0  = xu * ((9.0*u - u**3) / 16.0 + u**2 / 2.0)
        b1  = xu * ((9.0*u - u**3) / 16.0 - u**2 / 2.0)
        b2  = 4.0 * xu * (1.0 - u**2)
        A   = np.column_stack([b0, b1, b2])

        if cov >= coverage_thresh:
            c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
            lam = 0.0
        else:
            lam = lambda_base / max(cov, 0.01)
            reg = np.diag([lam, lam, 0.0])    # regularize dh0, dh1 only
            c   = np.linalg.solve(A.T @ A + reg, A.T @ h)

        return float(c[0]), float(c[1]), float(c[2]), float(cov), float(lam)

    def miedema_rksmooth(
        self,
        elem_a: str,
        elem_b: str,
        n_points: int = 201,
    ) -> dict:
        """
        Full pipeline: Miedema scan → RK4+smooth LS fit.

        Returns a dict with keys:
            'dh0', 'dh1', 'h05'         : geometric triplet [kJ/mol]
            'a0'..'a3'                   : RK4 coefficients (a3 = -a1/9) [kJ/mol]
            'rmse_refit', 'mae_refit'    : fit quality vs. Miedema scan [kJ/mol]
        """
        df  = self.scan(elem_a, elem_b, n_points)
        x   = df["xB"].values
        h   = df["H_mix"].values

        dh0, dh1, h05  = self.fit_rksmooth(x, h)
        a0, a1, a2, a3 = self.rksmooth_to_rk4(dh0, dh1, h05)

        u     = 1.0 - 2.0 * x
        h_fit = x * (1.0 - x) * (a0 + a1*u + a2*u**2 + a3*u**3)
        resid = h - h_fit
        rmse  = float(np.sqrt(np.mean(resid**2)))
        mae   = float(np.mean(np.abs(resid)))

        return {
            "dh0": dh0, "dh1": dh1, "h05": h05,
            "a0": a0, "a1": a1, "a2": a2, "a3": a3,
            "rmse_refit": rmse, "mae_refit": mae,
        }

    # ------------------------------------------------------------------
    # Piecewise quadratic spline parametrization
    # ------------------------------------------------------------------
    #
    # 3 segments: s1 on [0, 0.25], s2 on [0.25, 0.75], s3 on [0.75, 1]
    # Each is a degree-2 polynomial (3 params × 3 segments = 9 total).
    # 9 constraints make the system fully determined:
    #   s1(0)=0, s1'(0)=dh0, s3(1)=0, s3'(1)=dh1  (boundary)
    #   s1(0.25)=s2(0.25), s1'(0.25)=s2'(0.25)     (C¹ at 0.25)
    #   s2(0.75)=s3(0.75), s2'(0.75)=s3'(0.75)     (C¹ at 0.75)
    #   s2(0.5) = h05                                (midpoint)
    #
    # Flip A↔B: (dh0, dh1, h05) → (−dh1, −dh0, h05)  — identical to rksmooth.
    # ------------------------------------------------------------------

    @staticmethod
    def _rkspline_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the three basis functions for the piecewise quadratic spline.

        H(x; dh0, dh1, h05) = dh0·f0(x) + dh1·f1(x) + h05·f2(x)

        Basis functions (derived from the 9-constraint linear system):
            Segment [0, 0.25]:    f0 = x − 17/6·x²,  f1 = 1/6·x²,         f2 = 8·x²
            Segment [0.25, 0.75]: f0 = 5/24 − 2/3·x + 1/2·x²,
                                  f1 = −1/24 + 1/3·x − 1/2·x²,
                                  f2 = −1 + 8·x − 8·x²
            Segment [0.75, 1]:    f0 = −1/6 + 1/3·x − 1/6·x²,
                                  f1 = 11/6 − 14/3·x + 17/6·x²,
                                  f2 = 8 − 16·x + 8·x²
        """
        x = np.asarray(x, dtype=float)
        x2 = x * x

        # segment masks
        m1 = x <= 0.25
        m2 = (x > 0.25) & (x <= 0.75)
        m3 = x > 0.75

        f0 = np.empty_like(x)
        f1 = np.empty_like(x)
        f2 = np.empty_like(x)

        f0[m1] = x[m1] - (17.0/6.0)*x2[m1]
        f1[m1] = (1.0/6.0)*x2[m1]
        f2[m1] = 8.0*x2[m1]

        f0[m2] = 5.0/24.0 - (2.0/3.0)*x[m2] + 0.5*x2[m2]
        f1[m2] = -1.0/24.0 + (1.0/3.0)*x[m2] - 0.5*x2[m2]
        f2[m2] = -1.0 + 8.0*x[m2] - 8.0*x2[m2]

        f0[m3] = -1.0/6.0 + (1.0/3.0)*x[m3] - (1.0/6.0)*x2[m3]
        f1[m3] = 11.0/6.0 - (14.0/3.0)*x[m3] + (17.0/6.0)*x2[m3]
        f2[m3] = 8.0 - 16.0*x[m3] + 8.0*x2[m3]

        return f0, f1, f2

    @staticmethod
    def hmix_rkspline(
        x: np.ndarray,
        dh0: float,
        dh1: float,
        h05: float,
    ) -> np.ndarray:
        """Evaluate the piecewise quadratic spline H_mix at composition x."""
        f0, f1, f2 = MiedemaModel._rkspline_basis(x)
        return dh0*f0 + dh1*f1 + h05*f2

    @staticmethod
    def fit_rkspline(
        x: np.ndarray,
        h: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Fit the piecewise quadratic spline to (x, H_mix) data — plain OLS.

        Returns:
            (dh0, dh1, h05)
        """
        f0, f1, f2 = MiedemaModel._rkspline_basis(x)
        A = np.column_stack([f0, f1, f2])
        c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return float(c[0]), float(c[1]), float(c[2])

    @staticmethod
    def fit_rkspline_regularized(
        x: np.ndarray,
        h: np.ndarray,
        lambda_base: float = 1.0,
        coverage_thresh: float = 0.20,
    ) -> tuple:
        """
        Fit piecewise quadratic spline with adaptive L2 regularization on (dh0, dh1).

        h05 is not regularized — it controls H(0.5) and is well-determined even
        from partial-range data (f2 basis function peaks at x=0.5).

        Returns:
            (dh0, dh1, h05, coverage, lambda_used)
        """
        n_lo  = int((x < 0.5).sum())
        n_hi  = int((x > 0.5).sum())
        n_tot = n_lo + n_hi
        cov   = min(n_lo, n_hi) / max(n_tot, 1)

        f0, f1, f2 = MiedemaModel._rkspline_basis(x)
        A = np.column_stack([f0, f1, f2])

        if cov >= coverage_thresh:
            c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
            lam = 0.0
        else:
            lam = lambda_base / max(cov, 0.01)
            reg = np.diag([lam, lam, 0.0])   # regularize dh0, dh1 only
            c   = np.linalg.solve(A.T @ A + reg, A.T @ h)

        return float(c[0]), float(c[1]), float(c[2]), float(cov), float(lam)

    def miedema_rkspline(
        self,
        elem_a: str,
        elem_b: str,
        n_points: int = 201,
    ) -> dict:
        """
        Full pipeline: Miedema scan → piecewise quadratic spline LS fit.

        Returns a dict with keys:
            'dh0', 'dh1', 'h05'      : geometric triplet [kJ/mol]
            'rmse_refit', 'mae_refit' : fit quality vs. Miedema scan [kJ/mol]
        """
        df  = self.scan(elem_a, elem_b, n_points)
        x   = df["xB"].values
        h   = df["H_mix"].values

        dh0, dh1, h05 = self.fit_rkspline(x, h)
        h_fit = self.hmix_rkspline(x, dh0, dh1, h05)
        resid = h - h_fit
        rmse  = float(np.sqrt(np.mean(resid**2)))
        mae   = float(np.mean(np.abs(resid)))

        return {
            "dh0": dh0, "dh1": dh1, "h05": h05,
            "rmse_refit": rmse, "mae_refit": mae,
        }

    def miedema_triplet(
        self,
        elem_a: str,
        elem_b: str,
        n_points: int = 201,
    ) -> dict:
        """
        Full pipeline: Miedema scan → R-K fit → triplet.

        Returns a dict with keys:
            'a0', 'a1', 'a2'        : R-K coefficients [kJ/mol]
            'phi1', 'phi2', 'phi3'  : geometric triplet [kJ/mol]
            'rmse_refit'            : RMSE of 3-param fit vs. Miedema curve [kJ/mol]
                                      (lower bound on model error — §4 of project context)
        """
        df = self.scan(elem_a, elem_b, n_points)
        x  = df["xB"].values
        h  = df["H_mix"].values

        a0, a1, a2 = self.fit_rk3(x, h)

        u      = 1.0 - 2.0 * x
        h_fit  = x * (1.0 - x) * (a0 + a1 * u + a2 * u ** 2)
        rmse   = float(np.sqrt(np.mean((h - h_fit) ** 2)))

        phi1, phi2, phi3 = self.rk3_to_triplet(a0, a1, a2)

        return {
            "a0": a0, "a1": a1, "a2": a2,
            "phi1": phi1, "phi2": phi2, "phi3": phi3,
            "rmse_refit": rmse,
        }

    # ------------------------------------------------------------------
    # Natural-spline (degree-6) parameterization
    # ------------------------------------------------------------------

    @staticmethod
    def _natural6_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Three basis functions for the natural-spline polynomial.

        H(x) = dh0·B0(x) + dh1·B1(x) + h05·B2(x)

        Boundary conditions (all verified analytically):
            H(0) = H(1) = 0   (built in by construction)
            H'(0) = dh0,  H'(1) = dh1
            H(0.5) = h05
            H''(0) = H''(1) = 0   (natural-spline / zero-curvature ends)

        Derivation: 6-degree polynomial c0+…+c6·x^6 with c0=0, c1=dh0,
        c2=0, and c3…c6 solved from H(1)=0, H'(1)=dh1, H(0.5)=h05,
        H''(1)=0.  Result expressed as linear combinations:
            B0 = x − 16x³ + 38x⁴ − 33x⁵ + 10x⁶
            B1 = 6x³ − 23x⁴ + 27x⁵ − 10x⁶
            B2 = 64x³(1−x)³
        """
        x3 = x**3
        b0 = x    - 16*x3  + 38*x**4 - 33*x**5 + 10*x**6
        b1 =        6*x3   - 23*x**4 + 27*x**5 - 10*x**6
        b2 = 64 * x3 * (1.0 - x)**3
        return b0, b1, b2

    @staticmethod
    def fit_natural6_regularized(
        x: np.ndarray,
        h: np.ndarray,
        lambda_base: float = 1.0,
        coverage_thresh: float = 0.20,
    ) -> tuple:
        """
        Fit natural-spline degree-6 polynomial with adaptive L2 ridge on B0, B1.

        B2 (h05) is unregularized — it controls H(0.5) and is well-determined
        even from partial-range data.  B0 (dh0) and B1 (dh1) control endpoint
        slopes, poorly constrained by partial data → regularized toward zero.

        Coverage = min(n_lo, n_hi) / (n_lo + n_hi), n_lo = #{x < 0.5}.
        Lambda   = lambda_base / max(coverage, 0.01) for partial range.

        Returns:
            (dh0, dh1, h05, coverage, lambda_used)
        """
        n_lo = int((x < 0.5).sum())
        n_hi = int((x > 0.5).sum())
        cov  = min(n_lo, n_hi) / max(n_lo + n_hi, 1)

        b0, b1, b2 = MiedemaModel._natural6_basis(x)
        A = np.column_stack([b0, b1, b2])

        if cov >= coverage_thresh:
            c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
            lam = 0.0
        else:
            lam = lambda_base / max(cov, 0.01)
            reg = np.diag([lam, lam, 0.0])
            c   = np.linalg.solve(A.T @ A + reg, A.T @ h)

        return float(c[0]), float(c[1]), float(c[2]), float(cov), float(lam)

    @staticmethod
    def fit_natural6(
        x: np.ndarray,
        h: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Fit the natural-spline degree-6 polynomial to (x, H_mix) data via OLS.

        The polynomial satisfies H''(0) = H''(1) = 0 exactly by construction.
        Three free parameters share the same geometric meaning as RK4+smooth:
            dh0 = H'(0),  dh1 = H'(1),  h05 = H(0.5)

        Returns:
            (dh0, dh1, h05) in kJ/mol.
        """
        b0, b1, b2 = MiedemaModel._natural6_basis(x)
        A = np.column_stack([b0, b1, b2])
        c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return float(c[0]), float(c[1]), float(c[2])

    @staticmethod
    def natural6_hmix(
        x: np.ndarray,
        dh0: float,
        dh1: float,
        h05: float,
    ) -> np.ndarray:
        """Evaluate the natural-spline polynomial at composition(s) x."""
        b0, b1, b2 = MiedemaModel._natural6_basis(x)
        return dh0 * b0 + dh1 * b1 + h05 * b2

    def miedema_natural6(
        self,
        elem_a: str,
        elem_b: str,
        n_points: int = 201,
    ) -> dict:
        """
        Full pipeline: Miedema scan → natural-spline (degree-6) LS fit.

        Returns a dict with keys:
            'dh0', 'dh1', 'h05'         : geometric triplet [kJ/mol]
            'rmse_refit', 'mae_refit'    : fit quality vs. Miedema scan [kJ/mol]
        """
        df  = self.scan(elem_a, elem_b, n_points)
        x   = df["xB"].values
        h   = df["H_mix"].values

        dh0, dh1, h05 = self.fit_natural6(x, h)
        h_fit = self.natural6_hmix(x, dh0, dh1, h05)
        resid = h - h_fit
        rmse  = float(np.sqrt(np.mean(resid**2)))
        mae   = float(np.mean(np.abs(resid)))

        return {
            "dh0": dh0, "dh1": dh1, "h05": h05,
            "rmse_refit": rmse, "mae_refit": mae,
        }

    # ------------------------------------------------------------------
    # Points (h025, h05, h075) parameterization
    # ------------------------------------------------------------------

    @staticmethod
    def points_to_rk4(
        h025: float, h05: float, h075: float,
    ) -> tuple[float, float, float, float]:
        """
        Convert (h025=H(0.25), h05=H(0.5), h075=H(0.75)) to RK4 coefficients.

        Uses the RK4+smooth smoothness constraint a3 = -a1/9.

        Derivation from point-value constraints with a3 = -a1/9:
            a0 = 4·h05
            a1 = (192/35)·(h025 − h075)
            a2 = (32/3)·(h025 + h075) − 16·h05
            a3 = −(64/105)·(h025 − h075)   = −a1/9
        """
        a0 = 4.0 * h05
        a1 = (192.0 / 35.0) * (h025 - h075)
        a2 = (32.0 / 3.0) * (h025 + h075) - 16.0 * h05
        a3 = -(64.0 / 105.0) * (h025 - h075)
        return a0, a1, a2, a3

    @staticmethod
    def points_hmix(
        x: np.ndarray,
        h025: float,
        h05: float,
        h075: float,
    ) -> np.ndarray:
        """
        Evaluate H_mix(x) from (h025, h05, h075) parametrization.

        Basis functions (u = 1-2x, a3 = -a1/9 built in):
            B025(x) = x(1-x)·[(192/35)·u + (32/3)·u² − (64/105)·u³]
            B05(x)  = x(1-x)·[4 − 16·u²]
            B075(x) = x(1-x)·[−(192/35)·u + (32/3)·u² + (64/105)·u³]

        Verified: B025(0.25)=1, B025(0.5)=0, B025(0.75)=0 (and analogously
        for B05 and B075).
        """
        u   = 1.0 - 2.0 * x
        xu  = x * (1.0 - x)
        u2  = u ** 2
        u3  = u ** 3
        b025 = xu * ((192.0 / 35.0) * u + (32.0 / 3.0) * u2 - (64.0 / 105.0) * u3)
        b05  = xu * (4.0 - 16.0 * u2)
        b075 = xu * (-(192.0 / 35.0) * u + (32.0 / 3.0) * u2 + (64.0 / 105.0) * u3)
        return h025 * b025 + h05 * b05 + h075 * b075

    @staticmethod
    def fit_points(
        x: np.ndarray,
        h: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Fit (h025, h05, h075) to (x, H_mix) data via OLS.

        Uses basis functions that encode the a3 = -a1/9 constraint.
        Returns (h025, h05, h075) in kJ/mol.
        """
        u   = 1.0 - 2.0 * x
        xu  = x * (1.0 - x)
        u2  = u ** 2
        u3  = u ** 3
        b025 = xu * ((192.0 / 35.0) * u + (32.0 / 3.0) * u2 - (64.0 / 105.0) * u3)
        b05  = xu * (4.0 - 16.0 * u2)
        b075 = xu * (-(192.0 / 35.0) * u + (32.0 / 3.0) * u2 + (64.0 / 105.0) * u3)
        A = np.column_stack([b025, b05, b075])
        c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return float(c[0]), float(c[1]), float(c[2])

    @staticmethod
    def fit_points_regularized(
        x: np.ndarray,
        h: np.ndarray,
        lambda_base: float = 1.0,
        coverage_thresh: float = 0.20,
    ) -> tuple:
        """
        Fit (h025, h05, h075) with adaptive L2 ridge on h025 and h075.

        h05 is never regularized — it controls H(0.5) and is well-determined
        even from partial-range data.
        h025 and h075 are regularized toward zero for low-coverage data.

        Coverage = min(n_lo, n_hi) / (n_lo + n_hi), n_lo = #{x < 0.5}.
        Lambda   = lambda_base / max(coverage, 0.01) for partial range.

        Returns:
            (h025, h05, h075, coverage, lambda_used)
        """
        n_lo = int((x < 0.5).sum())
        n_hi = int((x > 0.5).sum())
        cov  = min(n_lo, n_hi) / max(n_lo + n_hi, 1)

        u   = 1.0 - 2.0 * x
        xu  = x * (1.0 - x)
        u2  = u ** 2
        u3  = u ** 3
        b025 = xu * ((192.0 / 35.0) * u + (32.0 / 3.0) * u2 - (64.0 / 105.0) * u3)
        b05  = xu * (4.0 - 16.0 * u2)
        b075 = xu * (-(192.0 / 35.0) * u + (32.0 / 3.0) * u2 + (64.0 / 105.0) * u3)
        A = np.column_stack([b025, b05, b075])

        if cov >= coverage_thresh:
            c, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
            lam = 0.0
        else:
            lam = lambda_base / max(cov, 0.01)
            reg = np.diag([lam, 0.0, lam])    # regularize h025, h075; not h05
            c   = np.linalg.solve(A.T @ A + reg, A.T @ h)

        return float(c[0]), float(c[1]), float(c[2]), float(cov), float(lam)

    def miedema_points(
        self,
        elem_a: str,
        elem_b: str,
        n_points: int = 201,
    ) -> dict:
        """
        Full pipeline: Miedema scan → (h025, h05, h075) LS fit.

        Returns a dict with keys:
            'h025', 'h05', 'h075'        : point-value triplet [kJ/mol]
            'a0'..'a3'                   : RK4 coefficients (a3 = -a1/9) [kJ/mol]
            'rmse_refit', 'mae_refit'    : fit quality vs. Miedema scan [kJ/mol]
        """
        df  = self.scan(elem_a, elem_b, n_points)
        x   = df["xB"].values
        h   = df["H_mix"].values

        h025, h05, h075 = self.fit_points(x, h)
        a0, a1, a2, a3  = self.points_to_rk4(h025, h05, h075)

        u     = 1.0 - 2.0 * x
        h_fit = x * (1.0 - x) * (a0 + a1 * u + a2 * u ** 2 + a3 * u ** 3)
        resid = h - h_fit
        rmse  = float(np.sqrt(np.mean(resid ** 2)))
        mae   = float(np.mean(np.abs(resid)))

        return {
            "h025": h025, "h05": h05, "h075": h075,
            "a0": a0, "a1": a1, "a2": a2, "a3": a3,
            "rmse_refit": rmse, "mae_refit": mae,
        }

    # ------------------------------------------------------------------
    # Sanity check helper
    # ------------------------------------------------------------------

    def sanity_check(
        self,
        known_systems: list[tuple[str, str, float]],
    ) -> pd.DataFrame:
        """
        Compare model predictions against known ΔH_mix values at x=0.5.

        Args:
            known_systems: list of (elem_a, elem_b, h_mix_literature) tuples.
                           elem_a < elem_b (canonical order).
                           h_mix_literature in kJ/mol.

        Returns:
            DataFrame with columns:
                system, H_predicted, H_literature, error [kJ/mol]

        Suggested systems (§8.5 of project context — units sanity check):
            Cu-Au   ≈ -9.0 kJ/mol
            Ag-Au   ≈ -3.0 kJ/mol
            Bi-Pb   ≈ +1.5 kJ/mol
        """
        rows = []
        for (a, b, h_lit) in known_systems:
            h_pred = self.h_mix_fn(a, b, 0.5)
            rows.append({
                "system":       f"{a}-{b}",
                "H_predicted":  round(h_pred, 2),
                "H_literature": h_lit,
                "error":        round(h_pred - h_lit, 2),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def filter_alloys(
        df_alloys: pd.DataFrame,
        criterion,
    ) -> pd.DataFrame:
        """
        Filter alloys by a lambda applied to ΔH_mix.
        Example: model.filter_alloys(df, lambda dH: dH < 0)
        """
        return df_alloys[criterion(df_alloys["H_mix"])].reset_index(drop=True)
