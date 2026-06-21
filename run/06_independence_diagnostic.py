"""
Independence-assumption diagnostic in P(HEA) terms.

Eq. sig_prop assumes distinct pairs are independent, omitting the off-diagonal
covariance the ARD kernel induces between pairs sharing a common element.
This script quantifies the effect *on the decision variable* P(HEA), not on a
(no-longer-existent) binary S/non-S class label.

For each subset (equimolar composition) it compares:
  sigma_indep = sqrt( w^T diag(K) w )      [independence, Eq. sig_prop]
  sigma_joint = sqrt( w^T K w )            [full GP covariance via predict_joint]
where w_pair = 4 c_i c_j (= 0.16 at equimolar) and K is the joint posterior
covariance of the C(5,2)=10 binary h05 predictions.

It then recomputes P(HEA) with each sigma (all other features fixed; sigma_dH is
the only feature that depends on the pair covariance) and reports the change.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np

from src.phea.predict import HEAPredictor
from src.phea.features import FEATURE_NAMES, DH_DEPLOY_LO, DH_DEPLOY_HI

POOL = [
    "Ag", "Al", "Au", "B",  "Ba", "Bi", "Ca", "Cd", "Ce", "Co",
    "Cr", "Cu", "Eu", "Fe", "Ga", "Gd", "Ge", "Hg", "In", "Ir",
    "Li", "Mg", "Mn", "Na", "Nb", "Ni", "Pb", "Pd", "Pt", "Rh",
    "Sb", "Sc", "Se", "Si", "Sn", "Sr", "Tb", "Te", "Ti", "Tl",
    "Y",  "Yb", "Zn", "Zr",
]
RARE_EARTH = ["Ce", "Eu", "Gd", "Tb", "Y", "Yb", "Sc"]   # chemically homogeneous
REFRACTORY_AU = ["Au", "Nb", "Ti", "Zr"]                  # shared-element + Au tail

pred = HEAPredictor.load()
gp = pred.gp


def diag_subset(elements):
    elements = sorted(elements)
    mu_kJ, K, _ = gp.predict_joint(elements, use_model_std=False)   # (10,), (10,10)
    P = len(mu_kJ)
    w = np.full(P, 4.0 * 0.2 * 0.2)                                 # equimolar pair weight
    mu_dH = float(w @ mu_kJ)
    var_indep = float((w**2) @ np.diag(K))
    var_joint = float(w @ K @ w)
    if var_indep <= 0:
        return None
    sig_indep = np.sqrt(var_indep)
    sig_joint = np.sqrt(max(var_joint, 0.0))
    eta = (var_joint - var_indep) / var_indep

    comp = {e: 0.2 for e in elements}
    fd = pred.feat.compute(comp)
    X = np.array([[fd[k] for k in FEATURE_NAMES]])
    P_indep = float(pred._features_to_phea(X)[0])
    Xj = X.copy()
    Xj[0, 1] = sig_joint                                            # replace sigma_dH only
    P_joint = float(pred._features_to_phea(Xj)[0])

    return dict(
        elements="-".join(elements), mu_dH=mu_dH,
        sig_indep=sig_indep, sig_joint=sig_joint, eta=eta,
        P_indep=P_indep, P_joint=P_joint, dP=P_joint - P_indep,
        in_range=(DH_DEPLOY_LO <= mu_dH <= DH_DEPLOY_HI),
        sigma_feat=fd["sigma_dH"],
    )


SUMMARY = {}


def summarise(rows, label):
    rows = [r for r in rows if r is not None and r["in_range"]]
    if not rows:
        print(f"\n[{label}] no in-range subsets")
        return
    eta = np.array([r["eta"] for r in rows])
    dP = np.array([abs(r["dP"]) for r in rows])
    worst = max(rows, key=lambda r: abs(r["dP"]))
    SUMMARY[label] = dict(
        n_in_range=len(rows),
        eta_median=float(np.median(eta)), eta_p95=float(np.percentile(eta, 95)),
        eta_max=float(eta.max()), eta_min=float(eta.min()),
        absdP_median=float(np.median(dP)), absdP_p95=float(np.percentile(dP, 95)),
        absdP_max=float(dP.max()),
        worst=dict(elements=worst["elements"], mu_dH=worst["mu_dH"],
                   sig_indep=worst["sig_indep"], sig_joint=worst["sig_joint"],
                   P_indep=worst["P_indep"], P_joint=worst["P_joint"], dP=worst["dP"]),
    )
    print(f"\n[{label}]  n_in_range = {len(rows)}")
    print(f"  eta (var_joint/var_indep - 1):  median={np.median(eta):+.3f}  "
          f"95th={np.percentile(eta,95):+.3f}  max={eta.max():+.3f}  min={eta.min():+.3f}")
    print(f"  |dP(HEA)| joint vs indep:        median={np.median(dP):.4f}  "
          f"95th={np.percentile(dP,95):.4f}  max={dP.max():.4f}")
    print(f"  worst dP: {worst['elements']}  mu_dH={worst['mu_dH']:.1f}  "
          f"sig {worst['sig_indep']:.2f}->{worst['sig_joint']:.2f}  "
          f"P {worst['P_indep']:.3f}->{worst['P_joint']:.3f}  (dP={worst['dP']:+.4f})")


# ---- sanity check: compute()'s sigma_dH must match predict_joint diagonal ----
chk = diag_subset(["Co", "Cr", "Cu", "Fe", "Ni"])
print(f"sanity: sigma_feat={chk['sigma_feat']:.4f}  sig_indep(joint-diag)={chk['sig_indep']:.4f}  "
      f"(rel.diff={abs(chk['sigma_feat']-chk['sig_indep'])/chk['sig_indep']:.1e})")

# ---- (1) uniform random sample of the atlas ----
random.seed(0)
seen = set()
rand_subsets = []
while len(rand_subsets) < 1000:
    s = tuple(sorted(random.sample(POOL, 5)))
    if s not in seen:
        seen.add(s)
        rand_subsets.append(s)
rand_rows = [diag_subset(s) for s in rand_subsets]
summarise(rand_rows, "uniform random atlas sample (n=1000)")

# ---- (2) top P(HEA) subsets (3d-TM chemistry, where positive recommendations live) ----
top = [
    ["Co", "Cr", "Cu", "Fe", "Ni"], ["Co", "Cu", "Fe", "Mn", "Ni"],
    ["Co", "Cr", "Cu", "Fe", "Mn"], ["Cr", "Cu", "Fe", "Mn", "Ni"],
    ["Co", "Cr", "Cu", "Mn", "Ni"], ["Co", "Cr", "Fe", "Mn", "Ni"],
    ["Al", "Co", "Cr", "Fe", "Ni"], ["Cu", "Fe", "Mn", "Ni", "Zn"],
]
summarise([diag_subset(s) for s in top], "top-P(HEA) subsets")

# ---- (3) worst case: chemically homogeneous (rare-earth) + refractory/Au tail ----
re_subsets = list(itertools.combinations(RARE_EARTH, 5))                 # C(7,5)=21
au_subsets = [tuple(sorted(REFRACTORY_AU + [x]))
              for x in POOL if x not in REFRACTORY_AU]                   # Au-Nb-Ti-Zr-X
summarise([diag_subset(s) for s in re_subsets], "rare-earth subsets (homogeneous)")
summarise([diag_subset(s) for s in au_subsets], "Au-Nb-Ti-Zr-X subsets (shared-element tail)")

out = Path(__file__).parent.parent / "out" / "data" / "independence_diagnostic.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(SUMMARY, indent=2))
print(f"\nwrote {out}")
