# Physics-Guided Bayesian Framework for HEA Decision Support

Calibrated probabilistic ranking for high-entropy alloy (HEA) discovery. The framework
outputs a calibrated **P(HEA) ∈ [0, 1]** for any 5-component alloy drawn from a
44-element pool, integrating three layers:

1. **Miedema physics** — semi-empirical mixing enthalpy as a domain-knowledge prior.
2. **Gaussian-process posterior** — trained on experimental binary mixing enthalpies,
   yielding a calibrated (μ_ΔH, σ_ΔH) per composition (σ propagated exactly via the
   full GP covariance).
3. **Calibration layer** — logistic regression on experimental HEA outcomes (n = 433),
   with temperature scaling, mapping thermodynamic + Hume-Rothery features to P(HEA).

This is the reproduction code for the manuscript *A Physics-Guided Bayesian Framework
for Decision Support in High-Entropy Alloy Discovery* (Knowledge-Based Systems).

## Install

Uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/rdebski/phea-atlas.git
cd phea-atlas
uv sync          # creates .venv from pyproject.toml + uv.lock (pinned)
```

## Quick start

```python
from src.phea.predict import HEAPredictor

pred = HEAPredictor.load()
p = pred.predict({"Co": 0.2, "Cr": 0.2, "Cu": 0.2, "Fe": 0.2, "Ni": 0.2})
print(p)   # ≈ 0.935
```

## Reproduce the paper

Run in order (each writes to `out/`). The GP model ships pre-trained
(`models/gp_full_model.pt`), so the GP need not be retrained.

| Step | Script | Produces |
|------|--------|----------|
| 1 | `run/01_prepare_exp_data.py`   | `out/data/exp_data_clean.csv` (433 compositions) |
| 2 | `run/02_run_calibration.py`    | `out/models/calibration_model.pkl`, LOO-CV metrics |
| 3 | `run/08_exact_atlas.py`        | `out/data/atlas_phea.csv` — **canonical atlas** (exact σ) |
| 4 | `run/05_decision_value.py`     | `out/data/decision_value.json` — discovery-efficiency analysis |
| 5 | `run/03_run_case_study.py`     | Al-Cu-Fe-Ni-Ti case study |
| 6 | `run/07_landscape.py`, `run/04_atlas_maps.py` | manuscript figures (read the atlas from step 3) |
| 7 | `run/06_independence_diagnostic.py` | independence-approximation diagnostic (appendix) |

```bash
uv run python run/01_prepare_exp_data.py
uv run python run/02_run_calibration.py
uv run python run/08_exact_atlas.py        # ~13 min, 1 CPU
uv run python run/05_decision_value.py
```

**ML baseline comparison (§4.1, optional, compute-heavy):**

```bash
uv run python run/00_ml_baselines.py       # GP vs RF / BNN / Deep Ensemble (LOSO)
```

This script is independent of the headline pipeline.

## Repository layout

```
data/
  processed/
    binary_h05_dataset.csv     # consolidated binary mixing-enthalpy dataset (see below)
    training_pairs.csv         # manifest: the 222 metallic pairs used to train the GP
  miedema_params.csv           # Miedema model parameters
  periodic_table/
    element_properties.csv     # VEC, atomic radius, T_melt, electronegativity
  database_of_HEAs.csv         # experimental HEA outcomes (calibration layer source)
  sources/                     # raw provenance snapshot (NOT read at run time)
    deffrennes2024_rk_params.csv
    entall_binary_enthalpies.csv
models/
  gp_full_model.pt             # pre-trained GP (Matérn-5/2 ARD, exact)
src/
  gp/                          # GP model: data prep, training, prediction, LOSO eval
  data_preparation/            # Miedema model, feature construction, data loaders
  phea/                        # P(HEA) framework: features + single-entry predictor
  baselines/                   # RF / BNN / Deep Ensemble (for the §4.1 comparison)
run/                           # numbered reproduction scripts
tests/                         # pytest suite + plan-consistency verification
out/                           # generated outputs (created on run)
```

## Dataset & provenance

`data/processed/binary_h05_dataset.csv` is our consolidated binary mixing-enthalpy
dataset: **324 binary pairs** with `h05` (ΔH_mix at x = 0.5, kJ/mol), data density
`n_pts`, and precomputed Miedema descriptors. The GP is trained on the **222 metallic
pairs** listed in `training_pairs.csv`.

Each pair carries a **`source`** column recording experimental provenance:

| `source` | meaning | pairs (of 324) |
|----------|---------|----------------|
| `Deffrennes2024`        | Deffrennes et al. (2024) CALPHAD assessment | 251 |
| `Deffrennes2024+Entall` | both sources present (merged)               | 63  |
| `Entall`                | Entall database only (no Deffrennes curve)  | 10  |

Within the 222-pair training set this is 212 Deffrennes + 10 Entall-only, matching the
manuscript. Units are unified to kJ/mol. The raw source files are kept under
`data/sources/` for auditability.

**Cite all three data sources:**

- **Deffrennes et al. (2024)**, *Calphad* **87**, 102745 — binary mixing enthalpies (GP training).
- **Entall database** — supplementary binary enthalpies (10 additional pairs).
- **Chizhevskiy et al. (2026)**, *Sci. Data* **13**(1), 612, DOI
  [10.1038/s41597-026-06930-z](https://doi.org/10.1038/s41597-026-06930-z); dataset
  Mendeley Data DOI [10.17632/j75v9bbbjz.1](https://doi.org/10.17632/j75v9bbbjz.1) —
  experimental HEA outcomes (calibration layer).

## Tests

```bash
uv run pytest                              # unit + integration tests
uv run python tests/verify_plan_consistency.py   # checks manuscript numbers vs outputs
```

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). Derived data under `data/` are redistributed under the terms
of their original sources (cite as above); see each source for its license.
