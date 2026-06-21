"""
Train and validate the calibration layer: composition features → P(HEA).

Architecture
------------
  GP posterior (models/gp_full_model.pt)
      → multicomponent features (src/phea/features.py)
      → StandardScaler + LogisticRegression
      → P(HEA) ∈ [0, 1]

Features (8):  mu_dH, sigma_dH, pen_lo, pen_hi, pen_omega, delta, dS_R, T_m
Label:         is_hea  (hard binary, p_label >= 0.5)
Sample weight: n_reports  (compositions with more experimental reports count more)
Validation:    Leave-One-Out CV  (n=433)

Outputs
-------
  out/models/calibration_model.pkl         — sklearn Pipeline (scaler + logreg)
  out/data/calibration_features.csv        — computed features for all 433 compositions
  out/data/calibration_predictions.csv     — LOO-CV predicted P(HEA) per composition
  out/data/calibration_metrics.json        — AUC, Brier, ECE, coefficients
  out/figures/reliability_diagram_phea.pdf/png
"""

import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP_CSV    = ROOT / "out" / "data" / "exp_data_clean.csv"
MODEL_PT   = ROOT / "models" / "gp_full_model.pt"
OUT_MODELS = ROOT / "out" / "models"
OUT_DATA   = ROOT / "out" / "data"
OUT_FIGS   = ROOT / "out" / "figures"
for d in [OUT_MODELS, OUT_DATA, OUT_FIGS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load experimental dataset
# ---------------------------------------------------------------------------
print("Loading experimental dataset...")
df = pd.read_csv(EXP_CSV)
df["composition"] = df["composition"].apply(json.loads)
df["elements"]    = df["elements"].apply(ast.literal_eval)

print(f"  Compositions: {len(df)}")
print(f"  is_hea=1:     {df['is_hea'].sum()}")
print(f"  is_hea=0:     {(df['is_hea'] == 0).sum()}")
print(f"  Conflicts:    {df['conflict'].sum()}")

# ---------------------------------------------------------------------------
# Compute multicomponent features
# ---------------------------------------------------------------------------
print("\nLoading GP model...")
from src.gp.predict import GPPredictor
from src.phea.features import MulticomponentFeatures, FEATURE_NAMES

gp   = GPPredictor.load(str(MODEL_PT))
feat = MulticomponentFeatures(gp)

print("Pre-computing GP pair cache for all 433 compositions...")
all_elements = set()
for elems in df["elements"]:
    all_elements.update(elems)
feat.precompute_pairs(sorted(all_elements))
print(f"  Unique pairs cached: {len(feat._pair_cache)}")

print("Computing features...")
compositions = df["composition"].tolist()
X_df = feat.compute_batch(compositions, verbose=True)

# Attach metadata for output
X_df.index = df.index
feat_csv = OUT_DATA / "calibration_features.csv"
pd.concat([df[["comp_key", "is_hea", "p_label", "n_reports",
               "conflict", "structure_type"]], X_df], axis=1).to_csv(
    feat_csv, index=False
)
print(f"  Features saved: {feat_csv}")

X = X_df[FEATURE_NAMES].values
y = df["is_hea"].values
w = df["n_reports"].values.astype(float)
p_soft = df["p_label"].values

# ---------------------------------------------------------------------------
# Feature summary
# ---------------------------------------------------------------------------
print("\n=== Feature summary ===")
summary = X_df.describe().loc[["mean", "std", "min", "max"]]
print(summary.to_string())

# ---------------------------------------------------------------------------
# Leave-One-Out Cross-Validation
# ---------------------------------------------------------------------------
print("\nRunning Leave-One-Out CV (n=433)...")

loo = LeaveOneOut()
p_loo = np.zeros(len(y))

for i, (train_idx, test_idx) in enumerate(loo.split(X)):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                       random_state=42)),
    ])
    pipe.fit(X[train_idx], y[train_idx], logreg__sample_weight=w[train_idx])
    p_loo[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(y)}")

print(f"  Done. P(HEA) range: [{p_loo.min():.3f}, {p_loo.max():.3f}]")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
auc    = roc_auc_score(y, p_loo)
brier  = brier_score_loss(y, p_loo)
brier_soft = float(np.mean((p_loo - p_soft) ** 2))

# ECE (10 equal-width bins)
def compute_ece(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        avg_conf = p_pred[mask].mean()
        frac_pos = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_conf - frac_pos)
    return ece

ece = compute_ece(y, p_loo)

# ---------------------------------------------------------------------------
# Temperature scaling on LOO logits
# ---------------------------------------------------------------------------
p_clip  = np.clip(p_loo, 1e-6, 1 - 1e-6)
logits  = np.log(p_clip / (1 - p_clip))

def nll(T):
    p = expit(logits / T)
    return -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

T_opt   = minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded").x
p_cal   = expit(logits / T_opt)

auc_cal   = roc_auc_score(y, p_cal)
brier_cal = brier_score_loss(y, p_cal)
ece_cal   = compute_ece(y, p_cal)

print(f"\n=== LOO-CV Metrics (before / after temperature scaling) ===")
print(f"  {'':22s}  {'Raw LogReg':>12s}  {'+ T-scaling':>12s}")
print(f"  {'Temperature T':22s}  {'1.000':>12s}  {T_opt:>12.3f}")
print(f"  {'AUC-ROC':22s}  {auc:>12.4f}  {auc_cal:>12.4f}")
print(f"  {'Brier (hard)':22s}  {brier:>12.4f}  {brier_cal:>12.4f}")
print(f"  {'Brier (soft)':22s}  {brier_soft:>12.4f}")
print(f"  {'ECE':22s}  {ece:>12.4f}  {ece_cal:>12.4f}")

# Threshold-based metrics at P=0.5
y_pred_bin = (p_loo >= 0.5).astype(int)
tp = int(((y == 1) & (y_pred_bin == 1)).sum())
fp = int(((y == 0) & (y_pred_bin == 1)).sum())
fn = int(((y == 1) & (y_pred_bin == 0)).sum())
tn = int(((y == 0) & (y_pred_bin == 0)).sum())
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
print(f"  @ threshold=0.5:")
print(f"    Precision      : {precision:.4f}")
print(f"    Recall (HEA=1) : {recall:.4f}")
print(f"    TP/FP/FN/TN    : {tp}/{fp}/{fn}/{tn}")

# ---------------------------------------------------------------------------
# Train final model on all data + fit temperature on all data
# ---------------------------------------------------------------------------
print("\nTraining final model on all 433 compositions...")
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                   random_state=42)),
])
final_model.fit(X, y, logreg__sample_weight=w)

# Temperature: re-fit on all data (for deployment; LOO value stored in metrics)
p_train_clip = np.clip(
    final_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6
)
logits_train = np.log(p_train_clip / (1 - p_train_clip))
T_final = minimize_scalar(
    lambda T: -np.mean(y * np.log(expit(logits_train / T) + 1e-15)
                       + (1 - y) * np.log(1 - expit(logits_train / T) + 1e-15)),
    bounds=(0.1, 20.0), method="bounded",
).x
print(f"  Temperature (on full data): {T_final:.3f}")

coef   = final_model.named_steps["logreg"].coef_[0]
scaler = final_model.named_steps["scaler"]
# Coefficients in original (unscaled) units for interpretability
coef_orig = coef / scaler.scale_

print("\n=== Logistic regression coefficients ===")
print(f"  {'Feature':<12s}  {'coef (scaled)':>14s}  {'coef / std':>12s}")
for name, c, cs in zip(FEATURE_NAMES, coef_orig, coef):
    print(f"  {name:<12s}  {c:>14.4f}  {cs:>12.4f}")

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
model_path = OUT_MODELS / "calibration_model.pkl"
with open(model_path, "wb") as fh:
    pickle.dump({"pipeline": final_model, "temperature": T_final}, fh)
print(f"\nModel saved: {model_path}")

pred_df = df[["comp_key", "is_hea", "p_label", "n_reports",
              "conflict", "structure_type"]].copy()
pred_df["P_HEA_loocv_raw"] = p_loo
pred_df["P_HEA_loocv_cal"] = p_cal
pred_path = OUT_DATA / "calibration_predictions.csv"
pred_df.to_csv(pred_path, index=False)
print(f"Predictions saved: {pred_path}")

metrics = {
    "n_compositions": int(len(y)),
    "n_hea_1": int(y.sum()),
    "n_hea_0": int((y == 0).sum()),
    "loocv_raw": {
        "auc_roc":    round(float(auc), 4),
        "brier_hard": round(float(brier), 4),
        "brier_soft": round(float(brier_soft), 4),
        "ece":        round(float(ece), 4),
    },
    "loocv_temperature_scaled": {
        "temperature": round(float(T_opt), 4),
        "auc_roc":     round(float(auc_cal), 4),
        "brier_hard":  round(float(brier_cal), 4),
        "ece":         round(float(ece_cal), 4),
    },
    "temperature_final": round(float(T_final), 4),
    "precision_at_0.5": round(float(precision), 4),
    "recall_at_0.5": round(float(recall), 4),
    "features": FEATURE_NAMES,
    "coefficients": {
        name: round(float(c), 6)
        for name, c in zip(FEATURE_NAMES, coef)
    },
}
metrics_path = OUT_DATA / "calibration_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics, fh, indent=2)
print(f"Metrics saved: {metrics_path}")

# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))

fp_raw, mp_raw = calibration_curve(y, p_loo,  n_bins=10, strategy="uniform")
fp_cal, mp_cal = calibration_curve(y, p_cal,  n_bins=10, strategy="uniform")

ax.plot(mp_raw, fp_raw, "s--", color="#aaaaaa", lw=1.5, ms=6,
        label=f"Raw LogReg  (ECE={ece:.3f})")
ax.plot(mp_cal, fp_cal, "s-",  color="#2166ac", lw=2,   ms=7,
        label=f"+ T-scaling (ECE={ece_cal:.3f}, T={T_opt:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

ax.set_xlabel("Mean predicted P(HEA)", fontsize=12)
ax.set_ylabel("Fraction of HEA (observed)", fontsize=12)
ax.set_title("Reliability diagram — calibration layer (LOO-CV)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax_in = ax.inset_axes([0.62, 0.10, 0.35, 0.25])
ax_in.hist(p_cal, bins=20, color="#2166ac", alpha=0.7, edgecolor="white")
ax_in.set_xlabel("P(HEA)", fontsize=8)
ax_in.set_ylabel("Count", fontsize=8)
ax_in.tick_params(labelsize=7)

plt.tight_layout()
for ext in ("pdf", "png"):
    fp = OUT_FIGS / f"reliability_diagram_phea.{ext}"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
print(f"Figure saved: {OUT_FIGS / 'reliability_diagram_phea.pdf'}")
plt.close()

print("\nDone.")
