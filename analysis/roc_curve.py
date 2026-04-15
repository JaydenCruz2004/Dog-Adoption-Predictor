## Dog Adoption Project
## Belmont University
## Jayden Cruz
## DSC-4900-01: Data Science Project/Portfolio (Spring 2026)

# roc_curve.py
# Generates the ROC curve for the trained LightGBM model
# Run this after main.py has been run at least once
# Output saved to outputs/roc_curve.png

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so it saves without popping up a window

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from config import TRAIN_PATH, FEATURES, TARGET, SPEED_TO_SCORE, DOG_TYPE
from features import add_sentiment_features
from model import train_model

nltk.download("vader_lexicon", quiet=True)

# ── 1. LOAD AND PREPARE DATA ──────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
train = train[train["Type"] == DOG_TYPE].copy()

# convert adoption speed to 0-1 score
train["AdoptionScore"] = train["AdoptionSpeed"].map(SPEED_TO_SCORE)

# add sentiment features
train = add_sentiment_features(train)

# split into train and validation
X = train[FEATURES]
y = train[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. TRAIN MODEL ────────────────────────────────────────────────────────────
print("Training model for ROC curve...")
model = train_model(X_train, y_train, X_val, y_val)

# ── 3. GET PREDICTIONS ────────────────────────────────────────────────────────
val_preds = np.clip(model.predict(X_val), 0, 1)

# convert to binary: score >= 0.5 means adopted within a month or faster
# score < 0.5 means adopted slowly or not at all
binary_actual = (y_val >= 0.5).astype(int)

# ── 4. COMPUTE ROC CURVE ──────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(binary_actual, val_preds)
auc = roc_auc_score(binary_actual, val_preds)

print("AUC:", round(auc, 4))

# ── 5. PLOT ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# ROC curve line
ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label="LightGBM (AUC = " + str(round(auc, 4)) + ")")

# diagonal reference line — this is what a random model would look like
ax.plot([0, 1], [0, 1], color="#9ca3af", linewidth=1, linestyle="--", label="Random Classifier (AUC = 0.50)")

# shade the area under the curve
ax.fill_between(fpr, tpr, alpha=0.08, color="#2563eb")

# labels and formatting
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve — Dog Adoption Predictor", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

# annotate the AUC value on the plot
ax.annotate(
    "AUC = " + str(round(auc, 4)),
    xy=(0.6, 0.4),
    fontsize=13,
    color="#2563eb",
    fontweight="bold"
)

plt.tight_layout()

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'roc_curve.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150)
print("ROC curve saved to outputs/roc_curve.png")