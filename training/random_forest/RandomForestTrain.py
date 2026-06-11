#!/usr/bin/env python3
"""
Train a Random Forest judge for particle re-identification.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import numpy as np

# ===== USER SETTINGS =====
DATASET_DIR = r"D:\Works\AI_Project\Process\05_Judge\Dataset_F4"  # folder with train/val/test.csv
MODEL_OUT   = r"D:\Works\AI_Project\Process\05_Judge\test.pkl"
N_ESTIMATORS = 200   # number of trees
MAX_DEPTH    = 9  # limit tree depth (None = expand fully)
RANDOM_STATE = 42
# =========================

# --- Load datasets ---
train = pd.read_csv(Path(DATASET_DIR) / "train.csv")
val   = pd.read_csv(Path(DATASET_DIR) / "val.csv")
test  = pd.read_csv(Path(DATASET_DIR) / "test.csv")

# Features = all columns except label
# keep label & video_id for targets/groups, but don't use them as features
drop_cols = {
    'label', 
    'video_id',
    'area_difference',      # Drop raw noise-sensitive metrics
    'Ibcnt_difference',     # Drop raw noise-sensitive metrics
    'fwhm_avg_difference'   # Drop raw noise-sensitive metrics
}

# numeric-only features
feature_cols = (
    train.select_dtypes(include=[np.number])
         .columns.difference(drop_cols)
         .tolist()
)

X_train, y_train = train[feature_cols], train["label"]
X_val,   y_val   = val[feature_cols],   val["label"]
X_test,  y_test  = test[feature_cols],  test["label"]

# --- Train Random Forest ---
clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# --- Evaluate ---
print("\nValidation set performance:")
y_val_pred = clf.predict(X_val)
print(classification_report(y_val, y_val_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_val, y_val_pred))
print("ROC AUC:", roc_auc_score(y_val, clf.predict_proba(X_val)[:,1]))

print("\nTest set performance:")
y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

# --- Save model ---
joblib.dump(clf, MODEL_OUT)
print(f"\n[✓] Model saved to {MODEL_OUT}")
