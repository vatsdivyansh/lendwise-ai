import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # non-interactive backend, saves without needing a display
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── 1. Load processed data ──────────────────────────────────────────────────
df = pd.read_csv('../data/processed/train_processed.csv')

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# ── 2. Split: 80% train / 10% val / 10% test ───────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. Handle class imbalance ───────────────────────────────────────────────
# 78% negatives / 22% positives ≈ ratio of 3.5
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = round(neg / pos, 2)
print(f"Class 0 (approved): {neg} | Class 1 (default): {pos} | scale_pos_weight: {scale}")

# ── 4. Train XGBoost ────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators       = 1000,
    learning_rate      = 0.05,
    max_depth          = 4,
    min_child_weight   = 10,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    scale_pos_weight   = scale,
    eval_metric        = 'auc',
    early_stopping_rounds = 30,
    random_state       = 42,
    verbosity          = 0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50           # prints every 50 rounds so you can watch it train
)

# ── 5. Evaluate on held-out test set ────────────────────────────────────────
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC-ROC (test): {auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ── 6. Confusion matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — XGBoost Loan Approval')
plt.tight_layout()
plt.savefig('../models/confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

# ── 7. Feature importance ────────────────────────────────────────────────────
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(8, 8))
importance.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — XGBoost')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../models/feature_importance.png', dpi=150)
plt.show()
print("Feature importance plot saved.")

# ── 8. Save model ────────────────────────────────────────────────────────────
joblib.dump(model, '../models/loan_model_v1.pkl')
feature_cols = X_train.columns.tolist()
joblib.dump(feature_cols, '../models/feature_cols.pkl')
print(f"Feature columns saved: {len(feature_cols)} cols")
print(f"\nModel saved to models/loan_model_v1.pkl")
print(f"Best iteration: {model.best_iteration}")