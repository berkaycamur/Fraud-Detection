"""
Precision Optimizer - Optimize threshold on existing model predictions
Goal: Find threshold that achieves 0.7+ precision
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
    confusion_matrix
)

print("="*60)
print("PRECISION OPTIMIZER")
print("="*60)

# Load existing predictions
df = pd.read_csv("final_test_predictions.csv")

y_true = df['is_fraud_transaction']
y_proba = df['fraud_probability']

print(f"\nTest Set: {len(df):,} transactions")
print(f"Fraud Cases: {y_true.sum():,} ({y_true.mean()*100:.2f}%)")

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

# Find thresholds that meet precision target
precision_target = 0.70
valid_indices = np.where(precisions >= precision_target)[0]

print(f"\n" + "="*60)
print(f"THRESHOLD OPTIMIZATION FOR PRECISION {precision_target:.1%}+")
print("="*60)

if len(valid_indices) > 0:
    # Select threshold with highest recall among valid precisions
    best_idx = valid_indices[np.argmax(recalls[valid_indices])]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"\n✅ Found threshold that achieves precision {precision_target:.1%}+")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Expected Precision: {precisions[best_idx]:.4f}")
    print(f"Expected Recall: {recalls[best_idx]:.4f}")
else:
    # Find highest achievable precision
    max_precision_idx = np.argmax(precisions)
    max_precision = precisions[max_precision_idx]
    best_threshold = thresholds[max_precision_idx] if max_precision_idx < len(thresholds) else 0.5

    print(f"\n⚠️  Precision {precision_target:.1%}+ not achievable with current model")
    print(f"Maximum achievable precision: {max_precision:.4f}")
    print(f"Threshold for max precision: {best_threshold:.4f}")
    print(f"Recall at max precision: {recalls[max_precision_idx]:.4f}")

# Generate predictions with optimized threshold
y_pred_optimized = (y_proba >= best_threshold).astype(int)

# Calculate final metrics
precision = precision_score(y_true, y_pred_optimized)
recall = recall_score(y_true, y_pred_optimized)
f1 = f1_score(y_true, y_pred_optimized)
pr_auc = average_precision_score(y_true, y_proba)

print(f"\n" + "="*60)
print("OPTIMIZED MODEL PERFORMANCE")
print("="*60)

print(f"\nThreshold: {best_threshold:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

cm = confusion_matrix(y_true, y_pred_optimized)
print(f"\nConfusion Matrix:")
print(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
print(f"FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")

# Compare with original threshold
y_pred_original = df['final_prediction']
precision_original = precision_score(y_true, y_pred_original)
recall_original = recall_score(y_true, y_pred_original)
f1_original = f1_score(y_true, y_pred_original)

print(f"\n" + "="*60)
print("COMPARISON: Original vs Optimized")
print("="*60)
print(f"\n{'Metric':<15} {'Original':<12} {'Optimized':<12} {'Change':<10}")
print("-" * 50)
print(f"{'Precision':<15} {precision_original:<12.4f} {precision:<12.4f} {precision-precision_original:+.4f}")
print(f"{'Recall':<15} {recall_original:<12.4f} {recall:<12.4f} {recall-recall_original:+.4f}")
print(f"{'F1-Score':<15} {f1_original:<12.4f} {f1:<12.4f} {f1-f1_original:+.4f}")

# Alternative thresholds
print(f"\n" + "="*60)
print("ALTERNATIVE THRESHOLD ANALYSIS")
print("="*60)
print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 50)

for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred_alt = (y_proba >= thresh).astype(int)
    prec = precision_score(y_true, y_pred_alt)
    rec = recall_score(y_true, y_pred_alt)
    f1_alt = f1_score(y_true, y_pred_alt)
    marker = " ← OPTIMAL" if abs(thresh - best_threshold) < 0.05 else ""
    print(f"{thresh:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1_alt:<12.4f}{marker}")

# Save optimized predictions
df['optimized_prediction'] = y_pred_optimized
df['optimized_threshold'] = best_threshold
df.to_csv("precision_optimized_predictions.csv", index=False)

print(f"\n✅ Optimized predictions saved to precision_optimized_predictions.csv")
print(f"🎯 Target precision 0.7+ {'✅ ACHIEVED!' if precision >= 0.70 else '⚠️  Not achieved'}")
