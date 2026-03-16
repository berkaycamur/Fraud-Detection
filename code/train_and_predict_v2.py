"""
Final Model Training & Prediction (Phase 4)
Focus: Final Fraud Prediction on Out-Of-Time (OOT) Test Set.
Engine: XGBoost
"""

import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    average_precision_score, precision_recall_curve, f1_score
)
import pickle
import time
import os

def run_training_pipeline(input_path):
    print(f"🎯 Loading enriched features from {input_path}...")
    start_time = time.time()
    
    # Load data
    df = pl.read_parquet(input_path)
    # Ensure payment_date is datetime for splitting
    if df["payment_date"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("payment_date").str.to_datetime())
    
    target = "is_fraud_transaction"
    split_date = pd.Timestamp("2024-09-01")
    
    # Identify features
    all_cols = df.columns
    # Drop identifying or leaky features (if any remains)
    drop_cols = [target, "payment_date", "merchant_id", "merchant_register_date"]
    feature_cols = [c for c in all_cols if c not in drop_cols]
    
    # 1. TIME-BASED SPLIT
    print(f"Splitting data at {split_date}...")
    train_pl = df.filter(pl.col("payment_date") < split_date)
    test_pl = df.filter(pl.col("payment_date") >= split_date)
    
    # Convert to pandas for training (XGBoost handles pandas well with categorical support or preprocessing)
    X_train_df = train_pl.select(feature_cols).to_pandas()
    y_train = train_pl.select(target).to_pandas().values.ravel()
    
    X_test_df = test_pl.select(feature_cols).to_pandas()
    y_test = test_pl.select(target).to_pandas().values.ravel()
    
    print(f"Train size: {len(y_train):,}, Test size: {len(y_test):,}")
    
    # 2. LABEL ENCODING FOR CATEGORICAL
    print("Encoding categorical features...")
    cat_cols = X_train_df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Handle unseen labels in test set
        X_train_df[col] = le.fit_transform(X_train_df[col].astype(str))
        X_test_df[col] = X_test_df[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        label_encoders[col] = le
    
    # 3. XGBOOST TRAINING
    print("Training XGBoost with 64 enriched features...")
    # Calculate scale_pos_weight for imbalance
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        tree_method='hist', # Fast training on large data
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    
    model.fit(
        X_train_df, y_train,
        eval_set=[(X_test_df, y_test)],
        verbose=100
    )
    
    # 4. PREDICTIONS & EVALUATION
    print("Generating predictions on OOT test set...")
    y_proba = model.predict_proba(X_test_df)[:, 1]
    
    # Finding optimal threshold for F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_t = thresholds[np.argmax(f1_scores)]
    
    y_pred = (y_proba >= best_t).astype(int)
    
    print(f"\nOptimal Threshold: {best_t:.4f}")
    print("\nModel Performance (at optimal threshold):")
    print(classification_report(y_test, y_pred))
    
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    
    # 5. SAVE RESULTS
    print("Saving prediction results...")
    results_df = test_pl.select(["payment_date", target]).to_pandas()
    results_df['fraud_probability'] = y_proba
    results_df['final_prediction'] = y_pred
    results_df.to_csv("final_test_predictions.csv", index=False)
    
    # Save model and encoders
    model.save_model("final_xgb_fraud_model.json")
    with open("label_encoders_v2.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
        
    end_time = time.time()
    print(f"\n✅ Prediction Phase completed in {end_time - start_time:.2f} seconds.")
    print(f"📊 Results saved to final_test_predictions.csv")

if __name__ == "__main__":
    run_pipeline_path = "/Users/berkay.camur/Desktop/Study/fraud_features_v2.parquet"
    if os.path.exists(run_pipeline_path):
        run_training_pipeline(run_pipeline_path)
    else:
        print(f"❌ Error: {run_pipeline_path} not found.")
