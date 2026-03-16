"""
Statistical Separation Analysis (Phase 2)
Focus: Identifying features that separate Fraud from Legit.
Tests: ANOVA (F-Test) for numeric, Chi-Square for categorical.
"""

import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import time

def run_separation_analysis(input_path):
    print(f"📈 Loading engineered features from {input_path}...")
    start_time = time.time()
    
    # Load parquet
    df = pl.read_parquet(input_path)
    target = "is_fraud_transaction"
    
    # Identify feature types
    all_cols = df.columns
    ignore_cols = [target, "payment_date", "merchant_id"] # ID/Time cols
    
    # Categorize automatically
    numeric_cols = [c for c in all_cols if df[c].dtype in [pl.Float64, pl.Int64, pl.Int32] and c not in ignore_cols]
    # We treat bool/int8 binary as categorical for Chi-Square or numeric for F-test. 
    # Let's treat everything numeric as F-test for simplicity and focus.
    
    cat_cols = [c for c in all_cols if df[c].dtype == pl.Utf8 and c not in ignore_cols]
    
    # Fraud vs Legit groups
    fraud = df.filter(pl.col(target) == 1)
    legit = df.filter(pl.col(target) == 0)
    
    print(f"Samples: Fraud={len(fraud):,}, Legit={len(legit):,}")
    
    # --- ANOVA FOR NUMERIC FEATURES ---
    print("Computing ANOVA F-test for numeric features...")
    anova_results = []
    for col in numeric_cols:
        f_stat, p_val = f_oneway(fraud[col].to_numpy(), legit[col].to_numpy())
        anova_results.append({
            "feature": col,
            "F_statistic": f_stat,
            "p_value": p_val,
            "mean_fraud": fraud[col].mean(),
            "mean_legit": legit[col].mean(),
            "diff_abs": abs(fraud[col].mean() - legit[col].mean())
        })
        
    anova_df = pd.DataFrame(anova_results).sort_values("F_statistic", ascending=False)
    anova_df.to_csv("anova_feature_ranking.csv", index=False)
    
    # --- CHI-SQUARE FOR CATEGORICAL FEATURES ---
    print("Computing Chi-Square for categorical features...")
    chi_results = []
    # Drop columns with too many levels for chi-square (like binnumber if it was string)
    for col in cat_cols:
        # Sample for speed if needed, but 3.1M is fine for contingency table
        contingency = pd.crosstab(df[col].to_numpy(), df[target].to_numpy())
        chi2, p, dof, ex = chi2_contingency(contingency)
        chi_results.append({
            "feature": col,
            "Chi2_statistic": chi2,
            "p_value": p,
            "unique_levels": df[col].n_unique()
        })
    
    chi_df = pd.DataFrame(chi_results).sort_values("Chi2_statistic", ascending=False)
    chi_df.to_csv("chi_square_ranking.csv", index=False)
    
    print("\nTop 15 Numeric Features (Separation Power):")
    print(anova_df[["feature", "F_statistic", "p_value"]].head(15))
    
    print("\nTop 5 Categorical Features:")
    print(chi_df[["feature", "Chi2_statistic", "p_value"]].head(5))
    
    # --- VISUALIZATION OF TOP FEATURES ---
    print("\nGenerating boxplots for top 6 numeric features...")
    top_features = anova_df["feature"].head(6).tolist()
    
    # Convert a small balanced sample to pandas for plotting
    sample_df = df.sample(n=min(len(df), 100_000)).to_pandas()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_features):
        sns.boxplot(data=sample_df, x=target, y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Separation: {col}")
        axes[i].set_yscale('log') if sample_df[col].max() > 1000 else None
        
    plt.tight_layout()
    plt.savefig("top_feature_separation.png")
    
    end_time = time.time()
    print(f"\n✅ Analysis completed in {end_time - start_time:.2f} seconds.")
    print("📊 Ranking tables and plots saved.")

if __name__ == "__main__":
    run_separation_analysis("/Users/berkay.camur/Desktop/Study/fraud_features_v2.parquet")
