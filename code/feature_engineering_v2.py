"""
Feature Engineering Pipeline v2
Focus: Fraud Separation, Historic Fraud, and Velocity
Engine: Polars (High Performance)
"""

import polars as pl
import numpy as np
import time

def run_feature_engineering_v2(input_path, output_path):
    print(f"🚀 Loading raw data from {input_path}...")
    start_all = time.time()
    
    # Load raw data
    df = pl.read_csv(input_path)
    
    # 1. 🕐 PREPROCESSING & TEMPORAL FEATURES
    print("[1/8] Processing temporal features...")
    df = df.with_columns(pl.col("payment_date").str.to_datetime())
    df = df.sort("payment_date") # Sort by date for cumulative/velocity calculations
    
    df = df.with_columns([
        pl.col("payment_date").dt.hour().alias("hour"),
        pl.col("payment_date").dt.weekday().alias("day_of_week"),
        pl.col("payment_date").dt.day().alias("day_of_month"),
        pl.col("payment_date").dt.month().alias("month"),
        ((pl.col("payment_date").dt.weekday() >= 6).cast(pl.Int8)).alias("is_weekend"), # Sat=6, Sun=7 in Polars
        ((pl.col("payment_date").dt.hour() < 6).cast(pl.Int8)).alias("is_night"),
        ((pl.col("payment_date").dt.hour().is_between(9, 17)).cast(pl.Int8)).alias("is_business_hours")
    ])

    # 2. 💰 TRANSACTION FEATURES
    print("[2/8] Processing transaction features...")
    df = df.with_columns([
        pl.col("price").log1p().alias("price_log"),
        pl.col("price").sqrt().alias("price_sqrt"),
        ((pl.col("price") % 100 == 0) & (pl.col("price") > 0)).cast(pl.Int8).alias("is_round_price"),
        (pl.col("installment") > 1).cast(pl.Int8).alias("has_installment"),
        # Transaction per price proxy
        (pl.lit(1.0) / (pl.col("price") + 1e-9)).alias("txn_per_price")
    ])

    # 3. 💳 CARD FEATURES
    print("[3/8] Processing card features...")
    df = df.with_columns([
        (pl.col("bin_number").cast(pl.Utf8) + "_" + pl.col("last_four_digits").cast(pl.Utf8)).alias("card_identifier"),
        pl.col("bin_number").cast(pl.Utf8).str.slice(0, 2).alias("bin_first_2"),
        pl.col("bin_number").cast(pl.Utf8).str.slice(0, 4).alias("bin_first_4"),
        (pl.col("card_type").is_null()).cast(pl.Int8).alias("is_foreign_card")
    ])

    # 4. 🏪 MERCHANT & 🌍 GEO FEATURES
    print("[4/8] Processing merchant and geographic features...")
    df = df.with_columns([
        (pl.col("payment_date") - pl.col("merchant_register_date").str.to_datetime()).dt.total_days().alias("merchant_age_days"),
        (pl.col("buyer_city") != pl.col("merchant_city")).cast(pl.Int8).alias("is_cross_border"),
        (pl.col("buyer_country") != "Turkey").cast(pl.Int8).alias("is_foreign_buyer")
    ])

    # 5. 📜 HISTORIC FRAUD FEATURES (Prior Frauds)
    print("[5/8] Calculating historic fraud involvement (target-leakage aware)...")
    # Cumulative sum of frauds EXCLUDING current row (shift 1)
    df = df.with_columns([
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("merchant_id").fill_null(0).alias("merchant_historic_fraud_count"),
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("buyer_email").fill_null(0).alias("email_has_prior_fraud_count"),
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("card_identifier").fill_null(0).alias("card_has_prior_fraud_count")
    ])
    
    # Cumulative transaction counts for rates
    df = df.with_columns(
        (pl.col("payment_id").cum_count().over("merchant_id") - 1).alias("_merchant_prior_txn_count")
    )
    df = df.with_columns(
        (pl.col("merchant_historic_fraud_count") / (pl.col("_merchant_prior_txn_count") + 1)).alias("merchant_fraud_rate")
    ).drop("_merchant_prior_txn_count")

    # 6. ⚡ VELOCITY FEATURES
    print("[6/8] Calculating velocity features (1h, 24h) using Polars rolling...")
    # Add dummy 'one' for rolling counts
    df = df.with_columns(pl.lit(1).alias("_one"))
    
    df = df.with_columns([
        # Email Velocity
        pl.col("_one").rolling_sum_by(window_size="1h", by="payment_date").over("buyer_email").alias("email_txn_last_1h") - 1,
        pl.col("_one").rolling_sum_by(window_size="24h", by="payment_date").over("buyer_email").alias("email_txn_last_24h") - 1,
        pl.col("price").rolling_sum_by(window_size="24h", by="payment_date").over("buyer_email").alias("email_amount_last_24h") - pl.col("price"),
        
        # Card Velocity
        pl.col("_one").rolling_sum_by(window_size="1h", by="payment_date").over("card_identifier").alias("card_txn_last_1h") - 1,
        pl.col("_one").rolling_sum_by(window_size="24h", by="payment_date").over("card_identifier").alias("card_txn_last_24h") - 1,
        
        # Merchant Velocity
        pl.col("_one").rolling_sum_by(window_size="1h", by="payment_date").over("merchant_id").alias("merchant_txn_last_1h") - 1
    ])
    
    df = df.with_columns(
        (pl.col("merchant_txn_last_1h") / (pl.col("price") + 1e-9)).alias("merchant_txn_per_price_1h")
    ).drop("_one")

    # 7. 📊 AGGREGATION & RISK FEATURES
    print("[7/8] Processing global aggregations...")
    # Global stats for entities
    df = df.with_columns([
        pl.col("payment_id").count().over("buyer_email").alias("email_total_txn_count"),
        pl.col("price").mean().over("buyer_email").alias("email_avg_price"),
        pl.col("merchant_id").n_unique().over("buyer_email").alias("email_unique_merchants"),
        pl.col("payment_id").count().over("card_identifier").alias("card_total_txn_count"),
        pl.col("merchant_id").n_unique().over("card_identifier").alias("card_unique_merchants")
    ])
    
    df = df.with_columns([
        (pl.col("email_total_txn_count") == 1).cast(pl.Int8).alias("is_new_email"),
        (pl.col("card_total_txn_count") == 1).cast(pl.Int8).alias("is_new_card"),
        (abs(pl.col("price") - pl.col("email_avg_price")) / (pl.col("email_avg_price") + 1e-9)).alias("price_deviation_from_avg")
    ])

    # 8. 💾 SAVE OUTPUT (Optimized for space)
    print("[8/8] Saving final features to Parquet (Compressed)...")
    
    # Drop heavyweight/identifying columns to save disk space and ensure privacy
    cols_to_drop = [
        'buyer_name', 'buyer_surname', 'buyer_email', 'buyer_gsm',
        'last_four_digits', 'card_identifier', 'payment_id'
        # payment_date is kept for time-based splitting later but cast to string/int if needed
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns])
    
    # Convert datetime to string for easier handle in some tools, or keep as datetime
    # Parquet handles datetime perfectly, so we keep it.
    
    df.write_parquet(output_path, compression="snappy")
    
    end_all = time.time()
    print(f"✅ Feature Engineering v2 completed in {end_all - start_all:.2f} seconds!")
    print(f"📁 Output saved: {output_path}")
    print(f"📊 Global row count: {len(df):,}")
    print(f"🔢 Total columns: {len(df.columns)}")

if __name__ == "__main__":
    run_feature_engineering_v2(
        "/Users/berkay.camur/Desktop/Study/DS Classification Case Data.csv",
        "/Users/berkay.camur/Desktop/Study/fraud_features_v2.parquet"
    )
