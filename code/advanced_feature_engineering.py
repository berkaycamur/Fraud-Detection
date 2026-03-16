"""
Advanced Feature Engineering - Enhanced Version
Focus: Additional Risk Signals & Behavioral Patterns
Goal: Improve precision from 0.627 to 0.7+
"""

import polars as pl
import numpy as np
import time

def run_advanced_feature_engineering(input_path, output_path):
    print(f"🚀 Loading raw data from {input_path}...")
    start_all = time.time()

    df = pl.read_csv(input_path)

    # 1. PREPROCESSING & TEMPORAL FEATURES
    print("[1/10] Processing temporal features...")
    df = df.with_columns(pl.col("payment_date").str.to_datetime())
    df = df.sort("payment_date")

    df = df.with_columns([
        pl.col("payment_date").dt.hour().alias("hour"),
        pl.col("payment_date").dt.weekday().alias("day_of_week"),
        pl.col("payment_date").dt.day().alias("day_of_month"),
        pl.col("payment_date").dt.month().alias("month"),
        ((pl.col("payment_date").dt.weekday() >= 5).cast(pl.Int8)).alias("is_weekend"),
        ((pl.col("payment_date").dt.hour() < 6).cast(pl.Int8)).alias("is_night"),
        ((pl.col("payment_date").dt.hour().is_between(9, 17)).cast(pl.Int8)).alias("is_business_hours"),
        # New: Late night (22-6)
        ((pl.col("payment_date").dt.hour() >= 22) | (pl.col("payment_date").dt.hour() < 6)).cast(pl.Int8).alias("is_late_night")
    ])

    # 2. TRANSACTION FEATURES
    print("[2/10] Processing transaction features...")
    df = df.with_columns([
        pl.col("price").log1p().alias("price_log"),
        pl.col("price").sqrt().alias("price_sqrt"),
        ((pl.col("price") % 100 == 0) & (pl.col("price") > 0)).cast(pl.Int8).alias("is_round_price"),
        (pl.col("installment") > 1).cast(pl.Int8).alias("has_installment"),
        (pl.lit(1.0) / (pl.col("price") + 1e-9)).alias("txn_per_price"),
        # New: Very small transactions (suspicious)
        (pl.col("price") < 50).cast(pl.Int8).alias("is_micro_txn"),
        # New: Very large transactions
        (pl.col("price") > 5000).cast(pl.Int8).alias("is_large_txn"),
        # New: Price bands
        (pl.col("price") // 500).alias("price_band")
    ])

    # 3. CARD FEATURES
    print("[3/10] Processing card features...")
    df = df.with_columns([
        (pl.col("bin_number").cast(pl.Utf8) + "_" + pl.col("last_four_digits").cast(pl.Utf8)).alias("card_identifier"),
        pl.col("bin_number").cast(pl.Utf8).str.slice(0, 2).alias("bin_first_2"),
        pl.col("bin_number").cast(pl.Utf8).str.slice(0, 4).alias("bin_first_4"),
        (pl.col("card_type").is_null()).cast(pl.Int8).alias("is_foreign_card")
    ])

    # 4. MERCHANT & GEO FEATURES
    print("[4/10] Processing merchant and geographic features...")
    df = df.with_columns([
        (pl.col("payment_date") - pl.col("merchant_register_date").str.to_datetime()).dt.total_days().alias("merchant_age_days"),
        (pl.col("buyer_city") != pl.col("merchant_city")).cast(pl.Int8).alias("is_cross_border"),
        (pl.col("buyer_country") != "Turkey").cast(pl.Int8).alias("is_foreign_buyer"),
        # New: Very young merchants (< 30 days)
        ((pl.col("payment_date") - pl.col("merchant_register_date").str.to_datetime()).dt.total_days() < 30).cast(pl.Int8).alias("is_new_merchant")
    ])

    # 5. HISTORIC FRAUD FEATURES
    print("[5/10] Calculating historic fraud involvement...")
    df = df.with_columns([
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("merchant_id").fill_null(0).alias("merchant_historic_fraud_count"),
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("buyer_email").fill_null(0).alias("email_has_prior_fraud_count"),
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("card_identifier").fill_null(0).alias("card_has_prior_fraud_count"),
        # New: BIN-level fraud history
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("bin_first_4").fill_null(0).alias("bin_historic_fraud_count"),
        # New: City-level fraud
        pl.col("is_fraud_transaction").shift(1).cum_sum().over("buyer_city").fill_null(0).alias("city_historic_fraud_count")
    ])

    # Fraud rates
    df = df.with_columns(
        (pl.col("payment_id").cum_count().over("merchant_id") - 1).alias("_merchant_prior_txn_count")
    )
    df = df.with_columns([
        (pl.col("merchant_historic_fraud_count") / (pl.col("_merchant_prior_txn_count") + 1)).alias("merchant_fraud_rate"),
        # New: Binary flag for any prior fraud
        (pl.col("email_has_prior_fraud_count") > 0).cast(pl.Int8).alias("email_ever_fraud"),
        (pl.col("card_has_prior_fraud_count") > 0).cast(pl.Int8).alias("card_ever_fraud")
    ]).drop("_merchant_prior_txn_count")

    # 6. VELOCITY FEATURES
    print("[6/10] Calculating velocity features...")
    df = df.with_columns(pl.lit(1).alias("_one"))

    df = df.with_columns([
        # Email Velocity
        pl.col("_one").rolling_sum_by(window_size="1h", by="payment_date").over("buyer_email").alias("email_txn_last_1h") - 1,
        pl.col("_one").rolling_sum_by(window_size="24h", by="payment_date").over("buyer_email").alias("email_txn_last_24h") - 1,
        pl.col("price").rolling_sum_by(window_size="24h", by="payment_date").over("buyer_email").alias("email_amount_last_24h") - pl.col("price"),
        # New: 6-hour velocity
        pl.col("_one").rolling_sum_by(window_size="6h", by="payment_date").over("buyer_email").alias("email_txn_last_6h") - 1,

        # Card Velocity
        pl.col("_one").rolling_sum_by(window_size="1h", by="payment_date").over("card_identifier").alias("card_txn_last_1h") - 1,
        pl.col("_one").rolling_sum_by(window_size="24h", by="payment_date").over("card_identifier").alias("card_txn_last_24h") - 1,
        # New: Card spending velocity
        pl.col("price").rolling_sum_by(window_size="1h", by="payment_date").over("card_identifier").alias("card_amount_last_1h") - pl.col("price"),

        # Merchant Velocity
        pl.col("_one").rolling_sum_by(window_size="1h", by="payment_date").over("merchant_id").alias("merchant_txn_last_1h") - 1
    ])

    df = df.with_columns([
        (pl.col("merchant_txn_last_1h") / (pl.col("price") + 1e-9)).alias("merchant_txn_per_price_1h"),
        # New: High velocity flags
        (pl.col("email_txn_last_1h") > 5).cast(pl.Int8).alias("email_high_velocity_1h"),
        (pl.col("card_txn_last_1h") > 3).cast(pl.Int8).alias("card_high_velocity_1h")
    ]).drop("_one")

    # 7. AGGREGATION & RISK FEATURES
    print("[7/10] Processing global aggregations...")
    df = df.with_columns([
        pl.col("payment_id").count().over("buyer_email").alias("email_total_txn_count"),
        pl.col("price").mean().over("buyer_email").alias("email_avg_price"),
        pl.col("price").std().over("buyer_email").alias("email_std_price"),
        pl.col("merchant_id").n_unique().over("buyer_email").alias("email_unique_merchants"),
        pl.col("payment_id").count().over("card_identifier").alias("card_total_txn_count"),
        pl.col("merchant_id").n_unique().over("card_identifier").alias("card_unique_merchants"),
        # New: Merchant statistics
        pl.col("price").mean().over("merchant_id").alias("merchant_avg_price"),
        pl.col("buyer_email").n_unique().over("merchant_id").alias("merchant_unique_customers"),
        # New: BIN statistics
        pl.col("payment_id").count().over("bin_first_4").alias("bin_total_txn_count")
    ])

    df = df.with_columns([
        (pl.col("email_total_txn_count") == 1).cast(pl.Int8).alias("is_new_email"),
        (pl.col("card_total_txn_count") == 1).cast(pl.Int8).alias("is_new_card"),
        (abs(pl.col("price") - pl.col("email_avg_price")) / (pl.col("email_avg_price") + 1e-9)).alias("price_deviation_from_avg"),
        # New: Unusual price for merchant
        (abs(pl.col("price") - pl.col("merchant_avg_price")) / (pl.col("merchant_avg_price") + 1e-9)).alias("price_deviation_from_merchant_avg"),
        # New: Price volatility indicator
        (pl.col("email_std_price") / (pl.col("email_avg_price") + 1e-9)).alias("email_price_cv")
    ])

    # 8. INTERACTION FEATURES
    print("[8/10] Creating interaction features...")
    df = df.with_columns([
        # Risk combinations
        (pl.col("is_new_email") * pl.col("is_new_card")).alias("new_email_and_card"),
        (pl.col("is_night") * pl.col("is_weekend")).alias("night_weekend"),
        (pl.col("is_foreign_card") * pl.col("is_foreign_buyer")).alias("foreign_buyer_and_card"),
        # Velocity x Historic fraud
        (pl.col("email_high_velocity_1h") * pl.col("email_ever_fraud")).alias("high_vel_with_fraud_history"),
        # Cross-border + High amount
        (pl.col("is_cross_border") * pl.col("is_large_txn")).alias("cross_border_large_txn")
    ])

    # 9. EMAIL/CARD RISK PATTERNS
    print("[9/10] Computing email and card risk patterns...")
    df = df.with_columns([
        # Email-Card relationship (same email using multiple cards is risky)
        pl.col("card_identifier").n_unique().over("buyer_email").alias("email_unique_cards"),
        # Card-Email relationship
        pl.col("buyer_email").n_unique().over("card_identifier").alias("card_unique_emails")
    ])

    df = df.with_columns([
        # Risk flags
        (pl.col("email_unique_cards") > 3).cast(pl.Int8).alias("email_many_cards"),
        (pl.col("card_unique_emails") > 2).cast(pl.Int8).alias("card_shared_emails")
    ])

    # 10. SAVE OUTPUT
    print("[10/10] Saving enhanced features...")

    cols_to_drop = [
        'buyer_name', 'buyer_surname', 'buyer_email', 'buyer_gsm',
        'last_four_digits', 'card_identifier', 'payment_id'
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns])

    df.write_parquet(output_path, compression="snappy")

    end_all = time.time()
    print(f"✅ Advanced Feature Engineering completed in {end_all - start_all:.2f} seconds!")
    print(f"📁 Output saved: {output_path}")
    print(f"📊 Global row count: {len(df):,}")
    print(f"🔢 Total columns: {len(df.columns)}")

if __name__ == "__main__":
    run_advanced_feature_engineering(
        "/Users/berkay.camur/Desktop/Study/DS Classification Case Data.csv",
        "/Users/berkay.camur/Desktop/Study/fraud_features_advanced.parquet"
    )
