"""
cleaning.py
-----------
Cleans both Amazon and Flipkart datasets, engineers features,
and saves cleaned CSVs ready for analysis.

Amazon  : Product/review-level data → price cleaning, rating parsing,
          discount calculation, category extraction, sentiment prep.
Flipkart: Transactional order data  → date parsing, numeric cleaning,
          feature engineering, synthetic customer_id generation.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import re

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

AMAZON_RAW = os.path.join(DATA_DIR, "amazon_sales.csv")
FLIPKART_RAW = os.path.join(DATA_DIR, "flipkart_sales.csv")

AMAZON_CLEAN = os.path.join(DATA_DIR, "amazon_cleaned.csv")
FLIPKART_CLEAN = os.path.join(DATA_DIR, "flipkart_cleaned.csv")


# ===================================================================
#  AMAZON CLEANING
# ===================================================================
def clean_amazon(filepath: str = AMAZON_RAW) -> pd.DataFrame:
    """
    Clean the Amazon product/review dataset.

    Steps:
      1. Standardize column names to snake_case
      2. Clean currency fields (₹, commas)
      3. Parse rating and rating_count
      4. Parse discount_percentage
      5. Extract main category from pipe-separated category string
      6. Engineer new features (discount_amount, price_tier)
      7. Add platform label
      8. Drop duplicates and handle nulls
    """
    print("\n🔧 Cleaning Amazon dataset...")
    df = pd.read_csv(filepath)
    print(f"   Raw shape: {df.shape}")

    # --- 1. Column names are already snake_case in this dataset ---
    # Verify and standardize
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- 2. Clean currency fields ---
    for col in ["discounted_price", "actual_price"]:
        if col in df.columns:
            # Remove ₹ symbol, commas, spaces, and any unicode currency chars
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace("â\x82¹", "", regex=False)  # UTF-8 encoded ₹
                .str.replace(",", "", regex=False)
                .str.replace("�", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 3. Clean rating and rating_count ---
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(
            df["rating"].astype(str).str.strip(), errors="coerce"
        )

    if "rating_count" in df.columns:
        df["rating_count"] = (
            df["rating_count"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

    # --- 4. Clean discount_percentage ---
    if "discount_percentage" in df.columns:
        df["discount_percentage"] = (
            df["discount_percentage"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["discount_percentage"] = pd.to_numeric(
            df["discount_percentage"], errors="coerce"
        )

    # --- 5. Extract main category ---
    if "category" in df.columns:
        df["main_category"] = (
            df["category"].astype(str).str.split("|").str[0].str.strip()
        )
        df["sub_category"] = (
            df["category"]
            .astype(str)
            .str.split("|")
            .str[1]
            .fillna("Unknown")
            .str.strip()
        )

    # --- 6. Feature engineering ---
    # Discount amount
    if "actual_price" in df.columns and "discounted_price" in df.columns:
        df["discount_amount"] = df["actual_price"] - df["discounted_price"]

    # Price tier
    if "discounted_price" in df.columns:
        df["price_tier"] = pd.cut(
            df["discounted_price"],
            bins=[0, 500, 2000, 10000, float("inf")],
            labels=["Budget", "Mid-Range", "Premium", "Luxury"],
            right=True,
        )

    # Rating tier
    if "rating" in df.columns:
        df["rating_tier"] = pd.cut(
            df["rating"],
            bins=[0, 2, 3, 4, 5],
            labels=["Poor", "Average", "Good", "Excellent"],
            right=True,
        )

    # --- 7. Platform label ---
    df["platform"] = "Amazon"

    # --- 8. Drop duplicates & handle nulls ---
    before_dedup = len(df)
    df.drop_duplicates(subset=["product_id"], keep="first", inplace=True)
    print(f"   Duplicates removed: {before_dedup - len(df)}")

    # Fill critical nulls
    df["rating"].fillna(df["rating"].median(), inplace=True)
    df["rating_count"].fillna(0, inplace=True)
    df["discounted_price"].fillna(df["discounted_price"].median(), inplace=True)
    df["actual_price"].fillna(df["actual_price"].median(), inplace=True)
    df.dropna(subset=["product_name"], inplace=True)

    print(f"   Cleaned shape: {df.shape}")
    print(f"   Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ===================================================================
#  FLIPKART CLEANING
# ===================================================================
def clean_flipkart(filepath: str = FLIPKART_RAW) -> pd.DataFrame:
    """
    Clean the Flipkart enriched sales dataset.

    Steps:
      1. Standardize column names to snake_case
      2. Parse dates
      3. Clean numeric columns
      4. Engineer new features (profit_margin, order_month/year/quarter)
      5. Generate synthetic customer_id for RFM analysis
      6. Add platform label
      7. Drop duplicates and handle nulls
    """
    print("\n🔧 Cleaning Flipkart dataset...")
    df = pd.read_csv(filepath)
    print(f"   Raw shape: {df.shape}")

    # --- 1. Standardize column names ---
    col_map = {
        "Order ID": "order_id",
        "Product Name": "product_name",
        "Category": "category",
        "Price (INR)": "price",
        "Quantity Sold": "quantity",
        "Total Sales (INR)": "total_sales",
        "Order Date": "order_date",
        "Payment Method": "payment_method",
        "Customer Rating": "customer_rating",
        "Month": "month",
        "Year": "year",
        "Profit (INR)": "profit",
        "Discount %": "discount_pct",
        "Customer Segment": "customer_segment",
        "Region": "region",
    }
    df.rename(columns=col_map, inplace=True)

    # --- 2. Parse dates ---
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # --- 3. Clean numeric columns ---
    for col in ["price", "total_sales", "profit", "discount_pct", "customer_rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)

    # --- 4. Feature engineering ---
    # Profit margin percentage
    if "profit" in df.columns and "total_sales" in df.columns:
        df["profit_margin_pct"] = (
            (df["profit"] / df["total_sales"].replace(0, np.nan)) * 100
        ).round(2)

    # Time features (derive from order_date if month/year already exist, keep both)
    df["order_month"] = df["order_date"].dt.month
    df["order_year"] = df["order_date"].dt.year
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_day_of_week"] = df["order_date"].dt.day_name()

    # Price tier
    df["price_tier"] = pd.cut(
        df["price"],
        bins=[0, 5000, 15000, 35000, float("inf")],
        labels=["Budget", "Mid-Range", "Premium", "Luxury"],
        right=True,
    )

    # --- 5. Generate synthetic customer_id ---
    # Strategy: distribute 1000 orders across ~200 synthetic customers
    # with varying purchase frequencies (realistic distribution)
    np.random.seed(42)
    n_customers = 200
    # Use a power-law-ish distribution: some customers buy more
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
    # Weighted random assignment — some customers get more orders
    weights = np.random.exponential(scale=1.0, size=n_customers)
    weights = weights / weights.sum()
    df["customer_id"] = np.random.choice(customer_ids, size=len(df), p=weights)

    # --- 6. Platform label ---
    df["platform"] = "Flipkart"

    # --- 7. Drop duplicates & handle nulls ---
    before_dedup = len(df)
    df.drop_duplicates(subset=["order_id"], keep="first", inplace=True)
    print(f"   Duplicates removed: {before_dedup - len(df)}")

    df["customer_rating"].fillna(df["customer_rating"].median(), inplace=True)
    df["profit"].fillna(0, inplace=True)
    df.dropna(subset=["order_date", "product_name"], inplace=True)

    print(f"   Cleaned shape: {df.shape}")
    print(f"   Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ===================================================================
#  MAIN EXECUTION
# ===================================================================
def main():
    """Run the full cleaning pipeline and save outputs."""
    print("\n" + "=" * 60)
    print("  🧹 E-Commerce Data Cleaning Pipeline")
    print("=" * 60)

    # Clean Amazon
    amazon_df = clean_amazon()
    amazon_df.to_csv(AMAZON_CLEAN, index=False)
    print(f"\n   ✅ Amazon cleaned data saved to: {AMAZON_CLEAN}")

    # Clean Flipkart
    flipkart_df = clean_flipkart()
    flipkart_df.to_csv(FLIPKART_CLEAN, index=False)
    print(f"   ✅ Flipkart cleaned data saved to: {FLIPKART_CLEAN}")

    # Summary
    print("\n" + "=" * 60)
    print("  📊 Cleaning Summary")
    print("=" * 60)
    print(f"  Amazon  : {amazon_df.shape[0]:,} rows × {amazon_df.shape[1]} cols")
    print(f"  Flipkart: {flipkart_df.shape[0]:,} rows × {flipkart_df.shape[1]} cols")
    print(f"  Amazon columns   : {list(amazon_df.columns)}")
    print(f"  Flipkart columns : {list(flipkart_df.columns)}")
    print("=" * 60)
    print("  ✅ All datasets cleaned and saved successfully!\n")


if __name__ == "__main__":
    main()
