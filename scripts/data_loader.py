"""
data_loader.py
--------------
Loads raw CSV datasets and prints initial inspection summaries.
Part of the E-Commerce Customer Behavior Analysis project.
"""

import pandas as pd
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

AMAZON_RAW = os.path.join(DATA_DIR, "amazon_sales.csv")
FLIPKART_RAW = os.path.join(DATA_DIR, "flipkart_sales.csv")
FLIPKART_BASIC_RAW = os.path.join(DATA_DIR, "flipkart_sales_basic.csv")


def load_amazon(filepath: str = AMAZON_RAW) -> pd.DataFrame:
    """Load the Amazon product/review dataset."""
    df = pd.read_csv(filepath)
    return df


def load_flipkart(filepath: str = FLIPKART_RAW) -> pd.DataFrame:
    """Load the Flipkart enriched sales dataset."""
    df = pd.read_csv(filepath)
    return df


def load_flipkart_basic(filepath: str = FLIPKART_BASIC_RAW) -> pd.DataFrame:
    """Load the Flipkart basic sales dataset."""
    df = pd.read_csv(filepath)
    return df


def inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    """Print summary statistics for a DataFrame."""
    print(f"\n{'='*60}")
    print(f"  📊 {name} — Inspection Report")
    print(f"{'='*60}")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory Usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\n  --- Column Names ---")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col} ({df[col].dtype})")
    print(f"\n  --- Missing Values ---")
    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df) * 100).round(2)
    for col in df.columns:
        if nulls[col] > 0:
            print(f"  {col}: {nulls[col]} ({nulls_pct[col]}%)")
    if nulls.sum() == 0:
        print("  ✅ No missing values found!")
    print(f"\n  --- Duplicate Rows ---")
    dup_count = df.duplicated().sum()
    print(f"  {dup_count} duplicates ({dup_count/len(df)*100:.1f}%)")
    print(f"\n  --- First 3 Rows ---")
    print(df.head(3).to_string(index=False))
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main — Run inspection when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 E-Commerce Data Loader — Initial Inspection\n")

    # Load datasets
    amazon_df = load_amazon()
    flipkart_df = load_flipkart()
    flipkart_basic_df = load_flipkart_basic()

    # Inspect each
    inspect_dataframe(amazon_df, "Amazon Product/Review Dataset")
    inspect_dataframe(flipkart_df, "Flipkart Enriched Sales Dataset")
    inspect_dataframe(flipkart_basic_df, "Flipkart Basic Sales Dataset")

    print("✅ All datasets loaded and inspected successfully!")
