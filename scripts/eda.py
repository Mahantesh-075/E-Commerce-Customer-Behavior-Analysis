"""
eda.py
------
Exploratory Data Analysis for both Amazon and Flipkart datasets.

Amazon  : Product-level analysis (prices, ratings, categories, reviews)
Flipkart: Transactional analysis (sales trends, regional, profit, segments)

Generates 14+ interactive Plotly and static Matplotlib/Seaborn charts.
All plots are saved to outputs/plots/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ===================================================================
#  FLIPKART EDA (Transactional Data)
# ===================================================================
def flipkart_eda(df: pd.DataFrame) -> None:
    """Run comprehensive EDA on Flipkart transactional data."""
    print("\n📊 Running Flipkart EDA...")

    # --- 1. Monthly Sales Trend ---
    print("   [1/7] Monthly Sales Trend...")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    monthly = (
        df.groupby(df["order_date"].dt.to_period("M"))["total_sales"]
        .sum()
        .reset_index()
    )
    monthly["order_date"] = monthly["order_date"].astype(str)

    fig = px.line(
        monthly,
        x="order_date",
        y="total_sales",
        title="📈 Flipkart — Monthly Sales Trend (2024)",
        labels={"total_sales": "Total Sales (₹)", "order_date": "Month"},
        template="plotly_white",
        markers=True,
    )
    fig.update_traces(line_color="#6366f1", line_width=3, marker_size=8)
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
        xaxis_tickangle=-45,
    )
    fig.write_html(os.path.join(PLOTS_DIR, "flipkart_monthly_sales.html"))

    # --- 2. Top 10 Categories by Revenue ---
    print("   [2/7] Top Categories by Revenue...")
    top_cats = (
        df.groupby("category")["total_sales"].sum().nlargest(10).reset_index()
    )
    fig = px.bar(
        top_cats,
        x="total_sales",
        y="category",
        orientation="h",
        title="🏆 Top 10 Categories by Revenue (Flipkart)",
        color="total_sales",
        color_continuous_scale="Purples",
        template="plotly_white",
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
    )
    fig.write_html(os.path.join(PLOTS_DIR, "flipkart_top_categories.html"))

    # --- 3. Regional Sales Distribution ---
    print("   [3/7] Regional Sales Distribution...")
    region_sales = df.groupby("region")["total_sales"].sum().reset_index()
    fig = px.pie(
        region_sales,
        names="region",
        values="total_sales",
        title="🌍 Regional Sales Distribution (Flipkart)",
        template="plotly_white",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "flipkart_region_distribution.html"))

    # --- 4. Discount vs Profit Correlation ---
    print("   [4/7] Discount vs Profit Correlation...")
    fig_mpl, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter_data = df.dropna(subset=["discount_pct", "profit"])
    sns.scatterplot(
        data=scatter_data,
        x="discount_pct",
        y="profit",
        hue="category",
        alpha=0.6,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Discount % vs Profit", fontsize=14, fontweight="bold")
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Discount %")
    axes[0].set_ylabel("Profit (₹)")

    corr_cols = ["total_sales", "profit", "discount_pct", "quantity", "customer_rating"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=axes[1],
        square=True,
        linewidths=0.5,
    )
    axes[1].set_title("Correlation Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "flipkart_discount_profit_correlation.png"), dpi=150
    )
    plt.close()

    # --- 5. Quarterly Sales vs Profit ---
    print("   [5/7] Quarterly Sales vs Profit...")
    quarterly = (
        df.groupby(["order_year", "order_quarter"])
        .agg(total_sales=("total_sales", "sum"), total_profit=("profit", "sum"))
        .reset_index()
    )
    quarterly["label"] = (
        "Q" + quarterly["order_quarter"].astype(str) + "-" + quarterly["order_year"].astype(str)
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=quarterly["label"],
            y=quarterly["total_sales"],
            name="Sales",
            marker_color="#6366f1",
            opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=quarterly["label"],
            y=quarterly["total_profit"],
            name="Profit",
            line=dict(color="#f59e0b", width=3),
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="📊 Quarterly Sales vs Profit (Flipkart)",
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
    )
    fig.update_yaxes(title_text="Total Sales (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Total Profit (₹)", secondary_y=True)
    fig.write_html(os.path.join(PLOTS_DIR, "flipkart_quarterly_sales_profit.html"))

    # --- 6. Payment Method Distribution ---
    print("   [6/7] Payment Method Distribution...")
    payment_dist = df["payment_method"].value_counts().reset_index()
    payment_dist.columns = ["method", "count"]
    fig = px.pie(
        payment_dist,
        names="method",
        values="count",
        title="💳 Payment Method Distribution (Flipkart)",
        template="plotly_white",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "flipkart_payment_methods.html"))

    # --- 7. Customer Segment Distribution ---
    print("   [7/7] Customer Segment Distribution...")
    if "customer_segment" in df.columns:
        seg_dist = df["customer_segment"].value_counts().reset_index()
        seg_dist.columns = ["segment", "count"]
        fig = px.bar(
            seg_dist,
            x="segment",
            y="count",
            color="segment",
            title="👥 Customer Segment Distribution (Flipkart)",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif"),
            title_font_size=18,
            showlegend=False,
        )
        fig.write_html(
            os.path.join(PLOTS_DIR, "flipkart_customer_segments.html")
        )

    print("   ✅ Flipkart EDA complete — 7 charts saved!")


# ===================================================================
#  AMAZON EDA (Product/Review Data)
# ===================================================================
def amazon_eda(df: pd.DataFrame) -> None:
    """Run comprehensive EDA on Amazon product/review data."""
    print("\n📊 Running Amazon EDA...")

    # --- 1. Category Distribution ---
    print("   [1/7] Category Distribution...")
    cat_counts = df["main_category"].value_counts().head(12).reset_index()
    cat_counts.columns = ["category", "count"]
    fig = px.bar(
        cat_counts,
        x="count",
        y="category",
        orientation="h",
        title="📦 Amazon — Product Category Distribution",
        color="count",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
    )
    fig.write_html(os.path.join(PLOTS_DIR, "amazon_category_distribution.html"))

    # --- 2. Price Distribution ---
    print("   [2/7] Price Distribution...")
    fig_mpl, axes = plt.subplots(1, 2, figsize=(16, 6))

    price_data = df["discounted_price"].dropna()
    price_data = price_data[price_data < price_data.quantile(0.95)]  # Remove outliers

    axes[0].hist(price_data, bins=50, color="#6366f1", edgecolor="white", alpha=0.85)
    axes[0].set_title("Price Distribution (Discounted)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Price (₹)")
    axes[0].set_ylabel("Count")

    # Box plot by main_category (top 8)
    top_cats = df["main_category"].value_counts().head(8).index
    box_data = df[df["main_category"].isin(top_cats)]
    sns.boxplot(
        data=box_data,
        x="main_category",
        y="discounted_price",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("Price by Category (Top 8)", fontsize=14, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Discounted Price (₹)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "amazon_price_distribution.png"), dpi=150)
    plt.close()

    # --- 3. Rating Distribution ---
    print("   [3/7] Rating Distribution...")
    fig = px.histogram(
        df.dropna(subset=["rating"]),
        x="rating",
        nbins=20,
        title="⭐ Amazon — Rating Distribution",
        template="plotly_white",
        color_discrete_sequence=["#f59e0b"],
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
        xaxis_title="Rating",
        yaxis_title="Number of Products",
    )
    fig.write_html(os.path.join(PLOTS_DIR, "amazon_rating_distribution.html"))

    # --- 4. Discount % vs Rating ---
    print("   [4/7] Discount vs Rating Analysis...")
    scatter_df = df.dropna(subset=["discount_percentage", "rating"]).copy()
    scatter_df["rating_count"] = scatter_df["rating_count"].fillna(1)
    fig = px.scatter(
        scatter_df,
        x="discount_percentage",
        y="rating",
        color="main_category",
        size="rating_count",
        size_max=25,
        title="🏷️ Amazon — Discount % vs Product Rating",
        template="plotly_white",
        opacity=0.6,
        hover_data=["product_name"],
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "amazon_discount_vs_rating.html"))

    # --- 5. Top 10 Most Reviewed Products ---
    print("   [5/7] Top Reviewed Products...")
    top_reviewed = (
        df.nlargest(10, "rating_count")[["product_name", "rating_count", "rating", "main_category"]]
        .copy()
    )
    top_reviewed["product_label"] = top_reviewed["product_name"].str[:40] + "..."
    fig = px.bar(
        top_reviewed,
        x="rating_count",
        y="product_label",
        orientation="h",
        color="rating",
        title="🔥 Top 10 Most Reviewed Products (Amazon)",
        template="plotly_white",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
    )
    fig.write_html(os.path.join(PLOTS_DIR, "amazon_top_reviewed.html"))

    # --- 6. Price vs Rating by Category ---
    print("   [6/7] Price vs Rating by Category...")
    fig = px.scatter(
        df.dropna(subset=["discounted_price", "rating"]),
        x="discounted_price",
        y="rating",
        color="main_category",
        title="💰 Amazon — Price vs Rating by Category",
        template="plotly_white",
        opacity=0.5,
        log_x=True,
        labels={"discounted_price": "Discounted Price (₹, log scale)", "rating": "Rating"},
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "amazon_price_vs_rating.html"))

    # --- 7. Discount Distribution by Category ---
    print("   [7/7] Discount Distribution by Category...")
    fig_mpl, ax = plt.subplots(figsize=(12, 6))
    top_cats = df["main_category"].value_counts().head(8).index
    box_data = df[df["main_category"].isin(top_cats)]
    sns.violinplot(
        data=box_data,
        x="main_category",
        y="discount_percentage",
        palette="Set2",
        ax=ax,
        inner="quartile",
    )
    ax.set_title("Discount % Distribution by Category", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("")
    ax.set_ylabel("Discount %")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "amazon_discount_by_category.png"), dpi=150)
    plt.close()

    print("   ✅ Amazon EDA complete — 7 charts saved!")


# ===================================================================
#  COMBINED INSIGHTS
# ===================================================================
def combined_insights(amazon_df: pd.DataFrame, flipkart_df: pd.DataFrame) -> None:
    """Generate cross-platform comparison insights."""
    print("\n📊 Generating Combined Insights...")

    # --- Platform Comparison: Average Rating ---
    amazon_avg_rating = amazon_df["rating"].mean()
    flipkart_avg_rating = flipkart_df["customer_rating"].mean()

    comparison_data = pd.DataFrame(
        {
            "Platform": ["Amazon", "Flipkart"],
            "Avg Rating": [amazon_avg_rating, flipkart_avg_rating],
            "Total Products/Orders": [len(amazon_df), len(flipkart_df)],
        }
    )

    fig = px.bar(
        comparison_data,
        x="Platform",
        y="Avg Rating",
        color="Platform",
        title="⭐ Platform Comparison — Average Rating",
        template="plotly_white",
        color_discrete_sequence=["#6366f1", "#f59e0b"],
        text="Avg Rating",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
        showlegend=False,
    )
    fig.write_html(os.path.join(PLOTS_DIR, "platform_comparison_rating.html"))

    # --- Category overlap analysis ---
    amazon_cats = set(amazon_df["main_category"].unique())
    flipkart_cats = set(flipkart_df["category"].unique())
    common = amazon_cats & flipkart_cats

    print(f"   Amazon unique categories : {len(amazon_cats)}")
    print(f"   Flipkart unique categories: {len(flipkart_cats)}")
    print(f"   Common categories          : {len(common)}")

    print("   ✅ Combined insights saved!")


# ===================================================================
#  MAIN
# ===================================================================
def main():
    """Run full EDA pipeline."""
    print("\n" + "=" * 60)
    print("  📊 E-Commerce Exploratory Data Analysis")
    print("=" * 60)

    # Load cleaned data
    amazon_df = pd.read_csv(os.path.join(DATA_DIR, "amazon_cleaned.csv"))
    flipkart_df = pd.read_csv(os.path.join(DATA_DIR, "flipkart_cleaned.csv"))

    print(f"  Amazon  : {amazon_df.shape[0]:,} rows loaded")
    print(f"  Flipkart: {flipkart_df.shape[0]:,} rows loaded")

    # Run EDA
    flipkart_eda(flipkart_df)
    amazon_eda(amazon_df)
    combined_insights(amazon_df, flipkart_df)

    print("\n" + "=" * 60)
    print("  ✅ All EDA visualizations generated successfully!")
    print(f"  📁 Plots saved to: {PLOTS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
