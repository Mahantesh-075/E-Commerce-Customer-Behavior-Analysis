"""
rfm.py
------
RFM (Recency, Frequency, Monetary) Analysis on Flipkart transactional data.

Calculates RFM scores per customer, assigns behavioral segments
(Champions, Loyal, Potential Loyalists, At-Risk, Lost), and generates
visualization charts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RFM scores for each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Flipkart transactional data with customer_id, order_date,
        order_id, and total_sales columns.

    Returns
    -------
    pd.DataFrame
        RFM table with scores and segments assigned.
    """
    print("\n👥 Calculating RFM Scores...")

    # Ensure date is parsed
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Snapshot date — 1 day after the last order
    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)
    print(f"   Snapshot date: {snapshot_date.date()}")

    # Aggregate per customer
    rfm = (
        df.groupby("customer_id")
        .agg(
            Recency=("order_date", lambda x: (snapshot_date - x.max()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("total_sales", "sum"),
        )
        .reset_index()
    )

    print(f"   Unique customers: {len(rfm)}")
    print(f"   Recency  range: {rfm['Recency'].min()} – {rfm['Recency'].max()} days")
    print(f"   Frequency range: {rfm['Frequency'].min()} – {rfm['Frequency'].max()} orders")
    print(f"   Monetary  range: ₹{rfm['Monetary'].min():,.0f} – ₹{rfm['Monetary'].max():,.0f}")

    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign R, F, M scores (1-5) using quintile binning.
    Higher scores = better customer.
    """
    print("\n   Assigning RFM scores (1–5)...")

    # Recency: lower is better → invert labels
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    )

    # Frequency: higher is better
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    )

    # Monetary: higher is better
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    )

    # Combined scores
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Total"] = (
        rfm[["R_Score", "F_Score", "M_Score"]].astype(int).sum(axis=1)
    )

    print(f"   RFM_Total range: {rfm['RFM_Total'].min()} – {rfm['RFM_Total'].max()}")
    return rfm


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign customer segments based on RFM_Total score.

    Segments:
      - Champions       : 13–15
      - Loyal Customers  : 10–12
      - Potential Loyalists : 7–9
      - At-Risk          : 5–6
      - Lost Customers   : 3–4
    """
    print("\n   Assigning customer segments...")

    def _segment(score: int) -> str:
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal Customers"
        elif score >= 7:
            return "Potential Loyalists"
        elif score >= 5:
            return "At-Risk"
        else:
            return "Lost Customers"

    rfm["Segment"] = rfm["RFM_Total"].apply(_segment)

    seg_counts = rfm["Segment"].value_counts()
    print("\n   Segment distribution:")
    for seg, count in seg_counts.items():
        pct = count / len(rfm) * 100
        print(f"     {seg:25s}: {count:4d} ({pct:5.1f}%)")

    return rfm


def visualize_segments(rfm: pd.DataFrame) -> None:
    """Generate RFM segment visualizations."""
    print("\n📊 Generating RFM visualizations...")

    # --- 1. Segment Count Bar Chart ---
    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    color_map = {
        "Champions": "#10b981",
        "Loyal Customers": "#6366f1",
        "Potential Loyalists": "#f59e0b",
        "At-Risk": "#f97316",
        "Lost Customers": "#ef4444",
    }

    fig = px.bar(
        segment_counts,
        x="Segment",
        y="Count",
        color="Segment",
        title="👥 Customer RFM Segments Distribution",
        template="plotly_white",
        color_discrete_map=color_map,
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
        showlegend=False,
    )
    fig.write_html(os.path.join(PLOTS_DIR, "rfm_segments_bar.html"))

    # --- 2. Revenue Per Segment (Pie) ---
    segment_monetary = (
        rfm.groupby("Segment")["Monetary"].mean().reset_index()
    )
    fig = px.pie(
        segment_monetary,
        names="Segment",
        values="Monetary",
        title="💰 Average Revenue per Customer Segment",
        hole=0.35,
        template="plotly_white",
        color="Segment",
        color_discrete_map=color_map,
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "rfm_monetary_pie.html"))

    # --- 3. RFM Score Distribution (Heatmap-style) ---
    fig_mpl, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (col, label, color) in enumerate(
        [
            ("Recency", "Recency (days)", "#6366f1"),
            ("Frequency", "Frequency (orders)", "#10b981"),
            ("Monetary", "Monetary (₹)", "#f59e0b"),
        ]
    ):
        axes[idx].hist(
            rfm[col], bins=25, color=color, edgecolor="white", alpha=0.85
        )
        axes[idx].set_title(f"{label} Distribution", fontsize=13, fontweight="bold")
        axes[idx].set_xlabel(label)
        axes[idx].set_ylabel("Count")
        axes[idx].axvline(
            rfm[col].median(), color="red", linestyle="--", label=f"Median: {rfm[col].median():.0f}"
        )
        axes[idx].legend()

    plt.suptitle("RFM Score Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rfm_distributions.png"), dpi=150)
    plt.close()

    # --- 4. Segment Treemap ---
    seg_summary = (
        rfm.groupby("Segment")
        .agg(
            Count=("customer_id", "count"),
            Avg_Monetary=("Monetary", "mean"),
            Avg_Frequency=("Frequency", "mean"),
        )
        .reset_index()
    )
    fig = px.treemap(
        seg_summary,
        path=["Segment"],
        values="Count",
        color="Avg_Monetary",
        title="🗺️ Customer Segments — Treemap (Size=Count, Color=Avg Revenue)",
        color_continuous_scale="RdYlGn",
        template="plotly_white",
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "rfm_treemap.html"))

    print("   ✅ RFM visualizations saved (4 charts)!")


# ===================================================================
#  MAIN
# ===================================================================
def main():
    """Run the full RFM analysis pipeline."""
    print("\n" + "=" * 60)
    print("  👥 RFM Analysis Pipeline")
    print("=" * 60)

    # Load cleaned Flipkart data
    flipkart_df = pd.read_csv(
        os.path.join(DATA_DIR, "flipkart_cleaned.csv"),
        parse_dates=["order_date"],
    )
    print(f"  Loaded {len(flipkart_df):,} orders")

    # RFM pipeline
    rfm = calculate_rfm(flipkart_df)
    rfm = score_rfm(rfm)
    rfm = assign_segments(rfm)

    # Visualize
    visualize_segments(rfm)

    # Save
    output_path = os.path.join(REPORTS_DIR, "rfm_segments.csv")
    rfm.to_csv(output_path, index=False)
    print(f"\n  ✅ RFM segments saved to: {output_path}")
    print("=" * 60 + "\n")

    return rfm


if __name__ == "__main__":
    main()
