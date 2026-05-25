"""
clustering.py
-------------
KMeans Customer Clustering on RFM scores.

1. Loads RFM segment data
2. Scales features (StandardScaler)
3. Determines optimal K via Elbow + Silhouette methods
4. Applies KMeans with optimal K
5. Generates 3D interactive cluster visualization
6. Saves clustered data
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def find_optimal_k(rfm_scaled: np.ndarray, k_range: range = range(2, 10)) -> int:
    """
    Determine optimal K using Elbow Method and Silhouette Score.

    Returns the K with the highest silhouette score.
    """
    print("\n🔍 Finding optimal K...")
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(rfm_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(rfm_scaled, km.labels_)
        silhouettes.append(sil)
        print(f"   K={k}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.4f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(list(k_range), inertias, "o-", color="#6366f1", linewidth=2.5, markersize=8)
    axes[0].set_title("Elbow Method — Inertia vs K", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[0].set_ylabel("Inertia", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "o-", color="#10b981", linewidth=2.5, markersize=8)
    axes[1].set_title("Silhouette Score vs K", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Highlight optimal
    optimal_k = list(k_range)[np.argmax(silhouettes)]
    axes[1].axvline(optimal_k, color="red", linestyle="--", alpha=0.7)
    axes[1].annotate(
        f"Optimal K={optimal_k}",
        xy=(optimal_k, max(silhouettes)),
        fontsize=12,
        fontweight="bold",
        color="red",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "elbow_silhouette.png"), dpi=150)
    plt.close()

    print(f"\n   ✅ Optimal K = {optimal_k} (Silhouette = {max(silhouettes):.4f})")
    return optimal_k


def apply_kmeans(rfm: pd.DataFrame, rfm_scaled: np.ndarray, k: int) -> pd.DataFrame:
    """Apply KMeans with the given K and add cluster labels."""
    print(f"\n🤖 Applying KMeans with K={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    # Cluster summary
    summary = (
        rfm.groupby("Cluster")
        .agg(
            Count=("customer_id", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .round(2)
    )
    print("\n   Cluster Summary:")
    print(summary.to_string())

    # Final silhouette
    sil = silhouette_score(rfm_scaled, rfm["Cluster"])
    print(f"\n   Final Silhouette Score: {sil:.4f}")

    return rfm


def visualize_clusters(rfm: pd.DataFrame) -> None:
    """Generate 3D and 2D cluster visualizations."""
    print("\n📊 Generating cluster visualizations...")

    # --- 1. 3D Scatter Plot ---
    fig = px.scatter_3d(
        rfm,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color=rfm["Cluster"].astype(str),
        title="🔵 3D Customer Clusters (RFM)",
        labels={"color": "Cluster"},
        template="plotly_white",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
        scene=dict(
            xaxis_title="Recency (days)",
            yaxis_title="Frequency (orders)",
            zaxis_title="Monetary (₹)",
        ),
    )
    fig.write_html(os.path.join(PLOTS_DIR, "3d_clusters.html"))

    # --- 2. 2D Pairwise Scatter ---
    fig = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color=rfm["Cluster"].astype(str),
        size="Recency",
        title="📌 Customer Clusters — Frequency vs Monetary",
        template="plotly_white",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "2d_clusters_freq_monetary.html"))

    # --- 3. Cluster Radar Chart ---
    cluster_means = (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )
    # Normalize for radar
    for col in ["Recency", "Frequency", "Monetary"]:
        max_val = cluster_means[col].max()
        if max_val > 0:
            cluster_means[col + "_norm"] = cluster_means[col] / max_val

    fig = px.line_polar(
        pd.melt(
            cluster_means,
            id_vars=["Cluster"],
            value_vars=["Recency_norm", "Frequency_norm", "Monetary_norm"],
            var_name="Metric",
            value_name="Value",
        ),
        r="Value",
        theta="Metric",
        color=cluster_means["Cluster"].astype(str).repeat(3).values,
        line_close=True,
        title="🎯 Cluster Profiles — Radar Chart",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "cluster_radar.html"))

    print("   ✅ Cluster visualizations saved (3 charts)!")


# ===================================================================
#  MAIN
# ===================================================================
def main():
    """Run the full clustering pipeline."""
    print("\n" + "=" * 60)
    print("  🤖 KMeans Clustering Pipeline")
    print("=" * 60)

    # Load RFM data
    rfm = pd.read_csv(os.path.join(REPORTS_DIR, "rfm_segments.csv"))
    print(f"  Loaded {len(rfm)} customers")

    # Prepare features
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(features)
    print("  Features scaled with StandardScaler")

    # Find optimal K
    optimal_k = find_optimal_k(rfm_scaled)

    # Apply KMeans
    rfm = apply_kmeans(rfm, rfm_scaled, optimal_k)

    # Visualize
    visualize_clusters(rfm)

    # Save
    output_path = os.path.join(REPORTS_DIR, "rfm_clustered.csv")
    rfm.to_csv(output_path, index=False)
    print(f"\n  ✅ Clustered data saved to: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
