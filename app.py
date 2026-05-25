"""
app.py
------
Streamlit Interactive Dashboard for E-Commerce Customer Behavior Analysis.

Features:
  - Sidebar filters: Platform, Category, Year, Region
  - KPI Cards: Total Revenue, Orders, Customers, AOV
  - Monthly Sales Trend
  - Top Categories by Revenue
  - Customer Segments (RFM) Pie Chart
  - 3D Customer Clusters
  - Payment Method Breakdown
  - Amazon Ratings Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main > div {
        padding: 1rem 2rem;
    }

    /* KPI Card Styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 500 !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: white !important;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 700 !important;
    }

    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.2);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")


@st.cache_data
def load_data():
    """Load all cleaned datasets."""
    amazon = pd.read_csv(os.path.join(DATA_DIR, "amazon_cleaned.csv"))
    flipkart = pd.read_csv(
        os.path.join(DATA_DIR, "flipkart_cleaned.csv"),
        parse_dates=["order_date"],
    )

    rfm_path = os.path.join(REPORTS_DIR, "rfm_clustered.csv")
    rfm = pd.read_csv(rfm_path) if os.path.exists(rfm_path) else None

    model_path = os.path.join(REPORTS_DIR, "model_results.csv")
    model_results = pd.read_csv(model_path) if os.path.exists(model_path) else None

    return amazon, flipkart, rfm, model_results


amazon_df, flipkart_df, rfm_df, model_results = load_data()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🛒 E-Commerce Customer Behavior Dashboard")
st.markdown(
    "**Amazon + Flipkart Sales Analysis** &nbsp;|&nbsp; "
    "Interactive Data Exploration &nbsp;|&nbsp; Campus Placement Portfolio Project"
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🔎 Filters")
st.sidebar.markdown("---")

# Platform filter
platform = st.sidebar.selectbox("📱 Platform", ["Both", "Amazon", "Flipkart"])

# Category filter (Flipkart)
all_categories = sorted(flipkart_df["category"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "📦 Category (Flipkart)",
    all_categories,
    default=all_categories[:5] if len(all_categories) >= 5 else all_categories,
)

# Year filter
available_years = sorted(flipkart_df["order_year"].dropna().unique())
selected_year = st.sidebar.selectbox("📅 Year", available_years)

# Region filter
all_regions = sorted(flipkart_df["region"].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "🌍 Region", all_regions, default=all_regions
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with ❤️ using **Python + Streamlit**\n\n"
    "*Campus Placement Portfolio Project*"
)

# ---------------------------------------------------------------------------
# Apply Filters
# ---------------------------------------------------------------------------
filtered_fk = flipkart_df[
    (flipkart_df["category"].isin(selected_categories))
    & (flipkart_df["order_year"] == selected_year)
    & (flipkart_df["region"].isin(selected_regions))
]

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Revenue",
    f"₹{filtered_fk['total_sales'].sum():,.0f}",
)
col2.metric(
    "📦 Total Orders",
    f"{filtered_fk['order_id'].nunique():,}",
)
col3.metric(
    "👥 Unique Customers",
    f"{filtered_fk['customer_id'].nunique():,}",
)
avg_order_val = (
    filtered_fk["total_sales"].mean() if len(filtered_fk) > 0 else 0
)
col4.metric(
    "📈 Avg Order Value",
    f"₹{avg_order_val:,.0f}",
)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Sales Analytics", "👥 Customer Segments", "🤖 ML Predictions", "📦 Amazon Insights"]
)

# ================================
# TAB 1 — Sales Analytics
# ================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Monthly Sales Trend")
        if len(filtered_fk) > 0:
            monthly = (
                filtered_fk.groupby(
                    filtered_fk["order_date"].dt.to_period("M")
                )["total_sales"]
                .sum()
                .reset_index()
            )
            monthly["order_date"] = monthly["order_date"].astype(str)
            fig = px.area(
                monthly,
                x="order_date",
                y="total_sales",
                title="Monthly Revenue",
                template="plotly_white",
                color_discrete_sequence=["#6366f1"],
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue (₹)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected filters.")

    with col2:
        st.markdown("#### 🏆 Top Categories by Revenue")
        top_cat = (
            filtered_fk.groupby("category")["total_sales"]
            .sum()
            .nlargest(8)
            .reset_index()
        )
        fig = px.bar(
            top_cat,
            x="total_sales",
            y="category",
            orientation="h",
            color="total_sales",
            color_continuous_scale="Purples",
            template="plotly_white",
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌍 Regional Sales Distribution")
        region_data = (
            filtered_fk.groupby("region")["total_sales"].sum().reset_index()
        )
        fig = px.pie(
            region_data,
            names="region",
            values="total_sales",
            hole=0.4,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 💳 Payment Method Breakdown")
        pay_data = filtered_fk["payment_method"].value_counts().reset_index()
        pay_data.columns = ["method", "count"]
        fig = px.pie(
            pay_data,
            names="method",
            values="count",
            hole=0.35,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

# ================================
# TAB 2 — Customer Segments
# ================================
with tab2:
    if rfm_df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👥 RFM Customer Segments")
            seg_counts = rfm_df["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            color_map = {
                "Champions": "#10b981",
                "Loyal Customers": "#6366f1",
                "Potential Loyalists": "#f59e0b",
                "At-Risk": "#f97316",
                "Lost Customers": "#ef4444",
            }
            fig = px.pie(
                seg_counts,
                names="Segment",
                values="Count",
                hole=0.35,
                color="Segment",
                color_discrete_map=color_map,
                template="plotly_white",
            )
            fig.update_layout(font=dict(family="Inter"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🔵 3D Customer Clusters")
            fig = px.scatter_3d(
                rfm_df,
                x="Recency",
                y="Frequency",
                z="Monetary",
                color=rfm_df["Cluster"].astype(str),
                opacity=0.7,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(
                font=dict(family="Inter"),
                scene=dict(
                    xaxis_title="Recency",
                    yaxis_title="Frequency",
                    zaxis_title="Monetary",
                ),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Segment Summary Table
        st.markdown("#### 📋 Segment Summary")
        seg_summary = (
            rfm_df.groupby("Segment")
            .agg(
                Customers=("customer_id", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
            )
            .round(2)
            .sort_values("Avg_Monetary", ascending=False)
        )
        st.dataframe(seg_summary, use_container_width=True)
    else:
        st.warning("⚠️ RFM data not found. Please run the RFM analysis first.")

# ================================
# TAB 3 — ML Predictions
# ================================
with tab3:
    if model_results is not None:
        st.markdown("#### 📊 Model Performance Comparison")

        col1, col2, col3 = st.columns(3)
        best_model = model_results.loc[model_results["ROC_AUC"].idxmax()]
        col1.metric("🏆 Best Model", best_model["Model"])
        col2.metric("🎯 Accuracy", f"{best_model['Accuracy']:.1%}")
        col3.metric("📈 ROC-AUC", f"{best_model['ROC_AUC']:.4f}")

        fig = px.bar(
            model_results.melt(
                id_vars="Model",
                value_vars=["Accuracy", "F1_Score", "ROC_AUC"],
                var_name="Metric",
                value_name="Score",
            ),
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            template="plotly_white",
            color_discrete_sequence=["#6366f1", "#10b981"],
            text="Score",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            font=dict(family="Inter"),
            yaxis_range=[0, 1.15],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display model evaluation image if exists
        eval_img = os.path.join(BASE_DIR, "outputs", "plots", "model_evaluation.png")
        if os.path.exists(eval_img):
            st.markdown("#### 📉 Confusion Matrices & ROC Curves")
            st.image(eval_img, use_container_width=True)

        # Feature importance
        fi_path = os.path.join(BASE_DIR, "outputs", "plots", "feature_importance.html")
        if os.path.exists(fi_path):
            st.markdown("#### 🔍 Feature Importance")
            import plotly.io as pio
            fig = pio.read_json(fi_path.replace(".html", ".json")) if os.path.exists(
                fi_path.replace(".html", ".json")
            ) else None
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance chart saved as HTML in outputs/plots/")
    else:
        st.warning("⚠️ Model results not found. Please run the prediction model first.")

# ================================
# TAB 4 — Amazon Insights
# ================================
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⭐ Rating Distribution")
        fig = px.histogram(
            amazon_df.dropna(subset=["rating"]),
            x="rating",
            nbins=20,
            template="plotly_white",
            color_discrete_sequence=["#f59e0b"],
        )
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Count",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📦 Category Distribution")
        if "main_category" in amazon_df.columns:
            cat_col = "main_category"
        else:
            cat_col = "category"
        cat_counts = amazon_df[cat_col].value_counts().head(10).reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(
            cat_counts,
            x="Count",
            y="Category",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏷️ Discount % vs Rating")
        scatter_data = amazon_df.dropna(
            subset=["discount_percentage", "rating"]
        )
        if "main_category" in scatter_data.columns:
            color_col = "main_category"
        else:
            color_col = "category"
        fig = px.scatter(
            scatter_data,
            x="discount_percentage",
            y="rating",
            color=color_col,
            opacity=0.6,
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="Discount %",
            yaxis_title="Rating",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 💰 Price Distribution by Category")
        if "main_category" in amazon_df.columns:
            cat_col = "main_category"
        else:
            cat_col = "category"
        top_cats = amazon_df[cat_col].value_counts().head(6).index
        box_data = amazon_df[amazon_df[cat_col].isin(top_cats)]
        fig = px.box(
            box_data,
            x=cat_col,
            y="discounted_price",
            color=cat_col,
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Price (₹)",
            font=dict(family="Inter"),
            showlegend=False,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with ❤️ using Python, Pandas, Scikit-learn, Plotly & Streamlit | "
    "E-Commerce Customer Behavior Analysis | Campus Placement Portfolio Project"
)
