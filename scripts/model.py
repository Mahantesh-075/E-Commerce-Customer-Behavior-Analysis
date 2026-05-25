"""
model.py
--------
Prediction Model for identifying high-value customers.

1. Loads clustered RFM data
2. Creates binary target (high-value vs. others at 75th percentile)
3. Trains Random Forest and Logistic Regression models
4. Evaluates with classification report, confusion matrix, ROC-AUC
5. Generates feature importance chart
6. Saves all evaluation plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
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


def prepare_data(rfm: pd.DataFrame):
    """
    Create target variable and prepare train/test split.

    Target: is_high_value = 1 if Monetary >= 75th percentile, else 0
    """
    print("\n📦 Preparing data for modeling...")

    # Create target
    threshold = rfm["Monetary"].quantile(0.75)
    rfm["is_high_value"] = (rfm["Monetary"] >= threshold).astype(int)

    print(f"   Monetary threshold (75th pct): ₹{threshold:,.0f}")
    print(f"   Class distribution:")
    print(f"     High-value (1): {rfm['is_high_value'].sum()} ({rfm['is_high_value'].mean()*100:.1f}%)")
    print(f"     Standard   (0): {(rfm['is_high_value']==0).sum()} ({(1-rfm['is_high_value'].mean())*100:.1f}%)")

    # Features
    feature_cols = ["Recency", "Frequency", "Monetary", "RFM_Total", "Cluster"]
    feature_cols = [c for c in feature_cols if c in rfm.columns]
    X = rfm[feature_cols].copy()
    y = rfm["is_high_value"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Features: {feature_cols}")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(X_train, X_test, y_train, y_test, feature_cols):
    """Train and evaluate Random Forest Classifier."""
    print("\n🌲 Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        max_depth=10,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n   📊 Random Forest Results:")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   ROC-AUC   : {roc_auc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Standard", "High-Value"]))

    return rf, y_pred, y_prob, {"Accuracy": acc, "F1": f1, "ROC-AUC": roc_auc}


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression (for comparison)."""
    print("\n📈 Training Logistic Regression...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(
        random_state=42, class_weight="balanced", max_iter=1000
    )
    lr.fit(X_train_s, y_train)

    y_pred = lr.predict(X_test_s)
    y_prob = lr.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n   📊 Logistic Regression Results:")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   ROC-AUC   : {roc_auc:.4f}")

    return lr, y_pred, y_prob, {"Accuracy": acc, "F1": f1, "ROC-AUC": roc_auc}


def plot_evaluation(
    y_test, rf_pred, rf_prob, lr_pred, lr_prob, rf_metrics, lr_metrics, feature_cols, rf_model
):
    """Generate model evaluation plots."""
    print("\n📊 Generating evaluation plots...")

    # --- 1. Confusion Matrices + ROC Curves ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # RF Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, rf_pred, ax=axes[0, 0], cmap="Blues", colorbar=False,
        display_labels=["Standard", "High-Value"],
    )
    axes[0, 0].set_title(
        f"Random Forest — Confusion Matrix\n(Acc={rf_metrics['Accuracy']:.3f})",
        fontsize=13, fontweight="bold",
    )

    # LR Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, lr_pred, ax=axes[0, 1], cmap="Greens", colorbar=False,
        display_labels=["Standard", "High-Value"],
    )
    axes[0, 1].set_title(
        f"Logistic Regression — Confusion Matrix\n(Acc={lr_metrics['Accuracy']:.3f})",
        fontsize=13, fontweight="bold",
    )

    # RF ROC Curve
    fpr, tpr, _ = roc_curve(y_test, rf_prob)
    axes[1, 0].plot(fpr, tpr, "b-", linewidth=2.5,
                    label=f"RF AUC = {rf_metrics['ROC-AUC']:.3f}")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1, 0].fill_between(fpr, tpr, alpha=0.15, color="blue")
    axes[1, 0].set_title("ROC Curve — Random Forest", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # LR ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    axes[1, 1].plot(fpr_lr, tpr_lr, "g-", linewidth=2.5,
                    label=f"LR AUC = {lr_metrics['ROC-AUC']:.3f}")
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1, 1].fill_between(fpr_lr, tpr_lr, alpha=0.15, color="green")
    axes[1, 1].set_title("ROC Curve — Logistic Regression", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("False Positive Rate")
    axes[1, 1].set_ylabel("True Positive Rate")
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_evaluation.png"), dpi=150)
    plt.close()

    # --- 2. Feature Importance (Random Forest) ---
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="🔍 Feature Importance — Random Forest",
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_white",
        text="Importance",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(font=dict(family="Inter, sans-serif"), title_font_size=18)
    fig.write_html(os.path.join(PLOTS_DIR, "feature_importance.html"))

    # --- 3. Model Comparison Bar Chart ---
    comparison = pd.DataFrame({
        "Model": ["Random Forest", "Random Forest", "Random Forest",
                   "Logistic Regression", "Logistic Regression", "Logistic Regression"],
        "Metric": ["Accuracy", "F1-Score", "ROC-AUC"] * 2,
        "Value": [rf_metrics["Accuracy"], rf_metrics["F1"], rf_metrics["ROC-AUC"],
                  lr_metrics["Accuracy"], lr_metrics["F1"], lr_metrics["ROC-AUC"]],
    })
    fig = px.bar(
        comparison,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="📊 Model Comparison — RF vs Logistic Regression",
        template="plotly_white",
        color_discrete_sequence=["#6366f1", "#10b981"],
        text="Value",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
        yaxis_range=[0, 1.15],
    )
    fig.write_html(os.path.join(PLOTS_DIR, "model_comparison.html"))

    print("   ✅ Evaluation plots saved (3 charts)!")


# ===================================================================
#  MAIN
# ===================================================================
def main():
    """Run the full prediction pipeline."""
    print("\n" + "=" * 60)
    print("  🧠 Prediction Model Pipeline")
    print("=" * 60)

    # Load clustered RFM data
    rfm = pd.read_csv(os.path.join(REPORTS_DIR, "rfm_clustered.csv"))
    print(f"  Loaded {len(rfm)} customers")

    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(rfm)

    # Train models
    rf_model, rf_pred, rf_prob, rf_metrics = train_random_forest(
        X_train, X_test, y_train, y_test, feature_cols
    )
    lr_model, lr_pred, lr_prob, lr_metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    # Evaluate & plot
    plot_evaluation(
        y_test, rf_pred, rf_prob, lr_pred, lr_prob,
        rf_metrics, lr_metrics, feature_cols, rf_model,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  📊 Final Model Summary")
    print("=" * 60)
    print(f"  Random Forest    : Acc={rf_metrics['Accuracy']:.4f}, F1={rf_metrics['F1']:.4f}, AUC={rf_metrics['ROC-AUC']:.4f}")
    print(f"  Logistic Reg.    : Acc={lr_metrics['Accuracy']:.4f}, F1={lr_metrics['F1']:.4f}, AUC={lr_metrics['ROC-AUC']:.4f}")

    best = "Random Forest" if rf_metrics["ROC-AUC"] >= lr_metrics["ROC-AUC"] else "Logistic Regression"
    print(f"\n  🏆 Best Model: {best}")

    # Save results
    results = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "Accuracy": [rf_metrics["Accuracy"], lr_metrics["Accuracy"]],
        "F1_Score": [rf_metrics["F1"], lr_metrics["F1"]],
        "ROC_AUC": [rf_metrics["ROC-AUC"], lr_metrics["ROC-AUC"]],
    })
    results.to_csv(os.path.join(REPORTS_DIR, "model_results.csv"), index=False)
    print(f"  ✅ Results saved to: {os.path.join(REPORTS_DIR, 'model_results.csv')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
