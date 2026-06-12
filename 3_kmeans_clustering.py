"""
3_kmeans_clustering.py
-----------------------
Groups stocks by risk level using the K-Means algorithm:
  - Cluster 0 → Low Risk
  - Cluster 1 → Medium Risk
  - Cluster 2 → High Risk

Input  : data/metrics.csv  (produced by 2_analysis.py)
Output : data/clustered.csv  +  data/clusters_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FEATURES     = ["Ann. Return (%)", "Ann. Volatility (%)", "Max Drawdown (%)"]
N_CLUSTERS   = 3
RANDOM_STATE = 42


def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map K-Means cluster numbers to meaningful risk labels.
    Sorted by average volatility per cluster:
      lowest volatility  → 'Low Risk'
      middle volatility  → 'Medium Risk'
      highest volatility → 'High Risk'
    """
    centers = df.groupby("Cluster")["Ann. Volatility (%)"].mean().sort_values()
    label_map = {
        centers.index[0]: "Low Risk",
        centers.index[1]: "Medium Risk",
        centers.index[2]: "High Risk",
    }
    df["Risk Level"] = df["Cluster"].map(label_map)
    return df


def run_kmeans(metrics: pd.DataFrame):
    X = metrics[FEATURES].copy()

    # Standardize features (K-Means is distance-sensitive)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    metrics["Cluster"] = km.fit_predict(X_scaled)
    metrics = assign_risk_labels(metrics)
    return metrics, km, scaler


def plot_clusters(metrics: pd.DataFrame):
    color_map = {
        "Low Risk":    "#2ecc71",
        "Medium Risk": "#f39c12",
        "High Risk":   "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for risk, group in metrics.groupby("Risk Level"):
        ax.scatter(
            group["Ann. Volatility (%)"],
            group["Ann. Return (%)"],
            color=color_map[risk],
            s=180,
            label=risk,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row.name,
                (row["Ann. Volatility (%)"], row["Ann. Return (%)"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
            )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Annual Volatility (%)", fontsize=11)
    ax.set_ylabel("Annual Average Return (%)", fontsize=11)
    ax.set_title("K-Means Clustering: Stock Risk Groups", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("data/clusters_plot.png", dpi=150)
    plt.show()
    print("  Chart saved: data/clusters_plot.png")


if __name__ == "__main__":
    print("=" * 55)
    print("  K-Means Clustering (Risk Grouping)")
    print("=" * 55)

    metrics = pd.read_csv("data/metrics.csv", index_col=0)
    metrics, km, scaler = run_kmeans(metrics)

    print("\n--- Risk Groups ---")
    result = metrics[["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", "Risk Level"]]
    print(result.to_string())

    metrics.to_csv("data/clustered.csv")
    print("\nResults saved: data/clustered.csv")

    print("\nGenerating chart...")
    plot_clusters(metrics)

    print("\nClustering complete.")
