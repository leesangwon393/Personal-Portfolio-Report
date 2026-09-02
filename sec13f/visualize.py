"""Section 9 figures. Matplotlib only (no seaborn — not in requirements.txt
and every plot here is simple enough not to need it), saved as PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sec13f.clustering import FEATURE_COLUMNS

FEATURE_LABELS = {
    "volatility": "Annualized Volatility",
    "mdd": "Maximum Drawdown (abs)",
    "beta": "Beta vs SPY",
    "sector_hhi": "Sector HHI",
}
_CLUSTER_COLORS = ["#2b6cb0", "#38a169", "#c53030", "#805ad5", "#dd6b20", "#319795"]


def _savefig(fig, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), FEATURE_COLUMNS):
        ax.hist(df[col].dropna(), bins=25, color="#4a5568", edgecolor="white")
        ax.set_title(FEATURE_LABELS[col])
        ax.set_xlabel(col)
        ax.set_ylabel("Number of portfolios")
    fig.suptitle("13F Portfolio Risk Metric Distributions (100 institutional filers)", fontsize=13)
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels([FEATURE_LABELS[c] for c in FEATURE_COLUMNS], rotation=30, ha="right")
    ax.set_yticklabels([FEATURE_LABELS[c] for c in FEATURE_COLUMNS])
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation Between Risk Metrics")
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_silhouette_by_k(evaluation_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in evaluation_df.groupby("method"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["silhouette"], marker="o", label=method.upper())
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by K (KMeans vs. Gaussian Mixture)")
    ax.axvline(3, color="gray", linestyle="--", linewidth=1, label="K=3 (target)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_pca_scatter(
    X_scaled: np.ndarray, labels: np.ndarray, cluster_names: dict[int, str], out_path: Path,
    seed: int = 42,
) -> None:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cid in enumerate(sorted(set(labels))):
        mask = labels == cid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=f"{cluster_names.get(cid, cid)} (n={mask.sum()})",
            color=_CLUSTER_COLORS[i % len(_CLUSTER_COLORS)], alpha=0.75, edgecolor="white", s=60,
        )
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("13F Portfolios in PCA Space, Colored by Risk Cluster")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_path)
    return pca, coords


def plot_centroid_heatmap(centroids_scaled: pd.DataFrame, cluster_names: dict[int, str], out_path: Path) -> None:
    labels = [cluster_names.get(cid, str(cid)) for cid in centroids_scaled.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(centroids_scaled[FEATURE_COLUMNS].values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels([FEATURE_LABELS[c] for c in FEATURE_COLUMNS], rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(FEATURE_COLUMNS)):
            ax.text(j, i, f"{centroids_scaled[FEATURE_COLUMNS].values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Scaled centroid value (higher = riskier)")
    ax.set_title("Cluster Centroids (Scaled Feature Space)")
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_cluster_sizes(labels: np.ndarray, cluster_names: dict[int, str], out_path: Path) -> None:
    counts = pd.Series(labels).value_counts().sort_index()
    names = [cluster_names.get(cid, str(cid)) for cid in counts.index]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, counts.values, color=_CLUSTER_COLORS[: len(counts)])
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(count),
                ha="center", va="bottom")
    ax.set_ylabel("Number of portfolios")
    ax.set_title("Portfolio Count per Risk Cluster")
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_representative_investor_positions(
    coords: np.ndarray, df: pd.DataFrame, representative_mask: pd.Series,
    labels: np.ndarray, cluster_names: dict[int, str], out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cid in enumerate(sorted(set(labels))):
        mask = (labels == cid) & (~representative_mask.values)
        ax.scatter(coords[mask, 0], coords[mask, 1], color=_CLUSTER_COLORS[i % len(_CLUSTER_COLORS)],
                   alpha=0.35, s=40)
    rep_idx = np.where(representative_mask.values)[0]
    for idx in rep_idx:
        cid = labels[idx]
        color_i = sorted(set(labels)).index(cid)
        ax.scatter(coords[idx, 0], coords[idx, 1], color=_CLUSTER_COLORS[color_i % len(_CLUSTER_COLORS)],
                   s=220, edgecolor="black", linewidth=1.5, marker="*", zorder=5)
        ax.annotate(
            df.iloc[idx]["manager_name"], (coords[idx, 0], coords[idx, 1]),
            textcoords="offset points", xytext=(8, 6), fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Representative Investors (Sanity Check) in PCA Space")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_cluster_boxplots(df: pd.DataFrame, labels: np.ndarray, cluster_names: dict[int, str], out_path: Path) -> None:
    plot_df = df[FEATURE_COLUMNS].copy()
    plot_df["cluster"] = [cluster_names.get(c, str(c)) for c in labels]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    order = [cluster_names.get(cid, str(cid)) for cid in sorted(set(labels))]
    for ax, col in zip(axes, FEATURE_COLUMNS):
        data = [plot_df.loc[plot_df["cluster"] == name, col].values for name in order]
        ax.boxplot(data, tick_labels=order)
        ax.set_title(FEATURE_LABELS[col])
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Risk Metric Distributions by Cluster (Original Units)", fontsize=13)
    fig.tight_layout()
    _savefig(fig, out_path)
