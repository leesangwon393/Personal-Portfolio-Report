"""Preprocessing, K-means/GMM model comparison, and cluster naming
(Sections 6-7), plus classify_portfolio() for new portfolios (Section 11).

All four risk features are oriented so "higher = riskier" (Section 6) before
scaling, which is what makes `ClusterRisk_k` (the mean of the four scaled
centroid components) a meaningful low-to-high risk ordering to name
DEFENSIVE / BALANCED / AGGRESSIVE off of (Section 7).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from sec13f import config

FEATURE_COLUMNS = ["volatility", "mdd", "beta", "sector_hhi"]


# ──────────────────────────────────────────────────────────────
# Section 6: preprocessing
# ──────────────────────────────────────────────────────────────
def winsorize(
    df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS,
    lower: float = config.WINSOR_LOWER_QUANTILE, upper: float = config.WINSOR_UPPER_QUANTILE,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        lo, hi = out[col].quantile([lower, upper])
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def check_missing_and_infinite(df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Section 6: "결측치와 무한값 점검" — returns a small report DataFrame,
    does not mutate/drop anything (caller decides what to do with rows that
    have issues).
    """
    rows = []
    for col in columns:
        series = df[col]
        rows.append({
            "column": col,
            "n_missing": int(series.isna().sum()),
            "n_infinite": int(np.isinf(series.astype(float)).sum()),
            "min": float(series.min(skipna=True)) if series.notna().any() else None,
            "max": float(series.max(skipna=True)) if series.notna().any() else None,
        })
    return pd.DataFrame(rows)


def fit_scaler(df: pd.DataFrame, kind: str, columns: list[str] = FEATURE_COLUMNS):
    """kind: "standard" or "robust". Section 6: RobustScaler is the default
    candidate (outlier-resistant), but both are fit so their effect on
    clustering quality can be compared (Section 6/7).
    """
    scaler = RobustScaler() if kind == "robust" else StandardScaler()
    scaler.fit(df[columns].values)
    return scaler


def transform(scaler, df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> np.ndarray:
    return scaler.transform(df[columns].values)


# ──────────────────────────────────────────────────────────────
# Section 7: model comparison across K and (KMeans, GMM)
# ──────────────────────────────────────────────────────────────
def _fit_kmeans(X: np.ndarray, k: int, seed: int = config.RANDOM_SEED) -> KMeans:
    return KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=seed).fit(X)


def _fit_gmm(X: np.ndarray, k: int, seed: int = config.RANDOM_SEED) -> GaussianMixture:
    return GaussianMixture(
        n_components=k, n_init=config.GMM_N_INIT, random_state=seed, covariance_type="full",
    ).fit(X)


def cluster_stability(
    X: np.ndarray, k: int, method: str,
    n_bootstrap: int = config.CLUSTER_STABILITY_N_BOOTSTRAP,
    sample_frac: float = config.CLUSTER_STABILITY_SAMPLE_FRAC,
    seed: int = config.RANDOM_SEED,
) -> float:
    """Section 7: bootstrap-based cluster stability. Fits the reference
    model on all of X; then, for n_bootstrap iterations, fits a fresh model
    on a random `sample_frac` subsample and predicts labels for the FULL
    X — stability is the mean Adjusted Rand Index between each bootstrap
    run's full-X labels and the reference full-X labels. 1.0 = perfectly
    stable, ~0 = no better than random relabeling.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    fit_fn = _fit_kmeans if method == "kmeans" else _fit_gmm
    reference = fit_fn(X, k, seed=seed)
    reference_labels = reference.predict(X)

    n = X.shape[0]
    sample_size = max(k + 1, int(n * sample_frac))
    scores = []
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=False)
        try:
            model = fit_fn(X[idx], k, seed=seed + i + 1)
            labels = model.predict(X)
        except Exception:  # noqa: BLE001 - degenerate bootstrap sample; skip
            continue
        scores.append(adjusted_rand_score(reference_labels, labels))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_models(
    X: np.ndarray, k_range=config.KMEANS_K_RANGE, seed: int = config.RANDOM_SEED,
    compute_stability: bool = True,
) -> pd.DataFrame:
    """Section 7: compares KMeans vs GaussianMixture across k_range on
    Silhouette / Calinski-Harabasz / Davies-Bouldin + per-cluster sample
    counts + bootstrap stability. One row per (method, k).
    """
    rows = []
    for method, fit_fn in (("kmeans", _fit_kmeans), ("gmm", _fit_gmm)):
        for k in k_range:
            if k >= X.shape[0]:
                continue
            model = fit_fn(X, k, seed=seed)
            labels = model.predict(X)
            if len(set(labels)) < 2:
                continue
            sizes = pd.Series(labels).value_counts().sort_index().to_dict()
            row = {
                "method": method,
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "cluster_sizes": json.dumps(sizes),
                "min_cluster_size": min(sizes.values()),
            }
            if compute_stability:
                row["stability_ari"] = cluster_stability(X, k, method, seed=seed)
            rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Section 7: centroids, ClusterRisk, and DEFENSIVE/BALANCED/AGGRESSIVE naming
# ──────────────────────────────────────────────────────────────
def compute_centroids(
    model, scaler, X_scaled: np.ndarray, labels: np.ndarray, columns: list[str] = FEATURE_COLUMNS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (centroids_scaled, centroids_original_units) DataFrames
    indexed by cluster_id.
    """
    k = len(set(labels))
    scaled_rows = []
    for c in sorted(set(labels)):
        scaled_rows.append(X_scaled[labels == c].mean(axis=0))
    centroids_scaled = pd.DataFrame(scaled_rows, columns=columns, index=sorted(set(labels)))
    centroids_scaled.index.name = "cluster_id"

    original = scaler.inverse_transform(centroids_scaled.values)
    centroids_original = pd.DataFrame(original, columns=columns, index=centroids_scaled.index)
    centroids_original.index.name = "cluster_id"
    return centroids_scaled, centroids_original


def compute_cluster_risk(centroids_scaled: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> pd.Series:
    """Section 7: ClusterRisk_k = mean(Z_Vol, Z_MDD, Z_Beta, Z_HHI) using
    each cluster's centroid in the SCALED feature space (RobustScaler/
    StandardScaler output stands in for "Z" here — both center-and-scale
    each feature, so a scaled centroid mean is the same "how many typical
    spreads above/below center" idea the spec's Z_k formula describes).
    """
    return centroids_scaled[columns].mean(axis=1).rename("cluster_risk")


def assign_risk_profile_names(cluster_risk: pd.Series) -> dict[int, str]:
    """Section 7: lowest ClusterRisk -> DEFENSIVE, ..., highest -> AGGRESSIVE.
    This is a POST-HOC interpretation of unsupervised cluster centroids, not
    supervised label training — see report.md for the explicit statement
    this function exists to operationalize.
    """
    ordered = cluster_risk.sort_values().index.tolist()
    names = config.RISK_PROFILE_NAMES
    if len(ordered) != len(names):
        # Only k == len(RISK_PROFILE_NAMES) has a canonical 3-way name set;
        # otherwise fall back to a generic RISK_TIER_i naming (still ordered
        # low -> high) so the function never silently mislabels a
        # non-3-cluster solution as DEFENSIVE/BALANCED/AGGRESSIVE.
        return {cid: f"RISK_TIER_{rank + 1}_OF_{len(ordered)}" for rank, cid in enumerate(ordered)}
    return {cid: names[rank] for rank, cid in enumerate(ordered)}


# ──────────────────────────────────────────────────────────────
# Model persistence + Section 11: classify_portfolio()
# ──────────────────────────────────────────────────────────────
def save_model_bundle(
    path: Path, scaler, model, method: str, cluster_names: dict[int, str],
    columns: list[str] = FEATURE_COLUMNS,
) -> None:
    bundle = {
        "scaler": scaler,
        "model": model,
        "method": method,
        "cluster_names": cluster_names,
        "columns": columns,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)


def classify_portfolio(
    volatility: float, mdd: float, beta: float, sector_hhi: float,
    model_bundle: dict | None = None, model_path: Path | None = None,
) -> dict:
    """Section 11. Applies the SAME fitted scaler + clustering model used
    for the 100-portfolio training run to a new portfolio's four risk
    metrics.

    K-means -> distance_or_probability is the Euclidean distance (in scaled
    feature space) to the assigned cluster's centroid; NOT reinterpreted as
    a probability. GaussianMixture -> distance_or_probability is
    predict_proba's probability for the assigned component.
    """
    if model_bundle is None:
        if model_path is None:
            raise ValueError("Provide either model_bundle or model_path")
        model_bundle = load_model_bundle(model_path)

    scaler = model_bundle["scaler"]
    model = model_bundle["model"]
    method = model_bundle["method"]
    cluster_names = model_bundle["cluster_names"]
    columns = model_bundle["columns"]

    metrics = {"volatility": volatility, "mdd": mdd, "beta": beta, "sector_hhi": sector_hhi}
    row = pd.DataFrame([metrics])[columns]
    X = scaler.transform(row.values)

    if method == "kmeans":
        distances = np.linalg.norm(model.cluster_centers_ - X, axis=1)
        cluster_id = int(np.argmin(distances))
        distance_or_probability = float(distances[cluster_id])
    else:
        proba = model.predict_proba(X)[0]
        cluster_id = int(np.argmax(proba))
        distance_or_probability = float(proba[cluster_id])

    return {
        "risk_profile": cluster_names.get(cluster_id, cluster_names.get(str(cluster_id))),
        "cluster_id": cluster_id,
        "distance_or_probability": distance_or_probability,
        "metrics": metrics,
    }
