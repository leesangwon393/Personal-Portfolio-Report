"""Tests for the SEC 13F risk-clustering pipeline (sec13f/*).

These are all offline/synthetic — no SEC/OpenFIGI/Yahoo Finance network
calls — so they run in CI without external dependencies. The end-to-end
network pipeline itself is exercised for real by
`scripts/run_13f_clustering.py` (see report.md for that run's actual
output), not by this file.

Run: python -m pytest test_13f_clustering.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sec13f import clustering, edgar, metrics

RNG_SEED = 42


# ──────────────────────────────────────────────────────────────
# Section 12: "포트폴리오 비중 합이 1인지"
# ──────────────────────────────────────────────────────────────
def test_weights_sum_to_one():
    values = pd.Series({"AAPL": 300.0, "MSFT": 500.0, "NVDA": 200.0})
    weights = metrics.compute_weights(values)
    assert weights.sum() == pytest.approx(1.0)
    assert weights["MSFT"] == pytest.approx(0.5)


def test_renormalize_after_dropping_a_ticker():
    values = pd.Series({"AAPL": 300.0, "MSFT": 500.0, "NVDA": 200.0})
    weights = metrics.compute_weights(values)
    remaining = weights.drop("NVDA")  # e.g. NVDA had no price history
    renorm = metrics.renormalize_weights(remaining)
    assert renorm.sum() == pytest.approx(1.0)
    assert renorm["MSFT"] == pytest.approx(500 / 800)


# ──────────────────────────────────────────────────────────────
# Section 12: "MDD 계산이 올바른지"
# ──────────────────────────────────────────────────────────────
def test_mdd_known_drawdown():
    # cumulative value path: 1 -> 1.2 -> 0.9 -> 1.0 (peak 1.2, trough 0.9)
    # -> drawdown = 1 - 0.9/1.2 = 0.25
    returns = pd.Series([0.20, -0.25, 0.1111111111])
    mdd = metrics.max_drawdown(returns)
    assert mdd == pytest.approx(0.25, abs=1e-6)


def test_mdd_is_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])
    assert metrics.max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)


def test_mdd_is_stored_as_positive_magnitude():
    returns = pd.Series([-0.1, -0.05, 0.02])
    assert metrics.max_drawdown(returns) > 0


# ──────────────────────────────────────────────────────────────
# Section 12: "Beta 계산이 알려진 예제와 일치하는지"
# ──────────────────────────────────────────────────────────────
def test_beta_matches_known_example_beta_two():
    # Portfolio that moves exactly 2x the market, no noise -> beta == 2.0
    rng = np.random.default_rng(RNG_SEED)
    market = pd.Series(rng.normal(0, 0.01, 300))
    portfolio = 2.0 * market
    assert metrics.beta(portfolio, market) == pytest.approx(2.0, abs=1e-9)


def test_beta_matches_known_example_beta_one():
    rng = np.random.default_rng(RNG_SEED)
    market = pd.Series(rng.normal(0, 0.01, 300))
    assert metrics.beta(market, market) == pytest.approx(1.0, abs=1e-9)


def test_beta_with_offsetting_noise_is_close_to_true_beta():
    # beta = 0.5 plus independent idiosyncratic noise; Cov/Var recovers the
    # true beta up to sampling noise for a large-enough sample.
    rng = np.random.default_rng(RNG_SEED)
    market = pd.Series(rng.normal(0, 0.01, 2000))
    idio = pd.Series(rng.normal(0, 0.002, 2000))
    portfolio = 0.5 * market + idio
    assert metrics.beta(portfolio, market) == pytest.approx(0.5, abs=0.05)


# ──────────────────────────────────────────────────────────────
# Section 12: "Sector HHI 범위가 0~1인지"
# ──────────────────────────────────────────────────────────────
def test_sector_hhi_range_fully_concentrated():
    weights = pd.Series({"AAPL": 1.0})
    hhi, _ = metrics.sector_hhi(weights, {"AAPL": "Technology"})
    assert hhi == pytest.approx(1.0)


def test_sector_hhi_range_diversified():
    weights = pd.Series({f"T{i}": 1 / 10 for i in range(10)})
    sector_map = {f"T{i}": f"Sector{i}" for i in range(10)}
    hhi, _ = metrics.sector_hhi(weights, sector_map)
    assert hhi == pytest.approx(0.1)
    assert 0.0 <= hhi <= 1.0


def test_sector_hhi_pools_unknown_sector_and_reports_its_weight():
    weights = pd.Series({"AAPL": 0.6, "OBSCURE": 0.4})
    hhi, detail = metrics.sector_hhi(weights, {"AAPL": "Technology"})  # OBSCURE unresolved
    assert detail["unknown_sector_weight"] == pytest.approx(0.4)
    assert hhi == pytest.approx(0.6 ** 2 + 0.4 ** 2)


def test_sector_hhi_never_exceeds_one_for_valid_weights():
    rng = np.random.default_rng(RNG_SEED)
    raw = rng.random(8)
    weights = pd.Series(raw / raw.sum(), index=[f"T{i}" for i in range(8)])
    sector_map = {f"T{i}": f"Sector{i % 3}" for i in range(8)}
    hhi, _ = metrics.sector_hhi(weights, sector_map)
    assert 0.0 <= hhi <= 1.0


# ──────────────────────────────────────────────────────────────
# compute_portfolio_metrics: end-to-end on synthetic data
# ──────────────────────────────────────────────────────────────
def _synthetic_returns(tickers, n_days=300, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, n_days)
    data = {"SPY": market}
    for i, t in enumerate(tickers):
        beta_i = 0.5 + 0.3 * i
        idio = rng.normal(0, 0.008, n_days)
        data[t] = beta_i * market + idio
    return pd.DataFrame(data)


def test_compute_portfolio_metrics_end_to_end():
    tickers = ["A", "B", "C"]
    values = pd.Series({"A": 500.0, "B": 300.0, "C": 200.0})
    returns = _synthetic_returns(tickers)
    sector_map = {"A": "Tech", "B": "Tech", "C": "Healthcare"}
    result = metrics.compute_portfolio_metrics(values, returns, returns["SPY"], sector_map)
    assert result["weights"].sum() == pytest.approx(1.0)
    assert result["volatility"] > 0
    assert 0.0 <= result["mdd"] <= 1.0
    assert 0.0 <= result["sector_hhi"] <= 1.0
    assert isinstance(result["beta"], float)


# ──────────────────────────────────────────────────────────────
# Section 12: clustering reproducibility, scaler/model persistence,
# new-portfolio classification
# ──────────────────────────────────────────────────────────────
def _synthetic_portfolio_metrics(n=60, seed=RNG_SEED) -> pd.DataFrame:
    """Three well-separated synthetic clusters in (vol, mdd, beta, hhi)
    space so KMeans has an unambiguous K=3 solution to test reproducibility
    and classify_portfolio() against.
    """
    rng = np.random.default_rng(seed)
    per_cluster = n // 3
    centers = [
        (0.10, 0.08, 0.6, 0.35),   # low-risk-ish
        (0.20, 0.20, 1.0, 0.15),   # mid
        (0.35, 0.40, 1.5, 0.10),   # high-risk-ish
    ]
    rows = []
    for cx in centers:
        for _ in range(per_cluster):
            rows.append([max(0.001, c + rng.normal(0, spread))
                         for c, spread in zip(cx, [0.02, 0.02, 0.08, 0.02])])
    df = pd.DataFrame(rows, columns=clustering.FEATURE_COLUMNS)
    df["manager_name"] = [f"Manager {i}" for i in range(len(df))]
    df["cik"] = [str(1000 + i) for i in range(len(df))]
    return df


def test_clustering_is_reproducible_with_fixed_seed():
    df = _synthetic_portfolio_metrics()
    scaler = clustering.fit_scaler(df, "robust")
    X = clustering.transform(scaler, df)

    model_a = clustering._fit_kmeans(X, 3, seed=42)
    model_b = clustering._fit_kmeans(X, 3, seed=42)
    assert np.array_equal(model_a.predict(X), model_b.predict(X))
    assert np.allclose(model_a.cluster_centers_, model_b.cluster_centers_)


def test_evaluate_models_covers_k_range_and_both_methods():
    df = _synthetic_portfolio_metrics()
    scaler = clustering.fit_scaler(df, "robust")
    X = clustering.transform(scaler, df)
    evaluation = clustering.evaluate_models(X, k_range=range(2, 5), compute_stability=False)
    assert set(evaluation["method"]) == {"kmeans", "gmm"}
    assert set(evaluation["k"]) >= {2, 3, 4}
    assert (evaluation["silhouette"] <= 1.0).all()
    assert (evaluation["silhouette"] >= -1.0).all()


def test_cluster_risk_orders_defensive_balanced_aggressive_correctly():
    df = _synthetic_portfolio_metrics()
    scaler = clustering.fit_scaler(df, "robust")
    X = clustering.transform(scaler, df)
    model = clustering._fit_kmeans(X, 3, seed=42)
    labels = model.predict(X)
    centroids_scaled, _ = clustering.compute_centroids(model, scaler, X, labels)
    cluster_risk = clustering.compute_cluster_risk(centroids_scaled)
    names = clustering.assign_risk_profile_names(cluster_risk)

    assert set(names.values()) == {"DEFENSIVE", "BALANCED", "AGGRESSIVE"}
    ordered_risk = cluster_risk.sort_values()
    assert names[ordered_risk.index[0]] == "DEFENSIVE"
    assert names[ordered_risk.index[-1]] == "AGGRESSIVE"


def test_assign_risk_profile_names_falls_back_for_non_three_k():
    fake_risk = pd.Series([0.5, -0.2, 1.0, -1.5], index=[0, 1, 2, 3])
    names = clustering.assign_risk_profile_names(fake_risk)
    assert "DEFENSIVE" not in names.values()
    assert names[3] == "RISK_TIER_1_OF_4"  # lowest risk


def test_model_bundle_save_and_reload_roundtrip(tmp_path):
    df = _synthetic_portfolio_metrics()
    scaler = clustering.fit_scaler(df, "robust")
    X = clustering.transform(scaler, df)
    model = clustering._fit_kmeans(X, 3, seed=42)
    labels = model.predict(X)
    centroids_scaled, _ = clustering.compute_centroids(model, scaler, X, labels)
    cluster_risk = clustering.compute_cluster_risk(centroids_scaled)
    names = clustering.assign_risk_profile_names(cluster_risk)

    path = tmp_path / "model.joblib"
    clustering.save_model_bundle(path, scaler, model, "kmeans", names)
    bundle = clustering.load_model_bundle(path)

    X_reloaded = bundle["scaler"].transform(df[clustering.FEATURE_COLUMNS].values)
    assert np.array_equal(bundle["model"].predict(X_reloaded), model.predict(X))


def test_classify_portfolio_returns_expected_shape_for_kmeans(tmp_path):
    df = _synthetic_portfolio_metrics()
    scaler = clustering.fit_scaler(df, "robust")
    X = clustering.transform(scaler, df)
    model = clustering._fit_kmeans(X, 3, seed=42)
    labels = model.predict(X)
    centroids_scaled, _ = clustering.compute_centroids(model, scaler, X, labels)
    names = clustering.assign_risk_profile_names(clustering.compute_cluster_risk(centroids_scaled))

    path = tmp_path / "model.joblib"
    clustering.save_model_bundle(path, scaler, model, "kmeans", names)

    result = clustering.classify_portfolio(
        volatility=0.35, mdd=0.40, beta=1.5, sector_hhi=0.10, model_path=path,
    )
    assert result["risk_profile"] == "AGGRESSIVE"
    assert result["cluster_id"] in {0, 1, 2}
    assert result["distance_or_probability"] >= 0.0  # KMeans -> a distance, not a probability
    assert result["metrics"]["beta"] == 1.5


def test_classify_portfolio_gmm_returns_a_probability(tmp_path):
    df = _synthetic_portfolio_metrics()
    scaler = clustering.fit_scaler(df, "robust")
    X = clustering.transform(scaler, df)
    model = clustering._fit_gmm(X, 3, seed=42)
    labels = model.predict(X)
    centroids_scaled, _ = clustering.compute_centroids(model, scaler, X, labels)
    names = clustering.assign_risk_profile_names(clustering.compute_cluster_risk(centroids_scaled))

    path = tmp_path / "gmm_model.joblib"
    clustering.save_model_bundle(path, scaler, model, "gmm", names)

    result = clustering.classify_portfolio(
        volatility=0.10, mdd=0.08, beta=0.6, sector_hhi=0.35, model_path=path,
    )
    assert 0.0 <= result["distance_or_probability"] <= 1.0  # GMM -> a probability


# ──────────────────────────────────────────────────────────────
# edgar.py pure-function tests (amendment resolution, PUT/CALL split,
# candidate selection) — synthetic SEC-shaped DataFrames, no network.
# ──────────────────────────────────────────────────────────────
def test_resolve_latest_filings_drops_notice_only_and_keeps_latest_amendment():
    submission = pd.DataFrame({
        "ACCESSION_NUMBER": ["A1", "A2", "A3", "B1"],
        "FILING_DATE": ["01-AUG-2026", "10-AUG-2026", "15-AUG-2026", "05-AUG-2026"],
        "SUBMISSIONTYPE": ["13F-HR", "13F-HR/A", "13F-NT", "13F-HR"],
        "CIK": ["0001", "0001", "0002", "0003"],
        "PERIODOFREPORT": ["31-MAR-2026"] * 3 + ["31-MAR-2026"],
    })
    submission["FILING_DATE_dt"] = edgar._parse_sec_date(submission["FILING_DATE"])
    latest = edgar.resolve_latest_filings(submission)

    # CIK 0002's only filing is 13F-NT (notice-only) -> should be dropped entirely.
    assert "0002" not in set(latest["CIK"])
    # CIK 0001 has original A1 + amendment A2 -> only the later-filed A2 kept.
    cik1 = latest[latest["CIK"] == "0001"]
    assert len(cik1) == 1
    assert cik1.iloc[0]["ACCESSION_NUMBER"] == "A2"
    # CIK 0003's single 13F-HR is kept as-is.
    assert "B1" in set(latest["ACCESSION_NUMBER"])


def test_split_long_vs_options_excludes_put_and_call():
    info = pd.DataFrame({
        "ACCESSION_NUMBER": ["A1"] * 4,
        "CUSIP": ["C1", "C2", "C3", "C4"],
        "VALUE": [100, 200, 300, 400],
        "PUTCALL": [None, "Put", "Call", ""],
    })
    long_only, options_only = edgar.split_long_vs_options(info)
    assert set(long_only["CUSIP"]) == {"C1", "C4"}  # None and "" (blank) are plain long positions
    assert set(options_only["CUSIP"]) == {"C2", "C3"}


def test_select_top_value_candidates_respects_min_positions_and_order():
    filer_table = pd.DataFrame({
        "manager_name": ["Big", "Small", "Mid"],
        "cik": ["1", "2", "3"],
        "accession_number": ["a1", "a2", "a3"],
        "FILING_DATE": ["01-AUG-2026"] * 3,
        "report_period": ["31-MAR-2026"] * 3,
        "reported_value": [1_000_000, 500_000, 750_000],
        "position_count": [50, 5, 20],  # Small has only 5 positions -> excluded by min_positions
    })
    top = edgar.select_top_value_candidates(filer_table, n=10, min_positions=10)
    assert list(top["manager_name"]) == ["Big", "Mid"]  # sorted desc, Small dropped


def test_select_top_value_candidates_prepends_guaranteed_ciks_not_otherwise_selected():
    filer_table = pd.DataFrame({
        "manager_name": ["Big", "Mid", "Famous-but-small"],
        "cik": ["1", "2", "9"],
        "accession_number": ["a1", "a2", "a9"],
        "FILING_DATE": ["01-AUG-2026"] * 3,
        "report_period": ["31-MAR-2026"] * 3,
        "reported_value": [1_000_000, 750_000, 10_000],  # too small to make top-1 by value
        "position_count": [50, 20, 15],
    })
    top1 = edgar.select_top_value_candidates(
        filer_table, n=1, min_positions=10, guaranteed_ciks={"9"},
    )
    # n=1 alone would only return "Big"; the guaranteed CIK is prepended on top.
    assert set(top1["cik"]) == {"1", "9"}
    assert top1.iloc[0]["cik"] == "9"  # guaranteed rows go first


def test_prepend_guaranteed_candidates_ignores_ineligible_or_absent_ciks():
    ranked = pd.DataFrame({"cik": ["1", "2"], "manager_name": ["A", "B"]})
    eligible = ranked.copy()
    # CIK "999" isn't in `eligible` at all (e.g. didn't clear min_positions) -> no-op.
    out = edgar._prepend_guaranteed_candidates(ranked, eligible, {"999"})
    assert list(out["cik"]) == ["1", "2"]
    # No guaranteed CIKs requested -> unchanged.
    out2 = edgar._prepend_guaranteed_candidates(ranked, eligible, None)
    assert out2 is ranked


def test_determine_report_period_picks_the_mode():
    coverpage = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2", "a3", "a4"],
        "REPORTCALENDARORQUARTER": ["31-MAR-2026", "31-MAR-2026", "31-MAR-2026", "31-DEC-2025"],
    })
    period = edgar.determine_report_period(coverpage)
    assert period == pd.Timestamp("2026-03-31")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
