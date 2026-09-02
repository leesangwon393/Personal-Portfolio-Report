"""End-to-end orchestration for scripts/run_13f_clustering.py.

Section 1-11 of the spec, wired together: quarter selection -> SEC bulk
13F data -> filer/portfolio selection -> CUSIP->ticker mapping -> price
history -> risk metrics -> preprocessing -> clustering -> figures -> saved
artifacts + report.md.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sec13f import clustering, config, edgar, mapping, metrics, prices, sectors, visualize


@dataclass
class PipelineResult:
    paths: dict
    quarter_info: dict
    n_candidates_scanned: int = 0
    n_selected: int = 0
    n_excluded: int = 0
    exclusion_reasons: dict = field(default_factory=dict)
    coverage_stats: dict = field(default_factory=dict)
    evaluation_df: pd.DataFrame | None = None
    chosen_method: str = ""
    chosen_k: int = 0
    cluster_names: dict = field(default_factory=dict)
    representative_results: list = field(default_factory=list)
    feature_summary: pd.DataFrame | None = None


def _log(msg: str) -> None:
    print(f"[13f] {msg}")


def run_pipeline(
    quarter: str | None,
    num_portfolios: int,
    lookback_days: int,
    output_dir: Path,
    selection_method: str = config.SELECTION_TOP_VALUE,
    candidate_pool_multiplier: int = 3,
    scaler_kind: str = "robust",
    fit_k: int = config.PREFERRED_K,
) -> PipelineResult:
    np.random.seed(config.RANDOM_SEED)
    paths = config.artifact_paths(output_dir)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # ---------- Section 1: quarter selection ----------
    _log("Selecting SEC 13F dataset quarter...")
    report_period, extracted_dir, quarter_info = edgar.select_dataset_for_quarter(
        quarter, paths["cache"], paths["cache"],
    )
    if not quarter_info["requested_quarter_available"] and quarter is not None:
        _log("WARNING: " + quarter_info["fallback_reason"])
    _log(f"Effective quarter: {report_period.date()} (dataset window {quarter_info['dataset_window']})")

    # ---------- Section 2: SEC 13F filer/holdings collection ----------
    _log("Loading SUBMISSION/COVERPAGE and resolving amendments...")
    submission, coverpage = edgar.load_submissions_and_coverpages(extracted_dir, report_period)
    latest_filings = edgar.resolve_latest_filings(submission)
    _log(f"{len(latest_filings)} distinct (CIK, period) 13F-HR/13F-HR/A filings after amendment resolution")

    target_accessions = set(latest_filings["ACCESSION_NUMBER"])
    infotable_cache = paths["cache"] / f"infotable_{report_period.date()}.parquet"
    info = edgar.load_infotable_for_accessions(extracted_dir, target_accessions, cache_path=infotable_cache)
    long_only, options_only = edgar.split_long_vs_options(info)
    _log(f"{len(info)} holding rows total; {len(options_only)} PUT/CALL rows excluded "
         f"({len(long_only)} plain long positions kept)")

    filer_table = edgar.build_filer_table(latest_filings, coverpage, long_only)
    filer_table.to_csv(paths["raw_filings"] / "filer_table_all.csv", index=False)

    guaranteed_ciks = set(config.REPRESENTATIVE_INVESTORS.keys())
    if selection_method == config.SELECTION_AUM_STRATIFIED:
        candidates = edgar.select_aum_stratified_candidates(
            filer_table, n=num_portfolios * candidate_pool_multiplier, min_positions=config.MIN_LONG_POSITIONS,
            guaranteed_ciks=guaranteed_ciks,
        )
    else:
        candidates = edgar.select_top_value_candidates(
            filer_table, n=num_portfolios * candidate_pool_multiplier, min_positions=config.MIN_LONG_POSITIONS,
            guaranteed_ciks=guaranteed_ciks,
        )
    _log(f"{len(candidates)} candidate filers ranked (selection_method={selection_method}, "
         f"pool = {candidate_pool_multiplier}x target)")

    # ---------- Section 3: CUSIP -> ticker mapping over the whole candidate pool ----------
    candidate_holdings = long_only[long_only["ACCESSION_NUMBER"].isin(candidates["accession_number"])]
    all_cusips = candidate_holdings["CUSIP"].dropna().unique().tolist()
    _log(f"Mapping {len(all_cusips)} unique CUSIPs across the candidate pool via OpenFIGI...")
    cusip_map = mapping.map_cusips_to_tickers(all_cusips, paths["cache_ticker_mapping"])
    cusip_to_ticker = dict(zip(cusip_map["cusip"], cusip_map["ticker"]))
    mapped_rate = cusip_map["mapped"].mean() if len(cusip_map) else 0.0
    _log(f"CUSIP mapping: {cusip_map['mapped'].sum()}/{len(cusip_map)} resolved ({mapped_rate:.1%})")

    candidate_holdings = candidate_holdings.copy()
    candidate_holdings["ticker"] = candidate_holdings["CUSIP"].map(cusip_to_ticker)

    # ---------- Per-candidate mapping coverage, walk down the ranked list ----------
    selected_rows = []
    excluded_rows = []
    exclusion_reasons: dict[str, int] = {}
    for _, cand in candidates.iterrows():
        if len(selected_rows) >= num_portfolios:
            break
        acc = cand["accession_number"]
        holdings = candidate_holdings[candidate_holdings["ACCESSION_NUMBER"] == acc]
        total_value = holdings["VALUE"].sum()
        mapped_value = holdings.loc[holdings["ticker"].notna(), "VALUE"].sum()
        coverage = mapped_value / total_value if total_value > 0 else 0.0
        n_unmapped = int(holdings["ticker"].isna().sum())
        unmapped_value_share = 1 - coverage

        record = {
            "manager_name": cand["manager_name"],
            "cik": cand["cik"],
            "accession_number": acc,
            "report_period": cand["report_period"],
            "reported_value": total_value,
            "position_count": cand["position_count"],
            "mapped_value": mapped_value,
            "mapping_coverage": coverage,
            "n_unmapped_positions": n_unmapped,
            "unmapped_value_share": unmapped_value_share,
        }
        if coverage < config.MIN_MAPPING_COVERAGE:
            record["exclusion_reason"] = "mapping_coverage_below_threshold"
            excluded_rows.append(record)
            exclusion_reasons["mapping_coverage_below_threshold"] = (
                exclusion_reasons.get("mapping_coverage_below_threshold", 0) + 1
            )
            continue
        selected_rows.append(record)

    if len(selected_rows) < num_portfolios:
        _log(
            f"WARNING: only {len(selected_rows)}/{num_portfolios} candidates cleared "
            f"{config.MIN_MAPPING_COVERAGE:.0%} mapping coverage out of the "
            f"{len(candidates)} scanned. Consider raising --candidate-pool-multiplier."
        )

    selected_df = pd.DataFrame(selected_rows)
    excluded_mapping_df = pd.DataFrame(excluded_rows)
    _log(f"{len(selected_df)} portfolios selected after mapping-coverage filtering "
         f"({len(excluded_mapping_df)} excluded for coverage)")

    # ---------- Section 4-5: prices + metrics per selected portfolio ----------
    selected_accessions = set(selected_df["accession_number"])
    sel_holdings = candidate_holdings[
        candidate_holdings["ACCESSION_NUMBER"].isin(selected_accessions) & candidate_holdings["ticker"].notna()
    ].copy()
    all_tickers = sorted(sel_holdings["ticker"].unique().tolist())
    _log(f"Fetching {len(all_tickers)} unique tickers + SPY, {lookback_days}-day lookback "
         f"ending {report_period.date()}...")
    returns_df, price_coverage = prices.build_returns_matrix(
        all_tickers, report_period, lookback_days, paths["cache_prices"],
    )
    price_coverage.to_csv(paths["processed"] / "price_coverage.csv", index=False)
    market_returns = returns_df[config.BENCHMARK_TICKER] if config.BENCHMARK_TICKER in returns_df.columns else None
    if market_returns is None or market_returns.empty:
        raise RuntimeError(
            f"Could not build a {config.BENCHMARK_TICKER} return series for {report_period.date()} — "
            "cannot compute Beta for any portfolio. Check network access to Yahoo Finance and retry."
        )

    _log("Looking up sectors for all mapped tickers...")
    sector_map = sectors.lookup_sectors(all_tickers, paths["cache_sectors"])

    metric_rows = []
    price_excluded_rows = []
    for _, sel in selected_df.iterrows():
        acc = sel["accession_number"]
        holdings = sel_holdings[sel_holdings["ACCESSION_NUMBER"] == acc]
        by_ticker_value = holdings.groupby("ticker")["VALUE"].sum()
        by_ticker_value = by_ticker_value[by_ticker_value > 0]
        available = [t for t in by_ticker_value.index if t in returns_df.columns]
        price_covered_value = by_ticker_value.loc[available].sum()
        price_coverage_ratio = price_covered_value / by_ticker_value.sum() if by_ticker_value.sum() > 0 else 0.0

        if price_coverage_ratio < config.MIN_MAPPING_COVERAGE or len(available) < config.MIN_LONG_POSITIONS:
            row = sel.to_dict()
            row["exclusion_reason"] = "price_history_coverage_below_threshold"
            row["price_history_coverage"] = price_coverage_ratio
            price_excluded_rows.append(row)
            continue

        try:
            result = metrics.compute_portfolio_metrics(
                by_ticker_value.loc[available], returns_df, market_returns, sector_map,
            )
        except Exception as exc:  # noqa: BLE001
            row = sel.to_dict()
            row["exclusion_reason"] = f"metrics_computation_failed:{exc}"
            price_excluded_rows.append(row)
            continue

        metric_rows.append({
            "manager_name": sel["manager_name"],
            "cik": sel["cik"],
            "accession_number": acc,
            "report_period": sel["report_period"],
            "portfolio_value": by_ticker_value.sum(),
            "position_count": len(available),
            "mapping_coverage": sel["mapping_coverage"],
            "price_history_coverage": price_coverage_ratio,
            "volatility": result["volatility"],
            "mdd": result["mdd"],
            "beta": result["beta"],
            "sector_hhi": result["sector_hhi"],
            "unknown_sector_weight": result["hhi_detail"]["unknown_sector_weight"],
            "n_sectors": result["hhi_detail"]["n_sectors"],
        })

    portfolio_metrics = pd.DataFrame(metric_rows)
    _log(f"{len(portfolio_metrics)} portfolios have final volatility/MDD/beta/HHI "
         f"({len(price_excluded_rows)} dropped for price-history coverage)")

    all_excluded = pd.concat(
        [excluded_mapping_df, pd.DataFrame(price_excluded_rows)], ignore_index=True, sort=False,
    ) if (len(excluded_mapping_df) or len(price_excluded_rows)) else pd.DataFrame()

    # ---------- Save Section 2-6 processed outputs ----------
    selected_df.to_csv(paths["processed"] / "portfolios_100.csv", index=False)
    portfolio_metrics.to_csv(paths["processed"] / "portfolio_metrics.csv", index=False)
    all_excluded.to_csv(paths["processed"] / "excluded_portfolios.csv", index=False)

    if portfolio_metrics.empty:
        raise RuntimeError(
            "No portfolio cleared every filter (mapping coverage, price-history coverage). "
            "Cannot proceed to clustering. See excluded_portfolios.csv for reasons."
        )

    # ---------- Section 6: preprocessing ----------
    winsorized = clustering.winsorize(portfolio_metrics)
    winsorized.to_csv(paths["processed"] / "portfolio_metrics_winsorized.csv", index=False)
    missing_report = clustering.check_missing_and_infinite(portfolio_metrics)
    missing_report.to_csv(paths["processed"] / "missing_infinite_report.csv", index=False)

    scalers = {
        "robust": clustering.fit_scaler(winsorized, "robust"),
        "standard": clustering.fit_scaler(winsorized, "standard"),
    }
    scaler = scalers[scaler_kind]
    X_scaled = clustering.transform(scaler, winsorized)

    # ---------- Section 7: clustering model comparison ----------
    _log(f"Evaluating KMeans/GMM for K in {list(config.KMEANS_K_RANGE)} "
         f"(n_init={config.KMEANS_N_INIT}, random_state={config.RANDOM_SEED})...")
    evaluation_df = clustering.evaluate_models(X_scaled)
    evaluation_df.to_csv(paths["clustering"] / "clustering_evaluation.csv", index=False)

    chosen_method, chosen_k = "kmeans", fit_k
    if not evaluation_df.empty:
        preferred = evaluation_df[(evaluation_df["k"] == fit_k) & (evaluation_df["method"] == "kmeans")]
        if not preferred.empty:
            chosen_method, chosen_k = "kmeans", fit_k

    final_model = (
        clustering._fit_kmeans(X_scaled, chosen_k) if chosen_method == "kmeans"
        else clustering._fit_gmm(X_scaled, chosen_k)
    )
    labels = final_model.predict(X_scaled)

    centroids_scaled, centroids_original = clustering.compute_centroids(
        final_model, scaler, X_scaled, labels,
    )
    cluster_risk = clustering.compute_cluster_risk(centroids_scaled)
    cluster_names = clustering.assign_risk_profile_names(cluster_risk)

    centroids_out = centroids_original.copy()
    centroids_out["cluster_risk_scaled"] = cluster_risk
    centroids_out["risk_profile"] = [cluster_names[c] for c in centroids_out.index]
    centroids_out.to_csv(paths["clustering"] / "cluster_centroids.csv")

    assignments = winsorized[["manager_name", "cik"]].copy()
    for col in clustering.FEATURE_COLUMNS:
        assignments[col] = portfolio_metrics[col].values
    assignments["cluster_id"] = labels
    assignments["risk_profile"] = [cluster_names[c] for c in labels]
    if chosen_method == "kmeans":
        dists = np.linalg.norm(final_model.cluster_centers_[labels] - X_scaled, axis=1)
        assignments["distance_or_probability"] = dists
    else:
        proba = final_model.predict_proba(X_scaled)
        assignments["distance_or_probability"] = proba[np.arange(len(labels)), labels]
    assignments.to_csv(paths["clustering"] / "cluster_assignments.csv", index=False)

    clustering.save_model_bundle(
        paths["clustering"] / "model.joblib", scaler, final_model, chosen_method, cluster_names,
    )
    (paths["clustering"] / "model_meta.json").write_text(
        json.dumps({
            "method": chosen_method, "k": chosen_k, "scaler_kind": scaler_kind,
            "random_seed": config.RANDOM_SEED, "features": clustering.FEATURE_COLUMNS,
        }, indent=2)
    )

    # ---------- Section 8: representative investor sanity check ----------
    rep_results = []
    all_ciks_seen = set(filer_table["cik"])
    for cik, label in config.REPRESENTATIVE_INVESTORS.items():
        match = assignments[assignments["cik"] == cik]
        if not match.empty:
            row = match.iloc[0]
            rep_results.append({
                "label": label, "cik": cik, "in_final_100": True,
                "manager_name": row["manager_name"], "cluster_id": int(row["cluster_id"]),
                "risk_profile": row["risk_profile"], "volatility": row["volatility"],
                "mdd": row["mdd"], "beta": row["beta"], "sector_hhi": row["sector_hhi"],
            })
        else:
            rep_results.append({
                "label": label, "cik": cik, "in_final_100": False,
                "in_broader_13f_universe": cik in all_ciks_seen,
            })
    pd.DataFrame(rep_results).to_csv(paths["clustering"] / "representative_investors.csv", index=False)

    # ---------- Section 9: figures ----------
    _log("Rendering figures...")
    visualize.plot_feature_distributions(portfolio_metrics, paths["figures"] / "01_feature_distributions.png")
    visualize.plot_correlation_heatmap(portfolio_metrics, paths["figures"] / "02_correlation_heatmap.png")
    visualize.plot_silhouette_by_k(evaluation_df, paths["figures"] / "03_silhouette_by_k.png")
    pca, coords = visualize.plot_pca_scatter(
        X_scaled, labels, cluster_names, paths["figures"] / "04_pca_scatter.png",
    )
    visualize.plot_centroid_heatmap(centroids_scaled, cluster_names, paths["figures"] / "05_centroid_heatmap.png")
    visualize.plot_cluster_sizes(labels, cluster_names, paths["figures"] / "06_cluster_sizes.png")
    rep_mask = assignments["cik"].isin({r["cik"] for r in rep_results if r.get("in_final_100")})
    if rep_mask.any():
        visualize.plot_representative_investor_positions(
            coords, assignments, rep_mask, labels, cluster_names,
            paths["figures"] / "07_representative_investors.png",
        )
    visualize.plot_cluster_boxplots(
        portfolio_metrics, labels, cluster_names, paths["figures"] / "08_cluster_boxplots.png",
    )

    result = PipelineResult(
        paths=paths,
        quarter_info=quarter_info,
        n_candidates_scanned=len(candidates),
        n_selected=len(portfolio_metrics),
        n_excluded=len(all_excluded),
        exclusion_reasons=exclusion_reasons,
        coverage_stats={
            "mean_mapping_coverage": float(selected_df["mapping_coverage"].mean()) if len(selected_df) else None,
            "min_mapping_coverage": float(selected_df["mapping_coverage"].min()) if len(selected_df) else None,
        },
        evaluation_df=evaluation_df,
        chosen_method=chosen_method,
        chosen_k=chosen_k,
        cluster_names=cluster_names,
        representative_results=rep_results,
        feature_summary=portfolio_metrics[clustering.FEATURE_COLUMNS].describe(),
    )
    return result
