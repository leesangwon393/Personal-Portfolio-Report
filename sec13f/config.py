"""Central configuration for the SEC 13F risk-clustering pipeline.

Every tunable threshold used by `sec13f/*` and `scripts/run_13f_clustering.py`
lives here so the whole pipeline's behavior is controlled from one file, and
so `--quarter`/`--num-portfolios`/etc. CLI flags have a single source of
defaults to override.
"""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "13f_clustering"

# --- Reproducibility ---
RANDOM_SEED = 42

# --- SEC EDGAR access ---
SEC_BASE_URL = "https://www.sec.gov"
SEC_13F_DATASETS_PAGE = (
    "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
)
DEFAULT_SEC_USER_AGENT = "PersonalPortfolioResearch contact@example.com"
SEC_REQUEST_MIN_INTERVAL_SECONDS = 0.15  # SEC allows <=10 req/s; stay well under
SEC_MAX_RETRIES = 5
SEC_BACKOFF_BASE_SECONDS = 1.5

# --- Portfolio selection (Section 2) ---
NUM_PORTFOLIOS = 100
MIN_LONG_POSITIONS = 10
MIN_MAPPING_COVERAGE = 0.80
VALID_HOLDING_SUBMISSION_TYPES = ("13F-HR", "13F-HR/A")
NOTICE_ONLY_SUBMISSION_TYPES = ("13F-NT", "13F-NT/A")

# Selection strategies exposed via --selection-method
SELECTION_TOP_VALUE = "top_value"
SELECTION_AUM_STRATIFIED = "aum_stratified"
SELECTION_METHODS = (SELECTION_TOP_VALUE, SELECTION_AUM_STRATIFIED)

# --- Price history / return construction (Section 4) ---
DEFAULT_LOOKBACK_DAYS = 252
BENCHMARK_TICKER = "SPY"
MIN_PRICE_HISTORY_COVERAGE = 0.90  # of lookback trading days, per ticker

# --- Risk metrics (Section 5) ---
TRADING_DAYS_PER_YEAR = 252

# --- Preprocessing (Section 6) ---
WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

# --- Clustering (Section 7) ---
KMEANS_K_RANGE = range(2, 7)  # 2..6 inclusive
KMEANS_N_INIT = 50
GMM_N_INIT = 10
PREFERRED_K = 3
CLUSTER_STABILITY_N_BOOTSTRAP = 30
CLUSTER_STABILITY_SAMPLE_FRAC = 0.8

RISK_PROFILE_NAMES = ["DEFENSIVE", "BALANCED", "AGGRESSIVE"]  # low -> high ClusterRisk

# --- Representative investors for sanity-check (Section 8) ---
# CIK -> label used only for post-hoc lookup/reporting; never used as a
# training/ground-truth label for the unsupervised clustering itself.
# CIKs verified directly against SEC EDGAR's own COVERPAGE.FILINGMANAGER_NAME
# for the 2026-03-31 dataset (artifacts/13f_clustering/raw/filings/
# filer_table_all.csv) rather than assumed from memory — several "well
# known" CIKs one might guess from memory turned out wrong on inspection
# (e.g. Renaissance Technologies LLC's real CIK is 0001037389, not
# 0001040273; ARK Investment Management LLC's is 0001697748).
REPRESENTATIVE_INVESTORS = {
    "0001067983": "Berkshire Hathaway Inc",
    "0001697748": "ARK Investment Management LLC",
    "0001536411": "Duquesne Family Office LLC (Druckenmiller)",
    "0001350694": "Bridgewater Associates, LP",
    "0001037389": "Renaissance Technologies LLC",
    "0001061768": "Baupost Group LLC/MA",
    "0001336528": "Pershing Square Capital Management, L.P.",
}

# --- Output layout (Section 10) ---
def artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    d = Path(artifacts_dir)
    return {
        "root": d,
        "raw": d / "raw",
        "raw_filings": d / "raw" / "filings",
        "raw_holdings": d / "raw" / "holdings",
        "cache": d / "cache",
        "cache_ticker_mapping": d / "cache" / "ticker_mapping",
        "cache_prices": d / "cache" / "prices",
        "cache_sectors": d / "cache" / "sectors",
        "processed": d / "processed",
        "clustering": d / "clustering",
        "figures": d / "figures",
    }
