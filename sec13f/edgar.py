"""SEC EDGAR access: bulk Form 13F structured data sets, quarter selection,
filer/holdings loading, amendment resolution, and portfolio selection.

Data source: SEC's official quarterly "Form 13F Data Sets"
(https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets),
published as one ZIP per ~3-month filing window containing SUBMISSION.tsv /
COVERPAGE.tsv / INFOTABLE.tsv / SUMMARYPAGE.tsv for every 13F filed in that
window. This is the same underlying data as scraping individual filings one
by one, but avoids ~10,000 individual HTTP requests per quarter.
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from sec13f import config
from secrets_config import get_sec_user_agent


class SecEdgarError(RuntimeError):
    """Raised when SEC EDGAR data cannot be fetched/parsed — never caught
    silently by callers; the pipeline is expected to stop and report this
    (Section: "가짜 결과를 만들지 말고 정확한 원인과 재실행 방법을 보고한다").
    """


# ──────────────────────────────────────────────────────────────
# Rate-limited, retrying, cached HTTP session
# ──────────────────────────────────────────────────────────────
class _SecSession:
    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or get_sec_user_agent() or config.DEFAULT_SEC_USER_AGENT
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = config.SEC_REQUEST_MIN_INTERVAL_SECONDS - elapsed
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, **kwargs) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(config.SEC_MAX_RETRIES):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=kwargs.pop("timeout", 60), **kwargs)
                self._last_request_ts = time.monotonic()
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise SecEdgarError(f"SEC EDGAR returned HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                return resp
            except (requests.RequestException, SecEdgarError) as exc:
                last_exc = exc
                if attempt < config.SEC_MAX_RETRIES - 1:
                    backoff = config.SEC_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    time.sleep(backoff)
        raise SecEdgarError(
            f"Failed to fetch {url} after {config.SEC_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc


_session: _SecSession | None = None


def _get_session() -> _SecSession:
    global _session
    if _session is None:
        _session = _SecSession()
    return _session


# ──────────────────────────────────────────────────────────────
# Bulk dataset discovery + download (cached)
# ──────────────────────────────────────────────────────────────
@dataclass
class DatasetWindow:
    label: str          # e.g. "01mar2026-31may2026" or "2023q4"
    url: str


def list_available_dataset_windows() -> list[DatasetWindow]:
    """Scrapes the SEC 13F data sets index page for downloadable ZIP links.
    Returned in page order (SEC lists newest first).
    """
    resp = _get_session().get(config.SEC_13F_DATASETS_PAGE)
    hrefs = re.findall(r'href="([^"]+form13f\.zip)"', resp.text)
    windows = []
    seen = set()
    for href in hrefs:
        if href in seen:
            continue
        seen.add(href)
        url = href if href.startswith("http") else f"https://www.sec.gov{href}"
        label = Path(href).name.replace("_form13f.zip", "")
        windows.append(DatasetWindow(label=label, url=url))
    if not windows:
        raise SecEdgarError(
            f"No 13F dataset ZIP links found on {config.SEC_13F_DATASETS_PAGE} "
            "(page structure may have changed)."
        )
    return windows


def download_dataset(window: DatasetWindow, cache_dir: Path) -> Path:
    """Downloads (or reuses a cached copy of) one quarterly 13F dataset ZIP."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{window.label}_form13f.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    resp = _get_session().get(window.url, timeout=300)
    zip_path.write_bytes(resp.content)
    if zip_path.stat().st_size == 0:
        raise SecEdgarError(f"Downloaded 0 bytes for {window.url}")
    return zip_path


def extract_dataset(zip_path: Path, extract_dir: Path) -> Path:
    """SEC's per-window ZIPs aren't consistently packaged: some put the
    TSVs at the archive root, others nest them inside a subfolder (with
    inconsistent casing, e.g. `01JUN2025-31AUG2025_form13f/`). Rather than
    assume a layout, extract once and then search for SUBMISSION.tsv
    wherever it landed.
    """
    out_dir = extract_dir / zip_path.stem
    existing = list(out_dir.rglob("SUBMISSION.tsv")) if out_dir.exists() else []
    if existing:
        return existing[0].parent

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    found = list(out_dir.rglob("SUBMISSION.tsv"))
    if not found:
        raise SecEdgarError(f"{zip_path} did not contain SUBMISSION.tsv after extraction")
    return found[0].parent


# ──────────────────────────────────────────────────────────────
# Quarter selection (Section 1)
# ──────────────────────────────────────────────────────────────
def _parse_sec_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d-%b-%Y", errors="coerce")


def determine_report_period(coverpage: pd.DataFrame) -> pd.Timestamp:
    """The dataset window's dominant REPORTCALENDARORQUARTER value is the
    "완전히 제출이 끝난 공통 13F 분기" this window was built to capture —
    the on-time deadline for that quarter fell inside the window, so the
    overwhelming majority of on-time filers report that period (stragglers
    for older periods make up a small tail). We pick the mode rather than
    trusting the filename, since SEC's window naming is by *filing* date
    range, not report period.
    """
    counts = coverpage["REPORTCALENDARORQUARTER"].value_counts()
    if counts.empty:
        raise SecEdgarError("COVERPAGE.tsv has no REPORTCALENDARORQUARTER values")
    mode_str = counts.idxmax()
    period = pd.to_datetime(mode_str, format="%d-%b-%Y", errors="coerce")
    if pd.isna(period):
        raise SecEdgarError(f"Could not parse dominant report period {mode_str!r}")
    return period


def select_dataset_for_quarter(
    requested_quarter: str | None,
    cache_dir: Path,
    extract_dir: Path,
) -> tuple[pd.Timestamp, Path, dict]:
    """Finds the most recent SEC 13F bulk dataset window, downloads/extracts
    it, and reports whether it actually covers `requested_quarter`
    (YYYY-MM-DD). If the requested quarter isn't the dominant period in any
    published window (e.g. its filing deadline hasn't passed / SEC hasn't
    published that window yet), falls back to the newest window whose
    dominant period IS a complete, fully-elapsed quarter, and records the
    substitution so it can be reported to the user — never silently.

    Returns (effective_quarter, extracted_dir, selection_info).
    """
    windows = list_available_dataset_windows()
    requested_ts = pd.to_datetime(requested_quarter) if requested_quarter else None

    # SEC lists windows newest-first, and a freshly-published window's
    # dominant period is, by construction, already "the newest fully
    # elapsed/published quarter" — so the newest window alone is the
    # correct fallback. We only look a few windows further back (in case
    # the caller asked for a *slightly* older quarter than the very latest
    # one) rather than downloading/extracting the entire historical archive
    # (50+ windows back to 2013) just to fail to find an exotic old quarter.
    MAX_WINDOWS_TO_CHECK = 4
    checked: list[dict] = []
    chosen: tuple[pd.Timestamp, Path, DatasetWindow] | None = None
    requested_found = False

    for window in windows[:MAX_WINDOWS_TO_CHECK]:
        zip_path = download_dataset(window, cache_dir)
        extracted = extract_dataset(zip_path, extract_dir)
        coverpage = pd.read_csv(
            extracted / "COVERPAGE.tsv", sep="\t",
            usecols=["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER"], dtype=str,
        )
        period = determine_report_period(coverpage)
        checked.append({"window": window.label, "dominant_period": str(period.date())})

        if chosen is None:
            # first (= newest, since SEC lists newest-first) window checked
            # becomes the fallback default if the requested quarter never
            # turns up as anyone's dominant period.
            chosen = (period, extracted, window)
        if requested_ts is not None and period == requested_ts:
            chosen = (period, extracted, window)
            requested_found = True
            break

    if chosen is None:
        raise SecEdgarError("No usable 13F dataset window found on SEC EDGAR")

    period, extracted, window = chosen
    info = {
        "requested_quarter": requested_quarter,
        "effective_quarter": str(period.date()),
        "requested_quarter_available": requested_found,
        "dataset_window": window.label,
        "dataset_url": window.url,
        "windows_checked": checked,
    }
    if requested_ts is not None and not requested_found:
        info["fallback_reason"] = (
            f"--quarter {requested_quarter} is not the dominant REPORTCALENDARORQUARTER "
            f"in any published SEC 13F bulk dataset window yet (its 45-day filing "
            f"deadline may not have passed, or SEC hasn't published that window's "
            f"structured data set). Falling back to the newest fully-published "
            f"complete quarter: {period.date()} (dataset window {window.label})."
        )
    return period, extracted, info


# ──────────────────────────────────────────────────────────────
# Loading + filtering the extracted TSVs (Section 2)
# ──────────────────────────────────────────────────────────────
def load_submissions_and_coverpages(extracted_dir: Path, report_period: pd.Timestamp):
    submission = pd.read_csv(extracted_dir / "SUBMISSION.tsv", sep="\t", dtype=str)
    coverpage = pd.read_csv(extracted_dir / "COVERPAGE.tsv", sep="\t", dtype=str)
    period_str = report_period.strftime("%d-%b-%Y").upper()
    target_accessions = set(
        coverpage.loc[coverpage["REPORTCALENDARORQUARTER"] == period_str, "ACCESSION_NUMBER"]
    )
    sub = submission[submission["ACCESSION_NUMBER"].isin(target_accessions)].copy()
    sub["FILING_DATE_dt"] = _parse_sec_date(sub["FILING_DATE"])
    cov = coverpage[coverpage["ACCESSION_NUMBER"].isin(target_accessions)].copy()
    return sub, cov


def resolve_latest_filings(submission: pd.DataFrame) -> pd.DataFrame:
    """Section 2: "13F-HR/A 수정공시가 존재하면 동일 기관·동일 보고기간에서
    최신 수정본을 사용한다. 원본과 수정본을 중복 계산하지 않는다."

    Keeps only 13F-HR / 13F-HR/A (drops 13F-NT notice-only filings, which
    have no holdings — the filer reports through another manager), then for
    each (CIK, PERIODOFREPORT) keeps the single filing with the latest
    FILING_DATE. Known simplification (documented in report.md): an
    AMENDMENTTYPE="NEW HOLDINGS" amendment is meant to be additive to the
    original filing, not a full replacement; picking "latest wins" like a
    restatement slightly under-counts that rare case rather than double-
    counting it.
    """
    holdings = submission[
        submission["SUBMISSIONTYPE"].isin(config.VALID_HOLDING_SUBMISSION_TYPES)
    ].copy()
    holdings = holdings.sort_values("FILING_DATE_dt")
    latest = holdings.groupby(["CIK", "PERIODOFREPORT"], as_index=False).tail(1)
    return latest.reset_index(drop=True)


def load_infotable_for_accessions(
    extracted_dir: Path, accessions: set[str], cache_path: Path | None = None
) -> pd.DataFrame:
    """INFOTABLE.tsv covers the *entire* ~3-month dataset window (millions
    of rows across every reported period, not just our target quarter), so
    this streams it in chunks and keeps only rows for the accessions we
    actually need — cached as parquet so repeat runs don't re-scan the TSV.
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)

    usecols = [
        "ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "FIGI",
        "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL", "INVESTMENTDISCRETION",
        "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE",
    ]
    chunks = pd.read_csv(
        extracted_dir / "INFOTABLE.tsv", sep="\t", dtype=str,
        usecols=usecols, chunksize=300_000,
    )
    kept = []
    for chunk in chunks:
        sel = chunk[chunk["ACCESSION_NUMBER"].isin(accessions)]
        if len(sel):
            kept.append(sel)
    if not kept:
        raise SecEdgarError("No INFOTABLE rows matched the selected accession numbers")
    info = pd.concat(kept, ignore_index=True)
    info["VALUE"] = pd.to_numeric(info["VALUE"], errors="coerce")
    info["SSHPRNAMT"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        info.to_parquet(cache_path)
    return info


def split_long_vs_options(info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 2: "PUT 포지션과 해석이 불명확한 옵션 포지션은 기본
    포트폴리오에서 제외하고, 제외 내역을 기록한다."

    A non-null PUTCALL means the row is an options position (SEC's 13F
    instructions have filers report VALUE for the underlying shares subject
    to the option, not the option's own market value or a comparable
    long-equity dollar exposure) — CALL is not excluded only, because "long
    call value" isn't economically equivalent to "long stock value" either;
    we exclude both PUT and CALL and keep plain long positions (PUTCALL is
    blank/NaN).
    """
    is_option = info["PUTCALL"].notna() & (info["PUTCALL"].str.strip() != "")
    return info[~is_option].copy(), info[is_option].copy()


# ──────────────────────────────────────────────────────────────
# Filer aggregation + candidate ranking (Section 2)
# ──────────────────────────────────────────────────────────────
def build_filer_table(
    latest_filings: pd.DataFrame, coverpage: pd.DataFrame, long_only_info: pd.DataFrame
) -> pd.DataFrame:
    agg = long_only_info.groupby("ACCESSION_NUMBER").agg(
        reported_value=("VALUE", "sum"),
        position_count=("CUSIP", "nunique"),
    ).reset_index()

    table = latest_filings.merge(
        coverpage[["ACCESSION_NUMBER", "FILINGMANAGER_NAME"]], on="ACCESSION_NUMBER", how="left"
    ).merge(agg, on="ACCESSION_NUMBER", how="left")
    table["reported_value"] = table["reported_value"].fillna(0.0)
    table["position_count"] = table["position_count"].fillna(0).astype(int)
    table = table.rename(columns={
        "FILINGMANAGER_NAME": "manager_name",
        "CIK": "cik",
        "PERIODOFREPORT": "report_period",
        "ACCESSION_NUMBER": "accession_number",
    })
    return table[[
        "manager_name", "cik", "accession_number", "FILING_DATE", "report_period",
        "reported_value", "position_count",
    ]]


def _prepend_guaranteed_candidates(
    ranked: pd.DataFrame, eligible: pd.DataFrame, guaranteed_ciks: set[str] | None,
) -> pd.DataFrame:
    """Section 8 sanity-check support: if a representative investor
    (`config.REPRESENTATIVE_INVESTORS`) clears the same eligibility bar
    (>=min_positions long positions) as everyone else, put it at the FRONT
    of the ranked candidate list so it isn't crowded out by the
    mapping/price-coverage walk-down before reaching `num_portfolios`
    selections.

    This does NOT feed these CIKs into the clustering as labels or give
    them different metrics/thresholds than any other candidate — it only
    affects which rows enter the candidate pool in the first place, so
    Section 8's "did this well-known filer end up in a cluster, and why"
    check has something to actually report instead of silently landing on
    zero matches under random/AUM-stratified sampling (as it did on the
    2026-03-31 aum_stratified run: all 7 configured representative
    investors existed in the broader universe but happened not to be drawn
    into the random 100-portfolio sample).
    """
    if not guaranteed_ciks:
        return ranked
    already = set(ranked["cik"])
    extra = eligible[eligible["cik"].isin(guaranteed_ciks) & ~eligible["cik"].isin(already)]
    if extra.empty:
        return ranked
    return pd.concat([extra, ranked], ignore_index=True)


def select_top_value_candidates(
    filer_table: pd.DataFrame, n: int, min_positions: int,
    guaranteed_ciks: set[str] | None = None,
) -> pd.DataFrame:
    eligible = filer_table[filer_table["position_count"] >= min_positions]
    ranked = eligible.sort_values("reported_value", ascending=False).head(n).reset_index(drop=True)
    return _prepend_guaranteed_candidates(ranked, eligible, guaranteed_ciks)


def select_aum_stratified_candidates(
    filer_table: pd.DataFrame, n: int, min_positions: int, n_bins: int = 5, seed: int = config.RANDOM_SEED,
    guaranteed_ciks: set[str] | None = None,
) -> pd.DataFrame:
    """Alternative selection (Section 2 fallback): rather than always taking
    the globally-largest filers (dominated by custodian/broker-dealer
    aggregators — see report.md for the diagnostic that motivated this),
    bucket eligible filers into AUM quantile bins and sample proportionally
    from each bin, so mid-size and smaller managers are represented too.
    """
    eligible = filer_table[filer_table["position_count"] >= min_positions].copy()
    if eligible.empty:
        return eligible
    try:
        eligible["aum_bin"] = pd.qcut(eligible["reported_value"], q=n_bins, duplicates="drop")
    except ValueError:
        eligible["aum_bin"] = 0
    rng_state = seed
    per_bin = max(1, n // eligible["aum_bin"].nunique())
    parts = []
    for _, group in eligible.groupby("aum_bin", observed=True):
        take = group.sort_values("reported_value", ascending=False).sample(
            n=min(per_bin, len(group)), random_state=rng_state
        )
        parts.append(take)
    sampled = pd.concat(parts).sort_values("reported_value", ascending=False)
    if len(sampled) < n:
        remaining = eligible.drop(sampled.index).sort_values("reported_value", ascending=False)
        sampled = pd.concat([sampled, remaining.head(n - len(sampled))])
    sampled = sampled.head(n).reset_index(drop=True)
    return _prepend_guaranteed_candidates(sampled, eligible, guaranteed_ciks)
