"""Portfolio construction + the four risk metrics (Sections 4-5).

Pure functions over pandas Series/DataFrames — no I/O — so they're directly
unit-testable without a network connection or the SEC/yfinance stack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sec13f import config


def compute_weights(values: pd.Series) -> pd.Series:
    """Section 4: w_i = Value_i / sum_j Value_j."""
    total = values.sum()
    if total <= 0:
        raise ValueError("Cannot compute weights: total portfolio value is <= 0")
    return values / total


def renormalize_weights(weights: pd.Series) -> pd.Series:
    """After dropping tickers with no/insufficient price history, rescale
    the remaining weights so they sum back to 1 (Section 4: "남은 종목의
    비중을 다시 합이 1이 되도록 정규화").
    """
    total = weights.sum()
    if total <= 0:
        raise ValueError("Cannot renormalize: remaining weight is <= 0")
    return weights / total


def compute_portfolio_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Section 4: R_p,t = sum_i w_i * R_i,t.

    `weights` index and `returns` columns must both be tickers; only the
    tickers present in both are used (weights are NOT renormalized here —
    call renormalize_weights() beforehand if you dropped any tickers from
    `returns`).
    """
    common = weights.index.intersection(returns.columns)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between weights and returns")
    aligned_weights = weights.loc[common]
    aligned_returns = returns[common].fillna(0.0)
    return aligned_returns.mul(aligned_weights, axis=1).sum(axis=1)


def volatility(portfolio_returns: pd.Series, trading_days: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """Section 5: Volatility = std(R_p) * sqrt(252), annualized."""
    if len(portfolio_returns) < 2:
        raise ValueError("Need at least 2 return observations to compute volatility")
    return float(portfolio_returns.std(ddof=1) * np.sqrt(trading_days))


def max_drawdown(portfolio_returns: pd.Series) -> float:
    """Section 5: MDD = max_t (1 - V_t / max_{s<=t} V_s), stored as a
    positive magnitude (0 = no drawdown, 1 = total loss).
    """
    cumulative = (1.0 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = 1.0 - cumulative / running_max
    return float(drawdown.max())


def beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
    """Section 5: Beta = Cov(R_p, R_m) / Var(R_m), against SPY."""
    aligned = pd.concat(
        [portfolio_returns.rename("p"), market_returns.rename("m")], axis=1
    ).dropna()
    if len(aligned) < 2:
        raise ValueError("Need at least 2 overlapping observations to compute beta")
    cov = aligned["p"].cov(aligned["m"])
    var = aligned["m"].var(ddof=1)
    if var == 0:
        raise ValueError("Market return variance is 0; beta is undefined")
    return float(cov / var)


def sector_hhi(weights: pd.Series, sector_map: dict[str, str]) -> tuple[float, dict]:
    """Section 5: HHI = sum_s w_s^2 over SECTOR-level weights (not
    per-ticker weights — do not confuse with a single-issuer concentration
    index).

    Tickers with an unresolved sector (sector_map value is falsy /
    "Unknown") are pooled into a single "Unknown" bucket rather than
    dropped, and its total weight is returned alongside the HHI so callers
    can record how much of the portfolio's HHI rests on an "Unknown"
    catch-all (Section 5: "업종을 확인할 수 없는 종목의 비중과 처리방법을
    기록한다").
    """
    sectors = weights.index.map(lambda t: sector_map.get(t) or "Unknown")
    sector_weights = weights.groupby(sectors).sum()
    hhi = float((sector_weights ** 2).sum())
    unknown_weight = float(sector_weights.get("Unknown", 0.0))
    detail = {
        "sector_weights": sector_weights.to_dict(),
        "unknown_sector_weight": unknown_weight,
        "n_sectors": int((sector_weights > 0).sum()),
    }
    return hhi, detail


def compute_portfolio_metrics(
    values: pd.Series,
    returns: pd.DataFrame,
    market_returns: pd.Series,
    sector_map: dict[str, str],
) -> dict:
    """End-to-end Section 4-5 pipeline for one portfolio:

    values         — Series indexed by ticker, reported market value (only
                      tickers that were successfully mapped+priced should
                      already be here; this function does the final
                      renormalization over whatever's passed in).
    returns         — DataFrame of daily simple returns, columns = tickers
                      (superset is fine; only `values.index` columns used).
    market_returns  — SPY daily simple returns, same date index space.
    sector_map      — ticker -> sector name (or None/"" for unresolved).
    """
    weights = compute_weights(values)
    available = [t for t in weights.index if t in returns.columns]
    weights = renormalize_weights(weights.loc[available])

    port_returns = compute_portfolio_returns(weights, returns)
    hhi, hhi_detail = sector_hhi(weights, sector_map)

    return {
        "weights": weights,
        "portfolio_returns": port_returns,
        "volatility": volatility(port_returns),
        "mdd": max_drawdown(port_returns),
        "beta": beta(port_returns, market_returns),
        "sector_hhi": hhi,
        "hhi_detail": hhi_detail,
    }
