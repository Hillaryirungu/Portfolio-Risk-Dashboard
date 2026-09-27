"""
risk_metrics.py
----------------
Core quantitative risk engine for the Portfolio Risk Dashboard.

All functions operate on plain pandas Series/DataFrames of *simple daily
returns* (not prices) unless stated otherwise, so they compose cleanly and
are trivial to unit test without any I/O.

Conventions:
    - `returns`      : pd.Series of daily simple returns for one asset/portfolio
    - `returns_df`   : pd.DataFrame of daily simple returns, one column per asset
    - `weights`      : dict[str, float] or pd.Series of portfolio weights (sum to 1)
    - Annualization assumes 252 trading days/year unless overridden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Portfolio construction
# ----------------------------------------------------------------------

def portfolio_returns(returns_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Combine asset returns into a single weighted portfolio return series.
    Weights are re-normalized to sum to 1 over the tickers actually present."""
    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers:
        raise ValueError("None of the provided weight tickers are in returns_df.")
    w = pd.Series({t: weights[t] for t in tickers}, dtype=float)
    w = w / w.sum()
    return returns_df[tickers].mul(w, axis=1).sum(axis=1).rename("portfolio")


# ----------------------------------------------------------------------
# Return / volatility metrics
# ----------------------------------------------------------------------

def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    growth = (1 + returns).prod()
    n = len(returns)
    if n == 0 or growth <= 0:
        return float("nan")
    return growth ** (periods_per_year / n) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio. risk_free_rate is an ANNUAL rate."""
    excess = returns - risk_free_rate / periods_per_year
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return (excess.mean() / vol) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Like Sharpe, but only penalizes downside deviation."""
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized return divided by max drawdown magnitude."""
    ar = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)["max_drawdown"]
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return ar / abs(mdd)


# ----------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown (as a negative fraction) from the running peak."""
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1


def max_drawdown(returns: pd.Series) -> dict:
    """Returns the max drawdown magnitude plus the peak/trough/recovery dates."""
    dd = drawdown_series(returns)
    trough_date = dd.idxmin()
    max_dd = dd.min()
    peak_date = (1 + returns).cumprod().loc[:trough_date].idxmax()

    recovery_date = None
    post_trough = dd.loc[trough_date:]
    recovered = post_trough[post_trough >= 0]
    if not recovered.empty:
        recovery_date = recovered.index[0]

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }


# ----------------------------------------------------------------------
# Value at Risk / Conditional VaR
# ----------------------------------------------------------------------

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk as a positive loss fraction.
    E.g. VaR_95 = 0.03 means: on the worst 5% of days, you lose >= 3%."""
    alpha = 1 - confidence
    return -np.percentile(returns.dropna(), alpha * 100)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Variance-covariance (Gaussian) VaR, using sample mean/std."""
    from scipy.stats import norm  # local import: optional dependency

    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall: average loss BEYOND the VaR
    threshold, as a positive loss fraction."""
    var = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    if tail.empty:
        return var
    return -tail.mean()


# ----------------------------------------------------------------------
# Relative risk: beta, correlation, covariance
# ----------------------------------------------------------------------

def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var = np.var(aligned.iloc[:, 1], ddof=1)
    return cov / var if var != 0 else float("nan")


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()


def covariance_matrix(returns_df: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Annualized covariance matrix."""
    return returns_df.cov() * periods_per_year


# ----------------------------------------------------------------------
# One-shot summary table (used by the dashboard)
# ----------------------------------------------------------------------

def summary_table(
    returns_df: pd.DataFrame,
    benchmark_col: str | None = None,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Build a tidy per-asset risk/return summary table -- the backbone of
    the dashboard's main metrics view."""
    rows = {}
    bench = returns_df[benchmark_col] if benchmark_col in (returns_df.columns if benchmark_col else []) else None

    for col in returns_df.columns:
        r = returns_df[col].dropna()
        mdd = max_drawdown(r)
        rows[col] = {
            "Ann. Return": annualized_return(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free_rate),
            "Sortino": sortino_ratio(r, risk_free_rate),
            "Calmar": calmar_ratio(r),
            "Max Drawdown": mdd["max_drawdown"],
            f"VaR {int(confidence * 100)}% (1d)": historical_var(r, confidence),
            f"CVaR {int(confidence * 100)}% (1d)": conditional_var(r, confidence),
            "Beta": beta(r, bench) if bench is not None and col != benchmark_col else np.nan,
        }
    return pd.DataFrame(rows).T
