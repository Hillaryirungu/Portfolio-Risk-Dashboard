"""
data_loader.py
---------------
ETL layer for the Portfolio Risk Dashboard.

Responsibilities:
    1. Pull raw adjusted-close price data for a list of tickers (via yfinance).
    2. Cache raw pulls to disk as CSV so the app is fast and works offline/demo mode.
    3. Clean the data (align dates, forward-fill small gaps, drop tickers with
       too much missing history) and hand back a tidy wide DataFrame of prices.
    4. Derive daily and cumulative returns.

Design notes:
    - Every function is pure / side-effect-explicit: network + disk I/O only
      happens in `fetch_prices`; everything downstream is deterministic and
      easy to unit test.
    - If `yfinance` is not installed or the network call fails (e.g. running
      in an offline sandbox), we fall back to a bundled synthetic sample
      dataset in `data/sample_prices.csv` so the dashboard is always runnable.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_PATH = os.path.join(DATA_DIR, "price_cache.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_prices.csv")


def _try_yfinance_download(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Attempt a live pull via yfinance. Returns None on any failure so the
    caller can gracefully fall back to cached/sample data instead of crashing
    the app (e.g. no internet, rate-limited, bad ticker)."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            prices = pd.concat(
                {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)},
                axis=1,
            )
        else:
            # Single ticker: yfinance returns a flat frame.
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        prices.index = pd.to_datetime(prices.index)
        return prices.sort_index()
    except Exception:
        return None


def fetch_prices(
    tickers: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Return a wide DataFrame of adjusted close prices, indexed by date,
    one column per ticker.

    Order of preference:
        1. Live yfinance pull (if package + network available).
        2. Local cache file (previous successful pull), filtered to tickers.
        3. Bundled synthetic sample dataset (always available, offline-safe).
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    end = end or date.today().isoformat()
    start = start or (date.today() - timedelta(days=5 * 365)).isoformat()

    live = _try_yfinance_download(tickers, start, end)
    if live is not None and not live.empty:
        os.makedirs(DATA_DIR, exist_ok=True)
        live.to_csv(CACHE_PATH)
        return _clean_prices(live)

    if use_cache and os.path.exists(CACHE_PATH):
        cached = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        available = [t for t in tickers if t in cached.columns]
        if available:
            return _clean_prices(cached[available])

    # Final fallback: synthetic sample data so the app always runs (demo mode).
    sample = pd.read_csv(SAMPLE_PATH, index_col=0, parse_dates=True)
    available = [t for t in tickers if t in sample.columns]
    cols = available if available else list(sample.columns)
    return _clean_prices(sample[cols])


def _clean_prices(prices: pd.DataFrame, max_missing_frac: float = 0.15) -> pd.DataFrame:
    """Align calendar, forward-fill short gaps, drop columns that are too
    sparse to trust, and drop any leading/trailing all-NaN rows."""
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    missing_frac = prices.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    prices = prices[keep_cols]

    prices = prices.ffill(limit=5)
    prices = prices.dropna(how="all")
    prices = prices.dropna(axis=0, how="any")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns."""
    return prices.pct_change().dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns -- additive over time, used for some risk calcs."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def compute_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative growth-of-$1 series from simple returns."""
    return (1 + returns).cumprod() - 1
