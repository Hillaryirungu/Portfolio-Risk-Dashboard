"""
Unit tests for src/risk_metrics.py

Run with:
    python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import risk_metrics as rm


@pytest.fixture
def flat_returns():
    """Zero-volatility series: every metric should degrade gracefully."""
    return pd.Series([0.0] * 100)


@pytest.fixture
def known_returns():
    """A hand-checkable small series."""
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.0, 0.015, -0.005])


def test_portfolio_returns_weights_normalize():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, -0.01]})
    # weights sum to 2, not 1 -- should be re-normalized internally
    result = rm.portfolio_returns(df, {"A": 1.0, "B": 1.0})
    expected = df.mean(axis=1)
    pd.testing.assert_series_equal(result, expected.rename("portfolio"))


def test_portfolio_returns_missing_ticker_ignored():
    df = pd.DataFrame({"A": [0.01, 0.02]})
    result = rm.portfolio_returns(df, {"A": 0.5, "GHOST": 0.5})
    pd.testing.assert_series_equal(result, df["A"].rename("portfolio"))


def test_portfolio_returns_raises_if_no_overlap():
    df = pd.DataFrame({"A": [0.01, 0.02]})
    with pytest.raises(ValueError):
        rm.portfolio_returns(df, {"GHOST": 1.0})


def test_annualized_return_compounding():
    # 1% daily for 252 days should compound to > 252% simple, verify formula matches manual calc
    r = pd.Series([0.01] * 252)
    expected = (1.01) ** 252 - 1
    assert rm.annualized_return(r, periods_per_year=252) == pytest.approx(expected, rel=1e-9)


def test_annualized_volatility_scales_with_sqrt_time():
    r = pd.Series(np.random.RandomState(0).normal(0, 0.01, 1000))
    vol = rm.annualized_volatility(r)
    assert vol == pytest.approx(r.std(ddof=1) * np.sqrt(252), rel=1e-9)


def test_sharpe_ratio_zero_vol_returns_nan(flat_returns):
    assert np.isnan(rm.sharpe_ratio(flat_returns))


def test_sortino_only_penalizes_downside():
    upside_only = pd.Series([0.01, 0.02, 0.0, 0.015, 0.03])
    assert np.isnan(rm.sortino_ratio(upside_only))  # no downside deviation -> nan


def test_max_drawdown_simple_case():
    # Wealth path: 1 -> 1.1 -> 0.99 -> 1.05  => drawdown from peak 1.1 to trough 0.99 = -10%
    r = pd.Series([0.10, -0.10, 0.0606])
    result = rm.max_drawdown(r)
    assert result["max_drawdown"] == pytest.approx(-0.10, abs=1e-3)


def test_drawdown_series_never_positive(known_returns):
    dd = rm.drawdown_series(known_returns)
    assert (dd <= 1e-12).all()


def test_historical_var_matches_percentile(known_returns):
    var_95 = rm.historical_var(known_returns, confidence=0.95)
    manual = -np.percentile(known_returns, 5)
    assert var_95 == pytest.approx(manual)


def test_conditional_var_ge_var(known_returns):
    var = rm.historical_var(known_returns, confidence=0.8)
    cvar = rm.conditional_var(known_returns, confidence=0.8)
    # CVaR (average of tail losses) should be at least as large as the VaR threshold
    assert cvar >= var - 1e-9


def test_beta_of_series_with_itself_is_one():
    r = pd.Series(np.random.RandomState(1).normal(0, 0.01, 500))
    assert rm.beta(r, r) == pytest.approx(1.0, rel=1e-9)


def test_beta_uncorrelated_near_zero():
    rng = np.random.RandomState(2)
    a = pd.Series(rng.normal(0, 0.01, 5000))
    b = pd.Series(rng.normal(0, 0.01, 5000))
    assert abs(rm.beta(a, b)) < 0.1


def test_correlation_matrix_diagonal_is_one():
    df = pd.DataFrame(np.random.RandomState(3).normal(0, 0.01, (200, 3)), columns=["A", "B", "C"])
    corr = rm.correlation_matrix(df)
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_summary_table_shape():
    df = pd.DataFrame(np.random.RandomState(4).normal(0.0005, 0.01, (300, 3)), columns=["A", "B", "SPY"])
    table = rm.summary_table(df, benchmark_col="SPY")
    assert list(table.index) == ["A", "B", "SPY"]
    assert "Sharpe" in table.columns
    assert "Beta" in table.columns
    assert np.isnan(table.loc["SPY", "Beta"])  # benchmark has no beta vs itself
