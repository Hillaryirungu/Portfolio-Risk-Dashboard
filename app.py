"""
Portfolio Risk Dashboard
=========================
A public, interactive dashboard for exploring portfolio risk: volatility,
Sharpe/Sortino, drawdowns, Value-at-Risk, and correlation structure.

Run locally:
    streamlit run app.py

Deploy publicly (free):
    Push this repo to GitHub -> https://share.streamlit.io -> "New app"
    -> point at this file. No server management needed.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import fetch_prices, compute_returns, compute_cumulative_returns
from src import risk_metrics as rm

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="\U0001F4C8",
    layout="wide",
)

st.title("\U0001F4C8 Portfolio Risk Dashboard")
st.caption(
    "Data engineering + quantitative risk analytics on live (or offline-demo) "
    "market data. Built with pandas, NumPy, and Streamlit."
)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio setup")

    default_tickers = "AAPL, MSFT, GOOGL, JPM, XOM"
    tickers_input = st.text_input("Tickers (comma-separated)", value=default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    benchmark = st.text_input("Benchmark ticker", value="SPY").strip().upper()
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))

    lookback_years = st.slider("Lookback window (years)", 1, 5, 3)

    st.subheader("Weights")
    st.caption("Leave equal, or set custom weights (auto-normalized to 100%).")
    weight_mode = st.radio("Weighting", ["Equal weight", "Custom"], horizontal=True)

    weights = {}
    if weight_mode == "Equal weight":
        for t in tickers:
            weights[t] = 1.0
    else:
        for t in tickers:
            weights[t] = st.slider(f"{t} weight", 0.0, 1.0, round(1 / len(tickers), 2), 0.05)

    st.subheader("Risk parameters")
    risk_free_rate = st.number_input("Annual risk-free rate", value=0.03, step=0.005, format="%.3f")
    confidence = st.select_slider("VaR / CVaR confidence", options=[0.90, 0.95, 0.975, 0.99], value=0.95)

    st.divider()
    st.caption(
        "Data source: Yahoo Finance via `yfinance` when available; otherwise "
        "falls back to a cached pull or a bundled synthetic sample dataset "
        "so the app always runs, even fully offline."
    )

# ----------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------
if not tickers:
    st.warning("Enter at least one ticker in the sidebar to get started.")
    st.stop()

start_date = (pd.Timestamp.today() - pd.DateOffset(years=lookback_years)).date().isoformat()

with st.spinner("Fetching and cleaning price data..."):
    prices = fetch_prices(all_tickers, start=start_date)
    returns = compute_returns(prices)

missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.info(f"No usable data for: {', '.join(missing)} (dropped).")

available_tickers = [t for t in tickers if t in prices.columns]
if not available_tickers:
    st.error("None of the requested tickers have usable data. Try different symbols.")
    st.stop()

weights = {t: w for t, w in weights.items() if t in available_tickers}
port_returns = rm.portfolio_returns(returns, weights)

combined_returns = returns.copy()
combined_returns["Portfolio"] = port_returns

# ----------------------------------------------------------------------
# Headline metrics
# ----------------------------------------------------------------------
mdd = rm.max_drawdown(port_returns)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Ann. Return", f"{rm.annualized_return(port_returns):.1%}")
col2.metric("Ann. Volatility", f"{rm.annualized_volatility(port_returns):.1%}")
col3.metric("Sharpe Ratio", f"{rm.sharpe_ratio(port_returns, risk_free_rate):.2f}")
col4.metric("Max Drawdown", f"{mdd['max_drawdown']:.1%}")
col5.metric(
    f"VaR {int(confidence*100)}% (1-day)",
    f"{rm.historical_var(port_returns, confidence):.2%}",
)

st.divider()

# ----------------------------------------------------------------------
# Cumulative returns chart
# ----------------------------------------------------------------------
st.subheader("Cumulative Returns")
cum = compute_cumulative_returns(combined_returns[available_tickers + [benchmark, "Portfolio"]].dropna(how="all", axis=1))
fig_cum = px.line(cum, labels={"value": "Cumulative Return", "index": "Date", "variable": "Series"})
fig_cum.update_layout(yaxis_tickformat=".0%", legend_title_text="")
st.plotly_chart(fig_cum, use_container_width=True)

# ----------------------------------------------------------------------
# Drawdown chart
# ----------------------------------------------------------------------
st.subheader("Portfolio Drawdown")
dd = rm.drawdown_series(port_returns)
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown", line=dict(color="crimson")))
fig_dd.update_layout(yaxis_tickformat=".0%", showlegend=False, height=300)
st.plotly_chart(fig_dd, use_container_width=True)

# ----------------------------------------------------------------------
# Risk summary table
# ----------------------------------------------------------------------
st.subheader("Risk & Return Summary")
summary = rm.summary_table(
    combined_returns[available_tickers + [benchmark, "Portfolio"]].dropna(how="all", axis=1),
    benchmark_col=benchmark,
    risk_free_rate=risk_free_rate,
    confidence=confidence,
)
pct_cols = [c for c in summary.columns if c != "Sharpe" and c != "Sortino" and c != "Calmar" and c != "Beta"]
styled = summary.style.format({c: "{:.2%}" for c in pct_cols}).format(
    {c: "{:.2f}" for c in ["Sharpe", "Sortino", "Calmar", "Beta"]}
)
st.dataframe(styled, use_container_width=True)

# ----------------------------------------------------------------------
# Correlation heatmap
# ----------------------------------------------------------------------
st.subheader("Correlation Matrix")
corr = rm.correlation_matrix(returns[available_tickers + [benchmark]].dropna(how="all", axis=1))
fig_corr = px.imshow(
    corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------------------------------------------------
# Rolling volatility
# ----------------------------------------------------------------------
st.subheader("Rolling 30-Day Annualized Volatility")
rolling_vol = port_returns.rolling(30).std() * np.sqrt(252)
fig_vol = px.line(rolling_vol, labels={"value": "Ann. Volatility", "index": "Date"})
fig_vol.update_layout(yaxis_tickformat=".0%", showlegend=False)
st.plotly_chart(fig_vol, use_container_width=True)

st.divider()
st.caption(
    "Built as a portfolio project demonstrating an end-to-end data pipeline: "
    "ingestion -> cleaning -> quantitative risk modeling -> interactive visualization. "
    "Not investment advice."
)
