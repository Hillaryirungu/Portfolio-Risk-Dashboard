# 📈 Portfolio Risk Dashboard

An end-to-end **data engineering + quantitative risk analytics** application:
pull market data, clean it, compute institutional-grade risk metrics, and
present it through a live, public, interactive dashboard.

**[Live demo →](#deployment)** *(fill in your Streamlit Cloud URL after deploying)*

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Why this project

Business and investment decisions run on data pipelines that are **correct**,
**explainable**, and **usable by non-technical stakeholders**. This project
demonstrates that full chain:

| Layer | What it shows |
|---|---|
| **ETL** (`src/data_loader.py`) | Ingest, clean, and cache real market data with graceful fallback (no single point of failure — the app never crashes even if the API is down) |
| **Quant modeling** (`src/risk_metrics.py`) | Volatility, Sharpe/Sortino/Calmar, drawdown, historical & parametric VaR, Conditional VaR, beta, correlation |
| **Testing** (`tests/`) | Unit tests with hand-verifiable expected values, not just smoke tests |
| **Delivery** (`app.py`) | A dashboard a portfolio manager (or a hiring manager) can actually use, no notebook required |

## Features

- 🔎 **Any tickers, any weights** — type in tickers, set custom or equal
  portfolio weights, pick a benchmark.
- 📊 **Cumulative return & drawdown charts** — see performance and pain
  side by side.
- 📉 **Value at Risk & Conditional VaR** — historical and parametric,
  adjustable confidence level.
- 🧮 **Full risk/return summary table** — Sharpe, Sortino, Calmar, beta,
  max drawdown, per asset and for the blended portfolio.
- 🔗 **Correlation heatmap** — spot diversification (or the lack of it).
- 🌐 **Works offline** — ships with a synthetic sample dataset so it
  always runs, even without an internet connection or API access.

## Architecture

```
portfolio-risk-dashboard/
├── app.py                    # Streamlit UI — the public-facing layer
├── src/
│   ├── data_loader.py        # ETL: fetch → cache → clean → return
│   └── risk_metrics.py       # Pure functions: the quant engine
├── data/
│   └── sample_prices.csv     # Offline-safe demo dataset
├── tests/
│   └── test_risk_metrics.py  # pytest unit tests
└── requirements.txt
```

The design deliberately separates **I/O** (`data_loader.py`) from **pure
computation** (`risk_metrics.py`) so the quant logic is trivially unit
tested without mocking a network call.

## Quickstart

```bash
git clone https://github.com/<your-username>/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Deployment

This app is designed to be deployed **for free, in under 5 minutes**, so
it's genuinely public — a link you can put on your CV or LinkedIn:

1. Push this repo to your own GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **"New app"**, select this repo and `app.py` as the entry point.
4. Deploy. You'll get a public URL like
   `https://your-app-name.streamlit.app`.

Paste that URL at the top of this README once it's live.

## Key risk metrics explained

- **Sharpe Ratio** — return earned per unit of total volatility, adjusted
  for the risk-free rate.
- **Sortino Ratio** — like Sharpe, but only penalizes *downside* volatility
  (upside swings aren't "risk").
- **Max Drawdown** — the largest peak-to-trough loss over the period; the
  number that best captures "how bad could it have felt to hold this."
- **Historical VaR (95%)** — the loss threshold you'd expect to exceed on
  the worst 5% of days, estimated directly from historical returns.
- **Conditional VaR (CVaR)** — the *average* loss on those worst days, not
  just the threshold — a more conservative tail-risk measure.
- **Beta** — sensitivity of an asset's returns to the benchmark's returns.

## Possible extensions

- Monte Carlo simulation for forward-looking VaR
- Factor model decomposition (Fama-French)
- PDF/email report generation on a schedule
- Multi-portfolio comparison view
- Docker container + CI (GitHub Actions running `pytest` on every push)

## Disclaimer

Built for educational/portfolio purposes. Not investment advice.

## License

MIT — see [LICENSE](LICENSE).
