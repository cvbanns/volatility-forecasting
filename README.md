# Volatility Forecasting and Risk Targeting

A reproducible Python study of whether better daily volatility forecasts improve a risk-targeted trading strategy after transaction costs. The project compares seven variance models on SPY, selects the model and trading rules on 2010–2015 data, then evaluates the frozen strategy on a 2016–2025 holdout and 11 additional ETFs.


## Results

GJR-GARCH with Student-t errors won the development forecast comparison. Its one-day holdout QLIKE was **0.434**, versus **0.575** for the 21-day rolling variance baseline. With identical trading rules and 5 bps transaction costs, the selected strategy's net Sharpe was **0.750**, versus **0.817** for the rolling baseline. The paired 95% confidence interval for the Sharpe difference was **−0.187 to +0.050**.

| SPY holdout strategy | Annualized volatility | Max drawdown | Net Sharpe |
| --- | ---: | ---: | ---: |
| Buy and hold | 18.01% | −33.72% | 0.732 |
| 21-day rolling forecast, matched rules | 10.26% | −12.51% | 0.817 |
| Selected GJR-GARCH strategy | 9.62% | −12.09% | 0.750 |

The selected strategy came within 0.38 percentage points of its 10% volatility target. It reduced drawdown relative to buy and hold, while the Sharpe comparison with the rolling baseline remained inconclusive.

[Full results and methodology](docs/results.md) · [Drawdown chart](assets/drawdown.png) · [ETF results](runs/holdout/generalization.csv)

## How it works

1. Use adjusted prices for returns and daily OHLC to estimate realized variance, including the overnight gap.
2. Compare rolling variance, EWMA, four GARCH variants, and HAR-RV using expanding windows and monthly refits. Score one- and five-day variance forecasts.
3. Select the strategy on development data. The selected rules target 10% annualized volatility, cap exposure at 100%, delay execution by one close, and skip changes below five percentage points of portfolio weight.
4. Evaluate net performance on the frozen holdout and apply the same rules to nine sector ETFs, TLT, and GLD. Compare Sharpe with paired stationary bootstrap intervals.

The study uses 2005–2009 for initialization, 2010–2015 for development, and 2016–2025 for holdout. See the [research protocol](docs/protocol.md), [registered settings](config.yaml), and [frozen selection](config_final.yaml).

## Quick start

Requires Git and Python 3.12. In a clone of this repository, create and activate a virtual environment, then install the pinned dependencies and run the tests.

```sh
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

The default branch contains the completed study. For an independent reproduction, create a clean checkout at the implementation tag:

```sh
git worktree add -b reproduce-study ../volatility-reproduction reproduction-base
cd ../volatility-reproduction
```

Use the Python environment created above. Copy the original raw CSVs into `data/raw/` for exact replication. The [snapshot manifest](data/manifest.json) lists each source and file hash. To run with fresh vendor data, first move the existing manifest aside, then install `setuptools==84.2.1` and run `python run.py download`. Vendor revisions can change results.

```sh
python run.py smoke
python run.py development
python run.py freeze
git add config_final.yaml
git commit -m "Freeze development selection before holdout"
python run.py holdout
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `run.py` | Download, smoke check, development, freeze, and holdout commands |
| `data.py`, `forecast.py` | Vendor snapshots, variance targets, and walk-forward forecasts |
| `evaluate.py`, `backtest.py`, `stats.py` | Forecast tests, trading simulation, and Sharpe inference |
| `report.py`, `assets/` | Plotting code and study charts |
| `tests/` | Forecast causality, reference checks, and trade accounting |
| `runs/` | Run ledger, daily outputs, and summary tables |

## Data and provenance

Prices come from Yahoo Finance via yfinance. The risk-free proxy is FRED's DTB3 series via pandas-datareader. Raw vendor downloads are kept outside Git. The `study-audit` branch preserves the original protocol, freeze, and holdout commits. The default branch organizes the same code and results into focused commits.

References and implementation details are in the [study report](docs/results.md).
