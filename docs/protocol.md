# Volatility Forecasting and Volatility-Targeted Trading: walk-forward protocol

A standard specification for a **standalone** trading-research project. It builds on your STAT 5140 GARCH work and turns it into a question a trading desk would ask:

> Do better volatility forecasts produce a better risk-managed strategy, after costs, on data the model never saw?

The project has two parts:

- **Part A** forecasts daily volatility and scores the forecasts.
- **Part B** uses those forecasts to size positions (volatility targeting) and backtests the strategy honestly.

The protocol is built so the result is believable whether or not the strategy "works". A careful null result is still a strong project; an overfit 2.5 Sharpe is a liability in an interview.

Reference ranges, not targets: volatility targeting on US equities typically lowers volatility and drawdowns, and changes risk-adjusted return only modestly (Moreira and Muir, 2017; Harvey et al., 2018). Report what your test period produces.

---

## 1. Data

| Item | Setting |
|---|---|
| Primary asset | SPY (S&P 500 ETF), daily open, high, low, close, adjusted close, volume |
| Generalization set | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY (sector ETFs), TLT, GLD |
| Risk-free rate | 3-month T-bill (FRED series `DTB3`), converted to a daily rate |
| Sample | First trading day of 2005 to last trading day of 2025 |
| Source | `yfinance` for prices, `pandas-datareader` for FRED; save raw downloads to `data/raw/` with the download date and a SHA-256 hash of each file |

**Returns.** Use daily log returns from adjusted close: `r_t = ln(P_t / P_{t-1})`. Use unadjusted OHLC only for the range-based variance below.

**Realized variance proxy (the forecasting target).** Use Garman–Klass plus the overnight gap, since daily close-to-close squared returns are too noisy to score forecasts well:

```
RV_t = (ln(O_t / C_{t-1}))^2 + 0.5 (ln(H_t / L_t))^2 - (2 ln 2 - 1) (ln(C_t / O_t))^2
```

**Splits (fixed before any modelling):**

| Period | Dates | Use |
|---|---|---|
| Burn-in | 2005–2009 | First model fits only; never scored |
| Development | 2010–2015 | All model and parameter choices |
| Holdout test | 2016–2025 | Touched **once**, for the configurations chosen on development |

The development period includes calm and stressed markets (2011, 2015). The holdout includes 2018, 2020, and 2022. Record the split in `config.yaml` and commit it before running anything.

---

## 2. Part A: volatility forecasting

**Forecast timing.** At the close of day *t*, using data up to and including day *t* only, forecast the variance of day *t+1*. Also produce a 5-day-ahead forecast (sum of daily forecasts) as a secondary horizon.

**Models:**

| Run | Model | Notes |
|---|---|---|
| `hist21` | 21-day rolling variance of returns | Naive baseline |
| `ewma` | EWMA, λ = 0.94 (RiskMetrics) | No fitting |
| `garch_n` | GARCH(1,1), normal errors | Constant mean |
| `garch_t` | GARCH(1,1), Student-t errors | |
| `gjr_t` | GJR-GARCH(1,1,1), Student-t | Leverage effect |
| `egarch_t` | EGARCH(1,1), Student-t | |
| `har_rv` | HAR-RV on log RV (daily, weekly, monthly lags) | Fit by OLS on the RV proxy |

**Walk-forward fitting.** Use an expanding window. Refit parameters on the first trading day of each month, then produce one-day forecasts for every day of that month with the frozen parameters, updating only the filtered variance. This mirrors how a desk would run it, and keeps runtime reasonable.

**Scoring (development period first, holdout once):**

- **QLIKE** (primary): `L = RV/h - ln(RV/h) - 1`, where `h` is the forecast. It is robust to noise in the RV proxy (Patton, 2011).
- **MSE** on variance (secondary).
- **Mincer–Zarnowitz regression** `RV_{t+1} = a + b h_{t+1} + e`, testing `a = 0, b = 1`. This checks forecast bias.
- **Diebold–Mariano tests** of each model against `hist21` and against the best model, with Newey–West standard errors.
- **Model Confidence Set** (Hansen, Lunde and Nason, 2011) at 90%, as the honest summary of "which models are indistinguishable".

### Reference walk-forward GARCH code

```python
import numpy as np
import pandas as pd
from arch import arch_model


def walk_forward_garch(r, start, vol="GARCH", o=0, dist="t"):
    """r: daily log returns (pd.Series). Returns one-day-ahead variance forecasts for dates >= start."""
    r100 = r * 100  # arch is better conditioned on percentage returns
    out = {}
    dates = r.index[r.index >= start]
    for month, month_dates in dates.to_series().groupby(dates.to_period("M")):
        fit_end = r.index[r.index.get_loc(month_dates.iloc[0]) - 1]   # last close before the month
        am = arch_model(r100.loc[:fit_end], mean="Constant", vol=vol, p=1, o=o, q=1, dist=dist)
        res = am.fit(disp="off")
        params = res.params
        for d in month_dates:
            prev = r.index[r.index.get_loc(d) - 1]                   # forecast for d uses data up to prev
            fixed = arch_model(r100.loc[:prev], mean="Constant", vol=vol, p=1, o=o, q=1, dist=dist).fix(params)
            out[d] = fixed.forecast(horizon=1, reindex=False).variance.iloc[-1, 0] / 1e4
    return pd.Series(out, name=f"{vol.lower()}_{dist}")
```

---

## 3. Part B: volatility-targeted strategy

**Position sizing.** At the close of day *t*, set the weight in the risky asset for day *t+1*:

```
w_{t+1} = min( L_max , sigma_target / sigma_hat_{t+1} )
sigma_hat_{t+1} = sqrt(252 * h_{t+1})          # annualized forecast volatility
```

| Setting | Canonical value |
|---|---|
| Target volatility `sigma_target` | 10% annualized |
| Leverage cap `L_max` | 1.5 |
| Execution | **One-day lag**: the weight decided at close *t* is traded at close *t+1* and earns returns from *t+2* |
| Rebalancing | Daily, with a no-trade band: skip trades where the weight change is under 0.05 |
| Cash | Unallocated weight earns the T-bill rate; borrowed weight (above 1.0) pays T-bill + 0.5% |
| Transaction costs | 5 bps of traded notional (sensitivity at 1 and 10 bps) |

**Daily strategy return:**

```
R_{t} = w_t * r_t + (1 - w_t) * rf_t - spread * max(w_t - 1, 0) - c * |w_t - w_{t-1}|
```


**Benchmarks:**

- Buy-and-hold SPY.
- Buy-and-hold SPY **scaled to the same realized volatility** as the strategy over the test period (an ex-post comparison that isolates timing from simple de-leveraging; label it clearly as ex-post).
- Volatility targeting with the naive `hist21` forecast. This is the key comparison: does a better forecast earn its complexity?

**Metrics (per run):** annualized return, volatility, Sharpe and Sortino (excess over T-bill), maximum drawdown, Calmar ratio, average leverage, annual turnover, total cost paid, and realized volatility versus target (how well the strategy actually hit 10%).

---

## 4. Model selection and evaluation

1. Make every choice (forecast model, target, cap, band, rebalance frequency) on the **development period only**, using QLIKE for Part A and Sharpe plus realized-vs-target volatility for Part B.
2. Log every configuration you run, including failed and discarded ones, to `runs/log.csv`. The count of configurations tried is needed for step 5.
3. Freeze the selected configuration in `config_final.yaml` and commit it before opening the holdout.
4. Run the holdout once. Do not change anything after seeing it. If you do change something, report both results and say why.
5. Report statistical uncertainty:
   - **Sharpe difference** between the selected strategy and each benchmark, with a 95% confidence interval from a stationary block bootstrap (Politis and Romano, 1994), mean block length 20 days, 10,000 resamples, seed 2026.
   - **Deflated Sharpe ratio** (Bailey and López de Prado, 2014), using the number of configurations in `runs/log.csv`.
6. Run the frozen configuration, without retuning, on the generalization set (sector ETFs, TLT, GLD) over the holdout period, and report one row per asset.

---

## 5. Experiment plan

Every row runs on the same splits. Part B rows change one factor relative to `vt_best`.

**Part A: forecasting** (rows from Section 2, scored on development, then the holdout once).

**Part B: strategy**

| Run | Compared with | Changed factor |
|---|---|---|
| `bh` | (benchmark) | Buy-and-hold SPY |
| `vt_hist21` | `vt_best` | Naive 21-day forecast |
| `vt_ewma` | `vt_best` | EWMA forecast |
| `vt_best` | (reference) | Best development forecast, canonical settings |
| `vt_best_nolev` | `vt_best` | Leverage cap 1.0 (no borrowing) |
| `vt_best_tgt15` | `vt_best` | Target volatility 15% |
| `vt_best_weekly` | `vt_best` | Weekly rebalancing instead of daily-with-band |
| `vt_best_cost1` / `vt_best_cost10` | `vt_best` | Transaction costs 1 bp and 10 bps |
| `vt_best_sameclose` | `vt_best` | No execution lag (trade at the same close the forecast uses). **Optimistic, look-ahead-prone sensitivity only**; shows how much the lag costs |

**Generalization**

| Run | Description |
|---|---|
| `vt_final_<ticker>` | Frozen configuration on each generalization asset, holdout period |

---

## 6. Common mistakes to avoid

- **Look-ahead bias:** using day *t+1* data in a day *t* forecast, fitting on the full sample, or computing normalization or thresholds over the whole period.
- Trading at the same close the signal was computed from in the main result.
- Using unadjusted prices for returns (splits and dividends create false jumps), or adjusted prices for the range-based variance.
- Ignoring costs, borrowing costs, or the cash return.
- Scoring forecasts with MSE alone; it is dominated by a few crisis days.
- Reporting only the best configuration without the number of configurations tried.
- Re-opening the holdout to "fix" something and reporting only the fixed number.
- Compounding log returns of a leveraged portfolio as if they were simple returns.

---

## 7. Project setup

- **Smoke test first:** run `ewma` and `vt_ewma` on 2010 only, and check by hand that a single day's weight, cost, and return match the formulas above. It is a correctness check, not a result.
- **Unit tests** (`tests/`): a test that a forecast for day *t+1* is unchanged when you append fake data after day *t* (the look-ahead test); a test of the return formula on a hand-built three-day example; a test that the QLIKE of a perfect forecast is 0.
- **Determinism:** pin package versions; set the bootstrap seed; keep raw data hashes in the README.
- **Repository layout (standalone):**

```
vol-forecasting-trading/
  README.md              results tables, protocol summary, how to reproduce
  docs/protocol.md       this document
  config.yaml            splits, assets, canonical settings (committed before modelling)
  config_final.yaml      frozen selection (committed before the holdout)
  data.py                download, hashing, returns, RV proxy
  forecast.py            the seven forecasting models, walk-forward engine
  evaluate.py            QLIKE, MSE, Mincer–Zarnowitz, Diebold–Mariano, MCS
  backtest.py            sizing, execution lag, costs, cash, metrics
  stats.py               block bootstrap, deflated Sharpe
  run.py                 one experiment row per invocation
  tests/                 look-ahead, return-formula, and metric tests
  runs/                  per-run outputs and log.csv of every configuration tried
  assets/                forecast plots, equity curves, drawdown charts
  requirements.txt       pinned versions (pandas, numpy, arch, statsmodels, yfinance, pandas-datareader)
```

- **README results tables:** Part A (QLIKE and MSE per model, development and holdout, MCS membership); Part B (metrics per row, holdout); Sharpe-difference confidence intervals; deflated Sharpe; generalization table.

---
