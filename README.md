# Volatility forecasting and volatility-targeted trading

The better variance forecast did not establish a trading advantage over the rolling variance baseline. On SPY in 2016–2025, the development-selected **vt_best_nolev** strategy realized **9.62% volatility** against a **10.00% target**. Its maximum drawdown was **-12.09%**, versus **-33.72%** for buy-and-hold. Its net Sharpe was **0.750**, versus **0.732** for buy-and-hold.

Against hist21 at the same strategy settings, the selected strategy changed Sharpe by **-0.066** (95% CI **[-0.187, +0.050]**). Against constant exposure matched ex post to its realized volatility, the change was **+0.019** (95% CI **[-0.297, +0.322]**).

The selected strategy turned over 7.60 times NAV per year versus 2.78 for hist21 at the same settings. At 5 bps per trade, the selected strategy incurred more fees under the protocol's cost model.

## Forecasts

GJR-GARCH with Student-t errors won development at both horizons. On the holdout, its one-day QLIKE was 0.434, versus 0.575 for hist21. HAR-RV had the lowest holdout QLIKE at both horizons. Both models were in the 90% holdout model confidence set at both horizons; the holdout ranking did not change the frozen strategy.

### One day

7 models, 1,510 development targets and 2,514 holdout targets.

| model    |   Dev QLIKE |   Dev MSE × 10⁸ | Dev MCS   |   Test QLIKE |   Test MSE × 10⁸ | Test MCS   |
|:---------|------------:|----------------:|:----------|-------------:|-----------------:|:-----------|
| hist21   |      0.5616 |          6.7912 | —         |       0.5748 |          19.8947 | —          |
| ewma     |      0.4714 |          6.5376 | —         |       0.5133 |          17.7989 | —          |
| garch_n  |      0.4284 |          6.3613 | Yes       |       0.4460 |          14.1862 | —          |
| garch_t  |      0.4270 |          6.3719 | —         |       0.4457 |          14.4055 | —          |
| gjr_t    |      0.3961 |          6.0497 | Yes       |       0.4336 |          13.5171 | Yes        |
| egarch_t |      0.4412 |          6.4942 | —         |       0.4545 |          14.4372 | —          |
| har_rv   |      0.4219 |          6.4310 | Yes       |       0.4158 |          15.3899 | Yes        |

### Five days

7 models, 1,506 development targets and 2,510 holdout targets.

| model    |   Dev QLIKE |   Dev MSE × 10⁸ | Dev MCS   |   Test QLIKE |   Test MSE × 10⁸ | Test MCS   |
|:---------|------------:|----------------:|:----------|-------------:|-----------------:|:-----------|
| hist21   |      0.4410 |         66.7494 | —         |       0.4564 |         346.1337 | —          |
| ewma     |      0.3421 |         61.1615 | —         |       0.3769 |         291.6076 | —          |
| garch_n  |      0.2847 |         58.6772 | Yes       |       0.3005 |         229.7020 | —          |
| garch_t  |      0.2846 |         60.2976 | Yes       |       0.3028 |         244.4856 | —          |
| gjr_t    |      0.2674 |         59.4458 | Yes       |       0.3052 |         230.3340 | Yes        |
| egarch_t |      0.2920 |         60.6998 | Yes       |       0.3098 |         228.5297 | —          |
| har_rv   |      0.2824 |         59.8127 | Yes       |       0.2715 |         250.0160 | Yes        |

The full forecast tables include Mincer–Zarnowitz intercepts, slopes and joint test p-values, plus MCS p-values. The DM tables report the loss difference, HAC test statistic, bandwidth and p-value against hist21 and the development winner.

- Development: [scores](runs/development/forecast_scores.csv), [DM tests](runs/development/dm.csv)
- Holdout: [scores](runs/holdout/forecast_scores.csv), [DM tests](runs/holdout/dm.csv)
- [Forecast plot](assets/forecast.png)

## SPY strategy holdout

| run                        | CAGR   | Vol.   | Target   | Vol. gap   |   Sharpe |   Sortino | Max DD   |   Calmar |
|:---------------------------|:-------|:-------|:---------|:-----------|---------:|----------:|:---------|---------:|
| bh                         | 14.75% | 18.01% | —        | —          |    0.732 |     1.025 | -33.72%  |    0.438 |
| vt_hist21                  | 10.75% | 11.37% | 10.00%   | +1.37 pp   |    0.760 |     1.022 | -15.28%  |    0.704 |
| vt_ewma                    | 10.17% | 10.62% | 10.00%   | +0.62 pp   |    0.757 |     1.017 | -15.09%  |    0.674 |
| vt_best                    | 9.36%  | 10.24% | 10.00%   | +0.24 pp   |    0.709 |     0.948 | -13.27%  |    0.705 |
| vt_best_nolev              | 9.38%  | 9.62%  | 10.00%   | -0.38 pp   |    0.750 |     1.006 | -12.09%  |    0.776 |
| vt_best_tgt15              | 12.87% | 14.57% | 15.00%   | -0.43 pp   |    0.752 |     1.010 | -18.83%  |    0.683 |
| vt_best_weekly             | 9.76%  | 10.94% | 10.00%   | +0.94 pp   |    0.704 |     0.937 | -16.28%  |    0.600 |
| vt_best_cost1              | 9.86%  | 10.24% | 10.00%   | +0.24 pp   |    0.754 |     1.010 | -12.91%  |    0.764 |
| vt_best_cost10             | 8.73%  | 10.24% | 10.00%   | +0.24 pp   |    0.653 |     0.871 | -13.72%  |    0.636 |
| vt_best_sameclose          | 9.93%  | 9.98%  | 10.00%   | -0.02 pp   |    0.777 |     1.047 | -12.11%  |    0.820 |
| vt_hist21_matched_settings | 10.59% | 10.26% | 10.00%   | +0.26 pp   |    0.817 |     1.110 | -12.51%  |    0.846 |
| bh_matched_vol_ex_post     | 9.19%  | 9.62%  | —        | —          |    0.732 |     1.025 | -19.09%  |    0.481 |

| run                        |   Avg. risky weight |   Annual turnover | Total fees / starting NAV   | Borrow spread / starting NAV   |
|:---------------------------|--------------------:|------------------:|:----------------------------|:-------------------------------|
| bh                         |               1.000 |             0.000 | 0.00%                       | 0.00%                          |
| vt_hist21                  |               0.853 |             5.637 | 4.88%                       | 0.71%                          |
| vt_ewma                    |               0.795 |             4.076 | 3.33%                       | 0.44%                          |
| vt_best                    |               0.802 |            11.486 | 9.36%                       | 0.44%                          |
| vt_best_nolev              |               0.737 |             7.596 | 6.29%                       | 0.00%                          |
| vt_best_tgt15              |               1.116 |            12.603 | 12.90%                      | 2.19%                          |
| vt_best_weekly             |               0.813 |             7.739 | 6.24%                       | 0.48%                          |
| vt_best_cost1              |               0.802 |            11.486 | 1.92%                       | 0.45%                          |
| vt_best_cost10             |               0.802 |            11.486 | 18.13%                      | 0.43%                          |
| vt_best_sameclose          |               0.802 |            11.486 | 9.74%                       | 0.46%                          |
| vt_hist21_matched_settings |               0.753 |             2.780 | 2.44%                       | 0.00%                          |
| bh_matched_vol_ex_post     |               0.534 |             0.000 | 0.00%                       | 0.00%                          |

The `vt_best` row uses the best development forecast with the canonical strategy settings; the separate `vt_best_nolev` row caps exposure at 1.0. The matched hist21 row uses the selected strategy settings and changes only the forecast. Same-close execution is an optimistic sensitivity. The constant weight matched to holdout volatility uses holdout outcomes by construction.

[Development strategies](runs/development/strategy_metrics.csv) · [Holdout CSV](runs/holdout/strategy_metrics.csv) · [Equity curves](assets/wealth.png) · [Drawdowns](assets/drawdown.png)

### Paired bootstrap: selected strategy minus benchmark

| benchmark                  |   Sharpe difference |   95% lower |   95% upper |
|:---------------------------|--------------------:|------------:|------------:|
| bh                         |               0.019 |      -0.297 |       0.322 |
| bh_matched_vol_ex_post     |               0.019 |      -0.297 |       0.322 |
| vt_hist21                  |              -0.010 |      -0.152 |       0.130 |
| vt_hist21_matched_settings |              -0.066 |      -0.187 |       0.050 |

Deflated Sharpe: **0.985** (probability scale), using **9** distinct development strategy configurations and an annualized reference Sharpe of **0.047**. This assesses the selected strategy's standalone Sharpe, not its advantage over hist21. See [DSR inputs](runs/holdout/dsr.json) and the [run log](runs/log.csv).

Buy-and-hold and its ex-post volatility match have the same excess Sharpe here: the latter holds a constant fraction of SPY plus T-bills, so its daily excess returns are a constant multiple of SPY's.

## Generalization

The same selected rules were applied to each asset over 2016–2025. Each model was estimated on that asset's own expanding history, using the frozen model and strategy settings. [Full metrics](runs/holdout/generalization.csv).

| ticker   | CAGR   | Vol.   | Vol. gap   |   Sharpe | Max DD   |   Avg. risky weight |   Annual turnover |
|:---------|:-------|:-------|:-----------|---------:|:---------|--------------------:|------------------:|
| XLB      | 3.91%  | 9.82%  | -0.18 pp   |    0.214 | -17.32%  |               0.580 |             3.926 |
| XLE      | 3.07%  | 10.06% | +0.06 pp   |    0.131 | -30.22%  |               0.436 |             2.236 |
| XLF      | 6.95%  | 10.09% | +0.09 pp   |    0.497 | -15.98%  |               0.585 |             5.765 |
| XLI      | 7.14%  | 10.04% | +0.04 pp   |    0.517 | -14.29%  |               0.650 |             5.200 |
| XLK      | 12.19% | 10.58% | +0.58 pp   |    0.932 | -12.49%  |               0.568 |             6.290 |
| XLP      | 4.54%  | 9.75%  | -0.25 pp   |    0.277 | -12.96%  |               0.815 |             5.932 |
| XLU      | 7.20%  | 10.13% | +0.13 pp   |    0.518 | -14.28%  |               0.660 |             4.152 |
| XLV      | 6.29%  | 9.92%  | -0.08 pp   |    0.441 | -11.95%  |               0.730 |             6.148 |
| XLY      | 9.02%  | 10.09% | +0.09 pp   |    0.687 | -16.97%  |               0.608 |             5.361 |
| TLT      | 1.16%  | 10.01% | +0.01 pp   |   -0.056 | -31.87%  |               0.753 |             1.690 |
| GLD      | 10.33% | 9.53%  | -0.47 pp   |    0.847 | -14.28%  |               0.696 |             1.504 |

## Protocol and reproduction

[Original protocol](docs/protocol.md) · [Registered splits and rules](config.yaml) · [Frozen settings](config_final.yaml)

- Burn-in: 2005–2009, unscored. Development: 2010–2015. Holdout: 2016–2025, opened once. The split and selection rule were committed at `bf378900abb6`; the final selection was committed at `9827765b05e7` before the [opening marker](runs/holdout/opened.json).
- Forecasts use adjusted-close log returns and are scored against overnight plus Garman–Klass variance from raw OHLC. Fits use all available history through the previous month-end close. Within the month the model updates its state, but parameters stay fixed. Earlier holdout observations may enter subsequent fits. Baseline five-day forecasts hold daily variance constant; GARCH/GJR use analytic recursion; EGARCH uses seeded empirical shocks for days 2–5. HAR uses averages of log RV (1, 5 and 22 days), direct step-specific OLS, and training residual smearing. Variance has a numerical floor of 1e-12. Five-day targets do not cross split boundaries.
- Development selected the lowest one-day QLIKE forecast, then the highest Sharpe among eligible strategy rows within two volatility percentage points of their own targets (otherwise the closest to target). Cost and same-close sensitivities were ineligible. Distinct non-BH strategy settings attempted in development, including sensitivity trials, count for DSR. Its dispersion estimate is from development Sharpes. Correlation between trials makes this adjustment approximate.
- A signal formed Monday at close for Tuesday is traded Tuesday at close and earns Wednesday's close-to-close return. Costs are booked on that earning row under the supplied formula. Turnover uses changes in portfolio weights as a proxy for traded notional; the model omits price drift between rebalances, slippage and fills. Weekly eligibility is the first trading close of each week, with the no-trade band retained. The strategy weights carry forward across the split; reported wealth resets to $1.
- Cash accrues the latest strictly prior DTB3 quote converted to an effective daily rate using a 91-day bill. Borrowing adds 50 bps annualized. CAGR, vol. and Sharpe annualize at 252 trading days; Sortino uses downside deviations from the T-bill rate. Cost and spread totals are dollars per dollar of starting wealth. Drawdowns include the starting NAV.
- MZ and DM use Newey–West errors. The MCS uses the range statistic and a stationary bootstrap. Sharpe differences use 10,000 paired stationary resamples (mean block length 20 days, seed 2026); the ex-post benchmark weight stays fixed during resampling.

Use Python 3.12 and Git. To reproduce without altering this run, make a clean checkout at implementation commit `5a606a64a06763eb82ad14bde3b874d89c188554` and copy the delivered `data/raw/` directory into it. Install from `requirements-lock.txt`. Run:

```sh
python -m pip install -r requirements-lock.txt
python -m pytest -q
python run.py smoke
python run.py development
python run.py freeze
git add config_final.yaml
git commit -m "Freeze settings before holdout"
python run.py holdout
```

The smoke test verifies a [real SPY trade by hand](runs/smoke/hand_check.json). Use `dev-forecast MODEL`, `dev-score` and `dev-strategy RUN` for individual development rows. The holdout command runs the frozen batch once and refuses to reopen it in the same checkout. To obtain a fresh vendor snapshot, use `python run.py download` in a separate checkout without an existing manifest. For downloading on Python 3.12, install `setuptools==84.2.1` to supply the legacy import used by pandas-datareader. Vendor history can be revised.

Implementation references: [arch forecast methods](https://arch.readthedocs.io/en/stable/univariate/forecasting.html), [FRED DTB3](https://fred.stlouisfed.org/series/DTB3), [Deflated Sharpe paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). The research bibliography is in the protocol.

## Raw snapshot

Yahoo Finance via yfinance (prices) and FRED via pandas-datareader (DTB3). [Machine-readable manifest](data/manifest.json). Each CSV is hashed before use.

| Series | Downloaded UTC | SHA-256 |
| --- | --- | --- |
| SPY | 2026-11-10T06:51:56+00:00 | `68b6185d9bc8b84b3c4c5b1634c010427644e1b5e79dc02f00a79439c103c395` |
| XLB | 2026-11-10T06:51:56+00:00 | `3f841e35138b97de62e8e6cdc1b73487236dd6f15e4573bbcb51bfac8a039a6e` |
| XLE | 2026-11-10T06:51:56+00:00 | `1762b3ab6607df0bbbc49c67376e1485d0aaa85289b19376bdd9ff4799e9c9f7` |
| XLF | 2026-11-10T06:51:56+00:00 | `12257bf7f81abe455219936fd64d9ea649c6c5cdd059416151c84ddf2c6f309e` |
| XLI | 2026-11-10T06:51:56+00:00 | `b037618c1ab37f8948fa2d79d38e7a418d19b9af312a69e193b4d2d0beb09709` |
| XLK | 2026-11-10T06:51:56+00:00 | `fa34952699fa83fd4aef2107ce5cfc98fc387e5aaf508725fdb902252c85da17` |
| XLP | 2026-11-10T06:51:56+00:00 | `e5ea7f13d26f1ca180813d1754d2c9c9b3843da6b0c8c1c6a0903fe44e61eabb` |
| XLU | 2026-11-10T06:51:56+00:00 | `eda635055de33791dd77411b7ce2a5e24abf297572d6adee821cd79cb37f963d` |
| XLV | 2026-11-10T06:51:56+00:00 | `e880ed14859d17ec279adee160137715a5b8ec82dc2086fba087b2ce00e06a97` |
| XLY | 2026-11-10T06:51:56+00:00 | `81bf7defefad3e576a117a80fb7bfa539e35326177d5476da9b19aeebd4f5633` |
| TLT | 2026-11-10T06:51:56+00:00 | `83297f2eea94fd04fc911373568f128fc14f4ac14e98e3d18a6477bf4175afd5` |
| GLD | 2026-11-10T06:51:56+00:00 | `050fc6d667a9b2df7ed38ff38beb8bf30e6012325811e47b56fe02668ab45d4d` |
| DTB3 | 2026-11-10T06:51:56+00:00 | `d340682767d85b652c33415fd4299726868abd23617442764aeed82219d50e22` |
