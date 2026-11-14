"""Daily simple P&L under the protocol's weight-change cost convention."""
import numpy as np
import pandas as pd

YEAR = 252


def pnl(frame, weights, cost=0., spread=0., initial=None):
    w = weights.reindex(frame.index)
    delta = w.diff()
    delta.iloc[0] = 0 if initial is None else w.iloc[0]-initial
    turnover = delta.abs()
    fees = cost * turnover
    borrowing = (spread/YEAR) * (w-1).clip(lower=0)
    net = w*frame.simple + (1-w)*frame.rf - fees - borrowing
    if (net <= -1).any() or not np.isfinite(net).all():
        raise ValueError("Invalid portfolio P&L.")
    return pd.DataFrame({"return":net, "rf":frame.rf, "weight":w, "turnover":turnover,
                         "fee":fees, "borrow_spread":borrowing})


def trade(frame, h, target, cap, lag, band, weekly, cost, spread):
    # h is dated by first target. h[d] was known at the previous close.
    # For lag=1 that signal trades at close d and earns the return dated d+1.
    desired = (target/np.sqrt(YEAR*h)).clip(upper=cap).reindex(frame.index).shift(lag)
    # An earning row's execution close is the immediately prior trading date.
    # A weekly trade occurs at the first close of a new trading week.
    weeks = frame.index.to_period("W-FRI")
    first = np.r_[True, weeks[1:] != weeks[:-1]]
    eligible = np.r_[False, first[:-1]] if weekly else np.ones(len(frame), dtype=bool)
    prev = 0.
    weights = []
    for want, allowed in zip(desired,eligible):
        if allowed and pd.notna(want) and abs(want-prev) >= band:
            prev = want
        weights.append(prev)
    return pnl(frame,pd.Series(weights,index=frame.index),cost,spread,initial=0.)


def sharpe(excess):
    return np.sqrt(YEAR)*np.mean(excess)/np.std(excess, ddof=1)


def drawdown(r):
    wealth = (1+r).cumprod()
    return wealth/wealth.cummax().clip(lower=1)-1


def metrics(bt, target=None):
    r = bt["return"]
    excess = r-bt.rf
    cagr = np.expm1(np.log1p(r).mean()*YEAR)
    vol = r.std()*np.sqrt(YEAR)
    dd = drawdown(r).min()
    downside = np.sqrt(np.minimum(excess,0).pow(2).mean())
    wealth_before = (1+r).cumprod().shift(fill_value=1.)
    return dict(cagr=cagr,vol=vol,sharpe=sharpe(excess),
                sortino=np.sqrt(YEAR)*excess.mean()/downside if downside else np.nan,
                max_drawdown=dd,calmar=cagr/-dd if dd < 0 else np.nan,
                avg_leverage=bt.weight.mean(),annual_turnover=YEAR*bt.turnover.mean(),
                total_cost=(wealth_before*bt.fee).sum(),
                total_borrow_spread=(wealth_before*bt.borrow_spread).sum(),
                target=target,vol_gap=vol-target if target is not None else None)


def matched(frame, reference, spread):
    # Solve Var(rf + w*(r-rf)) = Var(reference). Constant borrowing
    # spread shifts the mean only. This diagnostic can use holdout outcomes.
    x = frame.simple-frame.rf
    a = x.var()
    b = 2*x.cov(frame.rf)
    c = frame.rf.var()-reference["return"].var()
    disc = b*b-4*a*c
    if a <= 0 or disc < 0:
        raise ValueError("No positive constant weight matches realized volatility.")
    w = (-b+np.sqrt(disc))/(2*a)
    if w < 0:
        raise ValueError("No positive constant weight matches realized volatility.")
    return pnl(frame,pd.Series(w,index=frame.index),spread=spread)

