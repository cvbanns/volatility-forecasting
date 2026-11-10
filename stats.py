import numpy as np
from scipy.stats import norm, skew, kurtosis
from backtest import sharpe


def sharpe_difference(selected, comparator, cfg):
    x = np.column_stack([selected, comparator])
    n = len(x)
    rng = np.random.default_rng(cfg["seed"])
    reps = cfg["bootstrap_reps"]
    draws = np.empty(reps)
    # Circular stationary bootstrap with paired, geometrically sized blocks.
    for a in range(0,reps,256):
        b = min(reps,a+256)
        k = b-a
        starts = rng.integers(0,n,size=(k,n))
        reset = rng.random((k,n)) < 1/cfg["block"]
        reset[:,0] = True
        t = np.arange(n)
        recent = np.maximum.accumulate(np.where(reset,t,0),axis=1)
        ix = (np.take_along_axis(starts,recent,axis=1)+t-recent)%n
        sample = x[ix]
        s = np.sqrt(252)*sample.mean(axis=1)/sample.std(axis=1,ddof=1)
        draws[a:b] = s[:,0]-s[:,1]
    lo,hi = np.quantile(draws,[.025,.975])
    return dict(delta=sharpe(x[:,0])-sharpe(x[:,1]),lo95=lo,hi95=hi)


def deflated_sharpe(excess, trial_sharpes, n_trials):
    # Inputs: daily excess returns, annualized development trial Sharpe values.
    x = np.asarray(excess)
    sr = x.mean()/x.std(ddof=1)
    dispersion = np.std(trial_sharpes,ddof=1)/np.sqrt(252) if len(trial_sharpes)>1 else 0.
    gamma = np.euler_gamma
    threshold = 0.
    if n_trials > 1:
        threshold = dispersion*((1-gamma)*norm.ppf(1-1/n_trials)
                                +gamma*norm.ppf(1-1/(n_trials*np.e)))
    variance = 1-skew(x,bias=False)*sr+(kurtosis(x,fisher=False,bias=False)-1)*sr**2/4
    z = (sr-threshold)*np.sqrt((len(x)-1)/variance)
    return dict(dsr=float(norm.cdf(z)), n_trials=n_trials,
                threshold_annual=float(threshold*np.sqrt(252)),
                trial_sharpe_std_annual=float(dispersion*np.sqrt(252)),n=len(x))

