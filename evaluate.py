import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import MCS


def qlike(rv,h):
    ratio = rv/h
    return ratio-np.log(ratio)-1


def evaluate(rv, forecasts, cfg, best=None):
    scores, tests = [], []
    selected = {}
    targets = {1:rv, 5:rv.rolling(5).sum().shift(-4)}
    for horizon, target in targets.items():
        h = pd.DataFrame({k:v[f"h{horizon}"] for k,v in forecasts.items()})
        h = h.loc[target.dropna().index.intersection(h.index)]
        if h.isna().any().any() or h.empty:
            raise ValueError("Require a common nonempty forecast sample.")
        y = target.loc[h.index]
        loss = qlike(y.to_numpy()[:,None], h)
        winner = best[horizon] if best else loss.mean().idxmin()
        selected[horizon] = winner
        # Use a fixed bandwidth of at least horizon-1, including overlap.
        lags = max(horizon-1, int(4*(len(y)/100)**(2/9)))
        # Collapse identical loss vectors; include all aliases if representative survives.
        representatives = [k for i,k in enumerate(loss)
                           if not any(loss[k].equals(loss[j]) for j in list(loss)[:i])]
        if len(representatives) > 1:
            mcs = MCS(loss[representatives],size=cfg["mcs_size"],reps=cfg["bootstrap_reps"],
                      block_size=cfg["block"],method="R",bootstrap="stationary",seed=cfg["seed"])
            mcs.compute()
            included = mcs.included
            p_values = mcs.pvalues.iloc[:,0]
        else:
            included = representatives
            p_values = pd.Series(1.,index=representatives)
        for k in h:
            x = h[k].to_numpy()
            # Scale both sides to improve conditioning without changing the null.
            mz = sm.OLS(y.to_numpy()*1e4,sm.add_constant(x*1e4)).fit(
                cov_type="HAC",cov_kwds={"maxlags":lags})
            wald = mz.wald_test((np.eye(2),np.array([0.,1.])),scalar=True)
            rep = next(j for j in representatives if loss[k].equals(loss[j]))
            scores.append(dict(horizon=horizon,model=k,n=len(y),qlike=loss[k].mean(),
                               mse=((y-h[k])**2).mean(),mz_alpha=mz.params[0]/1e4,
                               mz_beta=mz.params[1],mz_p=float(wald.pvalue),mcs90=rep in included,
                               mcs_p=float(p_values.loc[rep])))
            for benchmark in dict.fromkeys(["hist21",winner]):
                if k == benchmark:
                    continue
                diff = loss[k]-loss[benchmark]
                if (diff == 0).all():
                    t,p = 0.,1.
                else:
                    dm = sm.OLS(diff.to_numpy(),np.ones((len(diff),1))).fit(
                        cov_type="HAC",cov_kwds={"maxlags":lags})
                    t,p = dm.tvalues[0],dm.pvalues[0]
                tests.append(dict(horizon=horizon,model=k,benchmark=benchmark,dm=t,p=p,
                                  mean_loss_difference=diff.mean(),nw_lags=lags))
    return pd.DataFrame(scores),pd.DataFrame(tests),selected

