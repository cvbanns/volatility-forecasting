"""Outputs indexed by first target date: h1 and sum of h1,...,h5."""
import numpy as np
import pandas as pd
from arch import arch_model

FLOOR = 1e-12
SPECS = {"garch_n": ("GARCH", 0, "normal"), "garch_t": ("GARCH", 0, "t"),
         "gjr_t": ("GARCH", 1, "t"), "egarch_t": ("EGARCH", 0, "t")}


def fitted_month(r, a, b, model, cfg):
    vol, o, dist = SPECS[model]
    fit = arch_model(r.iloc[:a]*100, vol=vol, p=1, o=o, q=1,
                     dist=dist, rescale=False).fit(disp="off", show_warning=False)
    if fit.convergence_flag != 0:
        # Prespecified optimizer retry, same data and model.
        fit = fit.model.fit(disp="off", show_warning=False,
                            options={"maxiter": 3000, "ftol": 1e-7})
    if fit.convergence_flag != 0:
        raise RuntimeError(f"{model}: fit failed for {r.index[a].date()}: {fit.optimization_result.message}")
    p = fit.params
    alpha, beta, omega = p["alpha[1]"], p["beta[1]"], p["omega"]
    gamma = p.get("gamma[1]", 0.)
    mu = p["mu"]
    h = float(fit.forecast(horizon=1, reindex=False).variance.iloc[-1,0])
    draws = None
    if vol == "EGARCH":
        pool = np.asarray(fit.std_resid.dropna())
        pool = (pool-pool.mean()) / pool.std()
        rng = np.random.default_rng(cfg["seed"] + r.index[a].toordinal())
        draws = rng.choice(pool, size=(4,cfg["egarch_paths"]))
    out = []
    for i in range(a,b):
        h = max(h, 1e-8)
        if draws is None:
            multi = h
            step = h
            for _ in range(4):
                step = omega + (alpha + gamma/2 + beta)*step
                multi += step
        else:
            logh = np.log(h)
            multi = h
            for z in draws:
                logh = omega + alpha*(abs(z)-np.sqrt(2/np.pi)) + beta*logh
                multi += np.exp(logh).mean()
        out.append((h/1e4, multi/1e4))
        if i+1 < b:
            e = 100*r.iloc[i]-mu
            if vol == "EGARCH":
                h = np.exp(omega + alpha*(abs(e/np.sqrt(h))-np.sqrt(2/np.pi)) + beta*np.log(h))
            else:
                h = omega + (alpha+gamma*(e<0))*e*e + beta*h
    return out


def har_month(z, x, a, b):
    preds = []
    for j in range(1,6):
        # Feature row k predicts log RV at k+j. Last training label is a-1.
        xx, yy = x[21:a-j], z[21+j:a]
        coef = np.linalg.lstsq(xx, yy, rcond=None)[0]
        smear = np.exp(yy-xx@coef).mean()
        preds.append(np.exp(x[a-1:b-1]@coef)*smear)
    return np.column_stack([preds[0], np.sum(preds, axis=0)])


def forecast(frame, model, start, cfg):
    r = frame.log_return
    idx = r.index
    first = idx.searchsorted(pd.Timestamp(start))
    if first < 100 or first >= len(idx):
        raise ValueError("Forecasting requires a burn-in and a nonempty test interval.")
    if model in ("hist21","ewma"):
        if model == "hist21":
            h = r.rolling(21).var()
        else:
            h = r.pow(2).ewm(alpha=1-cfg["ewma_decay"], adjust=False).mean()
        h = h.shift().iloc[first:].clip(lower=FLOOR)
        return pd.DataFrame({"h1":h, "h5":5*h})
    if model == "har_rv":
        z = np.log(frame.rv.clip(lower=FLOOR))
        x = np.column_stack([np.ones(len(z)),z,z.rolling(5).mean(),z.rolling(22).mean()])
        z = z.to_numpy()
    periods = idx.to_period("M")
    boundaries = np.flatnonzero(periods[1:] != periods[:-1])+1
    starts = np.r_[first, boundaries[boundaries>first], len(idx)]
    out = []
    for a,b in zip(starts[:-1],starts[1:]):
        if model == "har_rv":
            out.extend(har_month(z,x,a,b))
        else:
            out.extend(fitted_month(r,a,b,model,cfg))
    result = pd.DataFrame(out,index=idx[first:],columns=["h1","h5"])
    if not np.isfinite(result).all().all():
        raise RuntimeError(f"Nonfinite forecasts: {model}")
    return result.clip(lower=FLOOR)

