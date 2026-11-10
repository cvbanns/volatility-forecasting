from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest import drawdown

ROOT = Path(__file__).resolve().parent


def charts(frame, forecasts, strategies, selected, best):
    folder = ROOT/"assets"
    folder.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(np.sqrt(252*frame.rv.rolling(21).mean()),label="Realized proxy, 21-day smoothing",
            color="#9ba8af",lw=1)
    for model in dict.fromkeys([best,"hist21"]):
        ax.plot(np.sqrt(252*forecasts[model].h1.reindex(frame.index)),label=model,lw=1)
    ax.set(title="SPY variance forecasts | 2016–2025",ylabel="Annualized volatility")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/"forecast.png",dpi=160)
    plt.close(fig)
    for kind in ("wealth","drawdown"):
        fig, ax = plt.subplots(figsize=(10,4))
        for k in dict.fromkeys([selected,"bh","vt_hist21","bh_matched_vol_ex_post"]):
            r = strategies[k]["return"]
            values = (1+r).cumprod() if kind=="wealth" else drawdown(r)
            ax.plot(values,label=k,lw=1.25)
        ax.set(title=f"SPY {kind} | 2016–2025",
               ylabel="Growth of $1" if kind=="wealth" else "Peak-to-trough loss")
        if kind=="drawdown":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder/f"{kind}.png",dpi=160)
        plt.close(fig)

