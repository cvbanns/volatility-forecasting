"""Immutable snapshots of vendor frames (CSV) and causal feature construction."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
MANIFEST = ROOT / "data/manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    return json.loads(MANIFEST.read_text())


def download(cfg):
    import yfinance as yf
    from pandas_datareader import data as pdr
    RAW.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        raise FileExistsError("Snapshot already sealed; reuse it.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    files = {}

    def save(name, frame, source):
        if frame.empty:
            raise ValueError(f"Empty vendor response: {name}")
        path = RAW / f"{name}_{stamp}.csv"
        frame.to_csv(path, index_label="date")
        files[name] = {"file": str(path.relative_to(ROOT).as_posix()),
                       "downloaded_utc": now, "sha256": sha(path), "source": source}

    start, end = cfg["sample"]
    for ticker in [cfg["primary"], *cfg["generalization"]]:
        frame = yf.download(ticker, start=start, end=pd.Timestamp(end)+pd.Timedelta(days=1),
                            auto_adjust=False, progress=False, threads=False,
                            multi_level_index=False, timeout=30)
        save(ticker, frame[["Open","High","Low","Close","Adj Close","Volume"]], "Yahoo Finance via yfinance")
    rf = cfg["rf_series"]
    save(rf, pdr.DataReader(rf, "fred", pd.Timestamp(start)-pd.Timedelta(days=14), end),
         "FRED via pandas-datareader")
    MANIFEST.write_text(json.dumps(files, indent=2) + "\n")


def read_raw(name, end):
    entry = manifest()[name]
    path = ROOT / entry["file"]
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Snapshot hash mismatch: {name}")
    raw = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    if raw.index.has_duplicates:
        raise ValueError(f"Duplicate dates: {name}")
    return raw.loc[:end]


def features(prices, bills):
    cols = ["Open","High","Low","Close","Adj Close"]
    if not np.isfinite(prices[cols]).all().all() or (prices[cols] <= 0).any().any():
        raise ValueError("Invalid price row.")
    o,h,l,c,p = (np.log(prices[k]) for k in cols)
    rv = (o-c.shift()).pow(2) + .5*(h-l).pow(2) - (2*np.log(2)-1)*(c-o).pow(2)
    if (rv < -1e-12).any():
        raise ValueError("Negative realized variance: inspect OHLC.")
    # DTB3 bank-discount quote: illustrative constant 91-day bill, 360-day basis.
    # Use the latest observation strictly before the return date; carry over holidays.
    quotes = bills.dropna()
    quotes = quotes.reindex(quotes.index.union(prices.index)).ffill()
    d = quotes.shift().reindex(prices.index) / 100
    daily = (1-d*91/360).pow(-365/(91*252))-1
    out = pd.DataFrame({"log_return": p.diff(), "simple": np.expm1(p.diff()),
                        "rv": rv.clip(lower=1e-12), "rf": daily}).iloc[1:]
    if out.isna().any().any():
        raise ValueError("Missing market or prior bill observation.")
    return out


def load(cfg, ticker, end):
    return features(read_raw(ticker, end), read_raw(cfg["rf_series"], end)[cfg["rf_series"]])

