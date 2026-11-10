"""CLI for download, smoke, individual development rows, freeze and one holdout."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import data
from backtest import drawdown, matched, metrics, pnl, trade
from evaluate import evaluate
from forecast import forecast
from stats import deflated_sharpe, sharpe_difference

ROOT = data.ROOT
RUNS = ROOT / "runs"
CFG_FILE = ROOT / "config.yaml"
FINAL = ROOT / "config_final.yaml"
HOLDOUT = RUNS / "holdout"
CFG = json.loads(CFG_FILE.read_text())
FIELDS = ["timestamp_utc","phase","run","ticker","settings","status","error"]
DEV = "development"


def json_write(path, payload, mode="w"):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open(mode,encoding="utf8") as f:
        json.dump(payload,f,indent=2,allow_nan=False)
        f.write("\n")


def csv_write(obj,path,**kwargs):
    path.parent.mkdir(parents=True,exist_ok=True)
    obj.to_csv(path,**kwargs)


def git(*args):
    return subprocess.check_output(["git","-c",f"safe.directory={ROOT.as_posix()}",*args],
                                   cwd=ROOT,text=True).strip()


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    if git("hash-object","--path",rel,rel) != git("rev-parse",f"HEAD:{rel}"):
        raise RuntimeError(f"Commit {rel} before proceeding.")
    return git("log","-1","--format=%H","--",rel)


def record(phase,name,ticker,settings,fn):
    path = RUNS/"log.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    row = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),phase=phase,run=name,
               ticker=ticker,settings=json.dumps(settings,sort_keys=True),status="started",error="")

    def append():
        new = not path.exists()
        with path.open("a",newline="",encoding="utf8") as f:
            w = csv.DictWriter(f,fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    append()
    try:
        value = fn()
    except BaseException as exc:
        row.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        append()
        raise
    row["status"] = "ok"
    append()
    return value


def parameters(name, best):
    overrides = CFG["variants"][name].copy()
    model = overrides.pop("model",best)
    return {"model":model,**CFG["canonical"],**overrides}


def run_forecast(frame,model,phase,ticker,folder):
    def work():
        h = forecast(frame,model,CFG[DEV][0],CFG)
        csv_write(h,folder/f"forecast_{model}.csv",index_label="date")
        return h
    return record(phase,model,ticker,{"kind":"forecast","model":model},work)


def run_strategy(frame, name, forecasts, opts, phase, ticker, folder, bounds):
    def work():
        if name == "bh":
            bt = pnl(frame,pd.Series(1.,index=frame.index))
        else:
            model = opts["model"]
            bt = trade(frame,forecasts[model].h1,**{k:v for k,v in opts.items() if k!="model"})
        bt = bt.loc[bounds[0]:bounds[1]]
        csv_write(bt,folder/f"{name}.csv",index_label="date")
        return bt
    return record(phase,name,ticker,{"kind":"strategy",**opts},work)


def forecast_report(frame,forecasts,phase,folder,best=None):
    start,end = CFG[phase]
    target = frame.rv.loc[start:end]
    scores,dm,winners = evaluate(target,forecasts,CFG,best)
    csv_write(scores,folder/"forecast_scores.csv",index=False)
    csv_write(dm,folder/"dm.csv",index=False)
    return winners


def smoke():
    frame = data.load(CFG,CFG["primary"],"2010-12-31")
    folder = RUNS/"smoke"
    h = run_forecast(frame,"ewma","smoke",CFG["primary"],folder)
    opt = parameters("vt_ewma","ewma")
    bt = run_strategy(frame,"vt_ewma",{"ewma":h},opt,"smoke",CFG["primary"],folder,
                      ("2010-01-01","2010-12-31"))
    t = bt.index[bt.turnover.gt(0)][0]
    prior = frame.index[frame.index.get_loc(t)-1]
    prev_w = bt.weight.shift(fill_value=0).loc[t]
    rf,r = frame.loc[t,["rf","simple"]]
    h_used = float(h.h1.loc[prior])
    w = min(1.5,.10/(252*h_used)**.5)
    fee = .0005*abs(w-prev_w)
    spread = .005/252*max(w-1,0.)
    net = w*r+(1-w)*rf-spread-fee
    np.testing.assert_allclose(bt.loc[t,["weight","fee","return"]],[w,fee,net],rtol=1e-12)
    json_write(folder/"hand_check.json",dict(earning_date=str(t.date()),
        forecast_target_and_trade_close=str(prior.date()),variance=h_used,previous_weight=prev_w,
        asset_return=r,rf=rf,expected_weight=w,expected_fee=fee,expected_spread=spread,expected_return=net))


def dev_forecast(name):
    folder = RUNS/DEV
    frame = data.load(CFG,CFG["primary"],CFG[DEV][1])
    return run_forecast(frame,name,DEV,CFG["primary"],folder)


def saved_forecasts(folder):
    return {k:pd.read_csv(folder/f"forecast_{k}.csv",index_col="date",parse_dates=True)
            for k in CFG["models"]}


def dev_strategy(name):
    folder = RUNS/DEV
    frame = data.load(CFG,CFG["primary"],CFG[DEV][1])
    forecasts = saved_forecasts(folder)
    scores = pd.read_csv(folder/"forecast_scores.csv")
    best = scores[scores.horizon==1].sort_values("qlike",kind="stable").model.iloc[0]
    opt = {} if name=="bh" else parameters(name,best)
    bt = run_strategy(frame,name,forecasts,opt,DEV,CFG["primary"],folder,CFG[DEV])
    result = metrics(bt,opt.get("target"))
    pd.DataFrame([result],index=[name]).to_csv(folder/f"metrics_{name}.csv",index_label="run")


def development():
    folder = RUNS/DEV
    for k in CFG["models"]:
        dev_forecast(k)
    forecast_report(data.load(CFG,CFG["primary"],CFG[DEV][1]),saved_forecasts(folder),DEV,folder)
    for k in CFG["variants"]:
        dev_strategy(k)


def freeze():
    protocol_commit = committed(CFG_FILE)
    folder = RUNS/DEV
    scores = pd.read_csv(folder/"forecast_scores.csv")
    best = {str(i):scores[scores.horizon==i].sort_values("qlike",kind="stable").model.iloc[0]
            for i in (1,5)}
    table = pd.concat([pd.read_csv(folder/f"metrics_{k}.csv",index_col="run")
                       for k in CFG["variants"]])
    csv_write(table,folder/"strategy_metrics.csv",index_label="run")
    candidates = table.loc[CFG["selection"]["eligible"]].copy()
    candidates["gap"] = candidates.vol_gap.abs()
    good = candidates[candidates.gap <= CFG["selection"]["max_vol_gap"]]
    if len(good):
        selected = good.sort_values("sharpe",ascending=False,kind="stable").index[0]
    else:
        selected = candidates.sort_values(["gap","sharpe"],ascending=[True,False],kind="stable").index[0]
    trial = pd.read_csv(RUNS/"log.csv",keep_default_na=False)
    trial = trial[(trial.phase==DEV)&(trial.status=="started")]
    strategy_trials = trial[trial.settings.map(lambda s:json.loads(s)["kind"]=="strategy") & (trial.run!="bh")]
    unique = strategy_trials.drop_duplicates(["ticker","settings"])
    # Distinct settings, including failed/discarded attempts; repeated execution is not a new configuration.
    trial_sr = []
    seen = set()
    for name in table.index.drop("bh"):
        key = json.dumps(parameters(name,best["1"]),sort_keys=True)
        if key not in seen:
            trial_sr.append(float(table.loc[name,"sharpe"]))
            seen.add(key)
    json_write(FINAL,dict(protocol_commit=protocol_commit,config_sha256=data.sha(CFG_FILE),
               data_manifest_sha256=data.sha(data.MANIFEST),development_log_sha256=data.sha(RUNS/"log.csv"),
               forecast_winners=best,selected_run=selected,selected=parameters(selected,best["1"]),
               holdout_rows={k:parameters(k,best["1"]) if k!="bh" else {} for k in CFG["variants"]},
               n_trials=len(unique),trial_sharpes=trial_sr),mode="x")


def holdout():
    freeze_commit = committed(FINAL)
    final = json.loads(FINAL.read_text())
    for path,key in [(CFG_FILE,"config_sha256"),(data.MANIFEST,"data_manifest_sha256"),
                     (RUNS/"log.csv","development_log_sha256")]:
        if data.sha(path) != final[key]:
            raise RuntimeError(f"Frozen input changed: {path.name}")
    start,end = CFG["holdout"]
    json_write(HOLDOUT/"opened.json",dict(freeze_commit=freeze_commit,
               timestamp_utc=datetime.now(timezone.utc).isoformat()),mode="x")
    ticker = CFG["primary"]
    frame = data.load(CFG,ticker,end)
    forecasts = {k:run_forecast(frame,k,"holdout",ticker,HOLDOUT) for k in CFG["models"]}
    forecast_report(frame,forecasts,"holdout",HOLDOUT,
                    {int(k):v for k,v in final["forecast_winners"].items()})
    strategies, rows = {}, []
    for name,opts in final["holdout_rows"].items():
        bt = run_strategy(frame,name,forecasts,opts,"holdout",ticker,HOLDOUT,(start,end))
        strategies[name] = bt
        rows.append({"run":name,**metrics(bt,opts.get("target"))})
    name = final["selected_run"]
    selected = strategies[name]
    same = dict(final["selected"],model="hist21")
    if same != final["holdout_rows"]["vt_hist21"]:
        k = "vt_hist21_matched_settings"
        bt = run_strategy(frame,k,forecasts,same,"holdout",ticker,HOLDOUT,(start,end))
        strategies[k] = bt
        rows.append({"run":k,**metrics(bt,same["target"])})
    ex_post = "bh_matched_vol_ex_post"
    bt = record("holdout",ex_post,ticker,{"kind":"ex_post","reference":name},
                lambda:matched(frame.loc[start:end],selected,CFG["canonical"]["spread"]))
    strategies[ex_post] = bt
    csv_write(bt,HOLDOUT/f"{ex_post}.csv",index_label="date")
    rows.append({"run":ex_post,**metrics(bt)})
    csv_write(pd.DataFrame(rows),HOLDOUT/"strategy_metrics.csv",index=False)
    comparators = ["bh",ex_post,"vt_hist21"]
    if "vt_hist21_matched_settings" in strategies:
        comparators.append("vt_hist21_matched_settings")
    excess = selected["return"]-selected.rf
    ci = []
    for k in comparators:
        x = strategies[k]
        ci.append({"benchmark":k,**sharpe_difference(excess,x["return"]-x.rf,CFG)})
    csv_write(pd.DataFrame(ci),HOLDOUT/"sharpe_intervals.csv",index=False)
    json_write(HOLDOUT/"dsr.json",deflated_sharpe(excess,final["trial_sharpes"],final["n_trials"]))
    general = []
    for ticker2 in CFG["generalization"]:
        k = f"vt_final_{ticker2}"
        folder = HOLDOUT/"generalization"/ticker2
        f = data.load(CFG,ticker2,end)
        model = final["selected"]["model"]
        h = run_forecast(f,model,"holdout",ticker2,folder)
        bt = run_strategy(f,k,{model:h},final["selected"],"holdout",ticker2,folder,(start,end))
        general.append({"ticker":ticker2,**metrics(bt,final["selected"]["target"])})
    csv_write(pd.DataFrame(general),HOLDOUT/"generalization.csv",index=False)
    from report import charts
    charts(frame.loc[start:end],forecasts,strategies,name,final["forecast_winners"]["1"])
    json_write(HOLDOUT/"completed.json",{"freeze_commit":freeze_commit,
               "timestamp_utc":datetime.now(timezone.utc).isoformat()},mode="x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["download","smoke","development","dev-forecast","dev-score",
                                           "dev-strategy","freeze","holdout"])
    parser.add_argument("row",nargs="?")
    args = parser.parse_args()
    if args.command != "download":
        committed(CFG_FILE)
    if args.command in ["smoke","development","dev-forecast","dev-score","dev-strategy","freeze"] and FINAL.exists():
        parser.error("Development is sealed by config_final.yaml.")
    if args.command in ["development","dev-forecast","dev-score","dev-strategy","freeze"]:
        if not (RUNS/"smoke"/"hand_check.json").exists():
            parser.error("Complete the 2010 smoke check first.")
    if args.command == "download":
        data.download(CFG)
    elif args.command == "dev-forecast":
        if args.row not in CFG["models"]:
            parser.error("Pass a model name from config.yaml.")
        dev_forecast(args.row)
    elif args.command == "dev-score":
        folder = RUNS/DEV
        forecast_report(data.load(CFG,CFG["primary"],CFG[DEV][1]),saved_forecasts(folder),DEV,folder)
    elif args.command == "dev-strategy":
        if args.row not in CFG["variants"]:
            parser.error("Pass a strategy name from config.yaml.")
        dev_strategy(args.row)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()

