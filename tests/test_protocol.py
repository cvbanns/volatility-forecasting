import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from backtest import pnl,trade
from evaluate import qlike
from forecast import forecast
from arch import arch_model

CFG = json.loads((Path(__file__).parents[1]/"config.yaml").read_text())


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(59)
    idx = pd.bdate_range("2005-01-01", periods=520)
    r = rng.normal(0,.009,len(idx))
    return pd.DataFrame({"log_return":r,"rv":r*r+2e-5},index=idx)


@pytest.mark.parametrize("model",CFG["models"])
def test_future_data_do_not_change_forecasts(frame,model):
    start, through = frame.index[460],frame.index[485]
    prefix = frame.loc[:through]
    altered = frame.copy()
    altered.loc[frame.index>=through,["log_return","rv"]] = [.20,.04]
    a = forecast(prefix,model,start,CFG)
    b = forecast(altered,model,start,CFG).loc[:through]
    pd.testing.assert_frame_equal(a,b,rtol=1e-10,atol=1e-13)


@pytest.mark.parametrize("name,vol,o,dist",[("garch_n","GARCH",0,"normal"),
    ("garch_t","GARCH",0,"t"),("gjr_t","GARCH",1,"t"),("egarch_t","EGARCH",0,"t")])
def test_monthly_recursion_matches_arch_reference(frame,name,vol,o,dist):
    periods = frame.index.to_period("M")
    a = next(i for i in range(440,480) if periods[i]!=periods[i-1])
    end = a+6
    f = frame.iloc[:end]
    p = arch_model(f.log_return.iloc[:a]*100,vol=vol,o=o,dist=dist).fit(disp="off").params
    actual = forecast(f,name,f.index[a],CFG)
    for i in range(a,end):
        fixed = arch_model(f.log_return.iloc[:i]*100,vol=vol,o=o,dist=dist).fix(p)
        expected = fixed.forecast(horizon=1,reindex=False).variance.iloc[-1,0]/1e4
        assert actual.h1.loc[f.index[i]] == pytest.approx(expected,rel=1e-8)


def test_three_day_pnl():
    i = pd.bdate_range("2010-01-04",periods=3)
    f = pd.DataFrame({"simple":[.01,-.02,.03],"rf":[.0001]*3},index=i)
    got = pnl(f,pd.Series([.5,1.2,1.2],index=i),.0005,.005,initial=0.)
    expected = [.5*.01+.5*.0001-.0005*.5,
                1.2*(-.02)-.2*.0001-.2*.005/252-.0005*.7,
                1.2*.03-.2*.0001-.2*.005/252]
    np.testing.assert_allclose(got["return"],expected)


def test_trade_one_close_later():
    i = pd.bdate_range("2010-01-04",periods=4)
    f = pd.DataFrame({"simple":[.01]*4,"rf":[0.]*4},index=i)
    h = pd.Series([(.1/.5)**2/252,(.1/.53)**2/252,(.1/1.2)**2/252],index=i[:3])
    bt = trade(f,h,**CFG["canonical"])
    np.testing.assert_allclose(bt.weight,[0.,.5,.5,1.2])
    np.testing.assert_allclose(bt.fee,np.array([0.,.5,0.,.7])*.0005)


def test_qlike_perfect():
    rv = np.array([1e-7,1e-4,.01])
    np.testing.assert_array_equal(qlike(rv,rv),0.)

