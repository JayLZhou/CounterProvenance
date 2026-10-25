"""Multi-node SCM front end: the index must equal propagation through the DAG."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ScmOnline, bind_scm, fit_scm, star_bind

EDGES = [("A", "Z"), ("C", "Z"), ("Z", "W"), ("B", "W"), ("A", "Y"), ("Z", "Y"), ("W", "Y"), ("D", "Y")]


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    A, B, C, D = (rng.normal(size=n) for _ in range(4))
    Z = np.sin(2 * A) + 0.5 * C + 0.1 * rng.normal(size=n)
    W = Z * B + 0.1 * rng.normal(size=n)
    Y = Z + np.abs(W) + 0.5 * A * D + 0.1 * rng.normal(size=n)
    return pd.DataFrame(dict(A=A, B=B, C=C, D=D, Z=Z, W=W, Y=Y))


def _factories():
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    out = {
        "gbr": lambda: GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=15, early_stopping=False,
                                                     random_state=0),
        "rf": lambda: RandomForestRegressor(n_estimators=15, max_depth=5, random_state=0, n_jobs=1),
    }
    try:
        import xgboost as xgb

        out["xgb"] = lambda: xgb.XGBRegressor(n_estimators=30, max_depth=4, random_state=0, n_jobs=1)
    except ImportError:
        pass
    try:
        import lightgbm as lgb

        out["lgbm"] = lambda: lgb.LGBMRegressor(n_estimators=30, num_leaves=15, random_state=0, verbose=-1,
                                                n_jobs=1)
    except ImportError:
        pass
    return out


def _queries(df, treatments, bound, rng, n=60):
    lo = df[treatments].quantile(0.01).to_numpy()
    hi = df[treatments].quantile(0.99).to_numpy()
    pts = list(rng.uniform(lo, hi, size=(n, len(treatments))))
    th = [[b.hi[a] for b in bound.boxes if np.isfinite(b.hi[a])] for a in range(len(treatments))]
    for _ in range(n // 2):  # points exactly on a box boundary
        p = rng.uniform(lo, hi)
        a = int(rng.integers(len(treatments)))
        if th[a]:
            p[a] = th[a][int(rng.integers(len(th[a])))]
        pts.append(p)
    return pts


@pytest.mark.parametrize("family", list(_factories()))
@pytest.mark.parametrize("treatments", [["A"], ["A", "B"], ["B", "C", "D"]])
def test_index_equals_propagation(family, treatments):
    df = _data()
    scm = fit_scm(df, EDGES, "Y", treatments, _factories()[family])
    rng = np.random.default_rng(1)
    unit = df.iloc[int(rng.integers(len(df)))].to_dict()
    bound = bind_scm(scm, unit, treatments, "Y")
    index = ProvenanceIndex(bound)
    online = ScmOnline(scm, unit, treatments, "Y")
    tol = 5e-5 if family == "xgb" else 1e-9
    for t in _queries(df, treatments, bound, rng):
        got, ref = index.query(t).value, online.query(t)
        assert abs(got - ref) <= tol * max(1.0, abs(ref)), (family, treatments, t, got, ref)


@pytest.mark.parametrize("family", ["gbr", "lgbm"])
def test_star_graph_matches_single_model_binding(family):
    df = _data()
    edges = [(c, "Y") for c in ["A", "B", "C", "D"]]
    scm = fit_scm(df, edges, "Y", ["A", "B"], _factories()[family])
    unit = df.iloc[5].to_dict()
    b1 = bind_scm(scm, unit, ["A", "B"], "Y")
    b2 = star_bind(scm, unit, ["A", "B"], "Y")
    rng = np.random.default_rng(2)
    for t in _queries(df, ["A", "B"], b2, rng):
        assert abs(b1.value(t) - b2.value(t)) <= 1e-12 * max(1.0, abs(b2.value(t)))


@pytest.mark.parametrize("family", list(_factories()))
def test_dag_baselines_equal_propagation(family):
    from counterprovenance.scm_baselines import ScmNumba, ScmOnlinePy

    df = _data()
    treatments = ["A", "B"]
    scm = fit_scm(df, EDGES, "Y", treatments, _factories()[family])
    rng = np.random.default_rng(3)
    unit = df.iloc[int(rng.integers(len(df)))].to_dict()
    bound = bind_scm(scm, unit, treatments, "Y")
    ref = ScmOnline(scm, unit, treatments, "Y")
    py = ScmOnlinePy(scm, unit, treatments, "Y")
    nb = ScmNumba(scm, unit, treatments, "Y")
    tol = 5e-5 if family == "xgb" else 1e-9
    pts = _queries(df, treatments, bound, rng)
    batch = nb.batch(np.asarray(pts))
    for t, b in zip(pts, batch):
        r = ref.query(t)
        for got in (py.query(t)[0], nb.query(np.asarray(t)), b):
            assert abs(got - r) <= tol * max(1.0, abs(r)), (family, t, got, r)
