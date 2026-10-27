"""Walking the cell graph must give the value and the leaves of a fresh query at every step."""

from __future__ import annotations

import numpy as np
import pytest

from counterprovenance.scm import bind_scm, fit_scm, star_bind
from counterprovenance.walk import HAVE_NUMBA, NumbaWalk, Walk

from test_scm import EDGES, _data, _factories


def _steps(df, treatments, bound, rng, n=150):
    """Slider moves (one axis, small or large steps), jumps of every axis, and box boundaries."""
    lo = df[treatments].quantile(0.01).to_numpy()
    hi = df[treatments].quantile(0.99).to_numpy()
    th = [[b.hi[a] for b in bound.boxes if np.isfinite(b.hi[a])] for a in range(len(treatments))]
    t = rng.uniform(lo, hi)
    out = []
    for _ in range(n):
        t = t.copy()
        kind = rng.integers(4)
        a = int(rng.integers(len(treatments)))
        if kind == 0:
            t[a] += rng.normal() * 0.05 * (hi[a] - lo[a])
        elif kind == 1:
            t[a] = rng.uniform(lo[a], hi[a])
        elif kind == 2 and th[a]:
            t[a] = th[a][int(rng.integers(len(th[a])))]
        else:
            t = rng.uniform(lo, hi)
        out.append(t)
    return out


@pytest.mark.parametrize("family", list(_factories()))
@pytest.mark.parametrize("graph", ["star", "dag"])
def test_walk_equals_fresh_queries(family, graph):
    df = _data()
    treatments = ["A", "B"] if graph == "dag" else ["A", "B", "C"]
    edges = EDGES if graph == "dag" else [(c, "Y") for c in ["A", "B", "C", "D"]]
    scm = fit_scm(df, edges, "Y", treatments, _factories()[family])
    rng = np.random.default_rng(5)
    unit = df.iloc[int(rng.integers(len(df)))].to_dict()
    bound = bind_scm(scm, unit, treatments, "Y") if graph == "dag" else star_bind(scm, unit, treatments, "Y")
    steps = _steps(df, treatments, bound, rng)
    walk = Walk(bound, steps[0])
    nb = NumbaWalk(bound, np.asarray(steps[0]), plan=walk.plan) if HAVE_NUMBA else None
    for t in steps:
        ref = bound.value(t)
        got = walk.move(t)
        assert abs(got - ref) <= 1e-9 * max(1.0, abs(ref)), (family, graph, t, got, ref)
        assert sorted(walk.labels()) == sorted(bound.label(i) for i in bound.selected(t)
                                               if bound.boxes[i].tree in set(walk.plan.trees))
        if nb is not None:
            v = nb.move(np.asarray(t, dtype=np.float64))
            assert abs(v - ref) <= 1e-9 * max(1.0, abs(ref)), (family, graph, t, v, ref)
    assert walk.relocated < len(steps) * len(walk.plan.trees)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba")
@pytest.mark.parametrize("family", list(_factories()))
def test_incremental_baseline_equals_fresh_queries(family):
    from counterprovenance.scm_baselines import ScmIncremental, ScmNumba

    df = _data()
    treatments = ["A", "B"]
    scm = fit_scm(df, EDGES, "Y", treatments, _factories()[family])
    rng = np.random.default_rng(6)
    unit = df.iloc[int(rng.integers(len(df)))].to_dict()
    bound = bind_scm(scm, unit, treatments, "Y")
    steps = _steps(df, treatments, bound, rng)
    inc = ScmIncremental(scm, unit, treatments, "Y", np.asarray(steps[0]))
    full = ScmNumba(scm, unit, treatments, "Y")
    for t in steps:
        t = np.asarray(t, dtype=np.float64)
        assert inc.move(t) == full.query(t)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba")
@pytest.mark.parametrize("tau", [0, 5, 1e9, None])
def test_session_equals_fresh_queries(tau):
    from counterprovenance import ProvenanceIndex, accel
    from counterprovenance.walk import Session

    df = _data()
    treatments = ["A", "B", "C"]
    scm = fit_scm(df, EDGES, "Y", treatments, _factories()["gbr"])
    rng = np.random.default_rng(7)
    unit = df.iloc[int(rng.integers(len(df)))].to_dict()
    bound = bind_scm(scm, unit, treatments, "Y")
    steps = [np.asarray(t, dtype=np.float64) for t in _steps(df, treatments, bound, rng, n=300)]
    nb_index = accel.NumbaIndex(ProvenanceIndex(bound))
    session = Session(nb_index, NumbaWalk(bound, steps[0]), steps[0], tau=tau, calibrate=np.array(steps[:50]))
    for t in steps:
        ref = bound.value(t)
        got = session.move(t)
        assert abs(got - ref) <= 1e-9 * max(1.0, abs(ref)), (tau, t, got, ref)
    if tau == 0:
        assert session.walked < len(steps)
    if tau == 1e9:
        assert session.queried == 0
