#!/usr/bin/env python3
"""Query sessions: walking the cell graph vs fresh queries and incremental propagation.

For one dataset, every ``family:d`` and ``--units`` units: fit the SCM once,
bind and index the unit, and replay three workloads of ``--steps`` queries:

* ``slider``: one random axis moves by a Gaussian step of 2% of its range;
* ``sweep``: one axis at a time runs over a 100-point grid of its range;
* ``jump``: every step is a fresh uniform point (every axis changes; worst case).

Methods (numba, one call per step): the walk (``NumbaWalk``), a fresh query of
the index (``NumbaIndex``), incremental propagation (``ScmIncremental``) and
full propagation (``ScmNumba``); and in pure Python the walk (``Walk``), the
index (``ProvenanceIndex.query``) and full propagation (``ScmOnlinePy``).
Every step of the walk is checked against the index (exact at 1e-9), and a
sample against ``ScmOnlinePy``.  Output: ``results/session_<dataset>.csv`` and
per-step latencies in ``results/session_lat/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, per_call_seconds, per_query_us, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmIncremental, ScmNumba, ScmOnlinePy
from counterprovenance.walk import NumbaWalk, Session, Walk, _Plan
from scm_bench import choose_treatments, zoo


def workloads(rng, lo, hi, steps):
    d = len(lo)
    out = {}
    t = rng.uniform(lo, hi)
    seq = [t]
    for _ in range(steps - 1):
        t = t.copy()
        a = int(rng.integers(d))
        t[a] = float(np.clip(t[a] + rng.normal() * 0.02 * (hi[a] - lo[a]), lo[a], hi[a]))
        seq.append(t)
    out["slider"] = np.array(seq)
    t = rng.uniform(lo, hi)
    seq, flip = [], False
    while len(seq) < steps:
        a = int(rng.integers(d))
        grid = np.linspace(lo[a], hi[a], 100)
        for v in (grid[::-1] if flip else grid):
            t = t.copy()
            t[a] = v
            seq.append(t)
        flip = not flip
    out["sweep"] = np.array(seq[:steps])
    out["jump"] = rng.uniform(lo, hi, size=(steps, d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated family:d")
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--py-steps", type=int, default=2_000, help="steps timed in pure Python")
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    outcome = str(meta["target"])
    rows, fitted = [], {}
    lat_dir = RESULTS / "session_lat"
    lat_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.settings.split(","):
        fam, d = spec.split(":")
        d = int(d)
        if fam not in fitted:
            fitted[fam] = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]],
                                  zoo(1000, 1)[fam], max_rows=50_000, seed=0, n_jobs=4)
        scm = fitted[fam]
        T = choose_treatments(scm, outcome, d)
        if T is None:
            continue
        rng = np.random.default_rng(2028)
        lo, hi = df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy()
        for u in range(args.units):
            unit = df.iloc[int(rng.integers(len(df)))].to_dict()
            start = time.perf_counter()
            bound = bind_scm(scm, unit, T, outcome)
            index = ProvenanceIndex(bound)
            build_s = time.perf_counter() - start
            start = time.perf_counter()
            plan = _Plan(bound)
            plan_s = time.perf_counter() - start
            nb_index = accel.NumbaIndex(index)
            py = ScmOnlinePy(scm, unit, T, outcome)
            full = ScmNumba(scm, unit, T, outcome)
            loads = workloads(rng, lo, hi, args.steps)
            tau = Session(nb_index, NumbaWalk(bound, loads["jump"][0], plan=plan), loads["jump"][0],
                          calibrate=loads["jump"][:200]).tau
            for name, seq in loads.items():
                n = len(seq)
                row = dict(dataset=args.dataset, family=fam, d=d, unit=u, workload=name, steps=n,
                           mediators=len(py.mechs) - 1, boxes=len(bound.boxes), trees=len(plan.trees),
                           walk_depth=plan.depth, build_s=build_s, plan_s=plan_s, tau=tau)
                ref = np.array([nb_index.query(t) for t in seq])
                walk = NumbaWalk(bound, seq[0], plan=plan)
                got = np.array([walk.move(t) for t in seq])
                row["walk_err"] = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
                row["relocated_per_step"] = walk.relocated / n
                sample = seq[:: max(1, n // 200)]
                truth = np.array([py.query(t)[0] for t in sample])
                row["index_err"] = float(np.max(np.abs(ref[:: max(1, n // 200)] - truth) /
                                                 np.maximum(1.0, np.abs(truth))))
                if row["walk_err"] > 1e-9 or row["index_err"] > 1e-9:
                    raise AssertionError(f"{args.dataset} {fam} d={d} {name}: {row}")
                inc = ScmIncremental(scm, unit, T, outcome, seq[0])
                if not all(inc.move(t) == full.query(t) for t in seq[:200]):
                    raise AssertionError("incremental baseline differs from full propagation")

                mk_walk = lambda: NumbaWalk(bound, seq[0], plan=plan)
                mk_inc = lambda: ScmIncremental(scm, unit, T, outcome, seq[0])
                mk_ses = lambda: Session(nb_index, NumbaWalk(bound, seq[0], plan=plan), seq[0], tau=tau)
                ses = mk_ses()
                got = np.array([ses.move(t) for t in seq])
                row["session_err"] = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
                if row["session_err"] > 1e-9:
                    raise AssertionError(f"session differs by {row['session_err']}")
                row["session_walked"] = ses.walked / n
                row["session_us"] = session_seconds(mk_ses, seq) * 1e6
                row["walk_nb_us"] = session_seconds(mk_walk, seq) * 1e6
                row["index_nb_us"] = per_call_seconds(lambda: [nb_index.query(t) for t in seq], n) * 1e6
                row["incremental_nb_us"] = session_seconds(mk_inc, seq) * 1e6
                row["full_nb_us"] = per_call_seconds(lambda: [full.query(t) for t in seq], n) * 1e6
                lat = {"session_nb": per_query_us(mk_ses().move, seq), "walk_nb": per_query_us(mk_walk().move, seq), "index_nb": per_query_us(nb_index.query, seq),
                       "incremental_nb": per_query_us(mk_inc().move, seq), "full_nb": per_query_us(full.query, seq)}
                k = min(args.py_steps, n)
                pseq = seq[:k]
                row["walk_py_us"] = session_seconds(lambda: Walk(bound, pseq[0], plan=plan), pseq) * 1e6
                row["index_py_us"] = per_call_seconds(lambda: [index.query(t) for t in pseq], k) * 1e6
                row["full_py_us"] = per_call_seconds(lambda: [py.query(t) for t in pseq], k, repeats=1) * 1e6
                for m, v in lat.items():
                    for q in (25, 50, 75, 95):
                        row[f"{m}_p{q}_us"] = float(np.percentile(v, q))
                np.savez_compressed(lat_dir / f"{args.dataset}_{fam}_d{d}_u{u}_{name}.npz",
                                    **{m: v.astype(np.float32) for m, v in lat.items()})
                rows.append(row)
                write_csv(f"session_{args.dataset}.csv", rows)
                print(f"{args.dataset} {fam} d={d} u{u} {name:6s} med={row['mediators']} reloc/step="
                      f"{row['relocated_per_step']:.1f} | nb session {row['session_us']:.2f} walk {row['walk_nb_us']:.2f} index {row['index_nb_us']:.2f} "
                      f"incr {row['incremental_nb_us']:.2f} full {row['full_nb_us']:.1f} | py walk "
                      f"{row['walk_py_us']:.1f} index {row['index_py_us']:.1f} full {row['full_py_us']:.0f}", flush=True)


def session_seconds(make, seq, repeats=3):
    """Mean seconds per step of a session; the session is created outside the timed loop."""
    samples = []
    for _ in range(repeats):
        s = make()
        start = time.perf_counter()
        for t in seq:
            s.move(t)
        samples.append((time.perf_counter() - start) / len(seq))
    return float(np.median(samples))


if __name__ == "__main__":
    main()
