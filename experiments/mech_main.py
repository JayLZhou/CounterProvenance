#!/usr/bin/env python3
"""The main experiment: one index per mechanism, 1,000 random units with 10 interventions each, one query at a time.

The backend ``twin`` builds the index once, in one thread, before any of the units; the three baselines answer the same
queries.  A baseline stops after ``--hours``; what it reports is what really ran.  Every process runs in one thread.
Output: ``results/main/<dataset>_<family>_d<d>_<backend>.csv`` and ``.npy`` with the answers."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.scm import ancestors, fit_scm, parents_of, relevant_nodes
from counterprovenance.scm_baselines import ScmThreeStepPy
from mech_proto import MechanismIndex
from scm_bench import choose_treatments, zoo

DIMS = (1, 2, 4, 8, 16)


def results_dir(args):
    if args.out:
        return RESULTS / args.out
    return RESULTS / "main" if args.trees == 1000 else RESULTS / "trees" / str(args.trees)


def prepare(args):
    """The fitted SCM and the workload of a table and a family, kept so that every process sees the same ones."""
    cache = (RESULTS / "main" if args.trees == 1000 else results_dir(args)) / "cache" / f"{args.dataset}_{args.family}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    t0 = time.perf_counter()
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], zoo(args.trees, 1)[args.family], max_rows=50_000,
                  seed=0, n_jobs=4)
    fit_s = time.perf_counter() - t0
    idx = np.random.default_rng(2028).integers(len(df), size=args.units)
    work = {}
    for d in DIMS:
        T = choose_treatments(scm, y, d)
        if T:
            work[d] = dict(T=T, lo=df[T].quantile(0.01).to_numpy(), hi=df[T].quantile(0.99).to_numpy())
    state = dict(scm=scm, y=y, edges=edges, rows=df.iloc[idx].reset_index(drop=True), work=work, fit_s=fit_s, table_rows=len(df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(state))
    tmp.replace(cache)
    return state


def queries(state, d, per_unit):
    """For every unit, in order: values drawn uniformly between the 1st and 99th percentiles."""
    w = state["work"][d]
    rng = np.random.default_rng([2028, d])
    for _ in range(len(state["rows"])):
        yield rng.uniform(w["lo"], w["hi"], size=(per_unit, d))


def run_twin(args, state, out):
    scm, y, rows, work = state["scm"], state["y"], state["rows"], state["work"]
    nodes = sorted({x for w in work.values() for x in relevant_nodes(scm.parents, w["T"], y)})
    t0 = time.perf_counter()
    try:
        mech = MechanismIndex(scm, rows.iloc[0].to_dict(), nodes, args.budget, 1, args.one_pairs)
    except MemoryError as e:
        pd.DataFrame([dict(dataset=args.dataset, family=args.family, status=f"infeasible: {str(e)[:80]}", build_s=time.perf_counter() - t0)]).to_csv(out / f"{args.dataset}_{args.family}_build.csv", index=False)
        return
    build_s = time.perf_counter() - t0
    pd.DataFrame([dict(dataset=args.dataset, family=args.family, status="ok", node=x, **v) for x, v in mech.info.items()]).to_csv(out / f"{args.dataset}_{args.family}_build.csv", index=False)
    for d, w in work.items():
        T = w["T"]
        order = relevant_nodes(scm.parents, T, y)
        t0 = time.perf_counter()
        plan = mech.plan(T, order)
        plan_s = time.perf_counter() - t0
        q_abduct = q_bind = q_full = q_partial = 0.0
        n = 0
        got, lat_full, lat_partial = [], [], []
        for u, Q in enumerate(queries(state, d, args.per_unit)):
            unit = rows.iloc[u].to_dict()
            t0 = time.perf_counter()
            observed = mech.observed(unit, order)
            q_abduct += time.perf_counter() - t0
            t0 = time.perf_counter()
            bound = mech.bind(plan, unit, observed)
            q_bind += time.perf_counter() - t0
            for t in Q:
                t0 = time.perf_counter()
                got.append(mech.query(unit, T, t, order, observed))
                lat_full.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                mech.query_partial(unit, T, t, bound)
                lat_partial.append(time.perf_counter() - t0)
            n += len(Q)
        f, p = np.array(lat_full) * 1e6, np.array(lat_partial) * 1e6
        r = dict(dataset=args.dataset, family=args.family, d=d, backend="twin", units=len(rows), queries=n, path=len(order), table_rows=state["table_rows"],
                 build_s=build_s, plan_s=plan_s, fit_s=state["fit_s"], groups=sum(mech.info[x]["groups"] for x in order), read_groups=sum(len(pl[2].touched) for pl in plan),
                 bytes=sum(mech.info[x]["bytes"] for x in order), abduct_us=q_abduct / len(rows) * 1e6, bind_us=q_bind / len(rows) * 1e6,
                 full_us=float(f.mean()), full_p50_us=float(np.median(f)), full_p95_us=float(np.quantile(f, .95)),
                 partial_us=float(p.mean()), partial_p50_us=float(np.median(p)), partial_p95_us=float(np.quantile(p, .95)), status="ok")
        np.save(out / f"{args.dataset}_{args.family}_d{d}_twin.npy", np.array(got))
        np.save(out / f"{args.dataset}_{args.family}_d{d}_twin_lat.npy", np.stack([f, p]).astype(np.float32))
        pd.DataFrame([r]).to_csv(out / f"{args.dataset}_{args.family}_d{d}_twin.csv", index=False)
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k not in ("dataset", "family", "backend", "status")}), flush=True)


def run_baseline(args, state, out):
    d = args.d
    scm, y, rows, T = state["scm"], state["y"], state["rows"], state["work"][d]["T"]
    tag = out / f"{args.dataset}_{args.family}_d{d}_{args.backend}"
    if args.backend == "dowhy":
        from scm_full import dowhy_model
        gcm, cg = dowhy_model(scm, state["edges"])
    elif args.backend == "chirho":
        from chirho_baseline import ScmChiRho
    done = n = 0
    spent = 0.0
    got: list[float] = []
    lat: list[float] = []
    lat_path = tag.with_name(tag.name + "_lat.npy")
    if tag.with_suffix(".csv").exists():
        s = pd.read_csv(tag.with_suffix(".csv")).iloc[0]
        done, n, spent = int(s.units_done), int(s.queries), float(s.query_s)
        got = np.load(tag.with_suffix(".npy")).tolist()
        lat = np.load(lat_path).tolist() if lat_path.exists() else []
    start = time.perf_counter()
    status = "ok"

    def save():
        pd.DataFrame([dict(dataset=args.dataset, family=args.family, d=d, backend=args.backend, units=len(rows), units_done=done, queries=n, query_s=spent,
                           mean_us=spent / max(n, 1) * 1e6, status=status)]).to_csv(tag.with_suffix(".csv"), index=False)
        np.save(tag.with_suffix(".npy"), np.array(got))
        np.save(lat_path, np.array(lat, dtype=np.float32))

    for u, Q in enumerate(queries(state, d, args.per_unit)):
        if u < done:
            continue
        if spent + 0.0 > args.hours * 3600:
            status = f"stopped at {args.hours:g} hours"
            break
        row = rows.iloc[[u]]
        unit = row.iloc[0].to_dict()
        if args.backend == "three_step":
            three = ScmThreeStepPy(scm, unit, T, y)
            call = lambda t: three.query(t)[0]
        elif args.backend == "chirho":
            ch = ScmChiRho(scm, unit, T, y)
            call = ch.query
        else:
            call = lambda t: float(gcm.counterfactual_samples(cg, {name: (lambda _v, value=float(t[k]): value) for k, name in enumerate(T)}, observed_data=row)[y].iloc[0])
        for t in Q:
            t0 = time.perf_counter()
            v = call(t)
            took = time.perf_counter() - t0
            spent += took
            lat.append(took * 1e6)
            got.append(float(v))
            n += 1
        done = u + 1
        if done % 25 == 0:
            save()
    save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--backend", required=True, choices=["twin", "three_step", "chirho", "dowhy"])
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--out", default="", help="directory under results, for a rerun on the cached models")
    ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--per-unit", type=int, default=10)
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--one-pairs", type=float, default=5e10, help="an ensemble whose grid times its trees is at most this is one group")
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    out = results_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    state = prepare(args)
    if args.backend == "twin":
        run_twin(args, state, out)
    elif args.d in state["work"]:
        run_baseline(args, state, out)


if __name__ == "__main__":
    main()
