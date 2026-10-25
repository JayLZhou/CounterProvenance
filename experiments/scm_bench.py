#!/usr/bin/env python3
"""Counterfactual queries on the 20 causal datasets (``causal_datasets.py``).

For one dataset and one model family: fit a tree-ensemble mechanism for the
outcome and every non-root ancestor of it, fix a random unit, choose ``d``
treatments among the outcome's ancestors (preferring those the outcome's
mechanism splits on most, then the remaining ancestors), build the index
through the DAG (``scm.bind_scm``), check it query by query against
propagation with the libraries' ``predict`` (``scm.ScmOnline``), and time both.
Settings whose mediator grids or boxes exceed the limits are reported as
infeasible.  Output: ``results/scm_<tag>.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import Infeasible, ScmOnline, ancestors, bind_scm, fit_scm, parents_of


def zoo(n_trees: int, threads: int):
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    out = {
        "GBR": lambda: GradientBoostingRegressor(n_estimators=n_trees, max_depth=3, learning_rate=0.05, random_state=0),
        "HGB": lambda: HistGradientBoostingRegressor(max_iter=n_trees, learning_rate=0.05, max_leaf_nodes=31,
                                                     early_stopping=False, random_state=0),
        "RF": lambda: RandomForestRegressor(n_estimators=max(50, n_trees // 4), max_depth=10, max_features=0.33,
                                            random_state=0, n_jobs=threads),
    }
    import xgboost as xgb
    import lightgbm as lgb

    out["XGB"] = lambda: xgb.XGBRegressor(n_estimators=n_trees, max_depth=6, learning_rate=0.05, tree_method="hist",
                                          random_state=0, n_jobs=threads)
    out["LGBM"] = lambda: lgb.LGBMRegressor(n_estimators=n_trees, num_leaves=31, learning_rate=0.05, random_state=0,
                                            verbose=-1, n_jobs=threads)
    return out


def choose_treatments(scm, outcome, d):
    anc = ancestors(scm.parents, outcome)
    usage = Counter()
    ens = scm.ensembles[outcome]
    ps = scm.parents[outcome]
    for tree in ens.trees:
        for f in tree.feature:
            if f >= 0:
                usage[ps[f]] += 1
    ranked = [n for n, _ in usage.most_common() if n in anc]
    ranked += sorted(anc - set(ranked))
    return ranked[:d] if len(ranked) >= d else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory written by causal_datasets.py")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", default="LGBM")
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--dims", default="1,2,4,8,16")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--train-rows", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=8, help="threads for fitting only; queries are single-threaded")
    ap.add_argument("--max-boxes", type=int, default=5_000_000)
    ap.add_argument("--max-cells", type=int, default=5_000_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    outcome = str(meta["target"])
    parents = parents_of(edges)
    any_t = [sorted(ancestors(parents, outcome))[0]]
    start = time.perf_counter()
    scm = fit_scm(df, edges, outcome, any_t, zoo(args.trees, args.threads)[args.family],
                  max_rows=args.train_rows, seed=0)
    fit_s = time.perf_counter() - start
    print(f"{args.dataset} {args.family}: fitted {len(scm.models)} mechanisms in {fit_s:.0f}s", flush=True)

    rng = np.random.default_rng(2028)
    unit = df.iloc[int(rng.integers(len(df)))].to_dict()
    rows = []
    for d in (int(v) for v in args.dims.split(",")):
        T = choose_treatments(scm, outcome, d)
        if T is None:
            continue
        row = {"dataset": args.dataset, "family": args.family, "trees": args.trees, "d": d,
               "treatments": "|".join(T), "mechanisms": len(scm.models), "fit_s": fit_s}
        online = ScmOnline(scm, unit, T, outcome)
        row["mediators"] = len(online.order) - 1
        start = time.perf_counter()
        try:
            bound = bind_scm(scm, unit, T, outcome, max_boxes=args.max_boxes, max_cells=args.max_cells)
            index = ProvenanceIndex(bound)
        except (Infeasible, MemoryError) as e:
            row["status"] = f"infeasible: {e}"
            rows.append(row)
            print(f"  d={d:2d} mediators={row['mediators']} INFEASIBLE ({e})", flush=True)
            write_csv(f"scm_{args.tag or args.dataset + '_' + args.family}.csv", rows)
            continue
        row["build_s"] = time.perf_counter() - start
        row["boxes"] = len(bound.boxes)
        row["groups"] = len(index.groups)
        lo = df[T].quantile(0.01).to_numpy()
        hi = df[T].quantile(0.99).to_numpy()
        pts = list(rng.uniform(lo, hi, size=(args.queries, d)))
        th = [[b.hi[a] for b in bound.boxes if np.isfinite(b.hi[a])] for a in range(d)]
        for _ in range(args.queries // 3):
            p = rng.uniform(lo, hi)
            a = int(rng.integers(d))
            if th[a]:
                p[a] = th[a][int(rng.integers(len(th[a])))]
            pts.append(p)
        T_arr = np.asarray(pts)
        err = 0.0
        for t in T_arr:
            ref = online.query(t)
            err = max(err, abs(index.query(t).value - ref) / max(1.0, abs(ref)))
        row["max_rel_err"] = err
        n = len(T_arr)
        row["ours_us"] = per_call_seconds(lambda: [index.query(t) for t in T_arr], n) * 1e6
        row["online_us"] = per_call_seconds(lambda: [online.query(t) for t in T_arr], n, repeats=1) * 1e6
        row["status"] = "ok"
        rows.append(row)
        print(f"  d={d:2d} mediators={row['mediators']:3d} boxes={row['boxes']:>9,} groups={row['groups']:5d} "
              f"build={row['build_s']:7.1f}s ours={row['ours_us']:8.1f}us online={row['online_us']:9.1f}us "
              f"({row['online_us'] / row['ours_us']:7.1f}x) err={err:.1e}", flush=True)
        write_csv(f"scm_{args.tag or args.dataset + '_' + args.family}.csv", rows)


if __name__ == "__main__":
    main()
