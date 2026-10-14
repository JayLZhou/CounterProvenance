#!/usr/bin/env python3
"""Query benchmark of the compiled index against online evaluation.

For one dataset and one model (trained single-threaded), a fixed unit, and
``d`` intervened features (the ones the ensemble splits on most), compare per
query, everything single-threaded:

* pure Python - ours (``ProvenanceIndex.query``) vs the online loop over all
  trees and over the bound trees only (ablation);
* numba - ours (``NumbaIndex``) vs the numba online loop, one call per query
  and a batch inside compiled code;
* libraries - ours (numba, one call per query) vs ``model.predict`` on one row,
  the library's fastest single-row call, and batch ``predict`` per row;
* sliders - one feature changes per query: ``QuerySession`` vs the
  incremental online loop.

Values are checked against the online loop and the library's own prediction,
provenance against brute-force stabbing.  Output: ``results/qbench_<tag>.csv``.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter

import numpy as np

from common import environment, load_dataset, paper_zoo, per_call_seconds, value_tolerance, write_csv
from counterprovenance import IncrementalLoop, OnlineLoop, ProvenanceIndex, bind, from_model
from counterprovenance import accel


def query_points(X, axes, bound, rng, count):
    """``count`` uniform points inside the 1-99% range plus ``count // 3`` points
    with one coordinate exactly on a split threshold."""
    lo = np.percentile(X[:, axes], 1, axis=0)
    hi = np.percentile(X[:, axes], 99, axis=0)
    pts = list(rng.uniform(lo, hi, size=(count, len(axes))))
    for _ in range(count // 3):
        p = rng.uniform(lo, hi)
        a = int(rng.integers(len(axes)))
        th = [b.hi[a] for b in bound.boxes if math.isfinite(b.hi[a])]
        if th:
            p[a] = th[int(rng.integers(len(th)))]
        pts.append(p)
    return np.array(pts)


def single_row_calls(model):
    """(name, fn(row_2d)) for the library's single-row prediction paths."""
    calls = [("predict", lambda r: model.predict(r))]
    module = type(model).__module__
    if module.startswith("xgboost"):
        booster = model.get_booster()
        calls.append(("inplace_predict", lambda r: booster.inplace_predict(r)))
    elif module.startswith("lightgbm"):
        booster = model.booster_
        calls.append(("booster.predict", lambda r: booster.predict(r)))
    return calls


def bench(dataset, title, X, y, name, model, dims, n_queries, budget, rng) -> list[dict]:
    ensemble = from_model(model)
    usage = Counter(int(f) for t in ensemble.trees for f in t.feature if f >= 0)
    order = [f for f, _ in usage.most_common()] + [f for f in range(X.shape[1]) if f not in usage]
    unit_index = int(rng.integers(len(X)))
    unit = X[unit_index]
    noise = float(y[unit_index] - model.predict(unit[None, :])[0])
    tol = value_tolerance(model)
    rows = []
    for d in dims:
        if d > len(usage):
            continue
        axes = sorted(order[:d])
        bound = bind(ensemble, unit, axes, noise=noise)
        pts = query_points(X, axes, bound, rng, n_queries)
        T = np.ascontiguousarray(pts, dtype=np.float64)
        full = np.tile(unit, (len(T), 1))
        full[:, axes] = T
        full_rows = [full[q:q + 1] for q in range(len(full))]

        start = time.perf_counter()
        index = ProvenanceIndex(bound, budget=budget)
        build_s = time.perf_counter() - start
        st = index.stats
        loop_all = OnlineLoop(ensemble, unit, axes, noise=noise)
        loop_bound = OnlineLoop(ensemble, unit, axes, bound=bound)

        # --- exactness -------------------------------------------------
        lib = model.predict(full) + noise
        err_loop = err_lib = 0.0
        for q in range(len(T)):
            v = index.query(T[q]).value
            ref = loop_all.query(T[q])[0]
            err_loop = max(err_loop, abs(v - ref) / max(1.0, abs(ref)))
            err_lib = max(err_lib, abs(v - lib[q]) / max(1.0, abs(lib[q])))
        if err_loop > 1e-12 or err_lib > max(tol, 1e-9):
            raise AssertionError(f"{dataset} {name} d={d}: err vs loop {err_loop:.2e}, vs library {err_lib:.2e}")
        for q in range(0, len(T), max(1, len(T) // 12)):
            expected = Counter(bound.label(i) for i in bound.selected(T[q]))
            if index.query(T[q]).labels() != expected:
                raise AssertionError(f"{dataset} {name} d={d}: provenance mismatch at query {q}")

        n = len(T)
        row = {
            "dataset": dataset, "title": title, "features": X.shape[1], "model": name,
            "trees": ensemble.tree_count, "d": d, "varying_trees": len(loop_bound.trees),
            "boxes": len(bound.boxes), "groups": len(index.groups), "bitmask_trees": st.bitmask_trees,
            "records": st.records, "bytes": st.packed_bytes, "table_cells": st.table_cells,
            "build_s": build_s, "queries": n, "max_rel_err_vs_loop": err_loop, "max_rel_err_vs_library": err_lib,
        }
        # --- pure Python ----------------------------------------------
        row["py_ours_us"] = per_call_seconds(lambda: [index.query(t) for t in T], n) * 1e6
        row["py_loop_all_us"] = per_call_seconds(lambda: [loop_all.query(t) for t in T], n) * 1e6
        row["py_loop_bound_us"] = per_call_seconds(lambda: [loop_bound.query(t) for t in T], n) * 1e6
        # --- numba ----------------------------------------------------
        if accel.HAVE_NUMBA:
            nb = accel.NumbaIndex(index)
            nb_all = accel.NumbaLoop(ensemble, unit, axes, noise=noise)
            nb_bound = accel.NumbaLoop(ensemble, unit, axes, bound=bound)
            for f in (nb.query, nb_all.query, nb_bound.query):
                f(T[0])
            got = nb.batch(T)
            ref = nb_all.batch(T)
            if np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))) > 1e-12:
                raise AssertionError(f"{dataset} {name} d={d}: numba index differs from numba loop")
            row["nb_ours_call_us"] = per_call_seconds(lambda: [nb.query(t) for t in T], n) * 1e6
            row["nb_ours_batch_us"] = per_call_seconds(lambda: nb.batch(T), n) * 1e6
            row["nb_loop_all_call_us"] = per_call_seconds(lambda: [nb_all.query(t) for t in T], n) * 1e6
            row["nb_loop_all_batch_us"] = per_call_seconds(lambda: nb_all.batch(T), n) * 1e6
            row["nb_loop_bound_batch_us"] = per_call_seconds(lambda: nb_bound.batch(T), n) * 1e6
        # --- libraries ------------------------------------------------
        calls = single_row_calls(model)
        for label, fn in calls:
            row[f"lib_{label}_us"] = per_call_seconds(lambda: [fn(r) for r in full_rows], n) * 1e6
        row["lib_single_best_us"] = min(row[f"lib_{label}_us"] for label, _ in calls)
        row["lib_batch_us"] = per_call_seconds(lambda: model.predict(full), n) * 1e6
        # --- sliders --------------------------------------------------
        steps = []
        for _ in range(300):
            k = int(rng.integers(d))
            steps.append((k, float(T[int(rng.integers(n))][k])))

        def run_session():
            s = index.session(T[0])
            return [s.set(k, v) for k, v in steps]

        def run_incremental():
            s = IncrementalLoop(ensemble, bound, T[0])
            return [s.set(k, v) for k, v in steps]

        a, b = run_session(), run_incremental()
        if max(abs(p - q) / max(1.0, abs(q)) for p, q in zip(a, b)) > 1e-12:
            raise AssertionError(f"{dataset} {name} d={d}: session differs from incremental loop")
        row["slider_ours_us"] = per_call_seconds(run_session, len(steps)) * 1e6
        row["slider_loop_us"] = per_call_seconds(run_incremental, len(steps)) * 1e6
        rows.append(row)
        print(
            f"{dataset:5s} {name:12s} d={d:2d} trees={row['varying_trees']:4d}/{row['trees']} groups={row['groups']:4d} "
            f"bitmask={row['bitmask_trees']:4d} rec={row['records']:>11,} build={build_s:6.1f}s | "
            f"py {row['py_ours_us']:6.2f} vs loop {row['py_loop_all_us']:7.1f} ({row['py_loop_all_us'] / row['py_ours_us']:5.1f}x) | "
            + (f"nb {row['nb_ours_batch_us']:6.3f} vs {row['nb_loop_all_batch_us']:6.2f} "
               f"({row['nb_loop_all_batch_us'] / row['nb_ours_batch_us']:5.1f}x) | " if accel.HAVE_NUMBA else "")
            + f"lib {row['lib_single_best_us']:6.1f} vs nb-call {row.get('nb_ours_call_us', float('nan')):5.2f} | "
            f"slider {row['slider_ours_us']:5.2f} vs {row['slider_loop_us']:6.1f}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["cal", "yprop", "topo", "uji", "mars"])
    parser.add_argument("--models", default="", help="comma-separated subset of the paper zoo")
    parser.add_argument("--trees", default="1000", help="comma-separated tree counts")
    parser.add_argument("--dims", default="1,2,4,8,16")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--budget", type=int, default=2_000_000)
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    print(environment(), flush=True)
    X, y, title = load_dataset(args.dataset, args.data_home)
    print(f"{title}: {X.shape}", flush=True)
    dims = [int(v) for v in args.dims.split(",")]
    rows = []
    for n_trees in (int(v) for v in args.trees.split(",")):
        zoo = paper_zoo(n_trees)
        if args.models:
            zoo = {k: v for k, v in zoo.items() if k in args.models.split(",")}
        for name, factory in zoo.items():
            start = time.perf_counter()
            model = factory().fit(X, y)
            print(f"{name} ({n_trees} trees) fitted in {time.perf_counter() - start:.1f}s", flush=True)
            rng = np.random.default_rng(2028)
            rows += bench(args.dataset, title, X, y, name, model, dims, args.queries, args.budget, rng)
            write_csv(f"qbench_{args.tag or args.dataset}.csv", rows)


if __name__ == "__main__":
    main()
