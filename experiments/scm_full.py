#!/usr/bin/env python3
"""Full benchmark of counterfactual queries on the causal datasets.

For one dataset and one model family, every non-root ancestor of the outcome
gets a tree-ensemble mechanism (fitted once, multi-threaded, on at most
``--train-rows`` rows).  For every ``d`` and every unit (``--units`` random
units), the treatments are ``d`` ancestors of the outcome (those the outcome's
mechanism splits on most first), the index is built through the DAG, and each
query is answered by Tessera (``ProvenanceIndex.query``, pure Python with numpy), by the
textbook three-step counterfactual in pure Python that caches nothing (``ScmThreeStepPy``:
abduction, action and prediction on every query), by ChiRho (``chirho_baseline.ScmChiRho``,
when ``chirho`` is importable) and by DoWhy-GCM over the same DAG and the same fitted models,
all on ``--queries`` queries (``--dowhy-queries`` for the two libraries).  ``--extra-baselines`` also times pure-Python propagation
(``ScmOnlinePy``); it is not reported in the paper.

A value is computed as ``u + (f(pa') - f(pa))`` (see :mod:`counterprovenance.scm`).

Every answer is checked against the pure-Python propagation, which sums the
leaves exactly; that propagation is in turn checked against the libraries' own
``predict`` (``ScmOnline``), exactly for the float64 libraries and recorded as
``err_lib`` for XGBoost, whose ``predict`` accumulates in single precision.
Queries are timed single-threaded.
Output: ``results/scm_full_<tag>.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, per_call_seconds, per_query_us, write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import Infeasible, ScmOnline, ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmOnlinePy, ScmThreeStepPy
from scm_bench import choose_treatments, zoo

try:
    from chirho_baseline import ScmChiRho

    import chirho  # noqa: F401

    HAVE_CHIRHO = True
except ImportError:
    HAVE_CHIRHO = False


def dowhy_model(scm, edges):
    import networkx as nx
    from dowhy import gcm
    from dowhy.gcm.causal_models import PARENTS_DURING_FIT
    from dowhy.gcm.ml import SklearnRegressionModel
    from dowhy.graph import get_ordered_predecessors

    gcm.config.disable_progress_bars()
    g = nx.DiGraph(edges)
    keep = set(scm.models) | {p for n in scm.models for p in scm.parents[n]}
    g = g.subgraph(keep).copy()
    cg = gcm.InvertibleStructuralCausalModel(g)
    for n in g.nodes:
        ps = get_ordered_predecessors(cg.graph, n)
        if ps:
            if list(ps) != scm.parents[n]:
                raise ValueError(f"DoWhy parent order differs at {n}")
            cg.set_causal_mechanism(n, gcm.AdditiveNoiseModel(SklearnRegressionModel(scm.models[n])))
        else:
            cg.set_causal_mechanism(n, gcm.EmpiricalDistribution())
        cg.graph.nodes[n][PARENTS_DURING_FIT] = ps
    return gcm, cg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--dims", default="1,2,4,8,16")
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--queries", type=int, default=10_000,
                    help="queries per unit and d (3/4 uniform, 1/4 on box boundaries)")
    ap.add_argument("--dowhy-queries", type=int, default=10_000)
    ap.add_argument("--train-rows", type=int, default=50_000)
    ap.add_argument("--threads", type=int, default=1, help="threads per model fit (fitting only)")
    ap.add_argument("--fit-jobs", type=int, default=4, help="mechanisms fitted in parallel (fitting only)")
    ap.add_argument("--max-boxes", type=int, default=10**15, help="no limit by default")
    ap.add_argument("--max-mask-bytes", type=float, default=float("inf"), help="no limit by default")
    ap.add_argument("--build-threads", type=int, default=1, help="threads of the C++ leaf-box compiler")
    ap.add_argument("--extra-baselines", action="store_true",
                    help="also time pure-Python propagation (not reported in the paper)")
    ap.add_argument("--unit-ids", default="", help="comma-separated unit indices to run (default: all)")
    ap.add_argument("--grouping", default="tables")
    ap.add_argument("--budget", type=float, default=2e7, help="table cells of the greedy cell graphs")
    ap.add_argument("--ours-only", action="store_true", help="time only Tessera (exactness is still checked)")
    ap.add_argument("--baselines-only", action="store_true",
                    help="time only the baselines: the leaf boxes are compiled (they place the boundary queries), "
                         "the index is not built")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or f"{args.dataset}_{args.family}"
    # a run of some units only writes its own CSV (the latency files are per unit anyway)
    csv_name = f"scm_full_{tag}_u{args.unit_ids.replace(',', '-')}.csv" if args.unit_ids else f"scm_full_{tag}.csv"

    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    outcome = str(meta["target"])
    parents = parents_of(edges)
    start = time.perf_counter()
    scm = fit_scm(df, edges, outcome, [sorted(ancestors(parents, outcome))[0]],
                  zoo(args.trees, args.threads)[args.family], max_rows=args.train_rows, seed=0,
                  n_jobs=args.fit_jobs)
    fit_s = time.perf_counter() - start
    print(f"{args.dataset} {args.family}: {len(scm.models)} mechanisms fitted in {fit_s:.0f}s", flush=True)

    dims = [d for d in (int(v) for v in args.dims.split(",")) if choose_treatments(scm, outcome, d)]
    gcm, cg = dowhy_model(scm, edges)

    # every unit and every (unit, d) draws from its own seeded generator, so that jobs split by unit or d
    # reproduce the same units and queries
    unit_idx = np.random.default_rng(2028).integers(len(df), size=args.units)
    units_run = [int(u) for u in args.unit_ids.split(",")] if args.unit_ids else range(args.units)
    rows = []
    for u in units_run:
        unit_row = df.iloc[[int(unit_idx[u])]]
        unit = unit_row.iloc[0].to_dict()
        for d in dims:
            rng = np.random.default_rng([2028, u, d])
            T = choose_treatments(scm, outcome, d)
            ref = ScmOnline(scm, unit, T, outcome)
            row = {"dataset": args.dataset, "family": args.family, "trees": args.trees, "unit": u, "d": d,
                   "treatments": "|".join(T), "mechanisms": len(scm.models), "path_mechanisms": len(ref.order),
                   "mediators": len(ref.order) - 1, "fit_s": fit_s}
            start = time.perf_counter()
            try:
                bound = bind_scm(scm, unit, T, outcome, max_boxes=args.max_boxes, threads=args.build_threads)
                compile_s = time.perf_counter() - start
                index = None if args.baselines_only else ProvenanceIndex(bound, grouping=args.grouping, budget=int(args.budget),
                                                                         max_mask_bytes=args.max_mask_bytes,
                                                                         build_threads=args.build_threads)
            except (Infeasible, MemoryError) as e:
                row["status"] = f"infeasible: {e}"
                rows.append(row)
                write_csv(csv_name, rows)
                print(f"  unit {u} d={d:2d} INFEASIBLE ({e})", flush=True)
                continue
            row["compile_s"] = compile_s
            row["build_s"] = time.perf_counter() - start
            row["boxes"] = len(bound.boxes)
            if index is not None:
                st = index.stats
                row.update(groups=len(index.groups), bitmask_trees=st.bitmask_trees, bytes=st.packed_bytes,
                           choice=index.choice)
            lo = df[T].quantile(0.01).to_numpy()
            hi = df[T].quantile(0.99).to_numpy()
            n_unif = args.queries * 3 // 4
            pts = list(rng.uniform(lo, hi, size=(n_unif, d)))
            th = [[b.hi[a] for b in bound.boxes if np.isfinite(b.hi[a])] for a in range(d)]
            for _ in range(args.queries - n_unif):
                p = rng.uniform(lo, hi)
                a = int(rng.integers(d))
                if th[a]:
                    p[a] = th[a][int(rng.integers(len(th[a])))]
                pts.append(p)
            Q = np.ascontiguousarray(pts, dtype=np.float64)
            n = len(Q)
            py = ScmOnlinePy(scm, unit, T, outcome)
            truth = np.array([py.query(t)[0] for t in Q])
            lib = np.array([ref.query(t) for t in Q])
            row["err_lib"] = float(np.max(np.abs(lib - truth) / np.maximum(1.0, np.abs(truth))))
            if args.family != "XGB" and row["err_lib"] > 1e-9:
                raise AssertionError(f"{args.dataset} {args.family} d={d}: library differs by {row['err_lib']}")
            tol = 1e-9
            # answers of library-level evaluators (Treelite, DoWhy) count as wrong beyond the
            # library's own rounding: float64, or float32 accumulation for XGBoost
            if args.family == "XGB":
                ens = scm.ensembles[outcome]
                scale = abs(ens.base) + sum(float(np.abs(tr.value).max()) for tr in ens.trees)
                lib_tol = np.full(n, 4 * np.sqrt(len(ens.trees)) * 2.0 ** -24 * scale)
            else:
                lib_tol = 1e-9 * np.maximum(1.0, np.abs(lib))

            def check(name, got):
                err = float(np.max(np.abs(np.asarray(got) - truth[: len(got)]) / np.maximum(1.0, np.abs(truth[: len(got)]))))
                row[f"err_{name}"] = err
                if err > tol:
                    raise AssertionError(f"{args.dataset} {args.family} d={d}: {name} differs by {err}")

            lat = {}
            if index is not None:
                check("ours_py", [index.query(t).value for t in Q])
                row["ours_py_us"] = per_call_seconds(lambda: [index.query(t) for t in Q], n) * 1e6
                lat["ours_py"] = per_query_us(index.query, Q)
            if args.ours_only:
                for q in (25, 50, 75, 95):
                    row[f"ours_py_p{q}_us"] = float(np.percentile(lat["ours_py"], q))
                lat_dir = RESULTS / "scm_lat_ours"
                lat_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(lat_dir / f"{tag}_u{u}_d{d}.npz", ours_py=lat["ours_py"].astype(np.float32))
                row["status"] = "ok"
                rows.append(row)
                write_csv(csv_name, rows)
                print(f"  unit {u} d={d:2d} {index.choice} groups={row['groups']} | ours {row['ours_py_us']:.1f}us",
                      flush=True)
                continue
            three = ScmThreeStepPy(scm, unit, T, outcome)
            check("three_step", [three.query(t)[0] for t in Q])
            row["three_step_us"] = per_call_seconds(lambda: [three.query(t) for t in Q], n, repeats=1) * 1e6
            lat["three_step"] = per_query_us(three.query, Q)
            if args.extra_baselines:
                row["online_py_us"] = per_call_seconds(lambda: [py.query(t) for t in Q], n) * 1e6
                lat["online_py"] = per_query_us(py.query, Q)
            k = min(args.dowhy_queries, n)
            if HAVE_CHIRHO:
                ch = ScmChiRho(scm, unit, T, outcome)
                got = [ch.query(t) for t in Q[:k]]
                row["chirho_wrong"] = int((np.abs(np.asarray(got) - lib[:k]) > lib_tol[:k]).sum())
                lat["chirho"] = per_query_us(ch.query, Q[:k])
                row["chirho_us"] = float(lat["chirho"].mean())
            ivs = [{name: (lambda _v, value=float(t[j]): value) for j, name in enumerate(T)} for t in Q[:k]]
            got = [float(gcm.counterfactual_samples(cg, iv, observed_data=unit_row)[outcome].iloc[0]) for iv in ivs]
            # DoWhy abducts N = u - f(pa) and adds it back, so rounding can move an observed mediator
            # across a threshold of its child: count such answers instead of failing
            row["dowhy_wrong"] = int((np.abs(np.asarray(got) - lib[:k]) > lib_tol[:k]).sum())
            lat["dowhy"] = per_query_us(lambda iv: gcm.counterfactual_samples(cg, iv, observed_data=unit_row), ivs)
            row["dowhy_us"] = float(lat["dowhy"].mean())
            row["queries"], row["dowhy_queries"] = n, k
            for name, v in lat.items():
                for q in (25, 50, 75, 95):
                    row[f"{name}_p{q}_us"] = float(np.percentile(v, q))
            lat_dir = RESULTS / "scm_lat"
            lat_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(lat_dir / f"{tag}_u{u}_d{d}.npz", **{k_: v.astype(np.float32) for k_, v in lat.items()})
            row["status"] = "ok"
            rows.append(row)
            write_csv(csv_name, rows)
            print(f"  unit {u} d={d:2d} med={row['mediators']:3d} boxes={row['boxes']:>9,} groups={row.get('groups', -1):5d} "
                  f"build={row['build_s']:6.1f}s | ours {row.get('ours_py_us', float('nan')):7.1f}us | three-step "
                  f"{row['three_step_us']:8.0f}us | chirho {row.get('chirho_us', float('nan')):9.0f}us | dowhy "
                  f"{row['dowhy_us']:9.0f}us", flush=True)


if __name__ == "__main__":
    main()
