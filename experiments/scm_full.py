#!/usr/bin/env python3
"""Full benchmark of counterfactual queries on the 20 causal datasets.

For one dataset and one model family, every non-root ancestor of the outcome
gets a tree-ensemble mechanism (fitted once, multi-threaded, on at most
``--train-rows`` rows).  For every ``d`` and every unit (``--units`` random
units), the treatments are ``d`` ancestors of the outcome (those the outcome's
mechanism splits on most first), the index is built through the DAG, and each
query is answered by

* pure Python: ours (``ProvenanceIndex.query``) vs ``ScmOnlinePy`` (tree walks
  through every mechanism on the paths, recording the leaves);
* numba, one call per query: ours (``NumbaIndex``) vs ``ScmNumba``;
* Treelite (tl2cgen, gcc -O3): every mechanism compiled to a shared library and
  called in topological order, one call per mechanism and query;
* DoWhy-GCM over the same DAG and the same fitted models (``--dowhy-queries``).

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
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.scm import Infeasible, ScmOnline, ancestors, bind_scm, fit_scm, parents_of, relevant_nodes
from counterprovenance.scm_baselines import ScmNumba, ScmOnlinePy
from scm_bench import choose_treatments, zoo


def treelite_mechanisms(scm, nodes, workdir, threads):
    """Compile the mechanisms of ``nodes`` with tl2cgen; returns node -> (predictor, dtype, tl2cgen)."""
    from baselines_all import treelite_predictor

    out = {}
    for node in nodes:
        sub = Path(workdir) / f"m{len(out)}"
        sub.mkdir(parents=True, exist_ok=True)
        tl2cgen, predictor, dtype, _ = treelite_predictor(scm.models[node], str(sub))
        out[node] = (tl2cgen, predictor, dtype)
    return out


class ScmTreelite:
    def __init__(self, scm, compiled, unit, treatments, outcome):
        self.order = relevant_nodes(scm.parents, treatments, outcome)
        self.unit, self.treatments, self.outcome = dict(unit), list(treatments), outcome
        self.parents, self.compiled, self.ens = scm.parents, compiled, scm.ensembles
        self.observed = {n: self._predict(n, self.unit) for n in self.order}

    def _predict(self, node, values):
        tl2cgen, predictor, dtype = self.compiled[node]
        row = self.ens[node].cast(np.array([[values[p] for p in self.parents[node]]]))
        dm = tl2cgen.DMatrix(np.ascontiguousarray(row, dtype=np.float32 if dtype == "float32" else np.float64),
                             dtype=dtype)
        return float(np.asarray(predictor.predict(dm)).reshape(-1)[0])

    def query(self, t):
        values = dict(self.unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        for node in self.order:
            values[node] = float(self.unit[node]) + (self._predict(node, values) - self.observed[node])
        return values[self.outcome]


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
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--dowhy-queries", type=int, default=40)
    ap.add_argument("--train-rows", type=int, default=50_000)
    ap.add_argument("--threads", type=int, default=1, help="threads per model fit (fitting only)")
    ap.add_argument("--fit-jobs", type=int, default=4, help="mechanisms fitted in parallel (fitting only)")
    ap.add_argument("--max-boxes", type=int, default=5_000_000)
    ap.add_argument("--max-cells", type=int, default=5_000_000)
    ap.add_argument("--no-treelite", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or f"{args.dataset}_{args.family}"

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
    needed = set()
    for d in dims:
        needed |= set(relevant_nodes(scm.parents, choose_treatments(scm, outcome, d), outcome))
    tl_dir = tempfile.TemporaryDirectory()
    compiled = {}
    if not args.no_treelite:
        start = time.perf_counter()
        compiled = treelite_mechanisms(scm, sorted(needed), tl_dir.name, args.threads)
        print(f"  treelite: {len(compiled)} mechanisms compiled in {time.perf_counter() - start:.0f}s", flush=True)
    gcm, cg = dowhy_model(scm, edges)

    rng = np.random.default_rng(2028)
    rows = []
    for u in range(args.units):
        unit_row = df.iloc[[int(rng.integers(len(df)))]]
        unit = unit_row.iloc[0].to_dict()
        for d in dims:
            T = choose_treatments(scm, outcome, d)
            ref = ScmOnline(scm, unit, T, outcome)
            row = {"dataset": args.dataset, "family": args.family, "trees": args.trees, "unit": u, "d": d,
                   "treatments": "|".join(T), "mechanisms": len(scm.models), "path_mechanisms": len(ref.order),
                   "mediators": len(ref.order) - 1, "fit_s": fit_s}
            start = time.perf_counter()
            try:
                bound = bind_scm(scm, unit, T, outcome, max_boxes=args.max_boxes, max_cells=args.max_cells)
                compile_s = time.perf_counter() - start
                index = ProvenanceIndex(bound)
            except (Infeasible, MemoryError) as e:
                row["status"] = f"infeasible: {e}"
                rows.append(row)
                write_csv(f"scm_full_{tag}.csv", rows)
                print(f"  unit {u} d={d:2d} INFEASIBLE ({e})", flush=True)
                continue
            row["compile_s"] = compile_s
            row["build_s"] = time.perf_counter() - start
            st = index.stats
            row.update(boxes=len(bound.boxes), groups=len(index.groups), bitmask_trees=st.bitmask_trees,
                       bytes=st.packed_bytes)
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

            check("ours_py", [index.query(t).value for t in Q])
            row["ours_py_us"] = per_call_seconds(lambda: [index.query(t) for t in Q], n) * 1e6
            row["online_py_us"] = per_call_seconds(lambda: [py.query(t) for t in Q], n) * 1e6
            if accel.HAVE_NUMBA:
                nb_ours = accel.NumbaIndex(index)
                nb_onl = ScmNumba(scm, unit, T, outcome)
                nb_ours.query(Q[0]), nb_onl.query(Q[0])
                check("ours_nb", [nb_ours.query(t) for t in Q])
                check("online_nb", [nb_onl.query(t) for t in Q])
                row["ours_nb_us"] = per_call_seconds(lambda: [nb_ours.query(t) for t in Q], n) * 1e6
                row["online_nb_us"] = per_call_seconds(lambda: [nb_onl.query(t) for t in Q], n) * 1e6
            if compiled:
                tl = ScmTreelite(scm, compiled, unit, T, outcome)
                got = [tl.query(t) for t in Q]
                row["treelite_wrong"] = int((np.abs(np.asarray(got) - lib) > lib_tol).sum())
                row["treelite_us"] = per_call_seconds(lambda: [tl.query(t) for t in Q], n) * 1e6
            k = min(args.dowhy_queries, n)
            ivs = [{name: (lambda _v, value=float(t[j]): value) for j, name in enumerate(T)} for t in Q[:k]]
            got = [float(gcm.counterfactual_samples(cg, iv, observed_data=unit_row)[outcome].iloc[0]) for iv in ivs]
            # DoWhy abducts N = u - f(pa) and adds it back, so rounding can move an observed mediator
            # across a threshold of its child: count such answers instead of failing
            row["dowhy_wrong"] = int((np.abs(np.asarray(got) - lib[:k]) > lib_tol[:k]).sum())
            row["dowhy_us"] = per_call_seconds(
                lambda: [gcm.counterfactual_samples(cg, iv, observed_data=unit_row) for iv in ivs], k, repeats=1) * 1e6
            row["status"] = "ok"
            rows.append(row)
            write_csv(f"scm_full_{tag}.csv", rows)
            print(f"  unit {u} d={d:2d} med={row['mediators']:3d} boxes={row['boxes']:>9,} groups={row['groups']:5d} "
                  f"build={row['build_s']:6.1f}s | py {row['ours_py_us']:7.1f} vs {row['online_py_us']:9.1f} | "
                  f"nb {row.get('ours_nb_us', float('nan')):6.2f} vs {row.get('online_nb_us', float('nan')):8.1f} | "
                  f"tl {row.get('treelite_us', float('nan')):8.1f} | dowhy {row['dowhy_us']:9.0f}", flush=True)
    tl_dir.cleanup()


if __name__ == "__main__":
    main()
