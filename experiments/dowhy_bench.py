#!/usr/bin/env python3
"""End-to-end comparison with DoWhy-GCM on every dataset.

A star SCM (every feature -> Y, empirical root distributions, Y an
additive-noise mechanism whose prediction model is the tree ensemble) is
fitted with DoWhy-GCM.  For one observed unit and the ``d`` features the
ensemble splits on most, ``do(features = t)`` queries are answered by

* DoWhy fresh - ``counterfactual_samples(..., observed_data=unit)``;
* DoWhy cached noise - ``counterfactual_samples(..., noise_data=noise)``;
* the library's fastest single-row prediction plus the abducted noise;
* our index, pure Python (``ProvenanceIndex.query``) and numba (one call per query).

Values are compared query by query; fitting is excluded from all timings.
Output: ``results/dowhy_<tag>.csv``.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from common import environment, load_dataset, paper_zoo, per_call_seconds, value_tolerance, write_csv
from counterprovenance import ProvenanceIndex, bind, from_model
from counterprovenance import accel
from query_bench import single_row_calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", default="sklearn-GBR,sklearn-HGB,sklearn-RF,XGBoost,LightGBM")
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--dims", default="1,2,4,8,16")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dowhy-queries", type=int, default=40)
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    import networkx as nx
    import pandas as pd
    from dowhy import gcm
    from dowhy.gcm.ml import SklearnRegressionModel
    from dowhy.graph import get_ordered_predecessors

    gcm.config.disable_progress_bars()
    print(environment(), flush=True)
    X, y, title = load_dataset(args.dataset, args.data_home)
    names = [f"f{i:04d}" for i in range(X.shape[1])]  # sortable: DoWhy feeds parents sorted by name
    df = pd.DataFrame(X, columns=names)
    df["Y"] = y
    print(f"{title}: {X.shape}", flush=True)
    zoo = paper_zoo(args.trees)
    rows = []
    for name in args.models.split(","):
        rng = np.random.default_rng(11)
        unit = df.iloc[[int(rng.integers(len(df)))]].copy()
        scm = gcm.InvertibleStructuralCausalModel(nx.DiGraph([(f, "Y") for f in names]))
        for f in names:
            scm.set_causal_mechanism(f, gcm.EmpiricalDistribution())
        scm.set_causal_mechanism("Y", gcm.AdditiveNoiseModel(SklearnRegressionModel(zoo[name]())))
        start = time.perf_counter()
        gcm.fit(scm, df)
        fit_s = time.perf_counter() - start
        model = scm.causal_mechanism("Y").prediction_model.sklearn_model
        parents = get_ordered_predecessors(scm.graph, "Y")
        assert parents == names
        x_unit = unit[names].to_numpy()[0]
        noise = float(unit["Y"].iloc[0] - model.predict(x_unit[None, :])[0])
        noise_df = unit.copy()
        noise_df["Y"] = noise
        ensemble = from_model(model)
        usage = Counter(int(f) for t in ensemble.trees for f in t.feature if f >= 0)
        order = [f for f, _ in usage.most_common()]
        print(f"{name}: fitted in {fit_s:.1f}s, {ensemble.tree_count} trees", flush=True)
        for d in (int(v) for v in args.dims.split(",")):
            if d > len(order):
                continue
            axes = sorted(order[:d])
            bound = bind(ensemble, x_unit, axes, noise=noise)
            lo = np.percentile(X[:, axes], 2, axis=0)
            hi = np.percentile(X[:, axes], 98, axis=0)
            pts = np.ascontiguousarray(rng.uniform(lo, hi, size=(args.queries, d)))
            k = min(args.dowhy_queries, len(pts))
            interventions = [{names[a]: (lambda _v, value=float(t[j]): value) for j, a in enumerate(axes)} for t in pts[:k]]
            dowhy_vals = np.array([float(gcm.counterfactual_samples(scm, iv, observed_data=unit)["Y"].iloc[0])
                                   for iv in interventions])
            start = time.perf_counter()
            index = ProvenanceIndex(bound)
            build_s = time.perf_counter() - start
            ours = np.array([index.query(t).value for t in pts[:k]])
            err = float(np.max(np.abs(dowhy_vals - ours) / np.maximum(1.0, np.abs(dowhy_vals))))
            if err > max(value_tolerance(model), 1e-9):
                raise AssertionError(f"{name} d={d}: DoWhy and index differ by {err}")
            fresh = per_call_seconds(
                lambda: [gcm.counterfactual_samples(scm, iv, observed_data=unit) for iv in interventions], k, repeats=1)
            cached = per_call_seconds(
                lambda: [gcm.counterfactual_samples(scm, iv, noise_data=noise_df) for iv in interventions], k, repeats=1)
            full = np.tile(x_unit, (len(pts), 1))
            full[:, axes] = pts
            full_rows = [full[q:q + 1] for q in range(len(full))]
            lib = min(per_call_seconds(lambda: [fn(r) for r in full_rows], len(pts)) for _, fn in single_row_calls(model))
            py = per_call_seconds(lambda: [index.query(t) for t in pts], len(pts))
            row = {
                "dataset": args.dataset, "features": X.shape[1], "model": name, "trees": ensemble.tree_count, "d": d,
                "fit_s": fit_s, "build_s": build_s, "records": index.stats.records, "groups": len(index.groups),
                "bitmask_trees": index.stats.bitmask_trees, "max_rel_err_vs_dowhy": err,
                "dowhy_fresh_us": fresh * 1e6, "dowhy_cached_us": cached * 1e6, "lib_single_best_us": lib * 1e6,
                "ours_py_us": py * 1e6,
            }
            if accel.HAVE_NUMBA:
                nb = accel.NumbaIndex(index)
                nb.query(pts[0])
                row["ours_nb_us"] = per_call_seconds(lambda: [nb.query(t) for t in pts], len(pts)) * 1e6
            rows.append(row)
            print(f"{args.dataset} {name:12s} d={d:2d} DoWhy {row['dowhy_fresh_us']:9.0f}us cached {row['dowhy_cached_us']:9.0f}us "
                  f"lib {row['lib_single_best_us']:7.1f}us ours py {row['ours_py_us']:6.2f}us nb {row.get('ours_nb_us', float('nan')):5.2f}us "
                  f"-> {row['dowhy_fresh_us'] / row['ours_py_us']:7.0f}x (py) {row['dowhy_fresh_us'] / row.get('ours_nb_us', np.nan):7.0f}x (nb) "
                  f"err {err:.1e}", flush=True)
            write_csv(f"dowhy_{args.tag or args.dataset}.csv", rows)


if __name__ == "__main__":
    main()
