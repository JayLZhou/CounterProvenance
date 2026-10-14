#!/usr/bin/env python3
"""High-dimensional models: hundreds of features, d = 2, 8, 16 interventions.

For each OpenML regression dataset (hundreds of continuous features) and each
model, a star SCM ``every feature -> Y`` is fitted with DoWhy-GCM.  For one
observed unit we intervene on the d features the ensemble splits on most
(the hardest case for the index) and compare, query by query:

* DoWhy fresh   - ``counterfactual_samples(..., observed_data=unit)``
* DoWhy cached  - ``counterfactual_samples(..., noise_data=noise)``
* online replay - single-row ``model.predict`` + abducted noise
* index         - ``ProvenanceIndex`` chosen by ``autotune`` (pieces <= 16)

Everything is single-threaded.  Also reported: how many trees still touch an
intervened feature after binding the unit (``effective trees``) and the
number of leaf boxes ``N``.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter

import numpy as np

from common import environment, load_openml, model_zoo, per_call_seconds, value_tolerance, write_csv
from counterprovenance import autotune, bind, from_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="name:openml_id, e.g. topo_2_1:422")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dims", default="2,8,16")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dowhy-queries", type=int, default=40)
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--max-rows", type=int, default=20000)
    args = parser.parse_args()
    import networkx as nx
    from dowhy import gcm
    from dowhy.gcm.ml import SklearnRegressionModel
    from dowhy.graph import get_ordered_predecessors

    gcm.config.disable_progress_bars()
    print(environment(), flush=True)
    name, data_id = args.dataset.split(":")
    frame, y = load_openml(int(data_id), args.data_home, args.max_rows)
    p = frame.shape[1]
    print(f"{name}: {len(frame)} rows, {p} features", flush=True)
    df = frame.copy()
    df["Y"] = y
    graph = nx.DiGraph([(c, "Y") for c in frame.columns])
    scm = gcm.InvertibleStructuralCausalModel(graph)
    for c in frame.columns:
        scm.set_causal_mechanism(c, gcm.EmpiricalDistribution())
    scm.set_causal_mechanism("Y", gcm.AdditiveNoiseModel(SklearnRegressionModel(model_zoo()[args.model]())))
    start = time.perf_counter()
    gcm.fit(scm, df)
    fit_s = time.perf_counter() - start
    model = scm.causal_mechanism("Y").prediction_model.sklearn_model
    parents = get_ordered_predecessors(scm.graph, "Y")
    XP = df[parents].to_numpy()
    ensemble = from_model(model)
    check_rows = XP[:300]
    compile_err = float(np.max(np.abs(ensemble.predict(check_rows) - model.predict(check_rows))))
    print(f"{args.model}: fitted in {fit_s:.1f}s, {ensemble.tree_count} trees, compile error {compile_err:.1e}", flush=True)

    rng = np.random.default_rng(11)
    unit_index = int(rng.integers(len(df)))
    unit = df.iloc[[unit_index]].copy()
    x_unit = unit[parents].to_numpy()[0]
    noise = float(unit["Y"].iloc[0] - model.predict(unit[parents].to_numpy())[0])
    noise_df = unit.copy()
    noise_df["Y"] = noise
    usage = Counter(int(f) for t in ensemble.trees for f in t.feature if f >= 0)
    order = [f for f, _ in usage.most_common()]
    rows = []
    for d in (int(v) for v in args.dims.split(",")):
        if d > len(order):
            continue
        axes = sorted(order[:d])
        bound = bind(ensemble, x_unit, axes, noise=noise)
        effective = len({b.tree for b in bound.boxes if any(math.isfinite(v) for v in b.lo + b.hi)})
        lo = np.percentile(XP[:, axes], 2, axis=0)
        hi = np.percentile(XP[:, axes], 98, axis=0)
        pts = rng.uniform(lo, hi, size=(args.queries, d))
        start = time.perf_counter()
        tuned = autotune(bound, pts[rng.integers(len(pts), size=24)], taus=(0, 1024, 16384), budgets=(None, 4, 1),
                         pieces_budget=16, max_leaf_cells=400_000)
        build_s = time.perf_counter() - start
        index = tuned.index
        chosen = [r for r in tuned.table if r.get("chosen")][0]
        # online replay with the model's own single-row predict
        full_rows = np.tile(x_unit, (len(pts), 1))
        full_rows[:, axes] = pts
        predict_us = per_call_seconds(lambda: [model.predict(r[None, :]) for r in full_rows], len(full_rows)) * 1e6
        index_us = per_call_seconds(lambda: [index.query(t) for t in pts], len(pts)) * 1e6
        pieces = float(np.mean([len(index.query(t).pieces) for t in pts]))
        interventions = [
            {parents[a]: (lambda _v, value=float(t[k]): value) for k, a in enumerate(axes)} for t in pts[: args.dowhy_queries]
        ]
        dowhy_vals = np.array([
            float(gcm.counterfactual_samples(scm, iv, observed_data=unit)["Y"].iloc[0]) for iv in interventions
        ])
        index_vals = np.array([index.query(t).value for t in pts[: args.dowhy_queries]])
        err = float(np.max(np.abs(dowhy_vals - index_vals)))
        if err > value_tolerance(model):
            raise AssertionError(f"{name} {args.model} d={d}: DoWhy and index differ by {err}")
        for t in pts[:5]:  # provenance: one leaf per tree, identical to brute force
            answer = index.query(t)
            if answer.labels() != Counter(bound.label(j) for j in bound.selected(t)):
                raise AssertionError("provenance mismatch")
        fresh = per_call_seconds(
            lambda: [gcm.counterfactual_samples(scm, iv, observed_data=unit) for iv in interventions], len(interventions), repeats=1)
        cached = per_call_seconds(
            lambda: [gcm.counterfactual_samples(scm, iv, noise_data=noise_df) for iv in interventions], len(interventions), repeats=1)
        row = {
            "dataset": name, "features": p, "model": args.model, "trees": ensemble.tree_count,
            "effective_trees": effective, "d": d, "N_boxes": len(bound.boxes),
            "index_records": index.stats.records, "index_bytes": index.stats.packed_bytes,
            "chosen": f"tau={chosen['tau']}, g={chosen['group_budget']}", "build_s": build_s,
            "mean_pieces": pieces, "index_us": index_us, "predict_us": predict_us,
            "dowhy_fresh_us": fresh * 1e6, "dowhy_cached_us": cached * 1e6,
            "x_vs_dowhy_fresh": fresh * 1e6 / index_us, "x_vs_dowhy_cached": cached * 1e6 / index_us,
            "x_vs_predict": predict_us / index_us, "max_abs_error": err,
            "breakeven_queries_vs_fresh": build_s / max(1e-12, fresh - index_us * 1e-6),
        }
        rows.append(row)
        print(f"{name} p={p} {args.model:<13} d={d:>2} eff_trees={effective:>3}/{ensemble.tree_count} N={len(bound.boxes):>5} "
              f"pieces={pieces:5.1f} index={index_us:7.1f}us predict={predict_us:7.1f}us DoWhy fresh={fresh*1e6:9.1f}us "
              f"cached={cached*1e6:9.1f}us -> {row['x_vs_dowhy_fresh']:.0f}x / {row['x_vs_dowhy_cached']:.0f}x / "
              f"{row['x_vs_predict']:.1f}x vs predict, build {build_s:.1f}s, err {err:.1e}", flush=True)
        write_csv(f"r5_highdim_{name}_{args.model}.csv", rows)


if __name__ == "__main__":
    main()
