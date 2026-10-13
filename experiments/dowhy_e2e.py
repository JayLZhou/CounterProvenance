#!/usr/bin/env python3
"""End-to-end comparison with DoWhy-GCM counterfactual recomputation.

SCM on California housing: every feature is a root node, the target is an
additive-noise mechanism whose prediction model is a fitted tree ensemble.
For a fixed observed unit we issue many ``do(features = t)`` queries:

* DoWhy fresh: ``counterfactual_samples(..., observed_data=unit)`` per query
  (abduction, action, prediction);
* DoWhy cached noise: ``counterfactual_samples(..., noise_data=noise)``;
* index: one ``ProvenanceIndex.query`` per query (value and provenance).

Fitting is excluded from all timings.  Answers are compared query by query.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import california, environment, model_zoo, per_call_seconds, value_tolerance, write_csv
from counterprovenance import ProvenanceIndex, autotune, bind, from_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dims", default="2,4")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    import networkx as nx
    import pandas as pd
    from dowhy import gcm
    from dowhy.gcm.ml import SklearnRegressionModel
    from dowhy.graph import get_ordered_predecessors

    gcm.config.disable_progress_bars()
    print(environment(), flush=True)
    X, y, names = california(args.data_home)
    df = pd.DataFrame(X, columns=names)
    df["Y"] = y
    rng = np.random.default_rng(7)
    unit = df.iloc[[int(rng.integers(len(df)))]].copy()
    zoo = model_zoo(args.quick)
    rows = []
    for name in ("sklearn-GBR", "sklearn-HGB", "XGBoost-hist", "LightGBM"):
        if name not in zoo:
            continue
        graph = nx.DiGraph([(f, "Y") for f in names])
        scm = gcm.InvertibleStructuralCausalModel(graph)
        for f in names:
            scm.set_causal_mechanism(f, gcm.EmpiricalDistribution())
        scm.set_causal_mechanism("Y", gcm.AdditiveNoiseModel(SklearnRegressionModel(zoo[name]())))
        start = time.perf_counter()
        gcm.fit(scm, df)
        fit_s = time.perf_counter() - start
        fitted = scm.causal_mechanism("Y").prediction_model.sklearn_model
        parents = get_ordered_predecessors(scm.graph, "Y")  # DoWhy feeds parents sorted by name
        XP = df[parents].to_numpy()
        x_unit = unit[parents].to_numpy()[0]
        noise = float(unit["Y"].iloc[0] - fitted.predict(unit[parents].to_numpy())[0])
        noise_df = unit.copy()
        noise_df["Y"] = noise
        ensemble = from_model(fitted)
        for d in (int(v) for v in args.dims.split(",")):
            importance = np.argsort(-np.asarray(getattr(fitted, "feature_importances_", np.ones(len(parents)))))
            axes = sorted(int(a) for a in importance[:d])
            bound = bind(ensemble, x_unit, axes, noise=noise)
            lo = np.percentile(XP[:, axes], 2, axis=0)
            hi = np.percentile(XP[:, axes], 98, axis=0)
            pts = rng.uniform(lo, hi, size=(args.queries, d))
            interventions = [
                {parents[a]: (lambda _v, value=float(t[k]): value) for k, a in enumerate(axes)} for t in pts
            ]
            check = min(50, len(pts))
            dowhy_vals = np.array([
                float(gcm.counterfactual_samples(scm, interventions[i], observed_data=unit)["Y"].iloc[0])
                for i in range(check)
            ])
            for iv in interventions[:3]:
                gcm.counterfactual_samples(scm, iv, observed_data=unit)
            fresh = per_call_seconds(
                lambda: [gcm.counterfactual_samples(scm, iv, observed_data=unit) for iv in interventions],
                len(interventions), repeats=1)
            cached = per_call_seconds(
                lambda: [gcm.counterfactual_samples(scm, iv, noise_data=noise_df) for iv in interventions],
                len(interventions), repeats=1)
            configs = [("default (tau=4096, grouped)", lambda: ProvenanceIndex(bound))]
            sample = pts[rng.integers(len(pts), size=32)]
            configs.append(("autotuned (pieces<=4)", lambda: autotune(bound, sample, pieces_budget=4).index))
            for label, build in configs:
                start = time.perf_counter()
                index = build()
                build_s = time.perf_counter() - start
                index_vals = np.array([index.query(pts[i]).value for i in range(check)])
                err = float(np.max(np.abs(dowhy_vals - index_vals)))
                if err > value_tolerance(fitted):
                    raise AssertionError(f"{name} d={d}: DoWhy and index differ by {err}")
                ours = per_call_seconds(lambda: [index.query(t) for t in pts], len(pts))
                pieces = float(np.mean([len(index.query(t).pieces) for t in pts]))
                row = {
                    "model": name, "d": d, "intervened": "+".join(parents[a] for a in axes), "index": label,
                    "trees": ensemble.tree_count, "fit_s": fit_s, "index_build_s": build_s,
                    "index_records": index.stats.records, "mean_pieces": pieces,
                    "dowhy_fresh_us": fresh * 1e6, "dowhy_cached_noise_us": cached * 1e6, "index_us": ours * 1e6,
                    "speedup_vs_fresh": fresh / ours, "speedup_vs_cached": cached / ours, "max_abs_error": err,
                }
                rows.append(row)
                print(f"{name:<13} d={d} {label:<28} DoWhy fresh {row['dowhy_fresh_us']:8.1f}us cached "
                      f"{row['dowhy_cached_noise_us']:8.1f}us index {row['index_us']:6.1f}us -> "
                      f"{row['speedup_vs_fresh']:.0f}x / {row['speedup_vs_cached']:.0f}x (err {err:.1e}, pieces {pieces:.1f})",
                      flush=True)
            write_csv("r4_dowhy.csv", rows)


if __name__ == "__main__":
    main()
