#!/usr/bin/env python3
"""One index per mechanism on many units: 1,000 random rows of the table, 10 interventions each, one query at a time.

The index is built once, before any of these units.  ThreeStep and DoWhy-GCM answer the same queries, one at a time, as
in the main experiment.  ``abduct`` is what a unit costs once: the reads of its own values, one per mechanism.
Output: ``results/mech_units/<dataset>_<family>_d<d>.csv``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.scm import ancestors, fit_scm, parents_of, relevant_nodes
from counterprovenance.scm_baselines import ScmThreeStepPy
from mech_proto import MechanismIndex
from scm_bench import choose_treatments, zoo
from scm_full import dowhy_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--dims", default="2,4,8")
    ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--per-unit", type=int, default=10)
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], zoo(1000, 1)[args.family], max_rows=50_000,
                  seed=0, n_jobs=4)
    rows = np.random.default_rng(2028).integers(len(df), size=args.units)
    dims = [d for d in (int(v) for v in args.dims.split(",")) if choose_treatments(scm, y, d)]
    nodes = sorted({x for d in dims for x in relevant_nodes(scm.parents, choose_treatments(scm, y, d), y)})
    t0 = time.perf_counter()
    mech = MechanismIndex(scm, df.iloc[[int(rows[0])]].iloc[0].to_dict(), nodes, args.budget, args.threads)
    build_s = time.perf_counter() - t0
    gcm, cg = dowhy_model(scm, edges)
    out_dir = RESULTS / "mech_units"
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in dims:
        T = choose_treatments(scm, y, d)
        order = relevant_nodes(scm.parents, T, y)
        lo, hi = df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy()
        # 3/4 of the values uniform, 1/4 with one coordinate on a threshold of the outcome's trees
        pos = {p: i for i, p in enumerate(scm.parents[y])}
        cuts = {}
        for a, name in enumerate(T):
            if name in pos:
                vals = [t.threshold[t.feature == pos[name]] for t in scm.ensembles[y].trees]
                cuts[a] = np.unique(np.concatenate(vals)) if vals else np.zeros(0)
        q_abduct = q_mech = q_three = q_dowhy = 0.0
        n = wrong = 0
        rng = np.random.default_rng([2028, d])
        for i in rows:
            row = df.iloc[[int(i)]]
            unit = row.iloc[0].to_dict()
            Q = rng.uniform(lo, hi, size=(args.per_unit, d))
            for j in range(args.per_unit * 3 // 4, args.per_unit):
                a = int(rng.integers(d))
                if a in cuts and len(cuts[a]):
                    Q[j, a] = cuts[a][int(rng.integers(len(cuts[a])))]
            t0 = time.perf_counter()
            observed = mech.observed(unit, order)
            q_abduct += time.perf_counter() - t0
            got = []
            for t in Q:
                t0 = time.perf_counter()
                got.append(mech.query(unit, T, t, order, observed))
                q_mech += time.perf_counter() - t0
            three = ScmThreeStepPy(scm, unit, T, y)
            ref = []
            for t in Q:
                t0 = time.perf_counter()
                ref.append(three.query(t)[0])
                q_three += time.perf_counter() - t0
            for t in Q:
                iv = {name: (lambda _v, value=float(t[k]): value) for k, name in enumerate(T)}
                t0 = time.perf_counter()
                gcm.counterfactual_samples(cg, iv, observed_data=row)
                q_dowhy += time.perf_counter() - t0
            wrong += int((np.abs(np.array(got) - np.array(ref)) > 1e-6 * (1 + np.abs(np.array(ref)))).sum())
            n += len(Q)
        r = dict(dataset=args.dataset, family=args.family, d=d, units=args.units, per_unit=args.per_unit, queries=n, path=len(order),
                 build_s=build_s, sum_groups=sum(mech.info[x]["groups"] for x in order), wrong=wrong,
                 abduct_us=q_abduct / args.units * 1e6, mech_us=q_mech / n * 1e6, three_step_us=q_three / n * 1e6, dowhy_us=q_dowhy / n * 1e6)
        pd.DataFrame([r]).to_csv(out_dir / f"{args.dataset}_{args.family}_d{d}.csv", index=False)
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k not in ("dataset", "family")}), flush=True)


if __name__ == "__main__":
    main()
