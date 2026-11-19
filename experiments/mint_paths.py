#!/usr/bin/env python3
"""MINT by the length of the path of an action: for a sample of persons, the time per query of each backend for each
action set S, against the number of tree ensembles that a query of S replays (the mediators and the outcome).
Output: ``results/mint_paths/<dataset>_<family>.csv`` with one row per (person, action set, backend)."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.scm import ancestors, fit_scm, parents_of, relevant_nodes
from e2e_apps import Ctx, DoWhy, ThreeStep, CellForestMech, model_factory
from scm_full import dowhy_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--persons", type=int, default=30)
    ap.add_argument("--levels", type=int, default=20)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], model_factory(args.family), max_rows=50_000, seed=0, n_jobs=1)
    anc = ancestors(parents_of(edges), y)
    T = [v for v in meta["actionable"] if v in anc]
    order = np.random.default_rng(2028).permutation(len(df))
    first = df.iloc[[int(order[0])]]
    tw = CellForestMech(scm, first.iloc[0].to_dict(), T, y, 2e7, 1, max_size=3)
    gcm, cg = dowhy_model(scm, edges)
    base = Ctx(df, T, first, y, 2028, meta.get("increase_only", ()), meta.get("mcce_immutable", ()))
    grids = [np.linspace(base.lo[a], base.hi[a], args.levels) for a in range(len(T))]
    out = RESULTS / "mint_paths"
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for i in order[:args.persons]:
        row = df.iloc[[int(i)]]
        unit = row.iloc[0].to_dict()
        twu = tw.unit(unit)
        for size in range(1, min(len(T), 3) + 1):
            for S in itertools.combinations(range(len(T)), size):
                Q = np.column_stack([m.ravel() for m in np.meshgrid(*[grids[a] for a in S], indexing="ij")])
                names = [T[a] for a in S]
                path = relevant_nodes(scm.parents, names, y)
                for b in ("twin", "three_step", "dowhy"):
                    t0 = time.perf_counter()
                    if b == "twin":
                        fn = tw.subset(twu, S)
                    elif b == "three_step":
                        fn = ThreeStep(scm, unit, names, y)
                    else:
                        fn = DoWhy(gcm, cg, row, names, y)
                    v = fn(Q)
                    spent = time.perf_counter() - t0
                    got.append(dict(dataset=args.dataset, family=args.family, person=int(i), action=",".join(names), size=size,
                                    path=len(path), mediators=len(path) - 1, queries=len(Q), backend=b, seconds=spent,
                                    us_per_query=spent / len(Q) * 1e6, checksum=float(np.sum(v))))
        pd.DataFrame(got).to_csv(out / f"{args.dataset}_{args.family}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
