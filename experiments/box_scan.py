#!/usr/bin/env python3
"""Leaf-box counts of a dataset for every model, d and unit: fit, compile in C++, count.  No index, no queries.
Stops counting at ``--max-boxes``.  Output: ``results/box_scan_<dataset>_<family>.csv``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_csv
from counterprovenance import native_compile
from counterprovenance.scm import ancestors, fit_scm, parents_of, relevant_nodes
from scm_bench import choose_treatments, zoo


def count(scm, unit, T, outcome, max_boxes, threads):
    axis = {name: a for a, name in enumerate(T)}
    meds = {}
    for node in relevant_nodes(scm.parents, T, outcome):
        ens = scm.ensembles[node]
        inputs = []
        for p in scm.parents[node]:
            if p in axis:
                inputs.append(("axis", axis[p]))
            elif p in meds:
                inputs.append(("mediator", meds[p]))
            else:
                inputs.append(("fixed", float(ens.cast(np.array([unit[p]]))[0])))
        flat = native_compile.flat_trees(ens)
        boxes, status = native_compile.compile_trees(ens, flat, inputs, len(T), max_boxes, threads)
        if status != 0:
            return -1, len(meds)
        if node == outcome:
            return len(boxes.tree), len(meds)
        meds[node] = native_compile.mediator_from(boxes, flat, float(unit[node]), scm.observed_leaf_values(node, unit))
    raise AssertionError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--max-boxes", type=int, default=60_000_000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    outcome = str(meta["target"])
    scm = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]], zoo(1000, 1)[args.family],
                  max_rows=50_000, seed=0, n_jobs=4)
    unit_idx = np.random.default_rng(2028).integers(len(df), size=3)
    rows = []
    for u in range(3):
        unit = df.iloc[[int(unit_idx[u])]].iloc[0].to_dict()
        for d in (1, 2, 4, 8):
            T = choose_treatments(scm, outcome, d)
            if not T:
                continue
            t0 = time.perf_counter()
            n, med = count(scm, unit, T, outcome, args.max_boxes, args.threads)
            rows.append(dict(dataset=args.dataset, family=args.family, d=d, unit=u, boxes=n, mediators=med,
                             compile_s=time.perf_counter() - t0))
            write_csv(f"box_scan_{args.dataset}_{args.family}.csv", rows)
            print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
