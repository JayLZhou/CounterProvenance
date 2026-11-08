#!/usr/bin/env python3
"""The real memory of the index of every mechanism, after the build has returned its free memory, against the size formula.
Output: ``results/rss/<dataset>_<family>.csv``."""

from __future__ import annotations

import argparse
import gc
import math
import os
import pickle
import resource
import time

import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.ensemble import Slabs
from counterprovenance.scm import bind_scm


def rss() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--grouping", default="tables", choices=["tables", "none"])
    ap.add_argument("--threads", type=int, default=1, help="build threads: more only when the memory is the question, not the time")
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, rows = state["scm"], state["rows"]
    nodes = pd.read_csv(RESULTS / "main" / f"{args.dataset}_{args.family}_build.csv").node.tolist()
    out = RESULTS / ("rss" if args.grouping == "tables" else "rss_none")
    out.mkdir(parents=True, exist_ok=True)
    unit0 = rows.iloc[0].to_dict()
    got, kept = [], []
    for node in nodes:
        bound = bind_scm(scm, unit0, list(scm.parents[node]), node, threads=args.threads, native=True)
        gc.collect()
        before = rss()
        t0 = time.perf_counter()
        if args.grouping == "tables":
            index = ProvenanceIndex(bound, grouping="tables", budget=int(2e7), build_threads=args.threads)
        else:  # one group over all inputs: the index without grouping
            index = ProvenanceIndex(bound, grouping="none", native_cells=True, max_leaf_cells=10**15, build_threads=args.threads)
        build_s = time.perf_counter() - t0
        gc.collect()
        real = rss() - before
        # for the ablation: the cells, the leaves a table of all leaves per cell would store (each cell holds one leaf
        # of every tree of its group), and the grid of one group over all trees (the index without grouping)
        cells = [leaf.cells for leaf in index._leaves]
        faces = sum(int(c.overlay.n_faces) for c in cells)
        all_leaves = sum(int(c.overlay.n_faces) * int(c.stats.slots) for c in cells)
        grid_all = math.prod(int(c) for c in Slabs(bound).counts)
        got.append(dict(dataset=args.dataset, family=args.family, node=node, groups=len(index.groups), grid=index.stats.grid_cells, formula=index.stats.packed_bytes, real=real, total=rss(),
                        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, build_s=build_s, threads=args.threads,
                        trees=len(bound.ensemble.trees), faces=faces, all_leaves=all_leaves, grid_all=grid_all))
        kept.append(index)  # every index of the model stays loaded, as when the model serves queries
        pd.DataFrame(got).to_csv(out / f"{args.dataset}_{args.family}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
