#!/usr/bin/env python3
"""The real memory of the index of every mechanism, after the build has returned its free memory, against the size formula.
Output: ``results/rss/<dataset>_<family>.csv``."""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import resource
import time

import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import bind_scm


def rss() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, rows = state["scm"], state["rows"]
    nodes = pd.read_csv(RESULTS / "main" / f"{args.dataset}_{args.family}_build.csv").node.tolist()
    out = RESULTS / "rss"
    out.mkdir(parents=True, exist_ok=True)
    unit0 = rows.iloc[0].to_dict()
    got, kept = [], []
    for node in nodes:
        bound = bind_scm(scm, unit0, list(scm.parents[node]), node, threads=1, native=True)
        gc.collect()
        before = rss()
        t0 = time.perf_counter()
        index = ProvenanceIndex(bound, grouping="tables", budget=int(2e7), build_threads=1)
        build_s = time.perf_counter() - t0
        gc.collect()
        got.append(dict(dataset=args.dataset, family=args.family, node=node, formula=index.stats.packed_bytes, real=rss() - before,
                        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, build_s=build_s))
        kept.append(index)  # every index of the model stays loaded, as when the model serves queries
        pd.DataFrame(got).to_csv(out / f"{args.dataset}_{args.family}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
