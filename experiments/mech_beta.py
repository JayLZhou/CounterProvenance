#!/usr/bin/env python3
"""Exp.5: the groups of the greedy algorithm against the budget, for every mechanism, without building the index."""

from __future__ import annotations

import argparse
import math
import pickle
import time

import pandas as pd

from common import RESULTS
from counterprovenance.ensemble import Slabs
from counterprovenance.grouping import greedy_groups, tree_profiles
from counterprovenance.scm import bind_scm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, rows = state["scm"], state["rows"]
    unit0 = rows.iloc[0].to_dict()
    out = RESULTS / "structure" / f"{args.dataset}_{args.family}_beta.csv"
    got = []
    for node in sorted(scm.models):
        bound = bind_scm(scm, unit0, list(scm.parents[node]), node, threads=1, native=True)
        slabs = Slabs(bound)
        profiles = tree_profiles(bound, slabs)
        grid = math.prod(slabs.counts)
        for beta in (2e6, 2e7, 1e8, 5e8, 2e9):
            t0 = time.perf_counter()
            if grid <= 2_000_000:
                groups, deep = 1, 0
                cells = grid
            else:
                g, d = greedy_groups(profiles, int(beta), max(int(beta), 10**9))
                groups, deep, cells = len(g), len(d), sum(x.cells for x in g)
            got.append(dict(dataset=args.dataset, family=args.family, node=node, beta=beta, grid=grid, trees=len(profiles), groups=groups, too_large=deep, cells=cells, seconds=time.perf_counter() - t0))
            pd.DataFrame(got).to_csv(out, index=False)


if __name__ == "__main__":
    main()
