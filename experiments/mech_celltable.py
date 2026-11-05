#!/usr/bin/env python3
"""CellTable as the paper defines it: one table over all thresholds of the outcome's ensemble, without groups, really built for the first
M trees, until its grid exceeds ``--cap`` or the memory runs out.  Output: ``results/celltable/<dataset>_<family>.csv``."""

from __future__ import annotations

import argparse
import dataclasses
import math
import pickle
import time

import pandas as pd

from common import RESULTS
from counterprovenance.ensemble import Slabs
from counterprovenance.native_cells import NativeCellIndex
from counterprovenance.scm import bind_scm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--cap", type=float, default=1e9)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, y, rows = state["scm"], state["y"], state["rows"]
    out = RESULTS / "celltable"
    out.mkdir(parents=True, exist_ok=True)
    bound = bind_scm(scm, rows.iloc[0].to_dict(), list(scm.parents[y]), y, threads=1, native=True)
    got = []
    for m in (10, 20, 50, 100, 200, 500, 1000):
        sub = dataclasses.replace(bound, boxes=tuple(b for b in bound.boxes if b.tree < m))
        slabs = Slabs(sub)
        grid = math.prod(slabs.counts)
        r = dict(dataset=args.dataset, family=args.family, node=y, parents=sub.d, trees=m, grid=grid, leaves=len(sub.boxes))
        if grid > args.cap:
            r.update(status="grid over the cap", map_bytes_needed=4 * grid)
        else:
            t0 = time.perf_counter()
            try:
                cells = NativeCellIndex(items=list(range(len(sub.boxes))), ranges=[[slabs.ranges[i][a] for a in range(sub.d)] for i in range(len(sub.boxes))], slots=[b.tree for b in sub.boxes],
                                        slab_counts=slabs.counts, values=[b.value for b in sub.boxes], threads=1)
                s = cells.stats
                r.update(status="ok", cells=s.faces, entry_bytes=s.faces * (4 * s.slots + 8), map_bytes=4 * s.grid_cells, build_s=time.perf_counter() - t0, pft_nodes=s.pft_nodes)
                del cells
            except MemoryError as e:
                r.update(status=f"out of memory: {str(e)[:60]}")
        got.append(r)
        pd.DataFrame(got).to_csv(out / f"{args.dataset}_{args.family}.csv", index=False)
        print(r, flush=True)
        if r["status"] != "ok":
            break
    print("done", flush=True)


if __name__ == "__main__":
    main()
