#!/usr/bin/env python3
"""Exp.6: the total size of the index of every mechanism against CellTable, over all groups.

CellTable stores one entry per cell: the leaf of each tree (4 bytes) and the value (8 bytes), and a map from grid cells to
cells (4 bytes each).  ``table_groups`` keeps our groups; ``table_whole_min`` is one table for the whole ensemble, of which
only a lower bound is known when there are several groups: the whole ensemble has at least as many cells as any of them."""

from __future__ import annotations

import argparse
import gc
import pickle
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import bind_scm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--budget", type=float, default=2e7)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, rows = state["scm"], state["rows"]
    nodes = pd.read_csv(RESULTS / "main" / f"{args.dataset}_{args.family}_build.csv").node.tolist()
    out = RESULTS / "size"
    out.mkdir(parents=True, exist_ok=True)
    unit0 = rows.iloc[0].to_dict()
    got, groups = [], []
    for node in nodes:
        t0 = time.perf_counter()
        bound = bind_scm(scm, unit0, list(scm.parents[node]), node, threads=1, native=True)
        index = ProvenanceIndex(bound, grouping="tables", budget=int(args.budget), build_threads=1)
        st = index.stats
        trees = len(scm.ensembles[node].trees)
        leaves = [c for c in index.leaves]
        table = whole = pft = 0
        for c in leaves:
            s = c.stats
            one = s.faces * (4 * s.slots + 8)
            table += one + 4 * s.grid_cells
            whole = max(whole, s.faces)
            pft += 8 * s.pft_nodes + 16 * s.faces
            groups.append(dict(dataset=args.dataset, family=args.family, node=node, trees=s.slots, cells=s.faces, grid=s.grid_cells, nodes=s.pft_nodes))
        got.append(dict(dataset=args.dataset, family=args.family, node=node, trees=trees, groups=len(leaves), cells=sum(c.stats.faces for c in leaves),
                        grid=sum(c.stats.grid_cells for c in leaves), twin_bytes=st.packed_bytes, twin_cells_bytes=pft, table_groups=table,
                        table_groups_cells=sum(c.stats.faces * (4 * c.stats.slots + 8) for c in leaves), table_whole_min=whole * (4 * trees + 8),
                        whole_exact=len(leaves) == 1, build_s=time.perf_counter() - t0))
        pd.DataFrame(got).to_csv(out / f"{args.dataset}_{args.family}_size.csv", index=False)
        pd.DataFrame(groups).to_csv(out / f"{args.dataset}_{args.family}_groups.csv", index=False)
        del index, bound, leaves
        gc.collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
