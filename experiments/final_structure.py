#!/usr/bin/env python3
"""Exp.6 on the final indexes (leaves as a log along the MST): what each group stores, against CellTable, from the built
indexes of one model; and, on a sample of groups, the log along a random spanning tree and the changes of all edges.

``<out>/<model>_groups.csv``: per group, its trees, blocks, cells, the weight of its MST, the changes and the checkpoint
entries of its log, its bytes, and the leaves of CellTable (cells x trees).  ``<out>/<model>_sample.csv``: per sampled group
(at most ``--sample`` groups of at most ``--max-blocks`` blocks), the same group rebuilt along a random spanning tree, and
the changes on all edges of the cell graph."""
import argparse, glob, os, pickle, re, time
import numpy as np
import pandas as pd
from counterprovenance.native_cells import LogCellIndex


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="directory of the indexes of one model (one .pkl per tree ensemble)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--max-blocks", type=float, default=2e6)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    model = re.sub(r"_t\d+_b.*$", "", os.path.basename(args.dir.rstrip("/")))
    os.makedirs(args.out, exist_ok=True)
    rows, cand = [], []
    for f in sorted(glob.glob(args.dir + "/*.pkl")):
        if f.endswith(".tl.pkl"):
            continue
        index, info = pickle.load(open(f, "rb"))
        node = os.path.basename(f)[:-4]
        for gi, leaf in enumerate(getattr(index, "_fast_leaves", None) or []):
            c = leaf.cells
            if not isinstance(c, LogCellIndex):
                raise SystemExit(f"{f}: group {gi} is not a log")
            F, g, K = int(c.stats.faces), int(c.stats.slots), int(c.overlay.grid_cells)
            rows.append(dict(model=model, node=node, group=gi, trees=g, blocks=K, cells=F, omega=float(c.stats.tree_weight),
                             changes=int(c.log.changes), ckpt=int(len(c.log.snap)), direct=bool(c.log.direct), log_bytes=int(c.log.nbytes),
                             celltable_leaves=F * g, celltable_bytes=F * (4 * g + 8) + 4 * K))
            if g > 1 and K <= args.max_blocks:
                cand.append((f, gi))
        del index
    pd.DataFrame(rows).to_csv(os.path.join(args.out, model + "_groups.csv"), index=False)
    rng = np.random.default_rng(2028)
    pick = [cand[i] for i in sorted(rng.choice(len(cand), size=min(args.sample, len(cand)), replace=False))] if cand else []
    srows, loaded = [], {}
    for f, gi in pick:
        if f not in loaded:
            loaded = {f: pickle.load(open(f, "rb"))[0]}
        index = loaded[f]
        leaf = index._fast_leaves[gi]
        c, axes, slabs, boxes = leaf.cells, leaf.axes, index.slabs, index.bound.boxes
        kw = dict(items=c.items, ranges=[[slabs.ranges[i][a] for a in axes] for i in c.items], slots=[boxes[i].tree for i in c.items],
                  slab_counts=[slabs.counts[a] for a in axes], values=[boxes[i].value for i in c.items], threads=args.threads)
        t0 = time.perf_counter()
        r = LogCellIndex(**kw, seed=1)
        assert r.stats.faces == c.stats.faces
        srows.append(dict(model=model, node=os.path.basename(f)[:-4], group=gi, trees=int(c.stats.slots), blocks=int(c.overlay.grid_cells),
                          cells=int(c.stats.faces), omega=float(c.stats.tree_weight), mst_changes=int(c.log.changes), mst_ckpt=int(len(c.log.snap)),
                          random_changes=int(r.log.changes), random_ckpt=int(len(r.log.snap)), every_edge=int(r.edge_changes),
                          celltable_leaves=int(c.stats.faces) * int(c.stats.slots), sec=time.perf_counter() - t0))
    pd.DataFrame(srows).to_csv(os.path.join(args.out, model + "_sample.csv"), index=False)
    print(model, len(rows), "groups", len(srows), "sampled", flush=True)


if __name__ == "__main__":
    main()
