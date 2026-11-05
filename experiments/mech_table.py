#!/usr/bin/env python3
"""CellTable with our groups, really built: one row per cell with the leaf of each tree of the group, against the index with the
MST, on the same groups.  A read of the value is the same in both; a read of the leaves is a row against a walk of the persistent
tree.  ``fill_s`` is the time to fill the rows in one thread with numpy, given the cells.  Output: ``results/table/<dataset>_<family>.csv``."""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import resource
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import bind_scm


def rss() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def fill(index, ens):
    """The rows of every group: the leaf of each of its trees at one point of each cell."""
    th = [np.asarray(t, dtype=np.float64) for t in index.slabs.thresholds]
    d = index.bound.d
    tables, checked = [], True
    for leaf in index._fast_leaves:
        cells, o = leaf.cells, leaf.cells.overlay
        K = o.n_faces
        faces, first = np.unique(np.asarray(o.face_of_cell), return_index=True)
        ranks = np.unravel_index(first, o.shape)
        X = np.zeros((K, d), dtype=np.float64)
        for k, a in enumerate(leaf.axes):
            cuts = np.asarray(o.cuts[k], dtype=np.int64)
            slab = np.where(ranks[k] > 0, cuts[np.maximum(ranks[k] - 1, 0)] if len(cuts) else 0, 0)
            top = th[a][-1] + 1.0 if len(th[a]) else 0.0
            X[:, a] = np.append(th[a], top)[slab]
        trees = cells.slot_trees
        sig = np.empty((K, len(trees)), dtype=np.uint16)
        total = np.zeros(K, dtype=np.float64)
        at = np.arange(K)
        for j, m in enumerate(trees):
            t = ens.trees[m]
            feature, threshold, left, right = np.asarray(t.feature), np.asarray(t.threshold), np.asarray(t.left), np.asarray(t.right)
            n = np.zeros(K, dtype=np.int64)
            inner = np.flatnonzero(feature[n] >= 0)
            while len(inner):
                now = n[inner]
                go = X[at[inner], feature[now]] <= threshold[now]
                n[inner] = np.where(go, left[now], right[now])
                inner = inner[feature[n[inner]] >= 0]
            sig[:, j] = n
            total += np.asarray(t.value, dtype=np.float64)[n]
        if not np.allclose(total, np.asarray(cells.face_value)[faces], atol=1e-6):
            checked = False
        tables.append(sig)
    return tables, checked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--points", type=int, default=300)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, rows = state["scm"], state["rows"]
    nodes = pd.read_csv(RESULTS / "main" / f"{args.dataset}_{args.family}_build.csv").node.tolist()
    out = RESULTS / "table"
    out.mkdir(parents=True, exist_ok=True)
    unit0 = rows.iloc[0].to_dict()
    got = []
    for node in nodes:
        pa = list(scm.parents[node])
        gc.collect()
        rss0 = rss()
        t0 = time.perf_counter()
        bound = bind_scm(scm, unit0, pa, node, threads=1, native=True)
        index = ProvenanceIndex(bound, grouping="tables", budget=int(args.budget), build_threads=1)
        build_s = time.perf_counter() - t0
        gc.collect()
        rss1 = rss()
        st = index.stats
        ens = scm.ensembles[node]
        t0 = time.perf_counter()
        tables, checked = fill(index, ens)
        fill_s = time.perf_counter() - t0
        rss2 = rss()
        points = rows[pa].to_numpy(dtype=float)[:args.points]
        faces_of = [leaf.cells.overlay.face_of_cell for leaf in index._fast_leaves]
        value, mst, table = [], [], []
        same = True
        for i, x in enumerate(points):
            t0 = time.perf_counter()
            a = index.query(x)
            value.append(time.perf_counter() - t0)
            cells = a.cells
            t0 = time.perf_counter()
            boxes = a.boxes()
            mst.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            got_rows = [sig[f[c]] for sig, f, c in zip(tables, faces_of, cells)]
            table.append(time.perf_counter() - t0)
            if i < 5 and sum(len(r) for r in got_rows) < len(boxes):
                same = False
        leaf_bytes = sum(s.nbytes for s in tables)
        cells_bytes = 8 * st.pft_nodes + 16 * st.faces
        got.append(dict(dataset=args.dataset, family=args.family, node=node, groups=len(tables), cells=st.faces, map_bytes=st.map_bytes, ranks_bytes=st.packed_bytes - cells_bytes - st.map_bytes, twin_bytes=st.packed_bytes, twin_leaf_bytes=cells_bytes,
                        table_leaf_bytes=leaf_bytes, table_value_bytes=8 * st.faces, table_bytes=st.packed_bytes - cells_bytes + leaf_bytes + 8 * st.faces, build_s=build_s, fill_s=fill_s, rss_base=rss0, rss_index=rss1 - rss0, rss_rows=rss2 - rss1, peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, value_us=float(np.mean(value)) * 1e6,
                        mst_leaves_us=float(np.mean(mst)) * 1e6, table_leaves_us=float(np.mean(table)) * 1e6, checked=checked, same=same))
        pd.DataFrame(got).to_csv(out / f"{args.dataset}_{args.family}.csv", index=False)
        del index, bound, tables, faces_of
        gc.collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
