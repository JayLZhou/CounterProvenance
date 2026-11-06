#!/usr/bin/env python3
"""CellTable as the paper defines it, built naively: every grid cell of the outcome's ensemble is evaluated on every tree, grid cells
with the same leaves form one cell, and the table keeps one row per cell (the leaf of each tree and the value) and a map from grid
cells to cells.  No cell graph, no spanning tree, no groups.  It logs its progress, so that a build stopped for its memory or its
time still says how many cells it had found.  Output: ``results/naive/<dataset>_<family>.csv``."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import resource
import time

import numpy as np
import pandas as pd
from pandas._libs import hashtable as ht

from common import RESULTS
from counterprovenance.ensemble import Slabs
from counterprovenance.scm import bind_scm


def rss() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def walk(tree, X):
    """The leaf of ``tree`` for every row of ``X``."""
    feature, threshold, left, right = (np.asarray(a) for a in (tree.feature, tree.threshold, tree.left, tree.right))
    n = np.zeros(len(X), dtype=np.int64)
    inner = np.flatnonzero(feature[n] >= 0)
    at = np.arange(len(X))
    while len(inner):
        now = n[inner]
        go = X[at[inner], feature[now]] <= threshold[now]
        n[inner] = np.where(go, left[now], right[now])
        inner = inner[feature[n[inner]] >= 0]
    return n


class Leaves:
    """The leaf of one tree at every grid cell, through a table over the tree's own thresholds when that table is small."""

    def __init__(self, tree, values, limit=1 << 22):
        self.tree = tree
        feature, threshold = np.asarray(tree.feature), np.asarray(tree.threshold)
        self.axes = sorted({int(a) for a in feature[feature >= 0]})
        own = {a: np.unique(threshold[feature == a]) for a in self.axes}
        shape = [len(own[a]) + 1 for a in self.axes]
        self.table = None
        if math.prod(shape) <= limit:
            # rank of every slab of the input among the tree's own thresholds
            self.rank = {a: np.searchsorted(own[a], values[a], side="left") for a in self.axes}
            self.strides = np.cumprod([1] + shape[:0:-1])[::-1].tolist() if shape else []
            coarse = np.indices(shape).reshape(len(shape), -1) if shape else np.zeros((0, 1), dtype=np.int64)
            X = np.zeros((coarse.shape[1], len(values)))
            for k, a in enumerate(self.axes):
                X[:, a] = np.append(own[a], own[a][-1] + 1.0 if len(own[a]) else 0.0)[coarse[k]]
            self.table = walk(tree, X)

    def __call__(self, slabs, values):
        if self.table is not None:
            c = np.zeros(len(slabs[0]), dtype=np.int64)
            for k, a in enumerate(self.axes):
                c += self.rank[a][slabs[a]] * self.strides[k]
            return self.table[c]
        X = np.stack([values[a][slabs[a]] for a in range(len(values))], axis=1)
        return walk(self.tree, X)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--chunk", type=int, default=1 << 20)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, y, rows = state["scm"], state["y"], state["rows"]
    out = RESULTS / "naive"
    out.mkdir(parents=True, exist_ok=True)
    bound = bind_scm(scm, rows.iloc[0].to_dict(), list(scm.parents[y]), y, threads=1, native=True)
    slabs = Slabs(bound)
    counts = list(slabs.counts)
    values = [np.append(np.asarray(t, dtype=np.float64), (t[-1] + 1.0) if len(t) else 0.0) for t in slabs.thresholds]
    grid = math.prod(counts)
    ens = scm.ensembles[y]
    M = len(ens.trees)
    t0 = time.perf_counter()
    leaves = [Leaves(t, values) for t in ens.trees]
    rng = np.random.default_rng(2028)
    salt = rng.integers(1, 1 << 63, size=M, dtype=np.uint64) | np.uint64(1)
    print(f"grid {grid:,} trees {M} prepared in {time.perf_counter() - t0:.1f} s", flush=True)
    # pass 1: the cell of every grid cell, by a 64-bit hash of its leaves
    cell_of = np.empty(grid, dtype=np.uint32)
    table = ht.UInt64HashTable()
    first: list[np.ndarray] = []
    done_at = time.perf_counter()
    for lo in range(0, grid, args.chunk):
        hi = min(grid, lo + args.chunk)
        slab = np.unravel_index(np.arange(lo, hi, dtype=np.int64), counts)
        h = np.zeros(hi - lo, dtype=np.uint64)
        for m, leaf in enumerate(leaves):
            h += leaf(slab, values).astype(np.uint64) * salt[m]
        prior = len(table)
        labels = table.get_labels(h, ht.UInt64Vector(), prior, -1)
        cell_of[lo:hi] = labels
        new = labels >= prior
        if new.any():
            _, at = np.unique(labels[new], return_index=True)
            first.append(lo + np.flatnonzero(new)[at])
        if time.perf_counter() - done_at > 300 or hi == grid:
            done_at = time.perf_counter()
            print(f"{time.strftime('%F %T')} grid {hi:,}/{grid:,} cells {len(table):,} rss {rss() / 1e9:.1f} GB elapsed {time.perf_counter() - t0:.0f} s", flush=True)
    pass1 = time.perf_counter() - t0
    K = len(table)
    # pass 2: one row per cell, the leaf of every tree, and its value
    reps = np.concatenate(first) if first else np.zeros(0, dtype=np.int64)
    slab = np.unravel_index(reps, counts)
    rows_of = np.empty((K, M), dtype=np.uint16)
    value = np.zeros(K, dtype=np.float64)
    for m, (leaf, tree) in enumerate(zip(leaves, ens.trees)):
        n = leaf(slab, values)
        rows_of[:, m] = n
        value += np.asarray(tree.value, dtype=np.float64)[n]
    build_s = time.perf_counter() - t0
    r = dict(dataset=args.dataset, family=args.family, node=y, parents=len(counts), trees=M, grid=grid, cells=K, map_bytes=cell_of.nbytes, rows_bytes=rows_of.nbytes,
             value_bytes=value.nbytes, table_bytes=cell_of.nbytes + rows_of.nbytes + value.nbytes, pass1_s=pass1, build_s=build_s, rss=rss(),
             peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, status="ok")
    pd.DataFrame([r]).to_csv(out / f"{args.dataset}_{args.family}.csv", index=False)
    print(r, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
