#!/usr/bin/env python3
"""How small the map from grid cells to cells can get: cell numbers of the shortest width, identical slices merged along one
or two axes, and a full decision diagram; and the read in which single trees are walked in one batch instead of having a grid.
Output: ``results/map2/<dataset>_<family>_map.csv`` and ``_read.csv``."""

from __future__ import annotations

import argparse
import gc
import pickle
import time
from bisect import bisect_left

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import bind_scm


def width(cells):
    return 1 if cells <= 1 << 8 else 2 if cells <= 1 << 16 else 4


def merged(a, w):
    """Bytes with identical slices merged: along the best axis, then again in the kept slices; and at every level."""
    best1 = best2 = None
    for k in range(a.ndim):
        s = np.moveaxis(a, k, 0).reshape(a.shape[k], -1)
        u, inv = np.unique(s, axis=0, return_inverse=True)
        one = 4 * a.shape[k] + w * u.size
        if best1 is None or one < best1:
            best1 = one
        two = one
        if a.ndim > 1 and len(u) > 0:
            rest = u.reshape((len(u),) + tuple(n for i, n in enumerate(a.shape) if i != k))
            for j in range(1, rest.ndim):
                r = np.moveaxis(rest, j, 0).reshape(rest.shape[j], -1)
                v = np.unique(r, axis=0)
                two = min(two, 4 * a.shape[k] + 4 * len(u) * rest.shape[j] // rest.shape[j] + 4 * rest.shape[j] + w * v.size)
        if best2 is None or two < best2:
            best2 = two
    # a decision diagram: from the last axis up, every distinct slice is one node with one pointer per slab
    level = a.reshape(-1, a.shape[-1])
    total = 0
    for k in range(a.ndim - 1, -1, -1):
        u, inv = np.unique(level, axis=0, return_inverse=True)
        total += u.size * (w if k == a.ndim - 1 else 4)
        if k == 0:
            break
        level = np.asarray(inv).reshape(-1, a.shape[k - 1])
    return best1, best2, total


class Walk:
    """The trees of the single-tree groups, one step per level for all of them at once."""

    def __init__(self, ens, trees):
        feature, threshold, left, right, value, roots = [], [], [], [], [], []
        shift = 0
        depth = 0
        for tree in (ens.trees[i] for i in trees):
            n = len(tree.feature)
            leaf = np.asarray(tree.feature) < 0
            feature.append(np.where(leaf, 0, tree.feature))
            threshold.append(np.where(leaf, np.inf, tree.threshold))
            left.append(np.where(leaf, np.arange(n), tree.left) + shift)
            right.append(np.where(leaf, np.arange(n), tree.right) + shift)
            value.append(np.asarray(tree.value, dtype=np.float64))
            roots.append(shift)
            shift += n
            stack, d = [(0, 0)], 0
            while stack:
                i, e = stack.pop()
                if tree.feature[i] < 0:
                    d = max(d, e)
                else:
                    stack += [(int(tree.left[i]), e + 1), (int(tree.right[i]), e + 1)]
            depth = max(depth, d)
        self.feature, self.threshold = np.concatenate(feature).astype(np.int64), np.concatenate(threshold)
        self.left, self.right = np.concatenate(left).astype(np.int64), np.concatenate(right).astype(np.int64)
        self.value, self.roots, self.depth = np.concatenate(value), np.array(roots, dtype=np.int64), depth

    def __call__(self, x):
        n = self.roots
        for _ in range(self.depth):
            n = np.where(x[self.feature[n]] <= self.threshold[n], self.left[n], self.right[n])
        return float(self.value[n].sum())


def read_row(args, node, index, ens, points):
    leaves = index._fast_leaves
    single = np.array([len(leaf.cells.slot_trees) == 1 for leaf in leaves])
    if not single.any() or index._nbase is None:
        return None
    rest = np.flatnonzero(~single)
    faces, values, shift = [], [], 0
    for leaf in leaves:
        faces.append(np.asarray(leaf.cells.overlay.face_of_cell, dtype=np.int64) + shift)
        values.append(np.asarray(leaf.cells.face_value, dtype=np.float64))
        shift += len(values[-1])
    face_of, value = np.concatenate(faces), np.concatenate(values)
    del faces, values
    walk = Walk(ens, [leaf.cells.slot_trees[0] for leaf, s in zip(leaves, single) if s])
    wide = [(a, t, r[:, rest]) for a, t, r in index._nwide] if len(rest) else []
    base = index._nbase[rest]
    cast = index._cast
    now, two, hybrid = [], [], []
    for x in points:
        c_x = cast(x)
        t0 = time.perf_counter()
        c = index._nbase.copy()
        for a, t, r in index._nwide:
            c += r[bisect_left(t, c_x[a])]
        v1 = float(index._ntable[c].sum())
        t1 = time.perf_counter()
        c = index._nbase.copy()
        for a, t, r in index._nwide:
            c += r[bisect_left(t, c_x[a])]
        v2 = float(value[face_of[c]].sum())
        t2 = time.perf_counter()
        v3 = walk(np.asarray(c_x, dtype=np.float64))
        if len(rest):
            c = base.copy()
            for a, t, r in wide:
                c += r[bisect_left(t, c_x[a])]
            v3 += float(value[face_of[c]].sum())
        t3 = time.perf_counter()
        now.append(t1 - t0)
        two.append(t2 - t1)
        hybrid.append(t3 - t2)
        if abs(v1 - v2) > 1e-6 or abs(v1 - v3) > 1e-6:
            raise AssertionError(f"reads differ: {v1} {v2} {v3}")
    return dict(dataset=args.dataset, family=args.family, node=node, groups=len(leaves), single=int(single.sum()), depth=walk.depth, now_us=float(np.mean(now)) * 1e6, two_us=float(np.mean(two)) * 1e6, hybrid_us=float(np.mean(hybrid)) * 1e6,
                walk_nodes=len(walk.feature))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--budget", type=float, default=2e7)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, rows = state["scm"], state["rows"]
    nodes = pd.read_csv(RESULTS / "main" / f"{args.dataset}_{args.family}_build.csv").node.tolist()
    out = RESULTS / "map2"
    out.mkdir(parents=True, exist_ok=True)
    unit0 = rows.iloc[0].to_dict()
    maps, reads = [], []
    for node in nodes:
        pa = list(scm.parents[node])
        bound = bind_scm(scm, unit0, pa, node, threads=1, native=True)
        index = ProvenanceIndex(bound, grouping="tables", budget=int(args.budget), build_threads=1, np_read_groups=0)
        for leaf in index._fast_leaves:
            o = leaf.cells.overlay
            w = width(o.n_faces)
            kind = np.uint8 if w == 1 else np.uint16 if w == 2 else np.uint32
            a = np.asarray(o.face_of_cell, dtype=kind).reshape(o.shape)
            one, two, dd = merged(a, w) if a.size <= 300_000_000 else (a.size * w, a.size * w, a.size * w)
            maps.append(dict(dataset=args.dataset, family=args.family, node=node, trees=len(leaf.cells.slot_trees), cells=o.n_faces, grid=a.size, axes=a.ndim, four=4 * a.size, short=w * a.size, merge1=one, merge2=two, diagram=dd))
        pd.DataFrame(maps).to_csv(out / f"{args.dataset}_{args.family}_map.csv", index=False)
        points = rows[pa].to_numpy(dtype=float)[:300]
        r = read_row(args, node, index, scm.ensembles[node], points)
        if r:
            reads.append(r)
            pd.DataFrame(reads).to_csv(out / f"{args.dataset}_{args.family}_read.csv", index=False)
        del index, bound
        gc.collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
