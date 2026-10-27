#!/usr/bin/env python3
"""How many trees stay unresolved below a labelled space partition?

A k-d partition of the intervention space is grown best-first (always splitting
the node with the most unresolved trees, at the median wall of its unresolved
trees on the axis with the most such walls).  A tree is *resolved* in a node when
only one of its leaf boxes meets the node; its value then goes into the node's
label (stored once, at the node where the tree becomes resolved).  A query sums
the labels on its root-to-leaf path and evaluates only the trees unresolved at
its leaf node.  For each node budget this reports the mean number of unresolved
trees per query (sampled queries) and the label entries stored.
Output: ``results/kd_label_proto.csv``.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_csv
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from scm_bench import choose_treatments, zoo


class Node:
    __slots__ = ("lo", "hi", "boxes", "unresolved", "axis", "cut", "left", "right", "new_resolved")

    def __init__(self, lo, hi, boxes, unresolved, new_resolved):
        self.lo, self.hi, self.boxes, self.unresolved = lo, hi, boxes, unresolved
        self.new_resolved = new_resolved
        self.axis, self.cut, self.left, self.right = -1, 0.0, None, None


def make_node(lo, hi, cand, LO, HI, tree_of):
    """Boxes of ``cand`` meeting (lo, hi]; unresolved trees = trees with >= 2 such boxes."""
    m = (LO[cand] < hi).all(1) & (HI[cand] > lo).all(1)
    boxes = cand[m]
    trees, counts = np.unique(tree_of[boxes], return_counts=True)
    unresolved = trees[counts > 1]
    keep = np.isin(tree_of[boxes], unresolved)
    return boxes[keep], unresolved, int((counts == 1).sum())


def split(node, LO, HI, tree_of, d):
    best = None
    for a in range(d):
        w = np.concatenate((LO[node.boxes, a], HI[node.boxes, a]))
        w = w[(w > node.lo[a]) & (w < node.hi[a])]
        if len(w) and (best is None or len(w) > best[0]):
            best = (len(w), a, float(np.median(w)))
    if best is None:
        return False
    _, a, c = best
    hl, lr = node.hi.copy(), node.lo.copy()
    hl[a], lr[a] = c, c
    b1, u1, r1 = make_node(node.lo, hl, node.boxes, LO, HI, tree_of)
    b2, u2, r2 = make_node(lr, node.hi, node.boxes, LO, HI, tree_of)
    node.axis, node.cut = a, c
    node.left = Node(node.lo, hl, b1, u1, r1)
    node.right = Node(lr, node.hi, b2, u2, r2)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--budgets", default="1000,10000,100000")
    ap.add_argument("--queries", type=int, default=2000)
    args = ap.parse_args()
    rows, fitted = [], {}
    budgets = [int(b) for b in args.budgets.split(",")]
    for spec in args.settings.split(","):
        ds, fam, d = spec.split(":")
        d = int(d)
        if (ds, fam) not in fitted:
            d0 = Path(args.data) / ds
            meta = json.loads((d0 / "meta.json").read_text())
            df = pd.read_parquet(d0 / "data.parquet")
            df.columns = [str(c) for c in df.columns]
            edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
            outcome = str(meta["target"])
            scm = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]],
                          zoo(1000, 1)[fam], max_rows=50_000, seed=0, n_jobs=4)
            fitted[(ds, fam)] = (df, outcome, scm)
        df, outcome, scm = fitted[(ds, fam)]
        rng = np.random.default_rng(2028)
        unit = df.iloc[int(rng.integers(len(df)))].to_dict()
        T = choose_treatments(scm, outcome, d)
        bound = bind_scm(scm, unit, T, outcome)
        LO, HI = bound._LO.reshape(-1, d), bound._HI.reshape(-1, d)
        tree_of = np.array([b.tree for b in bound.boxes])
        X = bound.ensemble.cast(rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d)))
        # queries live in the data box; the partition only needs to cover it
        qlo = np.nextafter(X.min(0), -np.inf)
        qhi = X.max(0)
        boxes, unresolved, _ = make_node(qlo, qhi, np.arange(len(LO)), LO, HI, tree_of)
        root = Node(qlo, qhi, boxes, unresolved, 0)
        heap = [(-len(unresolved), 0, root)]
        nodes, tie = 1, 1
        labels = 0
        for budget in budgets:
            while heap and nodes < budget:
                negu, _, node = heapq.heappop(heap)
                if -negu == 0 or not split(node, LO, HI, tree_of, d):
                    continue
                nodes += 2
                labels += node.left.new_resolved + node.right.new_resolved
                for ch in (node.left, node.right):
                    heapq.heappush(heap, (-len(ch.unresolved), tie, ch))
                    tie += 1
            unres, depth = [], []
            for x in X:
                n, k = root, 0
                while n.left is not None:
                    n = n.left if x[n.axis] <= n.cut else n.right
                    k += 1
                unres.append(len(n.unresolved))
                depth.append(k)
            row = dict(dataset=ds, family=fam, d=d, trees=len(np.unique(tree_of)), budget=budget, nodes=nodes,
                       label_entries=labels, mean_unresolved=float(np.mean(unres)),
                       p90_unresolved=float(np.percentile(unres, 90)), mean_depth=float(np.mean(depth)))
            rows.append(row)
            write_csv("kd_label_proto.csv", rows)
            print(f"{ds} {fam} d={d} nodes={nodes:>7,} labels={labels:>9,} | unresolved mean "
                  f"{row['mean_unresolved']:6.1f} p90 {row['p90_unresolved']:6.0f} of {row['trees']} | "
                  f"depth {row['mean_depth']:.1f}", flush=True)


if __name__ == "__main__":
    main()
