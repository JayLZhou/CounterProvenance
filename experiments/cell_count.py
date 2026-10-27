#!/usr/bin/env python3
"""How many counterfactual cells are there, compared with the grid the groups pay for?

Without mediators every cell is a box (the intersection of one leaf box per
tree), so the cells of a set of trees are obtained by refining the boxes tree
by tree, without any grid.  For each ``dataset:family:d`` this reports

* the current grouping: number of groups, their total grid cells and their
  total faces (true cells);
* the cells of all varying trees together, added in order of support size,
  until ``--cap`` cells (then the number of trees that fit).

Output: ``results/cell_count.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from scm_bench import choose_treatments, zoo


def refine(cells_lo, cells_hi, box_lo, box_hi):
    """Nonempty intersections of every cell with every box of one tree (the boxes partition R^d)."""
    out_lo, out_hi = [], []
    for blo, bhi in zip(box_lo, box_hi):
        m = (cells_lo < bhi).all(1) & (cells_hi > blo).all(1)
        if m.any():
            out_lo.append(np.maximum(cells_lo[m], blo))
            out_hi.append(np.minimum(cells_hi[m], bhi))
    return np.concatenate(out_lo), np.concatenate(out_hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--cap", type=int, default=3_000_000)
    args = ap.parse_args()
    rows, fitted = [], {}
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
        index = ProvenanceIndex(bound)
        grid = sum(len(leaf.cells.overlay.face_of_cell) for leaf in index._fast_leaves)
        faces = sum(leaf.cells.overlay.n_faces for leaf in index._fast_leaves)
        by_tree = {}
        for b in bound.boxes:
            by_tree.setdefault(b.tree, []).append(b)
        varying = [bs for bs in by_tree.values() if len(bs) > 1]
        support = lambda bs: sum(any(math.isfinite(b.lo[a]) or math.isfinite(b.hi[a]) for b in bs) for a in range(d))
        varying.sort(key=support)
        lo = np.full((1, d), -np.inf)
        hi = np.full((1, d), np.inf)
        fit_trees, start, curve = 0, time.perf_counter(), []
        for k, bs in enumerate(varying):
            nlo, nhi = refine(lo, hi, np.array([b.lo for b in bs]), np.array([b.hi for b in bs]))
            if len(nlo) > args.cap:
                break
            lo, hi, fit_trees = nlo, nhi, k + 1
            if (k + 1) in (1, 10, 50, 100, 200, 500, 1000):
                curve.append(f"{k + 1}:{len(lo)}")
        row = dict(dataset=ds, family=fam, d=d, boxes=len(bound.boxes), varying_trees=len(varying),
                   groups=len(index.groups), deep_trees=index.stats.bitmask_trees, group_grid=grid,
                   group_faces=faces, all_trees_fit=fit_trees, cells_of_fit=len(lo), refine_s=time.perf_counter() - start,
                   curve=" ".join(curve))
        rows.append(row)
        write_csv("cell_count.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
