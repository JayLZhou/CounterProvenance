#!/usr/bin/env python3
"""Does wall alignment (refinement dominance) let many more trees share one cell graph?

A tree is dominated by a group when every wall of the tree is already a wall
of the group's cells, i.e., the group's partition refines the tree's: adding
it creates no cell.  Cells grow only with the walls a tree adds.  This script
fills one group up to ``--cap`` cells in two orders and reports how many trees
fit:

* ``arbitrary``: trees in index order;
* ``dominance``: at every step, among ``--sample`` random remaining trees (and
  all trees that add no cell), the one that adds the fewest new cells.

Output: ``results/dominance_groups.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cell_count import refine
from common import write_csv
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from scm_bench import choose_treatments, zoo


def added_cells(lo, hi, blo, bhi):
    """Cells that one tree adds to the refinement ``(lo, hi)``: sum over cells of (boxes hit - 1)."""
    hits = 0
    for a, b in zip(blo, bhi):
        hits += int(((lo < b).all(1) & (hi > a).all(1)).sum())
    return hits - len(lo)


def fill(trees, order_fn, cap, d):
    lo, hi = np.full((1, d), -np.inf), np.full((1, d), np.inf)
    left = list(trees)
    n = 0
    while left:
        k = order_fn(lo, hi, left)
        blo, bhi = trees[k]
        nlo, nhi = refine(lo, hi, blo, bhi)
        if len(nlo) > cap:
            break
        lo, hi = nlo, nhi
        left.remove(k)
        n += 1
    return n, len(lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--cap", type=int, default=300_000)
    ap.add_argument("--sample", type=int, default=20)
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
        by_tree: dict[int, list] = {}
        for b in bound.boxes:
            by_tree.setdefault(b.tree, []).append(b)
        trees = {m: (np.array([b.lo for b in bs]), np.array([b.hi for b in bs]))
                 for m, bs in by_tree.items() if len(bs) > 1}

        start = time.perf_counter()
        n_arb, k_arb = fill(trees, lambda lo, hi, left: left[0], args.cap, d)
        t_arb = time.perf_counter() - start

        free_seen = set()

        def dominance(lo, hi, left):
            pick = rng.choice(len(left), size=min(args.sample, len(left)), replace=False)
            best, best_cost = None, None
            for i in pick:
                k = left[i]
                cost = added_cells(lo, hi, *trees[k])
                if cost == 0:
                    free_seen.add(k)
                    return k
                if best is None or cost < best_cost:
                    best, best_cost = k, cost
            return best

        start = time.perf_counter()
        n_dom, k_dom = fill(trees, dominance, args.cap, d)
        row = dict(dataset=ds, family=fam, d=d, varying_trees=len(trees), cap=args.cap,
                   arbitrary_trees=n_arb, arbitrary_cells=k_arb, dominance_trees=n_dom, dominance_cells=k_dom,
                   free_trees=len(free_seen), arbitrary_s=t_arb, dominance_s=time.perf_counter() - start)
        rows.append(row)
        write_csv("dominance_groups.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
