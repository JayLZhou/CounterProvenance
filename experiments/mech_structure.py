#!/usr/bin/env python3
"""Exp.5 and Exp.6 on the models of the main experiment, without building the full index.

Exp.5: the number of groups of the greedy algorithm against the optimum on samples of trees of the outcome's mechanism.
Exp.6: for the groups of every mechanism whose grid is small enough for the Python build: changes between adjacent cells,
cells against grid cells, and the records of the MST, a random spanning tree and CellTable.
Output: ``results/structure/<dataset>_<family>_grouping.csv`` and ``_cells.csv``."""

from __future__ import annotations

import argparse
import math
import pickle
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.cellindex import CellIndex, iip_records
from counterprovenance.ensemble import Slabs
from counterprovenance.grouping import Group, greedy_groups, tree_profiles
from counterprovenance.scm import bind_scm


def optimum(profiles, budget):
    """The fewest groups with total cells at most ``budget``, over all partitions (dynamic program on subsets)."""
    n = len(profiles)
    full = (1 << n) - 1
    cells = [1] * (full + 1)
    for mask in range(1, full + 1):
        cuts = {}
        for i in range(n):
            if mask >> i & 1:
                for a, c in profiles[i].cuts.items():
                    cuts.setdefault(a, set()).update(c)
        cells[mask] = math.prod(len(c) + 1 for c in cuts.values())
    best = list(cells)
    for k in range(1, n + 1):
        if best[full] <= budget:
            return k, best[full]
        nxt = [math.inf] * (full + 1)
        for mask in range(1, full + 1):
            low = mask & -mask
            rest = mask ^ low
            sub = rest
            while True:
                if sub:  # ``low | (rest ^ sub)`` is one group, ``sub`` the k groups of the others
                    v = cells[low | (rest ^ sub)] + best[sub]
                    if v < nxt[mask]:
                        nxt[mask] = v
                if not sub:
                    break
                sub = (sub - 1) & rest
        best = nxt
    return None, None


def grouping_rows(args, bound, slabs, node):
    profiles = tree_profiles(bound, slabs)
    rows = []
    for n in (8, 10, 12):
        if len(profiles) < n:
            continue
        for seed in range(5):
            rng = np.random.default_rng([2028, n, seed])
            sample = [profiles[i] for i in sorted(rng.choice(len(profiles), n, replace=False))]
            singles = sum(p.cells for p in sample)
            merged = {}
            for p in sample:
                for a, c in p.cuts.items():
                    merged.setdefault(a, set()).update(c)
            whole = math.prod(len(c) + 1 for c in merged.values())
            for name, budget in (("mid", int(math.sqrt(singles * max(whole, singles)))), ("low", 2 * singles)):
                t0 = time.perf_counter()
                opt, opt_cells = optimum(sample, budget)
                opt_s = time.perf_counter() - t0
                groups, deep = greedy_groups(sample, budget, max(budget, 10**9))
                total = sum(g.cells for g in groups)
                start_cap = 1 << 10
                if total > budget:  # the start of the real index is far below its budget: here it must be too
                    start_cap = 1
                    groups, deep = greedy_groups(sample, budget, max(budget, 10**9), start_cap=1)
                    total = sum(g.cells for g in groups)
                rows.append(dict(dataset=args.dataset, family=args.family, node=node, trees=n, seed=seed, budget_kind=name, budget=budget, singles=singles, whole=whole,
                                 optimum=opt, optimum_cells=opt_cells, greedy=len(groups) + len(deep), greedy_cells=total, greedy_fits=total <= budget, start_cap=start_cap, opt_s=opt_s))
    return rows


def cell_rows(args, bound, slabs, node, budget, limit, per_mechanism):
    boxes = bound.boxes
    grid = math.prod(slabs.counts)
    if grid <= 2_000_000:
        groups = [Group(frozenset(range(bound.d)), {}, list(range(len(boxes))), sorted({b.tree for b in boxes}))]
        sizes = [grid]
    else:
        groups, _ = greedy_groups(tree_profiles(bound, slabs), budget, max(budget, 10**9))
        sizes = [g.cells for g in groups]
    rows = []
    small = [i for i in np.argsort(sizes).tolist() if sizes[i] <= limit]
    pick = small[-per_mechanism:]  # the largest ones that fit: the small ones say little about the tree
    for gi in pick:
        g = groups[gi]
        axes = sorted(g.axes)
        kw = dict(items=g.items, ranges=[[slabs.ranges[i][a] for a in axes] for i in g.items], slots=[boxes[i].tree for i in g.items],
                  slab_counts=[slabs.counts[a] for a in axes], values=[boxes[i].value for i in g.items], layout="path", exact=True)
        t0 = time.perf_counter()
        mst = CellIndex(**kw, spanning="mst")
        mst_s = time.perf_counter() - t0
        rnd = CellIndex(**kw, spanning="arbitrary")
        c = np.array([e.count for e in mst.overlay.edges], dtype=np.int64)
        st, sr = mst.stats, rnd.stats
        rows.append(dict(dataset=args.dataset, family=args.family, node=node, groups_total=len(groups), groups_fit=len(small), group=gi, group_trees=len(g.trees), axes=len(axes),
                         grid_cells=st.grid_cells, cells=st.faces, edges=len(c), ce_mean=float(c.mean()) if len(c) else 0.0, ce_max=int(c.max()) if len(c) else 0,
                         mst_records=st.pft_nodes + st.pft_entries, random_records=sr.pft_nodes + sr.pft_entries, celltable_records=iip_records(st.faces, st.slots), mst_s=mst_s))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--limit", type=int, default=300_000)
    ap.add_argument("--per-mechanism", type=int, default=10)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, y, rows = state["scm"], state["y"], state["rows"]
    out = RESULTS / "structure"
    out.mkdir(parents=True, exist_ok=True)
    unit0 = rows.iloc[0].to_dict()
    nodes = [y] + sorted(set(scm.models) - {y})
    grouping, cells = [], []
    for node in nodes:
        bound = bind_scm(scm, unit0, list(scm.parents[node]), node, threads=1, native=True)
        slabs = Slabs(bound)
        if node == y:
            grouping = grouping_rows(args, bound, slabs, node)
            pd.DataFrame(grouping).to_csv(out / f"{args.dataset}_{args.family}_grouping.csv", index=False)
        cells += cell_rows(args, bound, slabs, node, int(args.budget), args.limit, args.per_mechanism)
        pd.DataFrame(cells).to_csv(out / f"{args.dataset}_{args.family}_cells.csv", index=False)
        print(node, len(cells), flush=True)


if __name__ == "__main__":
    main()
