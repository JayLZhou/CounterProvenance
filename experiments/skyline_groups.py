#!/usr/bin/env python3
"""Group trees by the skyline of their axis supports and count the cells of each group.

The support of a tree is the set of intervention axes it still splits on.  A
tree whose support is contained in another support is dominated; the skyline
is the set of maximal supports.  Every tree joins the smallest skyline support
that contains it, and the cells of each group are obtained by refining the
boxes of its trees (no grid), up to ``--cap`` cells per group.

Output: ``results/skyline_groups.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cell_count import refine
from common import write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from scm_bench import choose_treatments, zoo


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
        groups_now = len(ProvenanceIndex(bound).groups)

        by_tree: dict[int, list] = {}
        for b in bound.boxes:
            by_tree.setdefault(b.tree, []).append(b)
        trees = {m: bs for m, bs in by_tree.items() if len(bs) > 1}
        support = {m: frozenset(a for a in range(d) if any(math.isfinite(b.lo[a]) or math.isfinite(b.hi[a])
                                                           for b in bs)) for m, bs in trees.items()}
        distinct = set(support.values())
        skyline = [s for s in distinct if not any(s < o for o in distinct)]
        members: dict[frozenset, list[int]] = {s: [] for s in skyline}
        for m, s in support.items():
            members[min((o for o in skyline if s <= o), key=lambda o: (len(o), sorted(o)))].append(m)

        cells, full, fit_trees = [], 0, 0
        for s, ms in members.items():
            lo, hi = np.full((1, d), -np.inf), np.full((1, d), np.inf)
            k = 0
            for m in ms:
                bs = trees[m]
                nlo, nhi = refine(lo, hi, np.array([b.lo for b in bs]), np.array([b.hi for b in bs]))
                if len(nlo) > args.cap:
                    break
                lo, hi, k = nlo, nhi, k + 1
            cells.append(len(lo))
            fit_trees += k
            full += k == len(ms)
        sizes = Counter(len(s) for s in support.values())
        row = dict(dataset=ds, family=fam, d=d, varying_trees=len(trees), groups_now=groups_now,
                   distinct_supports=len(distinct), skyline=len(skyline),
                   skyline_sizes=" ".join(f"{len(s)}" for s in sorted(skyline, key=len)),
                   support_hist=" ".join(f"{k}:{v}" for k, v in sorted(sizes.items())),
                   groups_within_cap=full, trees_within_cap=fit_trees,
                   cells_total=sum(cells), cells_max=max(cells) if cells else 0)
        rows.append(row)
        write_csv("skyline_groups.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
