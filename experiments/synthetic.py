#!/usr/bin/env python3
"""Controlled experiments on synthetic stump ensembles.

E1  1D sparse-change family (B = M = n, one tree changes per boundary).
E2  2D general-position stump grids: one CellTree over the whole overlay under
    every layout, versus the axis-support grouping of the same index.
E3  Coincident thresholds (kappa trees change per boundary).
E4  Heterogeneous edge costs: MST versus an arbitrary spanning tree.

Every index is checked against brute-force evaluation on random and
threshold points.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import cell_bytes, cell_records, environment, full_cell_index, write_csv
from counterprovenance import ProvenanceIndex, bind, iip_bytes, iip_records, synthetic


def check(bound, query, rng, count=200) -> float:
    pts = rng.uniform(-0.05, 1.05, size=(count, bound.d))
    worst = 0.0
    for t in pts:
        value, boxes = query(t)
        if sorted(boxes) != sorted(bound.selected(t).tolist()):
            raise AssertionError("provenance mismatch")
        worst = max(worst, abs(value - bound.value(t)))
    if worst > 1e-9:
        raise AssertionError(f"value mismatch {worst}")
    return worst


def cell_query(cells, slabs, bound):
    def q(t):
        piece = cells.piece(slabs.locate(bound.prepare(t)))
        return piece.value, piece.boxes()

    return q


def run_layouts(label, bound, rng, layouts, extra=None) -> list[dict]:
    rows = []
    for name, kwargs in layouts:
        start = time.perf_counter()
        cells, slabs = full_cell_index(bound, **kwargs)
        build = time.perf_counter() - start
        check(bound, cell_query(cells, slabs, bound), rng, 120)
        s = cells.stats
        rows.append(
            {
                "workload": label,
                "index": name,
                "M": bound.tree_count,
                "faces": s.faces,
                "grid_cells": s.grid_cells,
                "tree_weight": s.tree_weight,
                "walk_cost": s.walk_cost,
                "pft_nodes": s.pft_nodes,
                "pft_entries": s.pft_entries,
                "records": cell_records(cells),
                "bytes": cell_bytes(cells),
                "iip_records": iip_records(s.faces, bound.tree_count) + s.locator_words + len(cells.items),
                "iip_bytes": iip_bytes(s.faces, bound.tree_count) + 4 * (s.locator_words + len(cells.items)),
                "build_s": build,
                **(extra or {}),
            }
        )
        print(
            f"  {label:<22} {name:<26} faces={s.faces:>8,} pft={s.pft_nodes + s.pft_entries:>9,} "
            f"records={rows[-1]['records']:>10,} IIP={rows[-1]['iip_records']:>13,} build={build:6.2f}s",
            flush=True,
        )
    return rows


def e1(sizes, rng) -> list[dict]:
    print("E1  1D sparse-change family", flush=True)
    rows = []
    for n in sizes:
        order = rng.permutation(n)
        trees = [synthetic.stump(0, (int(order[i]) + 1) / (n + 1), float(rng.normal()), float(rng.normal())) for i in range(n)]
        bound = bind(synthetic.ensemble(trees, 1), np.zeros(1), [0])
        rows += run_layouts(
            f"1D n={n}",
            bound,
            rng,
            [
                ("path copying (chain)", {"layout": "path"}),
                ("node copying (chain)", {"layout": "node"}),
                ("hybrid (chain)", {"layout": "hybrid"}),
            ],
            {"n": n},
        )
    return rows


def e2(sizes, rng) -> list[dict]:
    print("E2  2D general-position stump grids", flush=True)
    rows = []
    for m in sizes:
        bound = bind(synthetic.stump_grid(m, rng), np.zeros(2), [0, 1])
        rows += run_layouts(
            f"2D M={m}",
            bound,
            rng,
            [
                ("CellTree path (rho-MST)", {"layout": "path"}),
                ("CellTree node (Euler tour)", {"layout": "node", "walk": "euler"}),
                ("CellTree node (best walk)", {"layout": "node"}),
                ("CellTree node (snake, grid)", {"layout": "node", "exact": False, "walk": "snake"}),
                ("CellTree hybrid", {"layout": "hybrid"}),
            ],
            {"M": m},
        )
        start = time.perf_counter()
        index = ProvenanceIndex(bound)  # axis-support grouping
        build = time.perf_counter() - start
        check(bound, lambda t: (lambda a: (a.value, a.boxes()))(index.query(t)), rng, 120)
        st = index.stats
        pieces = [len(index.query(t).pieces) for t in rng.uniform(0, 1, size=(200, 2))]
        rows.append(
            {
                "workload": f"2D M={m}",
                "index": "ProvenanceIndex (grouped)",
                "M": m,
                "faces": st.faces,
                "records": st.records,
                "bytes": st.packed_bytes,
                "groups": st.groups,
                "mean_pieces": float(np.mean(pieces)),
                "build_s": build,
            }
        )
        print(f"  {'2D M=' + str(m):<22} {'ProvenanceIndex (grouped)':<26} faces={st.faces:>8,} records={st.records:>10,} pieces={np.mean(pieces):.1f}", flush=True)
    return rows


def e3(groups, rng, m=256) -> list[dict]:
    print("E3  coincident thresholds", flush=True)
    rows = []
    for g in groups:
        bound = bind(synthetic.stump_grid(m, rng, group=g), np.zeros(2), [0, 1])
        rows += run_layouts(
            f"kappa={g}",
            bound,
            rng,
            [
                ("CellTree path (rho-MST)", {"layout": "path"}),
                ("CellTree node (best walk)", {"layout": "node"}),
                ("CellTree hybrid", {"layout": "hybrid"}),
            ],
            {"kappa": g},
        )
    return rows


def e4(rng) -> list[dict]:
    print("E4  heterogeneous costs: MST vs arbitrary spanning tree", flush=True)
    bound = bind(synthetic.stump_grid(64, rng, group=4, order="scattered"), np.zeros(2), [0, 1])
    return run_layouts(
        "ablation c_e=4",
        bound,
        rng,
        [
            ("path, rho-MST", {"layout": "path"}),
            ("path, c-MST", {"layout": "path", "plan_weight": "count"}),
            ("path, random spanning tree", {"layout": "path", "spanning": "arbitrary"}),
            ("node, best walk", {"layout": "node"}),
            ("node, walk of random tree", {"layout": "node", "spanning": "arbitrary"}),
            ("hybrid (MST plans)", {"layout": "hybrid"}),
            ("hybrid, random tree", {"layout": "hybrid", "spanning": "arbitrary"}),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(environment(), flush=True)
    rng = np.random.default_rng(2028)
    write_csv("e1_1d_chain.csv", e1((256, 1024) if args.quick else (256, 1024, 4096, 16384), rng))
    write_csv("e2_2d_grid.csv", e2((64, 128) if args.quick else (64, 256, 1024), rng))
    write_csv("e3_coincidence.csv", e3((1, 4) if args.quick else (1, 2, 4, 8, 16), rng))
    write_csv("e4_mst_ablation.csv", e4(rng))


if __name__ == "__main__":
    main()
