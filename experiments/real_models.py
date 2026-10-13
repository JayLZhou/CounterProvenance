#!/usr/bin/env python3
"""Real models on California housing: sklearn GBR/HGB/RF, XGBoost (exact, hist), LightGBM.

R1  compiler exactness: normalised ensemble vs ``model.predict`` and the
    model's own leaf ids (including rows on split thresholds);
R2  index size, pieces, latency and exactness for d = 1, 2, 3, 4, 6, 8
    intervention axes: IIP, one MCPST over the whole overlay, the grouped
    index, the chain endpoint (tau = 0) and the autotuned index;
R3  boundary statistics and the MST ablation on the d = 2 overlay of every model.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter

import numpy as np

from common import (
    california,
    cell_records,
    environment,
    full_cell_index,
    markdown_table,
    model_zoo,
    native_leaves,
    per_call_seconds,
    value_tolerance,
    write_csv,
)
from counterprovenance import ProvenanceIndex, autotune, bind, from_model, iip_bytes, iip_records

FULL_OVERLAY_LIMIT = 400_000


def what_if_points(X, axes, bound, rng, count):
    lo = np.percentile(X[:, axes], 1, axis=0)
    hi = np.percentile(X[:, axes], 99, axis=0)
    pts = list(rng.uniform(lo, hi, size=(count, len(axes))))
    # threshold points: every intervention coordinate set to a random split value
    for _ in range(count // 3):
        p = rng.uniform(lo, hi)
        a = int(rng.integers(len(axes)))
        th = [b.hi[a] for b in bound.boxes if math.isfinite(b.hi[a])]
        if th:
            p[a] = th[int(rng.integers(len(th)))]
        pts.append(p)
    return np.array(pts)


def r1(name, model, ensemble, X, rng) -> dict:
    rows = X[rng.integers(len(X), size=2000)]
    err = float(np.max(np.abs(ensemble.predict(rows) - model.predict(rows))))
    native = native_leaves(model, rows[:500])
    if native is not None:
        ours = ensemble.leaves(rows[:500])
        mapped = np.array([[t.native_leaf[n] for t, n in zip(ensemble.trees, row)] for row in ours])
        leaf_match = float(np.mean(mapped == native))
    else:
        leaf_match = float("nan")
    return {"model": name, "trees": ensemble.tree_count, "max_value_error": err, "leaf_id_match": leaf_match}


def evaluate(index, bound, model, pts, native_sample) -> dict:
    tol = value_tolerance(model)
    worst = 0.0
    mismatch = 0
    pieces = []
    rows = np.array([bound.full_input(t) for t in pts])
    predictions = model.predict(rows)
    native = native_leaves(model, rows[:native_sample])
    for i, t in enumerate(pts):
        answer = index.query(t)
        pieces.append(len(answer.pieces))
        worst = max(worst, abs(answer.value - predictions[i]))
        labels = answer.labels()
        expected = Counter(bound.label(j) for j in bound.selected(t))
        if labels != expected:
            mismatch += 1
        elif native is not None and i < native_sample:
            if labels != Counter((m, int(native[i, m])) for m in range(bound.tree_count)):
                mismatch += 1
    if worst > tol:
        raise AssertionError(f"value error {worst} > {tol}")
    if mismatch:
        raise AssertionError(f"{mismatch} provenance mismatches")
    q = per_call_seconds(lambda: [index.query(t) for t in pts[:200]], min(200, len(pts)))
    return {
        "mean_pieces": float(np.mean(pieces)),
        "max_pieces": int(np.max(pieces)),
        "query_us": q * 1e6,
        "max_value_error": worst,
    }


def r2(name, model, ensemble, X, y, axes_order, dims, unit_row, rng, quick) -> list[dict]:
    rows = []
    for d in dims:
        axes = sorted(axes_order[:d])
        bound = bind(ensemble, unit_row, axes)
        pts = what_if_points(X, axes, bound, rng, 150 if quick else 300)
        full_rows = np.array([bound.full_input(t) for t in pts[:200]])
        replay_us = per_call_seconds(lambda: [model.predict(r[None, :]) for r in full_rows], len(full_rows)) * 1e6
        from counterprovenance import Slabs

        k_grid = Slabs(bound).grid_cells
        base = {"model": name, "d": d, "N_boxes": len(bound.boxes), "K_grid": k_grid, "replay_us": replay_us}
        configs = []
        if k_grid <= FULL_OVERLAY_LIMIT:
            configs.append(("one MCPST (tau=inf, no groups)", dict(tau=math.inf, grouping="none")))
            configs.append(("grouped (tau=inf)", dict(tau=math.inf)))
        configs.append(("chain endpoint (tau=0)", dict(tau=0)))
        for label, kwargs in configs:
            start = time.perf_counter()
            try:
                index = ProvenanceIndex(bound, max_leaf_cells=FULL_OVERLAY_LIMIT, **kwargs)
            except MemoryError:
                rows.append({**base, "index": label, "status": "leaf overlay too large"})
                continue
            build = time.perf_counter() - start
            st = index.stats
            row = {**base, "index": label, "records": st.records, "bytes": st.packed_bytes, "faces": st.faces,
                   "groups": st.groups, "leaves": st.leaves, "build_s": build}
            row.update(evaluate(index, bound, model, pts, 60))
            if label.startswith("one MCPST"):
                row["iip_records"] = iip_records(st.faces, bound.tree_count) + st.locator_words + st.assignments
                row["iip_bytes"] = iip_bytes(st.faces, bound.tree_count) + 4 * (st.locator_words + st.assignments)
            rows.append(row)
            print(f"  {name:<14} d={d} {label:<32} records={row['records']:>9,} pieces={row['mean_pieces']:6.1f} "
                  f"q={row['query_us']:7.1f}us replay={replay_us:7.1f}us build={build:6.2f}s", flush=True)
        start = time.perf_counter()
        sample = pts[rng.integers(len(pts), size=48)]
        tuned = autotune(bound, sample, taus=(0, 256, 4096, 65536), budgets=(None, 8, 4, 1), pieces_budget=16,
                         max_leaf_cells=FULL_OVERLAY_LIMIT)
        build = time.perf_counter() - start
        chosen = [r for r in tuned.table if r.get("chosen")][0]
        st = tuned.index.stats
        row = {**base, "index": f"autotuned (pieces<=16): tau={chosen['tau']}, g={chosen['group_budget']}",
               "records": st.records, "bytes": st.packed_bytes, "faces": st.faces, "groups": st.groups,
               "leaves": st.leaves, "build_s": build, "configs_tried": len(tuned.table)}
        row.update(evaluate(tuned.index, bound, model, pts, 60))
        rows.append(row)
        print(f"  {name:<14} d={d} {'autotuned':<32} records={row['records']:>9,} pieces={row['mean_pieces']:6.1f} "
              f"q={row['query_us']:7.1f}us  [{row['index']}]", flush=True)
    return rows


def r3(name, model, ensemble, X, axes_order, unit_row) -> list[dict]:
    axes = sorted(axes_order[:2])
    bound = bind(ensemble, unit_row, axes)
    rows = []
    for layout in ("path", "hybrid", "node"):
        for spanning in ("mst", "arbitrary"):
            cells, _ = full_cell_index(bound, layout=layout, spanning=spanning)
            counts = [e.count for e in cells.overlay.edges]
            rows.append({
                "model": name, "layout": layout, "spanning": spanning, "faces": cells.stats.faces,
                "grid_cells": cells.stats.grid_cells, "face_edges": len(counts),
                "mean_c": float(np.mean(counts)) if counts else 0.0, "max_c": max(counts) if counts else 0,
                "frac_c_gt_1": float(np.mean([c > 1 for c in counts])) if counts else 0.0,
                "tree_weight": cells.stats.tree_weight, "walk_cost": cells.stats.walk_cost,
                "pft_records": cells.stats.pft_nodes + cells.stats.pft_entries, "records": cell_records(cells),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", default="")
    parser.add_argument("--suffix", default="", help="output file suffix (for parallel runs)")
    args = parser.parse_args()
    print(environment(), flush=True)
    X, y, names = california(args.data_home)
    rng = np.random.default_rng(2028)
    unit_index = int(rng.integers(len(X)))
    unit_row = X[unit_index]
    print(f"California housing {X.shape}, unit #{unit_index}", flush=True)
    zoo = model_zoo(args.quick)
    if args.models:
        zoo = {k: v for k, v in zoo.items() if k in args.models.split(",")}
    dims = (1, 2, 3) if args.quick else (1, 2, 3, 4, 6, 8)
    r1_rows, r2_rows, r3_rows = [], [], []
    for name, factory in zoo.items():
        start = time.perf_counter()
        model = factory().fit(X, y)
        print(f"{name}: fitted in {time.perf_counter() - start:.1f}s", flush=True)
        ensemble = from_model(model)
        r1_rows.append(r1(name, model, ensemble, X, rng))
        print(f"  R1 {r1_rows[-1]}", flush=True)
        # intervention axes: features ordered by how often the ensemble splits on them
        usage = Counter(int(f) for t in ensemble.trees for f in t.feature if f >= 0)
        axes_order = [f for f, _ in usage.most_common()] + [f for f in range(X.shape[1]) if f not in usage]
        r2_rows += r2(name, model, ensemble, X, y, axes_order, dims, unit_row, rng, args.quick)
        r3_rows += r3(name, model, ensemble, X, axes_order, unit_row)
        write_csv(f"r1_compilers{args.suffix}.csv", r1_rows)
        write_csv(f"r2_real_indexes{args.suffix}.csv", r2_rows)
        write_csv(f"r3_mst_real{args.suffix}.csv", r3_rows)
    print(markdown_table(r1_rows, [("model", "model"), ("trees", "trees"), ("max_value_error", "max |value error|"), ("leaf_id_match", "leaf-id match")]))


if __name__ == "__main__":
    main()
