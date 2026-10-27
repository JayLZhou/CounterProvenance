#!/usr/bin/env python3
"""Prototype: point location stored on the cells instead of the grid, at low d.

For every group of the index (an MCPST over the group's cells), axis ``k`` of the
group and local slab ``r``, a ``K``-bit mask marks the cells that meet slab ``r``
(``K`` = number of cells of the group).  A query ANDs one mask per axis; the
single bit left is the cell, whose value comes from the MCPST.  Compared with
the grid lookup of the index (one table read per group).  Output:
``results/cell_mask_proto.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from scm_bench import choose_treatments, zoo


def build(index):
    d = index.bound.d
    th = [np.asarray(t, dtype=np.float64) for t in index.slabs.thresholds]
    width = max(1, max(len(t) for t in th))
    tharr = np.full((d, width), np.inf)
    thlen = np.array([len(t) for t in th], dtype=np.int64)
    for a, t in enumerate(th):
        tharr[a, :len(t)] = t
    g_axis, g_off, g_W, g_vals = [], [], [], []   # per (group, axis) entries
    rows, locs, vals = [], [], []
    group_meta = []  # (first entry, n axes, W, value offset)
    rpos = lpos = vpos = 0
    cells_total = 0
    for leaf in index._fast_leaves:
        ov = leaf.cells.overlay
        K = ov.n_faces
        W = (K + 63) // 64
        F = ov.face_of_cell.reshape(ov.shape)
        first = len(g_axis)
        for k, a in enumerate(leaf.axes):
            n = ov.shape[k]
            M = np.zeros((n, W), dtype=np.uint64)
            for r in range(n):
                faces = np.unique(np.take(F, r, axis=k))
                np.bitwise_or.at(M[r], faces // 64, np.left_shift(np.uint64(1), (faces % 64).astype(np.uint64)))
            loc = np.searchsorted(np.asarray(ov.cuts[k]), np.arange(index.slabs.counts[a]), side="right")
            g_axis.append(a)
            g_off.append((rpos, lpos))
            rows.append(M.ravel())
            locs.append(loc.astype(np.int64))
            rpos += M.size
            lpos += len(loc)
        fv = np.asarray(leaf.cells.face_value, dtype=np.float64)
        group_meta.append((first, len(leaf.axes), W, vpos))
        vals.append(fv)
        vpos += len(fv)
        cells_total += K
    return dict(tharr=tharr, thlen=thlen, g_axis=np.asarray(g_axis, dtype=np.int64),
                g_off=np.asarray(g_off, dtype=np.int64).reshape(-1, 2), rows=np.concatenate(rows),
                locs=np.concatenate(locs), vals=np.concatenate(vals),
                meta=np.asarray(group_meta, dtype=np.int64).reshape(-1, 4), const=index._const,
                f32=int(index.bound.ensemble.input_dtype == np.float32),
                zt=index.bound.ensemble.zero_threshold if index.bound.ensemble.zero_threshold > 0 else -1.0,
                cells=cells_total, mask_bytes=sum(r.nbytes for r in rows) + sum(l.nbytes for l in locs))


@njit(cache=True)
def _query(t, tharr, thlen, g_axis, g_off, rows, locs, vals, meta, const, f32, zt, slab, work):
    d = tharr.shape[0]
    for a in range(d):
        v = t[a]
        if f32 == 1:
            v = np.float64(np.float32(v))
        if -zt <= v <= zt:
            v = 0.0
        lo, hi = 0, thlen[a]
        while lo < hi:
            mid = (lo + hi) >> 1
            if tharr[a, mid] < v:
                lo = mid + 1
            else:
                hi = mid
        slab[a] = lo
    total = const
    for g in range(meta.shape[0]):
        first, na, W, vo = meta[g, 0], meta[g, 1], meta[g, 2], meta[g, 3]
        for q in range(na):
            e = first + q
            r = locs[g_off[e, 1] + slab[g_axis[e]]]
            base = g_off[e, 0] + r * W
            if q == 0:
                for j in range(W):
                    work[j] = rows[base + j]
            else:
                for j in range(W):
                    work[j] &= rows[base + j]
        for j in range(W):
            m = work[j]
            if m != 0:
                total += vals[vo + j * 64 + np.int64(trailing_zeros(m))]
                break
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--queries", type=int, default=10_000)
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
        if T is None:
            continue
        bound = bind_scm(scm, unit, T, outcome)
        index = ProvenanceIndex(bound)
        if index._baxes:
            print(f"{spec}: has bitmask trees; skipped", flush=True)
            continue
        Q = np.ascontiguousarray(rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d)))
        nb = accel.NumbaIndex(index)
        ref = np.array([nb.query(t) for t in Q])
        p = build(index)
        maxW = max(1, int(p["meta"][:, 2].max())) if len(p["meta"]) else 1
        slab, work = np.zeros(d, dtype=np.int64), np.zeros(maxW, dtype=np.uint64)
        argv = (p["tharr"], p["thlen"], p["g_axis"], p["g_off"], p["rows"], p["locs"], p["vals"], p["meta"],
                p["const"], p["f32"], p["zt"], slab, work)
        got = np.array([_query(t, *argv) for t in Q])
        err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
        grid_cells = sum(len(leaf.cells.overlay.face_of_cell) for leaf in index._fast_leaves)
        row = dict(dataset=ds, family=fam, d=d, groups=len(index._fast_leaves), cells=p["cells"],
                   grid_cells=grid_cells, mask_mb=p["mask_bytes"] / 1e6, grid_mb=grid_cells * 8 / 1e6, err=err,
                   cell_mask_us=per_call_seconds(lambda: [_query(t, *argv) for t in Q], len(Q)) * 1e6,
                   grid_us=per_call_seconds(lambda: [nb.query(t) for t in Q], len(Q)) * 1e6)
        rows.append(row)
        write_csv("cell_mask_proto.csv", rows)
        print(f"{ds} {fam} d={d} groups={row['groups']} cells={row['cells']:,} grid={grid_cells:,} | masks "
              f"{row['mask_mb']:.2f} MB vs grid {row['grid_mb']:.2f} MB | cell-mask {row['cell_mask_us']:.2f}us "
              f"grid {row['grid_us']:.2f}us | err {err:.0e}", flush=True)


if __name__ == "__main__":
    main()
