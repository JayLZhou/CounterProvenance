#!/usr/bin/env python3
"""Prototype: leaf bitmasks stored along each axis like the cell graph's MST.

Every varying tree gets one 64-bit field per 64 leaf boxes.  For axis ``a`` and
global slab ``s``, row ``M_a[s]`` holds, per field, the boxes whose range on
``a`` covers ``s``; a query ANDs one row per axis and reads the lowest set bit
of every field.  Moving from slab ``s - 1`` to ``s`` crosses one cut, and only
the fields of trees with a wall there change: the slabs of an axis form a path,
the cell graph of that axis, whose MST is the path itself.  So the rows are
stored as in MCPST: a full row every ``k`` slabs (checkpoint) and, for every
other slab, only the fields that change.  ``k = 1`` is the plain bitmask.

For each ``dataset:family:d`` and ``k``: bytes, time of one numba call per query,
native time per query, checked against the index.  Output:
``results/mask_mst_proto.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.ensemble import Slabs
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmNumba
from scm_bench import choose_treatments, zoo


def build(bound, k):
    slabs = Slabs(bound)
    d = bound.d
    by_tree: dict[int, list[int]] = {}
    for i, b in enumerate(bound.boxes):
        by_tree.setdefault(b.tree, []).append(i)
    const = bound.offset
    fields = []
    for m, items in sorted(by_tree.items()):
        if len(items) == 1:
            const += bound.boxes[items[0]].value
            continue
        for s in range(0, len(items), 64):
            fields.append(items[s:s + 64])
    F = len(fields)
    vals = np.zeros((F, 64))
    for j, items in enumerate(fields):
        vals[j, :len(items)] = [bound.boxes[i].value for i in items]
    th = [np.asarray(t, dtype=np.float64) for t in slabs.thresholds]
    counts = slabs.counts
    rows = []  # full rows per axis, uint64 (C_a, F)
    for a in range(d):
        M = np.zeros((counts[a], F), dtype=np.uint64)
        for j, items in enumerate(fields):
            for bit, i in enumerate(items):
                lo, hi = slabs.ranges[i][a]
                M[lo:hi, j] |= np.uint64(1 << bit)
        rows.append(M)
    # checkpoints every k slabs, deltas (field, word) for every other slab
    cp_off, cp, dl_off, dl_field, dl_word, slab_off = [0], [], [], [], [], [0]
    for a in range(d):
        M = rows[a]
        C = M.shape[0]
        cps = M[::k]
        cp.append(cps.ravel())
        cp_off.append(cp_off[-1] + cps.size)
        offs = [len(dl_field)]
        for s in range(C):
            if s % k:
                ch = np.flatnonzero(M[s] != M[s - 1])
                dl_field.extend(ch.tolist())
                dl_word.extend(M[s, ch].tolist())
            offs.append(len(dl_field))
        dl_off.append(np.asarray(offs, dtype=np.int64))
        slab_off.append(slab_off[-1] + C + 1)
    width = max(1, max(len(t) for t in th))
    tharr = np.full((d, width), np.inf)
    thlen = np.zeros(d, dtype=np.int64)
    for a, t in enumerate(th):
        tharr[a, :len(t)] = t
        thlen[a] = len(t)
    pack = dict(
        F=F, k=k, tharr=tharr, thlen=thlen, cp=np.concatenate(cp) if cp else np.zeros(0, np.uint64),
        cp_off=np.asarray(cp_off, dtype=np.int64), dl_off=np.concatenate(dl_off), slab_off=np.asarray(slab_off),
        dl_field=np.asarray(dl_field, dtype=np.int64), dl_word=np.asarray(dl_word, dtype=np.uint64),
        vals=vals, const=const, full_bytes=sum(M.nbytes for M in rows),
        f32=int(bound.ensemble.input_dtype == np.float32),
        zt=bound.ensemble.zero_threshold if bound.ensemble.zero_threshold > 0 else -1.0)
    pack["bytes"] = pack["cp"].nbytes + pack["dl_field"].size * 4 + pack["dl_word"].nbytes + vals.nbytes
    return pack


@njit(cache=True)
def _query(t, F, k, tharr, thlen, cp, cp_off, dl_off, slab_off, dl_field, dl_word, vals, const, f32, zt, work, buf):
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
        s = lo
        c = s // k
        base = cp_off[a] + c * F
        target = work if a == 0 else buf
        for j in range(F):
            target[j] = cp[base + j]
        so = slab_off[a]
        for q in range(dl_off[so + c * k], dl_off[so + s + 1]):
            target[dl_field[q]] = dl_word[q]
        if a > 0:
            for j in range(F):
                work[j] &= buf[j]
    total = const
    for j in range(F):
        m = work[j]
        if m != 0:
            total += vals[j, np.int64(trailing_zeros(m))]
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--ks", default="1,4,16,64")
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
        bound = bind_scm(scm, unit, T, outcome)
        Q = np.ascontiguousarray(rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d)))
        index = ProvenanceIndex(bound)
        nb = accel.NumbaIndex(index)
        ref = np.array([nb.query(t) for t in Q])
        index_us = per_call_seconds(lambda: [nb.query(t) for t in Q], len(Q)) * 1e6
        online = ScmNumba(scm, unit, T, outcome)
        online.query(Q[0])
        online_us = per_call_seconds(lambda: [online.query(t) for t in Q], len(Q)) * 1e6
        for k in (int(v) for v in args.ks.split(",")):
            start = time.perf_counter()
            p = build(bound, k)
            build_s = time.perf_counter() - start
            work = np.zeros(max(1, p["F"]), dtype=np.uint64)
            buf = np.zeros_like(work)
            argv = (p["F"], p["k"], p["tharr"], p["thlen"], p["cp"], p["cp_off"], p["dl_off"], p["slab_off"],
                    p["dl_field"], p["dl_word"], p["vals"], p["const"], p["f32"], p["zt"], work, buf)
            got = np.array([_query(t, *argv) for t in Q])
            err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
            call_us = per_call_seconds(lambda: [_query(t, *argv) for t in Q], len(Q)) * 1e6
            row = dict(dataset=ds, family=fam, d=d, k=k, fields=p["F"], mask_mb=p["bytes"] / 1e6,
                       full_mask_mb=p["full_bytes"] / 1e6, index_mb=index.stats.packed_bytes / 1e6,
                       build_s=build_s, err=err, mask_us=call_us, index_us=index_us, online_us=online_us)
            rows.append(row)
            write_csv("mask_mst_proto.csv", rows)
            print(f"{ds} {fam} d={d} k={k:3d} fields={p['F']} masks {row['mask_mb']:7.2f} MB (full rows "
                  f"{row['full_mask_mb']:.2f}) | mask {call_us:6.2f}us index {index_us:6.2f}us online "
                  f"{online_us:6.1f}us | err {err:.0e}", flush=True)


if __name__ == "__main__":
    main()
