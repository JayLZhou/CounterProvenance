#!/usr/bin/env python3
"""How much of Treelite's time per query is its calling path?  Star graphs only (one model).

For ``dataset:family:d``: compile the outcome's model with tl2cgen and time one query
(a) through ``tl2cgen.DMatrix`` + ``Predictor.predict`` (what ``scm_full`` does),
(b) by calling the generated C ``predict`` directly with ctypes from Python,
(c) natively, in a C loop over the queries (``native/tl_loop.c``);
and our index (d) with one numba call per query from Python and (e) in a numba loop.
Output: ``results/tl_fair.csv``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baselines_all import treelite_predictor
from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.scm import ancestors, fit_scm, parents_of, star_bind
from scm_bench import choose_treatments, zoo

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--queries", type=int, default=10_000)
    args = ap.parse_args()
    exe = HERE / "native" / "tl_loop"
    subprocess.run(["gcc", "-O3", "-o", str(exe), str(HERE / "native" / "tl_loop.c"), "-ldl"], check=True)
    rows = []
    for spec in args.settings.split(","):
        ds, fam, d = spec.split(":")
        d = int(d)
        d0 = Path(args.data) / ds
        meta = json.loads((d0 / "meta.json").read_text())
        df = pd.read_parquet(d0 / "data.parquet")
        df.columns = [str(c) for c in df.columns]
        edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
        outcome = str(meta["target"])
        scm = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]], zoo(1000, 1)[fam],
                      max_rows=50_000, seed=0, n_jobs=4)
        ps = scm.parents[outcome]
        T = [t for t in choose_treatments(scm, outcome, d) or [] if t in ps][:d]
        if len(T) < d:
            print(f"{spec}: fewer than {d} treatments are parents of the outcome; skipped", flush=True)
            continue
        rng = np.random.default_rng(2028)
        unit = df.iloc[int(rng.integers(len(df)))].to_dict()
        bound = star_bind(scm, unit, T, outcome)
        nb = accel.NumbaIndex(ProvenanceIndex(bound))
        Q = np.ascontiguousarray(rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d)))
        base_row = np.array([unit[p] for p in ps], dtype=np.float64)
        full = np.repeat(base_row[None, :], len(Q), axis=0)
        full[:, [ps.index(t) for t in T]] = Q
        full = scm.ensembles[outcome].cast(full)
        noise = scm.abduct(unit, [outcome])[outcome]
        with tempfile.TemporaryDirectory() as wd:
            tl2cgen, predictor, dtype, _ = treelite_predictor(scm.models[outcome], wd)
            eb = 4 if dtype == "float32" else 8
            # (a) DMatrix + Predictor
            npdt = np.float32 if eb == 4 else np.float64
            a_us = per_call_seconds(lambda: [predictor.predict(tl2cgen.DMatrix(full[i:i + 1].astype(npdt), dtype=dtype))
                                             for i in range(len(Q))], len(Q)) * 1e6
            # (b) ctypes, one call per query
            lib = ctypes.CDLL(str(Path(wd) / "model.so"))
            fn = lib.predict
            fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            fn.restype = None
            entries = np.ascontiguousarray(full.astype(npdt))
            out = np.zeros(1)
            ptr, optr, stride = entries.ctypes.data, out.ctypes.data, entries.strides[0]

            def call(i):
                out[0] = 0.0
                fn(ptr + i * stride, 0, optr)
                return out[0]

            got_b = np.array([call(i) for i in range(len(Q))])
            b_us = per_call_seconds(lambda: [call(i) for i in range(len(Q))], len(Q)) * 1e6
            # (c) native C loop
            rows_bin, out_bin = Path(wd) / "rows.bin", Path(wd) / "out.bin"
            full.astype(np.float64).tofile(rows_bin)
            ns = float(subprocess.run([str(exe), str(Path(wd) / "model.so"), str(full.shape[1]), str(len(Q)), str(eb),
                                       str(rows_bin), str(out_bin)], check=True, capture_output=True, text=True).stdout)
            got_c = np.fromfile(out_bin)
        ref = np.array([nb.query(t) for t in Q])
        tl_err = float(np.max(np.abs(got_c + noise - ref) / np.maximum(1.0, np.abs(ref))))
        d_us = per_call_seconds(lambda: [nb.query(t) for t in Q], len(Q)) * 1e6
        nb.batch(Q)
        start = time.perf_counter()
        for _ in range(3):
            nb.batch(Q)
        e_us = (time.perf_counter() - start) / 3 / len(Q) * 1e6
        row = dict(dataset=ds, family=fam, d=d, trees=len(scm.ensembles[outcome].trees), tl_dmatrix_us=a_us,
                   tl_ctypes_us=b_us, tl_native_us=ns / 1e3, ours_call_us=d_us, ours_native_us=e_us,
                   tl_rel_dev=tl_err, ctypes_equals_native=bool(np.allclose(got_b, got_c)))
        rows.append(row)
        write_csv("tl_fair.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
