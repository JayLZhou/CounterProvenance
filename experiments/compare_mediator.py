#!/usr/bin/env python3
"""Mediator compilation: dense grids (``scm_grid``) vs the dominance test (``counterprovenance.scm``).

For one dataset, every family and every ``d`` given as ``family:d``: fit the SCM
once, then on the same unit compile the outcome's trees both ways, build the
index, check 400 queries of each against ``ScmOnlinePy``, and time the numba
index of each against ``ScmNumba``.  Output: ``results/compare_mediator_<dataset>.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import per_call_seconds, write_csv  # puts the package on sys.path
import scm_grid
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.scm import Infeasible, ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmNumba, ScmOnlinePy
from scm_bench import choose_treatments, zoo


def build(fn, Q, truth, row, tag):
    start = time.perf_counter()
    try:
        bound = fn()
        row[f"{tag}_compile_s"] = time.perf_counter() - start
        index = ProvenanceIndex(bound)
    except (scm_grid.Infeasible, Infeasible, MemoryError) as e:
        row[f"{tag}_status"] = f"infeasible: {e}"
        return None
    row[f"{tag}_build_s"] = time.perf_counter() - start
    row[f"{tag}_boxes"] = len(bound.boxes)
    row[f"{tag}_groups"] = len(index.groups)
    nb = accel.NumbaIndex(index)
    nb.query(Q[0])
    got = np.array([nb.query(t) for t in Q])
    row[f"{tag}_err"] = float(np.max(np.abs(got - truth) / np.maximum(1.0, np.abs(truth))))
    if row[f"{tag}_err"] > 1e-9:
        raise AssertionError(f"{tag} differs by {row[f'{tag}_err']}")
    row[f"{tag}_nb_us"] = per_call_seconds(lambda: [nb.query(t) for t in Q], len(Q)) * 1e6
    row[f"{tag}_status"] = "ok"
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated family:d")
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    outcome = str(meta["target"])
    rows, fitted = [], {}
    for spec in args.settings.split(","):
        fam, d = spec.split(":")
        d = int(d)
        if fam not in fitted:
            fitted[fam] = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]],
                                  zoo(1000, 1)[fam], max_rows=50_000, seed=0, n_jobs=4)
        scm = fitted[fam]
        rng = np.random.default_rng(2028)
        unit = df.iloc[int(rng.integers(len(df)))].to_dict()
        T = choose_treatments(scm, outcome, d)
        py = ScmOnlinePy(scm, unit, T, outcome)
        Q = rng.uniform(df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy(), size=(400, d))
        truth = np.array([py.query(t)[0] for t in Q])
        row = dict(dataset=args.dataset, family=fam, d=d, mediators=len(py.mechs) - 1)
        onl = ScmNumba(scm, unit, T, outcome)
        onl.query(Q[0])
        row["online_nb_us"] = per_call_seconds(lambda: [onl.query(t) for t in Q], len(Q)) * 1e6
        build(lambda: scm_grid.bind_scm(scm, unit, T, outcome), Q, truth, row, "grid")
        build(lambda: bind_scm(scm, unit, T, outcome), Q, truth, row, "dom")
        rows.append(row)
        write_csv(f"compare_mediator_{args.dataset}.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
