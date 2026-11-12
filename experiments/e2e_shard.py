#!/usr/bin/env python3
"""One shard of an application over a whole table, on one backend, in one thread: every row of the shard really runs.

The rows are in the same random order on all backends, the time of a table is the sum over its shards, and the answer of
every row is kept, so that the backends can be compared row by row.  A shard restarts from its last checkpoint.
Output: ``results/shard/<dataset>_<family>_<app>_<backend>_<start>.csv`` and ``.npy``."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.scm import ancestors, fit_scm, parents_of
from e2e_apps import APPS, SUBSET_APPS, Ctx, DoWhy, GeCo, Online, Oracle, SubsetOracle, ThreeStep, Twin, TwinMech, model_factory
from scm_full import dowhy_model


# the sensitive attributes of counterfactual fairness (binary): sex and birth country on Adult, gender on German (MINT's SCM)
SENSITIVE = {"adult": ["is_male", "native_us"], "german_mint": ["gender"]}


def at(base: Ctx, row: pd.DataFrame) -> Ctx:
    c = copy.copy(base)
    c.row = row
    c.u = np.clip(row[c.T].to_numpy(dtype=float)[0], c.lo, c.hi)
    return c


class Shifted:
    """The index of one unit for another unit: the same trees, the noise of the other unit."""

    def __init__(self, twin, shift):
        self.twin, self.shift = twin, shift

    def __call__(self, Q):
        return self.twin(Q) + self.shift


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--app", required=True)
    ap.add_argument("--backend", required=True, choices=["twin", "geco", "native", "three_step", "dowhy", "treeshap"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--actionable", action="store_true", help="intervene on the actionable ancestors of the outcome (MINT), with mediators")
    ap.add_argument("--sensitive", action="store_true", help="intervene on the sensitive attributes of the table (counterfactual fairness), with mediators")
    args = ap.parse_args()
    out_dir = RESULTS / "shard"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = out_dir / f"{args.dataset}_{args.family}_{args.app}_{args.backend}_{args.start}"
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], model_factory(args.family),
                  max_rows=50_000, seed=0, n_jobs=1)
    T = list(scm.parents[y])
    if args.actionable or args.sensitive:
        anc = ancestors(parents_of(edges), y)
        T = [v for v in (meta["actionable"] if args.actionable else SENSITIVE[args.dataset]) if v in anc]
    order = np.random.default_rng(2028).permutation(len(df))
    stop = len(df) if args.stop < 0 else min(args.stop, len(df))
    first = df.iloc[[int(order[0])]]
    base = Ctx(df, T, first, y, 2028, meta.get("increase_only", ()), meta.get("mcce_immutable", ()))
    app = APPS[args.app]
    done, q_s, run_s, n = 0, 0.0, 0.0, 0
    outs: list[float] = []
    state = tag.with_suffix(".csv")
    if state.exists():
        s = pd.read_csv(state).iloc[0]
        done, q_s, run_s, n = int(s.rows_done), float(s.query_s), float(s.run_s), int(s.queries)
        outs = np.load(tag.with_suffix(".npy")).tolist()[:done]
        if done >= stop - args.start:
            return
    build_s = 0.0
    if args.backend == "twin" and (args.actionable or args.sensitive):
        twinm = TwinMech(scm, first.iloc[0].to_dict(), T, y, 2e7, args.build_threads, max_size=3 if args.app in SUBSET_APPS else None)
        build_s = twinm.build_s
    elif args.backend == "twin":
        t0 = time.perf_counter()
        twin = Twin(scm, first.iloc[0].to_dict(), T, y, 2e7, args.build_threads)
        build_s = time.perf_counter() - t0
        f0 = float(twin(first[T].to_numpy(dtype=float))[0])
    elif args.backend == "dowhy":
        gcm, cg = dowhy_model(scm, edges)
    elif args.backend == "treeshap":
        # the same quantity by the algorithm for trees: interventional Shapley values against the same background
        import shap
        t0 = time.perf_counter()
        # the library samples 100 rows of a background unless it is told otherwise
        masker = shap.maskers.Independent(base.bg, max_samples=len(base.bg))
        explainer = shap.TreeExplainer(scm.models[y], data=masker, feature_perturbation="interventional")
        build_s = time.perf_counter() - t0
    base_row = dict(dataset=args.dataset, family=args.family, app=args.app, backend=args.backend, d=len(T), table_rows=len(df),
                    start=args.start, stop=stop, build_s=build_s)

    def save():
        pd.DataFrame([dict(base_row, rows_done=done, queries=n, query_s=q_s, run_s=run_s)]).to_csv(state, index=False)
        np.save(tag.with_suffix(".npy"), np.array(outs))

    for i in order[args.start + done:stop]:
        row = df.iloc[[int(i)]]
        unit = row.iloc[0].to_dict()
        if args.backend == "treeshap":
            c = at(base, row)
            t0 = time.perf_counter()
            phi = explainer.shap_values(c.u[None, :], check_additivity=False)
            spent = time.perf_counter() - t0
            outs.append(float(np.abs(phi).sum()))
            q_s += spent
            run_s += spent
            done += 1
            if done % 20 == 0:
                save()
            continue
        if args.backend == "twin" and (args.actionable or args.sensitive):
            backend = twinm.unit(unit)
        elif args.backend == "twin":
            shift = float(row[y].iloc[0]) - float(first[y].iloc[0]) - (float(twin(row[T].to_numpy(dtype=float))[0]) - f0)
            backend = Shifted(twin, shift)
        elif args.backend == "dowhy":
            backend = DoWhy(gcm, cg, row, T, y)
        elif args.backend == "native":
            # the application's own evaluation (e.g. MCCE): the library's predict over the whole call, the noise of the unit once
            backend = Online(scm, unit, T, y)
        elif args.backend == "geco":
            backend = GeCo(scm, unit, T, y)
        else:
            backend = ThreeStep(scm, unit, T, y)
        if args.app in SUBSET_APPS:
            if args.backend == "twin":
                make = lambda S, b=backend: b if S is None else twinm.subset(b, S)
            elif args.backend == "dowhy":
                make = lambda S, b=backend, r=row: b if S is None else DoWhy(gcm, cg, r, [T[i] for i in S], y)
            else:
                make = lambda S, b=backend, u=unit: b if S is None else ThreeStep(scm, u, [T[i] for i in S], y)
            f = SubsetOracle(make)
        else:
            f = Oracle(backend)
        t0 = time.perf_counter()
        outs.append(app(f, at(base, row)))
        run_s += time.perf_counter() - t0
        q_s += f.query_s
        n += f.queries
        done += 1
        if done % 20 == 0:
            save()
    save()


if __name__ == "__main__":
    main()
