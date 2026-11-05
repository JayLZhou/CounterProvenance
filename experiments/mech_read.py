#!/usr/bin/env python3
"""The read without the dense value table: grid cell -> cell -> value, against the table of one value per grid cell, on the
queries of the main experiment; and how far the map from grid cells to cells compresses by runs along one axis.
Output: ``results/read/<dataset>_<family>_time.csv`` and ``_map.csv``."""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import bind_scm, relevant_nodes
from mech_main import queries
from mech_proto import MechanismIndex


class TwoStep:
    def __init__(self, face_of, value):
        self.face_of, self.value = face_of, value

    def __getitem__(self, c):
        return self.value[self.face_of[c]]


def build(scm, unit0, nodes, budget):
    mech = MechanismIndex.__new__(MechanismIndex)
    mech.scm, mech.indexes, mech.info = scm, {}, {}
    for x in nodes:
        bound = bind_scm(scm, unit0, list(scm.parents[x]), x, threads=1, native=True)
        mech.indexes[x] = ProvenanceIndex(bound, grouping="tables", budget=int(budget), build_threads=1, np_read_groups=0)
    return mech


def two_step(index):
    faces, values, shift = [], [], 0
    for leaf in index._fast_leaves:
        c = leaf.cells
        faces.append(np.asarray(c.overlay.face_of_cell, dtype=np.int64) + shift)
        values.append(np.asarray(c.face_value, dtype=np.float64))
        shift += len(values[-1])
    two = TwoStep(np.concatenate(faces), np.concatenate(values))
    pick = np.random.default_rng(0).integers(len(index._ntable), size=min(100_000, len(index._ntable)))
    if not np.array_equal(index._ntable[pick], two[pick]):
        raise AssertionError("the two tables differ")
    return two


def map_rows(args, node, index):
    rows = []
    for leaf in index._fast_leaves:
        o = leaf.cells.overlay
        a = np.asarray(o.face_of_cell).reshape(o.shape)
        best = None
        for k in range(a.ndim):
            changes = int(np.count_nonzero(np.diff(a, axis=k))) if a.shape[k] > 1 else 0
            lines = a.size // a.shape[k]
            cost = 8 * (changes + lines) + 4 * lines  # a run: its first position and its cell; a line: where its runs start
            if best is None or cost < best[0]:
                best = (cost, k, changes + lines, lines)
        rows.append(dict(dataset=args.dataset, family=args.family, node=node, grid=a.size, cells=o.n_faces, trees=leaf.cells.stats.slots, map_bytes=4 * a.size, run_bytes=best[0], runs=best[2], lines=best[3],
                         value_table_bytes=8 * a.size))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--units", type=int, default=200)
    ap.add_argument("--budget", type=float, default=2e7)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, y, rows, work = state["scm"], state["y"], state["rows"], state["work"]
    out = RESULTS / "read"
    out.mkdir(parents=True, exist_ok=True)
    d = 4 if 4 in work else max(work)
    T = work[d]["T"]
    order = relevant_nodes(scm.parents, T, y)
    mech = build(scm, rows.iloc[0].to_dict(), order, args.budget)
    maps = [r for x in order for r in map_rows(args, x, mech.indexes[x])]
    pd.DataFrame(maps).to_csv(out / f"{args.dataset}_{args.family}_map.csv", index=False)
    plan = mech.plan(T, order)
    tables = {x: (mech.indexes[x]._ntable, two_step(mech.indexes[x])) for x in order}
    lat = {"full_one": [], "full_two": [], "partial_one": [], "partial_two": []}
    wrong = 0

    def use(i):
        for x in order:
            mech.indexes[x]._ntable = tables[x][i]

    for u, Q in enumerate(queries(state, d, 10)):
        if u >= args.units:
            break
        unit = rows.iloc[u].to_dict()
        use(0)
        observed = mech.observed(unit, order)
        bound = mech.bind(plan, unit, observed)
        for k, t in enumerate(Q):
            # alternate the order, so that neither read always comes first after the other has filled the cache
            got = {}
            for i in (0, 1) if k % 2 == 0 else (1, 0):
                use(i)
                name = "one" if i == 0 else "two"
                t0 = time.perf_counter()
                a = mech.query(unit, T, t, order, observed)
                lat["full_" + name].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                b = mech.query_partial(unit, T, t, bound)
                lat["partial_" + name].append(time.perf_counter() - t0)
                got[i] = (a, b)
            wrong += abs(got[0][0] - got[1][0]) > 1e-9 or abs(got[0][1] - got[1][1]) > 1e-9
    r = dict(dataset=args.dataset, family=args.family, d=d, path=len(order), queries=len(lat["full_one"]), groups=sum(len(mech.indexes[x]._fast_leaves) for x in order), wrong=wrong,
             **{k + "_us": float(np.mean(v)) * 1e6 for k, v in lat.items()})
    pd.DataFrame([r]).to_csv(out / f"{args.dataset}_{args.family}_time.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
