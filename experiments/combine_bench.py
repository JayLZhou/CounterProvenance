#!/usr/bin/env python3
"""Other combinations of the trees than their sum, chosen at query time (Section 6.4 of the paper).

The outcome's tree ensemble is combined by a function Phi of the selected leaf values, and the noise is abducted for it:
``Y = u[Y] + Phi(v(x)) - Phi(v(u))``; the mediators keep their own sums.  CellForest reads the selected leaves of the
outcome from the persistent trees of its cells (no tree is traversed); ThreeStep traverses every tree, as in the main
experiment.  The combinations:

* ``sum``: the model as it is (a check: the value of the main experiment);
* ``prefix``: the first M/2 trees (early stopping, XGBoost's ``iteration_range``, LightGBM's ``num_iteration``);
* ``weights``: one weight per tree, uniform in [0, 2], drawn per unit (a reweighted or pruned ensemble);
* ``median``: the median of the trees' values (a robust ensemble).

Same cached SCMs, units, and query points as the main experiment (mech_main.py), one thread.
Output: ``results/combine/<dataset>_<family>_d<d>.csv``."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.native_cells import NativeCellIndex
from counterprovenance.scm import relevant_nodes
from counterprovenance.scm_baselines import ScmThreeStepPy
from mech_main import prepare, queries
from mech_proto import MechanismIndex, index_dir


class LeafReader:
    """The selected leaf value of every tree of one ensemble, read from the persistent trees of its cells, all groups
    together: their node arrays are concatenated once, and a read walks the roots of all groups level by level, a few
    NumPy steps per level (no tree of the ensemble is traversed)."""

    def __init__(self, index, M: int) -> None:
        self.ix, self.M = index, M
        boxes = index.bound.boxes
        if index._baxes:
            raise ValueError("leaf masks are not read here")
        L, R, V, Tr, H, no, io, foc, ver = [], [], [], [], [], [], [], [], []
        n_nodes = n_items = 0
        for leaf in index._fast_leaves:
            c = leaf.cells
            if not isinstance(c, NativeCellIndex):
                raise ValueError("needs the persistent trees of NativeCellIndex")
            pft = c.pft
            L.append(np.asarray(pft.left, dtype=np.int64)); R.append(np.asarray(pft.right, dtype=np.int64))
            V.append(np.asarray(c.values, dtype=np.float64)); Tr.append(np.asarray(c.item_trees, dtype=np.int64))
            H.append(pft.height); no.append(n_nodes); io.append(n_items)
            n_nodes += len(pft.left); n_items += len(c.values)
            foc.append(c.overlay.face_of_cell); ver.append(c.face_version)
        self.L = np.concatenate(L) if L else np.zeros(0, np.int64)
        self.R = np.concatenate(R) if R else np.zeros(0, np.int64)
        self.V = np.concatenate(V) if V else np.zeros(0)
        self.Tr = np.concatenate(Tr) if Tr else np.zeros(0, np.int64)
        self.H, self.no, self.io = np.array(H, np.int64), np.array(no, np.int64), np.array(io, np.int64)
        self.Hmax = int(self.H.max()) if len(H) else 0
        # the root of a cell, all groups at once: the face maps and the face roots concatenated
        self.cell_off = np.concatenate(([0], np.cumsum([len(f) for f in foc])[:-1])).astype(np.int64)
        face_off = np.concatenate(([0], np.cumsum([len(v) for v in ver])[:-1])).astype(np.int64)
        self.FOC = np.concatenate([np.asarray(f, dtype=np.int64) for f in foc]) + np.repeat(face_off, [len(f) for f in foc]) if foc else np.zeros(0, np.int64)
        self.VER = np.concatenate([np.asarray(v, dtype=np.int64) for v in ver]) if ver else np.zeros(0, np.int64)
        const = [i for leaf in index._const_leaves for i in leaf.cells.items]
        self.const_trees = np.array([boxes[i].tree for i in const], dtype=np.int64)
        self.const_values = np.array([boxes[i].value for i in const], dtype=np.float64)

    def __call__(self, z) -> np.ndarray:
        ans = self.ix.query(z)
        v = np.empty(self.M)
        v[self.const_trees] = self.const_values
        roots = self.VER[self.FOC[np.asarray(ans._cells, dtype=np.int64) + self.cell_off]]
        codes, gid = roots, np.arange(len(roots))
        done_c, done_g = [], []
        for level in range(self.Hmax + 1):
            fin = self.H[gid] == level            # these groups reached their leaves: the codes are items
            if fin.any():
                done_c.append(codes[fin]); done_g.append(gid[fin])
            keep = ~fin
            if not keep.any() or level == self.Hmax:
                break
            c, g = codes[keep] + self.no[gid[keep]], gid[keep]
            codes, gid = np.concatenate((self.L[c], self.R[c])), np.concatenate((g, g))
        if done_c:
            items = -np.concatenate(done_c) - 2
            g = np.concatenate(done_g)
            ok = items >= 0
            gi = items[ok] + self.io[g[ok]]
            v[self.Tr[gi]] = self.V[gi]
        return v


class ThreeStepPhi(ScmThreeStepPy):
    """ThreeStep with the outcome combined by ``phi``: every query traverses every tree, twice (no caching)."""

    def query_phi(self, t, phi) -> float:
        unit = self.unit
        observed = [self._leaves(cast([unit[p] for p in parents]), trees) for _, parents, cast, trees in self.mechs]
        values = dict(unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        for (node, parents, cast, trees), obs in zip(self.mechs, observed):
            leaves = self._leaves(cast([values[p] for p in parents]), trees)
            if node == self.outcome:
                vx = np.array([tr[4][n] for tr, n in zip(trees, leaves)])
                vu = np.array([tr[4][o] for tr, o in zip(trees, obs)])
                values[node] = unit[node] + phi(vx) - phi(vu)
            else:
                values[node] = unit[node] + math.fsum(tr[4][n] - tr[4][o] for tr, n, o in zip(trees, leaves, obs))
        return values[self.outcome]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--per-unit", type=int, default=10)
    ap.add_argument("--base-units", type=int, default=100, help="units on which ThreeStep runs")
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--run-units", type=int, default=1000, help="units on which CellForest runs (all by default)")
    ap.add_argument("--out", default="combine")
    args = ap.parse_args()
    state = prepare(args)
    if args.d not in state["work"]:
        return
    scm, y, rows = state["scm"], state["y"], state["rows"]
    T = state["work"][args.d]["T"]
    order = relevant_nodes(scm.parents, T, y)
    t0 = time.perf_counter()
    mech = MechanismIndex(scm, rows.iloc[0].to_dict(), order, args.budget, args.build_threads,
                          cache=index_dir(RESULTS, args.dataset, args.family, args.trees, args.budget))
    build_s = time.perf_counter() - t0
    M = len(scm.ensembles[y].trees)
    reader = LeafReader(mech.indexes[y], M)
    pa_y = scm.parents[y]
    combos = {"sum": lambda v, w: float(v.sum()), "prefix": lambda v, w: float(v[: M // 2].sum()),
              "weights": lambda v, w: float(v @ w), "median": lambda v, w: float(np.median(v))}
    res = {k: dict(cf_s=0.0, cf_n=0, base_s=0.0, base_n=0, err=0.0) for k in combos}
    for u, Q in enumerate(queries(state, args.d, args.per_unit)):
        if u >= args.run_units:
            break
        unit = rows.iloc[u].to_dict()
        w = np.random.default_rng([2028, u]).uniform(0, 2, M)
        three = ThreeStepPhi(scm, unit, T, y) if u < args.base_units else None
        for k, phi in combos.items():
            t0 = time.perf_counter()  # per unit: the observed mediators and the factual leaves of the outcome
            observed = {x: mech.indexes[x].query([unit[p] for p in scm.parents[x]]).value for x in order[:-1]}
            vu = phi(reader([unit[p] for p in pa_y]), w)
            got = []
            for t in Q:
                values = dict(unit)
                for j, name in enumerate(T):
                    values[name] = float(t[j])
                for x in order[:-1]:
                    values[x] = unit[x] + (mech.indexes[x].query([values[p] for p in scm.parents[x]]).value - observed[x])
                got.append(unit[y] + phi(reader([values[p] for p in pa_y]), w) - vu)
            res[k]["cf_s"] += time.perf_counter() - t0
            res[k]["cf_n"] += len(Q)
            if three is not None:
                t0 = time.perf_counter()
                ref = [three.query_phi(t, lambda v, w=w, phi=phi: phi(v, w)) for t in Q]
                res[k]["base_s"] += time.perf_counter() - t0
                res[k]["base_n"] += len(Q)
                res[k]["err"] = max(res[k]["err"], float(np.max(np.abs(np.array(got) - np.array(ref)))))
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([dict(dataset=args.dataset, family=args.family, d=args.d, combo=k, trees=M, path=len(order),
                            groups=len(mech.indexes[y].groups), build_s=build_s, cf_us=r["cf_s"] / r["cf_n"] * 1e6,
                            base_us=r["base_s"] / max(r["base_n"], 1) * 1e6, queries=r["cf_n"], base_queries=r["base_n"],
                            max_abs_err=r["err"]) for k, r in res.items()])
    df.to_csv(out / f"{args.dataset}_{args.family}_d{args.d}.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
