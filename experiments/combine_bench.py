#!/usr/bin/env python3
"""Other combinations of the trees than their sum, chosen at query time (Section 6.4 of the paper).

The outcome's tree ensemble is combined by a function Phi of the selected leaf values, and the noise is abducted for it:
``Y = u[Y] + Phi(v(x)) - Phi(v(u))``; the mediators keep their own sums.  ThreeStep traverses every tree with the same
Phi, one query at a time, as in the main experiment.  CellForest:

* ``prefix`` (the first M/2 trees: early stopping, XGBoost's ``iteration_range``, LightGBM's ``num_iteration``) and
  ``weights`` (one weight per tree, uniform in [0, 2], one vector per setting: a reweighted or pruned ensemble) are linear:
  the value of every cell for the weights is computed once per setting from the persistent trees (the weighted sums of
  their nodes), and a query then reads one value per group, one query at a time, as the stored sum (``prep_s``);
* ``median`` and ``trimmed`` (the mean without the 10% largest and 10% smallest trees: robust ensembles) need every
  selected leaf: CellForest reads them from the persistent trees of all groups, one query at a time, with the walk
  compiled by numba (the only compiled step of a query);
* ``sum``: the stored value of the main experiment (a check).

Same cached SCMs, units, query points, and saved indexes as the main experiment (mech_main.py), one thread.
Output: ``results/combine/<dataset>_<family>_d<d>.csv``."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import copy
import math
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.index import _Table
from counterprovenance.native_cells import LogCellIndex, NativeCellIndex
from counterprovenance.scm import relevant_nodes
from counterprovenance.scm_baselines import ScmThreeStepPy
from mech_main import prepare, queries
from mech_proto import MechanismIndex, index_dir
from numba import njit


@njit(cache=True)
def _walk(roots, L, R, H, no, io, V, Tr, out):
    """The leaves of the versions ``roots``, one per group: a depth-first walk of each persistent tree."""
    stack = np.empty(256, np.int64)
    depth = np.empty(256, np.int64)
    for g in range(roots.shape[0]):
        stack[0] = roots[g]
        depth[0] = 0
        sp = 1
        while sp > 0:
            sp -= 1
            c = stack[sp]
            d = depth[sp]
            if d == H[g]:
                item = -c - 2
                if item >= 0:
                    out[Tr[item + io[g]]] = V[item + io[g]]
            else:
                n = c + no[g]
                stack[sp] = R[n]
                depth[sp] = d + 1
                stack[sp + 1] = L[n]
                depth[sp + 1] = d + 1
                sp += 2


@njit(cache=True)
def _median(v):
    return np.median(v)


@njit(cache=True)
def _trimmed(v):
    s = np.sort(v)
    k = s.shape[0] // 10
    return s[k:s.shape[0] - k].mean()


def _check(index):
    if index._baxes:
        raise ValueError("leaf masks are not read here")
    for leaf in index._fast_leaves:
        if not isinstance(leaf.cells, (NativeCellIndex, LogCellIndex)):
            raise ValueError("needs the leaves of NativeCellIndex or LogCellIndex")


def _is_log(index):
    return any(isinstance(leaf.cells, LogCellIndex) for leaf in index._fast_leaves)


@njit(cache=True)
def _log_sums(slot, item, wv, out):
    """The weighted sum of the leaves after every prefix of a log (``out[p]``: after ``p`` changes), compensated."""
    state = np.full(int(slot.max()) + 1 if slot.shape[0] else 1, -1, np.int64)
    s = 0.0; c = 0.0
    out[0] = 0.0
    for p in range(slot.shape[0]):
        j = slot[p]
        new = np.int64(item[p]) - 1
        old = state[j]
        x = (wv[new] if new >= 0 else 0.0) - (wv[old] if old >= 0 else 0.0)
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
        state[j] = new
        out[p + 1] = s + c


def _face_values_log(c, w):
    """The value of every cell of a LogCellIndex for the weights ``w`` of the trees."""
    lg = c.log
    wv = np.asarray(c.values, dtype=np.float64) * w[np.asarray(c.item_trees)]
    ver = np.asarray(c.face_version, dtype=np.int64)
    if lg.direct:
        rows = lg.snap.astype(np.int64).reshape(-1, lg.n_slots) - 1
        return np.where(rows >= 0, wv[np.maximum(rows, 0)], 0.0).sum(axis=1)[ver] + c.offset
    sums = np.empty(lg.changes + 1)
    _log_sums(lg.slot.astype(np.int64), lg.item.astype(np.int64), wv, sums)
    return sums[ver] + c.offset


@njit(cache=True)
def _replay(groups, versions, SLOT, ITEM, SNAP, so, no, G, SP, D, TR, V, vo, out):
    """The leaves of the versions ``versions`` of the log groups ``groups``: the last checkpoint, then the changes."""
    st = np.empty(int(G.max()) if G.shape[0] else 1, np.int64)
    for k in range(groups.shape[0]):
        g = groups[k]; v = versions[k]; n = G[g]
        if D[g]:
            base = no[g] + v * n
            for j in range(n):
                st[j] = SNAP[base + j]
        else:
            kk = v // SP[g]
            base = no[g] + kk * n
            for j in range(n):
                st[j] = SNAP[base + j]
            for p in range(kk * SP[g], v):
                st[SLOT[so[g] + p]] = ITEM[so[g] + p]
        for j in range(n):
            it = st[j] - 1
            if it >= 0:
                out[TR[vo[g] + it]] = V[vo[g] + it]


class LogLeafReader:
    """The selected leaf value of every tree, from the logs of the groups of several trees and from the two-level grids of
    the groups of one tree (no tree of the ensemble is traversed)."""

    def __init__(self, index, M: int) -> None:
        _check(index)
        self.ix, self.M = index, M
        boxes = index.bound.boxes
        two = index._two_level()
        self.tl = two.tl if two else None
        n = len(index._fast_leaves)
        self.multi = np.asarray(self.tl.multi if self.tl else np.arange(n), dtype=np.int64)
        SL, IT, SN, so, no, G, SP, D, TR, V, vo, foc, ver = [], [], [], [], [], [], [], [], [], [], [], [], []
        a = b = e = 0
        for g in self.multi:
            c = index._fast_leaves[g].cells; lg = c.log
            SL.append(lg.slot.astype(np.int64)); IT.append(lg.item.astype(np.int64)); SN.append(lg.snap.astype(np.int64))
            so.append(a); no.append(b); vo.append(e); a += lg.changes; b += len(lg.snap); e += len(c.values)
            G.append(lg.n_slots); SP.append(lg.spacing); D.append(lg.direct)
            TR.append(np.asarray(c.item_trees, dtype=np.int64)); V.append(np.asarray(c.values, dtype=np.float64))
            foc.append(np.asarray(c.overlay.face_of_cell)); ver.append(np.asarray(c.face_version, dtype=np.int64))
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(1, dt)
        self.SLOT, self.ITEM, self.SNAP = cat(SL, np.int64), cat(IT, np.int64), cat(SN, np.int64)
        self.so, self.no, self.vo = (np.array(x, dtype=np.int64) for x in (so, no, vo))
        self.G, self.SP, self.D = np.array(G, dtype=np.int64), np.array(SP, dtype=np.int64), np.array(D, dtype=np.bool_)
        self.TR, self.V = cat(TR, np.int64), cat(V, np.float64)
        self.foc, self.ver = foc, ver
        self.local = np.arange(len(self.multi), dtype=np.int64)
        if self.tl is not None:   # the tree of every leaf of the two-level grids
            pos = {id(r): i for i, r in enumerate(index.roots) if r is not None}
            self.single_tree = np.array([boxes[index.groups[pos[id(index._fast_leaves[g])]][1][0]].tree for g in self.tl.single], dtype=np.int64)
            self.view = self.tl.view(np.arange(len(self.tl.single)), range(self.tl.d))
        const = [i for leaf in index._const_leaves for i in leaf.cells.items]
        self.const_trees = np.array([boxes[i].tree for i in const], dtype=np.int64)
        self.const_values = np.array([boxes[i].value for i in const], dtype=np.float64)

    def one(self, z) -> np.ndarray:
        ix = self.ix
        x = ix._cast(z)
        out = np.empty(self.M)
        out[self.const_trees] = self.const_values
        if len(self.multi):
            # the cell of every group: the rank rows of a vector index, otherwise the cells of the query
            c = (ix._cell_vector(x) if ix._nbase is not None else np.asarray(ix.query(z).cells, dtype=np.int64))[self.multi]
            versions = np.array([self.ver[k][self.foc[k][c[k]]] for k in range(len(c))], dtype=np.int64)
            _replay(self.local, versions, self.SLOT, self.ITEM, self.SNAP, self.so, self.no, self.G, self.SP, self.D, self.TR, self.V, self.vo, out)
        if self.tl is not None:
            tl, v = self.tl, self.view
            rows = [tl.rowbase[a] + bisect_left(tl.th[a], x[a]) for a in range(tl.d)]
            t = v["TR"].take(rows, axis=0)
            c1 = tl.base1 + tl.C1.take(v["c1off"] + t).sum(axis=0)
            sub = v["subfirst"] + tl.E1.take(c1)
            c2 = tl.base2.take(sub) + tl.R2.take(v["r2off"].take(sub, axis=1) + t).sum(axis=0)
            out[self.single_tree] = tl.VAL.take(v["leafbase"] + tl.L2.take(c2))
        return out


def reweighted(index, w: np.ndarray):
    """The same index, whose cells hold the weighted sum of their leaves: the weighted sums of the nodes of every
    persistent tree in ``height`` vectorized passes, the value of a cell at the root of its version.  A query then
    reads it as the stored sum, one value per group."""
    _check(index)
    out = copy.copy(index)
    boxes = index.bound.boxes
    face_values = []
    for leaf in index._fast_leaves:
        c = leaf.cells
        if isinstance(c, LogCellIndex):
            face_values.append(_face_values_log(c, w))
            continue
        pft = c.pft
        vals = np.r_[np.asarray(c.values, dtype=np.float64) * w[np.asarray(c.item_trees)], 0.0]  # item -1 reads 0
        left, right = np.asarray(pft.left, dtype=np.int64), np.asarray(pft.right, dtype=np.int64)

        def total(child, sums):
            return np.where(child >= 0, sums[np.maximum(child, 0)], vals[np.where(child >= 0, -1, -child - 2)])
        sums = np.zeros(len(left))
        for _ in range(max(1, pft.height)):
            sums = total(left, sums) + total(right, sums)
        face_values.append(total(np.asarray(c.face_version, dtype=np.int64), sums))
    const = [i for leaf in index._const_leaves for i in leaf.cells.items]
    out._const = math.fsum([index.bound.offset] + [boxes[i].value * w[boxes[i].tree] for i in const])
    if index._nbase is not None:  # many groups: the tables in one array
        if index._ndense is not None:
            out._ndense = np.concatenate([fv[np.asarray(leaf.cells.overlay.face_of_cell)] for fv, leaf in zip(face_values, index._fast_leaves)])
        else:
            out._nvalue = np.concatenate(face_values)
    else:
        tables = []
        for fv, leaf, t in zip(face_values, index._fast_leaves, index._tables):
            m = leaf.cells.overlay.face_of_cell
            tables.append(memoryview(np.ascontiguousarray(fv[np.asarray(m)])) if isinstance(t, memoryview) else _Table(m, fv))
        out._tables = tables
    two = index._two_level()
    if two:   # the leaves of the two-level grids, weighted by their trees
        from counterprovenance.twolevel import Hybrid
        tl = copy.copy(two.tl)
        pos = {id(r): i for i, r in enumerate(index.roots) if r is not None}
        tree_of_leaf = np.concatenate([[boxes[i].tree for i in index.groups[pos[id(index._fast_leaves[g])]][1]] for g in tl.single])
        tl.VAL = two.tl.VAL * w[tree_of_leaf]
        out._two = Hybrid(out, tl, range(index.bound.d))
    else:
        out._two = False
    return out


class LeafReader:
    """The selected leaf value of every tree of one ensemble, for several inputs together: the persistent trees of all
    groups and all inputs are walked level by level, a few NumPy steps per level (no tree of the ensemble is traversed)."""

    def __init__(self, index, M: int) -> None:
        _check(index)
        self.ix, self.M = index, M
        boxes = index.bound.boxes
        L, R, V, Tr, H, no, io, foc, ver = [], [], [], [], [], [], [], [], []
        n_nodes = n_items = 0
        for leaf in index._fast_leaves:
            c, pft = leaf.cells, leaf.cells.pft
            L.append(np.asarray(pft.left, dtype=np.int64)); R.append(np.asarray(pft.right, dtype=np.int64))
            V.append(np.asarray(c.values, dtype=np.float64)); Tr.append(np.asarray(c.item_trees, dtype=np.int64))
            H.append(pft.height); no.append(n_nodes); io.append(n_items)
            n_nodes += len(pft.left); n_items += len(c.values)
            foc.append(c.overlay.face_of_cell); ver.append(c.face_version)
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)
        self.L, self.R, self.V, self.Tr = cat(L, np.int64), cat(R, np.int64), cat(V, np.float64), cat(Tr, np.int64)
        self.H, self.no, self.io = np.array(H, np.int64), np.array(no, np.int64), np.array(io, np.int64)
        self.Hmax = int(self.H.max()) if H else 0
        self.cell_off = np.concatenate(([0], np.cumsum([len(f) for f in foc])[:-1])).astype(np.int64)
        face_off = np.concatenate(([0], np.cumsum([len(v) for v in ver])[:-1])).astype(np.int64)
        self.FOC = (np.concatenate([np.asarray(f, dtype=np.int64) for f in foc]) + np.repeat(face_off, [len(f) for f in foc])) if foc else np.zeros(0, np.int64)
        self.VER = cat([np.asarray(v, dtype=np.int64) for v in ver], np.int64)
        const = [i for leaf in index._const_leaves for i in leaf.cells.items]
        self.const_trees = np.array([boxes[i].tree for i in const], dtype=np.int64)
        self.const_values = np.array([boxes[i].value for i in const], dtype=np.float64)

    def one(self, z) -> np.ndarray:
        """The leaf values of one input: the cells of the groups, then the compiled walk of their persistent trees."""
        cells = np.asarray(self.ix.query(z)._cells, dtype=np.int64)
        out = np.empty(self.M)
        out[self.const_trees] = self.const_values
        if len(cells):
            _walk(self.VER[self.FOC[cells + self.cell_off]], self.L, self.R, self.H, self.no, self.io, self.V, self.Tr, out)
        return out

    def many(self, Z) -> np.ndarray:
        """The leaf values, one row per input of ``Z``."""
        C = np.stack([np.asarray(self.ix.query(z)._cells, dtype=np.int64) for z in Z])
        q, G = C.shape
        out = np.empty((q, self.M))
        out[:, self.const_trees] = self.const_values
        if G == 0:
            return out
        codes = self.VER[self.FOC[(C + self.cell_off).ravel()]]
        gid, qid = np.tile(np.arange(G), q), np.repeat(np.arange(q), G)
        dc, dg, dq = [], [], []
        for level in range(self.Hmax + 1):
            fin = self.H[gid] == level            # these groups reached their leaves: the codes are items
            if fin.any():
                dc.append(codes[fin]); dg.append(gid[fin]); dq.append(qid[fin])
            keep = ~fin
            if not keep.any() or level == self.Hmax:
                break
            c, g, qq = codes[keep] + self.no[gid[keep]], gid[keep], qid[keep]
            codes, gid, qid = np.concatenate((self.L[c], self.R[c])), np.concatenate((g, g)), np.concatenate((qq, qq))
        items = -np.concatenate(dc) - 2
        g, qq = np.concatenate(dg), np.concatenate(dq)
        ok = items >= 0
        gi = items[ok] + self.io[g[ok]]
        out[qq[ok], self.Tr[gi]] = self.V[gi]
        return out


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
    ap.add_argument("--run-units", type=int, default=1000, help="units on which CellForest runs (all by default)")
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--build-threads", type=int, default=8)
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
    pa_y = scm.parents[y]
    ix_y = mech.indexes[y]
    w_set = np.random.default_rng([2028, args.d]).uniform(0, 2, M)          # one weight vector per setting
    w_pre = (np.arange(M) < M // 2).astype(np.float64)                      # the first half of the trees
    prep = {}
    t0 = time.perf_counter(); lin = {"prefix": reweighted(ix_y, w_pre)}; prep["prefix"] = time.perf_counter() - t0
    t0 = time.perf_counter(); lin["weights"] = reweighted(ix_y, w_set); prep["weights"] = time.perf_counter() - t0
    t0 = time.perf_counter(); reader = (LogLeafReader if _is_log(ix_y) else LeafReader)(ix_y, M); prep["median"] = time.perf_counter() - t0
    lin["sum"], prep["sum"] = ix_y, 0.0
    prep["trimmed"] = prep["median"]
    phis = {"sum": lambda v: float(v.sum()), "prefix": lambda v: float(v @ w_pre), "weights": lambda v: float(v @ w_set),
            "median": lambda v: float(_median(v)), "trimmed": lambda v: float(_trimmed(v))}
    reader.one([float(rows.iloc[0][p]) for p in pa_y])   # compile the walk outside the timer
    _median(np.zeros(3)); _trimmed(np.zeros(30))
    res = {k: dict(cf_s=0.0, cf_n=0, base_s=0.0, base_n=0, err=0.0) for k in phis}

    def mediators(unit, t, observed):
        values = dict(unit)
        for j, name in enumerate(T):
            values[name] = float(t[j])
        for x in order[:-1]:
            values[x] = unit[x] + (mech.indexes[x].query([values[p] for p in scm.parents[x]]).value - observed[x])
        return [values[p] for p in pa_y]

    for u, Q in enumerate(queries(state, args.d, args.per_unit)):
        if u >= args.run_units:
            break
        unit = rows.iloc[u].to_dict()
        three = ThreeStepPhi(scm, unit, T, y) if u < args.base_units else None
        for k, phi in phis.items():
            t0 = time.perf_counter()  # per unit: the observed mediators and the factual value of the outcome
            observed = {x: mech.indexes[x].query([unit[p] for p in scm.parents[x]]).value for x in order[:-1]}
            zu = [unit[p] for p in pa_y]
            if k in ("median", "trimmed"):  # every selected leaf, one query at a time
                fu = phi(reader.one(zu))
                got = [unit[y] + phi(reader.one(mediators(unit, t, observed))) - fu for t in Q]
            else:  # one query at a time: one value per group
                ix = lin[k]
                fu = ix.query(zu).value
                got = [unit[y] + ix.query(mediators(unit, t, observed)).value - fu for t in Q]
            res[k]["cf_s"] += time.perf_counter() - t0
            res[k]["cf_n"] += len(Q)
            if three is not None:
                t0 = time.perf_counter()
                ref = [three.query_phi(t, phi) for t in Q]
                res[k]["base_s"] += time.perf_counter() - t0
                res[k]["base_n"] += len(Q)
                res[k]["err"] = max(res[k]["err"], float(np.max(np.abs(np.array(got) - np.array(ref)))))
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([dict(dataset=args.dataset, family=args.family, d=args.d, combo=k, trees=M, path=len(order),
                            groups=len(ix_y.groups), build_s=build_s, prep_s=prep[k], cf_us=r["cf_s"] / r["cf_n"] * 1e6,
                            base_us=r["base_s"] / max(r["base_n"], 1) * 1e6, queries=r["cf_n"], base_queries=r["base_n"],
                            max_abs_err=r["err"]) for k, r in res.items()])
    df.to_csv(out / f"{args.dataset}_{args.family}_d{args.d}.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
