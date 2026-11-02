"""Online evaluation of counterfactuals on a DAG, at the same implementation level as the index.

* :class:`ScmOnlinePy` - pure Python: for every mechanism on a path from the
  treatments to the outcome, in topological order, walk every tree from its
  root to a leaf (recording the leaves), exactly like
  :class:`~counterprovenance.baselines.OnlineLoop` does for one model.
* :class:`ScmNumba` - the same propagation compiled with numba: all trees of
  all mechanisms are flattened into a few arrays and one call answers a query.

Both read inputs the way each library does (single-precision rounding,
LightGBM's zero threshold), compute a value as ``u_X + (f_X(pa') - f_X(pa))``
like :mod:`~counterprovenance.scm`, and agree with
:class:`~counterprovenance.scm.ScmOnline`.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .baselines import make_cast
from .scm import FittedSCM, relevant_nodes

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


class ScmOnlinePy:
    """Pure-Python propagation through the mechanisms of the DAG."""

    def __init__(self, scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> None:
        self.treatments = list(treatments)
        self.outcome = outcome
        order = relevant_nodes(scm.parents, treatments, outcome)
        self.unit = {k: float(v) for k, v in unit.items()}
        self.mechs = []
        for node in order:
            ens = scm.ensembles[node]
            obs = scm.observed_leaf_values(node, unit).tolist()
            trees = [(t.feature.tolist(), t.threshold.tolist(), t.left.tolist(), t.right.tolist(), t.value.tolist(), o)
                     for t, o in zip(ens.trees, obs)]
            self.mechs.append((node, list(scm.parents[node]), make_cast(ens), self.unit[node], trees))

    def query(self, t: Sequence[float]) -> tuple[float, list[int]]:
        values = dict(self.unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        leaves: list[int] = []
        for node, parents, cast, observed, trees in self.mechs:
            x = cast([values[p] for p in parents])
            diffs = []
            leaves = []
            for feature, threshold, left, right, value, obs in trees:
                n = 0
                while feature[n] >= 0:
                    n = left[n] if x[feature[n]] <= threshold[n] else right[n]
                leaves.append(n)
                diffs.append(value[n] - obs)
            values[node] = observed + math.fsum(diffs)
        return values[self.outcome], leaves


if HAVE_NUMBA:

    @njit(cache=True)
    def _propagate(x0, t, t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals, x):
        # mech[k] = (out node, first tree, end tree, f32 flag); cvals[k] = (observed value, zero threshold)
        # obs[m] = value of the leaf the unit reaches in tree m
        for i in range(x0.shape[0]):
            x[i] = x0[i]
        for i in range(t_idx.shape[0]):
            x[t_idx[i]] = t[i]
        for k in range(mech.shape[0]):
            f32 = mech[k, 3]
            zt = cvals[k, 1]
            total = 0.0
            for m in range(mech[k, 1], mech[k, 2]):
                base = tree_off[m]
                node = 0
                while feat[base + node] >= 0:
                    v = x[feat[base + node]]
                    if f32 == 1:
                        v = np.float64(np.float32(v))
                    if -zt <= v <= zt:
                        v = 0.0
                    if v <= thr[base + node]:
                        node = left[base + node]
                    else:
                        node = right[base + node]
                total += val[base + node] - obs[m]
            x[mech[k, 0]] = cvals[k, 0] + total
        return x[mech[mech.shape[0] - 1, 0]]

    @njit(cache=True)
    def _propagate_batch(x0, T, t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals, x, out):
        for q in range(T.shape[0]):
            out[q] = _propagate(x0, T[q], t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals, x)


class ScmNumba:
    """Propagation through the DAG compiled with numba (one call per query)."""

    def __init__(self, scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> None:
        if not HAVE_NUMBA:
            raise ImportError("numba is required")
        order = relevant_nodes(scm.parents, treatments, outcome)
        names = sorted(set(unit) | set(order))
        pos = {n: i for i, n in enumerate(names)}
        self.x0 = np.array([float(unit.get(n, 0.0)) for n in names])
        self.t_idx = np.array([pos[n] for n in treatments], dtype=np.int64)
        mech, cvals, tree_off = [], [], []
        feat, thr, left, right, val, obs = [], [], [], [], [], []
        off = 0
        for node in order:
            ens = scm.ensembles[node]
            pidx = np.array([pos[p] for p in scm.parents[node]], dtype=np.int64)
            first = len(tree_off)
            obs.append(scm.observed_leaf_values(node, unit))
            for tr in ens.trees:
                tree_off.append(off)
                f = np.where(tr.feature >= 0, pidx[np.maximum(tr.feature, 0)], -1)
                feat.append(f)
                thr.append(tr.threshold)
                left.append(tr.left)
                right.append(tr.right)
                val.append(tr.value)
                off += len(tr.feature)
            mech.append((pos[node], first, len(tree_off), int(ens.input_dtype == np.float32)))
            zt = ens.zero_threshold if ens.zero_threshold > 0 else -1.0
            cvals.append((float(unit[node]), zt))
        self.args = (
            np.array(self.t_idx), np.array(mech, dtype=np.int64), np.array(tree_off, dtype=np.int64),
            np.concatenate(feat).astype(np.int64), np.concatenate(thr).astype(np.float64),
            np.concatenate(left).astype(np.int64), np.concatenate(right).astype(np.int64),
            np.concatenate(val).astype(np.float64), np.concatenate(obs), np.array(cvals, dtype=np.float64),
        )
        self.x = np.zeros(len(names))

    def query(self, t: np.ndarray) -> float:
        return _propagate(self.x0, np.asarray(t, dtype=np.float64), *self.args, self.x)

    def batch(self, T: np.ndarray) -> np.ndarray:
        out = np.empty(len(T))
        _propagate_batch(self.x0, np.ascontiguousarray(T, dtype=np.float64), *self.args, self.x, out)
        return out


if HAVE_NUMBA:

    @njit(cache=True)
    def _walk_tree(x, base, f32, zt, feat, thr, left, right):
        node = 0
        while feat[base + node] >= 0:
            v = x[feat[base + node]]
            if f32 == 1:
                v = np.float64(np.float32(v))
            if -zt <= v <= zt:
                v = 0.0
            if v <= thr[base + node]:
                node = left[base + node]
            else:
                node = right[base + node]
        return node

    @njit(cache=True)
    def _incremental(t, t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals, x, cur,
                     par_off, par, dep_off, dep, changed, stamp, step):
        # mech[k] = (out node, first tree, end tree, f32 flag); cvals[k] = (observed value, zero threshold)
        # par[par_off[k]:par_off[k+1]] = parent nodes of mechanism k; dep[dep_off[j]:dep_off[j+1]] = trees
        # of mechanism k that split on its j-th parent; cur[m] = leaf value - observed leaf value of tree m
        for i in range(changed.shape[0]):
            changed[i] = False
        for i in range(t_idx.shape[0]):
            if x[t_idx[i]] != t[i]:
                x[t_idx[i]] = t[i]
                changed[t_idx[i]] = True
        for k in range(mech.shape[0]):
            f32 = mech[k, 3]
            zt = cvals[k, 1]
            dirty = False
            for j in range(par_off[k], par_off[k + 1]):
                if not changed[par[j]]:
                    continue
                for q in range(dep_off[j], dep_off[j + 1]):
                    m = dep[q]
                    if stamp[m] == step:
                        continue
                    stamp[m] = step
                    base = tree_off[m]
                    node = _walk_tree(x, base, f32, zt, feat, thr, left, right)
                    dv = val[base + node] - obs[m]
                    if dv != cur[m]:
                        cur[m] = dv
                        dirty = True
            if dirty:
                total = 0.0
                for m in range(mech[k, 1], mech[k, 2]):
                    total += cur[m]
                v = cvals[k, 0] + total
                if v != x[mech[k, 0]]:
                    x[mech[k, 0]] = v
                    changed[mech[k, 0]] = True
        return x[mech[mech.shape[0] - 1, 0]]


class ScmIncremental(ScmNumba):
    """Incremental propagation compiled with numba: keeps the leaf of every tree of every mechanism
    on the path and, when the intervention moves, re-walks only the trees that split on a changed
    input, re-sums only the mechanisms whose trees changed, and propagates only changed values.
    Its arithmetic is that of :class:`ScmNumba`, so both return the same floats."""

    def __init__(self, scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str,
                 t: np.ndarray) -> None:
        super().__init__(scm, unit, treatments, outcome)
        t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals = self.args
        par_off, par, dep_off, dep = [0], [], [0], []
        for k in range(len(mech)):
            first, end = int(mech[k, 1]), int(mech[k, 2])
            uses: dict[int, list[int]] = {}
            for m in range(first, end):
                stop = tree_off[m + 1] if m + 1 < len(tree_off) else len(feat)
                for f in set(feat[tree_off[m]:stop].tolist()) - {-1}:
                    uses.setdefault(f, []).append(m)
            for p, trees in sorted(uses.items()):
                par.append(p)
                dep.extend(trees)
                dep_off.append(len(dep))
            par_off.append(len(par))
        self.inc = (np.array(par_off, dtype=np.int64), np.array(par, dtype=np.int64),
                    np.array(dep_off, dtype=np.int64), np.array(dep, dtype=np.int64))
        self.cur = np.zeros(len(tree_off))
        self.changed = np.zeros(len(self.x), dtype=np.bool_)
        self.stamp = np.zeros(len(tree_off), dtype=np.int64)
        self.step = 0
        self.reset(t)

    def reset(self, t: np.ndarray) -> float:
        """Full propagation at ``t``, which also fills the leaf of every tree."""
        value = self.query(np.asarray(t, dtype=np.float64))
        t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals = self.args
        for k in range(len(mech)):
            f32, zt = int(mech[k, 3]), float(cvals[k, 1])
            for m in range(int(mech[k, 1]), int(mech[k, 2])):
                node = _walk_tree(self.x, int(tree_off[m]), f32, zt, feat, thr, left, right)
                self.cur[m] = val[tree_off[m] + node] - obs[m]
        return value

    def move(self, t: np.ndarray) -> float:
        self.step += 1
        t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals = self.args
        return _incremental(t, t_idx, mech, tree_off, feat, thr, left, right, val, obs, cvals, self.x, self.cur,
                            *self.inc, self.changed, self.stamp, self.step)


class ScmThreeStepPy:
    """Pure-Python counterfactual by the textbook three steps on every query, caching nothing:
    abduction walks every mechanism on the paths at the unit's observed parents, the action sets
    ``T``, and prediction walks them again at the counterfactual parents.  Same arithmetic as
    :class:`ScmOnlinePy` (``u + fsum`` of leaf differences), so both return the same floats."""

    def __init__(self, scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> None:
        self.treatments = list(treatments)
        self.outcome = outcome
        order = relevant_nodes(scm.parents, treatments, outcome)
        self.unit = {k: float(v) for k, v in unit.items()}
        self.mechs = []
        for node in order:
            ens = scm.ensembles[node]
            trees = [(t.feature.tolist(), t.threshold.tolist(), t.left.tolist(), t.right.tolist(), t.value.tolist())
                     for t in ens.trees]
            self.mechs.append((node, list(scm.parents[node]), make_cast(ens), trees))

    @staticmethod
    def _leaves(x, trees):
        out = []
        for feature, threshold, left, right, value in trees:
            n = 0
            while feature[n] >= 0:
                n = left[n] if x[feature[n]] <= threshold[n] else right[n]
            out.append(n)
        return out

    def query(self, t: Sequence[float]) -> tuple[float, list[int]]:
        unit = self.unit
        # abduction: the leaves (hence f(pa)) of every mechanism at the observed parents
        observed = [self._leaves(cast([unit[p] for p in parents]), trees)
                    for _, parents, cast, trees in self.mechs]
        # action
        values = dict(unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        # prediction: X = u_X + (f(pa') - f(pa)), in topological order
        leaves: list[int] = []
        for (node, parents, cast, trees), obs in zip(self.mechs, observed):
            leaves = self._leaves(cast([values[p] for p in parents]), trees)
            values[node] = unit[node] + math.fsum(tr[4][n] - tr[4][o] for tr, n, o in zip(trees, leaves, obs))
        return values[self.outcome], leaves

    def query_weighted(self, t: Sequence[float], w) -> float:
        """The three steps with weights on the outcome's trees; the mediators keep their full models."""
        unit = self.unit
        observed = [self._leaves(cast([unit[p] for p in parents]), trees)
                    for _, parents, cast, trees in self.mechs]
        values = dict(unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        for (node, parents, cast, trees), obs in zip(self.mechs, observed):
            leaves = self._leaves(cast([values[p] for p in parents]), trees)
            if node == self.outcome:
                values[node] = unit[node] + math.fsum(float(u) * (tr[4][n] - tr[4][o])
                                                      for u, tr, n, o in zip(w, trees, leaves, obs))
            else:
                values[node] = unit[node] + math.fsum(tr[4][n] - tr[4][o] for tr, n, o in zip(trees, leaves, obs))
        return values[self.outcome]

    def query_prefix(self, t: Sequence[float], n: int) -> float:
        """The three steps with the first ``n`` trees of the outcome; it walks only those trees of it."""
        unit = self.unit
        observed = []
        for node, parents, cast, trees in self.mechs:
            use = trees[:n] if node == self.outcome else trees
            observed.append(self._leaves(cast([unit[p] for p in parents]), use))
        values = dict(unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        for (node, parents, cast, trees), obs in zip(self.mechs, observed):
            use = trees[:n] if node == self.outcome else trees
            leaves = self._leaves(cast([values[p] for p in parents]), use)
            values[node] = unit[node] + math.fsum(tr[4][a] - tr[4][o] for tr, a, o in zip(use, leaves, obs))
        return values[self.outcome]
