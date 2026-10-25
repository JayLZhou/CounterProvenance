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
