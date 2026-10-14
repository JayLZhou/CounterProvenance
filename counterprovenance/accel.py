"""Compiled (numba) versions of the compiled query and of the online loop.

Both sides use the same compiler, so the comparison measures the algorithms:
:class:`NumbaIndex` runs the packed lookup (``d`` binary searches, one table
read per group) and the leaf bitmasks of a compiled
:class:`~counterprovenance.index.ProvenanceIndex`; :class:`NumbaLoop` walks the
trees.  ``query`` is one call from Python per query, ``batch`` loops over many
queries inside compiled code (sweeps).  numba is optional.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .baselines import varying_trees
from .ensemble import BoundEnsemble, TreeEnsemble

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

_DEBRUIJN = 0x03F79D71B4CB0A89
_MASK64 = (1 << 64) - 1
_DB = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _DB[((((1 << _i) * _DEBRUIJN) & _MASK64) >> 58)] = _i

if HAVE_NUMBA:

    @njit(cache=True)
    def _index_query(th, th_len, contrib, tables, offs, ub, bm, bvals, const, f32, t, slabs, cells, leaves, db):
        d = th.shape[0]
        g = offs.shape[0]
        for i in range(g):
            cells[i] = 0
        for k in range(d):
            v = t[k]
            if f32:
                v = np.float64(np.float32(v))
            lo = 0
            hi = th_len[k]
            while lo < hi:
                mid = (lo + hi) >> 1
                if th[k, mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            slabs[k] = lo
            if g > 0:
                row = contrib[k, lo]
                for i in range(g):
                    cells[i] += row[i]
        total = const
        for i in range(g):
            total += tables[offs[i] + cells[i]]
        nb = ub.shape[0]
        for j in range(bvals.shape[0]):
            m = bm[0, slabs[ub[0]], j]
            for u in range(1, nb):
                m &= bm[u, slabs[ub[u]], j]
            if m == 0:  # another 64-leaf field of the same tree holds the leaf
                leaves[j] = 0
                continue
            low = m & (~m + np.uint64(1))
            leaf = db[(low * np.uint64(_DEBRUIJN)) >> np.uint64(58)] + 1
            leaves[j] = leaf
            total += bvals[j, leaf]
        return total

    @njit(cache=True)
    def _index_batch(th, th_len, contrib, tables, offs, ub, bm, bvals, const, f32, T, out, slabs, cells, leaves, db):
        for q in range(T.shape[0]):
            out[q] = _index_query(th, th_len, contrib, tables, offs, ub, bm, bvals, const, f32, T[q],
                                  slabs, cells, leaves, db)

    @njit(cache=True)
    def _walk(feat, thr, left, right, val, unit, axes, f32, const, t, x, leaves):
        for i in range(unit.shape[0]):
            x[i] = unit[i]
        for k in range(axes.shape[0]):
            v = t[k]
            if f32:
                v = np.float64(np.float32(v))
            x[axes[k]] = v
        total = const
        for m in range(feat.shape[0]):
            node = 0
            while feat[m, node] >= 0:
                if x[feat[m, node]] <= thr[m, node]:
                    node = left[m, node]
                else:
                    node = right[m, node]
            leaves[m] = node
            total += val[m, node]
        return total

    @njit(cache=True)
    def _walk_batch(feat, thr, left, right, val, unit, axes, f32, const, T, out, x, leaves):
        for q in range(T.shape[0]):
            out[q] = _walk(feat, thr, left, right, val, unit, axes, f32, const, T[q], x, leaves)


def _require() -> None:
    if not HAVE_NUMBA:
        raise ImportError("numba is required for counterprovenance.accel")


class NumbaIndex:
    """Compiled query of a compiled ``ProvenanceIndex`` (same tables, same
    bitmasks; values agree up to floating-point summation order)."""

    def __init__(self, index) -> None:
        _require()
        if not index.compiled:
            raise ValueError("NumbaIndex needs a compiled index (tau = inf)")
        bound, slabs = index.bound, index.slabs
        d = bound.d
        counts = slabs.counts
        width = max(1, max(len(t) for t in slabs.thresholds))
        self.th = np.full((d, width), np.inf)
        self.th_len = np.zeros(d, dtype=np.int64)
        for a, t in enumerate(slabs.thresholds):
            self.th[a, : len(t)] = t
            self.th_len[a] = len(t)
        g = len(index._tables)
        self.contrib = np.zeros((d, max(counts), g), dtype=np.int64)
        for gi, a, c in index._contrib:
            self.contrib[a, : len(c), gi] = c
        parts = [np.frombuffer(t, dtype=np.float64) for t in index._tables]
        self.tables = np.concatenate(parts) if parts else np.zeros(1)
        self.offs = np.zeros(g, dtype=np.int64)
        if g:
            self.offs[1:] = np.cumsum([len(p) for p in parts])[:-1]
        K = len(index._bvals)
        used = sorted(index._bmasks)
        self.ub = np.array(used, dtype=np.int64)
        self.bm = np.zeros((max(1, len(used)), max(counts), K), dtype=np.uint64)
        for u, a in enumerate(used):
            self.bm[u, : counts[a], :] = index._bmasks[a]
        self.bvals = np.zeros((K, 65))
        for k, vals in enumerate(index._bvals):
            self.bvals[k, : len(vals)] = vals
        self.const = float(index._const)
        self.f32 = bound.ensemble.input_dtype == np.float32
        self.slabs = np.zeros(d, dtype=np.int64)
        self.cells = np.zeros(g, dtype=np.int64)
        self.leaves = np.zeros(K, dtype=np.int64)
        self._args = (self.th, self.th_len, self.contrib, self.tables, self.offs, self.ub, self.bm, self.bvals,
                      self.const, self.f32)

    def query(self, t: np.ndarray) -> float:
        """``t``: float64 array of the intervention values."""
        return _index_query(*self._args, t, self.slabs, self.cells, self.leaves, _DB)

    def query_full(self, t: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        value = self.query(t)
        return value, self.cells.copy(), self.leaves.copy()

    def batch(self, T: np.ndarray) -> np.ndarray:
        out = np.empty(len(T))
        _index_batch(*self._args, np.ascontiguousarray(T, dtype=np.float64), out, self.slabs, self.cells,
                     self.leaves, _DB)
        return out


class NumbaLoop:
    """Compiled online loop over all trees (or, with ``bound``, the varying ones)."""

    def __init__(self, ensemble: TreeEnsemble, unit: Sequence[float], axes: Sequence[int],
                 noise: float = 0.0, bound: BoundEnsemble | None = None) -> None:
        _require()
        if bound is None:
            trees = list(range(ensemble.tree_count))
            const = ensemble.base + noise
        else:
            trees = varying_trees(bound)
            keep = set(trees)
            const = math.fsum([bound.offset] + [b.value for b in bound.boxes if b.tree not in keep])
        n = max(1, max((len(ensemble.trees[m].feature) for m in trees), default=1))
        M = len(trees)
        self.feat = np.full((M, n), -1, dtype=np.int64)
        self.thr = np.zeros((M, n))
        self.left = np.zeros((M, n), dtype=np.int64)
        self.right = np.zeros((M, n), dtype=np.int64)
        self.val = np.zeros((M, n))
        for i, m in enumerate(trees):
            tr = ensemble.trees[m]
            k = len(tr.feature)
            self.feat[i, :k] = tr.feature
            self.thr[i, :k] = tr.threshold
            self.left[i, :k] = tr.left
            self.right[i, :k] = tr.right
            self.val[i, :k] = tr.value
        self.unit = ensemble.cast(np.asarray(unit, dtype=float).reshape(1, -1))[0].astype(np.float64)
        self.axes = np.asarray(axes, dtype=np.int64)
        self.f32 = ensemble.input_dtype == np.float32
        self.const = float(const)
        self.x = np.zeros(len(self.unit))
        self.leaves = np.zeros(M, dtype=np.int64)
        self._args = (self.feat, self.thr, self.left, self.right, self.val, self.unit, self.axes, self.f32,
                      self.const)

    def query(self, t: np.ndarray) -> float:
        return _walk(*self._args, t, self.x, self.leaves)

    def batch(self, T: np.ndarray) -> np.ndarray:
        out = np.empty(len(T))
        _walk_batch(*self._args, np.ascontiguousarray(T, dtype=np.float64), out, self.x, self.leaves)
        return out
