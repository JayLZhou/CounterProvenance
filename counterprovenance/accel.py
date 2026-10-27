"""Compiled (numba) versions of the compiled query and of the online loop.

Both sides use the same compiler and the same calling convention (a few flat
arrays per call), so the comparison measures the algorithms:
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

# int header of NumbaIndex.ints
_D, _W, _G, _K, _NB, _C, _O_THLEN, _O_OFFS, _O_UB, _O_CONTRIB, _O_DB, _O_TH, _O_TAB, _O_BV, _F32 = range(15)
# int header of NumbaLoop.ints
_M, _N, _P, _NA, _LF32, _O_FEAT, _O_LEFT, _O_RIGHT, _O_AXES, _O_THR, _O_VAL, _O_UNIT = range(12)

if HAVE_NUMBA:
    from numba.cpython.unsafe.numbers import trailing_zeros

    @njit(cache=True)
    def _index_query(il, fl, bm, t, work, mw):
        d = il[0]
        width = il[1]
        g = il[2]
        K = il[3]
        nb = il[4]
        C = il[5]
        o_thlen = il[6]
        o_offs = il[7]
        o_ub = il[8]
        o_contrib = il[9]
        o_db = il[10]
        o_th = il[11]
        o_tab = il[12]
        o_bv = il[13]
        f32 = il[14]
        for i in range(g):
            work[d + i] = 0
        zt = fl[1]
        for k in range(d):
            v = t[k]
            if f32 == 1:
                v = np.float64(np.float32(v))
            if -zt <= v <= zt:
                v = 0.0
            lo = 0
            hi = il[o_thlen + k]
            base = o_th + k * width
            while lo < hi:
                mid = (lo + hi) >> 1
                if fl[base + mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            work[k] = lo
            if g > 0:
                cb = o_contrib + (k * C + lo) * g
                for i in range(g):
                    work[d + i] += il[cb + i]
        total = fl[0]
        for i in range(g):
            total += fl[o_tab + il[o_offs + i] + work[d + i]]
        if K > 0:
            # AND the slab rows of the used axes, one contiguous row at a time
            row = work[il[o_ub]] * K
            for j in range(K):
                mw[j] = bm[row + j]
            for u in range(1, nb):
                row = (u * C + work[il[o_ub + u]]) * K
                for j in range(K):
                    mw[j] &= bm[row + j]
        for j in range(K):
            m = mw[j]
            if m == 0:  # another 64-leaf field of the same tree holds the leaf
                work[d + g + j] = 0
                continue
            leaf = np.int64(trailing_zeros(m)) + 1
            work[d + g + j] = leaf
            total += fl[o_bv + j * 65 + leaf]
        return total

    @njit(cache=True)
    def _index_batch(il, fl, bm, T, out, work, mw):
        for q in range(T.shape[0]):
            out[q] = _index_query(il, fl, bm, T[q], work, mw)

    @njit(cache=True)
    def _walk(il, fl, t, x, leaves):
        M = il[0]
        n = il[1]
        p = il[2]
        na = il[3]
        f32 = il[4]
        o_feat = il[5]
        o_left = il[6]
        o_right = il[7]
        o_axes = il[8]
        o_thr = il[9]
        o_val = il[10]
        o_unit = il[11]
        for i in range(p):
            x[i] = fl[o_unit + i]
        zt = fl[1]
        for k in range(na):
            v = t[k]
            if f32 == 1:
                v = np.float64(np.float32(v))
            if -zt <= v <= zt:
                v = 0.0
            x[il[o_axes + k]] = v
        total = fl[0]
        for m in range(M):
            base = m * n
            node = 0
            while il[o_feat + base + node] >= 0:
                if x[il[o_feat + base + node]] <= fl[o_thr + base + node]:
                    node = il[o_left + base + node]
                else:
                    node = il[o_right + base + node]
            leaves[m] = node
            total += fl[o_val + base + node]
        return total

    @njit(cache=True)
    def _walk_batch(il, fl, T, out, x, leaves):
        for q in range(T.shape[0]):
            out[q] = _walk(il, fl, T[q], x, leaves)


def _require() -> None:
    if not HAVE_NUMBA:
        raise ImportError("numba is required for counterprovenance.accel")


def _pack(parts):
    """Concatenate arrays; return (flat array, start offset of every part)."""
    offsets, pos = [], 0
    for p in parts:
        offsets.append(pos)
        pos += len(p)
    return np.concatenate(parts) if parts else np.zeros(0), offsets


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
        C = max(counts)
        width = max(1, max(len(t) for t in slabs.thresholds))
        th = np.full((d, width), np.inf)
        th_len = np.zeros(d, dtype=np.int64)
        for a, t in enumerate(slabs.thresholds):
            th[a, : len(t)] = t
            th_len[a] = len(t)
        g = len(index._tables)
        contrib = np.zeros((d, C, g), dtype=np.int64)
        for gi, a, c in index._contrib:
            contrib[a, : len(c), gi] = c
        tables = [np.frombuffer(t, dtype=np.float64) for t in index._tables]
        offs = np.zeros(g, dtype=np.int64)
        if g:
            offs[1:] = np.cumsum([len(t) for t in tables])[:-1]
        K = len(index._bvals)
        used = sorted(index._bmasks)
        bm = np.zeros((max(1, len(used)), C, K), dtype=np.uint64)
        for u, a in enumerate(used):
            bm[u, : counts[a], :] = index._bmasks[a]
        bvals = np.zeros((K, 65))
        for k, vals in enumerate(index._bvals):
            bvals[k, : len(vals)] = vals
        f32 = int(bound.ensemble.input_dtype == np.float32)
        zt = bound.ensemble.zero_threshold if bound.ensemble.zero_threshold > 0 else -1.0
        fl, (_, o_th, o_tab, o_bv) = _pack([
            np.array([index._const, zt]), th.ravel(), np.concatenate(tables) if tables else np.zeros(0), bvals.ravel(),
        ])
        header = np.zeros(15, dtype=np.int64)
        ints, (_, o_thlen, o_offs, o_ub, o_contrib, o_db) = _pack([
            header, th_len, offs, np.array(used, dtype=np.int64), contrib.ravel(), _DB,
        ])
        ints = ints.astype(np.int64)
        ints[: 15] = [d, width, g, K, len(used), C, o_thlen, o_offs, o_ub, o_contrib, o_db, o_th, o_tab, o_bv, f32]
        self.ints, self.floats, self.bm = ints, fl.astype(np.float64), bm.ravel()
        self.work = np.zeros(d + g + K, dtype=np.int64)
        self.mwork = np.zeros(max(1, K), dtype=np.uint64)
        self.d, self.g, self.K = d, g, K

    def query(self, t: np.ndarray) -> float:
        """``t``: float64 array of the intervention values."""
        return _index_query(self.ints, self.floats, self.bm, t, self.work, self.mwork)

    def query_full(self, t: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        value = self.query(t)
        d, g = self.d, self.g
        return value, self.work[d: d + g].copy(), self.work[d + g:].copy()

    def batch(self, T: np.ndarray) -> np.ndarray:
        out = np.empty(len(T))
        _index_batch(self.ints, self.floats, self.bm, np.ascontiguousarray(T, dtype=np.float64), out, self.work,
                     self.mwork)
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
        feat = np.full((M, n), -1, dtype=np.int64)
        thr = np.zeros((M, n))
        left = np.zeros((M, n), dtype=np.int64)
        right = np.zeros((M, n), dtype=np.int64)
        val = np.zeros((M, n))
        for i, m in enumerate(trees):
            tr = ensemble.trees[m]
            k = len(tr.feature)
            feat[i, :k] = tr.feature
            thr[i, :k] = tr.threshold
            left[i, :k] = tr.left
            right[i, :k] = tr.right
            val[i, :k] = tr.value
        unit_arr = ensemble.cast(np.asarray(unit, dtype=float).reshape(1, -1))[0].astype(np.float64)
        axes_arr = np.asarray(axes, dtype=np.int64)
        f32 = int(ensemble.input_dtype == np.float32)
        header = np.zeros(12, dtype=np.int64)
        ints, (_, o_feat, o_left, o_right, o_axes) = _pack([header, feat.ravel(), left.ravel(), right.ravel(), axes_arr])
        o_thr = 2
        o_val = o_thr + thr.size
        o_unit = o_val + val.size
        ints = ints.astype(np.int64)
        ints[:12] = [M, n, len(unit_arr), len(axes_arr), f32, o_feat, o_left, o_right, o_axes, o_thr, o_val, o_unit]
        self.ints = ints
        zt = ensemble.zero_threshold if ensemble.zero_threshold > 0 else -1.0
        self.floats = np.concatenate([np.array([const, zt]), thr.ravel(), val.ravel(), unit_arr]).astype(np.float64)
        self.x = np.zeros(len(unit_arr))
        self.leaves = np.zeros(M, dtype=np.int64)

    def query(self, t: np.ndarray) -> float:
        return _walk(self.ints, self.floats, t, self.x, self.leaves)

    def batch(self, T: np.ndarray) -> np.ndarray:
        out = np.empty(len(T))
        _walk_batch(self.ints, self.floats, np.ascontiguousarray(T, dtype=np.float64), out, self.x, self.leaves)
        return out
