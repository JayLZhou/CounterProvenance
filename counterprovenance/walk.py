"""Walking the cell graph: query sessions that move the intervention step by step.

Sweeps, sliders and recourse searches issue queries that differ from the
previous one in a few coordinates.  Moving ``t_a`` from ``v`` to ``v'`` crosses
the walls of the cell graph that lie on axis ``a`` between ``v`` and ``v'``,
and only the trees that own these walls can change their leaf.  A
:class:`Walk` keeps the current leaf of every tree and, per axis, the walls
sorted by position; a move re-locates only the trees whose walls it crosses and
updates the value by their differences.  The cost of a move is the number of
walls crossed (the weights of the crossed cell-graph edges), independent of the
number of trees and of ``d``; the cells themselves are never materialised, so
the walk works in any dimension.

Each tree is re-located with its *bound decision tree*: a binary tree over
``t`` alone, rebuilt from the tree's leaf boxes by guillotine cuts that do not
cut any box (tests on fixed features and on mediators are already resolved by
the binding), so its depth is about ``log`` of the tree's number of boxes.

:class:`Walk` is the pure-Python version; :class:`NumbaWalk` compiles the same
moves with numba.
"""

from __future__ import annotations

import bisect
import math
from typing import Sequence

import numpy as np

from .ensemble import BoundEnsemble

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# bound decision trees
# ---------------------------------------------------------------------------
def _guillotine(ids: np.ndarray, lo: np.ndarray, hi: np.ndarray, nodes: list, bucket: list) -> int:
    """Append the decision tree of the disjoint boxes ``ids`` to ``nodes``; return its root.

    A node is ``[axis, cut, left, right, bucket_start, bucket_end]``; a leaf has axis -1 and
    lists its boxes in ``bucket`` (one box unless no cut separates them).
    """
    me = len(nodes)
    nodes.append(None)
    best = None
    if len(ids) > 1:
        L, H = lo[ids], hi[ids]
        for a in range(lo.shape[1]):
            cuts = np.unique(H[:, a][np.isfinite(H[:, a])])
            if not len(cuts):
                continue
            left = (H[:, a][None, :] <= cuts[:, None]).sum(1)
            right = (L[:, a][None, :] >= cuts[:, None]).sum(1)
            ok = np.flatnonzero((left + right == len(ids)) & (left > 0) & (right > 0))
            if len(ok):
                k = ok[np.argmin(np.abs(left[ok] - right[ok]))]
                score = abs(int(left[k]) - int(right[k]))
                if best is None or score < best[0]:
                    best = (score, a, float(cuts[k]))
    if best is None:  # one box, or boxes no axis cut separates (not seen for tree bindings)
        nodes[me] = [-1, 0.0, -1, -1, len(bucket), len(bucket) + len(ids)]
        bucket.extend(int(i) for i in ids)
        return me
    _, a, c = best
    go_left = hi[ids, a] <= c
    left = _guillotine(ids[go_left], lo, hi, nodes, bucket)
    right = _guillotine(ids[~go_left], lo, hi, nodes, bucket)
    nodes[me] = [a, c, left, right, 0, 0]
    return me


class _Plan:
    """Decision trees and per-axis walls of a bound ensemble (shared by both walks)."""

    def __init__(self, bound: BoundEnsemble) -> None:
        d = bound.d
        by_tree: dict[int, list[int]] = {}
        for i, b in enumerate(bound.boxes):
            by_tree.setdefault(b.tree, []).append(i)
        lo, hi = bound._LO.reshape(-1, d), bound._HI.reshape(-1, d)
        self.trees = [m for m, ids in sorted(by_tree.items()) if len(ids) > 1]
        const = [bound.offset] + [bound.boxes[ids[0]].value for m, ids in sorted(by_tree.items()) if len(ids) == 1]
        self.const = math.fsum(const)
        nodes: list = []
        bucket: list = []
        self.root = []
        walls = [[] for _ in range(d)]
        for k, m in enumerate(self.trees):
            ids = np.asarray(by_tree[m])
            self.root.append(_guillotine(ids, lo, hi, nodes, bucket))
            for a in range(d):
                for c in np.unique(hi[ids, a][np.isfinite(hi[ids, a])]):
                    walls[a].append((float(c), k))
        self.nodes = nodes
        self.bucket = bucket
        self.lo, self.hi = lo, hi
        self.values = np.array([b.value for b in bound.boxes], dtype=np.float64)
        for w in walls:
            w.sort()
        self.wall_pos = [[c for c, _ in w] for w in walls]
        self.wall_tree = [[k for _, k in w] for w in walls]
        self.d = d
        self.box_tree = [bound.boxes[i].tree for i in range(len(bound.boxes))]
        self.depth = max((self._depth(r) for r in self.root), default=0)

    def _depth(self, n: int) -> int:
        node = self.nodes[n]
        return 0 if node[0] < 0 else 1 + max(self._depth(node[2]), self._depth(node[3]))

    def locate(self, k: int, x: Sequence[float]) -> int:
        nodes = self.nodes
        n = self.root[k]
        node = nodes[n]
        while node[0] >= 0:
            n = node[2] if x[node[0]] <= node[1] else node[3]
            node = nodes[n]
        lo, hi = self.lo, self.hi
        for i in self.bucket[node[4]:node[5]]:
            if all(lo[i, a] < x[a] <= hi[i, a] for a in range(self.d)):
                return i
        raise AssertionError("no box contains the point")


class Walk:
    """Pure-Python walk: ``move(t)`` returns the value at ``t`` from the previous position."""

    def __init__(self, bound: BoundEnsemble, t: Sequence[float], plan: _Plan | None = None) -> None:
        self.bound = bound
        self.plan = plan or _Plan(bound)
        self._cast = bound.ensemble.cast
        self.x = [float(v) for v in self._cast(np.asarray(t, dtype=float))]
        p = self.plan
        self.box = [p.locate(k, self.x) for k in range(len(p.trees))]
        self._stamp = [0] * len(p.trees)
        self._step = 0
        self.relocated = 0
        self._value = math.fsum([p.const] + [p.values[i] for i in self.box])

    @property
    def value(self) -> float:
        return self._value

    def move(self, t: Sequence[float]) -> float:
        p = self.plan
        x_new = [float(v) for v in self._cast(np.asarray(t, dtype=float))]
        self._step += 1
        step, stamp, touched = self._step, self._stamp, []
        for a in range(p.d):
            v, w = self.x[a], x_new[a]
            if v == w:
                continue
            pos, trees = p.wall_pos[a], p.wall_tree[a]
            lo, hi = (v, w) if v < w else (w, v)
            for j in range(bisect.bisect_left(pos, lo), bisect.bisect_left(pos, hi)):
                k = trees[j]
                if stamp[k] != step:
                    stamp[k] = step
                    touched.append(k)
        self.x = x_new
        if touched:
            delta = []
            for k in touched:
                i = p.locate(k, x_new)
                old = self.box[k]
                if i != old:
                    delta.append(p.values[i])
                    delta.append(-p.values[old])
                    self.box[k] = i
            self.relocated += len(touched)
            if delta:
                self._value = math.fsum([p.const] + [p.values[i] for i in self.box]) if len(delta) > 64 \
                    else math.fsum([self._value] + delta)
        return self._value

    def set(self, axis: int, value: float) -> float:
        t = list(self.x)
        t[axis] = value
        return self.move(t)

    def labels(self) -> list[tuple[int, int]]:
        """Provenance of the current position: ``(tree, native leaf)`` of every varying tree."""
        return [self.bound.label(i) for i in self.box]


# ---------------------------------------------------------------------------
# numba walk
# ---------------------------------------------------------------------------
# A numba call pays for every array argument, so the walk lives in four flat arrays:
# ``wi``/``wf`` (static ints and floats, laid out by the header below) and ``mi``/``mf``
# (mutable: box, stamp, touched / x, acc, last, session state).
_D, _NT, _F32, _NN, _NBK, _NBX, _NW = range(7)
_O_ROOT, _O_NAX, _O_NLEFT, _O_NRIGHT, _O_NB0, _O_NB1, _O_BUCKET, _O_WALL_OFF, _O_WALL_TREE = range(7, 16)
_O_NCUT, _O_WALL_POS, _O_VALUES, _O_LO, _O_HI = range(16, 21)
_HEADER = 21
_STATE = 6  # session state: step, mode (0 walk, 1 index), ema of walls per move, walked, queried, relocated

if HAVE_NUMBA:

    @njit(cache=True)
    def _views(wi, wf):
        d, nt, nn, nbk, nbx, nw = wi[_D], wi[_NT], wi[_NN], wi[_NBK], wi[_NBX], wi[_NW]
        return (wi[wi[_O_ROOT]:wi[_O_ROOT] + nt], wi[wi[_O_NAX]:wi[_O_NAX] + nn],
                wi[wi[_O_NLEFT]:wi[_O_NLEFT] + nn], wi[wi[_O_NRIGHT]:wi[_O_NRIGHT] + nn],
                wi[wi[_O_NB0]:wi[_O_NB0] + nn], wi[wi[_O_NB1]:wi[_O_NB1] + nn],
                wi[wi[_O_BUCKET]:wi[_O_BUCKET] + nbk], wi[wi[_O_WALL_OFF]:wi[_O_WALL_OFF] + d + 1],
                wi[wi[_O_WALL_TREE]:wi[_O_WALL_TREE] + nw], wf[wi[_O_NCUT]:wi[_O_NCUT] + nn],
                wf[wi[_O_WALL_POS]:wi[_O_WALL_POS] + nw], wf[wi[_O_VALUES]:wi[_O_VALUES] + nbx],
                wf[wi[_O_LO]:wi[_O_LO] + nbx * d].reshape((nbx, d)), wf[wi[_O_HI]:wi[_O_HI] + nbx * d].reshape((nbx, d)))

    @njit(cache=True)
    def _cast(v, f32, zt):
        if f32 == 1:
            v = np.float64(np.float32(v))
        if -zt <= v <= zt:
            v = 0.0
        return v

    @njit(cache=True)
    def _wall_range(wall_off, wall_pos, a, lo_v, hi_v):
        """Indices ``[s, e)`` of the walls on axis ``a`` in ``[lo_v, hi_v)``."""
        l, r = wall_off[a], wall_off[a + 1]
        while l < r:
            mid = (l + r) >> 1
            if wall_pos[mid] < lo_v:
                l = mid + 1
            else:
                r = mid
        s = l
        r = wall_off[a + 1]
        while l < r:
            mid = (l + r) >> 1
            if wall_pos[mid] < hi_v:
                l = mid + 1
            else:
                r = mid
        return s, l

    @njit(cache=True)
    def _locate(k, x, root, nax, ncut, nleft, nright, nb0, nb1, bucket, lo, hi):
        n = root[k]
        while nax[n] >= 0:
            if x[nax[n]] <= ncut[n]:
                n = nleft[n]
            else:
                n = nright[n]
        d = x.shape[0]
        for j in range(nb0[n], nb1[n]):
            i = bucket[j]
            inside = True
            for a in range(d):
                if not (lo[i, a] < x[a] <= hi[i, a]):
                    inside = False
                    break
            if inside:
                return i
        return -1

    @njit(cache=True)
    def _move(t, step, wi, wf, mi, mf):
        """Move the walk to ``t``; returns the number of re-located trees."""
        d, nt, f32, zt = wi[_D], wi[_NT], wi[_F32], wf[0]
        root, nax, nleft, nright, nb0, nb1, bucket, wall_off, wall_tree, ncut, wall_pos, values, lo, hi = \
            _views(wi, wf)
        box, stamp, touched = mi[:nt], mi[nt:2 * nt], mi[2 * nt:3 * nt]
        x, acc = mf[:d], mf[d:d + 2]
        n = 0
        for a in range(d):
            v = _cast(t[a], f32, zt)
            old = x[a]
            if v == old:
                continue
            x[a] = v
            s, e = _wall_range(wall_off, wall_pos, a, min(old, v), max(old, v))
            for j in range(s, e):
                k = wall_tree[j]
                if stamp[k] != step:
                    stamp[k] = step
                    touched[n] = k
                    n += 1
        for q in range(n):
            k = touched[q]
            i = _locate(k, x, root, nax, ncut, nleft, nright, nb0, nb1, bucket, lo, hi)
            o = box[k]
            if i != o:
                box[k] = i
                for y in (values[i], -values[o]):  # Neumaier summation into acc = (sum, compensation)
                    s0 = acc[0]
                    s1 = s0 + y
                    if abs(s0) >= abs(y):
                        acc[1] += (s0 - s1) + y
                    else:
                        acc[1] += (y - s1) + s0
                    acc[0] = s1
        return n

    @njit(cache=True)
    def _walls_crossed(t, origin, wi, wf):
        """Number of walls between ``origin`` (cast) and ``t`` (with multiplicity over trees)."""
        d, f32, zt = wi[_D], wi[_F32], wf[0]
        wall_off = wi[wi[_O_WALL_OFF]:wi[_O_WALL_OFF] + d + 1]
        wall_pos = wf[wi[_O_WALL_POS]:wi[_O_WALL_POS] + wi[_NW]]
        total = 0
        for a in range(d):
            v = _cast(t[a], f32, zt)
            if v != origin[a]:
                s, e = _wall_range(wall_off, wall_pos, a, min(origin[a], v), max(origin[a], v))
                total += e - s
        return total


class NumbaWalk:
    """The walk compiled with numba; ``move(t)`` takes a float64 array."""

    def __init__(self, bound: BoundEnsemble, t: np.ndarray, plan: _Plan | None = None) -> None:
        if not HAVE_NUMBA:
            raise ImportError("numba is required")
        from .accel import _pack

        p = plan or _Plan(bound)
        self.plan = p
        d, nt = p.d, len(p.trees)
        nodes = np.array(p.nodes, dtype=np.float64).reshape(-1, 6)
        wall_off = np.zeros(d + 1, dtype=np.int64)
        wall_off[1:] = np.cumsum([len(w) for w in p.wall_pos])
        ens = bound.ensemble
        zt = ens.zero_threshold if ens.zero_threshold > 0 else -1.0
        ints, io = _pack([np.zeros(_HEADER), np.asarray(p.root, dtype=np.float64), nodes[:, 0], nodes[:, 2],
                          nodes[:, 3], nodes[:, 4], nodes[:, 5], np.asarray(p.bucket, dtype=np.float64),
                          wall_off.astype(np.float64),
                          np.asarray([k for w in p.wall_tree for k in w], dtype=np.float64)])
        floats, fo = _pack([np.array([zt]), nodes[:, 1].copy(), np.asarray([c for w in p.wall_pos for c in w], dtype=float),
                            p.values, np.ascontiguousarray(p.lo).ravel(), np.ascontiguousarray(p.hi).ravel()])
        wi = ints.astype(np.int64)
        wi[:_HEADER] = 0
        wi[_D], wi[_NT], wi[_F32] = d, nt, int(ens.input_dtype == np.float32)
        wi[_NN], wi[_NBK], wi[_NBX], wi[_NW] = len(nodes), len(p.bucket), len(p.values), int(wall_off[-1])
        wi[_O_ROOT:_O_WALL_TREE + 1] = io[1:]
        wi[_O_NCUT:_O_HI + 1] = fo[1:]
        self.wi, self.wf = wi, floats.astype(np.float64)
        self.f32, self.zt = int(wi[_F32]), float(zt)
        x = np.asarray(bound.ensemble.cast(np.asarray(t, dtype=float)), dtype=np.float64)
        box = np.array([p.locate(k, x) for k in range(nt)], dtype=np.int64)
        self.mi = np.concatenate([box, np.zeros(2 * nt, dtype=np.int64)])
        acc = [math.fsum([p.const] + [p.values[i] for i in box]), 0.0]
        self.mf = np.concatenate([x, acc, x, np.zeros(_STATE)])
        self.step = 0
        self.relocated = 0

    @property
    def x(self) -> np.ndarray:
        return self.mf[:self.plan.d]

    @property
    def box(self) -> np.ndarray:
        return self.mi[:len(self.plan.trees)]

    def clone(self) -> "NumbaWalk":
        other = NumbaWalk.__new__(NumbaWalk)
        other.__dict__.update(self.__dict__)
        other.mi, other.mf = self.mi.copy(), self.mf.copy()
        return other

    def move(self, t: np.ndarray) -> float:
        self.step += 1
        self.relocated += _move(t, self.step, self.wi, self.wf, self.mi, self.mf)
        d = self.plan.d
        return self.mf[d] + self.mf[d + 1]


# ---------------------------------------------------------------------------
# sessions: walk while moves are local, otherwise query the index
# ---------------------------------------------------------------------------
if HAVE_NUMBA:

    from .accel import _index_query

    @njit(cache=True)
    def _session_move(t, tau, wi, wf, mi, mf, il, fl, bm, work, mw):
        d, f32, zt = wi[_D], wi[_F32], wf[0]
        last, state = mf[d + 2:2 * d + 2], mf[2 * d + 2:2 * d + 2 + _STATE]
        k = _walls_crossed(t, last, wi, wf)
        for a in range(d):
            last[a] = _cast(t[a], f32, zt)
        state[2] = 0.9 * state[2] + 0.1 * k
        if state[1] == 1.0 and state[2] < 0.8 * tau:
            state[1] = 0.0  # back to walking: this move re-synchronises the walk
        elif state[1] == 0.0 and state[2] > 1.25 * tau:
            state[1] = 1.0
        if state[1] == 0.0:
            state[0] += 1
            state[3] += 1
            state[5] += _move(t, np.int64(state[0]), wi, wf, mi, mf)
            return mf[d] + mf[d + 1]
        state[4] += 1
        return _index_query(il, fl, bm, t, work, mw)


class Session:
    """Consecutive queries of one unit.  The session tracks how many walls recent moves cross (an
    exponential average over moves, measured between consecutive query points).  While moves are
    local (fewer walls than ``tau``, the price of one index query in tree re-locations) it walks
    the cell graph on every move, big or small; otherwise it queries the index on every move and
    leaves the walk behind, which re-synchronises when walking resumes.  Hysteresis (0.8 and 1.25
    times ``tau``) avoids switching back and forth.  ``tau`` is measured on the unit.  The session
    owns the walk it is given."""

    def __init__(self, index_nb, walk: NumbaWalk, t: np.ndarray, tau: float | None = None,
                 calibrate: np.ndarray | None = None) -> None:
        self.index, self.walk = index_nb, walk
        d = walk.plan.d
        self.state = walk.mf[2 * d + 2:2 * d + 2 + _STATE]
        self.state[:] = 0.0
        self.state[0] = walk.step  # stamps continue the walk's steps
        walk.mf[d + 2:2 * d + 2] = walk.x
        if tau is None:
            tau = self._calibrate(calibrate if calibrate is not None else np.asarray(t, dtype=np.float64)[None, :])
        self.tau = float(tau)
        ix = index_nb
        self._args = (walk.wi, walk.wf, walk.mi, walk.mf, ix.ints, ix.floats, ix.bm, ix.work, ix.mwork)

    def _calibrate(self, points: np.ndarray) -> float:
        import time

        index = self.index
        pts = np.ascontiguousarray(np.repeat(points, max(1, 200 // len(points)), axis=0)[:200], dtype=np.float64)
        index.query(pts[0])
        start = time.perf_counter()
        for t in pts:
            index.query(t)
        index_s = (time.perf_counter() - start) / len(pts)
        probe = self.walk.clone()
        probe.move(pts[-1])
        before = probe.relocated
        start = time.perf_counter()
        for t in pts:
            probe.move(t)
        per_tree = (time.perf_counter() - start) / max(1, probe.relocated - before)
        return index_s / per_tree

    @property
    def walked(self) -> int:
        return int(self.state[3])

    @property
    def queried(self) -> int:
        return int(self.state[4])

    def move(self, t: np.ndarray) -> float:
        return _session_move(t, self.tau, *self._args)
