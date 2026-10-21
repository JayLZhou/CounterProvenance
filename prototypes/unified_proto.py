#!/usr/bin/env python3
"""Real-model case study + unified (grouped, adaptive) index + persistent-last-axis K-free.

Pipeline on a real dataset (California housing, 8 features):
  1. fit a GBDT (sklearn GradientBoostingRegressor; HistGradientBoostingRegressor optional);
  2. fix one unit u; intervention set S (d features); bind the unit context by following
     every split on a non-intervention feature with u's own value -> each tree becomes an
     axis-aligned box partition of R^d (reachable leaves only);
  3. build four indexes and compare: (a) cell/grid materialization (only if K_grid is
     feasible), (b) K-free with interval-tree last axis [O(log^d N) pieces],
     (c) K-free with persistent last axis [O(log^{d-1} N) pieces],
     (d) grouped + adaptive unified index (axis-support groups, recurse until K_hat <= tau);
  4. exactness against model.predict, latency against per-query replay, and a provenance
     use case: exact per-tree contrastive attribution between two interventions.
"""

from __future__ import annotations

import bisect
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO))

from counterprovenance.versioned import CircuitArena  # noqa: E402
from counterprovenance.mcpst import BalancedPFTLayout  # noqa: E402
from kfree_proto import Ensemble, KFreeIndex, LeafBox, Slabs  # noqa: E402
from nodecopy_proto import NodeCopyPFT, payload_ref  # noqa: E402


# --------------------------------------------------------------------------
# unit binding: restrict sklearn trees to the intervention axes
# --------------------------------------------------------------------------
def restrict_gbr(model, unit: np.ndarray, S: list[int]) -> tuple[list[LeafBox], float]:
    """GradientBoostingRegressor -> boxes over the d=|S| intervention axes for unit `unit`."""
    boxes: list[LeafBox] = []
    axis_of = {f: a for a, f in enumerate(S)}
    d = len(S)
    lr = model.learning_rate
    for m, est in enumerate(model.estimators_[:, 0]):
        t = est.tree_
        leaf_counter = [0]

        def rec(node: int, lo: list, hi: list) -> None:
            if t.children_left[node] == -1:
                boxes.append(LeafBox(m, leaf_counter[0], tuple(lo), tuple(hi), float(t.value[node][0][0]) * lr))
                leaf_counter[0] += 1
                return
            f, th = int(t.feature[node]), float(t.threshold[node])
            if f in axis_of:
                a = axis_of[f]
                if lo[a] < th < hi[a]:
                    hi2 = list(hi); hi2[a] = th
                    rec(t.children_left[node], lo, hi2)
                    lo2 = list(lo); lo2[a] = th
                    rec(t.children_right[node], lo2, hi)
                elif th >= hi[a]:  # whole box goes left
                    rec(t.children_left[node], lo, hi)
                else:
                    rec(t.children_right[node], lo, hi)
            else:
                nxt = t.children_left[node] if unit[f] <= th else t.children_right[node]
                rec(nxt, lo, hi)

        rec(0, [-math.inf] * d, [math.inf] * d)
    base = float(np.asarray(model.init_.constant_).ravel()[0])
    return boxes, base


def restrict_hgbr(model, unit: np.ndarray, S: list[int]) -> tuple[list[LeafBox], float]:
    """HistGradientBoostingRegressor -> boxes (thresholds are bin edges, heavily shared)."""
    boxes: list[LeafBox] = []
    axis_of = {f: a for a, f in enumerate(S)}
    d = len(S)
    for m, preds in enumerate(model._predictors):
        nodes = preds[0].nodes
        leaf_counter = [0]

        def rec(i: int, lo: list, hi: list) -> None:
            n = nodes[i]
            if n["is_leaf"]:
                boxes.append(LeafBox(m, leaf_counter[0], tuple(lo), tuple(hi), float(n["value"])))
                leaf_counter[0] += 1
                return
            f, th = int(n["feature_idx"]), float(n["num_threshold"])
            if f in axis_of:
                a = axis_of[f]
                if lo[a] < th < hi[a]:
                    hi2 = list(hi); hi2[a] = th
                    rec(int(n["left"]), lo, hi2)
                    lo2 = list(lo); lo2[a] = th
                    rec(int(n["right"]), lo2, hi)
                elif th >= hi[a]:
                    rec(int(n["left"]), lo, hi)
                else:
                    rec(int(n["right"]), lo, hi)
            else:
                nxt = int(n["left"]) if unit[f] <= th else int(n["right"])
                rec(nxt, lo, hi)

        rec(0, [-math.inf] * d, [math.inf] * d)
    base = float(np.asarray(model._baseline_prediction).ravel()[0])
    return boxes, base


# --------------------------------------------------------------------------
# K-free with persistent last axis: one pre-materialized root per slab
# --------------------------------------------------------------------------
class _PSegNode:
    __slots__ = ("L", "R", "left", "right", "sub")


class _SlabChain:
    """Last axis: PFT over the node's intervals, node-copied along the slab chain."""

    def __init__(self, arena: CircuitArena, items: list[int], ranges, values, axis: int) -> None:
        self.items = items
        n = len(items)
        self.n_slots = max(n, 2)  # dummy inactive slot when n == 1
        self.layout = BalancedPFTLayout(self.n_slots)
        self.pft = NodeCopyPFT(self.layout, capacity=2)
        los = sorted({ranges[it][axis][0] for it in items} | {ranges[it][axis][1] for it in items})
        self.breaks = los  # slab boundaries (slab index space already integer)
        # elementary slabs of this node: between consecutive distinct endpoints
        self.slab_starts = los[:-1] if len(los) > 1 else los
        V = len(self.slab_starts)
        self.root_of = [0] * V
        self.values = [0.0] * V
        zero = -1
        # initial state at the first slab
        s0 = self.slab_starts[0]
        state = [payload_ref(j, 1) if ranges[it][axis][0] <= s0 < ranges[it][axis][1] else zero
                 for j, it in enumerate(items)]
        state.extend([zero] * (self.n_slots - n))
        self.root_of[0] = self.pft.build(state)
        val = sum(values[it] for j, it in enumerate(items) if state[j] != zero)
        self.values[0] = val
        # events: interval j starts at lo, ends at hi
        starts: dict[int, list[int]] = {}
        ends: dict[int, list[int]] = {}
        for j, it in enumerate(items):
            lo, hi = ranges[it][axis]
            starts.setdefault(lo, []).append(j)
            ends.setdefault(hi, []).append(j)
        pre = list(range(V))
        for v in range(1, V):
            s = self.slab_starts[v]
            self.root_of[v] = self.root_of[v - 1]
            for j in ends.get(s, []):
                if state[j] != zero:
                    state[j] = zero
                    val -= values[items[j]]
                    self.pft.update(v, self.root_of, pre[v], V, V, j, zero)
            for j in starts.get(s, []):
                if state[j] == zero:
                    state[j] = payload_ref(j, 1)
                    val += values[items[j]]
                    self.pft.update(v, self.root_of, pre[v], V, V, j, payload_ref(j, 1))
            self.values[v] = val
        self.records = self.pft.records + V  # nodes + entries + slab directory

    def query(self, x: int):
        if x < self.breaks[0] or x >= self.breaks[-1]:
            return None
        v = bisect.bisect_right(self.slab_starts, x) - 1
        return (self.root_of[v], v), self.values[v]

    def atoms(self, piece, labels) -> list:
        root, v = piece
        out = []
        for j, ref in enumerate(self.pft.decode(root, v)):
            if ref != -1 and j < len(self.items):
                out.append(labels[self.items[j]])
        return out


class KFreePersistent:
    def __init__(self, arena: CircuitArena, boxes: list[LeafBox], slabs: Slabs) -> None:
        self.arena = arena
        self.slabs = slabs
        self.d = slabs.d
        self.ranges = [[slabs.box_range(b, a) for a in range(self.d)] for b in boxes]
        self.values = [b.value for b in boxes]
        self.labels = [("leaf", b.tree, b.leaf) for b in boxes]
        self.seg_nodes = 0
        self.chain_records = 0
        self.root = self._build(list(range(len(boxes))), 0)

    def _build(self, items, axis):
        if not items:
            return None
        if axis == self.d - 1:
            chain = _SlabChain(self.arena, items, self.ranges, self.values, axis)
            self.chain_records += chain.records
            return chain
        return self._build_seg(items, axis, 0, self.slabs.slab_counts[axis])

    def _build_seg(self, items, axis, L, R):
        node = _PSegNode()
        node.L, node.R = L, R
        self.seg_nodes += 1
        here, li, ri = [], [], []
        mid = (L + R) // 2
        for it in items:
            lo, hi = self.ranges[it][axis]
            if lo <= L and R <= hi:
                here.append(it)
            else:
                if lo < mid:
                    li.append(it)
                if hi > mid:
                    ri.append(it)
        node.sub = self._build(here, axis + 1)
        node.left = node.right = None
        if R - L > 1:
            if li:
                node.left = self._build_seg(li, axis, L, mid)
            if ri:
                node.right = self._build_seg(ri, axis, mid, R)
        return node

    def query(self, x: np.ndarray):
        slab = self.slabs.point_slab(x)
        pieces = []
        self._q(self.root, 0, slab, pieces)
        return pieces

    def _q(self, node, axis, slab, pieces):
        if node is None:
            return
        if axis == self.d - 1:
            r = node.query(slab[axis])
            if r is not None:
                pieces.append((node, r[0], r[1]))
            return
        x = slab[axis]
        while node is not None:
            if node.sub is not None:
                self._q(node.sub, axis + 1, slab, pieces)
            if node.R - node.L == 1:
                break
            mid = (node.L + node.R) // 2
            node = node.left if x < mid else node.right

    @property
    def total_records(self) -> int:
        return self.chain_records + self.seg_nodes


# --------------------------------------------------------------------------
# grouped + adaptive unified index (axis-support groups; recurse until K_hat <= tau)
# --------------------------------------------------------------------------
def multidim_snake(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Boustrophedon order over a grid: consecutive cells are grid-adjacent."""
    order: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...], axis: int, flip: bool) -> None:
        if axis == len(shape):
            order.append(prefix)
            return
        rng = range(shape[axis]) if not flip else range(shape[axis] - 1, -1, -1)
        for i, idx in enumerate(rng):
            rec(prefix + (idx,), axis + 1, flip ^ (i % 2 == 1))

    rec((), 0, False)
    return order


class GridLeaf:
    """Leaf-level materialisation over the node's remaining axes.

    by_item=True : one PFT slot per box (active/inactive); needed inside segment-tree
                   nodes, where the boxes of one tree do not tile the node.
    by_item=False: one slot per tree (global cell baseline; trees tile the domain).
    Cells are visited in boustrophedon order and materialised by node copying.
    """

    def __init__(self, items, boxes, slabs: Slabs, axes: list[int], values, by_item: bool = True) -> None:
        self.axes = axes
        self.items = list(items)
        self.by_item = by_item
        self.th = []
        for a in axes:
            vals = set()
            for it in self.items:
                lo, hi = slabs.box_range(boxes[it], a)
                vals.add(lo); vals.add(hi)
            vals.discard(0)
            vals.discard(slabs.slab_counts[a])
            self.th.append(sorted(vals))
        shape = tuple(len(t) + 1 for t in self.th)
        self.shape = shape
        K = int(np.prod(shape))
        self.K = K
        if by_item:
            n_slots = len(self.items)
            sig = np.full((n_slots,) + shape, -1, dtype=np.int64)
            for j, it in enumerate(self.items):
                sl = self._local_slices(boxes[it], slabs)
                sig[(j,) + sl] = it
        else:
            trees = sorted({boxes[it].tree for it in self.items})
            pos = {t: i for i, t in enumerate(trees)}
            n_slots = len(trees)
            sig = np.full((n_slots,) + shape, -1, dtype=np.int64)
            for it in self.items:
                sl = self._local_slices(boxes[it], slabs)
                sig[(pos[boxes[it].tree],) + sl] = it
            if np.any(sig < 0):
                raise AssertionError("trees do not tile the grid (tree-slot mode)")
        if n_slots == 1:  # PFT needs height >= 1: add a dummy, always-inactive slot
            sig = np.concatenate([sig, np.full((1,) + shape, -1, dtype=np.int64)], axis=0)
            n_slots = 2
        order = multidim_snake(shape)
        self.cell_index = {c: i for i, c in enumerate(order)}
        self.layout = BalancedPFTLayout(n_slots)
        self.pft = NodeCopyPFT(self.layout, capacity=2)
        V = len(order)
        self.root_of = [0] * V
        self.cell_values = np.zeros(V)
        vals_arr = np.asarray(values, dtype=float)
        prev = None
        for v, cell in enumerate(order):
            state = sig[(slice(None),) + cell]
            active = state[state >= 0]
            self.cell_values[v] = float(vals_arr[active].sum()) if len(active) else 0.0
            if prev is None:
                self.root_of[v] = self.pft.build([int(x) for x in state])
            else:
                self.root_of[v] = self.root_of[v - 1]
                for j in np.flatnonzero(state != prev):
                    self.pft.update(v, self.root_of, v, V, V, int(j), int(state[j]))
            prev = state
        self.records = self.pft.records + V

    def _local_slices(self, box, slabs):
        sl = []
        for ai, a in enumerate(self.axes):
            lo, hi = slabs.box_range(box, a)
            i0 = bisect.bisect_right(self.th[ai], lo)
            i1 = bisect.bisect_right(self.th[ai], hi - 1) + 1
            sl.append(slice(i0, i1))
        return tuple(sl)

    def locate(self, slab) -> int:
        cell = tuple(bisect.bisect_right(self.th[ai], slab[a]) for ai, a in enumerate(self.axes))
        return self.cell_index[cell]

    def query(self, slab):
        v = self.locate(slab)
        return (self, self.root_of[v], v), float(self.cell_values[v])

    def atoms(self, piece, boxes):
        _, root, v = piece
        return [("leaf", boxes[it].tree, boxes[it].leaf) for it in self.pft.decode(root, v) if it >= 0]


class AdaptiveNode:
    __slots__ = ("axis", "L", "R", "left", "right", "sub", "leaf")


class UnifiedIndex:
    def __init__(self, boxes: list[LeafBox], slabs: Slabs, tau: int, group_budget: int | None = None,
                 axis_order: str = "given") -> None:
        self.boxes = boxes
        self.slabs = slabs
        self.d = slabs.d
        self.values = [b.value for b in boxes]
        self.axis_order = axis_order
        support: dict[int, set[int]] = {}
        for b in boxes:
            s = support.setdefault(b.tree, set())
            for a in range(self.d):
                if math.isfinite(b.lo[a]) or math.isfinite(b.hi[a]):
                    s.add(a)
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, b in enumerate(boxes):
            key = tuple(sorted(support[b.tree])) or (0,)
            groups.setdefault(key, []).append(i)
        if group_budget is not None:
            groups = self._merge_groups(groups, group_budget)
        self.groups = groups
        self.tau = tau
        self.leaf_count = 0
        self.seg_count = 0
        self.records = 0
        self.decisions: list[tuple[tuple[int, ...], int, int]] = []
        self.roots = {}
        for key, items in groups.items():
            axes = list(key)
            if axis_order == "bycount":  # segment-tree the axes with many thresholds first
                counts = {a: self._distinct(items, a) for a in axes}
                axes = sorted(axes, key=lambda a: -counts[a])
            self.roots[key] = self._build(items, axes, 0, depth=0)

    def _distinct(self, items, a) -> int:
        vals = set()
        for it in items:
            lo, hi = self.slabs.box_range(self.boxes[it], a)
            vals.add(lo); vals.add(hi)
        return max(1, len(vals) - 1)

    def _merge_groups(self, groups, budget):
        """Greedy: merge the group with the fewest trees into the smallest superset group
        (or into the union with the most similar group) until at most `budget` groups remain."""
        groups = {k: list(v) for k, v in groups.items()}

        def ntrees(items):
            return len({self.boxes[i].tree for i in items})

        while len(groups) > budget:
            small = min(groups, key=lambda k: ntrees(groups[k]))
            items = groups.pop(small)
            ss = set(small)
            supersets = [k for k in groups if ss <= set(k)]
            if supersets:
                target = min(supersets, key=lambda k: (len(k), -ntrees(groups[k])))
                groups[target].extend(items)
            else:  # merge with the group sharing most axes; key becomes the union
                best = max(groups, key=lambda k: (len(ss & set(k)), -len(set(k) | ss)))
                merged = tuple(sorted(set(best) | ss))
                other = groups.pop(best)
                groups.setdefault(merged, []).extend(other + items)
        return groups

    def _khat(self, items, axes) -> int:
        out = 1
        for a in axes:
            vals = set()
            for it in items:
                lo, hi = self.slabs.box_range(self.boxes[it], a)
                vals.add(lo); vals.add(hi)
            out *= max(1, len(vals) - 1)
        return out

    def _build(self, items, axes, ai, depth):
        node = AdaptiveNode()
        node.left = node.right = node.sub = node.leaf = None
        remaining = axes[ai:]
        if not items:
            return None
        if self._khat(items, remaining) <= self.tau or len(remaining) == 1:
            node.leaf = GridLeaf(items, self.boxes, self.slabs, remaining, self.values, by_item=True)
            self.leaf_count += 1
            self.records += node.leaf.records
            self.decisions.append((tuple(axes), depth, node.leaf.K))
            return node
        # segment tree on axis axes[ai]
        a = axes[ai]
        node.axis = a
        return self._build_seg(items, axes, ai, 0, self.slabs.slab_counts[a], depth)

    def _build_seg(self, items, axes, ai, L, R, depth):
        node = AdaptiveNode()
        node.axis = axes[ai]
        node.L, node.R = L, R
        node.left = node.right = node.leaf = None
        self.seg_count += 1
        self.records += 1
        here, li, ri = [], [], []
        mid = (L + R) // 2
        for it in items:
            lo, hi = self.slabs.box_range(self.boxes[it], node.axis)
            if lo <= L and R <= hi:
                here.append(it)
            else:
                if lo < mid:
                    li.append(it)
                if hi > mid:
                    ri.append(it)
        node.sub = self._build(here, axes, ai + 1, depth + 1) if here else None
        if R - L > 1:
            if li:
                node.left = self._build_seg(li, axes, ai, L, mid, depth)
            if ri:
                node.right = self._build_seg(ri, axes, ai, mid, R, depth)
        return node

    def query(self, x: np.ndarray):
        slab = self.slabs.point_slab(x)
        pieces = []
        for key, root in self.roots.items():
            self._q(root, slab, pieces)
        return pieces

    def _q(self, node, slab, pieces):
        if node is None:
            return
        if node.leaf is not None:
            pieces.append(node.leaf.query(slab))
            return
        x = slab[node.axis]
        while node is not None and node.leaf is None:
            if node.sub is not None:
                self._q(node.sub, slab, pieces)
            if node.R - node.L == 1:
                return
            mid = (node.L + node.R) // 2
            node = node.left if x < mid else node.right
        if node is not None and node.leaf is not None:
            pieces.append(node.leaf.query(slab))


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def load_data(data_home: str):
    from sklearn.datasets import fetch_california_housing
    ds = fetch_california_housing(data_home=data_home)
    return ds.data, ds.target, list(ds.feature_names)


def model_predict_rows(model, unit, S, pts):
    rows = np.tile(unit, (len(pts), 1))
    rows[:, S] = pts
    return model.predict(rows)


def f32(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def run_setting(label, model, kind, X, unit, S, names, rng, tau=4096, n_q=400, grid_limit=300_000):
    if kind == "gbr":
        unit = f32(unit)
    boxes, base = (restrict_gbr if kind == "gbr" else restrict_hgbr)(model, unit, S)
    d = len(S)
    ens = Ensemble(boxes, d)
    slabs = Slabs(boxes, d)
    N = len(boxes)
    M = ens.tree_count
    K_grid = slabs.grid_cells
    lo = np.percentile(X[:, S], 2, axis=0)
    hi = np.percentile(X[:, S], 98, axis=0)
    pts = rng.uniform(lo, hi, size=(n_q, d))
    if kind == "gbr":
        pts = f32(pts)
    ref = model_predict_rows(model, unit, S, pts)
    out = [f"[{label}] d={d} axes={[names[f] for f in S]} M={M} N={N:,} K_grid={K_grid:,}"]
    # coincidence stats
    th_mult = []
    for a in range(d):
        cnt: dict[float, int] = {}
        for b in boxes:
            if math.isfinite(b.hi[a]):
                cnt[b.hi[a]] = cnt.get(b.hi[a], 0) + 1
        if cnt:
            th_mult.extend(cnt.values())
    out.append(f"   thresholds: distinct per axis={slabs.slab_counts}, mean boxes sharing a threshold={np.mean(th_mult):.2f}, max={max(th_mult)}")
    # replay latency (per-row predict = what-if replay)
    t0 = time.perf_counter()
    for p in pts[:200]:
        model_predict_rows(model, unit, S, p[None, :])
    replay_us = (time.perf_counter() - t0) / 200 * 1e6
    res = {}
    # (b) K-free interval-tree
    arena = CircuitArena()
    t0 = time.perf_counter()
    kf = KFreeIndex(arena, boxes, slabs)
    b_kf = time.perf_counter() - t0
    t0 = time.perf_counter()
    err = 0.0; pcs = []
    for i, p in enumerate(pts):
        ps = kf.query(p); pcs.append(len(ps))
        err = max(err, abs(base + sum(v for _, v in ps) - ref[i]))
    q_kf = (time.perf_counter() - t0) / n_q * 1e6
    res["kfree_itree"] = (kf.total_records, np.mean(pcs), max(pcs), q_kf, err, b_kf)
    # (c) K-free persistent last axis
    arena2 = CircuitArena()
    t0 = time.perf_counter()
    kp = KFreePersistent(arena2, boxes, slabs)
    b_kp = time.perf_counter() - t0
    t0 = time.perf_counter()
    err2 = 0.0; pcs2 = []
    for i, p in enumerate(pts):
        ps = kp.query(p); pcs2.append(len(ps))
        err2 = max(err2, abs(base + sum(v for _, _, v in ps) - ref[i]))
    q_kp = (time.perf_counter() - t0) / n_q * 1e6
    # multiset check on a sample
    for i in rng.choice(n_q, size=min(30, n_q), replace=False):
        ps = kp.query(pts[i])
        got = sorted(a for node, piece, _ in ps for a in node.atoms(piece, kp.labels))
        exp = sorted(("leaf", boxes[j].tree, boxes[j].leaf) for j in ens.selected(pts[i]))
        if got != exp:
            raise AssertionError("persistent K-free multiset mismatch")
    res["kfree_persistent"] = (kp.total_records, np.mean(pcs2), max(pcs2), q_kp, err2, b_kp)
    # (d) unified grouped adaptive
    t0 = time.perf_counter()
    un = UnifiedIndex(boxes, slabs, tau=tau)
    b_un = time.perf_counter() - t0
    t0 = time.perf_counter()
    err3 = 0.0; pcs3 = []
    for i, p in enumerate(pts):
        ps = un.query(p); pcs3.append(len(ps))
        err3 = max(err3, abs(base + sum(v for _, v in ps) - ref[i]))
    q_un = (time.perf_counter() - t0) / n_q * 1e6
    for i in rng.choice(n_q, size=min(30, n_q), replace=False):
        ps = un.query(pts[i])
        got = sorted(a for piece, _ in ps for a in piece[0].atoms(piece, boxes))
        exp = sorted(("leaf", boxes[j].tree, boxes[j].leaf) for j in ens.selected(pts[i]))
        if got != exp:
            raise AssertionError("unified index multiset mismatch")
    res["unified"] = (un.records, np.mean(pcs3), max(pcs3), q_un, err3, b_un)
    groups_desc = ", ".join(f"{[names[S[a]] for a in k]}:{len({boxes[i].tree for i in v})}trees" for k, v in un.groups.items())
    depth_hist: dict[int, int] = {}
    for _, k, _ in un.decisions:
        depth_hist[k] = depth_hist.get(k, 0) + 1
    out.append(f"   groups (axes:#trees) = {groups_desc}")
    out.append(f"   adaptive: leaf nodes={un.leaf_count}, seg nodes={un.seg_count}, recursion-depth histogram={dict(sorted(depth_hist.items()))}, sum K_leaf={sum(K for _,_,K in un.decisions):,}")
    # (a) cell/grid approach
    if K_grid <= grid_limit:
        t0 = time.perf_counter()
        leaf = GridLeaf(list(range(N)), boxes, slabs, list(range(d)), [b.value for b in boxes], by_item=False)
        b_cell = time.perf_counter() - t0
        t0 = time.perf_counter()
        err4 = 0.0
        for i, p in enumerate(pts):
            _, v = leaf.query(slabs.point_slab(p))
            err4 = max(err4, abs(base + v - ref[i]))
        q_cell = (time.perf_counter() - t0) / n_q * 1e6
        res["cells"] = (leaf.records, 1.0, 1, q_cell, err4, b_cell)
    else:
        res["cells"] = None
    out.append(f"   replay (model.predict per what-if row): {replay_us:.0f} us/q")
    for name, r in res.items():
        if r is None:
            out.append(f"   {name:<17} INFEASIBLE (K_grid={K_grid:,} > {grid_limit:,})")
        else:
            rec, pm, px, q, e, b = r
            out.append(f"   {name:<17} records={rec:>10,} pieces mean/max={pm:5.1f}/{px:<3} query={q:7.1f} us build={b:6.2f}s max_err={e:.1e}")
    print("\n".join(out), flush=True)
    return boxes, base, slabs, kp


def contrastive_case(model, kind, X, unit, S, names, rng):
    """Provenance use: exact per-tree attribution between two what-if interventions."""
    if kind == "gbr":
        unit = f32(unit)
    boxes, base = (restrict_gbr if kind == "gbr" else restrict_hgbr)(model, unit, S)
    d = len(S)
    ens = Ensemble(boxes, d)
    slabs = Slabs(boxes, d)
    kp = KFreePersistent(CircuitArena(), boxes, slabs)
    lo = np.percentile(X[:, S], 5, axis=0); hi = np.percentile(X[:, S], 95, axis=0)
    t1 = rng.uniform(lo, hi); t2 = rng.uniform(lo, hi)
    if kind == "gbr":
        t1, t2 = f32(t1), f32(t2)
    f1 = model_predict_rows(model, unit, S, t1[None, :])[0]
    f2 = model_predict_rows(model, unit, S, t2[None, :])[0]
    leaves1 = {("leaf", boxes[j].tree, boxes[j].leaf): boxes[j].value for j in ens.selected(t1)}
    leaves2 = {("leaf", boxes[j].tree, boxes[j].leaf): boxes[j].value for j in ens.selected(t2)}
    # from the index pieces
    p1 = {a for node, piece, _ in kp.query(t1) for a in node.atoms(piece, kp.labels)}
    p2 = {a for node, piece, _ in kp.query(t2) for a in node.atoms(piece, kp.labels)}
    assert p1 == set(leaves1) and p2 == set(leaves2)
    by_tree: dict[int, float] = {}
    for a, v in leaves2.items():
        by_tree[a[1]] = by_tree.get(a[1], 0.0) + v
    for a, v in leaves1.items():
        by_tree[a[1]] = by_tree.get(a[1], 0.0) - v
    changed = {t: v for t, v in by_tree.items() if abs(v) > 0}
    total = sum(changed.values())
    top = sorted(changed.items(), key=lambda kv: -abs(kv[1]))[:5]
    print(f"[contrast] unit prediction f(t1)={f1:.4f}, f(t2)={f2:.4f}, delta={f2-f1:.4f}; "
          f"trees changed leaf: {len(changed)}/{ens.tree_count}; sum of per-tree deltas={total:.4f} (exact match: {abs(total-(f2-f1))<1e-9})")
    print(f"[contrast] top-5 trees by |delta|: {[(t, round(v,4)) for t,v in top]}; "
          f"t1={np.round(t1,3).tolist()} t2={np.round(t2,3).tolist()} axes={[names[f] for f in S]}", flush=True)


def main() -> None:
    import platform
    print(f"host={platform.node()} python={platform.python_version()} numpy={np.__version__}", flush=True)
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    data_home = sys.argv[2] if len(sys.argv) > 2 else None
    X, y, names = load_data(data_home)
    print(f"California housing: {X.shape[0]} rows, features={names}", flush=True)
    rng = np.random.default_rng(2028)
    unit_idx = int(rng.integers(len(X)))
    unit = X[unit_idx].copy()
    print(f"fixed unit #{unit_idx}: {dict(zip(names, np.round(unit,3)))}", flush=True)

    t0 = time.perf_counter()
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=0).fit(X, y)
    print(f"GBR(300 trees, depth 3) fit in {time.perf_counter()-t0:.1f}s, R2={gbr.score(X,y):.3f}", flush=True)
    t0 = time.perf_counter()
    hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.1, random_state=0).fit(X, y)
    print(f"HGB(300 iters, depth 3, 255 bins) fit in {time.perf_counter()-t0:.1f}s, R2={hgb.score(X,y):.3f}", flush=True)

    # importance-ordered features for interventions
    order = list(np.argsort(-gbr.feature_importances_))
    print(f"feature importance order: {[names[i] for i in order]}", flush=True)

    import os
    dims = (2, 3) if os.environ.get("QUICK") else (2, 3, 4, 6, 8)
    print("== GradientBoostingRegressor (exact splits, near general position) ==")
    for d in dims:
        S = sorted(order[:d])
        run_setting(f"GBR d={d}", gbr, "gbr", X, unit, S, names, rng)
    print("== HistGradientBoostingRegressor (255 shared bin edges: coincident thresholds) ==")
    for d in dims:
        S = sorted(order[:d])
        try:
            run_setting(f"HGB d={d}", hgb, "hgb", X, unit, S, names, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"[HGB d={d}] skipped: {exc!r}", flush=True)
    print("== provenance use case: exact per-tree contrastive attribution (GBR, d=3) ==")
    contrastive_case(gbr, "gbr", X, unit, sorted(order[:3]), names, rng)
    print("== tau sensitivity of the adaptive rule (GBR, d=4) ==")
    S = sorted(order[:4])
    boxes, base = restrict_gbr(gbr, f32(unit), S)
    slabs = Slabs(boxes, 4)
    for tau in (256, 1024, 4096, 16384, 65536):
        t0 = time.perf_counter()
        un = UnifiedIndex(boxes, slabs, tau=tau)
        b = time.perf_counter() - t0
        lo = np.percentile(X[:, S], 2, axis=0); hi = np.percentile(X[:, S], 98, axis=0)
        pts = rng.uniform(lo, hi, size=(200, 4))
        pcs = [len(un.query(p)) for p in pts]
        print(f"tau={tau:>6}: records={un.records:>9,} leaf nodes={un.leaf_count:>5} pieces mean/max={np.mean(pcs):5.1f}/{max(pcs)} build={b:.2f}s", flush=True)


if __name__ == "__main__":
    main()
