#!/usr/bin/env python3
"""K-free semigroup point-enclosure index for tree ensembles in d dimensions.

Every reachable leaf of every tree is an axis-aligned box.  The ensemble
provenance at a point is the semiring sum of the payloads of all boxes that
contain the point (exactly one per tree).  Instead of enumerating the K
overlay cells, this index stores the N boxes in a (d-1)-level segment tree
whose last level is a centred interval tree with prefix/suffix aggregates
(the structure of `counterprovenance/tree.py`).  A query returns O(log^d N)
disjoint aggregate roots (pieces); their cached numeric values sum to the
counterfactual value.

Space is independent of the cell count K.  Exactness is checked against
direct ensemble evaluation (value and the multiset of selected leaves).
"""

from __future__ import annotations

import bisect
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO))

from counterprovenance.versioned import CircuitArena  # noqa: E402


# --------------------------------------------------------------------------
# synthetic axis-aligned regression trees on [0,1]^d
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LeafBox:
    tree: int
    leaf: int
    lo: tuple  # open lower bounds (-inf allowed)
    hi: tuple  # closed upper bounds (+inf allowed)
    value: float


def random_tree(rng: np.random.Generator, d: int, depth: int, tree_id: int) -> list[LeafBox]:
    leaves: list[LeafBox] = []

    def rec(lo: list, hi: list, level: int) -> None:
        if level == depth:
            leaves.append(LeafBox(tree_id, len(leaves), tuple(lo), tuple(hi), float(rng.normal())))
            return
        axis = int(rng.integers(d))
        a = lo[axis] if math.isfinite(lo[axis]) else 0.0
        b = hi[axis] if math.isfinite(hi[axis]) else 1.0
        theta = float(rng.uniform(a, b))
        hi2 = list(hi)
        hi2[axis] = theta
        rec(lo, hi2, level + 1)
        lo2 = list(lo)
        lo2[axis] = theta
        rec(lo2, hi, level + 1)

    rec([-math.inf] * d, [math.inf] * d, 0)
    return leaves


def random_ensemble(rng, d: int, depth: int, trees: int) -> list[LeafBox]:
    boxes: list[LeafBox] = []
    for m in range(trees):
        boxes.extend(random_tree(rng, d, depth, m))
    return boxes


class Ensemble:
    """Vectorised direct evaluation of an ensemble given as leaf boxes."""

    def __init__(self, boxes: list[LeafBox], d: int) -> None:
        self.boxes = boxes
        self.d = d
        self.LO = np.array([b.lo for b in boxes], dtype=float)
        self.HI = np.array([b.hi for b in boxes], dtype=float)
        self.values = np.array([b.value for b in boxes], dtype=float)
        self.tree_count = 1 + max(b.tree for b in boxes)

    def selected(self, x: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.all(self.LO < x, axis=1) & np.all(x <= self.HI, axis=1))

    def value(self, x: np.ndarray) -> float:
        return float(self.values[self.selected(x)].sum())


# --------------------------------------------------------------------------
# slab discretisation
# --------------------------------------------------------------------------
class Slabs:
    def __init__(self, boxes: list[LeafBox], d: int) -> None:
        self.d = d
        self.thresholds = []
        for a in range(d):
            vals = {b.lo[a] for b in boxes if math.isfinite(b.lo[a])}
            vals.update(b.hi[a] for b in boxes if math.isfinite(b.hi[a]))
            self.thresholds.append(sorted(vals))
        self.slab_counts = [len(t) + 1 for t in self.thresholds]

    def box_range(self, box: LeafBox, a: int) -> tuple[int, int]:
        th = self.thresholds[a]
        lo = 0 if not math.isfinite(box.lo[a]) else bisect.bisect_left(th, box.lo[a]) + 1
        hi = len(th) + 1 if not math.isfinite(box.hi[a]) else bisect.bisect_left(th, box.hi[a]) + 1
        return lo, hi

    def point_slab(self, x: np.ndarray) -> tuple[int, ...]:
        return tuple(bisect.bisect_left(self.thresholds[a], float(x[a])) for a in range(self.d))

    @property
    def grid_cells(self) -> int:
        out = 1
        for s in self.slab_counts:
            out *= s
        return out


# --------------------------------------------------------------------------
# the K-free index
# --------------------------------------------------------------------------
class _IntervalNode:
    __slots__ = ("center", "los", "his", "prefix_roots", "prefix_vals",
                 "suffix_roots", "suffix_vals", "left", "right")


class _SegNode:
    __slots__ = ("L", "R", "left", "right", "sub")


class KFreeIndex:
    def __init__(self, arena: CircuitArena, boxes: list[LeafBox], slabs: Slabs) -> None:
        self.arena = arena
        self.slabs = slabs
        self.d = slabs.d
        self.ranges = [[slabs.box_range(b, a) for a in range(self.d)] for b in boxes]
        self.roots = [arena.atom(("leaf", b.tree, b.leaf)) for b in boxes]
        self.values = [b.value for b in boxes]
        self.assignments = 0
        self.seg_nodes = 0
        self.interval_nodes = 0
        arena_before = arena.node_count
        items = list(range(len(boxes)))
        self.root = self._build_level(items, 0)
        self.arena_nodes = arena.node_count - arena_before

    # ---- construction -------------------------------------------------
    def _build_level(self, items: list[int], axis: int):
        if not items:
            return None
        if axis == self.d - 1:
            return self._build_interval(items, axis)
        return self._build_seg(items, axis, 0, self.slabs.slab_counts[axis])

    def _build_seg(self, items: list[int], axis: int, L: int, R: int):
        node = _SegNode()
        node.L, node.R = L, R
        self.seg_nodes += 1
        here: list[int] = []
        left_items: list[int] = []
        right_items: list[int] = []
        mid = (L + R) // 2
        for it in items:
            lo, hi = self.ranges[it][axis]
            if lo <= L and R <= hi:
                here.append(it)
            else:
                if lo < mid:
                    left_items.append(it)
                if hi > mid:
                    right_items.append(it)
        node.sub = self._build_level(here, axis + 1)
        node.left = node.right = None
        if R - L > 1:
            if left_items:
                node.left = self._build_seg(left_items, axis, L, mid)
            if right_items:
                node.right = self._build_seg(right_items, axis, mid, R)
        return node

    def _build_interval(self, items: list[int], axis: int):
        if not items:
            return None
        node = _IntervalNode()
        self.interval_nodes += 1
        mids = sorted((self.ranges[it][axis][0] + self.ranges[it][axis][1] - 1) // 2 for it in items)
        c = mids[len(mids) // 2]
        node.center = c
        here, left_items, right_items = [], [], []
        for it in items:
            lo, hi = self.ranges[it][axis]
            if hi <= c:
                left_items.append(it)
            elif lo > c:
                right_items.append(it)
            else:
                here.append(it)
        if not here:  # degenerate; fall back to a different centre
            c = self.ranges[items[0]][axis][0]
            node.center = c
            here, left_items, right_items = [], [], []
            for it in items:
                lo, hi = self.ranges[it][axis]
                if hi <= c:
                    left_items.append(it)
                elif lo > c:
                    right_items.append(it)
                else:
                    here.append(it)
        self.assignments += len(here)
        by_lo = sorted(here, key=lambda it: self.ranges[it][axis][0])
        by_hi = sorted(here, key=lambda it: self.ranges[it][axis][1])
        node.los = [self.ranges[it][axis][0] for it in by_lo]
        node.his = [self.ranges[it][axis][1] for it in by_hi]
        zero = self.arena.zero
        pr, pv = [zero], [0.0]
        for it in by_lo:
            pr.append(self.arena.add(pr[-1], self.roots[it]))
            pv.append(pv[-1] + self.values[it])
        sr, sv = [zero] * (len(by_hi) + 1), [0.0] * (len(by_hi) + 1)
        for i in range(len(by_hi) - 1, -1, -1):
            sr[i] = self.arena.add(self.roots[by_hi[i]], sr[i + 1])
            sv[i] = sv[i + 1] + self.values[by_hi[i]]
        node.prefix_roots, node.prefix_vals = pr, pv
        node.suffix_roots, node.suffix_vals = sr, sv
        node.left = self._build_interval(left_items, axis)
        node.right = self._build_interval(right_items, axis)
        return node

    # ---- queries ------------------------------------------------------
    def query(self, x: np.ndarray) -> list[tuple[int, float]]:
        slab = self.slabs.point_slab(x)
        pieces: list[tuple[int, float]] = []
        self._query_level(self.root, 0, slab, pieces)
        return pieces

    def _query_level(self, node, axis: int, slab, pieces) -> None:
        if node is None:
            return
        if axis == self.d - 1:
            self._query_interval(node, slab[axis], pieces)
            return
        x = slab[axis]
        while node is not None:
            if node.sub is not None:
                self._query_level(node.sub, axis + 1, slab, pieces)
            if node.R - node.L == 1:
                break
            mid = (node.L + node.R) // 2
            node = node.left if x < mid else node.right

    def _query_interval(self, node, x: int, pieces) -> None:
        zero = self.arena.zero
        while node is not None:
            if x < node.center:
                k = bisect.bisect_right(node.los, x)
                root = node.prefix_roots[k]
                if root != zero:
                    pieces.append((root, node.prefix_vals[k]))
                node = node.left
            else:
                k = bisect.bisect_right(node.his, x)
                root = node.suffix_roots[k]
                if root != zero:
                    pieces.append((root, node.suffix_vals[k]))
                node = node.right

    def atoms(self, root: int) -> list:
        """Expand an aggregate chain into its leaf atoms (for exactness checks)."""
        out = []
        stack = [root]
        while stack:
            n = stack.pop()
            node = self.arena.node(n)
            if node.op == "atom":
                out.append(node.label)
            elif node.op == "add":
                stack.extend(node.children)
            elif node.op == "zero":
                continue
            else:
                raise AssertionError(node.op)
        return out

    # ---- accounting ---------------------------------------------------
    @property
    def records(self) -> dict:
        return {
            "assignments": self.assignments,
            "sorted_keys": 2 * self.assignments,
            "aggregate_arena_nodes": self.arena_nodes,
            "numeric_cache": self.arena_nodes,
            "tree_nodes": self.seg_nodes + self.interval_nodes,
        }

    @property
    def total_records(self) -> int:
        return sum(self.records.values())


# --------------------------------------------------------------------------
# exact 2D overlay cell count by merging zero-change grid edges
# --------------------------------------------------------------------------
def exact_cells_2d(boxes: list[LeafBox], slabs: Slabs, tree_count: int) -> tuple[int, int]:
    sx, sy = slabs.slab_counts
    sig = np.full((tree_count, sx, sy), -1, dtype=np.int32)
    for b in boxes:
        (x0, x1), (y0, y1) = slabs.box_range(b, 0), slabs.box_range(b, 1)
        sig[b.tree, x0:x1, y0:y1] = b.leaf
    if np.any(sig < 0):
        raise AssertionError("leaf boxes do not tile the grid")
    same_x = np.all(sig[:, :-1, :] == sig[:, 1:, :], axis=0)  # (sx-1, sy)
    same_y = np.all(sig[:, :, :-1] == sig[:, :, 1:], axis=0)  # (sx, sy-1)
    parent = list(range(sx * sy))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in zip(*np.nonzero(same_x)):
        union(i * sy + j, (i + 1) * sy + j)
    for i, j in zip(*np.nonzero(same_y)):
        union(i * sy + j, i * sy + j + 1)
    components = len({find(v) for v in range(sx * sy)})
    return sx * sy, components


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def check_index(index: KFreeIndex, ens: Ensemble, slabs: Slabs, rng, n_queries: int, full_check: int):
    d = ens.d
    pts = rng.uniform(0.0, 1.0, size=(n_queries, d))
    # add points exactly on thresholds for boundary semantics
    for a in range(d):
        th = slabs.thresholds[a]
        if th:
            for _ in range(min(50, len(th))):
                p = rng.uniform(0.0, 1.0, size=d)
                p[a] = th[int(rng.integers(len(th)))]
                pts = np.vstack([pts, p])
    pieces_count = []
    t0 = time.perf_counter()
    max_err = 0.0
    for qi, x in enumerate(pts):
        pieces = index.query(x)
        pieces_count.append(len(pieces))
        val = sum(v for _, v in pieces)
        ref = ens.value(x)
        max_err = max(max_err, abs(val - ref))
        if qi < full_check:
            got = sorted(index.atoms(r) for r, _ in pieces)
            got_flat = sorted(a for chunk in got for a in chunk)
            exp = sorted(("leaf", ens.boxes[i].tree, ens.boxes[i].leaf) for i in ens.selected(x))
            if got_flat != exp:
                raise AssertionError(f"multiset mismatch at query {qi}")
            if len(got_flat) != ens.tree_count:
                raise AssertionError("did not select exactly one leaf per tree")
    elapsed = time.perf_counter() - t0
    if max_err > 1e-9:
        raise AssertionError(f"value mismatch {max_err}")
    return float(np.mean(pieces_count)), int(np.max(pieces_count)), elapsed / len(pts), max_err


def run(label: str, boxes: list[LeafBox], d: int, rng, n_queries=1000, full_check=200, overlay=False):
    ens = Ensemble(boxes, d)
    slabs = Slabs(boxes, d)
    arena = CircuitArena()
    t0 = time.perf_counter()
    index = KFreeIndex(arena, boxes, slabs)
    build = time.perf_counter() - t0
    mean_p, max_p, tq, err = check_index(index, ens, slabs, rng, n_queries, full_check)
    N = len(boxes)
    M = ens.tree_count
    K_grid = slabs.grid_cells
    rec = index.records
    line = (
        f"{label:<34} d={d} M={M:>5} N={N:>6} K_grid={K_grid:>14,} | kfree records={index.total_records:>9,} "
        f"(assign {rec['assignments']:,}, arena {rec['aggregate_arena_nodes']:,}, nodes {rec['tree_nodes']:,}) "
        f"build={build:.2f}s | pieces mean={mean_p:.1f} max={max_p} | {tq*1e6:.0f}us/q | max_err={err:.1e}"
    )
    extra = {}
    if overlay and d == 2 and K_grid <= 3_000_000:
        k_grid, k_exact = exact_cells_2d(boxes, slabs, M)
        extra["K_exact"] = k_exact
        line += f" | K_exact={k_exact:,} (grid/exact={k_grid/k_exact:.2f}x)"
    # cell-based reference costs (records): roots + cached values + persistent nodes
    cells_min = 2 * K_grid + int(1.5 * K_grid)  # node copying along a Hamiltonian path, ~1.5 rec/cell
    cells_path = 2 * K_grid + K_grid * max(1, math.ceil(math.log2(max(M, 2))))  # path copying, general position
    iip = 2 * K_grid + K_grid * (M - 1)
    line += f" | cells(grid): nodecopy~{cells_min:,} pathcopy~{cells_path:,} IIP~{iip:,}"
    print(line, flush=True)
    return index


def sklearn_boxes(model) -> tuple[list[LeafBox], int]:
    boxes: list[LeafBox] = []
    estimators = [row[0] for row in model.estimators_]
    for m, est in enumerate(estimators):
        t = est.tree_
        leaf_id = 0

        def rec(node: int, lo: list, hi: list) -> None:
            nonlocal leaf_id
            if t.children_left[node] == -1:
                boxes.append(LeafBox(m, leaf_id, tuple(lo), tuple(hi), float(t.value[node][0][0]) * model.learning_rate))
                leaf_id += 1
                return
            f, th = int(t.feature[node]), float(t.threshold[node])
            hi2 = list(hi)
            hi2[f] = min(hi[f], th)
            rec(t.children_left[node], lo, hi2)
            lo2 = list(lo)
            lo2[f] = max(lo[f], th)
            rec(t.children_right[node], lo2, hi)

        rec(0, [-math.inf, -math.inf], [math.inf, math.inf])
    return boxes, len(estimators)


def run_sklearn(sizes, seed: int) -> None:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        print("sklearn not available; skipping GBDT section")
        return
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(6000, 2))
    y = np.sin(6 * X[:, 0]) + 0.7 * np.cos(8 * X[:, 1]) + 1.2 * (X[:, 0] > X[:, 1]) + rng.normal(0, 0.03, len(X))
    for trees in sizes:
        model = GradientBoostingRegressor(n_estimators=trees, max_depth=2, learning_rate=0.08, random_state=seed)
        model.fit(X, y)
        boxes, M = sklearn_boxes(model)
        ens = Ensemble(boxes, 2)
        slabs = Slabs(boxes, 2)
        arena = CircuitArena()
        index = KFreeIndex(arena, boxes, slabs)
        # exactness against model.predict with sklearn's float32 input quantisation
        q = rng.uniform(0.0, 1.0, size=(2000, 2))
        q32 = q.astype(np.float32).astype(np.float64)
        pred = model.predict(q)
        base = float(model.init_.constant_[0][0]) if hasattr(model.init_, "constant_") else 0.0
        err = 0.0
        pieces = []
        for xi, x in enumerate(q32):
            ps = index.query(x)
            pieces.append(len(ps))
            err = max(err, abs(base + sum(v for _, v in ps) - pred[xi]))
        k_grid, k_exact = exact_cells_2d(boxes, slabs, M)
        print(
            f"{'sklearn GBDT depth2':<34} d=2 M={M:>5} N={len(boxes):>6} K_grid={k_grid:>14,} | kfree records={index.total_records:>9,} "
            f"| pieces mean={np.mean(pieces):.1f} max={max(pieces)} | max_err vs model.predict={err:.1e} "
            f"| K_exact={k_exact:,} (grid/exact={k_grid/k_exact:.2f}x)",
            flush=True,
        )


def main() -> None:
    import platform
    print(f"host={platform.node()} python={platform.python_version()} numpy={np.__version__}", flush=True)
    seed = 2028
    print("== 2D stumps (general position; same family as RQ1) ==")
    for M in (256, 512, 1024, 2048, 4096):
        rng = np.random.default_rng(seed + M)
        run(f"2D stumps M={M}", random_ensemble(rng, 2, 1, M), 2, rng, n_queries=500, full_check=100)
    print("== 2D depth-2 trees (conditional thresholds) ==")
    for M in (64, 256, 1024):
        rng = np.random.default_rng(seed + 10 + M)
        run(f"2D depth2 M={M}", random_ensemble(rng, 2, 2, M), 2, rng, n_queries=500, full_check=100, overlay=(M <= 256))
    print("== 2D depth-4 trees ==")
    for M in (64, 256):
        rng = np.random.default_rng(seed + 20 + M)
        run(f"2D depth4 M={M}", random_ensemble(rng, 2, 4, M), 2, rng, n_queries=300, full_check=50, overlay=(M <= 64))
    print("== 3D stumps and depth-2/3 trees (grid K explodes) ==")
    for M, depth in ((300, 1), (1000, 1), (3000, 1), (300, 2), (1000, 2), (300, 3)):
        rng = np.random.default_rng(seed + 30 + M + depth)
        run(f"3D depth{depth} M={M}", random_ensemble(rng, 3, depth, M), 3, rng, n_queries=300, full_check=50)
    print("== 4D stumps ==")
    for M in (400, 2000):
        rng = np.random.default_rng(seed + 40 + M)
        run(f"4D stumps M={M}", random_ensemble(rng, 4, 1, M), 4, rng, n_queries=200, full_check=50)
    print("== sklearn GradientBoostingRegressor (2D, depth 2) ==")
    run_sklearn((16, 32, 64, 128), seed + 400)


if __name__ == "__main__":
    main()
