"""The provenance index: one structure, two parameters.

``ProvenanceIndex`` answers ``do(T = t)`` for a fixed unit with the exact
counterfactual value and its exact semiring provenance.  It is built from:

1. **groups** - trees are grouped by the intervention axes they split on
   (their *axis support*), optionally merged down to ``group_budget`` groups;
   a group's state space is the overlay of its own trees only;
2. **adaptive segment tree** - inside a group, if the overlay of the remaining
   axes is small (``K_hat <= tau``) it is materialised as one MCPST leaf;
   otherwise a segment tree is built on the axis with the most boundaries and
   every node recurses on the remaining axes with the boxes assigned to it;
3. **MCPST leaves** (:mod:`cellindex`) - exact overlay, MST construction plan,
   persistent factor tree with one directly addressable root per face.

A query returns one pre-materialised root ("piece") per visited leaf.  The two
parameters move the index along the space / pieces trade-off: ``tau = inf``
with one group is a single MCPST over the whole overlay (one piece, Theta(K)
faces); ``tau = 0`` recurses to the last axis, whose leaves are 1D persistent
chains (``O(N log^{d-1} N)`` space, ``O(log^{d-1} N)`` pieces per group).
:func:`autotune` picks the parameters at build time from sample queries.

``grouping="auto"`` (used in the experiments) picks one of three configurations:
one cell graph for the whole forest when the slab grid of all axes has at most
``one_cells`` cells ("one"); otherwise the greedy groups below ("tables") or leaf
bitmasks for every tree ("masks"), whichever a per-query cost model
(``cost_lookup`` per table read, ``cost_masks_fixed`` + ``cost_word`` per mask
word) says is cheaper, decided from the grouping before any table is built.

Configuration ``grouping="greedy"``, ``tau=inf``: groups are merged
greedily under a total table budget (:mod:`grouping`), every group is one MCPST
leaf, and trees whose own table exceeds ``grid_cap``, or ``mask_ratio`` times
their number of leaf boxes, are answered by leaf bitmasks.  When every group is a leaf the query is *compiled*:

* packed lookup - per axis and slab, one Python int holds the contribution
  ``rank * stride`` of that slab to every group's cell number in a 32-bit
  field, so ``d`` additions give all cell numbers at once and the values are
  gathered at C speed;
* leaf bitmasks - per axis and slab, one int holds, for every deep tree, the
  leaves whose box contains the slab (one 64-bit field per tree); the AND over
  the axes leaves exactly one bit per tree.

Provenance is expanded on demand from the cell numbers (MCPST roots) and the
bitmask leaves (atoms), see :class:`LazyAnswer`.
"""

from __future__ import annotations

import math
import time
from array import array
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from itertools import chain
from operator import getitem
from typing import Iterable, Sequence

import numpy as np

from .baselines import make_cast
from .cellindex import CellIndex, Piece
from .ensemble import BoundEnsemble, Slabs
from .grouping import greedy_groups, tree_profiles
from .semiring import CircuitArena, NodeId

_FIELD = 0xFFFFFFFF


@dataclass
class Answer:
    value: float
    pieces: list[Piece]
    bound: BoundEnsemble = field(repr=False)

    def boxes(self) -> list[int]:
        out: list[int] = []
        for piece in self.pieces:
            out.extend(piece.boxes())
        return out

    def labels(self) -> Counter:
        """Multiset of provenance atoms ``(tree, native leaf id)``."""
        return Counter(self.bound.label(b) for b in self.boxes())

    def to_arena(self, arena: CircuitArena) -> NodeId:
        roots = [p.to_arena(arena, self.bound.label) for p in self.pieces]
        return arena.add_all(roots)


@dataclass(frozen=True)
class AtomPiece:
    """One leaf box of a bitmask tree: a provenance atom."""

    box: int
    value: float

    def boxes(self) -> list[int]:
        return [self.box]

    def to_arena(self, arena: CircuitArena, label) -> NodeId:
        return arena.atom(label(self.box))


class LazyAnswer:
    """Result of a compiled query: the exact value now, the provenance pieces on
    demand (from the per-group cell numbers and the bitmask leaves)."""

    __slots__ = ("value", "_index", "_cells", "_leaves")

    def __init__(self, value: float, index: "ProvenanceIndex", cells, leaves) -> None:
        self.value = value
        self._index = index
        self._cells = cells
        self._leaves = leaves

    @property
    def bound(self) -> BoundEnsemble:
        return self._index.bound

    @property
    def cells(self) -> list[int]:
        """Cell number of every group (the raw provenance handle)."""
        return list(self._cells)

    @property
    def pieces(self) -> list:
        idx = self._index
        out: list = []
        for leaf in idx._const_leaves:
            if not leaf.cells.face_empty[0]:
                out.append(Piece(leaf.cells, 0))
        for leaf, cell in zip(idx._fast_leaves, self._cells):
            cells = leaf.cells
            face = int(cells.overlay.face_of_cell[cell])
            if not cells.face_empty[face]:
                out.append(Piece(cells, face))
        for items, values, leaf in zip(idx._bitems, idx._bvals, self._leaves):
            if leaf:
                out.append(AtomPiece(items[leaf], values[leaf]))
        return out

    def boxes(self) -> list[int]:
        out: list[int] = []
        for piece in self.pieces:
            out.extend(piece.boxes())
        return out

    def labels(self) -> Counter:
        return Counter(self.bound.label(b) for b in self.boxes())

    def to_arena(self, arena: CircuitArena) -> NodeId:
        return arena.add_all([p.to_arena(arena, self.bound.label) for p in self.pieces])


class _SegNode:
    __slots__ = ("axis", "lo", "hi", "left", "right", "sub")

    def __init__(self, axis: int, lo: int, hi: int) -> None:
        self.axis, self.lo, self.hi = axis, lo, hi
        self.left = self.right = self.sub = None


class _Leaf:
    __slots__ = ("axes", "cells")

    def __init__(self, axes: tuple[int, ...], cells: CellIndex) -> None:
        self.axes, self.cells = axes, cells


@dataclass
class IndexStats:
    groups: int
    leaves: int
    segment_nodes: int
    assignments: int
    faces: int
    grid_cells: int
    pft_nodes: int
    pft_entries: int
    locator_words: int
    build_seconds: float
    leaf_depths: dict
    bitmask_trees: int = 0
    bitmask_words: int = 0   # 64-bit words of the per-slab leaf masks
    table_cells: int = 0     # dense value tables of the compiled query
    pack_words: int = 0      # 32-bit fields of the per-slab packed contributions

    @property
    def records(self) -> int:
        """PFT nodes + entries + root and value per face + locators + item tables
        + segment-tree nodes + compiled query tables + bitmask words."""
        return (
            self.pft_nodes
            + self.pft_entries
            + 2 * self.faces
            + self.locator_words
            + self.assignments
            + self.segment_nodes
            + self.table_cells
            + self.pack_words
            + self.bitmask_words
        )

    @property
    def packed_bytes(self) -> int:
        """Estimated packed layout: 8 B per PFT node (two int32 children), 12 B per
        entry, 16 B per face (root, version, float64 value), 4 B per locator word
        and item reference, 16 B per segment-tree node, 8 B per table cell,
        4 B per packed field, 8 B per bitmask word."""
        return (
            8 * self.pft_nodes
            + 12 * self.pft_entries
            + 16 * self.faces
            + 4 * (self.locator_words + self.assignments)
            + 16 * self.segment_nodes
            + 8 * self.table_cells
            + 4 * self.pack_words
            + 8 * self.bitmask_words
        )


class ProvenanceIndex:
    def __init__(
        self,
        bound: BoundEnsemble,
        tau: float = math.inf,
        group_budget: int | None = None,
        grouping: str = "greedy",
        budget: int = 2_000_000,
        grid_cap: int = 1 << 22,
        mask_ratio: float = 64.0,
        axis_order: str = "bycount",
        layout: str = "hybrid",
        exact: bool = True,
        alpha: float = 1.5,
        max_leaf_cells: int = 20_000_000,
        compile: bool = True,
        one_cells: int = 2_000_000,
        min_group_trees: int = 1,
        cost_lookup: float = 0.15,
        cost_masks_fixed: float = 8.0,
        cost_word: float = 0.02,
    ) -> None:
        """``grouping``: ``"auto"`` (one cell graph for the whole forest when the slab grid of all
        axes has at most ``one_cells`` cells; otherwise the greedy cell graphs or leaf bitmasks for
        every tree, whichever is cheaper per query under the cost model ``cost_lookup`` per table
        read, ``cost_masks_fixed`` + ``cost_word`` per mask word, in microseconds), ``"greedy"`` (merge groups under a total table budget ``budget``; trees whose own
        table exceeds ``grid_cap`` or ``mask_ratio`` times their leaf boxes use leaf bitmasks),
        ``"support"`` (one group per axis support) or ``"none"``."""
        t0 = time.perf_counter()
        self.bound = bound
        self.slabs = Slabs(bound)
        self.choice = None
        if grouping == "auto":
            grid = math.prod(self.slabs.counts)
            if grid <= one_cells:
                grouping, max_leaf_cells, self.choice = "none", max(max_leaf_cells, grid), "one"
            else:
                # greedy cell graphs, or leaf masks for every tree: whichever the cost model says is
                # cheaper per query (decided from the grouping alone, before any table is built)
                grouping = "greedy"
                profiles = tree_profiles(bound, self.slabs)
                groups, deep = greedy_groups(profiles, budget, grid_cap, mask_ratio=mask_ratio)
                words = lambda ps: sum((len(p.items) + 63) // 64 for p in ps)
                cost_tables = cost_lookup * len(groups) + (cost_masks_fixed + cost_word * words(deep) if deep else 0.0)
                cost_masks = cost_masks_fixed + cost_word * words(profiles)
                if cost_masks < cost_tables:
                    mask_ratio, grid_cap, self.choice = 0.0, 0, "masks"
                else:
                    self.choice = "tables"
        self.tau = tau
        self.budget = budget
        self.grid_cap = grid_cap
        self.mask_ratio = mask_ratio
        self.min_group_trees = min_group_trees
        self.layout = layout
        self.exact = exact
        self.alpha = alpha
        self.axis_order = axis_order
        self.max_leaf_cells = max_leaf_cells
        self._segment_nodes = 0
        self._leaves: list[_Leaf] = []
        self._leaf_depths: Counter = Counter()
        self._cast = make_cast(bound.ensemble)
        self.deep: list = []
        groups = self._groups(grouping)
        if group_budget is not None:
            merged: dict[tuple[int, ...], list[int]] = {}
            for axes, items in groups:
                merged.setdefault(axes, []).extend(items)
            groups = list(_merge_groups(merged, group_budget, bound).items())
        self.groups = groups
        self.roots = [self._build(items, tuple(axes), 0) for axes, items in groups]
        self._build_bitmasks()
        self._fast = compile and all(r is None or isinstance(r, _Leaf) for r in self.roots)
        self._fast_leaves: list[_Leaf] = []
        self._const_leaves: list[_Leaf] = []
        self._tables: list = []
        self._table_cells = self._pack_words = 0
        if self._fast:
            self._compile()
        self.build_seconds = time.perf_counter() - t0

    # -- grouping -----------------------------------------------------------
    def _groups(self, grouping: str) -> list[tuple[tuple[int, ...], list[int]]]:
        if grouping == "none":
            return [(tuple(range(self.bound.d)), list(range(len(self.bound.boxes))))]
        if grouping == "greedy":
            return self._greedy_groups()
        if grouping != "support":
            raise ValueError(f"unknown grouping {grouping!r}")
        support: dict[int, set[int]] = {}
        for b in self.bound.boxes:
            s = support.setdefault(b.tree, set())
            for a in range(self.bound.d):
                if math.isfinite(b.lo[a]) or math.isfinite(b.hi[a]):
                    s.add(a)
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, b in enumerate(self.bound.boxes):
            groups.setdefault(tuple(sorted(support[b.tree])), []).append(i)
        return list(groups.items())

    def _greedy_groups(self) -> list[tuple[tuple[int, ...], list[int]]]:
        profiles = tree_profiles(self.bound, self.slabs)
        table_groups, deep = greedy_groups(profiles, self.budget, self.grid_cap,
                                             mask_ratio=self.mask_ratio)
        # a cell graph must pay for its lookup: a group with fewer than min_group_trees trees costs
        # more to read than the leaf masks of its trees, so its trees use leaf masks instead
        by_tree = {p.tree: p for p in profiles}
        small = [g for g in table_groups if len(g.trees) < self.min_group_trees]
        table_groups = [g for g in table_groups if len(g.trees) >= self.min_group_trees]
        deep = deep + [by_tree[m] for g in small for m in g.trees]
        out = [(tuple(sorted(g.axes)), sorted(g.items)) for g in table_groups]
        self.deep = deep
        varying = {p.tree for p in profiles}
        constant = [i for i, b in enumerate(self.bound.boxes) if b.tree not in varying]
        if constant:
            out.append(((), constant))
        return out

    # -- leaf bitmasks for deep trees --------------------------------------
    def _build_bitmasks(self) -> None:
        """One 64-bit field per 64 leaves of a deep tree.  After the AND over the
        axes exactly one field of each tree holds one bit; the other fields of
        that tree are zero and read the dummy entry 0 (value 0.0, no box)."""
        deep = self.deep
        self._bitems: list[list] = []
        self._bvals: list[list[float]] = []
        self._baxes: list = []
        self._bmasks: dict[int, np.ndarray] = {}
        self._bitmask_words = 0
        fields = [p.items[s:s + 64] for p in deep for s in range(0, len(p.items), 64)]
        self._bbytes = 8 * len(fields)
        if not fields:
            return
        boxes, ranges, counts = self.bound.boxes, self.slabs.ranges, self.slabs.counts
        used = sorted(set().union(*(p.axes for p in deep)))
        masks = {a: np.zeros((counts[a], len(fields)), dtype=np.uint64) for a in used}
        for k, items in enumerate(fields):
            self._bitems.append([None] + list(items))  # indexed by bit_length
            self._bvals.append([0.0] + [boxes[i].value for i in items])
            for j, i in enumerate(items):
                bit = np.uint64(1 << j)
                for a in used:
                    lo, hi = ranges[i][a]
                    masks[a][lo:hi, k] |= bit
        th = [list(map(float, t)) for t in self.slabs.thresholds]
        for a in used:
            rows = [int.from_bytes(r.tobytes(), "little") for r in masks[a]]
            self._baxes.append((a, th[a], rows))
        self._bmasks = masks
        self._bitmask_words = sum(counts[a] for a in used) * len(fields)
        # vectorised read of the leaves: per axis the (slabs, fields) array, and the leaf values
        # as one array indexed by field * 65 + bit_length (bit_length 0 reads the dummy 0.0)
        self._bnp = [(th[a], masks[a]) for a in used]
        flat = np.zeros((len(fields), 65))
        for k, vals in enumerate(self._bvals):
            flat[k, :len(vals)] = vals
        self._bflat = flat.ravel()
        self._bbase = np.arange(len(fields), dtype=np.int64) * 65

    def _bitmask_np(self, x: Sequence[float]) -> tuple[np.ndarray, float]:
        """Leaf of every field (``bit_length`` of the remaining bit, as the ``frexp`` exponent)
        and the sum of their values, with one numpy AND per axis."""
        it = iter(self._bnp)
        th, M = next(it)
        m = M[bisect_left(th, x[self._baxes[0][0]])].copy()
        for (a, _, _), (th, M) in zip(self._baxes[1:], it):
            m &= M[bisect_left(th, x[a])]
        leaves = np.frexp(m.astype(np.float64))[1]
        return leaves, float(self._bflat[self._bbase + leaves].sum())

    def _bitmask_leaves(self, x: Sequence[float]) -> list[int]:
        it = iter(self._baxes)
        a, th, rows = next(it)
        mask = rows[bisect_left(th, x[a])]
        for a, th, rows in it:
            mask &= rows[bisect_left(th, x[a])]
        return list(map(int.bit_length, memoryview(mask.to_bytes(self._bbytes, "little")).cast("Q")))

    # -- compiled query ----------------------------------------------------
    def _compile(self) -> None:
        present = [r for r in self.roots if r is not None]
        # groups without intervention axes (constant trees) fold into the constant
        self._const_leaves = [leaf for leaf in present if not leaf.axes]
        leaves = [leaf for leaf in present if leaf.axes]
        self._fast_leaves = leaves
        d, counts = self.bound.d, self.slabs.counts
        th = [list(map(float, t)) for t in self.slabs.thresholds]
        rows = [[0] * counts[a] for a in range(d)]
        self._contrib: list[tuple[int, int, np.ndarray]] = []  # (group, axis, rank*stride per slab)
        groups_of_axis: dict[int, list[int]] = {}
        for gi, leaf in enumerate(leaves):
            cells, overlay = leaf.cells, leaf.cells.overlay
            dense = np.ascontiguousarray(cells.face_value[overlay.face_of_cell], dtype=np.float64)
            table = array("d")
            table.frombytes(dense.tobytes())
            self._tables.append(table)
            self._table_cells += len(table)
            shift = 32 * gi
            for k, a in enumerate(leaf.axes):
                if not overlay.cuts[k]:
                    continue
                contrib = np.searchsorted(np.asarray(overlay.cuts[k]), np.arange(counts[a]), side="right")
                contrib = contrib.astype(np.int64) * int(overlay.strides[k])
                self._contrib.append((gi, a, contrib))
                groups_of_axis.setdefault(a, []).append(gi)
                row = rows[a]
                for s in np.flatnonzero(contrib).tolist():
                    row[s] += int(contrib[s]) << shift
        self._paxes = [(a, th[a], rows[a]) for a in range(d) if any(rows[a])]
        self._groups_of_axis = groups_of_axis
        self._nbytes = 4 * len(leaves)
        self._pack_words = sum(counts[a] for a, _, _ in self._paxes) * len(leaves)
        self._const = math.fsum([self.bound.offset] + [float(leaf.cells.face_value[0]) for leaf in self._const_leaves])

    def _query_fast(self, t: Sequence[float]) -> LazyAnswer:
        x = self._cast(t)
        packed = 0
        for a, th, rows in self._paxes:
            packed += rows[bisect_left(th, x[a])]
        cells = memoryview(packed.to_bytes(self._nbytes, "little")).cast("I")
        if self._baxes:
            leaves, deep = self._bitmask_np(x)
            value = math.fsum(chain((self._const, deep), map(getitem, self._tables, cells)))
        else:
            leaves = ()
            value = math.fsum(chain((self._const,), map(getitem, self._tables, cells)))
        return LazyAnswer(value, self, cells, leaves)

    def session(self, t: Sequence[float]) -> "QuerySession":
        """Incremental queries that change one intervention value at a time."""
        if not self._fast:
            raise ValueError("sessions need a compiled index (tau = inf)")
        return QuerySession(self, t)

    # -- construction -------------------------------------------------------
    def _boundaries(self, items: Sequence[int], a: int) -> int:
        cuts = set()
        top = self.slabs.counts[a]
        for i in items:
            lo, hi = self.slabs.ranges[i][a]
            if lo > 0:
                cuts.add(lo)
            if hi < top:
                cuts.add(hi)
        return len(cuts)

    def _build(self, items: list[int], axes: tuple[int, ...], depth: int):
        if not items:
            return None
        counts = {a: self._boundaries(items, a) for a in axes}
        k_hat = 1
        for a in axes:
            k_hat *= counts[a] + 1
        if len(axes) <= 1 or k_hat <= self.tau:
            if k_hat > self.max_leaf_cells:
                raise MemoryError(f"leaf overlay of {k_hat:,} grid cells exceeds max_leaf_cells")
            return self._leaf(items, axes, depth)
        a = max(axes, key=lambda x: (counts[x], -x)) if self.axis_order == "bycount" else axes[0]
        rest = tuple(x for x in axes if x != a)
        return self._segment(items, a, rest, 0, self.slabs.counts[a], depth)

    def _segment(self, items, a, rest, lo, hi, depth):
        node = _SegNode(a, lo, hi)
        self._segment_nodes += 1
        here, left, right = [], [], []
        mid = (lo + hi) // 2
        for i in items:
            ilo, ihi = self.slabs.ranges[i][a]
            if ilo <= lo and hi <= ihi:
                here.append(i)
            else:
                if ilo < mid:
                    left.append(i)
                if ihi > mid:
                    right.append(i)
        node.sub = self._build(here, rest, depth + 1) if here else None
        if hi - lo > 1:
            if left:
                node.left = self._segment(left, a, rest, lo, mid, depth)
            if right:
                node.right = self._segment(right, a, rest, mid, hi, depth)
        return node

    def _leaf(self, items, axes, depth):
        boxes = self.bound.boxes
        cells = CellIndex(
            items=items,
            ranges=[[self.slabs.ranges[i][a] for a in axes] for i in items],
            slots=[boxes[i].tree for i in items],
            slab_counts=[self.slabs.counts[a] for a in axes],
            values=[boxes[i].value for i in items],
            offset=0.0,
            layout=self.layout,
            exact=self.exact,
            alpha=self.alpha,
        )
        leaf = _Leaf(tuple(axes), cells)
        self._leaves.append(leaf)
        self._leaf_depths[depth] += 1
        return leaf

    # -- queries ------------------------------------------------------------
    def query(self, t: Sequence[float]) -> Answer | LazyAnswer:
        if self._fast:
            return self._query_fast(t)
        x = self.bound.prepare(t)
        slab = self.slabs.locate(x)
        pieces: list = []
        for root in self.roots:
            self._collect(root, slab, pieces)
        if self._baxes:
            leaves = self._bitmask_leaves(self._cast(t))
            for items, values, leaf in zip(self._bitems, self._bvals, leaves):
                if leaf:
                    pieces.append(AtomPiece(items[leaf], values[leaf]))
        total = math.fsum([self.bound.offset] + [p.value for p in pieces])
        return Answer(total, pieces, self.bound)

    def _collect(self, node, slab, pieces) -> None:
        while node is not None:
            if isinstance(node, _Leaf):
                piece = node.cells.piece([slab[a] for a in node.axes])
                if not node.cells.face_empty[piece.face]:
                    pieces.append(piece)
                return
            if node.sub is not None:
                self._collect(node.sub, slab, pieces)
            if node.hi - node.lo == 1:
                return
            mid = (node.lo + node.hi) // 2
            node = node.left if slab[node.axis] < mid else node.right

    # -- accounting -----------------------------------------------------------
    @property
    def stats(self) -> IndexStats:
        cells = [leaf.cells for leaf in self._leaves]
        return IndexStats(
            groups=len(self.groups),
            leaves=len(cells),
            segment_nodes=self._segment_nodes,
            assignments=sum(len(c.items) for c in cells),
            faces=sum(c.stats.faces for c in cells),
            grid_cells=sum(c.stats.grid_cells for c in cells),
            pft_nodes=sum(c.stats.pft_nodes for c in cells),
            pft_entries=sum(c.stats.pft_entries for c in cells),
            locator_words=sum(c.stats.locator_words for c in cells),
            build_seconds=self.build_seconds,
            leaf_depths=dict(sorted(self._leaf_depths.items())),
            bitmask_trees=len(self.deep),
            bitmask_words=self._bitmask_words,
            table_cells=self._table_cells,
            pack_words=self._pack_words,
        )

    @property
    def leaves(self) -> list[CellIndex]:
        return [leaf.cells for leaf in self._leaves]

    @property
    def compiled(self) -> bool:
        return self._fast


class QuerySession:
    """A sequence of queries that change one intervention value at a time
    (sliders, dose-response sweeps).  A change re-locates one axis and
    re-reads only the groups that use it."""

    def __init__(self, index: ProvenanceIndex, t: Sequence[float]) -> None:
        self.index = index
        x = index._cast(t)
        self.x = x
        self._prow = {a: (th, rows) for a, th, rows in index._paxes}
        self._brow = {a: (th, rows) for a, th, rows in index._baxes}
        self.slab = {a: bisect_left(th, x[a]) for a, (th, _) in self._prow.items()}
        self.bslab = {a: bisect_left(th, x[a]) for a, (th, _) in self._brow.items()}
        packed = 0
        for a, (_, rows) in self._prow.items():
            packed += rows[self.slab[a]]
        self.packed = packed
        cells = memoryview(packed.to_bytes(index._nbytes, "little")).cast("I")
        self.vals = list(map(getitem, index._tables, cells))
        self.leaves = index._bitmask_leaves(x) if index._baxes else []
        self.bvals = list(map(getitem, index._bvals, self.leaves))

    @property
    def value(self) -> float:
        return math.fsum(chain((self.index._const,), self.vals, self.bvals))

    def set(self, axis: int, value: float) -> float:
        """Set the ``axis``-th intervention value and return the new value."""
        index = self.index
        v = index._cast([value])[0]
        self.x[axis] = v
        if axis in self._prow:
            th, rows = self._prow[axis]
            s = bisect_left(th, v)
            old = self.slab[axis]
            if s != old:
                self.slab[axis] = s
                packed = self.packed = self.packed + rows[s] - rows[old]
                groups = index._groups_of_axis[axis]
                if len(groups) > 16:
                    cells = memoryview(packed.to_bytes(index._nbytes, "little")).cast("I")
                    self.vals = list(map(getitem, index._tables, cells))
                else:
                    tables, vals = index._tables, self.vals
                    for g in groups:
                        vals[g] = tables[g][(packed >> (32 * g)) & _FIELD]
        if axis in self._brow:
            th, rows = self._brow[axis]
            s = bisect_left(th, v)
            if s != self.bslab[axis]:
                self.bslab[axis] = s
                it = iter(self._brow.items())
                a, (_, r) = next(it)
                mask = r[self.bslab[a]]
                for a, (_, r) in it:
                    mask &= r[self.bslab[a]]
                self.leaves = list(map(int.bit_length, memoryview(mask.to_bytes(index._bbytes, "little")).cast("Q")))
                self.bvals = list(map(getitem, index._bvals, self.leaves))
        return self.value

    def answer(self) -> LazyAnswer:
        cells = memoryview(self.packed.to_bytes(self.index._nbytes, "little")).cast("I")
        return LazyAnswer(self.value, self.index, cells, list(self.leaves))


def _merge_groups(groups, budget: int, bound: BoundEnsemble):
    """Greedy: fold the group with the fewest trees into its smallest superset
    group, or into the group sharing most axes (key becomes the union)."""
    groups = {k: list(v) for k, v in groups.items()}

    def trees(items):
        return len({bound.boxes[i].tree for i in items})

    while len(groups) > max(1, budget):
        small = min(groups, key=lambda k: (trees(groups[k]), k))
        items = groups.pop(small)
        ss = set(small)
        supersets = [k for k in groups if ss <= set(k)]
        if supersets:
            target = min(supersets, key=lambda k: (len(k), -trees(groups[k]), k))
            groups[target].extend(items)
        else:
            best = max(groups, key=lambda k: (len(ss & set(k)), -len(ss | set(k)), k))
            merged = tuple(sorted(ss | set(best)))
            other = groups.pop(best)
            groups.setdefault(merged, []).extend(other + items)
    return groups


@dataclass
class TuneResult:
    index: ProvenanceIndex
    table: list[dict]


def autotune(
    bound: BoundEnsemble,
    sample: np.ndarray,
    taus: Iterable[float] = (0, 256, 4096, 65536),
    budgets: Iterable[int | None] = (None, 8, 4, 1),
    pieces_budget: float | None = None,
    space_budget: int | None = None,
    **kwargs,
) -> TuneResult:
    """Build every (tau, group budget) configuration and pick one.

    With ``pieces_budget``: the smallest index whose mean piece count on
    ``sample`` is within budget.  With ``space_budget``: the fewest pieces
    among indexes within the record budget.  Otherwise: the configuration with
    the smallest records x pieces product (a balanced point).

    This tunes the segment-tree parameters; it uses axis-support groups unless
    ``grouping`` is given.  See :func:`tune_budget` for the compiled index.
    """
    kwargs.setdefault("grouping", "support")
    table: list[dict] = []
    for tau in taus:
        for budget in budgets:
            try:
                index = ProvenanceIndex(bound, tau=tau, group_budget=budget, **kwargs)
            except MemoryError:
                continue
            pieces = [len(index.query(t).pieces) for t in sample]
            table.append(
                {
                    "tau": tau,
                    "group_budget": budget,
                    "records": index.stats.records,
                    "mean_pieces": float(np.mean(pieces)),
                    "max_pieces": int(np.max(pieces)),
                    "build_s": index.build_seconds,
                }
            )
            del index
    if not table:
        raise MemoryError("no configuration fits max_leaf_cells")
    if pieces_budget is not None:
        ok = [i for i, r in enumerate(table) if r["mean_pieces"] <= pieces_budget]
        pick = min(ok, key=lambda i: table[i]["records"]) if ok else min(
            range(len(table)), key=lambda i: table[i]["mean_pieces"]
        )
    elif space_budget is not None:
        ok = [i for i, r in enumerate(table) if r["records"] <= space_budget]
        pick = min(ok, key=lambda i: table[i]["mean_pieces"]) if ok else min(
            range(len(table)), key=lambda i: table[i]["records"]
        )
    else:
        pick = min(range(len(table)), key=lambda i: table[i]["records"] * table[i]["mean_pieces"])
    table[pick]["chosen"] = True
    chosen = ProvenanceIndex(
        bound, tau=table[pick]["tau"], group_budget=table[pick]["group_budget"], **kwargs
    )
    return TuneResult(chosen, table)


def tune_budget(
    bound: BoundEnsemble,
    sample: np.ndarray,
    budgets: Iterable[int] = (200_000, 2_000_000, 20_000_000),
    space_budget: int | None = None,
    repeats: int = 3,
    **kwargs,
) -> TuneResult:
    """Build the compiled index for every table budget and keep the one with the
    fastest measured query time on ``sample`` (among those within
    ``space_budget`` records, if given).  Every configuration is exact."""
    table: list[dict] = []
    best, best_key = None, None
    for budget in budgets:
        index = ProvenanceIndex(bound, budget=budget, **kwargs)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            for t in sample:
                index.query(t)
            times.append((time.perf_counter() - start) / max(1, len(sample)))
        row = {
            "budget": budget,
            "records": index.stats.records,
            "groups": len(index.groups),
            "bitmask_trees": len(index.deep),
            "query_us": 1e6 * float(np.median(times)),
            "build_s": index.build_seconds,
        }
        table.append(row)
        fits = space_budget is None or row["records"] <= space_budget
        key = (not fits, row["query_us"] if fits else row["records"])
        if best_key is None or key < best_key:
            best, best_key = index, key
            for r in table:
                r.pop("chosen", None)
            row["chosen"] = True
        else:
            del index
    return TuneResult(best, table)
