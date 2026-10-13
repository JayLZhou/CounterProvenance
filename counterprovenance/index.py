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
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .cellindex import CellIndex, Piece
from .ensemble import BoundEnsemble, Slabs
from .semiring import CircuitArena, NodeId


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

    @property
    def records(self) -> int:
        """PFT nodes + entries + root and value per face + locators + item tables
        + segment-tree nodes."""
        return (
            self.pft_nodes
            + self.pft_entries
            + 2 * self.faces
            + self.locator_words
            + self.assignments
            + self.segment_nodes
        )

    @property
    def packed_bytes(self) -> int:
        """Estimated packed layout: 8 B per PFT node (two int32 children), 12 B per
        entry, 16 B per face (root, version, float64 value), 4 B per locator word
        and item reference, 16 B per segment-tree node."""
        return (
            8 * self.pft_nodes
            + 12 * self.pft_entries
            + 16 * self.faces
            + 4 * (self.locator_words + self.assignments)
            + 16 * self.segment_nodes
        )


class ProvenanceIndex:
    def __init__(
        self,
        bound: BoundEnsemble,
        tau: float = 4096,
        group_budget: int | None = None,
        grouping: str = "support",
        axis_order: str = "bycount",
        layout: str = "hybrid",
        exact: bool = True,
        alpha: float = 1.5,
        max_leaf_cells: int = 20_000_000,
    ) -> None:
        t0 = time.perf_counter()
        self.bound = bound
        self.slabs = Slabs(bound)
        self.tau = tau
        self.layout = layout
        self.exact = exact
        self.alpha = alpha
        self.axis_order = axis_order
        self.max_leaf_cells = max_leaf_cells
        self._segment_nodes = 0
        self._leaves: list[_Leaf] = []
        self._leaf_depths: Counter = Counter()
        groups = self._groups(grouping)
        if group_budget is not None:
            groups = _merge_groups(groups, group_budget, bound)
        self.groups = groups
        self.roots = [self._build(items, tuple(axes), 0) for axes, items in groups.items()]
        self.build_seconds = time.perf_counter() - t0

    # -- grouping -----------------------------------------------------------
    def _groups(self, grouping: str) -> dict[tuple[int, ...], list[int]]:
        if grouping == "none":
            return {tuple(range(self.bound.d)): list(range(len(self.bound.boxes)))}
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
        return groups

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
    def query(self, t: Sequence[float]) -> Answer:
        x = self.bound.prepare(t)
        slab = self.slabs.locate(x)
        pieces: list[Piece] = []
        for root in self.roots:
            self._collect(root, slab, pieces)
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
        )

    @property
    def leaves(self) -> list[CellIndex]:
        return [leaf.cells for leaf in self._leaves]


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
    """
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
