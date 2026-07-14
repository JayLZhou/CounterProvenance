"""Static exact provenance index for additive tree ensembles.

For one unit, every tree induces a partition of the intervention domain into
left-closed, right-open leaf intervals.  A point query must add the provenance
weight of the one interval selected in every tree.  This module stores those
weights in a centered interval tree.

Each leaf interval is assigned to exactly one interval-tree node.  Prefix and
suffix aggregate roots make the semigroup contribution of that node available
in O(1) after locating the query in its endpoint catalog.  Static fractional
cascading performs one binary search at the root and O(1) catalog work per
remaining level.  The resulting index uses only semiring addition: it never
assumes subtraction or additive inverses.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Hashable, Mapping, Sequence

from .versioned import CircuitArena, NodeId, Polynomial


@dataclass(frozen=True)
class TreeLeafInterval:
    """One tree leaf occurrence on an intervention line.

    Intervals use ``[left, right)`` semantics.  ``root`` is the provenance
    circuit for the tree contribution on this interval.
    """

    tree: Hashable
    left: float
    right: float
    root: NodeId


@dataclass(frozen=True)
class TreeStabbingClosure:
    """A factorized exact result returned by a tree stabbing query."""

    arena: CircuitArena
    factor_roots: tuple[NodeId, ...]

    @property
    def factor_count(self) -> int:
        return len(self.factor_roots)

    def materialize(self) -> NodeId:
        """Combine the O(log N) query factors into one canonical root."""

        return self.arena.add(*self.factor_roots)

    def evaluate(
        self,
        valuation: Mapping[Hashable, float],
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> float:
        return sum(
            self.arena.evaluate(root, valuation, selectors)
            for root in self.factor_roots
        )

    def polynomial(
        self,
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> Polynomial:
        result: Polynomial = Counter()
        for root in self.factor_roots:
            result.update(self.arena.polynomial(root, selectors))
        return result


@dataclass(frozen=True)
class TreeQueryStats:
    """Observable work for one fractional-cascading query."""

    nodes_visited: int
    root_binary_searches: int
    cascade_corrections: int
    factor_count: int


@dataclass(frozen=True)
class TreeIndexStats:
    """Structural records retained by a static tree provenance index."""

    tree_count: int
    interval_count: int
    breakpoint_count: int
    interval_tree_nodes: int
    catalog_entries: int
    rank_records: int
    aggregate_records: int
    cascade_links: int
    provenance_nodes_created: int

    @property
    def storage_records(self) -> int:
        """A word-level structural count, including interval records."""

        return (
            self.interval_count
            + self.interval_tree_nodes
            + self.catalog_entries
            + self.rank_records
            + self.aggregate_records
            + self.cascade_links
        )


@dataclass(frozen=True)
class _StoredInterval:
    tree: Hashable
    left: float
    right: float
    root: NodeId
    ordinal: int


@dataclass
class _IntervalNode:
    center: float
    crossing_count: int
    left_prefix: tuple[NodeId, ...]
    right_suffix: tuple[NodeId, ...]
    catalog: tuple[float, ...]
    left_ranks: tuple[int, ...]
    right_ranks: tuple[int, ...]
    left: "_IntervalNode | None" = None
    right: "_IntervalNode | None" = None
    left_down: tuple[int, ...] | None = None
    right_down: tuple[int, ...] | None = None


class TreeProvenanceIndex:
    """Exact static interval-stabbing index for one unit.

    The supplied intervals must form one complete, non-overlapping partition
    of ``domain`` for every tree.  Tree/rule provenance is shared through the
    roots already interned in ``arena``; this index stores only interval
    endpoints, aggregate roots, and fractional-cascading links.
    """

    def __init__(
        self,
        arena: CircuitArena,
        intervals: Sequence[TreeLeafInterval],
        domain: tuple[float, float],
        anchor_root: NodeId | None = None,
    ) -> None:
        self.arena = arena
        self.domain = (float(domain[0]), float(domain[1]))
        self.anchor_root = arena.zero if anchor_root is None else anchor_root
        self._validate_domain()
        self._validate_root(self.anchor_root)

        stored = tuple(
            _StoredInterval(
                record.tree,
                float(record.left),
                float(record.right),
                record.root,
                ordinal,
            )
            for ordinal, record in enumerate(intervals)
        )
        if not stored:
            raise ValueError(
                "A tree provenance index requires at least one interval."
            )

        self._validate_partitions(stored)
        self.intervals = tuple(
            TreeLeafInterval(record.tree, record.left, record.right, record.root)
            for record in stored
        )

        arena_before = arena.node_count
        build_order = tuple(
            sorted(
                stored,
                key=lambda record: (self._interior_point(record), record.ordinal),
            )
        )
        self._root = self._build_node(build_order)
        nodes = tuple(self._walk_nodes(self._root))
        tree_count = len({record.tree for record in stored})
        self.stats = TreeIndexStats(
            tree_count=tree_count,
            interval_count=len(stored),
            breakpoint_count=len(stored) - tree_count,
            interval_tree_nodes=len(nodes),
            catalog_entries=sum(len(node.catalog) for node in nodes),
            rank_records=sum(
                len(node.left_ranks) + len(node.right_ranks) for node in nodes
            ),
            aggregate_records=sum(
                len(node.left_prefix) + len(node.right_suffix) for node in nodes
            ),
            cascade_links=sum(
                (len(node.left_down) if node.left_down is not None else 0)
                + (len(node.right_down) if node.right_down is not None else 0)
                for node in nodes
            ),
            provenance_nodes_created=arena.node_count - arena_before,
        )

    def closure_for(self, intervention: float) -> TreeStabbingClosure:
        closure, _ = self.query_with_stats(intervention)
        return closure

    def query_with_stats(
        self,
        intervention: float,
    ) -> tuple[TreeStabbingClosure, TreeQueryStats]:
        value = float(intervention)
        self._validate_query(value)

        factors: list[NodeId] = []
        if self.anchor_root != self.arena.zero:
            factors.append(self.anchor_root)

        node = self._root
        position = bisect_right(node.catalog, value)
        visited = 0
        corrections = 0

        while node is not None:
            visited += 1
            go_left = value < node.center
            if go_left:
                rank = node.left_ranks[position]
                contribution = node.left_prefix[rank]
                child = node.left
                down = node.left_down
            else:
                rank = node.right_ranks[position]
                contribution = node.right_suffix[rank]
                child = node.right
                down = node.right_down

            if contribution != self.arena.zero:
                factors.append(contribution)

            if child is None:
                break
            assert down is not None
            position = down[position]
            if position < len(child.catalog) and child.catalog[position] <= value:
                position += 1
                corrections += 1
            if position < len(child.catalog) and child.catalog[position] <= value:
                raise AssertionError("Fractional-cascading bridge skipped two entries.")
            node = child

        closure = TreeStabbingClosure(self.arena, tuple(factors))
        return closure, TreeQueryStats(
            nodes_visited=visited,
            root_binary_searches=1,
            cascade_corrections=corrections,
            factor_count=closure.factor_count,
        )

    def root_for(self, intervention: float) -> NodeId:
        return self.closure_for(intervention).materialize()

    def evaluate(
        self,
        intervention: float,
        valuation: Mapping[Hashable, float],
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> float:
        return self.closure_for(intervention).evaluate(valuation, selectors)

    def polynomial(
        self,
        intervention: float,
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> Polynomial:
        return self.closure_for(intervention).polynomial(selectors)

    def _validate_domain(self) -> None:
        left, right = self.domain
        if math.isnan(left) or math.isnan(right) or left >= right:
            raise ValueError(f"Invalid intervention domain {self.domain!r}.")

    def _validate_query(self, intervention: float) -> None:
        left, right = self.domain
        if math.isnan(intervention) or not left <= intervention < right:
            raise ValueError(
                f"Intervention {intervention} is outside indexed domain "
                f"[{left}, {right})."
            )

    def _validate_root(self, root: NodeId) -> None:
        if not isinstance(root, int) or not 0 <= root < self.arena.node_count:
            raise ValueError(f"Unknown provenance root {root!r}.")

    def _validate_partitions(self, intervals: Sequence[_StoredInterval]) -> None:
        by_tree: dict[Hashable, list[_StoredInterval]] = defaultdict(list)
        for record in intervals:
            if math.isnan(record.left) or math.isnan(record.right):
                raise ValueError("Tree interval endpoints cannot be NaN.")
            if record.left >= record.right:
                raise ValueError(
                    f"Invalid interval [{record.left}, {record.right}) for "
                    f"tree {record.tree!r}."
                )
            self._validate_root(record.root)
            by_tree[record.tree].append(record)

        domain_left, domain_right = self.domain
        for tree, records in by_tree.items():
            ordered = sorted(
                records,
                key=lambda item: (item.left, item.right, item.ordinal),
            )
            if ordered[0].left != domain_left or ordered[-1].right != domain_right:
                raise ValueError(
                    f"Tree {tree!r} does not cover domain "
                    f"[{domain_left}, {domain_right})."
                )
            for previous, current in zip(ordered, ordered[1:]):
                if previous.right < current.left:
                    raise ValueError(
                        f"Tree {tree!r} has a gap between {previous.right} "
                        f"and {current.left}."
                    )
                if previous.right > current.left:
                    raise ValueError(
                        f"Tree {tree!r} has overlapping intervals at "
                        f"{current.left}."
                    )

    def _build_node(self, intervals: Sequence[_StoredInterval]) -> _IntervalNode:
        # ``intervals`` arrives sorted by its representative point.  Stable
        # subsequences keep that order recursively, avoiding a fresh median
        # sort at every level.
        center = self._interior_point(intervals[len(intervals) // 2])

        left_records: list[_StoredInterval] = []
        right_records: list[_StoredInterval] = []
        crossing: list[_StoredInterval] = []
        for record in intervals:
            if record.right <= center:
                left_records.append(record)
            elif record.left > center:
                right_records.append(record)
            else:
                crossing.append(record)

        if not crossing:
            raise AssertionError("Interval-tree center must stab at least one interval.")

        left_child = self._build_node(left_records) if left_records else None
        right_child = self._build_node(right_records) if right_records else None

        by_left = sorted(
            crossing,
            key=lambda item: (item.left, item.right, item.ordinal),
        )
        by_right = sorted(
            crossing,
            key=lambda item: (item.right, item.left, item.ordinal),
        )

        left_prefix = [self.arena.zero]
        for record in by_left:
            left_prefix.append(self.arena.add(left_prefix[-1], record.root))

        right_suffix = [self.arena.zero] * (len(by_right) + 1)
        for index in range(len(by_right) - 1, -1, -1):
            right_suffix[index] = self.arena.add(
                by_right[index].root,
                right_suffix[index + 1],
            )

        own_catalog = {record.left for record in crossing}
        own_catalog.update(record.right for record in crossing)
        augmented = set(own_catalog)
        if left_child is not None:
            augmented.update(left_child.catalog[::2])
        if right_child is not None:
            augmented.update(right_child.catalog[::2])
        catalog = tuple(sorted(augmented))

        left_values = tuple(record.left for record in by_left)
        right_values = tuple(record.right for record in by_right)
        left_ranks = self._gap_ranks(catalog, left_values)
        right_ranks = self._gap_ranks(catalog, right_values)

        return _IntervalNode(
            center=center,
            crossing_count=len(crossing),
            left_prefix=tuple(left_prefix),
            right_suffix=tuple(right_suffix),
            catalog=catalog,
            left_ranks=left_ranks,
            right_ranks=right_ranks,
            left=left_child,
            right=right_child,
            left_down=(
                self._down_bridges(catalog, left_child.catalog)
                if left_child is not None
                else None
            ),
            right_down=(
                self._down_bridges(catalog, right_child.catalog)
                if right_child is not None
                else None
            ),
        )

    @staticmethod
    def _interior_point(record: _StoredInterval) -> float:
        left, right = record.left, record.right
        if math.isinf(left) and math.isinf(right):
            return 0.0
        if math.isinf(left):
            return math.nextafter(right, -math.inf)
        if math.isinf(right):
            return left
        midpoint = left + (right - left) / 2.0
        return midpoint if left <= midpoint < right else left

    @staticmethod
    def _gap_ranks(
        catalog: tuple[float, ...],
        values: tuple[float, ...],
    ) -> tuple[int, ...]:
        ranks = [0]
        for endpoint in catalog:
            ranks.append(bisect_right(values, endpoint))
        return tuple(ranks)

    @staticmethod
    def _down_bridges(
        parent_catalog: tuple[float, ...],
        child_catalog: tuple[float, ...],
    ) -> tuple[int, ...]:
        bridges = [0]
        for endpoint in parent_catalog:
            bridges.append(bisect_right(child_catalog, endpoint))
        return tuple(bridges)

    @staticmethod
    def _walk_nodes(root: _IntervalNode):
        pending = [root]
        while pending:
            node = pending.pop()
            yield node
            if node.right is not None:
                pending.append(node.right)
            if node.left is not None:
                pending.append(node.left)
