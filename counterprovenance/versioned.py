"""Persistent semiring circuits for versioned counterfactual provenance.

The intervention line induces an ordered sequence of exact regions.  A naive
index stores one complete provenance circuit per region.  This module instead
keeps all circuit nodes in a global hash-consed arena and records one root per
region.  When a tree leaf or neural activation changes, only the affected
template nodes are rebuilt; unchanged subcircuits remain shared.

The implementation is deliberately independent of staircase or laminar
supports.  Region transitions may turn any set of slots on or off in any
order.  It also preserves both semiring addition and multiplication, so it
does not require path expansion into an additive-only fragment matrix.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence


NodeId = int
TemplateNodeId = int
Monomial = tuple[str, ...]
Polynomial = Counter[Monomial]


@dataclass(frozen=True)
class CircuitNode:
    """One immutable node in a canonical semiring circuit arena."""

    op: str
    children: tuple[NodeId, ...] = ()
    label: Hashable | None = None


class CircuitArena:
    """Hash-consed nodes for a commutative provenance semiring.

    Addition is not idempotent: ``a + a`` remains two occurrences.  Child IDs
    are sorted only to canonicalize commutativity; duplicates are retained.
    """

    def __init__(self) -> None:
        self._nodes: list[CircuitNode] = []
        self._interned: dict[CircuitNode, NodeId] = {}
        self.zero = self._intern(CircuitNode("zero"))
        self.one = self._intern(CircuitNode("one"))

    @property
    def nodes(self) -> tuple[CircuitNode, ...]:
        return tuple(self._nodes)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def node(self, node_id: NodeId) -> CircuitNode:
        return self._nodes[node_id]

    def _intern(self, node: CircuitNode) -> NodeId:
        existing = self._interned.get(node)
        if existing is not None:
            return existing
        node_id = len(self._nodes)
        self._nodes.append(node)
        self._interned[node] = node_id
        return node_id

    def atom(self, label: Hashable) -> NodeId:
        return self._intern(CircuitNode("atom", label=label))

    def add(self, *children: NodeId) -> NodeId:
        kept = [child for child in children if child != self.zero]
        if not kept:
            return self.zero
        if len(kept) == 1:
            return kept[0]
        return self._intern(CircuitNode("add", tuple(sorted(kept))))

    def mul(self, *children: NodeId) -> NodeId:
        if any(child == self.zero for child in children):
            return self.zero
        kept = [child for child in children if child != self.one]
        if not kept:
            return self.one
        if len(kept) == 1:
            return kept[0]
        return self._intern(CircuitNode("mul", tuple(sorted(kept))))

    def switch(self, selector: Hashable, when_true: NodeId, when_false: NodeId) -> NodeId:
        if when_true == when_false:
            return when_true
        return self._intern(
            CircuitNode("switch", (when_true, when_false), label=selector)
        )

    def evaluate(
        self,
        root: NodeId,
        valuation: Mapping[Hashable, float],
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> float:
        """Evaluate a root under a numeric semiring homomorphism."""

        selector_values = selectors or {}
        memo: dict[NodeId, float] = {}

        def visit(node_id: NodeId) -> float:
            if node_id in memo:
                return memo[node_id]

            node = self._nodes[node_id]
            if node.op == "zero":
                value = 0.0
            elif node.op == "one":
                value = 1.0
            elif node.op == "atom":
                try:
                    value = float(valuation[node.label])
                except KeyError as exc:
                    raise KeyError(f"Missing valuation for atom {node.label!r}") from exc
            elif node.op == "add":
                value = sum(visit(child) for child in node.children)
            elif node.op == "mul":
                value = 1.0
                for child in node.children:
                    value *= visit(child)
            elif node.op == "switch":
                try:
                    enabled = selector_values[node.label]
                except KeyError as exc:
                    raise KeyError(f"Missing value for selector {node.label!r}") from exc
                value = visit(node.children[0 if enabled else 1])
            else:  # pragma: no cover - construction prevents this branch.
                raise ValueError(f"Unknown circuit operation {node.op!r}")

            memo[node_id] = value
            return value

        return visit(root)

    def polynomial(
        self,
        root: NodeId,
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> Polynomial:
        """Expand a small circuit in the free commutative semiring.

        This is intended for exactness tests and explanations, not large-model
        execution: multiplication can make the logical polynomial exponential
        even when the stored circuit is compact.
        """

        selector_values = selectors or {}
        memo: dict[NodeId, Polynomial] = {}

        def visit(node_id: NodeId) -> Polynomial:
            if node_id in memo:
                return memo[node_id].copy()

            node = self._nodes[node_id]
            if node.op == "zero":
                result: Polynomial = Counter()
            elif node.op == "one":
                result = Counter({(): 1})
            elif node.op == "atom":
                result = Counter({(str(node.label),): 1})
            elif node.op == "add":
                result = Counter()
                for child in node.children:
                    result.update(visit(child))
            elif node.op == "mul":
                result = Counter({(): 1})
                for child in node.children:
                    product: Polynomial = Counter()
                    child_poly = visit(child)
                    for left, left_count in result.items():
                        for right, right_count in child_poly.items():
                            product[tuple(sorted(left + right))] += left_count * right_count
                    result = product
            elif node.op == "switch":
                try:
                    enabled = selector_values[node.label]
                except KeyError as exc:
                    raise KeyError(f"Missing value for selector {node.label!r}") from exc
                result = visit(node.children[0 if enabled else 1])
            else:  # pragma: no cover - construction prevents this branch.
                raise ValueError(f"Unknown circuit operation {node.op!r}")

            memo[node_id] = result.copy()
            return result

        return visit(root)

    def reachable_nodes(self, root: NodeId) -> frozenset[NodeId]:
        pending = [root]
        seen: set[NodeId] = set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            pending.extend(self._nodes[node_id].children)
        return frozenset(seen)


@dataclass(frozen=True)
class TemplateNode:
    """A fixed operator-tree node whose slot leaves may change by version."""

    op: str
    children: tuple[TemplateNodeId, ...] = ()
    slot: str | None = None
    fixed_root: NodeId | None = None


class CircuitTemplateBuilder:
    """Build a fixed expression topology for persistent region versions."""

    def __init__(self, arena: CircuitArena) -> None:
        self.arena = arena
        self._nodes: list[TemplateNode] = []
        self._slots: dict[str, TemplateNodeId] = {}

    def slot(self, name: str) -> TemplateNodeId:
        if name in self._slots:
            raise ValueError(f"Duplicate circuit slot {name!r}")
        node_id = len(self._nodes)
        self._nodes.append(TemplateNode("slot", slot=name))
        self._slots[name] = node_id
        return node_id

    def fixed(self, root: NodeId) -> TemplateNodeId:
        node_id = len(self._nodes)
        self._nodes.append(TemplateNode("fixed", fixed_root=root))
        return node_id

    def add(self, left: TemplateNodeId, right: TemplateNodeId) -> TemplateNodeId:
        return self._operator("add", left, right)

    def mul(self, left: TemplateNodeId, right: TemplateNodeId) -> TemplateNodeId:
        return self._operator("mul", left, right)

    def _operator(
        self,
        op: str,
        left: TemplateNodeId,
        right: TemplateNodeId,
    ) -> TemplateNodeId:
        if left >= len(self._nodes) or right >= len(self._nodes):
            raise ValueError("Template children must be created before their parent.")
        node_id = len(self._nodes)
        self._nodes.append(TemplateNode(op, (left, right)))
        return node_id

    def fold_balanced(
        self,
        op: str,
        nodes: Sequence[TemplateNodeId],
    ) -> TemplateNodeId:
        if op not in {"add", "mul"}:
            raise ValueError(f"Unsupported associative operation {op!r}")
        if not nodes:
            identity = self.arena.zero if op == "add" else self.arena.one
            return self.fixed(identity)

        def fold(values: Sequence[TemplateNodeId]) -> TemplateNodeId:
            if len(values) == 1:
                return values[0]
            midpoint = len(values) // 2
            left = fold(values[:midpoint])
            right = fold(values[midpoint:])
            return self._operator(op, left, right)

        return fold(tuple(nodes))

    def build(self, root: TemplateNodeId) -> "CircuitTemplate":
        if root >= len(self._nodes):
            raise ValueError(f"Unknown template root {root}")
        return CircuitTemplate(self.arena, tuple(self._nodes), root, dict(self._slots))


class CircuitTemplate:
    """Immutable expression topology shared by every unit and region."""

    def __init__(
        self,
        arena: CircuitArena,
        nodes: tuple[TemplateNode, ...],
        root: TemplateNodeId,
        slots: Mapping[str, TemplateNodeId],
    ) -> None:
        self.arena = arena
        self.nodes = nodes
        self.root = root
        self.slots = dict(slots)
        parents: list[list[TemplateNodeId]] = [[] for _ in nodes]
        for parent_id, node in enumerate(nodes):
            for child in node.children:
                parents[child].append(parent_id)
        self.parents = tuple(tuple(values) for values in parents)

    def new_version_builder(
        self,
        initial_slots: Mapping[str, NodeId],
    ) -> "CircuitVersionBuilder":
        return CircuitVersionBuilder(self, initial_slots)


@dataclass(frozen=True)
class VersionUpdateStats:
    changed_slots: int
    affected_template_nodes: int
    new_arena_nodes: int
    root: NodeId


class CircuitVersionBuilder:
    """Sequentially build persistent versions of one circuit template."""

    def __init__(
        self,
        template: CircuitTemplate,
        initial_slots: Mapping[str, NodeId],
    ) -> None:
        missing = set(template.slots) - set(initial_slots)
        extra = set(initial_slots) - set(template.slots)
        if missing or extra:
            raise ValueError(
                f"Initial slot mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )

        self.template = template
        self.arena = template.arena
        self._current: list[NodeId] = [self.arena.zero] * len(template.nodes)

        for node_id, node in enumerate(template.nodes):
            if node.op == "slot":
                assert node.slot is not None
                self._current[node_id] = initial_slots[node.slot]
            elif node.op == "fixed":
                assert node.fixed_root is not None
                self._current[node_id] = node.fixed_root
            else:
                self._current[node_id] = self._evaluate_template_node(node_id)

        self.roots: list[NodeId] = [self._current[template.root]]
        self.stats: list[VersionUpdateStats] = [
            VersionUpdateStats(
                changed_slots=len(template.slots),
                affected_template_nodes=len(template.nodes),
                new_arena_nodes=0,
                root=self._current[template.root],
            )
        ]

    @property
    def current_root(self) -> NodeId:
        return self._current[self.template.root]

    def current_slot(self, name: str) -> NodeId:
        return self._current[self.template.slots[name]]

    def update(self, updates: Mapping[str, NodeId]) -> NodeId:
        unknown = set(updates) - set(self.template.slots)
        if unknown:
            raise ValueError(f"Unknown circuit slots: {sorted(unknown)}")

        changed: list[TemplateNodeId] = []
        for name, value in updates.items():
            slot_id = self.template.slots[name]
            if self._current[slot_id] != value:
                self._current[slot_id] = value
                changed.append(slot_id)

        arena_before = self.arena.node_count
        # Count the changed leaves themselves as part of the affected closure.
        # This makes the update statistic include both slot writes and rebuilt
        # ancestors, which is the full template work charged by the bound.
        affected: set[TemplateNodeId] = set(changed)
        pending = list(changed)
        while pending:
            child = pending.pop()
            for parent in self.template.parents[child]:
                if parent not in affected:
                    affected.add(parent)
                    pending.append(parent)

        for node_id in sorted(affected):
            if self.template.nodes[node_id].op not in {"slot", "fixed"}:
                self._current[node_id] = self._evaluate_template_node(node_id)

        root = self.current_root
        self.roots.append(root)
        self.stats.append(
            VersionUpdateStats(
                changed_slots=len(changed),
                affected_template_nodes=len(affected),
                new_arena_nodes=self.arena.node_count - arena_before,
                root=root,
            )
        )
        return root

    def _evaluate_template_node(self, node_id: TemplateNodeId) -> NodeId:
        node = self.template.nodes[node_id]
        if node.op == "add":
            return self.arena.add(*(self._current[child] for child in node.children))
        if node.op == "mul":
            return self.arena.mul(*(self._current[child] for child in node.children))
        if node.op in {"slot", "fixed"}:
            return self._current[node_id]
        raise ValueError(f"Unsupported template operation {node.op!r}")


@dataclass(frozen=True)
class UnitRegionVersions:
    """Left-closed, right-open regions and their persistent roots."""

    starts: tuple[float, ...]
    roots: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        if not self.starts:
            raise ValueError("A unit must contain at least one region.")
        if len(self.starts) != len(self.roots):
            raise ValueError("Region starts and roots must have equal length.")
        if any(left >= right for left, right in zip(self.starts, self.starts[1:])):
            raise ValueError("Region starts must be strictly increasing.")

    def region_index(self, intervention: float) -> int:
        index = bisect_right(self.starts, intervention) - 1
        if index < 0:
            raise ValueError(
                f"Intervention {intervention} is below indexed domain {self.starts[0]}."
            )
        return index

    def root_for(self, intervention: float) -> NodeId:
        return self.roots[self.region_index(intervention)]


class VersionedSemiringIndex:
    """Population directory over globally shared persistent circuit roots."""

    def __init__(self, arena: CircuitArena) -> None:
        self.arena = arena
        self._units: dict[Hashable, UnitRegionVersions] = {}

    @property
    def unit_count(self) -> int:
        return len(self._units)

    @property
    def region_count(self) -> int:
        return sum(len(unit.starts) for unit in self._units.values())

    def add_unit(
        self,
        unit: Hashable,
        starts: Sequence[float],
        roots: Sequence[NodeId],
    ) -> None:
        if unit in self._units:
            raise ValueError(f"Duplicate unit {unit!r}")
        self._units[unit] = UnitRegionVersions(tuple(starts), tuple(roots))

    def versions_for(self, unit: Hashable) -> UnitRegionVersions:
        try:
            return self._units[unit]
        except KeyError as exc:
            raise KeyError(f"Unknown indexed unit {unit!r}") from exc

    def root_for(self, unit: Hashable, intervention: float) -> NodeId:
        return self.versions_for(unit).root_for(intervention)

    def evaluate(
        self,
        unit: Hashable,
        intervention: float,
        valuation: Mapping[Hashable, float],
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> float:
        root = self.root_for(unit, intervention)
        return self.arena.evaluate(root, valuation, selectors)

    def polynomial(
        self,
        unit: Hashable,
        intervention: float,
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> Polynomial:
        root = self.root_for(unit, intervention)
        return self.arena.polynomial(root, selectors)


def independent_node_occurrences(arena: CircuitArena, roots: Iterable[NodeId]) -> int:
    """Count nodes if every region stored its reachable circuit independently."""

    return sum(len(arena.reachable_nodes(root)) for root in roots)


def balanced_slot_template(
    arena: CircuitArena,
    slot_names: Sequence[str],
    op: str = "add",
) -> CircuitTemplate:
    """Create a balanced associative template over mutable slots."""

    builder = CircuitTemplateBuilder(arena)
    leaves = [builder.slot(name) for name in slot_names]
    root = builder.fold_balanced(op, leaves)
    return builder.build(root)
