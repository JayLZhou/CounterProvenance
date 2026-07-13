"""Two exact layouts for versioned counterfactual provenance closures.

The intervention line induces an ordered sequence of exact regions.  A naive
index stores one complete provenance circuit per region.  This module exposes
one logical object, a shared semiring template paired with a versioned slot
environment, through two physical layouts:

* ``VersionedSlotEnvironment`` stores only effective slot changes;
* ``CircuitVersionBuilder`` materializes canonical roots by path copying.

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
    """Immutable expression topology shared by compatible units and regions."""

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
        self.slot_names = frozenset(self.slots)
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
class EnvironmentVersionStats:
    """Storage work for one compact slot-environment version."""

    version: int
    changed_slots: int


class VersionedSlotEnvironment:
    """Compact partial persistence for a template's mutable slots.

    Each slot stores its initial root followed by only the versions at which
    that root actually changes.  The representation therefore uses one record
    per initial slot and one per effective update, independently of the slot's
    fan-out in the circuit template.
    """

    def __init__(self, initial_slots: Mapping[str, NodeId]) -> None:
        self._slot_names = tuple(initial_slots)
        self._slot_name_set = frozenset(self._slot_names)
        self._versions = {name: [0] for name in self._slot_names}
        self._values = {name: [initial_slots[name]] for name in self._slot_names}
        self._current = dict(initial_slots)
        self._version_count = 1
        self._change_count = 0
        self.stats: list[EnvironmentVersionStats] = [
            EnvironmentVersionStats(version=0, changed_slots=len(initial_slots))
        ]

    @property
    def slot_names(self) -> tuple[str, ...]:
        return self._slot_names

    @property
    def slot_name_set(self) -> frozenset[str]:
        return self._slot_name_set

    @property
    def slot_count(self) -> int:
        return len(self._slot_names)

    @property
    def version_count(self) -> int:
        return self._version_count

    @property
    def change_count(self) -> int:
        return self._change_count

    @property
    def storage_records(self) -> int:
        """Initial slot records plus effective update records."""

        return self.slot_count + self.change_count

    def current_root(self, name: str) -> NodeId:
        try:
            return self._current[name]
        except KeyError as exc:
            raise KeyError(f"Unknown circuit slot {name!r}") from exc

    def root_at(self, name: str, version: int) -> NodeId:
        if not 0 <= version < self._version_count:
            raise IndexError(
                f"Version {version} is outside [0, {self._version_count})."
            )
        try:
            versions = self._versions[name]
            values = self._values[name]
        except KeyError as exc:
            raise KeyError(f"Unknown circuit slot {name!r}") from exc
        position = bisect_right(versions, version) - 1
        return values[position]

    def append_version(self, updates: Mapping[str, NodeId]) -> int:
        unknown = set(updates).difference(self._slot_name_set)
        if unknown:
            raise ValueError(f"Unknown circuit slots: {sorted(unknown)}")

        version = self._version_count
        changed = 0
        for name, value in updates.items():
            if self._current[name] == value:
                continue
            self._current[name] = value
            self._versions[name].append(version)
            self._values[name].append(value)
            changed += 1

        self._version_count += 1
        self._change_count += changed
        self.stats.append(
            EnvironmentVersionStats(version=version, changed_slots=changed)
        )
        return version


@dataclass(frozen=True)
class CircuitClosure:
    """An exact template circuit paired with one slot-environment version."""

    template: CircuitTemplate
    environment: VersionedSlotEnvironment
    version: int

    def __post_init__(self) -> None:
        if self.template.slot_names != self.environment.slot_name_set:
            raise ValueError("Template slots and environment slots must match.")
        if not 0 <= self.version < self.environment.version_count:
            raise IndexError(f"Unknown environment version {self.version}.")

    @classmethod
    def _from_validated(
        cls,
        template: CircuitTemplate,
        environment: VersionedSlotEnvironment,
        version: int,
    ) -> "CircuitClosure":
        """Create a handle after its region directory validated the schema."""

        closure = object.__new__(cls)
        object.__setattr__(closure, "template", template)
        object.__setattr__(closure, "environment", environment)
        object.__setattr__(closure, "version", version)
        return closure

    def materialize(self) -> NodeId:
        """Resolve this closure to a canonical arena root."""

        resolved: list[NodeId] = [self.template.arena.zero] * len(self.template.nodes)
        for node_id, node in enumerate(self.template.nodes):
            if node.op == "slot":
                assert node.slot is not None
                resolved[node_id] = self.environment.root_at(node.slot, self.version)
            elif node.op == "fixed":
                assert node.fixed_root is not None
                resolved[node_id] = node.fixed_root
            elif node.op == "add":
                resolved[node_id] = self.template.arena.add(
                    *(resolved[child] for child in node.children)
                )
            elif node.op == "mul":
                resolved[node_id] = self.template.arena.mul(
                    *(resolved[child] for child in node.children)
                )
            else:  # pragma: no cover - template construction prevents this.
                raise ValueError(f"Unsupported template operation {node.op!r}")
        return resolved[self.template.root]

    def evaluate(
        self,
        valuation: Mapping[Hashable, float],
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> float:
        values: list[float] = [0.0] * len(self.template.nodes)
        root_values: dict[NodeId, float] = {}

        def evaluate_root(root: NodeId) -> float:
            if root not in root_values:
                root_values[root] = self.template.arena.evaluate(
                    root, valuation, selectors
                )
            return root_values[root]

        for node_id, node in enumerate(self.template.nodes):
            if node.op == "slot":
                assert node.slot is not None
                root = self.environment.root_at(node.slot, self.version)
                values[node_id] = evaluate_root(root)
            elif node.op == "fixed":
                assert node.fixed_root is not None
                values[node_id] = evaluate_root(node.fixed_root)
            elif node.op == "add":
                values[node_id] = sum(values[child] for child in node.children)
            elif node.op == "mul":
                value = 1.0
                for child in node.children:
                    value *= values[child]
                values[node_id] = value
            else:  # pragma: no cover - template construction prevents this.
                raise ValueError(f"Unsupported template operation {node.op!r}")
        return values[self.template.root]

    def polynomial(
        self,
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> Polynomial:
        values: list[Polynomial] = [Counter() for _ in self.template.nodes]
        root_values: dict[NodeId, Polynomial] = {}

        def polynomial_for_root(root: NodeId) -> Polynomial:
            if root not in root_values:
                root_values[root] = self.template.arena.polynomial(root, selectors)
            return root_values[root].copy()

        for node_id, node in enumerate(self.template.nodes):
            if node.op == "slot":
                assert node.slot is not None
                root = self.environment.root_at(node.slot, self.version)
                values[node_id] = polynomial_for_root(root)
            elif node.op == "fixed":
                assert node.fixed_root is not None
                values[node_id] = polynomial_for_root(node.fixed_root)
            elif node.op == "add":
                result: Polynomial = Counter()
                for child in node.children:
                    result.update(values[child])
                values[node_id] = result
            elif node.op == "mul":
                result = Counter({(): 1})
                for child in node.children:
                    product: Polynomial = Counter()
                    for left, left_count in result.items():
                        for right, right_count in values[child].items():
                            monomial = tuple(sorted(left + right))
                            product[monomial] += left_count * right_count
                    result = product
                values[node_id] = result
            else:  # pragma: no cover - template construction prevents this.
                raise ValueError(f"Unsupported template operation {node.op!r}")
        return values[self.template.root].copy()


@dataclass(frozen=True)
class UnitClosureVersions:
    """Region directory whose payload is a compact circuit closure."""

    starts: tuple[float, ...]
    template: CircuitTemplate
    environment: VersionedSlotEnvironment

    def __post_init__(self) -> None:
        if not self.starts:
            raise ValueError("A unit must contain at least one region.")
        if len(self.starts) != self.environment.version_count:
            raise ValueError("Each region must have one environment version.")
        if any(left >= right for left, right in zip(self.starts, self.starts[1:])):
            raise ValueError("Region starts must be strictly increasing.")
        if self.template.slot_names != self.environment.slot_name_set:
            raise ValueError("Template slots and environment slots must match.")

    def region_index(self, intervention: float) -> int:
        index = bisect_right(self.starts, intervention) - 1
        if index < 0:
            raise ValueError(
                f"Intervention {intervention} is below indexed domain {self.starts[0]}."
            )
        return index

    def closure_for(self, intervention: float) -> CircuitClosure:
        return CircuitClosure._from_validated(
            self.template,
            self.environment,
            self.region_index(intervention),
        )


class VersionedClosureIndex:
    """Population index for compact template-plus-environment provenance."""

    def __init__(self) -> None:
        self._units: dict[Hashable, UnitClosureVersions] = {}

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
        template: CircuitTemplate,
        environment: VersionedSlotEnvironment,
    ) -> None:
        if unit in self._units:
            raise ValueError(f"Duplicate unit {unit!r}")
        self._units[unit] = UnitClosureVersions(
            tuple(starts), template, environment
        )

    def versions_for(self, unit: Hashable) -> UnitClosureVersions:
        try:
            return self._units[unit]
        except KeyError as exc:
            raise KeyError(f"Unknown indexed unit {unit!r}") from exc

    def closure_for(self, unit: Hashable, intervention: float) -> CircuitClosure:
        return self.versions_for(unit).closure_for(intervention)

    def evaluate(
        self,
        unit: Hashable,
        intervention: float,
        valuation: Mapping[Hashable, float],
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> float:
        return self.closure_for(unit, intervention).evaluate(valuation, selectors)

    def polynomial(
        self,
        unit: Hashable,
        intervention: float,
        selectors: Mapping[Hashable, bool] | None = None,
    ) -> Polynomial:
        return self.closure_for(unit, intervention).polynomial(selectors)


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
        unknown = set(updates).difference(self.template.slot_names)
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
