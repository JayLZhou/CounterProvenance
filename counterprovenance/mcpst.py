"""Minimum-change persistent factor-tree indexes over counterfactual cells.

The module implements the physical model used by the MCPST paper design:

* every cell selects one payload for each component-tree slot;
* a balanced persistent factor tree aggregates the slots;
* adjacent cells are labelled by the slots whose payloads change;
* a source-augmented MST chooses the single-parent construction plan;
* path copying materializes a directly addressable root for every cell.

Payload nodes are shared separately and are not charged as PFT internal nodes.
PFT nodes created on different construction branches are deliberately not
globally hash-consed: this keeps transition weights fixed and additive, which
is the cost model for the MST optimality theorem.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Literal, Mapping, Sequence

import numpy as np


Payload = Hashable
Objective = Literal["pft", "changed", "unweighted"]


def _as_payload(value: object) -> Payload:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
    except TypeError as exc:  # pragma: no cover - caller contract.
        raise TypeError(f"PFT payloads must be hashable, got {value!r}") from exc
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PFTNode:
    """One immutable internal node in a persistent factor tree."""

    left: int
    right: int


class BalancedPFTLayout:
    """A fixed balanced layout over component-tree slots."""

    def __init__(self, component_count: int) -> None:
        if component_count < 1:
            raise ValueError("A PFT needs at least one component slot.")
        self.component_count = component_count
        self.padded_leaf_count = 1 << (component_count - 1).bit_length()
        self.height = self.padded_leaf_count.bit_length() - 1
        self.build_cost = self.padded_leaf_count - 1

    def transition_cost(self, changed_components: Iterable[int]) -> int:
        """Return the exact number of internal ancestors that must be copied."""

        affected = set(changed_components)
        if any(index < 0 or index >= self.component_count for index in affected):
            raise IndexError("Changed component is outside the PFT leaf range.")

        cost = 0
        width = self.padded_leaf_count
        while width > 1 and affected:
            affected = {index // 2 for index in affected}
            cost += len(affected)
            width //= 2
        return cost


class PFTArena:
    """Append-only PFT nodes with parent-based structural sharing."""

    _PADDING_REF = -1

    def __init__(self, layout: BalancedPFTLayout) -> None:
        self.layout = layout
        self._nodes: list[PFTNode] = []
        self._payload_refs: dict[tuple[int, Payload], int] = {}
        self._ref_payloads: dict[int, tuple[int, Payload]] = {}
        self._next_payload_ref = -2

    @property
    def nodes(self) -> tuple[PFTNode, ...]:
        return tuple(self._nodes)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def payload_count(self) -> int:
        return len(self._payload_refs)

    def _payload_ref(self, component: int, payload: object) -> int:
        normalized = _as_payload(payload)
        key = (component, normalized)
        existing = self._payload_refs.get(key)
        if existing is not None:
            return existing
        ref = self._next_payload_ref
        self._next_payload_ref -= 1
        self._payload_refs[key] = ref
        self._ref_payloads[ref] = key
        return ref

    def _append(self, left: int, right: int) -> int:
        node_id = len(self._nodes)
        self._nodes.append(PFTNode(left, right))
        return node_id

    def build(self, payloads: Sequence[object]) -> tuple[int, int]:
        """Build one root independently and return ``(root, new_nodes)``."""

        if len(payloads) != self.layout.component_count:
            raise ValueError(
                f"Expected {self.layout.component_count} payloads, got {len(payloads)}."
            )
        before = self.node_count
        level = [
            self._payload_ref(component, payload)
            for component, payload in enumerate(payloads)
        ]
        level.extend(
            [self._PADDING_REF]
            * (self.layout.padded_leaf_count - self.layout.component_count)
        )
        while len(level) > 1:
            level = [
                self._append(level[index], level[index + 1])
                for index in range(0, len(level), 2)
            ]
        return level[0], self.node_count - before

    def batch_update(
        self,
        parent_root: int,
        updates: Mapping[int, object],
    ) -> tuple[int, int]:
        """Path-copy the union of root paths for effective slot updates."""

        if not updates:
            return parent_root, 0
        positions = tuple(sorted(updates))
        if positions[0] < 0 or positions[-1] >= self.layout.component_count:
            raise IndexError("Updated component is outside the PFT leaf range.")
        payload_refs = tuple(
            self._payload_ref(position, updates[position]) for position in positions
        )
        before = self.node_count

        def copy(ref: int, left: int, right: int, lo: int, hi: int) -> int:
            if lo >= hi:
                return ref
            if right - left == 1:
                if hi - lo != 1 or positions[lo] != left:
                    raise AssertionError("Malformed batch-update leaf partition.")
                return payload_refs[lo]
            if ref < 0:
                raise AssertionError(
                    "An internal PFT interval cannot be a payload ref."
                )

            midpoint = (left + right) // 2
            split = bisect_left(positions, midpoint, lo, hi)
            node = self._nodes[ref]
            new_left = copy(node.left, left, midpoint, lo, split)
            new_right = copy(node.right, midpoint, right, split, hi)
            return self._append(new_left, new_right)

        root = copy(
            parent_root,
            0,
            self.layout.padded_leaf_count,
            0,
            len(positions),
        )
        return root, self.node_count - before

    def payloads(self, root: int) -> tuple[Payload, ...]:
        """Decode component payloads for exactness tests and small explanations."""

        refs: list[int] = []

        def visit(ref: int, width: int) -> None:
            if width == 1:
                refs.append(ref)
                return
            if ref < 0:
                raise AssertionError(
                    "An internal PFT interval cannot be a payload ref."
                )
            node = self._nodes[ref]
            visit(node.left, width // 2)
            visit(node.right, width // 2)

        visit(root, self.layout.padded_leaf_count)
        decoded: list[Payload] = []
        for ref in refs[: self.layout.component_count]:
            try:
                _, payload = self._ref_payloads[ref]
            except KeyError as exc:
                raise AssertionError(
                    f"Unknown payload ref {ref} in live PFT leaf."
                ) from exc
            decoded.append(payload)
        return tuple(decoded)


@dataclass(frozen=True, slots=True)
class CellTransition:
    """One candidate transition between adjacent counterfactual cells."""

    left: int
    right: int
    changed: tuple[int, ...]
    weight: int

    @property
    def changed_count(self) -> int:
        return len(self.changed)


@dataclass(frozen=True, slots=True)
class SelectedEdge:
    """One edge selected by a source-augmented spanning plan."""

    left: int
    right: int
    changed: tuple[int, ...]
    weight: int
    source_edge: bool


@dataclass(frozen=True)
class MCPSTPlan:
    """A source-rooted construction tree and its actual PFT cost."""

    parents: tuple[int, ...]
    parent_changes: tuple[tuple[int, ...], ...]
    parent_weights: tuple[int, ...]
    traversal: tuple[int, ...]
    selected_edges: tuple[SelectedEdge, ...]
    total_cost: int
    source_roots: int
    objective: Objective


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def grid_adjacency(x_cells: int, y_cells: int) -> tuple[tuple[int, int], ...]:
    """Return four-neighbour edges for a row-major rectangular cell grid."""

    if x_cells < 1 or y_cells < 1:
        raise ValueError("A cell grid needs positive dimensions.")
    edges: list[tuple[int, int]] = []
    for y_index in range(y_cells):
        row_start = y_index * x_cells
        for x_index in range(x_cells - 1):
            left = row_start + x_index
            edges.append((left, left + 1))
    for y_index in range(y_cells - 1):
        row_start = y_index * x_cells
        next_row = row_start + x_cells
        for x_index in range(x_cells):
            edges.append((row_start + x_index, next_row + x_index))
    return tuple(edges)


def transitions_from_states(
    states: np.ndarray,
    adjacency: Iterable[tuple[int, int]],
    layout: BalancedPFTLayout,
) -> tuple[CellTransition, ...]:
    """Label candidate edges by exact component changes and PFT copy costs."""

    state_array = np.asarray(states)
    if state_array.ndim != 2 or state_array.shape[1] != layout.component_count:
        raise ValueError("states must have shape (cell_count, layout.component_count).")
    cell_count = state_array.shape[0]
    transitions: list[CellTransition] = []
    for left, right in adjacency:
        if not 0 <= left < cell_count or not 0 <= right < cell_count:
            raise IndexError("Adjacency endpoint is outside the cell range.")
        changed = tuple(
            int(value)
            for value in np.flatnonzero(state_array[left] != state_array[right])
        )
        transitions.append(
            CellTransition(left, right, changed, layout.transition_cost(changed))
        )
    return tuple(transitions)


def build_spanning_plan(
    cell_count: int,
    transitions: Sequence[CellTransition],
    layout: BalancedPFTLayout,
    objective: Objective = "pft",
) -> MCPSTPlan:
    """Build a source-augmented spanning plan with Kruskal's algorithm.

    ``objective='pft'`` is MCPST.  The other objectives are deterministic
    ablations; ``total_cost`` always reports the actual PFT node-copy cost.
    """

    if cell_count < 1:
        raise ValueError("At least one cell is required.")
    if objective not in {"pft", "changed", "unweighted"}:
        raise ValueError(f"Unknown spanning objective {objective!r}.")

    source = cell_count
    # (objective score, source tie-break, left, right, actual weight, changes)
    candidates: list[tuple[int, int, int, int, int, tuple[int, ...]]] = []
    for edge in transitions:
        if not 0 <= edge.left < cell_count or not 0 <= edge.right < cell_count:
            raise IndexError("Transition endpoint is outside the cell range.")
        if objective == "pft":
            score = edge.weight
        elif objective == "changed":
            score = edge.changed_count
        else:
            score = 1
        candidates.append((score, 0, edge.left, edge.right, edge.weight, edge.changed))

    if objective == "pft":
        source_score = layout.build_cost
    elif objective == "changed":
        source_score = layout.component_count
    else:
        source_score = 2
    for cell in range(cell_count):
        candidates.append((source_score, 1, source, cell, layout.build_cost, ()))

    candidates.sort(key=lambda item: item[:4])
    disjoint = _DisjointSet(cell_count + 1)
    selected: list[SelectedEdge] = []
    for _, source_flag, left, right, actual_weight, changed in candidates:
        if not disjoint.union(left, right):
            continue
        selected.append(
            SelectedEdge(
                left=left,
                right=right,
                changed=changed,
                weight=actual_weight,
                source_edge=bool(source_flag),
            )
        )
        if len(selected) == cell_count:
            break
    if len(selected) != cell_count:
        raise ValueError("Candidate transitions do not connect all cells to source.")

    tree: list[list[tuple[int, SelectedEdge]]] = [[] for _ in range(cell_count + 1)]
    for edge in selected:
        tree[edge.left].append((edge.right, edge))
        tree[edge.right].append((edge.left, edge))

    parents = [-2] * cell_count
    changes: list[tuple[int, ...]] = [()] * cell_count
    weights = [0] * cell_count
    traversal: list[int] = []
    visited = {source}
    pending: deque[int] = deque((source,))
    while pending:
        parent = pending.popleft()
        for child, edge in tree[parent]:
            if child in visited:
                continue
            visited.add(child)
            pending.append(child)
            if child == source:
                continue
            parents[child] = -1 if parent == source else parent
            changes[child] = edge.changed
            weights[child] = edge.weight
            traversal.append(child)

    if len(traversal) != cell_count or any(parent == -2 for parent in parents):
        raise AssertionError("Rooted MST traversal did not reach every cell.")

    return MCPSTPlan(
        parents=tuple(parents),
        parent_changes=tuple(changes),
        parent_weights=tuple(weights),
        traversal=tuple(traversal),
        selected_edges=tuple(selected),
        total_cost=sum(edge.weight for edge in selected),
        source_roots=sum(edge.source_edge for edge in selected),
        objective=objective,
    )


@dataclass(frozen=True)
class MCPSTMaterialization:
    """All directly addressable PFT roots produced by one spanning plan."""

    arena: PFTArena
    roots: tuple[int, ...]
    plan: MCPSTPlan

    @property
    def internal_nodes(self) -> int:
        return self.arena.node_count


def materialize_plan(
    states: np.ndarray,
    layout: BalancedPFTLayout,
    plan: MCPSTPlan,
) -> MCPSTMaterialization:
    """Materialize every cell root and verify the predicted copy count."""

    state_array = np.asarray(states)
    if state_array.shape != (len(plan.parents), layout.component_count):
        raise ValueError(
            "states must match the plan cell count and PFT component count."
        )

    arena = PFTArena(layout)
    roots = [0] * len(plan.parents)
    for cell in plan.traversal:
        parent = plan.parents[cell]
        if parent == -1:
            root, created = arena.build(state_array[cell])
        else:
            changed = plan.parent_changes[cell]
            updates = {index: state_array[cell, index] for index in changed}
            root, created = arena.batch_update(roots[parent], updates)
        if created != plan.parent_weights[cell]:
            raise AssertionError(
                f"Cell {cell} allocated {created} PFT nodes; "
                f"plan predicted {plan.parent_weights[cell]}."
            )
        roots[cell] = root

    if arena.node_count != plan.total_cost:
        raise AssertionError(
            f"Materialization allocated {arena.node_count} nodes; "
            f"plan predicted {plan.total_cost}."
        )
    return MCPSTMaterialization(arena, tuple(roots), plan)


def independent_pft_nodes(cell_count: int, layout: BalancedPFTLayout) -> int:
    """Exact IIP internal-node count under the same fixed PFT layout."""

    if cell_count < 0:
        raise ValueError("cell_count cannot be negative.")
    return cell_count * layout.build_cost


def fully_duplicated_pft_records(
    cell_count: int,
    layout: BalancedPFTLayout,
    payload_records_per_component: int = 1,
) -> int:
    """Count a baseline that duplicates every active payload and PFT per cell.

    The count includes one root-directory record per cell. Padding references
    are constants and do not create payload records.
    """

    if cell_count < 0:
        raise ValueError("cell_count cannot be negative.")
    if payload_records_per_component < 1:
        raise ValueError("Each active component payload needs at least one record.")
    records_per_cell = (
        layout.component_count * payload_records_per_component + layout.build_cost + 1
    )
    return cell_count * records_per_cell


class GridPointLocator:
    """Exact point location for a full two-dimensional threshold grid.

    Threshold equality belongs to the lower/left cell, matching numerical
    regression-tree predicates of the form ``feature <= threshold``.
    """

    def __init__(
        self,
        x_thresholds: Sequence[float],
        y_thresholds: Sequence[float],
        domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        query_dtype: object | None = None,
    ) -> None:
        self.x_thresholds = np.asarray(x_thresholds, dtype=float)
        self.y_thresholds = np.asarray(y_thresholds, dtype=float)
        self.query_dtype = None if query_dtype is None else np.dtype(query_dtype)
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Grid domain must have positive width and height.")
        if np.any(np.diff(self.x_thresholds) <= 0) or np.any(
            np.diff(self.y_thresholds) <= 0
        ):
            raise ValueError("Grid thresholds must be strictly increasing.")
        if (
            np.any(self.x_thresholds <= self.x_min)
            or np.any(self.x_thresholds >= self.x_max)
            or np.any(self.y_thresholds <= self.y_min)
            or np.any(self.y_thresholds >= self.y_max)
        ):
            raise ValueError("Grid thresholds must lie strictly inside the domain.")
        self.x_cells = len(self.x_thresholds) + 1
        self.y_cells = len(self.y_thresholds) + 1
        self.cell_count = self.x_cells * self.y_cells

    def cell_id(self, x_value: float, y_value: float) -> int:
        if self.query_dtype is not None:
            x_value = float(np.asarray(x_value, dtype=self.query_dtype))
            y_value = float(np.asarray(y_value, dtype=self.query_dtype))
        if not self.x_min <= x_value <= self.x_max:
            raise ValueError(f"x={x_value} is outside the indexed domain.")
        if not self.y_min <= y_value <= self.y_max:
            raise ValueError(f"y={y_value} is outside the indexed domain.")
        x_index = int(np.searchsorted(self.x_thresholds, x_value, side="left"))
        y_index = int(np.searchsorted(self.y_thresholds, y_value, side="left"))
        return y_index * self.x_cells + x_index

    def cell_ids(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(
            points,
            dtype=float if self.query_dtype is None else self.query_dtype,
        )
        if values.dtype != np.dtype(float):
            values = values.astype(float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("points must have shape (query_count, 2).")
        if np.any(values[:, 0] < self.x_min) or np.any(values[:, 0] > self.x_max):
            raise ValueError("At least one x coordinate is outside the indexed domain.")
        if np.any(values[:, 1] < self.y_min) or np.any(values[:, 1] > self.y_max):
            raise ValueError("At least one y coordinate is outside the indexed domain.")
        x_indices = np.searchsorted(self.x_thresholds, values[:, 0], side="left")
        y_indices = np.searchsorted(self.y_thresholds, values[:, 1], side="left")
        return y_indices * self.x_cells + x_indices


@dataclass(frozen=True)
class MCPSTGridIndex:
    """A 2D point locator paired with persistent roots and cached values."""

    locator: GridPointLocator
    roots: tuple[int, ...]
    cached_values: np.ndarray
    _root_array: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.cached_values, dtype=float)
        if len(self.roots) != self.locator.cell_count:
            raise ValueError("Every grid cell must have one persistent root.")
        if values.shape != (self.locator.cell_count,):
            raise ValueError("Every grid cell must have one cached value.")
        object.__setattr__(self, "cached_values", values)
        object.__setattr__(self, "_root_array", np.asarray(self.roots, dtype=np.int64))

    def query(self, x_value: float, y_value: float) -> tuple[float, int]:
        cell = self.locator.cell_id(x_value, y_value)
        return float(self.cached_values[cell]), self.roots[cell]

    def query_batch(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cells = self.locator.cell_ids(points)
        return self.cached_values[cells], self._root_array[cells]
