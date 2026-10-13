"""MCPST leaf: exact overlay + MST construction plan + persistent materialisation.

Every face of the overlay gets one directly addressable provenance root (a
version of a persistent factor tree) and a cached numeric value.  Three
layouts realise the same logical index:

* ``"path"``  - immutable path copying along the source-augmented MST under the
  exact copy cost ``rho_e``; optimal among adjacency-constrained single-parent
  plans (it may branch).
* ``"node"``  - node copying along a covering walk of the face graph (the
  cheaper of the MST shortcut walk and a greedy DFS walk); O(1) amortised
  records per slot change (partial persistence on a chain), and the MST walk
  costs at most twice the cheapest covering walk.
* ``"hybrid"`` (default) - compares the exact path-copy cost of the rho-MST tree
  plan with the surrogate cost of a covering walk whose steps use path copying
  when ``rho <= alpha * c`` and node copying otherwise, and materialises the
  cheaper plan.  The MST minimises the surrogate, not the realised allocation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np

from .overlay import Overlay
from .persistence import INACTIVE, PersistentFactorTree, ancestor_union_size
from .plan import best_walk, boustrophedon, edge_weights, euler_walk, path_walk, spanning_forest
from .semiring import CircuitArena, NodeId

LAYOUTS = ("path", "node", "hybrid")


class _Neumaier:
    """Compensated running sum, so long walks do not drift."""

    __slots__ = ("s", "c")

    def __init__(self, s: float = 0.0, c: float = 0.0) -> None:
        self.s, self.c = s, c

    def add(self, x: float) -> None:
        t = self.s + x
        if abs(self.s) >= abs(x):
            self.c += (self.s - t) + x
        else:
            self.c += (x - t) + self.s
        self.s = t

    def copy(self) -> "_Neumaier":
        return _Neumaier(self.s, self.c)

    @property
    def value(self) -> float:
        return self.s + self.c


@dataclass(frozen=True)
class Piece:
    """One pre-materialised provenance root returned by a query."""

    leaf: "CellIndex"
    face: int

    @property
    def value(self) -> float:
        return float(self.leaf.face_value[self.face])

    def boxes(self) -> list[int]:
        """Global box ids of the atoms under this root (output-sensitive in the slots)."""
        version = int(self.leaf.face_version[self.face])
        return [self.leaf.items[i] for i in self.leaf.pft.active(version)]

    def to_arena(self, arena: CircuitArena, label) -> NodeId:
        version = int(self.leaf.face_version[self.face])
        return self.leaf.pft.to_arena(version, arena, lambda i: label(self.leaf.items[i]))


@dataclass
class CellStats:
    grid_cells: int
    faces: int
    face_edges: int
    slots: int
    tree_weight: float
    walk_cost: float
    sources: int
    pft_nodes: int
    pft_entries: int
    node_copies: int
    path_nodes: int
    locator_words: int
    seconds: dict = field(default_factory=dict)

    @property
    def records(self) -> int:
        """PFT nodes + entries + one root and one value per face + locator."""
        return self.pft_nodes + self.pft_entries + 2 * self.faces + self.locator_words


class CellIndex:
    def __init__(
        self,
        items: Sequence[int],
        ranges: Sequence[Sequence[tuple[int, int]]],
        slots: Sequence[int],
        slab_counts: Sequence[int],
        values: Sequence[float],
        offset: float = 0.0,
        layout: str = "hybrid",
        exact: bool = True,
        walk: str = "best",
        alpha: float = 1.5,
        capacity: int = 2,
        spanning: str = "mst",
        plan_weight: str | None = None,
    ) -> None:
        """``items`` are global box ids; ``ranges``, ``slots`` and ``values`` are
        aligned with them.  ``slots`` is the component tree of each box; a slot
        that holds no box at a point is inactive (semiring zero).

        ``walk`` selects the covering walk for node copying: ``"best"`` (cheaper
        of the MST shortcut walk and the greedy DFS walk), ``"euler"`` or
        ``"snake"`` (boustrophedon over the full grid, ``exact=False`` only).
        ``plan_weight`` overrides the edge weight used to plan (ablations)."""
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}")
        t0 = time.perf_counter()
        self.items = list(items)
        self.values = [float(v) for v in values]
        self.offset = float(offset)
        self.layout = layout
        dense = {s: i for i, s in enumerate(sorted(set(slots)))}
        self.slot_of = [dense[s] for s in slots]
        self.overlay = Overlay(ranges, self.slot_of, slab_counts, exact=exact)
        t1 = time.perf_counter()
        n_slots = max(1, len(dense))
        self.pft = PersistentFactorTree(n_slots, capacity=capacity)
        width = self.pft.width
        edges = self.overlay.edges
        n_faces = self.overlay.n_faces

        def make_walk(step_weights):
            if walk == "snake":
                if exact:
                    raise ValueError("the snake walk needs the full grid (exact=False)")
                return path_walk(boustrophedon(self.overlay.shape), edges, step_weights)
            if walk == "euler":
                return euler_walk(spanning_forest(n_faces, edges, step_weights, method=spanning))
            return best_walk(n_faces, edges, step_weights, method=spanning)

        forest = None
        plan = None
        if layout in ("path", "hybrid"):
            tree_weights = edge_weights(edges, width, plan_weight or "rho", alpha)
            forest = spanning_forest(n_faces, edges, tree_weights, method=spanning)
        if layout in ("node", "hybrid"):
            step_kind = plan_weight or ("count" if layout == "node" else "hybrid")
            plan = make_walk(edge_weights(edges, width, step_kind, alpha))
        if layout == "hybrid":
            # exact path-copy cost of the tree plan versus the walk's surrogate cost
            path_cost = sum(
                ancestor_union_size(list(edges[forest.parent_edge[v]].changes), width)
                for v in range(n_faces)
                if forest.parent_edge[v] >= 0
            )
            use_tree = path_cost <= plan.cost
        else:
            use_tree = layout == "path"
        t2 = time.perf_counter()
        self.face_version = np.full(n_faces, -1, dtype=np.int64)
        self.face_value = np.zeros(n_faces)
        self.face_empty = np.zeros(n_faces, dtype=bool)
        if use_tree:
            self.plan_kind = "tree"
            walk_cost = self._materialise_tree(forest)
        else:
            self.plan_kind = "walk"
            walk_cost = plan.cost
            self._materialise_walk(plan, "node" if layout == "node" else "auto", alpha)
        t3 = time.perf_counter()
        self.stats = CellStats(
            grid_cells=self.overlay.grid_cells,
            faces=n_faces,
            face_edges=len(edges),
            slots=n_slots,
            tree_weight=forest.weight if forest is not None else plan.cost,
            walk_cost=walk_cost,
            sources=sum(1 for kind, _, _ in plan.events if kind == "source") if not use_tree else len(forest.roots),
            pft_nodes=self.pft.node_count,
            pft_entries=self.pft.entries,
            node_copies=self.pft.node_copies,
            path_nodes=self.pft.path_nodes,
            locator_words=self.overlay.locator_words,
            seconds={"overlay": t1 - t0, "plan": t2 - t1, "materialise": t3 - t2},
        )

    # ------------------------------------------------------------------
    def _value_of(self, item: int) -> float:
        return 0.0 if item == INACTIVE else self.values[item]

    def _source(self, face: int) -> tuple[list[int], _Neumaier, int]:
        sig = self.overlay.signature(face)
        acc = _Neumaier(self.offset)
        active = 0
        for item in sorted(i for i in sig if i != INACTIVE):
            acc.add(self.values[item])
            active += 1
        return sig, acc, active

    def _record(self, face: int, version: int, acc: _Neumaier, active: int) -> None:
        if self.face_version[face] < 0:
            self.face_version[face] = version
            self.face_value[face] = acc.value
            self.face_empty[face] = active == 0

    def _apply(self, edge, forward: bool, state: list[int], acc: _Neumaier, active: int):
        changes = {}
        for slot, (in_a, in_b) in edge.changes.items():
            old, new = (in_a, in_b) if forward else (in_b, in_a)
            state[slot] = new
            acc.add(self._value_of(new) - self._value_of(old))
            active += (new != INACTIVE) - (old != INACTIVE)
            changes[slot] = new
        return changes, active

    def _materialise_walk(self, plan, mode: str, alpha: float) -> None:
        state: list[int] = []
        acc = _Neumaier()
        active = 0
        edges = self.overlay.edges
        current = -1
        for kind, face, e in plan.events:
            if kind == "source":
                state, acc, active = self._source(face)
                version = self.pft.build(state)
            else:
                edge = edges[e]
                changes, active = self._apply(edge, current == edge.a, state, acc, active)
                version = self.pft.advance(changes, mode=mode, alpha=alpha)
            current = face
            self._record(face, version, acc, active)

    def _materialise_tree(self, forest) -> float:
        """Depth-first apply/undo along the spanning tree; immutable path copying."""
        edges = self.overlay.edges
        cost = 0.0
        for root in forest.roots:
            state, acc, active = self._source(root)
            version = self.pft.build(state)
            self._record(root, version, acc, active)
            stack = [(root, 0, acc, active)]
            while stack:
                v, k, vacc, vact = stack[-1]
                kids = forest.children[v]
                if k < len(kids):
                    stack[-1] = (v, k + 1, vacc, vact)
                    c = kids[k]
                    edge = edges[forest.parent_edge[c]]
                    cacc = vacc.copy()
                    changes, cact = self._apply(edge, v == edge.a, state, cacc, vact)
                    cost += forest.edge_weight[forest.parent_edge[c]]
                    version = self.pft.derive_path(int(self.face_version[v]), changes)
                    self._record(c, version, cacc, cact)
                    stack.append((c, 0, cacc, cact))
                else:
                    stack.pop()
                    if stack:  # undo the edge back to the parent
                        parent = stack[-1][0]
                        edge = edges[forest.parent_edge[v]]
                        for slot, (in_a, in_b) in edge.changes.items():
                            state[slot] = in_a if parent == edge.a else in_b
        return cost

    # ------------------------------------------------------------------
    def face_of(self, slab: Sequence[int]) -> int:
        return self.overlay.face_of(slab)

    def piece(self, slab: Sequence[int]) -> Piece:
        return Piece(self, self.face_of(slab))


def iip_records(faces: int, slots: int) -> int:
    """Independent per-face materialisation with the same layout and locator-free
    accounting: one full balanced aggregation tree, root and value per face."""
    width = 2
    while width < slots:
        width *= 2
    return faces * (width - 1) + 2 * faces


def iip_bytes(faces: int, slots: int) -> int:
    """Packed IIP: 8 B per aggregation node, 12 B per face (root, float64 value)."""
    width = 2
    while width < slots:
        width *= 2
    return faces * (8 * (width - 1) + 12)
