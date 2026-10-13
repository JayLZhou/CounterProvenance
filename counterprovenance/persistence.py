"""Persistent factor tree (PFT): one directly addressable root per version.

The PFT is a balanced binary tree whose leaves are *slots* (one per component
tree, or one per box).  A version assigns a payload to every slot; payload
``-1`` means "inactive" (contributes the semiring zero).  Two update
mechanisms are supported and can be mixed per update:

* **path copying** (immutable): copy the union of the root paths of the
  changed slots.  Cost ``rho = |Anc(Delta)|`` new nodes, exactly.  Any earlier
  version can be the parent, so construction plans may branch.
* **node copying** (Sarnak-Tarjan / Driscoll-Sarnak-Sleator-Tarjan partial
  persistence): each node owns ``capacity`` version-stamped modification
  entries; a node is copied only when its entries are full.  Because every
  node has in-degree one, this costs O(1) amortised records per changed slot,
  but versions must form a single chain (each update derives from the latest
  version).

Reading version ``v`` costs O(capacity) per visited node.  A version is a
provenance root: its semiring meaning is the sum of the payload atoms of its
active slots, and :meth:`to_arena` materialises that sum as a canonical
circuit on demand.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Callable, Hashable, Mapping, Sequence

from .semiring import CircuitArena, NodeId

INACTIVE = -1


def _encode(payload: int) -> int:
    return -(payload + 2)


def _decode(child: int) -> int:
    return -child - 2


def ancestor_union_size(slots: Sequence[int], width: int) -> int:
    """Number of internal nodes on the union of root paths of ``slots``."""
    affected = set(slots)
    cost = 0
    while width > 1 and affected:
        affected = {i // 2 for i in affected}
        cost += len(affected)
        width //= 2
    return cost


class PersistentFactorTree:
    def __init__(self, n_slots: int, capacity: int = 2) -> None:
        if n_slots < 1:
            raise ValueError("a PFT needs at least one slot")
        if capacity < 1:
            raise ValueError("node copying needs at least one modification entry per node")
        self.n_slots = n_slots
        self.width = 2
        while self.width < n_slots:
            self.width *= 2
        self.height = self.width.bit_length() - 1
        self.capacity = capacity
        self._left: list[int] = []
        self._right: list[int] = []
        self._mods: list[list[tuple[int, int, int]]] = []  # (version, field, value)
        self.root_of: list[int] = []
        self.entries = 0
        self.node_copies = 0
        self.path_nodes = 0
        self._max_entry_version = -1

    # -- accounting --------------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self._left)

    @property
    def records(self) -> int:
        """Internal nodes plus modification entries."""
        return self.node_count + self.entries

    @property
    def version_count(self) -> int:
        return len(self.root_of)

    def packed_bytes(self) -> int:
        """Estimated packed size: 8 B per node, 12 B per entry, 4 B per root."""
        return 8 * self.node_count + 12 * self.entries + 4 * self.version_count

    def transition_cost(self, slots: Sequence[int]) -> int:
        return ancestor_union_size(slots, self.width)

    # -- primitives --------------------------------------------------------
    def _new(self, left: int, right: int) -> int:
        self._left.append(left)
        self._right.append(right)
        self._mods.append([])
        return len(self._left) - 1

    def _read(self, node: int, field: int, version: int) -> int:
        mods = self._mods[node]
        for i in range(len(mods) - 1, -1, -1):
            stamp, f, value = mods[i]
            if f == field and stamp <= version:
                return value
        return self._left[node] if field == 0 else self._right[node]

    # -- construction ------------------------------------------------------
    def build(self, payloads: Sequence[int]) -> int:
        """Create a version from scratch (a new source root)."""
        if len(payloads) != self.n_slots:
            raise ValueError(f"expected {self.n_slots} payloads, got {len(payloads)}")
        level = [_encode(int(p)) for p in payloads]
        level.extend([_encode(INACTIVE)] * (self.width - self.n_slots))
        while len(level) > 1:
            level = [self._new(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        self.root_of.append(level[0])
        return len(self.root_of) - 1

    def derive_path(self, parent: int, changes: Mapping[int, int]) -> int:
        """Immutable path copying from any earlier version (plans may branch)."""
        if parent < self._max_entry_version and parent != len(self.root_of) - 1:
            # entries stamped in (parent, latest] belong to another branch and
            # would become visible to the new version
            raise RuntimeError("cannot branch below a node-copied version")
        version = len(self.root_of)
        self.root_of.append(self.root_of[parent])
        if changes:
            self.root_of[version] = self._path_copy(self.root_of[parent], parent, changes)
        return version

    def advance(self, changes: Mapping[int, int], mode: str = "node", alpha: float = 1.5) -> int:
        """Append a version derived from the latest one.

        ``mode`` is ``"node"``, ``"path"`` or ``"auto"`` (path copying when
        ``rho <= alpha * |changes|``, node copying otherwise).
        """
        if not self.root_of:
            raise RuntimeError("build() a first version before advancing")
        latest = len(self.root_of) - 1
        version = latest + 1
        self.root_of.append(self.root_of[latest])
        if not changes:
            return version
        if mode == "auto":
            rho = self.transition_cost(list(changes))
            mode = "path" if rho <= alpha * len(changes) else "node"
        if mode == "path":
            self.root_of[version] = self._path_copy(self.root_of[latest], latest, changes)
            return version
        if mode != "node":
            raise ValueError(f"unknown update mode {mode!r}")
        for slot, payload in changes.items():
            self._node_copy_update(version, int(slot), _encode(int(payload)))
        return version

    def _path_copy(self, root: int, read_version: int, changes: Mapping[int, int]) -> int:
        positions = sorted(changes)
        refs = [_encode(int(changes[p])) for p in positions]
        before = self.node_count

        def copy(node: int, lo: int, hi: int, a: int, b: int) -> int:
            if a >= b:
                return node
            if hi - lo == 1:
                return refs[a]
            mid = (lo + hi) // 2
            split = bisect_left(positions, mid, a, b)
            new_left = copy(self._read(node, 0, read_version), lo, mid, a, split)
            new_right = copy(self._read(node, 1, read_version), mid, hi, split, b)
            return self._new(new_left, new_right)

        new_root = copy(root, 0, self.width, 0, len(positions))
        self.path_nodes += self.node_count - before
        return new_root

    def _node_copy_update(self, version: int, slot: int, value: int) -> None:
        path: list[tuple[int, int]] = []
        node = self.root_of[version]
        for depth in range(self.height):
            direction = (slot >> (self.height - depth - 1)) & 1
            path.append((node, direction))
            if depth + 1 < self.height:
                node = self._read(node, direction, version)
        index = len(path) - 1
        while True:
            node, field = path[index]
            mods = self._mods[node]
            for i in range(len(mods) - 1, -1, -1):
                stamp, f, _ = mods[i]
                if stamp == version and f == field:
                    mods[i] = (version, field, value)
                    return
                if stamp < version:
                    break
            if len(mods) < self.capacity:
                mods.append((version, field, value))
                self.entries += 1
                self._max_entry_version = version
                return
            left = self._read(node, 0, version)
            right = self._read(node, 1, version)
            if field == 0:
                left = value
            else:
                right = value
            copy = self._new(left, right)
            self.node_copies += 1
            if index == 0:
                self.root_of[version] = copy
                return
            index -= 1
            value = copy

    # -- reading -----------------------------------------------------------
    def payloads(self, version: int) -> list[int]:
        out: list[int] = []

        def visit(node: int, depth: int) -> None:
            if depth == self.height:
                out.append(_decode(node))
                return
            visit(self._read(node, 0, version), depth + 1)
            visit(self._read(node, 1, version), depth + 1)

        visit(self.root_of[version], 0)
        return out[: self.n_slots]

    def slot(self, version: int, slot: int) -> int:
        node = self.root_of[version]
        for depth in range(self.height):
            node = self._read(node, (slot >> (self.height - depth - 1)) & 1, version)
        return _decode(node)

    def active(self, version: int) -> list[int]:
        return [p for p in self.payloads(version) if p != INACTIVE]

    def to_arena(
        self,
        version: int,
        arena: CircuitArena,
        atom_of: Callable[[int], Hashable],
    ) -> NodeId:
        """Materialise a version as a canonical semiring sum circuit."""

        def visit(node: int, depth: int) -> NodeId:
            if depth == self.height:
                payload = _decode(node)
                return arena.zero if payload == INACTIVE else arena.atom(atom_of(payload))
            return arena.add(
                visit(self._read(node, 0, version), depth + 1),
                visit(self._read(node, 1, version), depth + 1),
            )

        return visit(self.root_of[version], 0)
