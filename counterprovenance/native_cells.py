"""One cell graph built in C++ (native/cell_graph.cpp): the same cells, edges, MST and persistent factor tree as
``cellindex.CellIndex(layout="path")``, with the same interface for the index (``tests/test_native_cells.py``)."""

from __future__ import annotations

import bisect
import ctypes as C
import os
import subprocess
import time
from pathlib import Path
from typing import Hashable, Callable, Sequence

import numpy as np

from .cellindex import CellStats, Piece
from .persistence import INACTIVE

_DIR = Path(__file__).parent / "native"
_SRC = _DIR / "cell_graph.cpp"
_LIB = None
_I64, _F64, _I32, _U32, _U8 = (C.POINTER(C.c_int64), C.POINTER(C.c_double), C.POINTER(C.c_int32),
                               C.POINTER(C.c_uint32), C.POINTER(C.c_uint8))


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    so = _DIR / f"cell_graph_{os.uname().machine}.so"
    if not so.exists() or so.stat().st_mtime < _SRC.stat().st_mtime:
        tmp = so.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off", "-pthread", str(_SRC),
                        "-o", str(tmp)], check=True, capture_output=True)
        os.replace(tmp, so)
    lib = C.CDLL(str(so))
    lib.tw_cell_graph.restype = C.c_void_p
    lib.tw_cell_graph.argtypes = [C.c_int, C.c_int64, _I64, _I64, _F64, _I64, C.c_double, C.c_int, C.c_int64, _I64]
    lib.tw_cg_info.argtypes = [C.c_void_p, _I64, _F64]
    lib.tw_cg_ncuts.restype = C.c_int64
    lib.tw_cg_ncuts.argtypes = [C.c_void_p, C.c_int]
    lib.tw_cg_cuts.argtypes = [C.c_void_p, C.c_int, _I64]
    lib.tw_cg_take.argtypes = [C.c_void_p, _U32, _F64, _I32, _U8, _I32, _I32, _F64]
    lib.tw_cg_free.argtypes = [C.c_void_p]
    lib.tw_cell_log.restype = C.c_void_p
    lib.tw_cell_log.argtypes = [C.c_int, C.c_int64, _I64, _I64, _F64, _I64, C.c_double, C.c_int, C.c_int64, C.c_int64, _I64]
    lib.tw_cell_log_seed.restype = C.c_void_p
    lib.tw_cell_log_seed.argtypes = [C.c_int, C.c_int64, _I64, _I64, _F64, _I64, C.c_double, C.c_int, C.c_int64, C.c_int64,
                                     C.c_uint64, _I64]
    lib.tw_cl_info.argtypes = [C.c_void_p, _I64]
    lib.tw_cl_take.argtypes = [C.c_void_p, _U32, _F64, _I64, _U8, _I32, _I32, _I32]
    lib.tw_lean_table.restype = C.c_void_p
    lib.tw_lean_table.argtypes = [C.c_int, C.c_int64, _I64, _I64, _F64, _I64, C.c_double, C.c_int, C.c_int64, _I64]
    lib.tw_lean_take.argtypes = [C.c_void_p, _F64]
    _LIB = lib
    return lib


def available() -> bool:
    try:
        _load()
        return True
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        return False


class _Overlay:
    """The part of ``overlay.Overlay`` that a query needs."""

    def __init__(self, cuts, slab_counts, face_of_cell, n_faces) -> None:
        self.cuts = cuts
        self.d = len(cuts)
        self.slab_counts = list(slab_counts)
        self.shape = tuple(len(c) + 1 for c in cuts)
        self.strides = []
        stride = 1
        for k in range(self.d - 1, -1, -1):
            self.strides.insert(0, stride)
            stride *= self.shape[k]
        self.grid_cells = stride
        self.face_of_cell = face_of_cell
        self.n_faces = n_faces

    def cell_of(self, slab: Sequence[int]) -> int:
        return sum(bisect.bisect_right(self.cuts[k], int(slab[k])) * self.strides[k] for k in range(self.d))

    def face_of(self, slab: Sequence[int]) -> int:
        return int(self.face_of_cell[self.cell_of(slab)])

    @property
    def locator_words(self) -> int:
        return self.grid_cells + sum(len(c) for c in self.cuts)


class _Pft:
    """Read-only persistent factor tree: a version is a root node."""

    def __init__(self, left: np.ndarray, right: np.ndarray, width: int, values: np.ndarray) -> None:
        self.left, self.right, self.width = left, right, width
        self.height = width.bit_length() - 1
        self.values = values
        self._sums = None
        self._left = None  # Python lists for the prefix sums of weighted reads, made on their first use

    @property
    def sums(self) -> np.ndarray:
        """Subtree sums of the nodes, only for weighted reads: not stored with the index, but derived on first
        use from the children in ``height`` vectorized passes, since a node is at most ``height`` levels above
        the items."""
        if self._sums is None:
            vals = np.r_[np.asarray(self.values, dtype=np.float64), 0.0]   # item -1 (no item) reads 0
            def total(child, sums):
                return np.where(child >= 0, sums[np.maximum(child, 0)], vals[np.where(child >= 0, -1, -child - 2)])
            sums = np.zeros(len(self.left), dtype=np.float64)
            for _ in range(max(1, self.height)):
                sums = total(self.left, sums) + total(self.right, sums)
            self._sums = sums
        return self._sums

    def _lists(self) -> None:
        self._left, self._right, self._sums = self.left.tolist(), self.right.tolist(), self.sums.tolist()
        self._vals = self.values.tolist()

    def _total(self, child: int) -> float:
        if child >= 0:
            return self._sums[child]
        item = -child - 2
        return 0.0 if item < 0 else self._vals[item]

    def prefix(self, root: int, p: int) -> float:
        """Sum of the leaf values of the slots below ``p``: one root-to-leaf path, O(log M)."""
        if p <= 0:
            return 0.0
        if self._left is None:
            self._lists()
        if p >= self.width:
            return self._sums[root]
        left, right = self._left, self._right
        node, lo, span, out = root, 0, self.width, 0.0
        for _ in range(self.height):
            span //= 2
            if p >= lo + span:
                out += self._total(left[node])
                node = right[node]
                lo += span
            else:
                node = left[node]
            if lo == p:
                return out
        return out

    def items(self, root: int) -> np.ndarray:
        """The items of the active slots, one numpy step per level."""
        f = np.array([root], dtype=np.int32)
        for _ in range(self.height):
            f = np.concatenate((self.left[f], self.right[f]))
        f = -f.astype(np.int64) - 2
        return f[f >= 0]

    def active(self, root: int) -> list[int]:
        out, stack = [], [(int(root), 0)]
        while stack:
            node, depth = stack.pop()
            if depth == self.height:
                item = -node - 2
                if item != INACTIVE:
                    out.append(item)
                continue
            stack.append((int(self.right[node]), depth + 1))
            stack.append((int(self.left[node]), depth + 1))
        return out

    def to_arena(self, root: int, arena, atom_of: Callable[[int], Hashable]):
        def visit(node: int, depth: int):
            if depth == self.height:
                item = -node - 2
                return arena.zero if item == INACTIVE else arena.atom(atom_of(item))
            return arena.add(visit(int(self.left[node]), depth + 1), visit(int(self.right[node]), depth + 1))
        return visit(int(root), 0)


class NativeCellIndex:
    """Same attributes as ``CellIndex`` for the index and its queries; a version is a root node of the tree."""

    plan_kind = "tree"

    def __init__(self, items: Sequence[int], ranges, slots: Sequence[int], slab_counts: Sequence[int], values,
                 offset: float = 0.0, threads: int = 1, max_cells: int = 1 << 40) -> None:
        t0 = time.perf_counter()
        lib = _load()
        self.items = list(items)
        d = len(slab_counts)
        n = len(self.items)
        dense = {s: i for i, s in enumerate(sorted(set(slots)))}
        slot_of = np.array([dense[s] for s in slots], dtype=np.int64)
        rr = np.asarray(ranges, dtype=np.int64).reshape(n, d, 2)
        vals = np.asarray(values, dtype=np.float64)
        counts = np.asarray(slab_counts, dtype=np.int64)
        self.values = vals
        self.offset = float(offset)
        status = C.c_int64(0)
        h = lib.tw_cell_graph(d, n, rr.ctypes.data_as(_I64), slot_of.ctypes.data_as(_I64), vals.ctypes.data_as(_F64),
                              counts.ctypes.data_as(_I64), self.offset, threads, int(max_cells), C.byref(status))
        try:
            if status.value != 0:
                raise MemoryError(f"cell graph: status {status.value}")
            info = (C.c_int64 * 8)()
            weight = C.c_double(0.0)
            lib.tw_cg_info(h, info, C.byref(weight))
            K, faces, n_edges, labels, labelled, roots, nodes, width = list(info)
            cuts = []
            for k in range(d):
                c = np.empty(lib.tw_cg_ncuts(h, k), dtype=np.int64)
                lib.tw_cg_cuts(h, k, c.ctypes.data_as(_I64))
                cuts.append(c.tolist())
            face_of_cell = np.empty(K, dtype=np.uint32)
            self.face_value = np.empty(faces, dtype=np.float64)
            self.face_version = np.empty(faces, dtype=np.int32)
            self.face_empty = np.empty(faces, dtype=bool)
            left, right = np.empty(nodes, dtype=np.int32), np.empty(nodes, dtype=np.int32)
            sums = np.empty(nodes, dtype=np.float64)
            lib.tw_cg_take(h, face_of_cell.ctypes.data_as(_U32), self.face_value.ctypes.data_as(_F64),
                           self.face_version.ctypes.data_as(_I32), self.face_empty.ctypes.data_as(_U8),
                           left.ctypes.data_as(_I32), right.ctypes.data_as(_I32), sums.ctypes.data_as(_F64))
        finally:
            lib.tw_cg_free(h)
        self.overlay = _Overlay(cuts, slab_counts, face_of_cell, faces)
        del sums   # derived again only for weighted reads (``_Pft.sums``)
        self.pft = _Pft(left, right, int(width), vals)
        self.slot_trees = sorted(dense)   # the tree of every slot, ascending
        self.item_trees = np.asarray(slots, dtype=np.int64)
        self.stats = CellStats(grid_cells=K, faces=faces, face_edges=n_edges, slots=len(dense),
                               tree_weight=weight.value, walk_cost=weight.value, sources=roots, pft_nodes=nodes,
                               pft_entries=0, node_copies=0, path_nodes=nodes, locator_words=self.overlay.locator_words,
                               seconds={"native": time.perf_counter() - t0})

    def face_of(self, slab: Sequence[int]) -> int:
        return self.overlay.face_of(slab)

    def piece(self, slab: Sequence[int]) -> Piece:
        return Piece(self, self.face_of(slab))


def _small(a: np.ndarray, hi: int, signed: bool = False) -> np.ndarray:
    """``a`` in the smallest integer type that holds 0..hi (``signed``: -1..hi)."""
    for dt in ((np.int8, np.int16, np.int32) if signed else (np.uint8, np.uint16, np.uint32)):
        if hi <= np.iinfo(dt).max:
            return a.astype(dt)
    return a.astype(np.int64)


class _Log:
    """The leaves of every cell as a log: the cells follow a depth-first walk of the minimum spanning tree, the log holds
    the changes (slot, item) from each cell to the next one, and a checkpoint holds all slots after every ``spacing``
    changes.  A version is the length of the log at a cell: its leaves are the last checkpoint before it and the
    changes in between, O(slots + spacing).  ``direct``: a small group keeps one row per cell instead (version = row).
    Items are stored +1 (0: the slot has no item)."""

    def __init__(self, slot: np.ndarray, item: np.ndarray, snap: np.ndarray, n_slots: int, spacing: int, direct: bool,
                 n_items: int, values: np.ndarray) -> None:
        self.n_slots, self.spacing, self.direct, self.values = int(n_slots), int(spacing), bool(direct), values
        self.slot = _small(slot, max(0, n_slots - 1))
        self.item = _small(item + 1, n_items)
        self.snap = _small(snap + 1, n_items)
        self.changes = len(self.slot)

    @property
    def nbytes(self) -> int:
        return self.slot.nbytes + self.item.nbytes + self.snap.nbytes

    @property
    def records(self) -> int:
        """Stored leaves: the changes of the log and the entries of its checkpoints (or rows)."""
        return len(self.item) + len(self.snap)

    def state(self, version: int) -> np.ndarray:
        """The item of every slot (-1: none)."""
        g = self.n_slots
        if self.direct:
            return self.snap[version * g:(version + 1) * g].astype(np.int64) - 1
        k = version // self.spacing
        st = self.snap[k * g:(k + 1) * g].astype(np.int64)
        a = k * self.spacing
        if version > a:
            st[self.slot[a:version]] = self.item[a:version]   # in order: the last change of a slot stays
        return st - 1

    def items(self, version: int) -> np.ndarray:
        st = self.state(version)
        return st[st >= 0]

    def active(self, version: int) -> list[int]:
        return self.items(version).tolist()

    def to_arena(self, version: int, arena, atom_of: Callable[[int], Hashable]):
        return arena.add_all([arena.atom(atom_of(int(i))) for i in self.items(version)])

    def prefix(self, version: int, p: int) -> float:
        """Sum of the leaf values of the slots below ``p``, O(slots + spacing)."""
        st = self.state(version)[:max(0, p)]
        return float(self.values[st[st >= 0]].sum())


class LogCellIndex:
    """The cell graph of ``NativeCellIndex`` with the leaves in a log along the minimum spanning tree (``_Log``) instead
    of a persistent tree: the edges are weighted by their number of changes, and ``face_version`` is the version of a
    cell in the log.  ``spacing_factor`` times the slots is the number of changes between two checkpoints."""

    plan_kind = "tree"

    def __init__(self, items: Sequence[int], ranges, slots: Sequence[int], slab_counts: Sequence[int], values,
                 offset: float = 0.0, threads: int = 1, max_cells: int = 1 << 40, spacing_factor: float = 1.0,
                 seed: int = 0) -> None:
        t0 = time.perf_counter()
        lib = _load()
        self.items = list(items)
        d = len(slab_counts)
        n = len(self.items)
        dense = {s: i for i, s in enumerate(sorted(set(slots)))}
        slot_of = np.array([dense[s] for s in slots], dtype=np.int64)
        rr = np.asarray(ranges, dtype=np.int64).reshape(n, d, 2)
        vals = np.asarray(values, dtype=np.float64)
        counts = np.asarray(slab_counts, dtype=np.int64)
        self.values = vals
        self.offset = float(offset)
        status = C.c_int64(0)
        spacing = max(1, int(round(spacing_factor * len(dense))))
        args = (d, n, rr.ctypes.data_as(_I64), slot_of.ctypes.data_as(_I64), vals.ctypes.data_as(_F64),
                counts.ctypes.data_as(_I64), self.offset, threads, int(max_cells), spacing)
        # seed > 0: along a random spanning tree instead of the minimum one (ablation)
        h = lib.tw_cell_log_seed(*args, int(seed), C.byref(status)) if seed else lib.tw_cell_log(*args, C.byref(status))
        try:
            if status.value != 0:
                raise MemoryError(f"cell graph: status {status.value}")
            info = (C.c_int64 * 8)()
            weight = C.c_double(0.0)
            lib.tw_cg_info(h, info, C.byref(weight))
            K, faces, n_edges, labels, labelled, roots, _, width = list(info)
            li = (C.c_int64 * 6)()
            lib.tw_cl_info(h, li)
            changes, snap_len, n_slots, spacing, direct, self.edge_changes = list(li)
            cuts = []
            for k in range(d):
                c = np.empty(lib.tw_cg_ncuts(h, k), dtype=np.int64)
                lib.tw_cg_cuts(h, k, c.ctypes.data_as(_I64))
                cuts.append(c.tolist())
            face_of_cell = np.empty(K, dtype=np.uint32)
            self.face_value = np.empty(faces, dtype=np.float64)
            pos = np.empty(faces, dtype=np.int64)
            self.face_empty = np.empty(faces, dtype=bool)
            slot, item = np.empty(changes, dtype=np.int32), np.empty(changes, dtype=np.int32)
            snap = np.empty(snap_len, dtype=np.int32)
            lib.tw_cl_take(h, face_of_cell.ctypes.data_as(_U32), self.face_value.ctypes.data_as(_F64),
                           pos.ctypes.data_as(_I64), self.face_empty.ctypes.data_as(_U8),
                           slot.ctypes.data_as(_I32), item.ctypes.data_as(_I32), snap.ctypes.data_as(_I32))
        finally:
            lib.tw_cg_free(h)
        self.overlay = _Overlay(cuts, slab_counts, face_of_cell, faces)
        self.face_version = _small(pos, int(pos.max()) if faces else 0)
        self.log = _Log(slot, item, snap, n_slots, spacing, bool(direct), n, vals)
        self.pft = self.log      # the same reads as the persistent tree: active, items, to_arena
        self.slot_trees = sorted(dense)   # the tree of every slot, ascending
        self.item_trees = np.asarray(slots, dtype=np.int64)
        self.stats = CellStats(grid_cells=K, faces=faces, face_edges=n_edges, slots=len(dense),
                               tree_weight=weight.value, walk_cost=float(changes), sources=roots, pft_nodes=0,
                               pft_entries=0, node_copies=0, path_nodes=0,   # the log is counted in leaf_bytes (IndexStats)
                               locator_words=self.overlay.locator_words, seconds={"native": time.perf_counter() - t0})

    @property
    def leaf_bytes(self) -> int:
        """The log with its checkpoints and the version of every cell."""
        return self.log.nbytes + self.face_version.nbytes

    def face_of(self, slab: Sequence[int]) -> int:
        return self.overlay.face_of(slab)

    def piece(self, slab: Sequence[int]) -> Piece:
        return Piece(self, self.face_of(slab))


class LeanCellIndex:
    """A large cell graph: walls with their weights, the minimum spanning tree over the grid cells, and one value per
    cell computed along it; no persistent tree (a read of the leaves goes through the items).  ``table`` is the value
    of every grid cell, in the numbering of the overlay."""

    plan_kind = "tree"

    def __init__(self, items: Sequence[int], ranges, slots: Sequence[int], slab_counts: Sequence[int], values,
                 offset: float = 0.0, threads: int = 1, max_cells: int = 1 << 40) -> None:
        t0 = time.perf_counter()
        lib = _load()
        self.items = list(items)
        d = len(slab_counts)
        n = len(self.items)
        dense = {s: i for i, s in enumerate(sorted(set(slots)))}
        slot_of = np.array([dense[s] for s in slots], dtype=np.int64)
        rr = np.asarray(ranges, dtype=np.int64).reshape(n, d, 2)
        vals = np.asarray(values, dtype=np.float64)
        counts = np.asarray(slab_counts, dtype=np.int64)
        status = C.c_int64(0)
        h = lib.tw_lean_table(d, n, rr.ctypes.data_as(_I64), slot_of.ctypes.data_as(_I64), vals.ctypes.data_as(_F64),
                              counts.ctypes.data_as(_I64), float(offset), threads, int(max_cells), C.byref(status))
        try:
            if status.value != 0:
                raise MemoryError(f"cell graph: status {status.value}")
            info = (C.c_int64 * 8)()
            weight = C.c_double(0.0)
            lib.tw_cg_info(h, info, C.byref(weight))
            K = list(info)[0]
            cuts = []
            for k in range(d):
                c = np.empty(lib.tw_cg_ncuts(h, k), dtype=np.int64)
                lib.tw_cg_cuts(h, k, c.ctypes.data_as(_I64))
                cuts.append(c.tolist())
            self.table = np.empty(K, dtype=np.float64)
            lib.tw_lean_take(h, self.table.ctypes.data_as(_F64))
        finally:
            lib.tw_cg_free(h)
        self.overlay = _Overlay(cuts, slab_counts, None, K)
        self.face_value = self.table
        self.stats = CellStats(grid_cells=K, faces=K, face_edges=0, slots=len(dense), tree_weight=weight.value,
                               walk_cost=weight.value, sources=1, pft_nodes=0, pft_entries=0, node_copies=0, path_nodes=0,
                               locator_words=sum(len(c) for c in cuts), seconds={"native": time.perf_counter() - t0})
