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
    lib.tw_cg_take.argtypes = [C.c_void_p, _U32, _F64, _I32, _U8, _I32, _I32]
    lib.tw_cg_free.argtypes = [C.c_void_p]
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

    def __init__(self, left: np.ndarray, right: np.ndarray, width: int) -> None:
        self.left, self.right, self.width = left, right, width
        self.height = width.bit_length() - 1

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
            lib.tw_cg_take(h, face_of_cell.ctypes.data_as(_U32), self.face_value.ctypes.data_as(_F64),
                           self.face_version.ctypes.data_as(_I32), self.face_empty.ctypes.data_as(_U8),
                           left.ctypes.data_as(_I32), right.ctypes.data_as(_I32))
        finally:
            lib.tw_cg_free(h)
        self.overlay = _Overlay(cuts, slab_counts, face_of_cell, faces)
        self.pft = _Pft(left, right, int(width))
        self.stats = CellStats(grid_cells=K, faces=faces, face_edges=n_edges, slots=len(dense),
                               tree_weight=weight.value, walk_cost=weight.value, sources=roots, pft_nodes=nodes,
                               pft_entries=0, node_copies=0, path_nodes=nodes, locator_words=self.overlay.locator_words,
                               seconds={"native": time.perf_counter() - t0})

    def face_of(self, slab: Sequence[int]) -> int:
        return self.overlay.face_of(slab)

    def piece(self, slab: Sequence[int]) -> Piece:
        return Piece(self, self.face_of(slab))
