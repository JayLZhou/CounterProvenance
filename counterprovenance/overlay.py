"""Exact overlay of a family of slotted leaf boxes.

Each *item* is a leaf box that occupies one *slot* (a component tree, or the
box itself).  Inside a cell of the overlay every slot holds one fixed item (or
is inactive), so the cell has a fixed provenance signature.

Construction works on the compressed grid of the items' own boundaries:

1. every face of every box labels the grid edges it covers with
   ``(slot, item below, item above)`` - these are the boundary labels
   ``Delta_e``; no ``K x M`` signature matrix is ever built;
2. grid cells joined by an unlabelled edge have the same signature, so the
   connected components of unlabelled edges are exactly the faces of the
   *non-Cartesian* overlay (``exact=True``); with ``exact=False`` every grid
   cell is its own face;
3. labelled grid edges between two faces give the face adjacency graph and its
   change sets.

Point location maps the query's slab indices to the local grid cell and then
to its face; it stores one integer per grid cell.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .persistence import INACTIVE

try:  # scipy is a soft dependency for large overlays
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
except ImportError:  # pragma: no cover
    coo_matrix = None
    connected_components = None


@dataclass(frozen=True)
class FaceEdge:
    a: int  # face id (a < b)
    b: int
    changes: dict  # slot -> (item in a, item in b)

    @property
    def count(self) -> int:
        return len(self.changes)


class Overlay:
    def __init__(
        self,
        ranges: Sequence[Sequence[tuple[int, int]]],
        slots: Sequence[int],
        slab_counts: Sequence[int],
        exact: bool = True,
    ) -> None:
        """``ranges[i][k]`` is item ``i``'s global slab range on the k-th axis."""
        self.n_items = len(ranges)
        self.slots = list(slots)
        self.n_slots = (max(self.slots) + 1) if self.slots else 0
        self.d = len(slab_counts)
        self.slab_counts = list(slab_counts)
        self.exact = exact
        # local grid: the items' own boundaries only
        self.cuts: list[list[int]] = []
        for k in range(self.d):
            cuts = set()
            for r in ranges:
                lo, hi = r[k]
                if lo > 0:
                    cuts.add(lo)
                if hi < self.slab_counts[k]:
                    cuts.add(hi)
            self.cuts.append(sorted(cuts))
        self.shape = tuple(len(c) + 1 for c in self.cuts)
        self.grid_cells = int(np.prod(self.shape)) if self.d else 1
        self.strides = []
        stride = 1
        for k in range(self.d - 1, -1, -1):
            self.strides.insert(0, stride)
            stride *= self.shape[k]
        self.local = [
            tuple(
                (bisect.bisect_right(self.cuts[k], r[k][0]), bisect.bisect_right(self.cuts[k], r[k][1] - 1) + 1)
                for k in range(self.d)
            )
            for r in ranges
        ]
        self._build()

    # ------------------------------------------------------------------
    def _face_cells(self, item_local, k: int, coord: int) -> np.ndarray:
        """Flat indices of the grid cells with coordinate ``coord`` on axis k
        and inside the item's range on every other axis."""
        axes = []
        for j in range(self.d):
            if j == k:
                axes.append(np.array([coord], dtype=np.int64))
            else:
                lo, hi = item_local[j]
                axes.append(np.arange(lo, hi, dtype=np.int64))
        mesh = np.meshgrid(*axes, indexing="ij")
        flat = np.zeros(mesh[0].shape, dtype=np.int64)
        for j in range(self.d):
            flat += mesh[j] * self.strides[j]
        return flat.ravel()

    def _build(self) -> None:
        K = self.grid_cells
        # 1. boundary labels on grid edges: edge id = k * K + flat(upper cell)
        edge_ids: list[np.ndarray] = []
        edge_slot: list[np.ndarray] = []
        edge_side: list[np.ndarray] = []  # 0: item is below the edge, 1: above
        edge_item: list[np.ndarray] = []
        for i, loc in enumerate(self.local):
            slot = self.slots[i]
            for k in range(self.d):
                lo, hi = loc[k]
                if lo > 0:
                    cells = self._face_cells(loc, k, lo)
                    edge_ids.append(k * K + cells)
                    edge_slot.append(np.full(len(cells), slot, dtype=np.int64))
                    edge_side.append(np.ones(len(cells), dtype=np.int8))
                    edge_item.append(np.full(len(cells), i, dtype=np.int64))
                if hi < self.shape[k]:
                    cells = self._face_cells(loc, k, hi)
                    edge_ids.append(k * K + cells)
                    edge_slot.append(np.full(len(cells), slot, dtype=np.int64))
                    edge_side.append(np.zeros(len(cells), dtype=np.int8))
                    edge_item.append(np.full(len(cells), i, dtype=np.int64))
        if edge_ids:
            eid = np.concatenate(edge_ids)
            eslot = np.concatenate(edge_slot)
            eside = np.concatenate(edge_side)
            eitem = np.concatenate(edge_item)
        else:
            eid = eslot = eitem = np.zeros(0, dtype=np.int64)
            eside = np.zeros(0, dtype=np.int8)
        self.label_count = int(len(eid))
        labelled = np.unique(eid)
        self.labelled_grid_edges = int(len(labelled))

        # 2. faces = components of the unlabelled grid-edge graph
        if self.exact and self.d > 0 and K > 1:
            rows, cols = [], []
            for k in range(self.d):
                upper = np.arange(K, dtype=np.int64)
                upper = upper[(upper // self.strides[k]) % self.shape[k] > 0]
                ids = k * K + upper
                keep = ~np.isin(ids, labelled, assume_unique=False)
                rows.append(upper[keep])
                cols.append(upper[keep] - self.strides[k])
            rows_a = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
            cols_a = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
            if coo_matrix is None:
                raise ImportError("scipy is required for exact overlays")
            graph = coo_matrix((np.ones(len(rows_a), dtype=np.int8), (rows_a, cols_a)), shape=(K, K))
            n_faces, face_of_cell = connected_components(graph, directed=False)
            # renumber faces in order of first grid cell (deterministic)
            _, first = np.unique(face_of_cell, return_index=True)
            order = np.argsort(first)
            renumber = np.empty(n_faces, dtype=np.int64)
            renumber[order] = np.arange(n_faces)
            face_of_cell = renumber[face_of_cell]
        else:
            n_faces = K
            face_of_cell = np.arange(K, dtype=np.int64)
        self.n_faces = int(n_faces)
        self.face_of_cell = face_of_cell.astype(np.int64)
        _, rep = np.unique(self.face_of_cell, return_index=True)
        self.representative = rep.astype(np.int64)  # one grid cell per face

        # 3. face adjacency with change sets
        edges: dict[tuple[int, int], dict] = {}
        if len(eid):
            order = np.lexsort((eslot, eid))
            eid, eslot, eside, eitem = eid[order], eslot[order], eside[order], eitem[order]
            boundaries = np.flatnonzero(np.diff(eid)) + 1
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [len(eid)]))
            for s, t in zip(starts, stops):
                e = int(eid[s])
                k, upper = divmod(e, K)
                lower = upper - self.strides[k]
                fl = int(self.face_of_cell[lower])
                fu = int(self.face_of_cell[upper])
                key = (fl, fu) if fl < fu else (fu, fl)
                if key in edges:
                    continue
                change: dict[int, list[int]] = {}
                for j in range(s, t):
                    # entry = [item below the edge, item above the edge]
                    entry = change.setdefault(int(eslot[j]), [INACTIVE, INACTIVE])
                    entry[int(eside[j])] = int(eitem[j])
                oriented = {
                    slot: ((below, above) if fl < fu else (above, below))
                    for slot, (below, above) in change.items()
                }
                edges[key] = oriented
        if not self.exact and self.d > 0 and K > 1:
            # grid cells are faces: keep the zero-change adjacencies so that
            # plans can move between cells with identical signatures for free
            for k in range(self.d):
                upper = np.arange(K, dtype=np.int64)
                upper = upper[(upper // self.strides[k]) % self.shape[k] > 0]
                ids = k * K + upper
                for u in upper[~np.isin(ids, labelled)]:
                    edges.setdefault((int(u) - self.strides[k], int(u)), {})
        self.edges = [FaceEdge(a, b, ch) for (a, b), ch in sorted(edges.items())]

    # ------------------------------------------------------------------
    def cell_of(self, slab: Sequence[int]) -> int:
        flat = 0
        for k in range(self.d):
            flat += bisect.bisect_right(self.cuts[k], int(slab[k])) * self.strides[k]
        return flat

    def face_of(self, slab: Sequence[int]) -> int:
        return int(self.face_of_cell[self.cell_of(slab)])

    def signature(self, face: int) -> list[int]:
        """Item held by every slot inside ``face`` (O(items * d); for sources and tests)."""
        cell = int(self.representative[face])
        coords = [(cell // self.strides[k]) % self.shape[k] for k in range(self.d)]
        sig = [INACTIVE] * self.n_slots
        for i, loc in enumerate(self.local):
            if all(loc[k][0] <= coords[k] < loc[k][1] for k in range(self.d)):
                sig[self.slots[i]] = i
        return sig

    @property
    def locator_words(self) -> int:
        return self.grid_cells + sum(len(c) for c in self.cuts)
