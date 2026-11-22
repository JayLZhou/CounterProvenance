"""Counterfactual queries whose combination of the trees is chosen at query time.

With weights ``w`` on the outcome's trees the answer is ``Y(t) = u_Y + sum_m w_m (v_m(t) - v_m(u))``: the noise is
abducted for the weighted model, and the mediators keep their own full models.  A table of values cannot answer
this, since it stores one combination.  Every cell of a cell graph holds a root of the persistent factor tree,
which gives

* :meth:`WeightedIndex.prefix`: the first ``N`` trees (early stopping, ``iteration_range`` in XGBoost, ``num_iteration``
  in LightGBM), one root-to-leaf path per cell graph using the sums stored in the nodes;
* :meth:`WeightedIndex.weighted`: any weights, by reading the selected leaves.

Deep trees answer both from their leaf masks.  The index must be built with ``native_cells=True``.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Sequence

import numpy as np

from .index import ProvenanceIndex
from .native_cells import LogCellIndex, NativeCellIndex


class WeightedIndex:
    def __init__(self, index: ProvenanceIndex, observed: float, factual: Sequence[float]) -> None:
        if not index.compiled:
            raise ValueError("needs a compiled index")
        self.index = index
        self.observed = float(observed)
        self.factual = list(map(float, factual))
        boxes = index.bound.boxes
        self.groups = []
        for leaf in index._fast_leaves:
            cells = leaf.cells
            if not isinstance(cells, (NativeCellIndex, LogCellIndex)):
                raise ValueError("build the index with native_cells=True")
            self.groups.append((cells, cells.overlay.face_of_cell, cells.face_version.tolist(), cells.slot_trees,
                                cells.item_trees, cells.values))
        const = [i for leaf in index._const_leaves for i in leaf.cells.items]
        const.sort(key=lambda i: boxes[i].tree)
        self.const_trees = np.array([boxes[i].tree for i in const], dtype=np.int64)
        self.const_values = np.array([boxes[i].value for i in const], dtype=np.float64)
        self.const_prefix = np.concatenate(([0.0], np.cumsum(self.const_values)))
        self.field_trees = np.array([boxes[it[1]].tree for it in index._bitems], dtype=np.int64)
        self._memo: dict[int, float] = {}

    def _cells(self, t: Sequence[float]):
        ix = self.index
        x = ix._cast(t)
        packed = 0
        for a, th, rows in ix._paxes:
            packed += rows[bisect_left(th, x[a])]
        return x, memoryview(packed.to_bytes(ix._nbytes, "little")).cast("I")

    def _mask_values(self, x) -> np.ndarray:
        ix = self.index
        leaves, _ = ix._bitmask_np(x)
        return ix._bflat[ix._bbase + leaves]

    # -- the first N trees -------------------------------------------------
    def _prefix_sum(self, t: Sequence[float], n: int) -> float:
        x, cells = self._cells(t)
        out = float(self.const_prefix[np.searchsorted(self.const_trees, n)])
        for (leaf_cells, foc, ver, slot_trees, _, _), cell in zip(self.groups, cells):
            out += leaf_cells.pft.prefix(ver[foc[cell]], bisect_left(slot_trees, n))
        if self.index._baxes:
            out += float(self._mask_values(x)[self.field_trees < n].sum())
        return out

    def prefix(self, t: Sequence[float], n: int) -> float:
        base = self._memo.get(n)
        if base is None:
            base = self._memo[n] = self._prefix_sum(self.factual, n)
        return self.observed + self._prefix_sum(t, n) - base

    # -- any weights -------------------------------------------------------
    def _weighted_sum(self, t: Sequence[float], w: np.ndarray) -> float:
        x, cells = self._cells(t)
        out = float(self.const_values @ w[self.const_trees]) if len(self.const_trees) else 0.0
        for (leaf_cells, foc, ver, _, item_trees, values), cell in zip(self.groups, cells):
            items = leaf_cells.pft.items(ver[foc[cell]])
            out += float(values[items] @ w[item_trees[items]])
        if self.index._baxes:
            out += float(self._mask_values(x) @ w[self.field_trees])
        return out

    def weighted(self, t: Sequence[float], w: np.ndarray) -> float:
        return self.observed + self._weighted_sum(t, w) - self._weighted_sum(self.factual, w)
