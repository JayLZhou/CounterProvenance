"""Online baselines: evaluate the fitted ensemble for every query.

:class:`OnlineLoop` is what one writes without an index: copy the unit, set the
intervened features, walk every tree from the root to a leaf.  With ``bound``
it only walks the trees that still depend on an intervened feature (binding is
the first step of our method, so this variant is an ablation).
:class:`IncrementalLoop` re-walks only the trees that use a changed feature
(the online counterpart of :class:`~counterprovenance.index.QuerySession`).
Both return the exact value and the selected leaf of every walked tree.
"""

from __future__ import annotations

import math
from array import array
from itertools import chain
from typing import Sequence

import numpy as np

from .ensemble import BoundEnsemble, TreeEnsemble


def make_cast(ensemble: TreeEnsemble):
    """Read intervention values the way the model does (plain floats): round to
    the input precision, and map |x| <= zero_threshold to 0 (LightGBM)."""
    if ensemble.input_dtype == np.float32:
        def cast(t):
            return array("f", t.tolist() if isinstance(t, np.ndarray) else t).tolist()
    else:
        def cast(t):
            return t.tolist() if isinstance(t, np.ndarray) else [float(v) for v in t]
    zt = ensemble.zero_threshold
    if zt > 0.0:
        rounded = cast

        def cast(t):
            return [0.0 if -zt <= v <= zt else v for v in rounded(t)]
    return cast


def _lists(tree):
    return (tree.feature.tolist(), tree.threshold.tolist(), tree.left.tolist(),
            tree.right.tolist(), tree.value.tolist())


def varying_trees(bound: BoundEnsemble) -> list[int]:
    return sorted({b.tree for b in bound.boxes if any(math.isfinite(v) for v in b.lo + b.hi)})


class OnlineLoop:
    def __init__(self, ensemble: TreeEnsemble, unit: Sequence[float], axes: Sequence[int],
                 noise: float = 0.0, bound: BoundEnsemble | None = None) -> None:
        self.axes = list(axes)
        self._cast = make_cast(ensemble)
        self._unit = make_cast(ensemble)(np.asarray(unit, dtype=float).ravel())
        if bound is None:
            self.trees = list(range(ensemble.tree_count))
            self.const = ensemble.base + noise
        else:
            self.trees = varying_trees(bound)
            keep = set(self.trees)
            self.const = math.fsum([bound.offset] + [b.value for b in bound.boxes if b.tree not in keep])
        self._lists = [_lists(ensemble.trees[m]) for m in self.trees]

    def query(self, t: Sequence[float]) -> tuple[float, list[int]]:
        x = list(self._unit)
        for a, v in zip(self.axes, self._cast(t)):
            x[a] = v
        vals, leaves = [], []
        for feature, threshold, left, right, value in self._lists:
            node = 0
            while feature[node] >= 0:
                node = left[node] if x[feature[node]] <= threshold[node] else right[node]
            leaves.append(node)
            vals.append(value[node])
        return math.fsum(chain((self.const,), vals)), leaves


class IncrementalLoop:
    """Keeps the current leaf of every tree; a change of one intervened
    feature re-walks only the trees that split on it."""

    def __init__(self, ensemble: TreeEnsemble, bound: BoundEnsemble, t: Sequence[float]) -> None:
        self.axes = list(bound.axes)
        self._cast = make_cast(ensemble)
        self.x = self._cast(np.asarray(bound.unit, dtype=float).ravel())
        for a, v in zip(self.axes, self._cast(t)):
            self.x[a] = v
        self.trees = varying_trees(bound)
        keep = set(self.trees)
        self.const = math.fsum([bound.offset] + [b.value for b in bound.boxes if b.tree not in keep])
        self._lists = {m: _lists(ensemble.trees[m]) for m in self.trees}
        # trees whose reachable leaves (after binding) vary along axis k
        self._trees_of = {
            k: sorted({b.tree for b in bound.boxes if math.isfinite(b.lo[k]) or math.isfinite(b.hi[k])})
            for k in range(len(self.axes))
        }
        self.vals = {m: self._walk(m) for m in self.trees}

    def _walk(self, m: int) -> float:
        feature, threshold, left, right, value = self._lists[m]
        x, node = self.x, 0
        while feature[node] >= 0:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        return value[node]

    def set(self, axis: int, value: float) -> float:
        self.x[self.axes[axis]] = self._cast([value])[0]
        for m in self._trees_of[axis]:
            self.vals[m] = self._walk(m)
        return math.fsum(chain((self.const,), self.vals.values()))
