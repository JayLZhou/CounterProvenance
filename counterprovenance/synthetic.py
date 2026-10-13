"""Synthetic ensembles for tests and controlled experiments."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .ensemble import Tree, TreeEnsemble, _tree_from_arrays


def stump(feature: int, threshold: float, left_value: float, right_value: float) -> Tree:
    return _tree_from_arrays(
        [feature, -1, -1], [threshold, 0.0, 0.0], [1, -1, -1], [2, -1, -1], [0.0, left_value, right_value]
    )


def random_tree(rng: np.random.Generator, n_features: int, depth: int, low: float = 0.0, high: float = 1.0) -> Tree:
    """Full binary tree of the given depth with thresholds inside the parent box."""
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []

    def grow(lo: list[float], hi: list[float], level: int) -> int:
        index = len(feature)
        feature.append(-1)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(0.0)
        if level == depth:
            value[index] = float(rng.normal())
            return index
        f = int(rng.integers(n_features))
        theta = float(rng.uniform(lo[f], hi[f]))
        feature[index] = f
        threshold[index] = theta
        hi_left = list(hi)
        hi_left[f] = theta
        lo_right = list(lo)
        lo_right[f] = theta
        left[index] = grow(lo, hi_left, level + 1)
        right[index] = grow(lo_right, hi, level + 1)
        return index

    grow([low] * n_features, [high] * n_features, 0)
    return _tree_from_arrays(feature, threshold, left, right, value)


def ensemble(trees: Sequence[Tree], n_features: int, base: float = 0.0) -> TreeEnsemble:
    return TreeEnsemble(tuple(trees), float(base), np.dtype(np.float64), n_features, "synthetic")


def stump_grid(m: int, rng: np.random.Generator, group: int = 1, order: str = "grouped") -> TreeEnsemble:
    """``m`` stumps, half on axis 0 and half on axis 1; ``group`` stumps share
    each threshold (coincident boundaries).  ``order='scattered'`` interleaves
    the coincident stumps of axis 1 so their slots are far apart in the PFT."""
    half = m // 2
    if half % group:
        raise ValueError("group must divide m/2")
    specs = []
    for axis in (0, 1):
        count = half // group
        for g in range(count):
            for member in range(group):
                specs.append((axis, (g + 1) / (count + 1), member, g))
    if order == "scattered":
        x = [s for s in specs if s[0] == 0]
        y = sorted((s for s in specs if s[0] == 1), key=lambda s: (s[2], s[3]))
        specs = x + y
    trees = [stump(axis, theta, float(rng.normal()), float(rng.normal())) for axis, theta, _, _ in specs]
    return ensemble(trees, 2, base=float(rng.normal()))


def boundary_points(thresholds: Sequence[float]) -> list[float]:
    """Each threshold and its two floating-point neighbours."""
    out: list[float] = []
    for t in thresholds:
        out.extend((math.nextafter(t, -math.inf), t, math.nextafter(t, math.inf)))
    return out
