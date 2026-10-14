"""Fitted tree ensembles, unit binding, and the leaf-box geometry.

A fitted additive tree ensemble predicts ``base + sum_m value_m(leaf_m(x))``.
Every supported model family is converted to one normalised representation in
which a split routes left iff ``x[feature] <= threshold`` after the input has
been cast to the model's own input dtype.  Models with strict ``x < theta``
splits (XGBoost) are converted exactly by moving the threshold to the previous
representable float32.

Binding a fixed unit ``u`` to a set of intervention axes ``S`` follows every
split on a non-intervention feature with ``u``'s own value.  What remains of
each tree is a partition of ``R^|S|`` into axis-aligned *leaf boxes*
``lo < t <= hi`` (reachable leaves only), which is the geometric object every
index in this package works on.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------
# normalised ensemble
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Tree:
    """Binary regression tree; leaves have ``feature == -1``."""

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    native_leaf: np.ndarray  # leaf id reported by the model's own leaf API

    def route(self, x: np.ndarray) -> int:
        node = 0
        feature, threshold, left, right = self.feature, self.threshold, self.left, self.right
        while feature[node] >= 0:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        return int(node)


@dataclass(frozen=True)
class TreeEnsemble:
    trees: tuple[Tree, ...]
    base: float
    input_dtype: np.dtype
    n_features: int
    source: str
    zero_threshold: float = 0.0  # inputs with |x| <= this are read as 0 (LightGBM)

    @property
    def tree_count(self) -> int:
        return len(self.trees)

    def cast(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype=self.input_dtype).astype(np.float64)
        if self.zero_threshold > 0.0:
            out = np.where(np.abs(out) <= self.zero_threshold, 0.0, out)
        return out

    def leaves(self, X: np.ndarray) -> np.ndarray:
        """Node index of the selected leaf, shape (n, M)."""
        X = np.atleast_2d(self.cast(X))
        return np.array([[tree.route(x) for tree in self.trees] for x in X], dtype=np.int64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(self.cast(X))
        out = np.empty(len(X))
        for i, x in enumerate(X):
            out[i] = self.base + math.fsum(tree.value[tree.route(x)] for tree in self.trees)
        return out


def _tree_from_arrays(feature, threshold, left, right, value, native_leaf=None) -> Tree:
    feature = np.asarray(feature, dtype=np.int64)
    if native_leaf is None:
        native_leaf = np.arange(len(feature), dtype=np.int64)
    return Tree(
        feature=feature,
        threshold=np.asarray(threshold, dtype=np.float64),
        left=np.asarray(left, dtype=np.int64),
        right=np.asarray(right, dtype=np.int64),
        value=np.asarray(value, dtype=np.float64),
        native_leaf=np.asarray(native_leaf, dtype=np.int64),
    )


def _sklearn_tree(estimator, scale: float) -> Tree:
    t = estimator.tree_
    feature = np.where(t.children_left < 0, -1, t.feature)
    return _tree_from_arrays(
        feature,
        t.threshold,
        t.children_left,
        t.children_right,
        t.value[:, 0, 0] * scale,
    )


def from_sklearn_gbr(model) -> TreeEnsemble:
    """``GradientBoostingRegressor`` (squared/absolute/huber loss, identity link)."""
    trees = tuple(_sklearn_tree(est, model.learning_rate) for est in model.estimators_[:, 0])
    if model.init_ == "zero":
        base = 0.0
    else:
        base = float(np.asarray(model.init_.constant_).ravel()[0])
    return TreeEnsemble(trees, base, np.dtype(np.float32), model.n_features_in_, "sklearn-gbr")


def from_sklearn_forest(model) -> TreeEnsemble:
    """``RandomForestRegressor`` / ``ExtraTreesRegressor`` / ``DecisionTreeRegressor``."""
    estimators = getattr(model, "estimators_", None) or [model]
    scale = 1.0 / len(estimators)
    trees = tuple(_sklearn_tree(est, scale) for est in estimators)
    return TreeEnsemble(trees, 0.0, np.dtype(np.float32), model.n_features_in_, "sklearn-forest")


def from_sklearn_hgb(model) -> TreeEnsemble:
    """``HistGradientBoostingRegressor`` with numerical features and no missing values."""
    trees = []
    for predictors in model._predictors:
        nodes = predictors[0].nodes
        if np.any(nodes["is_categorical"]):
            raise NotImplementedError("categorical splits are not supported")
        is_leaf = nodes["is_leaf"].astype(bool)
        trees.append(
            _tree_from_arrays(
                np.where(is_leaf, -1, nodes["feature_idx"]),
                nodes["num_threshold"],
                nodes["left"],
                nodes["right"],
                nodes["value"],
            )
        )
    base = float(np.asarray(model._baseline_prediction).ravel()[0])
    return TreeEnsemble(tuple(trees), base, np.dtype(np.float64), model.n_features_in_, "sklearn-hgb")


_XGB_IDENTITY_OBJECTIVES = {"reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"}


def from_xgboost(model) -> TreeEnsemble:
    """XGBoost ``gbtree`` regressor with an identity link.

    XGBoost routes left iff ``float32(x) < split_condition``; this equals
    ``float32(x) <= nextafter32(split_condition, -inf)``.
    """
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    config = json.loads(booster.save_raw(raw_format="json"))
    learner = config["learner"]
    objective = learner["objective"]["name"]
    if objective not in _XGB_IDENTITY_OBJECTIVES:
        raise NotImplementedError(f"objective {objective!r} has a non-identity link")
    gb = learner["gradient_booster"]
    if gb["name"] != "gbtree":
        raise NotImplementedError(f"booster {gb['name']!r} is not supported")
    trees = []
    for tree in gb["model"]["trees"]:
        if int(tree["tree_param"].get("size_leaf_vector", "1") or 1) > 1:
            raise NotImplementedError("multi-target trees are not supported")
        left = np.asarray(tree["left_children"], dtype=np.int64)
        right = np.asarray(tree["right_children"], dtype=np.int64)
        cond = np.asarray(tree["split_conditions"], dtype=np.float32)
        is_leaf = left < 0
        threshold = np.where(
            is_leaf,
            0.0,
            np.nextafter(cond, np.float32(-np.inf)).astype(np.float64),
        )
        value = np.where(is_leaf, cond.astype(np.float64), 0.0)
        feature = np.where(is_leaf, -1, np.asarray(tree["split_indices"], dtype=np.int64))
        trees.append(_tree_from_arrays(feature, threshold, left, right, value))
    base_text = str(learner["learner_model_param"]["base_score"]).strip("[]")
    base = float(base_text)
    n_features = int(learner["learner_model_param"]["num_feature"])
    return TreeEnsemble(tuple(trees), base, np.dtype(np.float32), n_features, "xgboost")


def from_lightgbm(model) -> TreeEnsemble:
    """LightGBM regressor (numerical ``<=`` splits, identity link)."""
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()
    trees = []
    for info in dump["tree_info"]:
        feature: list[int] = []
        threshold: list[float] = []
        left: list[int] = []
        right: list[int] = []
        value: list[float] = []
        native: list[int] = []

        def visit(node: dict) -> int:
            index = len(feature)
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(0.0)
            native.append(-1)
            if "leaf_value" in node:
                value[index] = float(node["leaf_value"])
                native[index] = int(node.get("leaf_index", 0))
                return index
            if node["decision_type"] != "<=":
                raise NotImplementedError("categorical LightGBM splits are not supported")
            if node.get("missing_type") == "Zero":
                raise NotImplementedError("zero_as_missing is not supported")
            feature[index] = int(node["split_feature"])
            threshold[index] = float(node["threshold"])
            left[index] = visit(node["left_child"])
            right[index] = visit(node["right_child"])
            return index

        visit(info["tree_structure"])
        trees.append(_tree_from_arrays(feature, threshold, left, right, value, native))
    if dump.get("average_output"):
        scale = 1.0 / len(trees)
        trees = [
            _tree_from_arrays(t.feature, t.threshold, t.left, t.right, t.value * scale, t.native_leaf)
            for t in trees
        ]
    n_features = int(dump["max_feature_idx"]) + 1
    # LightGBM drops |x| <= kZeroThreshold (1e-35f) from every input row, i.e.
    # reads such values as exact zeros before comparing with thresholds
    return TreeEnsemble(tuple(trees), 0.0, np.dtype(np.float64), n_features, "lightgbm",
                        zero_threshold=float(np.float32(1e-35)))


def from_model(model) -> TreeEnsemble:
    """Dispatch on the fitted model's type."""
    name = type(model).__name__
    module = type(model).__module__
    if name == "GradientBoostingRegressor":
        return from_sklearn_gbr(model)
    if name == "HistGradientBoostingRegressor":
        return from_sklearn_hgb(model)
    if name in {"RandomForestRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor"}:
        return from_sklearn_forest(model)
    if module.startswith("xgboost"):
        return from_xgboost(model)
    if module.startswith("lightgbm"):
        return from_lightgbm(model)
    raise NotImplementedError(f"unsupported model type {module}.{name}")


# --------------------------------------------------------------------------
# unit binding: leaf boxes over the intervention axes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LeafBox:
    """Reachable leaf of one tree: ``lo < t <= hi`` on every intervention axis."""

    tree: int
    node: int
    native_leaf: int
    lo: tuple[float, ...]
    hi: tuple[float, ...]
    value: float


@dataclass
class BoundEnsemble:
    """An ensemble with a fixed unit context; only the intervention axes vary."""

    ensemble: TreeEnsemble
    unit: np.ndarray
    axes: tuple[int, ...]
    boxes: list[LeafBox]
    offset: float  # base score + abducted noise
    _LO: np.ndarray = field(init=False, repr=False)
    _HI: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._LO = np.array([b.lo for b in self.boxes], dtype=float)
        self._HI = np.array([b.hi for b in self.boxes], dtype=float)

    @property
    def d(self) -> int:
        return len(self.axes)

    @property
    def tree_count(self) -> int:
        return self.ensemble.tree_count

    def prepare(self, t: Sequence[float]) -> np.ndarray:
        """Cast an intervention vector to the model's input precision."""
        return self.ensemble.cast(np.asarray(t, dtype=float).reshape(self.d))

    def selected(self, t: Sequence[float]) -> np.ndarray:
        """Brute-force stabbing: indices of the boxes that contain ``t``."""
        x = self.prepare(t)
        return np.flatnonzero(np.all(self._LO < x, axis=1) & np.all(x <= self._HI, axis=1))

    def value(self, t: Sequence[float]) -> float:
        return self.offset + math.fsum(self.boxes[i].value for i in self.selected(t))

    def full_input(self, t: Sequence[float]) -> np.ndarray:
        row = np.array(self.unit, dtype=float)
        row[list(self.axes)] = np.asarray(t, dtype=float)
        return row

    def label(self, box: int) -> tuple[int, int]:
        """Provenance atom of a box: ``(tree index, native leaf id)``."""
        b = self.boxes[box]
        return (b.tree, b.native_leaf)


def bind(ensemble: TreeEnsemble, unit: Sequence[float], axes: Sequence[int], noise: float = 0.0) -> BoundEnsemble:
    """Restrict every tree to the intervention ``axes`` for the fixed ``unit``."""
    axes = tuple(int(a) for a in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("intervention axes must be distinct")
    position = {feature: i for i, feature in enumerate(axes)}
    u = ensemble.cast(np.asarray(unit, dtype=float).reshape(1, -1))[0]
    d = len(axes)
    boxes: list[LeafBox] = []
    for m, tree in enumerate(ensemble.trees):
        stack = [(0, [-math.inf] * d, [math.inf] * d)]
        while stack:
            node, lo, hi = stack.pop()
            f = int(tree.feature[node])
            if f < 0:
                boxes.append(
                    LeafBox(m, node, int(tree.native_leaf[node]), tuple(lo), tuple(hi), float(tree.value[node]))
                )
                continue
            theta = float(tree.threshold[node])
            if f not in position:
                stack.append((int(tree.left[node]) if u[f] <= theta else int(tree.right[node]), lo, hi))
                continue
            a = position[f]
            if theta >= hi[a]:
                stack.append((int(tree.left[node]), lo, hi))
            elif theta <= lo[a]:
                stack.append((int(tree.right[node]), lo, hi))
            else:
                hi_left = list(hi)
                hi_left[a] = theta
                lo_right = list(lo)
                lo_right[a] = theta
                stack.append((int(tree.right[node]), lo_right, hi))
                stack.append((int(tree.left[node]), lo, hi_left))
    return BoundEnsemble(ensemble, u, axes, boxes, ensemble.base + float(noise))


# --------------------------------------------------------------------------
# slab discretisation shared by every index
# --------------------------------------------------------------------------
class Slabs:
    """Per-axis elementary slabs induced by all finite box bounds.

    Slab ``s`` of axis ``a`` is ``(th[s-1], th[s]]`` with ``th[-1] = -inf`` and
    ``th[len] = +inf``.  A box covers the half-open slab range ``[lo, hi)``.
    """

    def __init__(self, bound: BoundEnsemble) -> None:
        self.d = bound.d
        self.thresholds: list[list[float]] = []
        for a in range(self.d):
            values = {b.lo[a] for b in bound.boxes if math.isfinite(b.lo[a])}
            values.update(b.hi[a] for b in bound.boxes if math.isfinite(b.hi[a]))
            self.thresholds.append(sorted(values))
        self.counts = [len(t) + 1 for t in self.thresholds]
        self.ranges = [
            tuple(self._range(b.lo[a], b.hi[a], a) for a in range(self.d)) for b in bound.boxes
        ]

    def _range(self, lo: float, hi: float, a: int) -> tuple[int, int]:
        th = self.thresholds[a]
        start = 0 if not math.isfinite(lo) else bisect.bisect_left(th, lo) + 1
        stop = len(th) + 1 if not math.isfinite(hi) else bisect.bisect_left(th, hi) + 1
        return start, stop

    def locate(self, x: np.ndarray) -> tuple[int, ...]:
        return tuple(bisect.bisect_left(self.thresholds[a], float(x[a])) for a in range(self.d))

    @property
    def grid_cells(self) -> int:
        out = 1
        for c in self.counts:
            out *= c
        return out
