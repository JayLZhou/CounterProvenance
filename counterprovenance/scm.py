"""Counterfactual queries on an SCM over a causal DAG with tree-ensemble mechanisms.

Every variable ``X`` on a path from the intervened variables ``T`` to the
outcome ``Y`` has an additive-noise mechanism ``X := f_X(pa(X)) + N_X`` whose
``f_X`` is a fitted tree ensemble.  For a fixed unit, abduction recovers every
noise, the non-descendants of ``T`` keep their observed values, and every
mediator ``Z`` (a descendant of ``T`` and an ancestor of ``Y``) becomes a
piecewise-constant function ``Z(t)`` over the intervention space.

A counterfactual value is computed as ``u_X + (f_X(pa') - f_X(pa))``, which is
``f_X(pa') + N_X`` in exact arithmetic but equals the observed ``u_X`` exactly
whenever ``pa'`` reaches the same leaves as the observed parents
(consistency).  Observed values often lie on split thresholds of the children
(XGBoost cuts are data values, and integer columns are common), and the
rounding of ``(u_X - f_X(pa)) + f_X(pa')`` would otherwise move them across.

:func:`bind_scm` compiles the outcome's trees into leaf boxes over ``t``,
exactly as :func:`~counterprovenance.ensemble.bind` does for a single model:
a test on an intervened variable splits the current box, a test on a non-
intervened, non-mediator variable is decided by the unit, and a test
``Z <= theta`` on a mediator splits the current box along the grid on which
``Z(t)`` is constant.  The result is an ordinary
:class:`~counterprovenance.ensemble.BoundEnsemble`, so every index of the
package applies unchanged.  The provenance atoms are the leaves of the
outcome's trees.

:class:`ScmOnline` evaluates the same counterfactual by propagating through the
mechanisms in topological order with each library's own ``predict``: it is the
online baseline on a DAG and the reference for exactness.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .ensemble import BoundEnsemble, LeafBox, Slabs, TreeEnsemble, bind, from_model


class Infeasible(RuntimeError):
    """The compiled boxes or a mediator grid exceed the configured limits."""


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------
def parents_of(edges: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a, b in edges:
        out.setdefault(b, []).append(a)
        out.setdefault(a, [])
    return {k: sorted(v) for k, v in out.items()}


def topological(parents: dict[str, list[str]]) -> list[str]:
    indeg = {n: len(p) for n, p in parents.items()}
    children: dict[str, list[str]] = {n: [] for n in parents}
    for n, ps in parents.items():
        for p in ps:
            children[p].append(n)
    stack = sorted((n for n, k in indeg.items() if k == 0), reverse=True)
    order = []
    while stack:
        n = stack.pop()
        order.append(n)
        for c in sorted(children[n], reverse=True):
            indeg[c] -= 1
            if indeg[c] == 0:
                stack.append(c)
    if len(order) != len(parents):
        raise ValueError("graph has a cycle")
    return order


def ancestors(parents: dict[str, list[str]], node: str) -> set[str]:
    out, stack = set(), [node]
    while stack:
        for p in parents.get(stack.pop(), []):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


def descendants(parents: dict[str, list[str]], nodes: Sequence[str]) -> set[str]:
    children: dict[str, list[str]] = {}
    for n, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(n)
    out, stack = set(), list(nodes)
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


# ---------------------------------------------------------------------------
# fitted SCM
# ---------------------------------------------------------------------------
@dataclass
class FittedSCM:
    """Additive-noise SCM: one fitted regressor per non-root variable that is needed."""

    parents: dict[str, list[str]]
    models: dict[str, object]  # node -> fitted model (predict on its parents, in ``parents[node]`` order)
    ensembles: dict[str, TreeEnsemble] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for node, model in self.models.items():
            if node not in self.ensembles:
                self.ensembles[node] = from_model(model)

    def predict(self, node: str, values: dict[str, float]) -> float:
        row = np.array([[values[p] for p in self.parents[node]]], dtype=float)
        return float(self.models[node].predict(row)[0])

    def abduct(self, unit: dict[str, float], nodes: Sequence[str]) -> dict[str, float]:
        """Noise of ``unit`` for every node in ``nodes`` that has a mechanism."""
        return {n: float(unit[n]) - self.predict(n, unit) for n in nodes if n in self.models}

    def observed_leaf_values(self, node: str, unit: dict[str, float]) -> np.ndarray:
        """Value of the leaf the unit reaches in every tree of ``node``'s mechanism."""
        ens = self.ensembles[node]
        leaves = ens.leaves(np.array([[unit[p] for p in self.parents[node]]], dtype=float))[0]
        return np.array([tree.value[leaf] for tree, leaf in zip(ens.trees, leaves)], dtype=np.float64)


def _fit_one(factory, X, y):
    return factory().fit(X, y)


def fit_scm(data, edges: Sequence[tuple[str, str]], outcome: str, treatments: Sequence[str],
            factory: Callable[[], object], max_rows: int | None = None, seed: int = 0,
            n_jobs: int = 1) -> FittedSCM:
    """Fit a mechanism for the outcome and for every mediator between ``treatments`` and it.

    ``data`` is a pandas DataFrame whose columns include every graph node.
    """
    parents = parents_of(edges)
    for n in data.columns:
        parents.setdefault(n, [])
    relevant_nodes(parents, treatments, outcome)  # validates the treatments
    # one fit serves every choice of treatments: all non-root ancestors of the outcome, and the outcome
    keep = ancestors(parents, outcome) | {outcome}
    needed = [n for n in topological(parents) if n in keep]
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed)
    jobs = [n for n in needed if parents[n]]
    args = [(data[parents[n]].to_numpy(dtype=float), data[n].to_numpy(dtype=float)) for n in jobs]
    if n_jobs > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed

        fitted = Parallel(n_jobs=min(n_jobs, len(jobs)))(delayed(_fit_one)(factory, X, y) for X, y in args)
    else:
        fitted = [_fit_one(factory, X, y) for X, y in args]
    return FittedSCM(parents, dict(zip(jobs, fitted)))


def relevant_nodes(parents: dict[str, list[str]], treatments: Sequence[str], outcome: str) -> list[str]:
    """Mediators (descendants of T that are ancestors of Y, not in T) and Y, in topological order."""
    t = set(treatments)
    anc_y = ancestors(parents, outcome)
    missing = t - anc_y
    if missing:
        raise ValueError(f"treatments {sorted(missing)} are not ancestors of {outcome}")
    mediators = (descendants(parents, list(t)) & anc_y) - t
    keep = mediators | {outcome}
    return [n for n in topological(parents) if n in keep]


# ---------------------------------------------------------------------------
# online evaluation through the DAG (baseline and reference)
# ---------------------------------------------------------------------------
class ScmOnline:
    """Counterfactual by propagation with each model's own ``predict``."""

    def __init__(self, scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> None:
        self.scm = scm
        self.treatments = list(treatments)
        self.outcome = outcome
        self.order = relevant_nodes(scm.parents, treatments, outcome)
        self.unit = dict(unit)
        self.observed = {n: scm.predict(n, unit) for n in self.order}

    def query(self, t: Sequence[float]) -> float:
        values = dict(self.unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        for node in self.order:
            values[node] = float(self.unit[node]) + (self.scm.predict(node, values) - self.observed[node])
        return values[self.outcome]


# ---------------------------------------------------------------------------
# compilation of the outcome's trees into leaf boxes over t
# ---------------------------------------------------------------------------
@dataclass
class _Map:
    """A piecewise-constant function of ``t``: dense values on the slab grid of its own cuts."""

    thresholds: list[list[float]]  # per axis; empty for axes the function does not depend on
    values: np.ndarray  # shape = (len(th_a) + 1 for a in axes)


def _slab_span(th: list[float], lo: float, hi: float) -> range:
    """Slabs ``(th[s-1], th[s]]`` that intersect the interval ``(lo, hi]``."""
    start = 0 if lo == -math.inf else bisect.bisect_right(th, lo)
    stop = len(th) if hi == math.inf else bisect.bisect_left(th, hi)
    return range(start, stop + 1)


def _compile_trees(ens: TreeEnsemble, inputs: list[tuple[str, object]], d: int, max_boxes: int) -> list[LeafBox]:
    """Leaf boxes of every tree of ``ens``.

    ``inputs[j]`` describes model input ``j``: ``("axis", a)``, ``("fixed", value)`` or
    ``("map", _Map)``.  A test on a mapped input splits the box along the map's grid.
    """
    boxes: list[LeafBox] = []
    for m, tree in enumerate(ens.trees):
        stack = [(0, [-math.inf] * d, [math.inf] * d)]
        while stack:
            node, lo, hi = stack.pop()
            f = int(tree.feature[node])
            if f < 0:
                boxes.append(LeafBox(m, node, int(tree.native_leaf[node]), tuple(lo), tuple(hi),
                                     float(tree.value[node])))
                if len(boxes) > max_boxes:
                    raise Infeasible(f"more than {max_boxes:,} leaf boxes")
                continue
            theta = float(tree.threshold[node])
            left, right = int(tree.left[node]), int(tree.right[node])
            kind, arg = inputs[f]
            if kind == "fixed":
                stack.append((left if arg <= theta else right, lo, hi))
            elif kind == "axis":
                a = arg
                if theta >= hi[a]:
                    stack.append((left, lo, hi))
                elif theta <= lo[a]:
                    stack.append((right, lo, hi))
                else:
                    hl, lr = list(hi), list(lo)
                    hl[a], lr[a] = theta, theta
                    stack.append((right, lr, hi))
                    stack.append((left, lo, hl))
            else:  # mediator: split the box along the map's grid
                mp: _Map = arg
                spans = [_slab_span(mp.thresholds[a], lo[a], hi[a]) for a in range(d)]
                idx = np.ix_(*[np.asarray(s) for s in spans])
                go_left = mp.values[idx] <= theta
                if go_left.all():
                    stack.append((left, lo, hi))
                    continue
                if not go_left.any():
                    stack.append((right, lo, hi))
                    continue
                for cell in np.ndindex(go_left.shape):
                    clo, chi = list(lo), list(hi)
                    for a in range(d):
                        th = mp.thresholds[a]
                        if not th:
                            continue
                        s = spans[a][cell[a]]
                        if s > 0:
                            clo[a] = max(clo[a], th[s - 1])
                        if s < len(th):
                            chi[a] = min(chi[a], th[s])
                    stack.append((left if go_left[cell] else right, clo, chi))
    return boxes


def _as_map(bound_boxes: list[LeafBox], observed: float, observed_leaf: np.ndarray, d: int,
            cast: Callable[[np.ndarray], np.ndarray], max_cells: int) -> _Map:
    """Dense values of ``observed + sum of (box value - observed leaf value of its tree)`` on the
    grid of the boxes' own cuts; a cell whose boxes are the observed leaves gets ``observed``."""
    shim = BoundEnsemble.__new__(BoundEnsemble)
    shim.boxes = bound_boxes
    shim.axes = tuple(range(d))
    slabs = Slabs(shim)
    cells = math.prod(slabs.counts)
    if cells > max_cells:
        raise Infeasible(f"mediator grid of {cells:,} cells exceeds {max_cells:,}")
    acc = np.zeros(slabs.counts)
    for box, rng in zip(bound_boxes, slabs.ranges):
        acc[tuple(slice(s, e) for s, e in rng)] += box.value - observed_leaf[box.tree]
    return _Map([list(t) for t in slabs.thresholds], cast(observed + acc))


def bind_scm(scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str,
             max_boxes: int = 5_000_000, max_cells: int = 5_000_000) -> BoundEnsemble:
    """Leaf boxes over ``t`` of the outcome's trees, through the mediators of the DAG."""
    treatments = list(treatments)
    d = len(treatments)
    axis = {name: a for a, name in enumerate(treatments)}
    order = relevant_nodes(scm.parents, treatments, outcome)
    maps: dict[str, _Map] = {}
    out_ens = scm.ensembles[outcome]
    for node in order:
        ens = scm.ensembles[node]
        inputs = []
        for p in scm.parents[node]:
            if p in axis:
                inputs.append(("axis", axis[p]))
            elif p in maps:
                inputs.append(("map", maps[p]))
            else:
                inputs.append(("fixed", float(ens.cast(np.array([unit[p]]))[0])))
        boxes = _compile_trees(ens, inputs, d, max_boxes)
        if node == outcome:
            row = np.array([unit[p] for p in scm.parents[node]], dtype=float)
            return BoundEnsemble(out_ens, row, tuple(range(d)), boxes,
                                 out_ens.base + scm.abduct(unit, [node])[node])
        # the mediator's value is read by its children through their own input casts; all
        # mechanisms of one SCM come from one model family, so the outcome's cast applies
        maps[node] = _as_map(boxes, float(unit[node]), scm.observed_leaf_values(node, unit), d, out_ens.cast,
                             max_cells)
    raise AssertionError("outcome not reached")


def star_bind(scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> BoundEnsemble:
    """Single-model binding when every treatment is a parent of the outcome and there are no mediators."""
    ps = scm.parents[outcome]
    row = np.array([unit[p] for p in ps], dtype=float)
    noise = scm.abduct(unit, [outcome])[outcome]
    return bind(scm.ensembles[outcome], row, [ps.index(t) for t in treatments], noise=noise)
