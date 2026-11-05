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
``Z <= theta`` on a mediator splits the current box only where ``Z(t)`` straddles
``theta`` (``_Mediator.split``).  The result is an ordinary
:class:`~counterprovenance.ensemble.BoundEnsemble`, so every index of the
package applies unchanged.  The provenance atoms are the leaves of the
outcome's trees.

:class:`ScmOnline` evaluates the same counterfactual by propagating through the
mechanisms in topological order with each library's own ``predict``: it is the
online baseline on a DAG and the reference for exactness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import native_compile
from .ensemble import BoundEnsemble, LeafBox, TreeEnsemble, bind, from_model


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
        """Noise of ``unit`` for every node in ``nodes`` that has a mechanism, against the trees
        summed exactly (XGBoost's own ``predict`` accumulates in single precision)."""
        out = {}
        for n in nodes:
            if n in self.models:
                row = np.array([[unit[p] for p in self.parents[n]]], dtype=float)
                out[n] = float(unit[n]) - float(self.ensembles[n].predict(row)[0])
        return out

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
class _Mediator:
    """A mediator ``Z(t) = u_Z + sum_m (w_m(t) - w_m(u))`` given by the leaf boxes of its trees.

    Trees with a single box always select the unit's own leaf and are dropped.  Over a box ``B``
    of ``t``, ``Z`` lies between ``u_Z + sum_m min_B`` and ``u_Z + sum_m max_B`` of the per-tree
    differences (sums taken with ``fsum``, so that both bounds are attained exactly when every
    tree has one box in ``B``).  :meth:`split` uses these bounds as a dominance test: a box on
    one side of a threshold is decided without being split, and only a box that straddles it is
    cut, at a boundary of the mediator's own boxes inside it.
    """

    dominance = True  # False (for the experiment): a box is cut at every boundary inside it, even when it is decided

    def __init__(self, boxes: list[LeafBox], observed: float, observed_leaf: np.ndarray, d: int) -> None:
        trees = np.array([b.tree for b in boxes], dtype=np.int64)
        counts = np.bincount(trees, minlength=len(observed_leaf)) if len(boxes) else np.zeros(0, dtype=np.int64)
        keep = [i for i, b in enumerate(boxes) if counts[b.tree] > 1]
        self.lo = np.array([boxes[i].lo for i in keep], dtype=np.float64).reshape(-1, d)
        self.hi = np.array([boxes[i].hi for i in keep], dtype=np.float64).reshape(-1, d)
        self.tree = trees[keep] if keep else np.zeros(0, dtype=np.int64)
        self.diff = np.array([boxes[i].value - observed_leaf[boxes[i].tree] for i in keep], dtype=np.float64)
        self.observed = float(observed)
        self.d = d

    def _range(self, idx: np.ndarray) -> tuple[float, float]:
        if not len(idx):
            return self.observed, self.observed
        tr, df = self.tree[idx], self.diff[idx]
        starts = np.flatnonzero(np.r_[True, tr[1:] != tr[:-1]])
        lo = self.observed + math.fsum(np.minimum.reduceat(df, starts).tolist())
        hi = self.observed + math.fsum(np.maximum.reduceat(df, starts).tolist())
        return lo, hi

    def _has_boundary(self, idx: np.ndarray, blo: np.ndarray, bhi: np.ndarray) -> bool:
        return bool((((self.lo[idx] > blo) & (self.lo[idx] < bhi)) | ((self.hi[idx] > blo) & (self.hi[idx] < bhi))).any())

    def split(self, theta: float, lo: list[float], hi: list[float],
              cast: Callable[[np.ndarray], np.ndarray]) -> list[tuple[bool, list[float], list[float]]]:
        """Pieces ``(Z <= theta, lo, hi)`` of the box ``(lo, hi]``, as read through ``cast``."""
        out = []
        stack = [(lo, hi, np.arange(len(self.tree)))]
        while stack:
            lo, hi, idx = stack.pop()
            blo, bhi = np.asarray(lo), np.asarray(hi)
            idx = idx[(self.lo[idx] < bhi).all(1) & (self.hi[idx] > blo).all(1)]
            vlo, vhi = self._range(idx)
            vlo, vhi = cast(np.array([vlo, vhi]))
            if self.dominance or not self._has_boundary(idx, blo, bhi):
                if vhi <= theta:
                    out.append((True, lo, hi))
                    continue
                if vlo > theta:
                    out.append((False, lo, hi))
                    continue
            # some tree still has two boxes in the current box: cut at the median of the
            # boundaries strictly inside it, on the axis that has the most of them
            best = None
            for a in range(self.d):
                c = np.concatenate((self.lo[idx, a], self.hi[idx, a]))
                c = np.unique(c[(c > blo[a]) & (c < bhi[a])])
                if len(c) and (best is None or len(c) > len(best[1])):
                    best = (a, c)
            a, c = best
            cut = float(c[len(c) // 2])
            hl, lr = list(hi), list(lo)
            hl[a], lr[a] = cut, cut
            stack.append((lr, hi, idx))
            stack.append((lo, hl, idx))
        return out


def _compile_trees(ens: TreeEnsemble, inputs: list[tuple[str, object]], d: int, max_boxes: int) -> list[LeafBox]:
    """Leaf boxes of every tree of ``ens``.

    ``inputs[j]`` describes model input ``j``: ``("axis", a)``, ``("fixed", value)`` or
    ``("mediator", _Mediator)``.  A test on a mediator keeps the box when the mediator lies on
    one side of the threshold over the whole box, and splits it otherwise.
    """
    cast = ens.cast
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
            else:
                for go_left, plo, phi in arg.split(theta, lo, hi, cast):
                    stack.append((left if go_left else right, plo, phi))
    return boxes


def _bind_scm_native(scm: FittedSCM, unit: dict[str, float], treatments: list[str], outcome: str,
                     max_boxes: int, threads: int) -> BoundEnsemble:
    """:func:`bind_scm` with the C++ compiler: the same boxes, in the same order."""
    d = len(treatments)
    axis = {name: a for a, name in enumerate(treatments)}
    meds: dict[str, native_compile.NativeMediator] = {}
    for node in relevant_nodes(scm.parents, treatments, outcome):
        ens = scm.ensembles[node]
        inputs = []
        for p in scm.parents[node]:
            if p in axis:
                inputs.append(("axis", axis[p]))
            elif p in meds:
                inputs.append(("mediator", meds[p]))
            else:
                inputs.append(("fixed", float(ens.cast(np.array([unit[p]]))[0])))
        flat = native_compile.flat_trees(ens)
        boxes, status = native_compile.compile_trees(ens, flat, inputs, d, max_boxes, threads)
        if status == 1:
            raise Infeasible(f"more than {max_boxes:,} leaf boxes")
        if status != 0:
            raise AssertionError("a straddling box has no boundary inside")
        if node == outcome:
            at = flat["ptr"][boxes.tree] + boxes.node
            out = [LeafBox(*f) for f in zip(boxes.tree.tolist(), boxes.node.tolist(), flat["native_leaf"][at].tolist(),
                                            map(tuple, boxes.lo.tolist()), map(tuple, boxes.hi.tolist()),
                                            flat["value"][at].tolist())]
            row = np.array([unit[p] for p in scm.parents[node]], dtype=float)
            return BoundEnsemble(ens, row, tuple(range(d)), out, ens.base + scm.abduct(unit, [node])[node])
        meds[node] = native_compile.mediator_from(boxes, flat, float(unit[node]),
                                                  scm.observed_leaf_values(node, unit))
    raise AssertionError("outcome not reached")


def bind_scm(scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str,
             max_boxes: int = 50_000_000, native: bool = True, threads: int = 1) -> BoundEnsemble:
    """Leaf boxes over ``t`` of the outcome's trees, through the mediators of the DAG.

    ``native``: compile in C++ with ``threads`` threads when a compiler is available (the same boxes)."""
    treatments = list(treatments)
    if native and native_compile.available():
        return _bind_scm_native(scm, unit, treatments, outcome, max_boxes, threads)
    d = len(treatments)
    axis = {name: a for a, name in enumerate(treatments)}
    order = relevant_nodes(scm.parents, treatments, outcome)
    meds: dict[str, _Mediator] = {}
    out_ens = scm.ensembles[outcome]
    for node in order:
        ens = scm.ensembles[node]
        inputs = []
        for p in scm.parents[node]:
            if p in axis:
                inputs.append(("axis", axis[p]))
            elif p in meds:
                inputs.append(("mediator", meds[p]))
            else:
                inputs.append(("fixed", float(ens.cast(np.array([unit[p]]))[0])))
        boxes = _compile_trees(ens, inputs, d, max_boxes)
        if node == outcome:
            row = np.array([unit[p] for p in scm.parents[node]], dtype=float)
            return BoundEnsemble(out_ens, row, tuple(range(d)), boxes,
                                 out_ens.base + scm.abduct(unit, [node])[node])
        meds[node] = _Mediator(boxes, float(unit[node]), scm.observed_leaf_values(node, unit), d)
    raise AssertionError("outcome not reached")


def star_bind(scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> BoundEnsemble:
    """Single-model binding when every treatment is a parent of the outcome and there are no mediators."""
    ps = scm.parents[outcome]
    row = np.array([unit[p] for p in ps], dtype=float)
    noise = scm.abduct(unit, [outcome])[outcome]
    return bind(scm.ensembles[outcome], row, [ps.index(t) for t in treatments], noise=noise)
