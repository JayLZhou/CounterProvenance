"""Online evaluation of counterfactuals on a DAG, in pure Python.

* :class:`ScmOnlinePy` - for every mechanism on a path from the treatments to
  the outcome, in topological order, walk every tree from its root to a leaf
  (recording the leaves), exactly like
  :class:`~counterprovenance.baselines.OnlineLoop` does for one model.
* :class:`ScmThreeStepPy` - the textbook three steps (abduction, action,
  prediction) on every query, caching nothing.

Both read inputs the way each library does (single-precision rounding,
LightGBM's zero threshold), compute a value as ``u_X + (f_X(pa') - f_X(pa))``
like :mod:`~counterprovenance.scm`, and agree with
:class:`~counterprovenance.scm.ScmOnline`.
"""

from __future__ import annotations

import math
from typing import Sequence

from .baselines import make_cast
from .scm import FittedSCM, relevant_nodes


class ScmOnlinePy:
    """Pure-Python propagation through the mechanisms of the DAG."""

    def __init__(self, scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> None:
        self.treatments = list(treatments)
        self.outcome = outcome
        order = relevant_nodes(scm.parents, treatments, outcome)
        self.unit = {k: float(v) for k, v in unit.items()}
        self.mechs = []
        for node in order:
            ens = scm.ensembles[node]
            obs = scm.observed_leaf_values(node, unit).tolist()
            trees = [(t.feature.tolist(), t.threshold.tolist(), t.left.tolist(), t.right.tolist(), t.value.tolist(), o)
                     for t, o in zip(ens.trees, obs)]
            self.mechs.append((node, list(scm.parents[node]), make_cast(ens), self.unit[node], trees))

    def query(self, t: Sequence[float]) -> tuple[float, list[int]]:
        values = dict(self.unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        leaves: list[int] = []
        for node, parents, cast, observed, trees in self.mechs:
            x = cast([values[p] for p in parents])
            diffs = []
            leaves = []
            for feature, threshold, left, right, value, obs in trees:
                n = 0
                while feature[n] >= 0:
                    n = left[n] if x[feature[n]] <= threshold[n] else right[n]
                leaves.append(n)
                diffs.append(value[n] - obs)
            values[node] = observed + math.fsum(diffs)
        return values[self.outcome], leaves


class ScmThreeStepPy:
    """Pure-Python counterfactual by the textbook three steps on every query, caching nothing:
    abduction walks every mechanism on the paths at the unit's observed parents, the action sets
    ``T``, and prediction walks them again at the counterfactual parents.  Same arithmetic as
    :class:`ScmOnlinePy` (``u + fsum`` of leaf differences), so both return the same floats."""

    def __init__(self, scm: FittedSCM, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> None:
        self.treatments = list(treatments)
        self.outcome = outcome
        order = relevant_nodes(scm.parents, treatments, outcome)
        self.unit = {k: float(v) for k, v in unit.items()}
        self.mechs = []
        for node in order:
            ens = scm.ensembles[node]
            trees = [(t.feature.tolist(), t.threshold.tolist(), t.left.tolist(), t.right.tolist(), t.value.tolist())
                     for t in ens.trees]
            self.mechs.append((node, list(scm.parents[node]), make_cast(ens), trees))

    @staticmethod
    def _leaves(x, trees):
        out = []
        for feature, threshold, left, right, value in trees:
            n = 0
            while feature[n] >= 0:
                n = left[n] if x[feature[n]] <= threshold[n] else right[n]
            out.append(n)
        return out

    def query(self, t: Sequence[float]) -> tuple[float, list[int]]:
        unit = self.unit
        # abduction: the leaves (hence f(pa)) of every mechanism at the observed parents
        observed = [self._leaves(cast([unit[p] for p in parents]), trees)
                    for _, parents, cast, trees in self.mechs]
        # action
        values = dict(unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        # prediction: X = u_X + (f(pa') - f(pa)), in topological order
        leaves: list[int] = []
        for (node, parents, cast, trees), obs in zip(self.mechs, observed):
            leaves = self._leaves(cast([values[p] for p in parents]), trees)
            values[node] = unit[node] + math.fsum(tr[4][n] - tr[4][o] for tr, n, o in zip(trees, leaves, obs))
        return values[self.outcome], leaves

    def query_weighted(self, t: Sequence[float], w) -> float:
        """The three steps with weights on the outcome's trees; the mediators keep their full models."""
        unit = self.unit
        observed = [self._leaves(cast([unit[p] for p in parents]), trees)
                    for _, parents, cast, trees in self.mechs]
        values = dict(unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        for (node, parents, cast, trees), obs in zip(self.mechs, observed):
            leaves = self._leaves(cast([values[p] for p in parents]), trees)
            if node == self.outcome:
                values[node] = unit[node] + math.fsum(float(u) * (tr[4][n] - tr[4][o])
                                                      for u, tr, n, o in zip(w, trees, leaves, obs))
            else:
                values[node] = unit[node] + math.fsum(tr[4][n] - tr[4][o] for tr, n, o in zip(trees, leaves, obs))
        return values[self.outcome]

    def query_prefix(self, t: Sequence[float], n: int) -> float:
        """The three steps with the first ``n`` trees of the outcome; it walks only those trees of it."""
        unit = self.unit
        observed = []
        for node, parents, cast, trees in self.mechs:
            use = trees[:n] if node == self.outcome else trees
            observed.append(self._leaves(cast([unit[p] for p in parents]), use))
        values = dict(unit)
        for name, v in zip(self.treatments, t):
            values[name] = float(v)
        for (node, parents, cast, trees), obs in zip(self.mechs, observed):
            use = trees[:n] if node == self.outcome else trees
            leaves = self._leaves(cast([values[p] for p in parents]), use)
            values[node] = unit[node] + math.fsum(tr[4][a] - tr[4][o] for tr, a, o in zip(use, leaves, obs))
        return values[self.outcome]
