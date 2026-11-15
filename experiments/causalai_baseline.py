"""Salesforce CausalAI (github.com/salesforce/causalai, commit e03c9d8, archived) made exact for one counterfactual
query on one unit.

CausalAI's ``CausalInference.ate_causal_path`` answers an intervention by copying its data, setting the treated
columns, and predicting every ancestor of the outcome in topological order with one model per variable; its
``counterfactual`` then fits a new regression per query.  It has no abduction, so neither returns the counterfactual
value of an SCM.  We keep its objects (``TabularData``, ``_TabularGraph``, ``_DataProcessor``) and the treated pass of
``ate_causal_path`` line by line, and change only what exactness needs:

1. its data is the unit, one row, so that the pass computes the unit and nothing else;
2. the model of a variable is the fitted mechanism of our SCM plus the noise of the unit, abducted once per unit with
   the same ``predict`` of the library (as DoWhy does), instead of a model trained by CausalAI;
3. its normalization is the identity (mean 0, std 1), so that the inputs reach the trees unchanged;
4. only the treated pass runs; ``ate`` also runs a control pass, which a counterfactual does not need.

Unlike ``ThreeStep``, it replays every ancestor of the outcome (CausalAI's ``relevant_nodes``), not only the
mediators, and adds the noise to the library's ``predict`` (``f(pa') + eps``) instead of adding the change of the
leaves to the observed value (``u + f(pa') - f(pa)``).  Both give the same answer up to rounding; on XGBoost, whose
``predict`` rounds its inputs to single precision, they differ as DoWhy and ChiRho do.

``ScmCausalAIBatch`` is CausalAI as it was designed: its data are many units, and one pass answers the same
intervention for all of them, one ``predict`` call per variable."""

from __future__ import annotations

import os
import sys
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.environ.get("CAUSALAI_PATH", "/mnt/data/yingli/counterprovenance/causalai_src/repo"))
from causalai.models.tabular.causal_inference import CausalInference  # noqa: E402


class _Noisy:
    """A fitted mechanism plus the noise of the unit: the model CausalAI calls ``predict`` on."""

    def __init__(self, model, eps: float) -> None:
        self.model, self.eps = model, eps

    def predict(self, x):
        return self.model.predict(x) + self.eps


class ScmCausalAI:
    """One unit (``unit`` a dict) or many (``unit`` a DataFrame, see ``ScmCausalAIBatch``)."""

    def __init__(self, scm, unit, treatments, outcome: str) -> None:
        if isinstance(unit, dict):
            names = [str(n) for n in unit]
            arr = np.array([[float(unit[n]) for n in names]])
        else:
            names = [str(n) for n in unit.columns]
            arr = unit.to_numpy(dtype=float)
        graph = {n: (list(scm.parents[n]) if n in scm.models else []) for n in names}
        ci = CausalInference(arr, names, graph, prediction_model=object)
        ci.data_processor.mean = np.zeros_like(ci.data_processor.mean)
        ci.data_processor.std = np.ones_like(ci.data_processor.std)
        relevant = ci.graph_obj.relevant_nodes([outcome])
        seq = [n for n in ci.node_seq if n in relevant] + [outcome]
        self.T = [str(t) for t in treatments]
        col = {n: i for i, n in enumerate(names)}
        eps = {n: arr[:, col[n]] - scm.models[n].predict(arr[:, [col[p] for p in graph[n]]]) for n in seq if graph[n]}
        if arr.shape[0] == 1:
            eps = {n: float(v[0]) for n, v in eps.items()}
        ci.models = {n: _Noisy(scm.models[n], eps[n]) for n in seq if n not in self.T and graph[n]}
        ci.target_var, ci.relevant_node_seq, ci.treatment_vars = outcome, seq, self.T
        self.ci, self.y = ci, outcome

    def query(self, t) -> float:
        """The treated pass of ``CausalInference.ate_causal_path`` on the unit."""
        return float(self.query_all(t)[0])

    def query_all(self, t) -> np.ndarray:
        """The treated pass of ``CausalInference.ate_causal_path`` on all units."""
        ci = self.ci
        treatment_data = deepcopy(ci.data)
        for name, v in zip(self.T, t):
            treatment_data.data_arrays[0][:, ci.data.var_name2index(name)] = float(v)
        for var in ci.relevant_node_seq:
            if var not in ci.treatment_vars and ci.causal_graph[var] != []:
                x, _, _ = treatment_data.get_causal_Xy(var, ci.causal_graph[var])
                if x is not None:
                    parents = ci.causal_graph[var]
                    x = ci.data_processor.transformX(x, parents)
                    pred = ci.models[var].predict(x)
                    pred = ci.data_processor.inv_transform(pred, [var])
                    treatment_data.data_arrays[0][:, ci.data.var_name2index(var)] = pred
        return treatment_data.data_arrays[0][:, ci.data.var_name2index(self.y)].copy()


    def query_many(self, Q) -> np.ndarray:
        """Many interventions on the one unit in one pass: its row repeated once per intervention, each row with its own
        values of the intervened variables (CausalAI accepts one treatment value per row)."""
        ci, Q = self.ci, np.asarray(Q, dtype=float)
        treatment_data = deepcopy(ci.data)
        treatment_data.data_arrays[0] = np.repeat(treatment_data.data_arrays[0][:1], len(Q), axis=0)
        for j, name in enumerate(self.T):
            treatment_data.data_arrays[0][:, ci.data.var_name2index(name)] = Q[:, j]
        for var in ci.relevant_node_seq:
            if var not in ci.treatment_vars and ci.causal_graph[var] != []:
                x, _, _ = treatment_data.get_causal_Xy(var, ci.causal_graph[var])
                if x is not None:
                    parents = ci.causal_graph[var]
                    x = ci.data_processor.transformX(x, parents)
                    pred = ci.models[var].predict(x)
                    pred = ci.data_processor.inv_transform(pred, [var])
                    treatment_data.data_arrays[0][:, ci.data.var_name2index(var)] = pred
        return treatment_data.data_arrays[0][:, ci.data.var_name2index(self.y)].copy()


ScmCausalAIBatch = ScmCausalAI
