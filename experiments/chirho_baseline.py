"""Counterfactuals with ChiRho (Pyro) over the same DAG and the same fitted models.

The SCM is a Pyro program: a treatment is a sample site at its observed value, a
non-descendant of the treatments keeps its observed value, and every mediator and
the outcome is ``X := f_X(pa) + N_X`` with ``f_X`` the fitted model's own
``predict``.  Each query abducts the noise again (``N_X = u_X - f_X(pa_u)``, no
caching), runs the program under ``TwinWorldCounterfactual`` and ``do(T = t)``, and
reads the outcome of the counterfactual world.  The additive noise is determined by
the observation, so its posterior is a point mass and the answer is exact up to
the libraries' rounding.  Needs ``torch``, ``pyro-ppl`` and ``chirho`` on the path.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class ScmChiRho:
    def __init__(self, scm, unit: dict[str, float], treatments: Sequence[str], outcome: str) -> None:
        import pyro
        import pyro.distributions as dist
        import torch
        from chirho.counterfactual.handlers import TwinWorldCounterfactual
        from chirho.indexed.ops import IndexSet, gather
        from chirho.interventional.handlers import do

        from counterprovenance.scm import relevant_nodes

        self.torch, self.pyro, self.dist = torch, pyro, dist
        self.Twin, self.do, self.IndexSet, self.gather = TwinWorldCounterfactual, do, IndexSet, gather
        self.scm, self.unit, self.treatments, self.outcome = scm, dict(unit), list(treatments), outcome
        self.order = relevant_nodes(scm.parents, treatments, outcome)
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)  # queries are timed single-threaded, like every other method

    def _f(self, node, parents):
        """The fitted model's predict on (possibly world-batched) parent tensors."""
        torch = self.torch
        cols = torch.broadcast_tensors(*[torch.as_tensor(p, dtype=torch.float64) for p in parents])
        shape = cols[0].shape
        X = torch.stack([c.reshape(-1) for c in cols], dim=1).numpy()
        y = np.asarray(self.scm.models[node].predict(X), dtype=np.float64)
        return torch.as_tensor(y).reshape(shape)

    def _program(self):
        torch, pyro, dist = self.torch, self.pyro, self.dist
        unit, scm = self.unit, self.scm
        values = {}
        for name in self.treatments:
            values[name] = pyro.sample(name, dist.Delta(torch.tensor(float(unit[name]))))
        for node in self.order:
            parents = scm.parents[node]
            # abduction, redone on every query: N = u - f(pa observed)
            f_obs = self._f(node, [torch.tensor(float(unit[p])) for p in parents])
            noise = torch.tensor(float(unit[node])) - f_obs
            pa = [values[p] if p in values else torch.tensor(float(unit[p])) for p in parents]
            values[node] = pyro.sample(node, dist.Delta(self._f(node, pa) + noise))
        return values[self.outcome]

    def query(self, t: Sequence[float]) -> float:
        torch = self.torch
        actions = {name: torch.tensor(float(v)) for name, v in zip(self.treatments, t)}
        with self.Twin(), self.do(actions=actions):
            y = self._program()
            # the twin worlds are indexed by TwinWorldCounterfactual's split name: 0 factual, 1 counterfactual
            y_cf = self.gather(y, self.IndexSet(**{self.Twin.fresh_prefix: {1}}))
        return float(torch.as_tensor(y_cf).reshape(-1)[0])
