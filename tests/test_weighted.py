"""Queries that choose the trees' weights at query time agree with the three steps."""

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from counterprovenance import ProvenanceIndex, native_cells
from counterprovenance.scm import bind_scm, fit_scm
from counterprovenance.scm_baselines import ScmThreeStepPy
from counterprovenance.weighted import WeightedIndex


@unittest.skipUnless(native_cells.available(), "no C++ compiler")
class Weighted(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 2000
        x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.uniform(-2, 2, size=n)
        z = np.sin(x1) + 0.5 * x2 + 0.2 * rng.normal(size=n)
        y = z + x1 * x3 + 0.1 * rng.normal(size=n)
        self.df = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, z=z, y=y))
        self.edges = [("x1", "z"), ("x2", "z"), ("z", "y"), ("x1", "y"), ("x3", "y")]

    def check(self, factory, grouping, **kw):
        scm = fit_scm(self.df, self.edges, "y", ["x1"], factory)
        rng = np.random.default_rng(1)
        M = len(scm.ensembles["y"].trees)
        for u in (0, 11):
            unit = self.df.iloc[u].to_dict()
            for T in (["x1"], ["x1", "x2"], ["x3", "x1", "x2"]):
                bound = bind_scm(scm, unit, T, "y", threads=2)
                index = ProvenanceIndex(bound, grouping=grouping, native_cells=True, build_threads=2, **kw)
                wi = WeightedIndex(index, unit["y"], [unit[a] for a in T])
                three = ScmThreeStepPy(scm, unit, T, "y")
                Q = rng.normal(size=(40, len(T)))
                for t in Q:
                    for n in (0, 1, M // 3, M - 1, M):
                        self.assertAlmostEqual(wi.prefix(t, n), three.query_prefix(t, n), places=9)
                    w = rng.uniform(0, 2, size=M)
                    self.assertAlmostEqual(wi.weighted(t, w), three.query_weighted(t, w), places=9)
                    self.assertAlmostEqual(wi.weighted(t, np.ones(M)), three.query(t)[0], places=9)

    def test_one_graph(self):
        self.check(lambda: GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0), "none",
                   max_leaf_cells=10**8)

    def test_greedy_groups(self):
        self.check(lambda: GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=0), "greedy", budget=2000)

    def test_deep_trees_in_masks(self):
        self.check(lambda: RandomForestRegressor(n_estimators=10, max_depth=6, random_state=0), "greedy", budget=100,
                   grid_cap=50)


if __name__ == "__main__":
    unittest.main()
