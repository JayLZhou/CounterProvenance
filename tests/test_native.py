"""The C++ compiler returns the same leaf boxes as the Python one, in the same order."""

import math
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from counterprovenance import native_compile
from counterprovenance.scm import bind_scm, fit_scm


def same_boxes(a, b):
    assert len(a.boxes) == len(b.boxes), (len(a.boxes), len(b.boxes))
    for x, y in zip(a.boxes, b.boxes):
        assert (x.tree, x.node, x.native_leaf) == (y.tree, y.node, y.native_leaf)
        assert x.lo == y.lo and x.hi == y.hi and x.value == y.value
    assert a.offset == b.offset


@unittest.skipUnless(native_compile.available(), "no C++ compiler")
class NativeCompile(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 3000
        x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.uniform(-2, 2, size=n)
        z1 = np.sin(x1) + 0.5 * x2 + 0.2 * rng.normal(size=n)
        z2 = z1 * x2 + 0.2 * rng.normal(size=n)
        y = z1 + np.abs(z2) + x1 * x3 + 0.1 * rng.normal(size=n)
        self.df = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, z1=z1, z2=z2, y=y))
        self.edges = [("x1", "z1"), ("x2", "z1"), ("z1", "z2"), ("x2", "z2"), ("z1", "y"), ("z2", "y"),
                      ("x1", "y"), ("x3", "y")]

    def check(self, factory):
        scm = fit_scm(self.df, self.edges, "y", ["x1"], factory)
        for u in (0, 11):
            unit = self.df.iloc[u].to_dict()
            for T in (["x1"], ["x1", "x2"], ["x3", "x1", "x2"], ["z1"], ["x2", "z2"]):
                py = bind_scm(scm, unit, T, "y", native=False)
                same_boxes(bind_scm(scm, unit, T, "y", native=True), py)
                same_boxes(bind_scm(scm, unit, T, "y", native=True, threads=4), py)

    def test_gbr(self):
        self.check(lambda: GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0))

    def test_hgb(self):
        self.check(lambda: HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=15, random_state=0))

    def test_rf(self):
        self.check(lambda: RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0))

    def test_xgboost(self):
        try:
            import xgboost as xgb
        except ImportError:
            self.skipTest("no xgboost")
        self.check(lambda: xgb.XGBRegressor(n_estimators=40, max_depth=4, tree_method="hist", n_jobs=1, random_state=0))

    def test_lightgbm(self):
        try:
            import lightgbm as lgb
        except ImportError:
            self.skipTest("no lightgbm")
        self.check(lambda: lgb.LGBMRegressor(n_estimators=40, num_leaves=15, verbose=-1, n_jobs=1, random_state=0))

    def test_max_boxes(self):
        scm = fit_scm(self.df, self.edges, "y", ["x1"],
                      lambda: GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0))
        from counterprovenance.scm import Infeasible
        with self.assertRaises(Infeasible):
            bind_scm(scm, self.df.iloc[0].to_dict(), ["x1", "x2"], "y", max_boxes=5, native=True)

    def test_fsum_matches_math(self):
        # the sum the mediators use, on values whose naive sum is wrong
        self.assertEqual(math.fsum([1e16, 1.0, -1e16]), 1.0)


if __name__ == "__main__":
    unittest.main()
