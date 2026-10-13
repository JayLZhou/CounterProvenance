"""Model compilers and unit binding against the fitted models themselves."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from counterprovenance import bind, from_model

HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_LGBM = importlib.util.find_spec("lightgbm") is not None


def _data(seed: int = 0, n: int = 1500, d: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, d))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * (X[:, 2] > 0) + 0.3 * X[:, 3] + rng.normal(0, 0.1, n)
    return X, y


def _threshold_rows(ensemble, X, rng, count=300):
    """Rows whose features sit exactly on split thresholds (boundary semantics)."""
    rows = X[rng.integers(len(X), size=count)].copy()
    for row in rows:
        tree = ensemble.trees[int(rng.integers(ensemble.tree_count))]
        internal = np.flatnonzero(tree.feature >= 0)
        if len(internal):
            node = int(rng.choice(internal))
            row[tree.feature[node]] = tree.threshold[node]
    return rows


class SklearnCompilerTests(unittest.TestCase):
    def _check(self, model, X, leaf_api=None) -> None:
        ensemble = from_model(model)
        rng = np.random.default_rng(1)
        rows = np.vstack([X[:400], _threshold_rows(ensemble, X, rng)])
        np.testing.assert_allclose(ensemble.predict(rows), model.predict(rows), rtol=0, atol=1e-9)
        if leaf_api is not None:
            np.testing.assert_array_equal(ensemble.leaves(rows), leaf_api(rows))

    def test_gradient_boosting(self) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        X, y = _data()
        model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0).fit(X, y)

        def leaves(R):
            out = model.apply(R)
            return out[:, :, 0] if out.ndim == 3 else out

        self._check(model, X, leaves)

    def test_hist_gradient_boosting(self) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        X, y = _data(1)
        model = HistGradientBoostingRegressor(max_iter=60, max_depth=3, random_state=0).fit(X, y)
        self._check(model, X)

    def test_random_forest(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        X, y = _data(2)
        model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
        self._check(model, X, model.apply)


@unittest.skipUnless(HAS_XGB, "xgboost not installed")
class XGBoostCompilerTests(unittest.TestCase):
    def test_strict_less_than_splits_and_leaves(self) -> None:
        import xgboost as xgb

        X, y = _data(3)
        for method in ("exact", "hist"):
            model = xgb.XGBRegressor(n_estimators=60, max_depth=3, tree_method=method, random_state=0).fit(X, y)
            ensemble = from_model(model)
            rng = np.random.default_rng(4)
            # points exactly on the original float32 split values (x < theta goes left)
            rows = X[rng.integers(len(X), size=300)].copy()
            booster_trees = model.get_booster()
            for row in rows:
                tree = ensemble.trees[int(rng.integers(ensemble.tree_count))]
                internal = np.flatnonzero(tree.feature >= 0)
                node = int(rng.choice(internal))
                original = np.nextafter(np.float32(tree.threshold[node]), np.float32(np.inf))
                row[tree.feature[node]] = float(original)
            rows = np.vstack([X[:300], rows])
            leaves = booster_trees.predict(xgb.DMatrix(rows), pred_leaf=True)
            np.testing.assert_array_equal(ensemble.leaves(rows), leaves)
            np.testing.assert_allclose(ensemble.predict(rows), model.predict(rows), rtol=0, atol=2e-5)


@unittest.skipUnless(HAS_LGBM, "lightgbm not installed")
class LightGBMCompilerTests(unittest.TestCase):
    def test_leaves_and_values(self) -> None:
        import lightgbm as lgb

        X, y = _data(5)
        model = lgb.LGBMRegressor(n_estimators=60, num_leaves=15, verbose=-1, random_state=0).fit(X, y)
        ensemble = from_model(model)
        rng = np.random.default_rng(6)
        rows = np.vstack([X[:300], _threshold_rows(ensemble, X, rng)])
        native = model.predict(rows, pred_leaf=True)
        ours = ensemble.leaves(rows)
        mapped = np.array([[t.native_leaf[n] for t, n in zip(ensemble.trees, row)] for row in ours])
        np.testing.assert_array_equal(mapped, native)
        np.testing.assert_allclose(ensemble.predict(rows), model.predict(rows), rtol=0, atol=1e-9)


class BindingTests(unittest.TestCase):
    def test_bound_value_equals_full_prediction(self) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        X, y = _data(7)
        model = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0).fit(X, y)
        ensemble = from_model(model)
        rng = np.random.default_rng(8)
        for axes in ([0], [1, 2], [0, 2, 3]):
            unit = X[int(rng.integers(len(X)))]
            bound = bind(ensemble, unit, axes)
            ts = rng.uniform(-2.5, 2.5, size=(200, len(axes)))
            # boundary points on the intervention axes
            for a in range(len(axes)):
                th = sorted({b.hi[a] for b in bound.boxes if np.isfinite(b.hi[a])})
                for v in th[:20]:
                    t = rng.uniform(-2.5, 2.5, size=len(axes))
                    t[a] = v
                    ts = np.vstack([ts, t])
            for t in ts:
                selected = bound.selected(t)
                self.assertEqual(len({bound.boxes[i].tree for i in selected}), ensemble.tree_count)
                self.assertEqual(len(selected), ensemble.tree_count)
                full = bound.full_input(t)
                self.assertAlmostEqual(bound.value(t), float(model.predict(full[None, :])[0]), places=9)


if __name__ == "__main__":
    unittest.main()
