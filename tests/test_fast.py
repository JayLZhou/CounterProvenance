"""Compiled query (greedy groups, packed lookup, leaf bitmasks), sessions,
online baselines and the numba kernels against brute force."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from counterprovenance import (
    CircuitArena,
    IncrementalLoop,
    OnlineLoop,
    ProvenanceIndex,
    Slabs,
    bind,
    greedy_groups,
    synthetic,
    tree_profiles,
    tune_budget,
)
from counterprovenance import accel


def _bound(seed: int, d: int, m: int, depth: int, n_features: int | None = None):
    rng = np.random.default_rng(seed)
    p = n_features or d
    trees = [synthetic.random_tree(rng, p, depth) for _ in range(m)]
    ensemble = synthetic.ensemble(trees, p, base=0.25)
    unit = rng.uniform(0, 1, size=p)
    axes = sorted(rng.choice(p, size=d, replace=False).tolist())
    return bind(ensemble, unit, axes, noise=-0.5), rng


def _queries(bound, rng, count: int = 120) -> np.ndarray:
    d = bound.d
    pts = list(rng.uniform(-0.1, 1.1, size=(count, d)))
    for a in range(d):
        th = sorted({b.hi[a] for b in bound.boxes if np.isfinite(b.hi[a])})
        for v in th[:: max(1, len(th) // 10)]:
            for w in (np.nextafter(v, -np.inf), v, np.nextafter(v, np.inf)):
                p = rng.uniform(-0.1, 1.1, size=d)
                p[a] = w
                pts.append(p)
    return np.array(pts)


class GroupingTests(unittest.TestCase):
    def test_greedy_respects_budget_and_partitions_trees(self) -> None:
        bound, _ = _bound(1, 4, 40, 3)
        slabs = Slabs(bound)
        profiles = tree_profiles(bound, slabs)
        for budget in (50, 5_000, 10**7):
            groups, deep = greedy_groups(profiles, budget, cap=10**6)
            trees = sorted(t for g in groups for t in g.trees) + sorted(p.tree for p in deep)
            self.assertEqual(sorted(trees), sorted(p.tree for p in profiles))
            for g in groups:
                self.assertLessEqual(g.cells, 10**6)
                for t in g.trees:
                    prof = next(p for p in profiles if p.tree == t)
                    self.assertTrue(prof.axes <= g.axes)
        few, _ = greedy_groups(profiles, 10**7, cap=10**6)
        many, _ = greedy_groups(profiles, 50, cap=10**6)
        self.assertLessEqual(len(few), len(many))

    def test_cap_sends_trees_to_bitmasks(self) -> None:
        bound, _ = _bound(2, 4, 30, 3)
        profiles = tree_profiles(bound, Slabs(bound))
        groups, deep = greedy_groups(profiles, 10**7, cap=4)
        self.assertTrue(deep)
        self.assertTrue(all(p.cells > 4 for p in deep))


class CompiledIndexTests(unittest.TestCase):
    def _check(self, index, bound, rng, provenance_every: int = 7) -> None:
        arena = CircuitArena()
        for q, t in enumerate(_queries(bound, rng, 80)):
            answer = index.query(t)
            self.assertAlmostEqual(answer.value, bound.value(t), places=10)
            if q % provenance_every == 0:
                expected = Counter(bound.label(i) for i in bound.selected(t))
                self.assertEqual(answer.labels(), expected)
                self.assertEqual(arena.atoms(answer.to_arena(arena)), expected)

    def test_greedy_compiled_is_exact(self) -> None:
        for seed, d, m, depth, p in ((20, 1, 30, 3, 1), (21, 2, 25, 3, 3), (22, 4, 30, 3, 6), (23, 6, 40, 2, 9)):
            for budget, cap in ((10**7, 1 << 22), (200, 1 << 22), (10**7, 6)):
                bound, rng = _bound(seed, d, m, depth, p)
                index = ProvenanceIndex(bound, budget=budget, grid_cap=cap)
                self.assertTrue(index.compiled)
                self._check(index, bound, rng)

    def test_bitmask_only(self) -> None:
        bound, rng = _bound(24, 5, 30, 3, 8)
        index = ProvenanceIndex(bound, grid_cap=1)
        self.assertEqual(index.stats.bitmask_trees, len(tree_profiles(bound, Slabs(bound))))
        self._check(index, bound, rng, provenance_every=3)

    def test_segment_tree_path_with_bitmasks(self) -> None:
        bound, rng = _bound(25, 3, 25, 3, 5)
        index = ProvenanceIndex(bound, tau=0, grid_cap=6)
        self.assertFalse(index.compiled)
        self._check(index, bound, rng)

    def test_session_matches_fresh_queries(self) -> None:
        bound, rng = _bound(26, 4, 35, 3, 6)
        for cap in (1 << 22, 6):
            index = ProvenanceIndex(bound, grid_cap=cap, budget=500)
            pts = _queries(bound, rng, 40)
            session = index.session(pts[0])
            loop = IncrementalLoop(bound.ensemble, bound, pts[0])
            for _ in range(150):
                k = int(rng.integers(bound.d))
                v = float(pts[int(rng.integers(len(pts)))][k])
                got = session.set(k, v)
                ref = loop.set(k, v)
                fresh = index.query(session.x).value
                self.assertAlmostEqual(got, fresh, places=10)
                self.assertAlmostEqual(got, ref, places=10)
            expected = Counter(bound.label(i) for i in bound.selected(session.x))
            self.assertEqual(session.answer().labels(), expected)

    def test_online_loops(self) -> None:
        bound, rng = _bound(27, 3, 30, 3, 7)
        full = OnlineLoop(bound.ensemble, bound.unit, bound.axes, noise=-0.5)
        bound_only = OnlineLoop(bound.ensemble, bound.unit, bound.axes, bound=bound)
        self.assertLessEqual(len(bound_only.trees), len(full.trees))
        for t in _queries(bound, rng, 40):
            self.assertAlmostEqual(full.query(t)[0], bound.value(t), places=10)
            self.assertAlmostEqual(bound_only.query(t)[0], bound.value(t), places=10)

    def test_tune_budget(self) -> None:
        bound, rng = _bound(28, 4, 30, 3, 6)
        sample = rng.uniform(0, 1, size=(30, 4))
        result = tune_budget(bound, sample, budgets=(100, 10_000), repeats=1)
        self.assertEqual(sum(1 for r in result.table if r.get("chosen")), 1)
        self._check(result.index, bound, rng)


@unittest.skipUnless(accel.HAVE_NUMBA, "numba not installed")
class NumbaTests(unittest.TestCase):
    def test_numba_index_and_loop(self) -> None:
        for seed, cap in ((30, 1 << 22), (31, 6), (32, 1)):
            bound, rng = _bound(seed, 4, 30, 3, 7)
            index = ProvenanceIndex(bound, grid_cap=cap, budget=2_000)
            fast = accel.NumbaIndex(index)
            loop = accel.NumbaLoop(bound.ensemble, bound.unit, bound.axes, noise=-0.5)
            loop_b = accel.NumbaLoop(bound.ensemble, bound.unit, bound.axes, bound=bound)
            T = _queries(bound, rng, 60)
            ref = np.array([bound.value(t) for t in T])
            np.testing.assert_allclose([fast.query(t) for t in T], ref, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(fast.batch(T), ref, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose([loop.query(t) for t in T], ref, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(loop_b.batch(T), ref, rtol=1e-12, atol=1e-12)
            for t in T[:20]:
                _, cells, leaves = fast.query_full(t)
                answer = index.query(t)
                self.assertEqual(list(cells), answer.cells)
                self.assertEqual(list(leaves), list(answer._leaves))


class WideTreeTests(unittest.TestCase):
    def test_trees_with_more_than_64_leaves(self) -> None:
        bound, rng = _bound(50, 3, 6, 8, 4)  # up to 256 leaves per tree
        self.assertGreater(max(sum(1 for b in bound.boxes if b.tree == m) for m in range(6)), 64)
        index = ProvenanceIndex(bound, grid_cap=1)
        CompiledIndexTests._check(self, index, bound, rng, provenance_every=2)
        if accel.HAVE_NUMBA:
            fast = accel.NumbaIndex(index)
            T = _queries(bound, rng, 40)
            np.testing.assert_allclose(fast.batch(T), [bound.value(t) for t in T], rtol=1e-12, atol=1e-12)


class RealModelTests(unittest.TestCase):
    def test_xgboost_deep_trees(self) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            self.skipTest("xgboost not installed")
        from counterprovenance import from_model

        rng = np.random.default_rng(40)
        X = rng.uniform(0, 1, size=(2000, 6))
        y = X[:, 0] * 3 + np.sin(5 * X[:, 1]) + X[:, 2] * X[:, 3] + rng.normal(0, 0.05, 2000)
        model = xgb.XGBRegressor(n_estimators=60, max_depth=6, learning_rate=0.1, n_jobs=1, random_state=0).fit(X, y)
        bound = bind(from_model(model), X[3], [0, 1, 2, 3, 4])
        for cap in (1 << 22, 64):
            index = ProvenanceIndex(bound, grid_cap=cap)
            T = rng.uniform(-0.1, 1.1, size=(150, 5))
            for t in T:
                full = bound.full_input(t)
                self.assertAlmostEqual(index.query(t).value, float(model.predict(full[None, :])[0]), delta=5e-5)
            for t in T[:10]:
                expected = Counter(bound.label(i) for i in bound.selected(t))
                self.assertEqual(index.query(t).labels(), expected)


if __name__ == "__main__":
    unittest.main()
