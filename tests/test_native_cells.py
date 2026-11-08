"""The C++ cell graph has the same cells, edges, MST weight and persistent tree as the Python one."""

import unittest

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from counterprovenance import ProvenanceIndex, native_cells
from counterprovenance.cellindex import CellIndex
from counterprovenance.ensemble import Slabs, bind, from_model
from counterprovenance.native_cells import LeanCellIndex, NativeCellIndex


@unittest.skipUnless(native_cells.available(), "no C++ compiler")
class NativeCells(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(1500, 5))
        self.y = np.sin(self.X[:, 0]) * self.X[:, 1] + np.abs(self.X[:, 2]) + 0.1 * rng.normal(size=1500)

    def binds(self):
        for model in (GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0),
                      RandomForestRegressor(n_estimators=8, max_depth=5, random_state=0)):
            ens = from_model(model.fit(self.X, self.y))
            for axes in ([0], [0, 1], [2, 0, 1]):
                yield bind(ens, self.X[5], axes=axes, noise=0.25)

    def test_subtree_sums(self):
        # the sums are not stored with the index: derived from the children, a version sums its items
        for bound in self.binds():
            sl = Slabs(bound)
            d = bound.d
            items = list(range(len(bound.boxes)))
            kw = dict(items=items, ranges=[[sl.ranges[i][a] for a in range(d)] for i in items],
                      slots=[bound.boxes[i].tree for i in items], slab_counts=[sl.counts[a] for a in range(d)],
                      values=[bound.boxes[i].value for i in items])
            cc = NativeCellIndex(**kw)
            self.assertIsNone(cc.pft._sums)
            sums = cc.pft.sums
            vals = np.asarray(kw["values"], dtype=np.float64)
            for root in sorted(set(cc.face_version.tolist())):
                if root >= 0:
                    self.assertAlmostEqual(sums[root], float(vals[cc.pft.items(root)].sum()), places=9)

    def test_same_graph(self):
        for bound in self.binds():
            sl = Slabs(bound)
            d = bound.d
            items = list(range(len(bound.boxes)))
            kw = dict(items=items, ranges=[[sl.ranges[i][a] for a in range(d)] for i in items],
                      slots=[bound.boxes[i].tree for i in items], slab_counts=[sl.counts[a] for a in range(d)],
                      values=[bound.boxes[i].value for i in items])
            py = CellIndex(layout="path", **kw)
            for threads in (1, 4):
                cc = NativeCellIndex(threads=threads, **kw)
                self.assertEqual(py.overlay.cuts, cc.overlay.cuts)
                self.assertEqual(py.stats.faces, cc.stats.faces)
                self.assertEqual(py.stats.face_edges, cc.stats.face_edges)
                self.assertEqual(py.stats.tree_weight, cc.stats.tree_weight)
                self.assertEqual(py.stats.pft_nodes, cc.stats.pft_nodes)
                np.testing.assert_array_equal(py.overlay.face_of_cell, cc.overlay.face_of_cell)
                np.testing.assert_allclose(py.face_value, cc.face_value, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(py.face_empty, cc.face_empty)
                for f in range(py.stats.faces):
                    self.assertEqual(sorted(py.pft.active(int(py.face_version[f]))),
                                     sorted(cc.pft.active(int(cc.face_version[f]))))

    def test_lean_table(self):
        for bound in self.binds():
            sl = Slabs(bound)
            d = bound.d
            items = list(range(len(bound.boxes)))
            kw = dict(items=items, ranges=[[sl.ranges[i][a] for a in range(d)] for i in items],
                      slots=[bound.boxes[i].tree for i in items], slab_counts=[sl.counts[a] for a in range(d)],
                      values=[bound.boxes[i].value for i in items])
            # the Python cell graph over the grid cells with the same edge weights
            py = CellIndex(layout="path", exact=False, plan_weight="count", **kw)
            for threads in (1, 4):
                cc = LeanCellIndex(threads=threads, **kw)
                self.assertEqual(py.overlay.cuts, cc.overlay.cuts)
                self.assertEqual(py.stats.tree_weight, cc.stats.tree_weight)
                np.testing.assert_allclose(py.face_value[py.overlay.face_of_cell], cc.table, rtol=1e-12, atol=1e-12)

    def test_lean_queries(self):
        rng = np.random.default_rng(2)
        for bound in self.binds():
            a = ProvenanceIndex(bound, grouping="none", max_leaf_cells=10**8)
            b = ProvenanceIndex(bound, grouping="greedy", budget=10**7, grid_cap=10**7, mask_ratio=float("inf"),
                                lean_cells=True, build_threads=2)
            for t in rng.normal(size=(300, bound.d)):
                self.assertAlmostEqual(a.query(t).value, b.query(t).value, places=10)

    def test_tables_mode(self):
        rng = np.random.default_rng(4)
        for bound in self.binds():
            a = ProvenanceIndex(bound, grouping="none", max_leaf_cells=10**8)
            for kw in (dict(budget=10**7), dict(budget=20000, np_read_groups=1, one_cells=1)):
                b = ProvenanceIndex(bound, grouping="tables", build_threads=2, **kw)
                self.assertFalse(b.deep)
                for t in rng.normal(size=(200, bound.d)):
                    self.assertAlmostEqual(a.query(t).value, b.query(t).value, places=9)

    def test_one_group_fallback(self):
        # one group that does not fit (as a persistent tree over 2^31 nodes would) falls back to the greedy groups
        from counterprovenance import index as index_module
        rng = np.random.default_rng(5)
        real = index_module.NativeCellIndex
        n_trees = {}

        class Failing(real):
            def __init__(self, items, ranges, slots, *args, **kw):
                if not n_trees.get("failed") and len(set(slots)) == n_trees["all"] and n_trees["all"] > 1:
                    n_trees["failed"] = True   # only the first build, the one group of all trees
                    raise MemoryError("cell graph: status 2")
                super().__init__(items, ranges, slots, *args, **kw)

        for bound in self.binds():
            a = ProvenanceIndex(bound, grouping="none", max_leaf_cells=10**8)
            n_trees.clear()
            n_trees["all"] = len({box.tree for box in bound.boxes})
            index_module.NativeCellIndex = Failing
            try:
                b = ProvenanceIndex(bound, grouping="tables", budget=20000, one_cells=10**9)
            finally:
                index_module.NativeCellIndex = real
            if n_trees["all"] > 1:
                self.assertTrue(b.one_fallback)
                self.assertEqual(b.choice, "tables")
            for t in rng.normal(size=(200, bound.d)):
                self.assertAlmostEqual(a.query(t).value, b.query(t).value, places=9)

    def test_queries(self):
        rng = np.random.default_rng(1)
        for bound in self.binds():
            a = ProvenanceIndex(bound, grouping="none", max_leaf_cells=10**8)
            b = ProvenanceIndex(bound, grouping="none", max_leaf_cells=10**8, native_cells=True, build_threads=2)
            for t in rng.normal(size=(200, bound.d)):
                x, y = a.query(t), b.query(t)
                self.assertAlmostEqual(x.value, y.value, places=10)
                self.assertEqual(sorted(x.boxes()), sorted(y.boxes()))


    def test_query_batch(self):
        for bound in self.binds():
            for budget in (64, 1 << 20):
                try:
                    index = ProvenanceIndex(bound, grouping="tables", budget=budget, one_cells=16)
                except MemoryError:
                    continue
                rng = np.random.default_rng(1)
                Q = rng.normal(size=(300, bound.d))
                one = np.array([index.query(t).value for t in Q])
                self.assertLess(np.abs(index.query_batch(Q, chunk=70) - one).max(), 1e-9)


    def test_nearest_is_optimal(self):
        for bound in self.binds():
            if bound.d > 2:
                continue
            many = ProvenanceIndex(bound, grouping="tables", budget=16, one_cells=4, np_read_groups=2)
            index = ProvenanceIndex(bound, grouping="tables", budget=1 << 22, one_cells=1 << 22)
            rng = np.random.default_rng(2)
            values = [rng.normal(size=40) for _ in range(bound.d)]
            scale = rng.uniform(0.5, 2.0, size=bound.d)
            u = rng.normal(size=bound.d)
            axes = [np.r_[values[a], u[a]] for a in range(bound.d)]
            grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(bound.d, -1).T
            y = np.array([index.query(t).value for t in grid])
            brute = (np.abs(grid - u) / scale).sum(axis=1)
            for tau in np.quantile(y, [0.5, 0.9, 0.99]) - 1e-9:     # below a value, whose sum may round differently
                t, cost, value = index.nearest(u, tau, values, scale, k=1)
                self.assertAlmostEqual(cost[0], brute[y >= tau].min(), places=9)
                self.assertGreaterEqual(index.query(t[0]).value, tau - 1e-9)
                self.assertAlmostEqual(index.query(t[0]).value, value[0], places=9)
                self.assertAlmostEqual(many.nearest(u, tau, values, scale, k=1)[1][0], cost[0], places=9)
                t5, cost5, _ = index.nearest(u, tau, values, scale, k=5)
                self.assertAlmostEqual(cost5[0], cost[0], places=12)
                self.assertTrue(np.all(np.diff(cost5) >= 0))


if __name__ == "__main__":
    unittest.main()
