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

    def test_queries(self):
        rng = np.random.default_rng(1)
        for bound in self.binds():
            a = ProvenanceIndex(bound, grouping="none", max_leaf_cells=10**8)
            b = ProvenanceIndex(bound, grouping="none", max_leaf_cells=10**8, native_cells=True, build_threads=2)
            for t in rng.normal(size=(200, bound.d)):
                x, y = a.query(t), b.query(t)
                self.assertAlmostEqual(x.value, y.value, places=10)
                self.assertEqual(sorted(x.boxes()), sorted(y.boxes()))


if __name__ == "__main__":
    unittest.main()
