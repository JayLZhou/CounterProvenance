"""Exact overlay, MCPST leaves, and the provenance index against brute force."""

from __future__ import annotations

import itertools
import unittest
from collections import Counter

import numpy as np

from counterprovenance import (
    INACTIVE,
    CellIndex,
    CircuitArena,
    Overlay,
    ProvenanceIndex,
    Slabs,
    autotune,
    bind,
)
from counterprovenance import synthetic


def _bound(seed: int, d: int, m: int, depth: int, stumps: bool = False):
    rng = np.random.default_rng(seed)
    if stumps:
        trees = [synthetic.stump(int(rng.integers(d)), float(rng.uniform()), float(rng.normal()), float(rng.normal())) for _ in range(m)]
    else:
        trees = [synthetic.random_tree(rng, d, depth) for _ in range(m)]
    ensemble = synthetic.ensemble(trees, d, base=0.25)
    return bind(ensemble, np.zeros(d), list(range(d)), noise=-0.5), rng


def _queries(bound, rng, count: int = 150) -> np.ndarray:
    d = bound.d
    pts = list(rng.uniform(-0.1, 1.1, size=(count, d)))
    for a in range(d):
        th = sorted({b.hi[a] for b in bound.boxes if np.isfinite(b.hi[a])})
        for v in th[:: max(1, len(th) // 15)]:
            for w in (np.nextafter(v, -np.inf), v, np.nextafter(v, np.inf)):
                p = rng.uniform(-0.1, 1.1, size=d)
                p[a] = w
                pts.append(p)
    return np.array(pts)


def _cell_index(bound, **kwargs) -> tuple[CellIndex, Slabs]:
    slabs = Slabs(bound)
    items = list(range(len(bound.boxes)))
    cells = CellIndex(
        items=items,
        ranges=[slabs.ranges[i] for i in items],
        slots=[bound.boxes[i].tree for i in items],
        slab_counts=slabs.counts,
        values=[bound.boxes[i].value for i in items],
        offset=bound.offset,
        **kwargs,
    )
    return cells, slabs


class OverlayTests(unittest.TestCase):
    def test_faces_have_constant_signatures(self) -> None:
        for seed, depth in ((0, 2), (1, 3)):
            bound, _ = _bound(seed, 2, 12, depth)
            slabs = Slabs(bound)
            items = list(range(len(bound.boxes)))
            overlay = Overlay([slabs.ranges[i] for i in items], [bound.boxes[i].tree for i in items], slabs.counts)
            grid = Overlay([slabs.ranges[i] for i in items], [bound.boxes[i].tree for i in items], slabs.counts, exact=False)
            self.assertLessEqual(overlay.n_faces, grid.n_faces)
            self.assertEqual(grid.n_faces, grid.grid_cells)
            # brute-force signature of every grid cell
            sig_of_face = {}
            for cell in range(overlay.grid_cells):
                coords = [(cell // overlay.strides[k]) % overlay.shape[k] for k in range(2)]
                sig = [INACTIVE] * overlay.n_slots
                for i, loc in enumerate(overlay.local):
                    if all(loc[k][0] <= coords[k] < loc[k][1] for k in range(2)):
                        sig[overlay.slots[i]] = i
                face = int(overlay.face_of_cell[cell])
                sig_of_face.setdefault(face, sig)
                self.assertEqual(sig_of_face[face], sig)
            for e in overlay.edges:
                sa, sb = sig_of_face[e.a], sig_of_face[e.b]
                changed = {s for s in range(overlay.n_slots) if sa[s] != sb[s]}
                self.assertEqual(set(e.changes), changed)
                for s, (ia, ib) in e.changes.items():
                    self.assertEqual((sa[s], sb[s]), (ia, ib))


class CellIndexTests(unittest.TestCase):
    def test_every_layout_is_exact(self) -> None:
        for (seed, d, m, depth, stumps), layout, exact in itertools.product(
            [(2, 2, 14, 2, False), (3, 2, 30, 1, True), (4, 3, 8, 2, False), (5, 1, 20, 3, False)],
            ("path", "node", "hybrid"),
            (True, False),
        ):
            bound, rng = _bound(seed, d, m, depth, stumps)
            cells, slabs = _cell_index(bound, layout=layout, exact=exact)
            for face in range(cells.overlay.n_faces):
                version = int(cells.face_version[face])
                self.assertGreaterEqual(version, 0)
                self.assertEqual(cells.pft.payloads(version), cells.overlay.signature(face))
            for t in _queries(bound, rng, 60):
                x = bound.prepare(t)
                piece = cells.piece(slabs.locate(x))
                self.assertAlmostEqual(piece.value, bound.value(t), places=10)
                self.assertEqual(sorted(piece.boxes()), sorted(bound.selected(t).tolist()))

    def test_snake_walk_on_full_grid(self) -> None:
        bound, rng = _bound(6, 2, 24, 1, stumps=True)
        cells, slabs = _cell_index(bound, layout="node", exact=False, walk="snake")
        self.assertEqual(cells.stats.sources, 1)
        for t in _queries(bound, rng, 40):
            piece = cells.piece(slabs.locate(bound.prepare(t)))
            self.assertEqual(sorted(piece.boxes()), sorted(bound.selected(t).tolist()))

    def test_node_copying_beats_path_copying_on_sparse_changes(self) -> None:
        rng = np.random.default_rng(7)
        trees = [synthetic.stump(i % 2, (i // 2 + 1) / 66, float(rng.normal()), float(rng.normal())) for i in range(128)]
        bound = bind(synthetic.ensemble(trees, 2), np.zeros(2), [0, 1])
        path, _ = _cell_index(bound, layout="path", exact=False)
        node, _ = _cell_index(bound, layout="node", exact=False, walk="snake")
        self.assertLess(node.pft.records, path.pft.records)


class ProvenanceIndexTests(unittest.TestCase):
    def _check(self, index: ProvenanceIndex, bound, rng) -> None:
        arena = CircuitArena()
        for t in _queries(bound, rng, 80):
            answer = index.query(t)
            self.assertAlmostEqual(answer.value, bound.value(t), places=10)
            expected = Counter(bound.label(i) for i in bound.selected(t))
            self.assertEqual(answer.labels(), expected)
        root = answer.to_arena(arena)
        self.assertEqual(arena.atoms(root), expected)

    def test_all_configurations_are_exact(self) -> None:
        for seed, d, m, depth in ((10, 1, 25, 3), (11, 2, 20, 2), (12, 3, 15, 2), (13, 3, 25, 1)):
            for tau, budget, grouping, layout in itertools.product(
                (0, 16, float("inf")), (None, 2, 1), ("support", "none"), ("hybrid", "path")
            ):
                bound, rng = _bound(seed, d, m, depth, stumps=depth == 1)
                index = ProvenanceIndex(bound, tau=tau, group_budget=budget, grouping=grouping, layout=layout)
                self._check(index, bound, rng)

    def test_endpoints(self) -> None:
        bound, rng = _bound(14, 2, 20, 2)
        single = ProvenanceIndex(bound, tau=float("inf"), grouping="none")
        self.assertEqual(single.stats.leaves, 1)
        for t in _queries(bound, rng, 30):
            self.assertEqual(len(single.query(t).pieces), 1)
        chains = ProvenanceIndex(bound, tau=0, grouping="none")
        self.assertTrue(all(len(leaf.overlay.cuts) <= 1 for leaf in chains.leaves))
        for t in _queries(bound, rng, 30):
            self.assertLessEqual(len(chains.query(t).pieces), 2 * int(np.ceil(np.log2(len(bound.boxes)))) + 2)

    def test_autotune_respects_piece_budget(self) -> None:
        bound, rng = _bound(15, 3, 20, 2)
        sample = rng.uniform(0, 1, size=(40, 3))
        result = autotune(bound, sample, taus=(0, 64, 4096), budgets=(None, 2, 1), pieces_budget=6)
        chosen = [row for row in result.table if row.get("chosen")]
        self.assertEqual(len(chosen), 1)
        feasible = [row for row in result.table if row["mean_pieces"] <= 6]
        if feasible:
            self.assertLessEqual(chosen[0]["mean_pieces"], 6)
            self.assertEqual(chosen[0]["records"], min(r["records"] for r in feasible))
        self._check(result.index, bound, rng)

    def test_sklearn_end_to_end(self) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        from counterprovenance import from_model

        rng = np.random.default_rng(16)
        X = rng.uniform(0, 1, size=(1200, 4))
        y = X[:, 0] * 3 + np.sin(5 * X[:, 1]) + (X[:, 2] > 0.5) + rng.normal(0, 0.05, 1200)
        model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=0).fit(X, y)
        unit = X[5]
        bound = bind(from_model(model), unit, [0, 1, 2])
        index = ProvenanceIndex(bound, tau=512)
        for t in rng.uniform(-0.1, 1.1, size=(100, 3)):
            full = bound.full_input(t)
            self.assertAlmostEqual(index.query(t).value, float(model.predict(full[None, :])[0]), places=9)


if __name__ == "__main__":
    unittest.main()
