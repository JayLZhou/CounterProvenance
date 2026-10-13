"""Semiring circuits, the persistent factor tree, and construction plans."""

from __future__ import annotations

import itertools
import random
import unittest
from collections import Counter

from counterprovenance import (
    CircuitArena,
    FaceEdge,
    PersistentFactorTree,
    ancestor_union_size,
    best_walk,
    boustrophedon,
    edge_weights,
    euler_walk,
    greedy_dfs_forest,
    shortcut_walk,
    spanning_forest,
)


class SemiringTests(unittest.TestCase):
    def test_addition_keeps_multiplicity(self) -> None:
        arena = CircuitArena()
        a = arena.atom("a")
        root = arena.add(a, a)
        self.assertEqual(arena.atoms(root), Counter({"a": 2}))
        self.assertEqual(arena.evaluate(root, {"a": 3.0}), 6.0)
        self.assertEqual(arena.add(a, arena.zero), a)

    def test_hash_consing_and_balanced_sum(self) -> None:
        arena = CircuitArena()
        atoms = [arena.atom(i) for i in range(7)]
        r1 = arena.add_all(atoms)
        r2 = arena.add_all(atoms)
        self.assertEqual(r1, r2)
        self.assertEqual(arena.atoms(r1), Counter(range(7)))


class PersistentFactorTreeTests(unittest.TestCase):
    def test_ancestor_union(self) -> None:
        self.assertEqual(ancestor_union_size([0], 4), 2)
        self.assertEqual(ancestor_union_size([0, 1], 4), 2)
        self.assertEqual(ancestor_union_size([1, 2], 4), 3)
        self.assertEqual(ancestor_union_size([0, 1, 2, 3], 4), 3)
        self.assertEqual(ancestor_union_size([], 4), 0)

    def _random_chain(self, mode: str, seed: int, n: int = 37, versions: int = 300, k: int = 3) -> None:
        rng = random.Random(seed)
        pft = PersistentFactorTree(n, capacity=2)
        state = [rng.randrange(-1, 5) for _ in range(n)]
        reference = [list(state)]
        pft.build(state)
        changes_total = 0
        for _ in range(versions):
            changes = {}
            for slot in rng.sample(range(n), rng.randint(0, k)):
                value = rng.randrange(-1, 5)
                if value != state[slot]:
                    changes[slot] = value
                    state[slot] = value
            changes_total += len(changes)
            pft.advance(changes, mode=mode)
            reference.append(list(state))
        for v, expected in enumerate(reference):
            self.assertEqual(pft.payloads(v), expected)
            slot = rng.randrange(n)
            self.assertEqual(pft.slot(v, slot), expected[slot])
        if mode == "node":
            self.assertEqual(pft.path_nodes, 0)
            self.assertLessEqual(pft.entries, changes_total)
            # O(1) amortised records per change on a chain
            self.assertLessEqual(pft.records, (pft.width - 1) + 3 * changes_total)

    def test_chain_node_copying(self) -> None:
        for seed in range(5):
            self._random_chain("node", seed)

    def test_chain_path_copying(self) -> None:
        self._random_chain("path", 11)

    def test_chain_auto(self) -> None:
        for seed in range(3):
            self._random_chain("auto", 20 + seed, k=9)

    def test_branching_path_copying(self) -> None:
        rng = random.Random(3)
        n = 21
        pft = PersistentFactorTree(n)
        states = [[rng.randrange(4) for _ in range(n)]]
        pft.build(states[0])
        for _ in range(200):
            parent = rng.randrange(len(states))
            state = list(states[parent])
            changes = {}
            for slot in rng.sample(range(n), 2):
                state[slot] = rng.randrange(4)
                changes[slot] = state[slot]
            pft.derive_path(parent, changes)
            states.append(state)
        for v, expected in enumerate(states):
            self.assertEqual(pft.payloads(v), expected)
        self.assertEqual(pft.entries, 0)

    def test_cannot_branch_below_node_copies(self) -> None:
        pft = PersistentFactorTree(8, capacity=1)
        pft.build([0] * 8)
        pft.advance({1: 1}, mode="node")
        pft.advance({2: 1}, mode="node")
        with self.assertRaises(RuntimeError):
            pft.derive_path(0, {3: 1})

    def test_to_arena_matches_active_payloads(self) -> None:
        pft = PersistentFactorTree(5)
        pft.build([0, -1, 2, 2, 4])
        pft.advance({1: 7, 4: -1}, mode="node")
        arena = CircuitArena()
        root = pft.to_arena(1, arena, lambda p: f"x{p}")
        self.assertEqual(arena.atoms(root), Counter({"x0": 1, "x7": 1, "x2": 2}))


class PlanTests(unittest.TestCase):
    def _random_graph(self, rng: random.Random, n: int, extra: int) -> list[FaceEdge]:
        edges = {}
        for v in range(1, n):  # connected
            u = rng.randrange(v)
            edges[(u, v)] = {s: (0, 1) for s in rng.sample(range(8), rng.randint(1, 4))}
        for _ in range(extra):
            a, b = sorted(rng.sample(range(n), 2))
            edges.setdefault((a, b), {s: (0, 1) for s in rng.sample(range(8), rng.randint(1, 4))})
        return [FaceEdge(a, b, c) for (a, b), c in sorted(edges.items())]

    def test_mst_is_minimum_over_all_spanning_trees(self) -> None:
        rng = random.Random(5)
        for _ in range(10):
            n = 6
            edges = self._random_graph(rng, n, 5)
            weights = edge_weights(edges, 8, "rho")
            forest = spanning_forest(n, edges, weights)
            best = float("inf")
            for subset in itertools.combinations(range(len(edges)), n - 1):
                parent = list(range(n))

                def find(x):
                    while parent[x] != x:
                        x = parent[x]
                    return x

                ok = True
                for i in subset:
                    ra, rb = find(edges[i].a), find(edges[i].b)
                    if ra == rb:
                        ok = False
                        break
                    parent[rb] = ra
                if ok:
                    best = min(best, sum(weights[i] for i in subset))
            self.assertEqual(forest.weight, best)

    def test_euler_walk_covers_and_is_two_approximate(self) -> None:
        rng = random.Random(9)
        for _ in range(20):
            n = rng.randint(2, 40)
            edges = self._random_graph(rng, n, n)
            weights = edge_weights(edges, 8, "count")
            forest = spanning_forest(n, edges, weights)
            walk = euler_walk(forest)
            self.assertTrue(all(f >= 0 for f in walk.first_visit))
            self.assertLessEqual(walk.cost, 2 * forest.weight)
            self.assertGreaterEqual(walk.cost, forest.weight)
            current = None
            for kind, face, e in walk.events:
                if kind == "step":
                    self.assertIn(current, (edges[e].a, edges[e].b))
                    self.assertIn(face, (edges[e].a, edges[e].b))
                current = face

    def test_shortcut_and_greedy_walks(self) -> None:
        rng = random.Random(10)
        for _ in range(20):
            n = rng.randint(2, 60)
            edges = self._random_graph(rng, n, 2 * n)
            weights = edge_weights(edges, 8, "count")
            forest = spanning_forest(n, edges, weights)
            euler = euler_walk(forest)
            short = shortcut_walk(forest, edges, weights)
            greedy = shortcut_walk(greedy_dfs_forest(n, edges, weights), edges, weights, heavy_last=False)
            best = best_walk(n, edges, weights)
            self.assertLessEqual(short.cost, euler.cost + 1e-9)
            self.assertLessEqual(best.cost, min(short.cost, greedy.cost) + 1e-9)
            for walk in (short, greedy, best):
                self.assertTrue(all(f >= 0 for f in walk.first_visit))
                current = None
                total = 0.0
                for kind, face, e in walk.events:
                    if kind == "step":
                        self.assertEqual({current, face}, {edges[e].a, edges[e].b})
                        total += weights[e]
                    current = face
                self.assertAlmostEqual(total, walk.cost)

    def test_boustrophedon_moves_between_neighbours(self) -> None:
        order = boustrophedon((3, 4, 2))
        self.assertEqual(sorted(order), list(range(24)))
        coords = [(i // 8, (i // 2) % 4, i % 2) for i in order]
        for p, q in zip(coords, coords[1:]):
            self.assertEqual(sum(abs(x - y) for x, y in zip(p, q)), 1)


if __name__ == "__main__":
    unittest.main()
