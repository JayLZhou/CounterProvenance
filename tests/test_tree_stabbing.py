from __future__ import annotations

from collections import Counter
import math
import random
import unittest

from counterprovenance import (
    CircuitArena,
    TreeLeafInterval,
    TreeProvenanceIndex,
)


def brute_polynomial(
    arena: CircuitArena,
    intervals: list[TreeLeafInterval],
    intervention: float,
    anchor: int | None = None,
) -> Counter[tuple[str, ...]]:
    result: Counter[tuple[str, ...]] = Counter()
    if anchor is not None:
        result.update(arena.polynomial(anchor))
    for record in intervals:
        if record.left <= intervention < record.right:
            result.update(arena.polynomial(record.root))
    return result


class TreeProvenanceIndexTests(unittest.TestCase):
    def test_exact_at_leaf_and_global_boundaries(self) -> None:
        arena = CircuitArena()
        roots = {name: arena.atom(name) for name in ("a0", "a1", "a2", "b0", "b1", "b2")}
        noise = arena.atom("noise")
        intervals = [
            TreeLeafInterval("a", 0.0, 2.0, roots["a0"]),
            TreeLeafInterval("a", 2.0, 5.0, roots["a1"]),
            TreeLeafInterval("a", 5.0, 10.0, roots["a2"]),
            TreeLeafInterval("b", 0.0, 1.0, roots["b0"]),
            TreeLeafInterval("b", 1.0, 7.0, roots["b1"]),
            TreeLeafInterval("b", 7.0, 10.0, roots["b2"]),
        ]
        index = TreeProvenanceIndex(
            arena,
            intervals,
            domain=(0.0, 10.0),
            anchor_root=noise,
        )

        for intervention in (0.0, 0.5, 1.0, 1.999, 2.0, 5.0, 6.999, 7.0, 9.999):
            self.assertEqual(
                index.polynomial(intervention),
                brute_polynomial(arena, intervals, intervention, noise),
            )

        self.assertEqual(index.stats.tree_count, 2)
        self.assertEqual(index.stats.interval_count, 6)
        self.assertEqual(index.stats.breakpoint_count, 4)
        with self.assertRaises(ValueError):
            index.polynomial(10.0)

    def test_preserves_semiring_multiplicity_without_subtraction(self) -> None:
        arena = CircuitArena()
        shared = arena.atom("shared")
        intervals = [
            TreeLeafInterval("left", 0.0, 1.0, shared),
            TreeLeafInterval("right", 0.0, 1.0, shared),
        ]
        index = TreeProvenanceIndex(arena, intervals, domain=(0.0, 1.0))

        self.assertEqual(index.polynomial(0.5), Counter({("shared",): 2}))
        self.assertEqual(index.evaluate(0.5, {"shared": 3.0}), 6.0)

    def test_unbounded_intervention_domain(self) -> None:
        arena = CircuitArena()
        negative = arena.atom("negative")
        nonnegative = arena.atom("nonnegative")
        global_root = arena.atom("global")
        intervals = [
            TreeLeafInterval("split", -math.inf, 0.0, negative),
            TreeLeafInterval("split", 0.0, math.inf, nonnegative),
            TreeLeafInterval("constant", -math.inf, math.inf, global_root),
        ]
        index = TreeProvenanceIndex(
            arena,
            intervals,
            domain=(-math.inf, math.inf),
        )

        self.assertEqual(
            index.polynomial(-1e100),
            Counter({("negative",): 1, ("global",): 1}),
        )
        self.assertEqual(
            index.polynomial(0.0),
            Counter({("nonnegative",): 1, ("global",): 1}),
        )
        self.assertEqual(
            index.polynomial(1e100),
            Counter({("nonnegative",): 1, ("global",): 1}),
        )

    def test_random_partitions_match_brute_force_with_one_binary_search(self) -> None:
        rng = random.Random(41)
        arena = CircuitArena()
        intervals: list[TreeLeafInterval] = []
        breakpoints: list[float] = []

        for tree in range(12):
            threshold_count = rng.randint(1, 8)
            thresholds = sorted(rng.sample(range(1, 1000), threshold_count))
            points = [0.0, *(value / 1000.0 for value in thresholds), 1.0]
            breakpoints.extend(points[1:-1])
            for leaf, (left, right) in enumerate(zip(points, points[1:])):
                root = arena.atom(f"t{tree}:l{leaf}")
                intervals.append(TreeLeafInterval(tree, left, right, root))

        index = TreeProvenanceIndex(arena, intervals, domain=(0.0, 1.0))
        queries = [rng.random() for _ in range(300)]
        for breakpoint in breakpoints:
            queries.extend(
                (
                    math.nextafter(breakpoint, -math.inf),
                    breakpoint,
                    math.nextafter(breakpoint, math.inf),
                )
            )

        height_bound = math.ceil(math.log2(len(intervals))) + 1
        for intervention in queries:
            closure, stats = index.query_with_stats(intervention)
            self.assertEqual(
                closure.polynomial(),
                brute_polynomial(arena, intervals, intervention),
            )
            self.assertEqual(stats.root_binary_searches, 1)
            self.assertLessEqual(stats.nodes_visited, height_bound)
            self.assertLessEqual(stats.cascade_corrections, stats.nodes_visited - 1)
            self.assertLessEqual(stats.factor_count, stats.nodes_visited)

    def test_partition_validation_rejects_gaps_and_overlaps(self) -> None:
        arena = CircuitArena()
        a = arena.atom("a")
        b = arena.atom("b")

        with self.assertRaisesRegex(ValueError, "gap"):
            TreeProvenanceIndex(
                arena,
                [
                    TreeLeafInterval("t", 0.0, 0.4, a),
                    TreeLeafInterval("t", 0.5, 1.0, b),
                ],
                domain=(0.0, 1.0),
            )

        with self.assertRaisesRegex(ValueError, "overlapping"):
            TreeProvenanceIndex(
                arena,
                [
                    TreeLeafInterval("t", 0.0, 0.6, a),
                    TreeLeafInterval("t", 0.5, 1.0, b),
                ],
                domain=(0.0, 1.0),
            )

    def test_static_index_records_and_circuit_nodes_scale_linearly(self) -> None:
        def build(tree_count: int) -> TreeProvenanceIndex:
            arena = CircuitArena()
            intervals: list[TreeLeafInterval] = []
            for tree in range(tree_count):
                threshold = (tree + 1) / (tree_count + 1)
                intervals.extend(
                    (
                        TreeLeafInterval(
                            tree,
                            0.0,
                            threshold,
                            arena.atom(f"t{tree}:left"),
                        ),
                        TreeLeafInterval(
                            tree,
                            threshold,
                            1.0,
                            arena.atom(f"t{tree}:right"),
                        ),
                    )
                )
            return TreeProvenanceIndex(arena, intervals, domain=(0.0, 1.0))

        small = build(64)
        large = build(128)

        self.assertLessEqual(
            small.stats.provenance_nodes_created,
            2 * small.stats.interval_count,
        )
        self.assertLessEqual(
            large.stats.provenance_nodes_created,
            2 * large.stats.interval_count,
        )
        self.assertLess(
            large.stats.storage_records / small.stats.storage_records,
            2.6,
        )

        independent_leaf_references = 128 * (128 + 1)
        self.assertGreater(
            independent_leaf_references,
            large.stats.storage_records,
        )


if __name__ == "__main__":
    unittest.main()
