import unittest

import numpy as np

from counterprovenance.mcpst import (
    BalancedPFTLayout,
    GridPointLocator,
    MCPSTGridIndex,
    build_spanning_plan,
    fully_duplicated_pft_records,
    grid_adjacency,
    independent_pft_nodes,
    materialize_plan,
    transitions_from_states,
)


class BalancedPFTLayoutTests(unittest.TestCase):
    def test_batch_copy_cost_counts_union_of_ancestor_paths(self) -> None:
        layout = BalancedPFTLayout(4)

        self.assertEqual(layout.build_cost, 3)
        self.assertEqual(layout.transition_cost([]), 0)
        self.assertEqual(layout.transition_cost([0]), 2)
        self.assertEqual(layout.transition_cost([0, 1]), 2)
        self.assertEqual(layout.transition_cost([1, 2]), 3)
        self.assertEqual(layout.transition_cost([0, 1, 2, 3]), 3)

    def test_non_power_of_two_layout_uses_padding(self) -> None:
        layout = BalancedPFTLayout(5)

        self.assertEqual(layout.padded_leaf_count, 8)
        self.assertEqual(layout.height, 3)
        self.assertEqual(layout.build_cost, 7)
        self.assertEqual(layout.transition_cost([4]), 3)

    def test_fully_duplicated_baseline_counts_payloads_and_roots(self) -> None:
        layout = BalancedPFTLayout(4)

        self.assertEqual(fully_duplicated_pft_records(3, layout), 24)
        self.assertEqual(
            fully_duplicated_pft_records(3, layout, payload_records_per_component=4),
            60,
        )


class MCPSTConstructionTests(unittest.TestCase):
    def test_small_grid_avoids_expensive_transition(self) -> None:
        states = np.asarray(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=np.int8,
        )
        layout = BalancedPFTLayout(4)
        transitions = transitions_from_states(states, grid_adjacency(2, 2), layout)
        plan = build_spanning_plan(len(states), transitions, layout)
        materialized = materialize_plan(states, layout, plan)

        self.assertEqual(sorted(edge.weight for edge in transitions), [2, 2, 2, 3])
        self.assertEqual(plan.total_cost, 9)
        self.assertEqual(plan.source_roots, 1)
        self.assertEqual(materialized.internal_nodes, 9)
        self.assertEqual(independent_pft_nodes(len(states), layout), 12)
        for cell, root in enumerate(materialized.roots):
            self.assertEqual(materialized.arena.payloads(root), tuple(states[cell]))

    def test_random_grid_prediction_equals_real_allocation(self) -> None:
        rng = np.random.default_rng(7)
        states = rng.integers(0, 4, size=(30, 13), dtype=np.int8)
        layout = BalancedPFTLayout(states.shape[1])
        transitions = transitions_from_states(states, grid_adjacency(6, 5), layout)
        plan = build_spanning_plan(len(states), transitions, layout)
        materialized = materialize_plan(states, layout, plan)

        self.assertEqual(materialized.internal_nodes, plan.total_cost)
        self.assertLessEqual(
            materialized.internal_nodes,
            independent_pft_nodes(len(states), layout),
        )
        for cell, root in enumerate(materialized.roots):
            self.assertEqual(materialized.arena.payloads(root), tuple(states[cell]))

    def test_exact_pft_objective_dominates_spanning_ablations(self) -> None:
        rng = np.random.default_rng(11)
        states = rng.integers(0, 3, size=(36, 16), dtype=np.int8)
        layout = BalancedPFTLayout(states.shape[1])
        transitions = transitions_from_states(states, grid_adjacency(6, 6), layout)

        exact = build_spanning_plan(len(states), transitions, layout, "pft")
        changed = build_spanning_plan(len(states), transitions, layout, "changed")
        unweighted = build_spanning_plan(len(states), transitions, layout, "unweighted")

        self.assertLessEqual(exact.total_cost, changed.total_cost)
        self.assertLessEqual(exact.total_cost, unweighted.total_cost)


class GridQueryTests(unittest.TestCase):
    def test_threshold_equality_uses_lower_cell(self) -> None:
        locator = GridPointLocator([0.25, 0.75], [0.5])

        self.assertEqual(locator.cell_id(0.25, 0.5), 0)
        self.assertEqual(locator.cell_id(0.2500001, 0.5), 1)
        self.assertEqual(locator.cell_id(0.75, 0.5000001), 4)
        self.assertEqual(locator.cell_id(0.7500001, 0.5000001), 5)

    def test_query_dtype_matches_predictor_input_quantization(self) -> None:
        threshold = 0.1683676838874817
        locator = GridPointLocator([threshold], [0.5], query_dtype=np.float32)
        rounded = float(np.float32(threshold))
        expected_x_cell = 0 if rounded <= threshold else 1

        self.assertEqual(locator.cell_id(threshold, 0.1), expected_x_cell)
        cells = locator.cell_ids(np.asarray([[threshold, 0.1]]))
        self.assertEqual(int(cells[0]), expected_x_cell)

    def test_batch_query_returns_cached_values_and_roots(self) -> None:
        locator = GridPointLocator([0.5], [0.5])
        index = MCPSTGridIndex(
            locator=locator,
            roots=(10, 11, 12, 13),
            cached_values=np.asarray([1.0, 2.0, 3.0, 4.0]),
        )
        points = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])

        values, roots = index.query_batch(points)

        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(roots, [10, 11, 12, 13])


if __name__ == "__main__":
    unittest.main()
