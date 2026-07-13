from __future__ import annotations

from collections import Counter
import math
import random
import unittest

from counterprovenance import (
    CircuitArena,
    CircuitTemplateBuilder,
    VersionedSemiringIndex,
    balanced_slot_template,
    independent_node_occurrences,
)


class CircuitArenaTests(unittest.TestCase):
    def test_free_semiring_preserves_addition_multiplicity_and_products(self) -> None:
        arena = CircuitArena()
        a = arena.atom("a")
        b = arena.atom("b")
        c = arena.atom("c")

        root = arena.mul(a, arena.add(b, c))
        self.assertEqual(
            arena.polynomial(root),
            Counter({("a", "b"): 1, ("a", "c"): 1}),
        )
        self.assertEqual(arena.polynomial(arena.add(a, a)), Counter({("a",): 2}))
        self.assertEqual(arena.evaluate(root, {"a": 2, "b": 3, "c": 5}), 16.0)

    def test_switch_is_exact_under_selector_valuation(self) -> None:
        arena = CircuitArena()
        left = arena.atom("left")
        right = arena.atom("right")
        root = arena.switch("gate", left, right)

        self.assertEqual(arena.polynomial(root, {"gate": True}), Counter({("left",): 1}))
        self.assertEqual(arena.polynomial(root, {"gate": False}), Counter({("right",): 1}))
        self.assertEqual(
            arena.evaluate(root, {"left": 7, "right": 11}, {"gate": False}),
            11.0,
        )


class VersionedSemiringCircuitTests(unittest.TestCase):
    def test_arbitrary_toggle_sequence_matches_independent_materialization(self) -> None:
        arena = CircuitArena()
        names = ("f0", "f1", "f2")
        atoms = {name: arena.atom(name) for name in names}
        template = balanced_slot_template(arena, names)

        states = (
            frozenset(("f0",)),
            frozenset(("f0", "f1")),
            frozenset(("f1", "f2")),
            frozenset(("f0", "f2")),
            frozenset(("f1",)),
            frozenset(("f0", "f1")),
        )
        version_builder = template.new_version_builder(
            {name: atoms[name] if name in states[0] else arena.zero for name in names}
        )
        for state in states[1:]:
            version_builder.update(
                {name: atoms[name] if name in state else arena.zero for name in names}
            )

        index = VersionedSemiringIndex(arena)
        index.add_unit("u0", tuple(float(i) for i in range(len(states))), version_builder.roots)

        for region, state in enumerate(states):
            expected = Counter({(name,): 1 for name in state})
            self.assertEqual(index.polynomial("u0", region + 0.5), expected)

        # Regions are left-closed: an exact breakpoint belongs to the new version.
        self.assertEqual(index.polynomial("u0", 2.0), Counter({("f1",): 1, ("f2",): 1}))
        self.assertEqual(index.evaluate("u0", 3.25, {"f0": 2, "f1": 3, "f2": 5}), 7.0)

        # The same non-adjacent state is globally hash-consed to the same root.
        self.assertEqual(version_builder.roots[1], version_builder.roots[5])

        # f0 follows ON -> OFF -> ON, so this is not a staircase sequence.
        self.assertIn("f0", states[1])
        self.assertNotIn("f0", states[2])
        self.assertIn("f0", states[3])

    def test_mixed_add_and_multiply_template_updates_exactly(self) -> None:
        arena = CircuitArena()
        atoms = {name: arena.atom(name) for name in ("a", "b", "c", "d")}
        template_builder = CircuitTemplateBuilder(arena)
        slots = {name: template_builder.slot(name) for name in atoms}
        left = template_builder.add(slots["a"], slots["b"])
        right = template_builder.add(slots["c"], slots["d"])
        root = template_builder.mul(left, right)
        template = template_builder.build(root)

        versions = template.new_version_builder(
            {"a": atoms["a"], "b": arena.zero, "c": atoms["c"], "d": atoms["d"]}
        )
        self.assertEqual(
            arena.polynomial(versions.current_root),
            Counter({("a", "c"): 1, ("a", "d"): 1}),
        )

        versions.update({"b": atoms["b"], "c": arena.zero})
        self.assertEqual(
            arena.polynomial(versions.current_root),
            Counter({("a", "d"): 1, ("b", "d"): 1}),
        )
        self.assertEqual(
            arena.evaluate(versions.current_root, {"a": 2, "b": 3, "c": 5, "d": 7}),
            35.0,
        )

    def test_relu_style_multiplicative_gates_can_turn_off_and_back_on(self) -> None:
        arena = CircuitArena()
        template_builder = CircuitTemplateBuilder(arena)

        gate_a = template_builder.slot("gate_a")
        gate_b = template_builder.slot("gate_b")
        gate_out = template_builder.slot("gate_out")
        weight_a = template_builder.fixed(arena.atom("w_a"))
        weight_b = template_builder.fixed(arena.atom("w_b"))
        weight_out = template_builder.fixed(arena.atom("w_out"))
        source = template_builder.fixed(arena.atom("x"))

        path_a = template_builder.mul(gate_a, weight_a)
        path_a = template_builder.mul(path_a, source)
        path_b = template_builder.mul(gate_b, weight_b)
        path_b = template_builder.mul(path_b, source)
        hidden = template_builder.add(path_a, path_b)
        output = template_builder.mul(gate_out, hidden)
        output = template_builder.mul(weight_out, output)
        template = template_builder.build(output)

        gate_atoms = {
            "gate_a": arena.atom("relu_a"),
            "gate_b": arena.atom("relu_b"),
            "gate_out": arena.atom("relu_out"),
        }
        versions = template.new_version_builder(
            {
                "gate_a": gate_atoms["gate_a"],
                "gate_b": arena.zero,
                "gate_out": gate_atoms["gate_out"],
            }
        )
        first_root = versions.current_root
        versions.update({"gate_a": arena.zero, "gate_b": gate_atoms["gate_b"]})
        versions.update({"gate_a": gate_atoms["gate_a"]})
        both_paths_root = versions.current_root
        versions.update({"gate_out": arena.zero})
        self.assertEqual(versions.current_root, arena.zero)
        versions.update({"gate_out": gate_atoms["gate_out"]})

        self.assertEqual(versions.current_root, both_paths_root)
        self.assertNotEqual(first_root, both_paths_root)
        self.assertEqual(
            arena.polynomial(both_paths_root),
            Counter(
                {
                    ("relu_a", "relu_out", "w_a", "w_out", "x"): 1,
                    ("relu_b", "relu_out", "w_b", "w_out", "x"): 1,
                }
            ),
        )
        self.assertEqual(
            arena.evaluate(
                both_paths_root,
                {
                    "relu_a": 1,
                    "relu_b": 1,
                    "relu_out": 1,
                    "w_a": 2,
                    "w_b": 3,
                    "w_out": 5,
                    "x": 7,
                },
            ),
            175.0,
        )

    def test_cross_unit_structure_shares_roots_but_not_bindings(self) -> None:
        arena = CircuitArena()
        a = arena.atom("a")
        b = arena.atom("b")
        template = balanced_slot_template(arena, ("left", "right"))

        first = template.new_version_builder({"left": a, "right": arena.zero})
        first.update({"right": b})
        second = template.new_version_builder({"left": a, "right": arena.zero})
        second.update({"right": b})

        self.assertEqual(first.roots, second.roots)

        index = VersionedSemiringIndex(arena)
        index.add_unit("u1", (0.0, 1.0), first.roots)
        index.add_unit("u2", (0.0, 1.0), second.roots)
        self.assertEqual(index.evaluate("u1", 1.5, {"a": 2, "b": 3}), 5.0)
        self.assertEqual(index.evaluate("u2", 1.5, {"a": 20, "b": 30}), 50.0)

    def test_noop_version_reuses_root_without_new_nodes(self) -> None:
        arena = CircuitArena()
        atom = arena.atom("x")
        template = balanced_slot_template(arena, ("x",))
        versions = template.new_version_builder({"x": atom})
        node_count = arena.node_count
        root = versions.update({"x": atom})

        self.assertEqual(root, versions.roots[0])
        self.assertEqual(arena.node_count, node_count)
        self.assertEqual(versions.stats[-1].changed_slots, 0)
        self.assertEqual(versions.stats[-1].new_arena_nodes, 0)

    def test_affected_work_includes_changed_slot_for_single_leaf_template(self) -> None:
        arena = CircuitArena()
        atom = arena.atom("x")
        template = balanced_slot_template(arena, ("x",))
        versions = template.new_version_builder({"x": arena.zero})

        versions.update({"x": atom})

        self.assertEqual(versions.current_root, atom)
        self.assertEqual(versions.stats[-1].changed_slots, 1)
        self.assertEqual(versions.stats[-1].affected_template_nodes, 1)
        self.assertEqual(versions.stats[-1].new_arena_nodes, 0)

    def test_balanced_batch_updates_follow_path_union_bound(self) -> None:
        component_count = 64
        names = tuple(f"f{index}" for index in range(component_count))

        for changed_count in (1, 2, 4, 8, 16, 32, 64):
            with self.subTest(changed_count=changed_count):
                arena = CircuitArena()
                atoms = {name: arena.atom(name) for name in names}
                template = balanced_slot_template(arena, names)
                versions = template.new_version_builder(
                    {name: arena.zero for name in names}
                )
                versions.update(
                    {name: atoms[name] for name in names[:changed_count]}
                )

                stats = versions.stats[-1]
                path_union_bound = changed_count * (
                    math.ceil(math.log2(component_count / changed_count)) + 2
                )
                self.assertLessEqual(
                    stats.affected_template_nodes,
                    path_union_bound,
                )
                self.assertLessEqual(
                    stats.new_arena_nodes,
                    stats.affected_template_nodes,
                )

    def test_non_staircase_versions_have_change_sensitive_space(self) -> None:
        component_count = 64
        region_count = 257
        rng = random.Random(17)

        arena = CircuitArena()
        names = tuple(f"f{index}" for index in range(component_count))
        atoms = {name: arena.atom(name) for name in names}
        template = balanced_slot_template(arena, names)
        active = {name for index, name in enumerate(names) if index % 2 == 0}
        versions = template.new_version_builder(
            {name: atoms[name] if name in active else arena.zero for name in names}
        )

        for _ in range(region_count - 1):
            changed = rng.sample(names, 2)
            updates = {}
            for name in changed:
                if name in active:
                    active.remove(name)
                    updates[name] = arena.zero
                else:
                    active.add(name)
                    updates[name] = atoms[name]
            versions.update(updates)

        independent = independent_node_occurrences(arena, versions.roots)
        persistent = arena.node_count + len(versions.roots)

        self.assertLess(persistent, independent // 3)
        max_single_paths = 2 * (math.ceil(math.log2(component_count)) + 1)
        for stats in versions.stats[1:]:
            self.assertEqual(stats.changed_slots, 2)
            self.assertLessEqual(stats.affected_template_nodes, max_single_paths)
            self.assertLessEqual(stats.new_arena_nodes, stats.affected_template_nodes)


if __name__ == "__main__":
    unittest.main()
