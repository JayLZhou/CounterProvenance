from __future__ import annotations

import functools
import unittest
from collections import defaultdict


Factor = tuple[int, frozenset[int]]


def _runs(mask: int) -> int:
    return (mask & ~(mask << 1)).bit_count()


def _runs_by_chain(mask: int, chain_lengths: tuple[int, ...]) -> int:
    total = 0
    offset = 0
    for length in chain_lengths:
        local_mask = (mask >> offset) & ((1 << length) - 1)
        total += _runs(local_mask)
        offset += length
    return total


def _is_laminar(supports: list[int]) -> bool:
    for index, left in enumerate(supports):
        for right in supports[index + 1 :]:
            if left & right and left & ~right and right & ~left:
                return False
    return True


def _inclusion_forest(
    columns: list[tuple[int, int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for column_id, support in columns:
        if support:
            groups[support].append(column_id)

    supports = sorted(groups, key=lambda support: (support.bit_count(), support))
    children = {support: [] for support in supports}
    for support in supports:
        supersets = [
            candidate
            for candidate in supports
            if candidate != support and not support & ~candidate
        ]
        if supersets:
            parent = min(supersets, key=lambda candidate: candidate.bit_count())
            children[parent].append(support)
    return dict(groups), children


def _best_reset(
    residual: int,
    children: list[int],
    chain_lengths: tuple[int, ...],
) -> tuple[set[int], int]:
    best_key: tuple[int, int, int] | None = None
    best_children: set[int] = set()
    best_guard = residual
    for choice in range(1 << len(children)):
        selected = {
            child for index, child in enumerate(children) if choice & (1 << index)
        }
        guard = residual
        for child in selected:
            guard |= child
        key = (_runs_by_chain(guard, chain_lengths), guard.bit_count(), choice)
        if best_key is None or key < best_key:
            best_key = key
            best_children = selected
            best_guard = guard
    return best_children, best_guard


def _run_aware_lcf(
    columns: list[tuple[int, int]],
    chain_lengths: tuple[int, ...] | None = None,
) -> tuple[list[Factor], int, int, int, int]:
    if chain_lengths is None:
        row_count = max((support.bit_length() for _, support in columns), default=0)
        chain_lengths = (row_count,)

    groups, children = _inclusion_forest(columns)
    child_supports = {child for values in children.values() for child in values}
    roots = [support for support in groups if support not in child_supports]
    factors: list[Factor] = []
    atom_count = 0
    support_runs = 0
    atom_runs = 0

    def factor(support: int, inherited: frozenset[int]) -> None:
        nonlocal atom_count, support_runs, atom_runs
        bundle = inherited | frozenset(groups[support])
        child_union = 0
        for child in children[support]:
            child_union |= child
        residual = support & ~child_union

        if not residual:
            for child in children[support]:
                factor(child, bundle)
            return

        atom_count += 1
        support_runs += _runs_by_chain(support, chain_lengths)
        atom_runs += _runs_by_chain(residual, chain_lengths)
        reset_children, guard = _best_reset(
            residual, children[support], chain_lengths
        )
        factors.append((guard, bundle))
        for child in children[support]:
            factor(child, frozenset() if child in reset_children else bundle)

    for root in roots:
        factor(root, frozenset())

    total_support_runs = sum(
        _runs_by_chain(support, chain_lengths) for support in groups
    )
    return factors, atom_count, total_support_runs, support_runs, atom_runs


def _assert_exact(
    testcase: unittest.TestCase,
    supports: list[int],
    factors: list[Factor],
    row_count: int,
) -> None:
    for row in range(row_count):
        for column, support in enumerate(supports):
            multiplicity = sum(
                1
                for guard, payload in factors
                if guard & (1 << row) and column in payload
            )
            testcase.assertEqual(multiplicity, int(bool(support & (1 << row))))


def _minimum_factor_count(supports: list[int], row_count: int) -> int:
    cells = [
        (row, column)
        for column, support in enumerate(supports)
        for row in range(row_count)
        if support & (1 << row)
    ]
    cell_index = {cell: index for index, cell in enumerate(cells)}
    full = (1 << len(cells)) - 1
    rectangles: set[int] = set()

    for guard in range(1, 1 << row_count):
        eligible = [
            column
            for column, support in enumerate(supports)
            if not guard & ~support
        ]
        for payload_bits in range(1, 1 << len(eligible)):
            rectangle = 0
            for payload_index, column in enumerate(eligible):
                if not payload_bits & (1 << payload_index):
                    continue
                for row in range(row_count):
                    if guard & (1 << row):
                        rectangle |= 1 << cell_index[(row, column)]
            rectangles.add(rectangle)

    by_cell: list[list[int]] = [[] for _ in cells]
    for rectangle in rectangles:
        for index in range(len(cells)):
            if rectangle & (1 << index):
                by_cell[index].append(rectangle)

    @functools.cache
    def solve(remaining: int) -> int:
        if not remaining:
            return 0
        first = (remaining & -remaining).bit_length() - 1
        return 1 + min(
            solve(remaining ^ rectangle)
            for rectangle in by_cell[first]
            if rectangle & remaining == rectangle
        )

    return solve(full)


def _greedy_laminar_blocks(supports: list[int]) -> list[list[tuple[int, int]]]:
    distinct = sorted(set(supports))
    adjacency = {support: set() for support in distinct}
    for index, left in enumerate(distinct):
        for right in distinct[index + 1 :]:
            if left & right and left & ~right and right & ~left:
                adjacency[left].add(right)
                adjacency[right].add(left)

    colors: dict[int, int] = {}
    order = sorted(distinct, key=lambda support: (-len(adjacency[support]), support))
    for support in order:
        used = {colors[neighbor] for neighbor in adjacency[support] if neighbor in colors}
        color = 0
        while color in used:
            color += 1
        colors[support] = color

    blocks: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for column, support in enumerate(supports):
        blocks[colors[support]].append((column, support))
    return list(blocks.values())


class LCFTheoryTests(unittest.TestCase):
    def test_all_laminar_families_through_four_regions(self) -> None:
        checked = 0
        for row_count in range(1, 5):
            candidates = list(range(1, 1 << row_count))
            for family_bits in range(1, 1 << len(candidates)):
                supports = [
                    support
                    for index, support in enumerate(candidates)
                    if family_bits & (1 << index)
                ]
                if not _is_laminar(supports):
                    continue
                factors, atoms, total_runs, support_runs, atom_runs = _run_aware_lcf(
                    list(enumerate(supports))
                )
                _assert_exact(self, supports, factors, row_count)
                self.assertEqual(len(factors), atoms)
                guard_runs = sum(_runs(guard) for guard, _ in factors)
                self.assertLessEqual(guard_runs, min(support_runs, atom_runs))
                self.assertLessEqual(guard_runs, total_runs)
                checked += 1
        self.assertEqual(checked, 902)

    def test_general_three_region_approximation_against_exact_oracle(self) -> None:
        candidates = list(range(1, 1 << 3))
        for family_bits in range(1, 1 << len(candidates)):
            supports = [
                support
                for index, support in enumerate(candidates)
                if family_bits & (1 << index)
            ]
            blocks = _greedy_laminar_blocks(supports)
            factors: list[Factor] = []
            total_runs = support_runs = atom_runs = 0
            for block in blocks:
                block_factors, _, block_total, block_support, block_atom = (
                    _run_aware_lcf(block)
                )
                factors.extend(block_factors)
                total_runs += block_total
                support_runs += block_support
                atom_runs += block_atom

            _assert_exact(self, supports, factors, 3)
            optimum = _minimum_factor_count(supports, 3)
            self.assertLessEqual(len(factors), len(blocks) * optimum)
            guard_runs = sum(_runs(guard) for guard, _ in factors)
            self.assertLessEqual(guard_runs, min(support_runs, atom_runs))
            self.assertLessEqual(guard_runs, total_runs)

    def test_run_aware_reset_strictly_improves_both_endpoints(self) -> None:
        supports = [0b10111, 0b00010, 0b10000]
        factors, atoms, _, support_runs, atom_runs = _run_aware_lcf(
            list(enumerate(supports))
        )
        _assert_exact(self, supports, factors, 5)
        self.assertEqual(len(factors), atoms)
        self.assertEqual(support_runs, 4)
        self.assertEqual(atom_runs, 4)
        self.assertEqual(sum(_runs(guard) for guard, _ in factors), 3)

    def test_equal_support_columns_share_one_local_group(self) -> None:
        supports = [0b111, 0b111, 0b010]
        factors, atoms, _, _, _ = _run_aware_lcf(list(enumerate(supports)))
        _assert_exact(self, supports, factors, 3)
        self.assertEqual(len(factors), atoms)
        self.assertEqual(len(factors), _minimum_factor_count(supports, 3))

    def test_population_runs_do_not_cross_unit_boundaries(self) -> None:
        chain_lengths = (1, 1)
        support = 0b11
        self.assertEqual(_runs(support), 1)
        self.assertEqual(_runs_by_chain(support, chain_lengths), 2)

        factors, _, total_runs, support_runs, atom_runs = _run_aware_lcf(
            [(0, support)], chain_lengths
        )
        _assert_exact(self, [support], factors, 2)
        self.assertEqual(total_runs, 2)
        self.assertEqual(support_runs, 2)
        self.assertEqual(atom_runs, 2)
        self.assertEqual(
            sum(
                _runs_by_chain(guard, chain_lengths) for guard, _ in factors
            ),
            2,
        )

    def test_population_factorization_survives_arbitrary_unit_bindings(self) -> None:
        chain_lengths = (2, 2)
        supports = [0b1111, 0b0011, 0b1100, 0b1010]
        factors: list[Factor] = []
        for block in _greedy_laminar_blocks(supports):
            block_factors, _, _, _, _ = _run_aware_lcf(block, chain_lengths)
            factors.extend(block_factors)
        _assert_exact(self, supports, factors, 4)

        bindings = (
            (2.0, -3.0, 101.0, 0.5),
            (-7.0, 11.0, -13.0, 17.0),
        )
        row_to_unit = (0, 0, 1, 1)
        for row, unit in enumerate(row_to_unit):
            direct = sum(
                bindings[unit][column]
                for column, support in enumerate(supports)
                if support & (1 << row)
            )
            factored = sum(
                sum(bindings[unit][column] for column in payload)
                for guard, payload in factors
                if guard & (1 << row)
            )
            self.assertEqual(factored, direct)


if __name__ == "__main__":
    unittest.main()
