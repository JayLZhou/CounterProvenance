#!/usr/bin/env python3
"""Benchmark exact tree provenance layouts on theorem-shaped workloads.

The workload has M trees and k effective breakpoints per tree.  ``rho`` trees
share each breakpoint, so rho=1 is the maximum-separation family and rho=M is
the fully synchronized family.  The script compares:

* replay: one predecessor search per tree;
* IIP: one complete M-leaf provenance root per global region;
* materialized VSC: persistent balanced aggregation over tree slots;
* TPSI: static interval stabbing with fractional cascading.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterprovenance import (
    CircuitArena,
    TreeLeafInterval,
    TreeProvenanceIndex,
    VersionedSemiringIndex,
    balanced_slot_template,
)


@dataclass(frozen=True)
class Workload:
    tree_count: int
    breakpoints_per_tree: int
    rho: int
    thresholds: tuple[tuple[float, ...], ...]
    events: tuple[tuple[float, tuple[tuple[int, int], ...]], ...]

    @property
    def interval_count(self) -> int:
        return self.tree_count * (self.breakpoints_per_tree + 1)

    @property
    def region_count(self) -> int:
        return len(self.events) + 1


@dataclass
class BuiltLayout:
    arena: CircuitArena
    index: object
    leaf_roots: tuple[tuple[int, ...], ...]
    build_seconds: float
    aggregation_nodes: int
    arena_edges: int
    structural_records: int


def make_workload(tree_count: int, breakpoints_per_tree: int, rho: int) -> Workload:
    if tree_count <= 0 or breakpoints_per_tree <= 0:
        raise ValueError("tree_count and breakpoints_per_tree must be positive")
    if rho <= 0 or tree_count % rho:
        raise ValueError("rho must be a positive divisor of tree_count")

    group_count = tree_count // rho
    denominator = breakpoints_per_tree * group_count + 1
    thresholds: list[tuple[float, ...]] = []
    events: dict[float, list[tuple[int, int]]] = defaultdict(list)

    for tree in range(tree_count):
        group = tree // rho
        values = tuple(
            (level * group_count + group + 1) / denominator
            for level in range(breakpoints_per_tree)
        )
        thresholds.append(values)
        for leaf, value in enumerate(values, start=1):
            events[value].append((tree, leaf))

    ordered_events = tuple(
        (value, tuple(changes)) for value, changes in sorted(events.items())
    )
    return Workload(
        tree_count,
        breakpoints_per_tree,
        rho,
        tuple(thresholds),
        ordered_events,
    )


def make_leaf_roots(
    arena: CircuitArena,
    workload: Workload,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            arena.atom(f"t{tree}:l{leaf}")
            for leaf in range(workload.breakpoints_per_tree + 1)
        )
        for tree in range(workload.tree_count)
    )


def build_tpsi(workload: Workload) -> BuiltLayout:
    arena = CircuitArena()
    leaf_roots = make_leaf_roots(arena, workload)
    intervals: list[TreeLeafInterval] = []
    for tree, thresholds in enumerate(workload.thresholds):
        points = (0.0, *thresholds, 1.0)
        for leaf, (left, right) in enumerate(zip(points, points[1:])):
            intervals.append(
                TreeLeafInterval(tree, left, right, leaf_roots[tree][leaf])
            )

    started = perf_counter()
    index = TreeProvenanceIndex(arena, intervals, domain=(0.0, 1.0))
    elapsed = perf_counter() - started
    edges = sum(len(node.children) for node in arena.nodes)
    return BuiltLayout(
        arena=arena,
        index=index,
        leaf_roots=leaf_roots,
        build_seconds=elapsed,
        aggregation_nodes=index.stats.provenance_nodes_created,
        arena_edges=edges,
        structural_records=index.stats.storage_records,
    )


def build_vsc(workload: Workload) -> BuiltLayout:
    arena = CircuitArena()
    leaf_roots = make_leaf_roots(arena, workload)
    atom_count = sum(len(roots) for roots in leaf_roots)
    names = tuple(f"t{tree}" for tree in range(workload.tree_count))

    started = perf_counter()
    template = balanced_slot_template(arena, names)
    versions = template.new_version_builder(
        {name: leaf_roots[tree][0] for tree, name in enumerate(names)}
    )
    for _, changes in workload.events:
        versions.update(
            {names[tree]: leaf_roots[tree][leaf] for tree, leaf in changes}
        )
    starts = (0.0, *(value for value, _ in workload.events))
    index = VersionedSemiringIndex(arena)
    index.add_unit("u", starts, versions.roots)
    elapsed = perf_counter() - started

    edges = sum(len(node.children) for node in arena.nodes)
    aggregation_nodes = arena.node_count - atom_count - 2
    structural = len(template.nodes) + len(versions.roots) + arena.node_count + edges
    return BuiltLayout(
        arena,
        index,
        leaf_roots,
        elapsed,
        aggregation_nodes,
        edges,
        structural,
    )


def build_iip(workload: Workload) -> BuiltLayout:
    arena = CircuitArena()
    leaf_roots = make_leaf_roots(arena, workload)
    atom_count = sum(len(roots) for roots in leaf_roots)

    started = perf_counter()
    active = [0] * workload.tree_count
    roots = [arena.add(*(leaf_roots[tree][0] for tree in range(workload.tree_count)))]
    for _, changes in workload.events:
        for tree, leaf in changes:
            active[tree] = leaf
        roots.append(
            arena.add(
                *(leaf_roots[tree][active[tree]] for tree in range(workload.tree_count))
            )
        )
    starts = (0.0, *(value for value, _ in workload.events))
    index = VersionedSemiringIndex(arena)
    index.add_unit("u", starts, roots)
    elapsed = perf_counter() - started

    edges = sum(len(node.children) for node in arena.nodes)
    aggregation_nodes = arena.node_count - atom_count - 2
    structural = len(roots) + arena.node_count + edges
    return BuiltLayout(
        arena,
        index,
        leaf_roots,
        elapsed,
        aggregation_nodes,
        edges,
        structural,
    )


def timed_queries(
    query: Callable[[float], int],
    values: tuple[float, ...],
    repeats: int = 3,
) -> tuple[float, int]:
    best = float("inf")
    checksum = 0
    for _ in range(repeats):
        local = 0
        started = perf_counter()
        for value in values:
            local ^= query(value)
        elapsed = perf_counter() - started
        if elapsed < best:
            best = elapsed
            checksum = local
    return best / len(values), checksum


def run_case(workload: Workload, query_count: int) -> dict[str, float | int]:
    tpsi = build_tpsi(workload)
    vsc = build_vsc(workload)
    iip = build_iip(workload)
    rng = random.Random(1000 + workload.rho)
    values = tuple(rng.random() for _ in range(query_count))

    tpsi_index = tpsi.index
    vsc_index = vsc.index
    iip_index = iip.index

    for value in values[: min(100, len(values))]:
        expected = iip.arena.polynomial(iip_index.root_for("u", value))
        actual_vsc = vsc.arena.polynomial(vsc_index.root_for("u", value))
        actual_tpsi = tpsi_index.polynomial(value)
        if actual_vsc != expected or actual_tpsi != expected:
            raise AssertionError(f"Exactness mismatch at intervention {value}")

    replay_seconds, replay_checksum = timed_queries(
        lambda value: sum(
            tpsi.leaf_roots[tree][bisect_right(thresholds, value)]
            for tree, thresholds in enumerate(workload.thresholds)
        ),
        values,
    )
    iip_seconds, iip_checksum = timed_queries(
        lambda value: iip_index.root_for("u", value),
        values,
    )
    vsc_seconds, vsc_checksum = timed_queries(
        lambda value: vsc_index.root_for("u", value),
        values,
    )
    tpsi_seconds, tpsi_checksum = timed_queries(
        lambda value: sum(tpsi_index.closure_for(value).factor_roots),
        values,
    )
    # Keep all query loops observable without requiring root IDs to match
    # across their separate canonical arenas.
    if replay_checksum == iip_checksum == vsc_checksum == tpsi_checksum == -1:
        raise AssertionError("unreachable checksum")

    return {
        "rho": workload.rho,
        "regions": workload.region_count,
        "intervals": workload.interval_count,
        "iip_leaf_refs": workload.region_count * workload.tree_count,
        "vsc_agg_nodes": vsc.aggregation_nodes,
        "tpsi_agg_nodes": tpsi.aggregation_nodes,
        "tpsi_records": tpsi.structural_records,
        "iip_build_ms": iip.build_seconds * 1e3,
        "vsc_build_ms": vsc.build_seconds * 1e3,
        "tpsi_build_ms": tpsi.build_seconds * 1e3,
        "replay_us": replay_seconds * 1e6,
        "iip_us": iip_seconds * 1e6,
        "vsc_us": vsc_seconds * 1e6,
        "tpsi_us": tpsi_seconds * 1e6,
    }


def print_table(rows: list[dict[str, float | int]]) -> None:
    print("=== Structural scaling ===")
    print(
        "| rho | B | N | IIP leaf refs | VSC agg nodes | "
        "TPSI agg nodes | TPSI records |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['rho']} | {row['regions']} | {row['intervals']} | "
            f"{row['iip_leaf_refs']} | {row['vsc_agg_nodes']} | "
            f"{row['tpsi_agg_nodes']} | {row['tpsi_records']} |"
        )

    print("\n=== Build and query time ===")
    print(
        "| rho | IIP build ms | VSC build ms | TPSI build ms | "
        "replay us/q | IIP us/q | VSC us/q | TPSI us/q |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['rho']} | {row['iip_build_ms']:.3f} | "
            f"{row['vsc_build_ms']:.3f} | {row['tpsi_build_ms']:.3f} | "
            f"{row['replay_us']:.3f} | {row['iip_us']:.3f} | "
            f"{row['vsc_us']:.3f} | {row['tpsi_us']:.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--breakpoints", type=int, default=4)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument(
        "--rho",
        type=int,
        nargs="*",
        help="Breakpoint multiplicities; defaults to 1, 4, and M when valid.",
    )
    args = parser.parse_args()

    requested = args.rho or (1, 4, args.trees)
    rhos = tuple(dict.fromkeys(rho for rho in requested if args.trees % rho == 0))
    rows = [
        run_case(
            make_workload(args.trees, args.breakpoints, rho),
            args.queries,
        )
        for rho in rhos
    ]
    print(
        f"M={args.trees}, k={args.breakpoints}, queries={args.queries}; "
        "all layouts passed exact polynomial checks."
    )
    print_table(rows)


if __name__ == "__main__":
    main()
