#!/usr/bin/env python3
"""Benchmark arbitrary-toggle provenance with a Versioned Semiring Circuit.

The workload intentionally violates staircase monotonicity: every boundary
toggles randomly selected fragments, so individual fragments may follow
ON -> OFF -> ON repeatedly.  We compare:

* independent per-region circuit materialization (IIP);
* event replay from the first region for each query;
* materialized VSC path-copy construction and indexed root lookup;
* compact template-plus-versioned-environment closures.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from pathlib import Path
import random
from statistics import median
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterprovenance import (
    CircuitArena,
    CircuitTemplate,
    VersionedClosureIndex,
    VersionedSemiringIndex,
    VersionedSlotEnvironment,
    balanced_slot_template,
    independent_node_occurrences,
)


def make_events(
    component_count: int,
    region_count: int,
    changes_per_boundary: int,
    seed: int,
) -> tuple[set[int], list[tuple[int, ...]]]:
    rng = random.Random(seed)
    initial = {index for index in range(component_count) if rng.random() < 0.5}
    events = [
        tuple(rng.sample(range(component_count), changes_per_boundary))
        for _ in range(region_count - 1)
    ]
    return initial, events


def build_vsc(
    component_count: int,
    initial: set[int],
    events: list[tuple[int, ...]],
) -> tuple[CircuitArena, VersionedSemiringIndex, object, float]:
    start = time.perf_counter()
    arena = CircuitArena()
    names = tuple(f"f{index}" for index in range(component_count))
    atoms = tuple(arena.atom(name) for name in names)
    template = balanced_slot_template(arena, names)
    active = set(initial)
    versions = template.new_version_builder(
        {
            name: atoms[index] if index in active else arena.zero
            for index, name in enumerate(names)
        }
    )

    for event in events:
        updates = {}
        for index in event:
            if index in active:
                active.remove(index)
                updates[names[index]] = arena.zero
            else:
                active.add(index)
                updates[names[index]] = atoms[index]
        versions.update(updates)

    index = VersionedSemiringIndex(arena)
    index.add_unit("u0", tuple(float(i) for i in range(len(events) + 1)), versions.roots)
    elapsed = time.perf_counter() - start
    return arena, index, versions, elapsed


def build_independent_materialization(
    component_count: int,
    initial: set[int],
    events: list[tuple[int, ...]],
) -> tuple[int, float]:
    start = time.perf_counter()
    active = set(initial)
    total_nodes = 0
    names = tuple(f"f{index}" for index in range(component_count))

    for region in range(len(events) + 1):
        arena = CircuitArena()
        template = balanced_slot_template(arena, names)
        slot_values = {
            name: arena.atom(name) if index in active else arena.zero
            for index, name in enumerate(names)
        }
        version = template.new_version_builder(slot_values)
        total_nodes += len(arena.reachable_nodes(version.current_root))

        if region < len(events):
            for index in events[region]:
                if index in active:
                    active.remove(index)
                else:
                    active.add(index)

    return total_nodes, time.perf_counter() - start


def build_compact_closures(
    component_count: int,
    initial: set[int],
    events: list[tuple[int, ...]],
) -> tuple[
    CircuitArena,
    CircuitTemplate,
    VersionedSlotEnvironment,
    VersionedClosureIndex,
    float,
]:
    start = time.perf_counter()
    arena = CircuitArena()
    names = tuple(f"f{index}" for index in range(component_count))
    atoms = tuple(arena.atom(name) for name in names)
    template = balanced_slot_template(arena, names)
    active = set(initial)
    environment = VersionedSlotEnvironment(
        {
            name: atoms[index] if index in active else arena.zero
            for index, name in enumerate(names)
        }
    )

    for event in events:
        updates = {}
        for index in event:
            if index in active:
                active.remove(index)
                updates[names[index]] = arena.zero
            else:
                active.add(index)
                updates[names[index]] = atoms[index]
        environment.append_version(updates)

    index = VersionedClosureIndex()
    index.add_unit(
        "u0",
        tuple(float(i) for i in range(len(events) + 1)),
        template,
        environment,
    )
    elapsed = time.perf_counter() - start
    return arena, template, environment, index, elapsed


def replay_region(initial: set[int], events: list[tuple[int, ...]], region: int) -> int:
    active = set(initial)
    for event_index in range(region):
        event = events[event_index]
        for index in event:
            if index in active:
                active.remove(index)
            else:
                active.add(index)
    return len(active)


def benchmark_queries(
    vsc_index: VersionedSemiringIndex,
    closure_index: VersionedClosureIndex,
    initial: set[int],
    events: list[tuple[int, ...]],
    query_count: int,
    seed: int,
) -> tuple[float, float, float, int, int, int]:
    rng = random.Random(seed + 1)
    queries = [rng.randrange(len(events) + 1) + 0.5 for _ in range(query_count)]

    start = time.perf_counter()
    vsc_checksum = 0
    for query in queries:
        vsc_checksum ^= vsc_index.root_for("u0", query)
    vsc_seconds = time.perf_counter() - start

    start = time.perf_counter()
    closure_checksum = 0
    for query in queries:
        closure_checksum ^= closure_index.closure_for("u0", query).version
    closure_seconds = time.perf_counter() - start

    start = time.perf_counter()
    replay_checksum = 0
    for query in queries:
        region = bisect_right(range(len(events) + 1), query) - 1
        replay_checksum += replay_region(initial, events, region)
    replay_seconds = time.perf_counter() - start
    return (
        vsc_seconds,
        closure_seconds,
        replay_seconds,
        vsc_checksum,
        closure_checksum,
        replay_checksum,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--regions", type=int, default=1000)
    parser.add_argument("--changes", type=int, default=3)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.components < 1 or args.regions < 1:
        raise SystemExit("--components and --regions must be positive")
    if not 1 <= args.changes <= args.components:
        raise SystemExit("--changes must be in [1, components]")
    if args.queries < 1 or args.trials < 1:
        raise SystemExit("--queries and --trials must be positive")

    initial, events = make_events(
        args.components, args.regions, args.changes, args.seed
    )
    vsc_build_trials = []
    closure_build_trials = []
    independent_build_trials = []
    arena = index = versions = None
    closure_arena = closure_template = environment = closure_index = None
    independent_nodes = 0
    for trial in range(args.trials):
        if trial % 3 == 0:
            arena, index, versions, elapsed = build_vsc(
                args.components, initial, events
            )
            vsc_build_trials.append(elapsed)
            closure_arena, closure_template, environment, closure_index, elapsed = (
                build_compact_closures(args.components, initial, events)
            )
            closure_build_trials.append(elapsed)
            independent_nodes, elapsed = build_independent_materialization(
                args.components, initial, events
            )
            independent_build_trials.append(elapsed)
        elif trial % 3 == 1:
            closure_arena, closure_template, environment, closure_index, elapsed = (
                build_compact_closures(args.components, initial, events)
            )
            closure_build_trials.append(elapsed)
            independent_nodes, elapsed = build_independent_materialization(
                args.components, initial, events
            )
            independent_build_trials.append(elapsed)
            arena, index, versions, elapsed = build_vsc(
                args.components, initial, events
            )
            vsc_build_trials.append(elapsed)
        else:
            independent_nodes, elapsed = build_independent_materialization(
                args.components, initial, events
            )
            independent_build_trials.append(elapsed)
            arena, index, versions, elapsed = build_vsc(
                args.components, initial, events
            )
            vsc_build_trials.append(elapsed)
            closure_arena, closure_template, environment, closure_index, elapsed = (
                build_compact_closures(args.components, initial, events)
            )
            closure_build_trials.append(elapsed)

    assert arena is not None and index is not None and versions is not None
    assert closure_arena is not None and closure_template is not None
    assert environment is not None and closure_index is not None
    vsc_build = median(vsc_build_trials)
    closure_build = median(closure_build_trials)
    independent_build = median(independent_build_trials)

    independent_records = independent_nodes + args.regions
    vsc_records = arena.node_count + len(versions.template.nodes) + index.region_count
    closure_records = (
        closure_arena.node_count
        + len(closure_template.nodes)
        + environment.storage_records
        + closure_index.region_count
    )
    reachable_occurrences = independent_node_occurrences(arena, versions.roots)
    vsc_query_trials = []
    closure_query_trials = []
    replay_query_trials = []
    vsc_checksum = closure_checksum = replay_checksum = 0
    for _ in range(args.trials):
        (
            vsc_query,
            closure_query,
            replay_query,
            vsc_checksum,
            closure_checksum,
            replay_checksum,
        ) = benchmark_queries(
            index, closure_index, initial, events, args.queries, args.seed
        )
        vsc_query_trials.append(vsc_query)
        closure_query_trials.append(closure_query)
        replay_query_trials.append(replay_query)
    vsc_query = median(vsc_query_trials)
    closure_query = median(closure_query_trials)
    replay_query = median(replay_query_trials)

    changed_slots = sum(stats.changed_slots for stats in versions.stats[1:])
    affected_nodes = sum(stats.affected_template_nodes for stats in versions.stats[1:])
    new_nodes = sum(stats.new_arena_nodes for stats in versions.stats[1:])

    print("=== TESSERA Versioned Provenance Closure demo ===")
    print(
        f"components={args.components:,}, regions={args.regions:,}, "
        f"changes/boundary={args.changes}, queries={args.queries:,}, "
        f"trials={args.trials} (median)"
    )
    print("workload: arbitrary ON/OFF toggles (not staircase)")
    print()
    print("Construction and storage")
    print(f"IIP independent build:       {independent_build:.6f}s")
    print(f"VSC materialized build:      {vsc_build:.6f}s")
    print(f"Closure compact build:       {closure_build:.6f}s")
    print(f"IIP independent records:     {independent_records:,}")
    print(f"IIP occurrences on VSC DAG:  {reachable_occurrences:,}")
    print(f"VSC materialized records:    {vsc_records:,}")
    print(f"Closure compact records:     {closure_records:,}")
    print(
        f"VSC compression vs IIP:      "
        f"{independent_records / max(vsc_records, 1):.2f}x"
    )
    print(
        f"Closure compression vs IIP:  "
        f"{independent_records / max(closure_records, 1):.2f}x"
    )
    print(f"changed selector slots:      {changed_slots:,}")
    print(f"closure change records:      {environment.change_count:,}")
    print(f"affected template nodes:     {affected_nodes:,}")
    print(f"new hash-consed nodes:       {new_nodes:,}")
    print()
    print("Provenance-root queries")
    print(f"event replay:                {replay_query:.6f}s")
    print(f"VSC indexed root lookup:     {vsc_query:.6f}s")
    print(f"Closure indexed handle:      {closure_query:.6f}s")
    print(
        f"VSC speedup vs event replay: {replay_query / max(vsc_query, 1e-15):.2f}x"
    )
    print(
        f"Closure vs event replay:     "
        f"{replay_query / max(closure_query, 1e-15):.2f}x"
    )
    print(
        f"checksums: root={vsc_checksum}, closure={closure_checksum}, "
        f"replay={replay_checksum}"
    )
    print()
    print("IIP, VSC, and Closure all provide logarithmic region lookup. VSC and")
    print("Closure trade materialized output speed for different storage bounds. The")
    print("query speedup above is against event replay, not IIP root lookup.")


if __name__ == "__main__":
    main()
