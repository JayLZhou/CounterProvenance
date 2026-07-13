#!/usr/bin/env python3
"""Measure the storage separation on the theorem-aligned workload.

For every requested n, this benchmark sets B=C=n and changes exactly one slot
per boundary.  It reports normalized record counts for the predicted IIP
quadratic, Materialized-VSC n-log-n, and Compact-Closure linear growth.
"""

from __future__ import annotations

import argparse
import math

from versioned_circuit_demo import (
    build_compact_closures,
    build_independent_materialization,
    build_vsc,
    make_events,
)
from counterprovenance import (
    CircuitArena,
    CircuitTemplateBuilder,
    VersionedSlotEnvironment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="64,128,256,512",
        help="comma-separated values for B=C=n",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def build_fanout_records(size: int) -> tuple[int, int]:
    materialized_arena = CircuitArena()
    materialized_gates = [
        materialized_arena.atom(f"g{version}") for version in range(size)
    ]
    materialized_builder = CircuitTemplateBuilder(materialized_arena)
    materialized_slot = materialized_builder.slot("gate")
    materialized_branches = []
    for index in range(size):
        fixed = materialized_builder.fixed(materialized_arena.atom(f"a{index}"))
        materialized_branches.append(
            materialized_builder.mul(materialized_slot, fixed)
        )
    materialized_template = materialized_builder.build(
        materialized_builder.fold_balanced("add", materialized_branches)
    )
    versions = materialized_template.new_version_builder(
        {"gate": materialized_gates[0]}
    )
    for gate in materialized_gates[1:]:
        versions.update({"gate": gate})
    materialized_records = (
        materialized_arena.node_count
        + len(materialized_template.nodes)
        + len(versions.roots)
    )

    closure_arena = CircuitArena()
    closure_gates = [closure_arena.atom(f"g{version}") for version in range(size)]
    closure_builder = CircuitTemplateBuilder(closure_arena)
    closure_slot = closure_builder.slot("gate")
    closure_branches = []
    for index in range(size):
        fixed = closure_builder.fixed(closure_arena.atom(f"a{index}"))
        closure_branches.append(closure_builder.mul(closure_slot, fixed))
    closure_template = closure_builder.build(
        closure_builder.fold_balanced("add", closure_branches)
    )
    environment = VersionedSlotEnvironment({"gate": closure_gates[0]})
    for gate in closure_gates[1:]:
        environment.append_version({"gate": gate})
    closure_records = (
        closure_arena.node_count
        + len(closure_template.nodes)
        + environment.storage_records
        + size
    )
    return materialized_records, closure_records


def main() -> None:
    args = parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("Every scaling size must be at least 2.")

    print("B=C=n, one arbitrary slot toggle per boundary")
    print()
    print(
        "| n | IIP records | IIP/n^2 | VSC records | "
        "VSC/(n log2 n) | Closure records | Closure/n |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")

    for size in sizes:
        initial, events = make_events(size, size, 1, args.seed)
        independent_nodes, _ = build_independent_materialization(
            size, initial, events
        )
        arena, index, versions, _ = build_vsc(size, initial, events)
        closure_arena, template, environment, closure_index, _ = (
            build_compact_closures(size, initial, events)
        )

        independent_records = independent_nodes + size
        vsc_records = (
            arena.node_count + len(versions.template.nodes) + index.region_count
        )
        closure_records = (
            closure_arena.node_count
            + len(template.nodes)
            + environment.storage_records
            + closure_index.region_count
        )
        if environment.change_count != size - 1:
            raise AssertionError("Every boundary should create one effective change.")

        print(
            f"| {size:,} | {independent_records:,} | "
            f"{independent_records / size**2:.3f} | {vsc_records:,} | "
            f"{vsc_records / (size * math.log2(size)):.3f} | "
            f"{closure_records:,} | {closure_records / size:.3f} |"
        )

    print()
    print("B=F=n, one changing slot with fan-out F")
    print()
    print(
        "| n | Materialized records | Materialized/n^2 | "
        "Closure records | Closure/n |"
    )
    print("|---:|---:|---:|---:|---:|")
    for size in sizes:
        materialized_records, closure_records = build_fanout_records(size)
        print(
            f"| {size:,} | {materialized_records:,} | "
            f"{materialized_records / size**2:.3f} | "
            f"{closure_records:,} | {closure_records / size:.3f} |"
        )


if __name__ == "__main__":
    main()
