#!/usr/bin/env python3
"""Large-scale packed-byte experiment for general-position 2D stumps.

This benchmark uses an implicit snake spanning path over the cell grid, so it
does not materialize the K-by-M selector matrix or Python graph objects. It
still constructs every persistent PFT internal node and every cell root in
packed NumPy arrays. The workload has one changed component per transition.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit


REPO_ROOT = Path(__file__).resolve().parents[2]


@njit(cache=True)
def _path_copy(
    children: np.ndarray,
    cursor: int,
    root: int,
    component: int,
    new_state: int,
    height: int,
    path_nodes: np.ndarray,
    directions: np.ndarray,
) -> tuple[int, int]:
    old_ref = root
    for depth in range(height):
        path_nodes[depth] = old_ref
        direction = (component >> (height - depth - 1)) & 1
        directions[depth] = direction
        old_ref = children[old_ref, direction]

    new_ref = -(2 * component + new_state + 1)
    for depth in range(height - 1, -1, -1):
        old_node = path_nodes[depth]
        if directions[depth] == 0:
            children[cursor, 0] = new_ref
            children[cursor, 1] = children[old_node, 1]
        else:
            children[cursor, 0] = children[old_node, 0]
            children[cursor, 1] = new_ref
        new_ref = cursor
        cursor += 1
    return new_ref, cursor


@njit(cache=True)
def build_packed_general_position(
    component_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize all PFT nodes and cell roots for a power-of-two M."""

    height = int(math.log2(component_count))
    thresholds_per_axis = component_count // 2
    side = thresholds_per_axis + 1
    cell_count = side * side
    node_count = component_count - 1 + (cell_count - 1) * height
    children = np.empty((node_count, 2), dtype=np.int32)
    roots = np.empty(cell_count, dtype=np.int32)

    current = np.empty(component_count, dtype=np.int32)
    for component in range(component_count):
        current[component] = -(2 * component + 1)
    cursor = 0
    width = component_count
    while width > 1:
        next_level = np.empty(width // 2, dtype=np.int32)
        for index in range(width // 2):
            children[cursor, 0] = current[2 * index]
            children[cursor, 1] = current[2 * index + 1]
            next_level[index] = cursor
            cursor += 1
        current = next_level
        width //= 2

    root = current[0]
    x_index = 0
    roots[0] = root
    path_nodes = np.empty(height, dtype=np.int32)
    directions = np.empty(height, dtype=np.int8)

    for y_index in range(side):
        if y_index % 2 == 0:
            for boundary in range(thresholds_per_axis):
                root, cursor = _path_copy(
                    children,
                    cursor,
                    root,
                    boundary,
                    1,
                    height,
                    path_nodes,
                    directions,
                )
                x_index = boundary + 1
                roots[y_index * side + x_index] = root
        else:
            for boundary in range(thresholds_per_axis - 1, -1, -1):
                root, cursor = _path_copy(
                    children,
                    cursor,
                    root,
                    boundary,
                    0,
                    height,
                    path_nodes,
                    directions,
                )
                x_index = boundary
                roots[y_index * side + x_index] = root

        if y_index + 1 < side:
            root, cursor = _path_copy(
                children,
                cursor,
                root,
                thresholds_per_axis + y_index,
                1,
                height,
                path_nodes,
                directions,
            )
            roots[(y_index + 1) * side + x_index] = root

    if cursor != node_count:
        raise AssertionError("Packed construction did not fill the node arena.")
    return children, roots


def payload_at(children: np.ndarray, root: int, component: int, height: int) -> int:
    ref = int(root)
    for depth in range(height):
        direction = (component >> (height - depth - 1)) & 1
        ref = int(children[ref, direction])
    return ref


def verify_samples(
    children: np.ndarray,
    roots: np.ndarray,
    component_count: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    half = component_count // 2
    side = half + 1
    height = component_count.bit_length() - 1
    sample_cells = rng.integers(0, len(roots), size=64)
    sample_components = rng.integers(0, component_count, size=64)
    for cell in sample_cells:
        x_index = int(cell % side)
        y_index = int(cell // side)
        root = int(roots[cell])
        for component in sample_components:
            if component < half:
                state = int(x_index > component)
            else:
                state = int(y_index > component - half)
            expected = -(2 * int(component) + state + 1)
            actual = payload_at(children, root, int(component), height)
            if actual != expected:
                raise AssertionError(
                    f"Cell {cell}, component {component}: {actual} != {expected}."
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="256,512,1024,2048")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument(
        "--persist-size",
        type=int,
        default=0,
        help="Also persist one packed index size as memory-mappable .npy arrays.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def persist_packed_index(component_count: int, artifact_dir: Path, seed: int) -> Path:
    """Persist one inspectable packed index and return its directory."""

    if component_count < 2 or component_count & (component_count - 1):
        raise ValueError("persist-size must be a power of two >= 2.")
    children, roots = build_packed_general_position(component_count)
    verify_samples(children, roots, component_count, seed + component_count)
    cell_count = len(roots)
    cached_values = np.arange(cell_count, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, component_count + 2, dtype=np.float64)[1:-1]
    payload_table = np.empty((2 * component_count, 2), dtype=np.int32)
    payload_table[:, 0] = np.repeat(np.arange(component_count, dtype=np.int32), 2)
    payload_table[:, 1] = np.tile(np.asarray([0, 1], dtype=np.int32), component_count)

    destination = artifact_dir / f"packed_m{component_count}"
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {
        "children": children,
        "roots": roots,
        "cached_values": cached_values,
        "thresholds": thresholds,
        "payload_table": payload_table,
    }
    files: dict[str, dict[str, object]] = {}
    for name, values in arrays.items():
        path = destination / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        files[name] = {
            "file": path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "array_bytes": values.nbytes,
            "file_bytes": path.stat().st_size,
        }
    metadata = {
        "format": "CounterProvenance packed general-position PFT v1",
        "M": component_count,
        "K_cells": cell_count,
        "height": component_count.bit_length() - 1,
        "note": (
            "cached_values are deterministic cell IDs for structural inspection, "
            "not predictions from a fitted model"
        ),
        "load": "numpy.load(path, mmap_mode='r', allow_pickle=False)",
        "files": files,
        "total_array_bytes": sum(values.nbytes for values in arrays.values()),
        "total_file_bytes": sum(int(record["file_bytes"]) for record in files.values()),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return destination


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("Unreachable byte unit.")


def run_size(component_count: int, seed: int) -> dict[str, object]:
    if component_count < 2 or component_count & (component_count - 1):
        raise ValueError("Packed scaling sizes must be powers of two >= 2.")
    height = component_count.bit_length() - 1
    side = component_count // 2 + 1
    cell_count = side * side

    started = time.perf_counter()
    children, roots = build_packed_general_position(component_count)
    build_seconds = time.perf_counter() - started
    verify_samples(children, roots, component_count, seed + component_count)

    cached_values = np.arange(cell_count, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, component_count + 2, dtype=np.float64)[1:-1]
    payload_table = np.empty((2 * component_count, 2), dtype=np.int32)
    payload_table[:, 0] = np.repeat(np.arange(component_count, dtype=np.int32), 2)
    payload_table[:, 1] = np.tile(np.asarray([0, 1], dtype=np.int32), component_count)

    ours_bytes = (
        children.nbytes
        + roots.nbytes
        + cached_values.nbytes
        + thresholds.nbytes
        + payload_table.nbytes
    )
    common_bytes = cached_values.nbytes + thresholds.nbytes
    independent_internal_bytes = (
        cell_count * (component_count - 1) * 2 * np.dtype(np.int32).itemsize
    )
    strong_iip_bytes = common_bytes + payload_table.nbytes + independent_internal_bytes
    active_payload_bytes = (
        cell_count * component_count * 2 * np.dtype(np.int32).itemsize
    )
    fully_duplicated_bytes = (
        common_bytes + independent_internal_bytes + active_payload_bytes
    )
    row = {
        "M": component_count,
        "K_cells": cell_count,
        "height": height,
        "packed_internal_nodes": len(children),
        "packed_root_count": len(roots),
        "mcpst_bytes": ours_bytes,
        "strong_iip_bytes": strong_iip_bytes,
        "fully_duplicated_bytes_q1": fully_duplicated_bytes,
        "compression_vs_strong_iip_x": strong_iip_bytes / ours_bytes,
        "compression_vs_full_x": fully_duplicated_bytes / ours_bytes,
        "build_s": build_seconds,
        "verified_payload_lookups": 64 * 64,
    }
    del children, roots, cached_values, thresholds, payload_table
    gc.collect()
    return row


def write_outputs(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "packed_scaling.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = []
    for row in rows:
        table.append(
            "| "
            f"{row['M']:,} | {row['K_cells']:,} | "
            f"{human_bytes(int(row['mcpst_bytes']))} | "
            f"{human_bytes(int(row['strong_iip_bytes']))} | "
            f"{human_bytes(int(row['fully_duplicated_bytes_q1']))} | "
            f"{row['compression_vs_strong_iip_x']:.2f}x | "
            f"{row['compression_vs_full_x']:.2f}x | "
            f"{row['build_s']:.3f} |"
        )
    summary = [
        "# Packed MCPST Scaling",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- Numba: `{numba.__version__}`",
        "- MCPST arrays are actually allocated and populated.",
        "- Baseline bytes are exact counts under the same packed schema; very large"
        " baselines are not allocated to avoid exhausting machine memory.",
        "- Schema: each internal node has two int32 refs; each MCPST root is int32;"
        " cached values and thresholds are float64; each q=1 payload is two int32 fields.",
        "- Strong IIP is allowed to derive each independent root offset implicitly and"
        " therefore is not charged a root directory.",
        "- This is a general-position specialized builder: its snake path is already an"
        " optimal spanning tree because every transition has the same one-component cost."
        " Build time excludes cell discovery, graph construction, and generic MST planning.",
        "",
        "| M | Cells K | MCPST actual | Strong IIP | Full duplicate q=1 | vs IIP | vs full | Build (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table,
        "",
        "Every row verifies 4,096 random (cell, component) payload lookups against the"
        " closed-form stump state. The packed builder excludes one-time Numba JIT time.",
        "",
    ]
    (output_dir / "PACKED_RESULTS.md").write_text("\n".join(summary), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency.
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    figure.patch.set_facecolor("white")
    sizes = [int(row["M"]) for row in rows]
    axes[0].plot(
        sizes,
        [int(row["mcpst_bytes"]) for row in rows],
        marker="s",
        label="MCPST actual",
    )
    axes[0].plot(
        sizes,
        [int(row["strong_iip_bytes"]) for row in rows],
        marker="o",
        label="Strong IIP",
    )
    axes[0].plot(
        sizes,
        [int(row["fully_duplicated_bytes_q1"]) for row in rows],
        marker="^",
        label="Full duplicate",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels([str(size) for size in sizes])
    axes[0].set_xlabel("Component trees M")
    axes[0].set_ylabel("Packed bytes (log scale)")
    axes[0].set_title("Packed memory footprint")
    axes[0].legend()

    axes[1].plot(
        sizes,
        [float(row["compression_vs_strong_iip_x"]) for row in rows],
        marker="o",
        label="vs strong IIP",
    )
    axes[1].plot(
        sizes,
        [float(row["compression_vs_full_x"]) for row in rows],
        marker="^",
        label="vs full duplicate",
    )
    axes[1].axhline(100.0, color="black", linestyle="--", label="100x")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels([str(size) for size in sizes])
    axes[1].set_xlabel("Component trees M")
    axes[1].set_ylabel("Compression ratio (log scale)")
    axes[1].set_title("Compression threshold")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "packed_results.png", dpi=180, facecolor="white")
    figure.savefig(output_dir / "packed_results.pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    # Exclude one-time compilation from the reported construction latency.
    build_packed_general_position(2)
    rows = []
    for size in sizes:
        print(f"Building packed M={size}", flush=True)
        rows.append(run_size(size, args.seed))
    write_outputs(args.output_dir, rows)
    if args.persist_size:
        destination = persist_packed_index(
            args.persist_size, args.artifact_dir, args.seed
        )
        print(f"Packed index persisted to {destination}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
