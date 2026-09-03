#!/usr/bin/env python3
"""Reproducible experiments for the 2D MCPST provenance index.

The benchmark measures exact storage records under one shared physical model:

* IIP independently materializes one balanced factor-tree root per cell;
* MCPST stores the same payload references and cell-root directory, but obtains
  roots by path-copying along a source-augmented minimum spanning tree.

It also validates exact answers against synthetic stump replay and sklearn's
GradientBoostingRegressor.  Timing numbers are medians after warm-up.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterprovenance.mcpst import (  # noqa: E402
    BalancedPFTLayout,
    GridPointLocator,
    MCPSTGridIndex,
    MCPSTMaterialization,
    MCPSTPlan,
    build_spanning_plan,
    fully_duplicated_pft_records,
    grid_adjacency,
    independent_pft_nodes,
    materialize_plan,
    transitions_from_states,
)


@dataclass(frozen=True)
class SyntheticGridWorkload:
    locator: GridPointLocator
    states: np.ndarray
    slot_axes: np.ndarray
    slot_thresholds: np.ndarray
    left_values: np.ndarray
    deltas: np.ndarray
    cached_values: np.ndarray
    base_value: float

    @property
    def component_count(self) -> int:
        return self.states.shape[1]

    @property
    def cell_count(self) -> int:
        return self.states.shape[0]

    def replay(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        selected_coordinates = values[:, self.slot_axes]
        right_leaf = selected_coordinates > self.slot_thresholds
        return (
            self.base_value + float(np.sum(self.left_values)) + right_leaf @ self.deltas
        )


@dataclass(frozen=True)
class Compilation:
    layout: BalancedPFTLayout
    transitions: tuple
    plan: MCPSTPlan
    materialization: MCPSTMaterialization
    index: MCPSTGridIndex
    transition_seconds: float
    mst_seconds: float
    materialize_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.transition_seconds + self.mst_seconds + self.materialize_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--query-count", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller workloads for a fast correctness/debug run.",
    )
    parser.add_argument(
        "--skip-gbdt",
        action="store_true",
        help="Skip the sklearn GradientBoostingRegressor sanity check.",
    )
    return parser.parse_args()


def make_stump_grid(
    x_groups: int,
    y_groups: int,
    group_size: int,
    seed: int,
    order: str = "grouped",
) -> SyntheticGridWorkload:
    """Build a full grid induced by axis-aligned tree-stump thresholds."""

    if x_groups < 1 or y_groups < 1 or group_size < 1:
        raise ValueError("Synthetic grids need positive group counts and size.")
    if order not in {"grouped", "x_scattered", "y_scattered"}:
        raise ValueError(f"Unknown component order {order!r}.")

    x_thresholds = np.arange(1, x_groups + 1, dtype=float) / (x_groups + 1)
    y_thresholds = np.arange(1, y_groups + 1, dtype=float) / (y_groups + 1)
    locator = GridPointLocator(x_thresholds, y_thresholds)

    x_components = [
        (0, group, member) for group in range(x_groups) for member in range(group_size)
    ]
    y_components = [
        (1, group, member) for group in range(y_groups) for member in range(group_size)
    ]
    if order == "x_scattered":
        x_components.sort(key=lambda item: (item[2], item[1]))
    if order == "y_scattered":
        y_components.sort(key=lambda item: (item[2], item[1]))
    components = x_components + y_components

    slot_axes = np.asarray([item[0] for item in components], dtype=np.int8)
    slot_groups = np.asarray([item[1] for item in components], dtype=np.int32)
    slot_thresholds = np.where(
        slot_axes == 0,
        x_thresholds[slot_groups],
        y_thresholds[slot_groups],
    )

    x_cell = np.tile(np.arange(locator.x_cells), locator.y_cells)
    y_cell = np.repeat(np.arange(locator.y_cells), locator.x_cells)
    states = np.where(
        slot_axes[None, :] == 0,
        x_cell[:, None] > slot_groups[None, :],
        y_cell[:, None] > slot_groups[None, :],
    ).astype(np.int8)

    rng = np.random.default_rng(seed)
    left_values = rng.normal(0.0, 0.5, len(components))
    deltas = rng.normal(0.0, 1.0, len(components))
    base_value = float(rng.normal())
    cached_values = base_value + float(np.sum(left_values)) + states @ deltas
    return SyntheticGridWorkload(
        locator=locator,
        states=states,
        slot_axes=slot_axes,
        slot_thresholds=slot_thresholds,
        left_values=left_values,
        deltas=deltas,
        cached_values=cached_values,
        base_value=base_value,
    )


def compile_grid(
    locator: GridPointLocator,
    states: np.ndarray,
    cached_values: np.ndarray,
    objective: str = "pft",
) -> Compilation:
    layout = BalancedPFTLayout(states.shape[1])
    adjacency = grid_adjacency(locator.x_cells, locator.y_cells)

    started = time.perf_counter()
    transitions = transitions_from_states(states, adjacency, layout)
    transition_seconds = time.perf_counter() - started

    started = time.perf_counter()
    plan = build_spanning_plan(len(states), transitions, layout, objective=objective)
    mst_seconds = time.perf_counter() - started

    started = time.perf_counter()
    materialization = materialize_plan(states, layout, plan)
    materialize_seconds = time.perf_counter() - started
    index = MCPSTGridIndex(locator, materialization.roots, cached_values)
    return Compilation(
        layout=layout,
        transitions=transitions,
        plan=plan,
        materialization=materialization,
        index=index,
        transition_seconds=transition_seconds,
        mst_seconds=mst_seconds,
        materialize_seconds=materialize_seconds,
    )


def storage_counts(compilation: Compilation, cell_count: int) -> dict[str, int]:
    payloads = compilation.materialization.arena.payload_count
    iip_nodes = independent_pft_nodes(cell_count, compilation.layout)
    ours_nodes = compilation.materialization.internal_nodes
    fully_duplicated = fully_duplicated_pft_records(cell_count, compilation.layout)
    return {
        "shared_payload_records": payloads,
        "iip_internal_nodes": iip_nodes,
        "mcpst_internal_nodes": ours_nodes,
        "iip_total_records": payloads + cell_count + iip_nodes,
        "mcpst_total_records": payloads + cell_count + ours_nodes,
        "fully_duplicated_records_q1": fully_duplicated,
    }


def changed_statistics(compilation: Compilation) -> tuple[int, float, int]:
    counts = [edge.changed_count for edge in compilation.transitions]
    positive = [count for count in counts if count]
    return (
        max(counts, default=0),
        float(np.mean(positive)) if positive else 0.0,
        sum(count == 0 for count in counts),
    )


def run_scaling(sizes: Sequence[int], seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component_count in sizes:
        x_groups = component_count // 2
        y_groups = component_count - x_groups
        workload = make_stump_grid(x_groups, y_groups, 1, seed + component_count)
        compilation = compile_grid(
            workload.locator, workload.states, workload.cached_values
        )
        counts = storage_counts(compilation, workload.cell_count)
        max_changed, mean_changed, zero_edges = changed_statistics(compilation)
        predicted = (
            compilation.layout.build_cost
            + (workload.cell_count - 1) * compilation.layout.height
        )
        if compilation.materialization.internal_nodes != predicted:
            raise AssertionError("General-position workload missed its exact bound.")
        rows.append(
            {
                "M": component_count,
                "Bx": x_groups,
                "By": y_groups,
                "K_cells": workload.cell_count,
                "E_adjacencies": len(compilation.transitions),
                "max_c_e": max_changed,
                "mean_positive_c_e": mean_changed,
                "zero_change_edges": zero_edges,
                **counts,
                "record_compression_x": (
                    counts["iip_total_records"] / counts["mcpst_total_records"]
                ),
                "full_dup_compression_x": (
                    counts["fully_duplicated_records_q1"]
                    / counts["mcpst_total_records"]
                ),
                "predicted_mcpst_nodes": predicted,
                "source_roots": compilation.plan.source_roots,
                "transition_s": compilation.transition_seconds,
                "mst_s": compilation.mst_seconds,
                "materialize_s": compilation.materialize_seconds,
                "total_index_s": compilation.total_seconds,
            }
        )
    return rows


def run_coincidence(
    group_sizes: Sequence[int], seed: int, component_count: int = 256
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if component_count % 2:
        raise ValueError("Coincidence experiment needs an even component count.")
    components_per_axis = component_count // 2
    for group_size in group_sizes:
        if components_per_axis % group_size:
            raise ValueError("Every coincidence group must divide M/2.")
        x_groups = y_groups = components_per_axis // group_size
        workload = make_stump_grid(x_groups, y_groups, group_size, seed + group_size)
        compilation = compile_grid(
            workload.locator, workload.states, workload.cached_values
        )
        counts = storage_counts(compilation, workload.cell_count)
        max_changed, mean_changed, zero_edges = changed_statistics(compilation)
        rows.append(
            {
                "kappa": group_size,
                "M": workload.component_count,
                "K_cells": workload.cell_count,
                "E_adjacencies": len(compilation.transitions),
                "max_c_e": max_changed,
                "mean_positive_c_e": mean_changed,
                "zero_change_edges": zero_edges,
                **counts,
                "record_compression_x": (
                    counts["iip_total_records"] / counts["mcpst_total_records"]
                ),
                "full_dup_compression_x": (
                    counts["fully_duplicated_records_q1"]
                    / counts["mcpst_total_records"]
                ),
                "total_index_s": compilation.total_seconds,
            }
        )
    return rows


def run_ablation(seed: int) -> list[dict[str, object]]:
    workload = make_stump_grid(8, 8, 4, seed, order="y_scattered")
    layout = BalancedPFTLayout(workload.component_count)
    transitions = transitions_from_states(
        workload.states,
        grid_adjacency(workload.locator.x_cells, workload.locator.y_cells),
        layout,
    )
    compiled: list[tuple[str, MCPSTPlan, MCPSTMaterialization, float]] = []
    for label, objective in (
        ("MCPST (exact PFT weight)", "pft"),
        ("Change-count MST", "changed"),
        ("Unweighted MST", "unweighted"),
    ):
        started = time.perf_counter()
        plan = build_spanning_plan(
            workload.cell_count, transitions, layout, objective=objective
        )
        materialization = materialize_plan(workload.states, layout, plan)
        elapsed = time.perf_counter() - started
        compiled.append((label, plan, materialization, elapsed))

    best_nodes = compiled[0][2].internal_nodes
    if compiled[1][2].internal_nodes <= best_nodes:
        raise AssertionError("Ablation workload must separate exact and count weights.")
    payloads = compiled[0][2].arena.payload_count
    iip_nodes = independent_pft_nodes(workload.cell_count, layout)
    iip_total = payloads + workload.cell_count + iip_nodes
    rows: list[dict[str, object]] = []
    for label, plan, materialization, elapsed in compiled:
        total = payloads + workload.cell_count + materialization.internal_nodes
        selected_transition_weights = [
            edge.weight for edge in plan.selected_edges if not edge.source_edge
        ]
        rows.append(
            {
                "strategy": label,
                "M": workload.component_count,
                "K_cells": workload.cell_count,
                "source_roots": plan.source_roots,
                "internal_nodes": materialization.internal_nodes,
                "total_records": total,
                "iip_total_records": iip_total,
                "compression_vs_iip_x": iip_total / total,
                "overhead_vs_mcpst_x": materialization.internal_nodes / best_nodes,
                "mean_selected_transition_weight": float(
                    np.mean(selected_transition_weights)
                ),
                "plan_and_materialize_s": elapsed,
            }
        )
    return rows


def median_seconds(
    operation: Callable[[], object], repeats: int, warmups: int = 2
) -> float:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return median(samples)


def run_query_benchmark(
    sizes: Sequence[int],
    query_count: int,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, size=(query_count, 2))
    for component_count in sizes:
        workload = make_stump_grid(
            component_count // 2,
            component_count - component_count // 2,
            1,
            seed + component_count,
        )
        compilation = compile_grid(
            workload.locator, workload.states, workload.cached_values
        )
        replay_values = workload.replay(points)
        index_values, _ = compilation.index.query_batch(points)
        max_error = float(np.max(np.abs(replay_values - index_values)))
        if max_error > 1e-10:
            raise AssertionError(f"Synthetic index error {max_error} is not exact.")

        replay_seconds = median_seconds(lambda: workload.replay(points), repeats)
        index_seconds = median_seconds(
            lambda: compilation.index.query_batch(points), repeats
        )
        locator_seconds = median_seconds(
            lambda: workload.locator.cell_ids(points), repeats
        )
        rows.append(
            {
                "M": component_count,
                "K_cells": workload.cell_count,
                "queries": query_count,
                "max_abs_error": max_error,
                "raw_replay_s": replay_seconds,
                "mcpst_query_s": index_seconds,
                "point_location_s": locator_seconds,
                "raw_replay_us_per_query": replay_seconds * 1e6 / query_count,
                "mcpst_us_per_query": index_seconds * 1e6 / query_count,
                "speedup_vs_raw_replay_x": replay_seconds / index_seconds,
                "speedup_vs_iip_query_x": 1.0,
            }
        )
    return rows


def sklearn_thresholds(model: object) -> tuple[np.ndarray, np.ndarray, list[object]]:
    estimators = [row[0] for row in model.estimators_]
    by_feature: list[list[float]] = [[], []]
    for estimator in estimators:
        tree = estimator.tree_
        for feature, threshold in zip(tree.feature, tree.threshold):
            if feature in (0, 1) and 0.0 < threshold < 1.0:
                by_feature[int(feature)].append(float(threshold))
    return (
        np.unique(np.asarray(by_feature[0], dtype=float)),
        np.unique(np.asarray(by_feature[1], dtype=float)),
        estimators,
    )


def grid_centers(locator: GridPointLocator) -> np.ndarray:
    x_edges = np.concatenate(([locator.x_min], locator.x_thresholds, [locator.x_max]))
    y_edges = np.concatenate(([locator.y_min], locator.y_thresholds, [locator.y_max]))
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def run_gbdt(
    sizes: Sequence[int], query_count: int, repeats: int, seed: int
) -> list[dict[str, object]]:
    try:
        import sklearn
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - optional experiment dependency.
        raise RuntimeError("The GBDT experiment requires scikit-learn.") from exc

    rng = np.random.default_rng(seed)
    training_x = rng.uniform(0.0, 1.0, size=(6_000, 2))
    training_y = (
        np.sin(6.0 * training_x[:, 0])
        + 0.7 * np.cos(8.0 * training_x[:, 1])
        + 1.2 * (training_x[:, 0] > training_x[:, 1])
        + rng.normal(0.0, 0.03, len(training_x))
    )
    random_queries = rng.uniform(0.0, 1.0, size=(query_count, 2))

    rows: list[dict[str, object]] = []
    for component_count in sizes:
        model = GradientBoostingRegressor(
            n_estimators=component_count,
            max_depth=2,
            learning_rate=0.08,
            random_state=seed,
        )
        started = time.perf_counter()
        model.fit(training_x, training_y)
        train_seconds = time.perf_counter() - started

        x_thresholds, y_thresholds, estimators = sklearn_thresholds(model)
        # sklearn tree predictors quantize numeric inputs to float32 before
        # threshold comparisons; the locator must use the same boundary rule.
        locator = GridPointLocator(x_thresholds, y_thresholds, query_dtype=np.float32)
        centers = grid_centers(locator)
        states = np.column_stack(
            [estimator.apply(centers).astype(np.int32) for estimator in estimators]
        )
        cached_values = model.predict(centers)
        compilation = compile_grid(locator, states, cached_values)
        counts = storage_counts(compilation, locator.cell_count)
        max_changed, mean_changed, zero_edges = changed_statistics(compilation)

        index_values, _ = compilation.index.query_batch(random_queries)
        model_values = model.predict(random_queries)
        random_error = float(np.max(np.abs(index_values - model_values)))

        boundary_points: list[tuple[float, float]] = []
        boundary_points.extend((float(value), 0.371) for value in x_thresholds)
        boundary_points.extend((0.619, float(value)) for value in y_thresholds)
        boundary_array = np.asarray(boundary_points, dtype=float)
        boundary_values, _ = compilation.index.query_batch(boundary_array)
        boundary_error = float(
            np.max(np.abs(boundary_values - model.predict(boundary_array)))
        )
        if max(random_error, boundary_error) > 1e-10:
            raise AssertionError(
                "Global-threshold grid did not exactly reproduce sklearn GBDT."
            )

        model_seconds = median_seconds(lambda: model.predict(random_queries), repeats)
        index_seconds = median_seconds(
            lambda: compilation.index.query_batch(random_queries), repeats
        )
        rows.append(
            {
                "sklearn_version": sklearn.__version__,
                "M": component_count,
                "Bx": len(x_thresholds),
                "By": len(y_thresholds),
                "K_cells": locator.cell_count,
                "E_adjacencies": len(compilation.transitions),
                "max_c_e": max_changed,
                "mean_positive_c_e": mean_changed,
                "zero_change_edges": zero_edges,
                **counts,
                "record_compression_x": (
                    counts["iip_total_records"] / counts["mcpst_total_records"]
                ),
                "full_dup_compression_x": (
                    counts["fully_duplicated_records_q1"]
                    / counts["mcpst_total_records"]
                ),
                "max_random_error": random_error,
                "max_boundary_error": boundary_error,
                "train_s": train_seconds,
                "index_build_s": compilation.total_seconds,
                "model_predict_s": model_seconds,
                "index_query_s": index_seconds,
                "query_speedup_x": model_seconds / index_seconds,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1_000:
            return f"{float(value):,.1f}"
        if abs(float(value)) >= 10:
            return f"{float(value):.2f}"
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def markdown_table(
    rows: Sequence[dict[str, object]], columns: Sequence[tuple[str, str]]
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---:" for _ in columns) + "|"
    body = [
        "| " + " | ".join(format_number(row[key]) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_summary(
    path: Path,
    scaling: Sequence[dict[str, object]],
    coincidence: Sequence[dict[str, object]],
    ablation: Sequence[dict[str, object]],
    query: Sequence[dict[str, object]],
    gbdt: Sequence[dict[str, object]],
) -> None:
    largest = scaling[-1]
    query_largest = query[-1]
    component_count = int(largest["M"])
    cell_count = int(largest["K_cells"])
    payload_count = int(largest["shared_payload_records"])
    internal_nodes = int(largest["mcpst_internal_nodes"])
    build_cost = int(largest["iip_internal_nodes"]) // cell_count
    payload_sensitivity = []
    for payload_size in (1, 4, 16):
        fully_duplicated = cell_count * (
            component_count * payload_size + build_cost + 1
        )
        ours = payload_count * payload_size + cell_count + internal_nodes
        payload_sensitivity.append(
            {
                "q": payload_size,
                "fully_duplicated_records": fully_duplicated,
                "mcpst_records": ours,
                "compression_x": fully_duplicated / ours,
            }
        )
    lines = [
        "# MCPST 二维实验结果",
        "",
        "## 结论摘要",
        "",
        (
            f"- 在 general-position stump workload 上，最大规模 "
            f"$M={largest['M']}$、$K={largest['K_cells']:,}$ 时，MCPST 的总记录数"
            f"相对 IIP 压缩 **{largest['record_compression_x']:.2f}x**。"
        ),
        (
            "- 若连每个 cell 的 active leaf payload 也完全复制，同一配置的压缩为 "
            f"**{largest['full_dup_compression_x']:.2f}x**；前一个数字使用的是更强、更保守的 IIP。"
        ),
        (
            f"- 对 {query_largest['queries']:,} 个查询，$M={query_largest['M']}$ 时"
            f"相对 raw ensemble replay 加速 **{query_largest['speedup_vs_raw_replay_x']:.2f}x**；"
            "IIP 与 MCPST 使用相同 point-location/root-directory，因此二者查询复杂度相同。"
        ),
        (
            f"- 构造消融中，change-count MST 的内部节点开销是精确 PFT copy-cost "
            f"MST 的 **{ablation[1]['overhead_vs_mcpst_x']:.2f}x**。"
        ),
        "- 所有 synthetic 与 sklearn GBDT 随机点、阈值等号点均通过 exactness 校验。",
        "",
        "## 实验环境",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- Platform: `{platform.platform()}`",
        "- 时间为 2 次 warm-up 后多次运行的 median；空间为结构记录数，不是 Python 对象字节数。",
        "",
        "## General-Position Scaling",
        "",
        markdown_table(
            scaling,
            (
                ("M", "M"),
                ("K_cells", "K"),
                ("iip_total_records", "IIP records"),
                ("fully_duplicated_records_q1", "Full duplicate"),
                ("mcpst_total_records", "MCPST records"),
                ("record_compression_x", "vs IIP"),
                ("full_dup_compression_x", "vs full"),
                ("total_index_s", "Build (s)"),
            ),
        ),
        "",
        "这里每个 cell boundary 仅改变一个 component，实测内部节点数严格等于 "
        "$b+(K-1)\\log_2 M$，不是拟合曲线。两种方法都计入共享 payload 和每-cell root pointer。",
        "",
        "## Fully Duplicated Payload Sensitivity",
        "",
        markdown_table(
            payload_sensitivity,
            (
                ("q", "Payload nodes q"),
                ("fully_duplicated_records", "Full duplicate"),
                ("mcpst_records", "MCPST"),
                ("compression_x", "Compression"),
            ),
        ),
        "",
        "这里 q 是一个 active tree-leaf provenance payload 的平均 circuit-node 数。"
        "q=1 对应当前 leaf-ID 原型；q>1 是 final provenance payload 的参数化空间预测，"
        "不是尚未实现的实测字节数。",
        "在 general-position、$q=1$、$M$ 为 2 的幂且 $K$ 足够大时，"
        "strong IIP 与 full-duplicate 的压缩率分别近似 "
        "$M/(\\log_2 M+1)$ 和 $2M/(\\log_2 M+1)$。最坏情况下若每条边几乎改变全部"
        " components，MCPST 会退化到与 IIP 同阶，因此不存在对所有模型都固定的压缩倍数。",
        "",
        "## Coincident Thresholds",
        "",
        markdown_table(
            coincidence,
            (
                ("kappa", "κ"),
                ("M", "M"),
                ("K_cells", "K"),
                ("mcpst_internal_nodes", "MCPST nodes"),
                ("record_compression_x", "Compression"),
            ),
        ),
        "",
        f"本组固定 $M={coincidence[0]['M']}$，将多个 components 合并到同一 threshold，"
        "使 κ 增加且 cell 数相应减少，用来检验 copy-cost 对 coincident selectors 的敏感性。",
        "",
        "## Spanning-Plan Ablation",
        "",
        markdown_table(
            ablation,
            (
                ("strategy", "Strategy"),
                ("internal_nodes", "Internal nodes"),
                ("compression_vs_iip_x", "vs IIP"),
                ("overhead_vs_mcpst_x", "vs MCPST cost"),
            ),
        ),
        "",
        "该 workload 让 horizontal boundaries 改变连续 leaves，vertical boundaries 改变"
        "分散 leaves；两者 $c_e$ 相同但 $\\rho_e$ 不同。因此只看 changed count 无法选出"
        "最省 path-copy 的 construction tree。",
        "",
        "## Query",
        "",
        markdown_table(
            query,
            (
                ("M", "M"),
                ("queries", "Queries"),
                ("raw_replay_us_per_query", "Replay μs/q"),
                ("mcpst_us_per_query", "Index μs/q"),
                ("speedup_vs_raw_replay_x", "Speedup"),
            ),
        ),
        "",
        "MCPST 的查询为 predecessor point location 后直接读取 root/value。它加速的是从头"
        " replay；相对已完整物化所有 cells 的 IIP，查询时间预期持平，优势在空间与构建写入量。",
    ]
    if gbdt:
        lines.extend(
            [
                "",
                "## sklearn GBDT Sanity Check",
                "",
                markdown_table(
                    gbdt,
                    (
                        ("M", "Trees"),
                        ("K_cells", "Global-grid K"),
                        ("zero_change_edges", "Zero-change edges"),
                        ("record_compression_x", "Compression"),
                        ("query_speedup_x", "Query speedup"),
                        ("max_random_error", "Max error"),
                    ),
                ),
                "",
                "这里使用所有 sklearn split thresholds 的 global-grid refinement。它是 exact "
                "但不保证 cell decomposition 最小，因此这是端到端正确性与趋势验证，不应被写成"
                "最优 partition 结果。",
            ]
        )
    lines.extend(
        [
            "",
            "## DoWhy-GCM Comparison",
            "",
            "完整 abduction-action-prediction 重算使用独立脚本运行；见 "
            "`DOWHY_RESULTS.md` 和 `dowhy_query.csv`。主表应使用 online scalar speedup，"
            "不要用包含 NumPy vectorization 的 batch throughput 代替。",
        ]
    )
    lines.extend(
        [
            "",
            "## Packed Million-Cell Scaling",
            "",
            "去除 Python object overhead 的 packed general-position builder 使用独立脚本；"
            "见 `PACKED_RESULTS.md` 和 `packed_scaling.csv`。该实验实际分配 MCPST arrays，"
            "并在同一字宽下精确计算 strong IIP 与 fully duplicated baseline bytes。",
        ]
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "1. 记录数比较采用相同 balanced PFT layout，并让 IIP 也共享 payload，因此不是"
            "靠弱化 baseline 得到压缩率。",
            "2. MCPST 的最优性只针对 adjacency-constrained、single-parent、fixed-layout、"
            "无跨分支全局 internal-node hash-consing 的构造模型。",
            "3. 本实验验证二维 exact tree ensembles；高维 cell enumeration 仍是后续工作。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(
    output_dir: Path,
    scaling: Sequence[dict[str, object]],
    coincidence: Sequence[dict[str, object]],
    ablation: Sequence[dict[str, object]],
    query: Sequence[dict[str, object]],
    gbdt: Sequence[dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional experiment dependency.
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    figure.patch.set_facecolor("white")

    m_values = [int(row["M"]) for row in scaling]
    axes[0, 0].plot(
        m_values,
        [int(row["iip_total_records"]) for row in scaling],
        marker="o",
        color="#457b9d",
        label="Strong IIP",
    )
    axes[0, 0].plot(
        m_values,
        [int(row["fully_duplicated_records_q1"]) for row in scaling],
        marker="^",
        color="#e76f51",
        label="Full duplicate",
    )
    axes[0, 0].plot(
        m_values,
        [int(row["mcpst_total_records"]) for row in scaling],
        marker="s",
        color="#2a9d8f",
        label="MCPST",
    )
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(m_values)
    axes[0, 0].set_xticklabels([str(value) for value in m_values])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Component trees M")
    axes[0, 0].set_ylabel("Storage records")
    axes[0, 0].set_title("General-position storage")
    axes[0, 0].legend()

    kappa_values = [int(row["kappa"]) for row in coincidence]
    axes[0, 1].plot(
        kappa_values,
        [float(row["record_compression_x"]) for row in coincidence],
        marker="o",
        color="#2a9d8f",
    )
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xticks(kappa_values)
    axes[0, 1].set_xticklabels([str(value) for value in kappa_values])
    axes[0, 1].set_xlabel(r"Coincident changes $\kappa$")
    axes[0, 1].set_ylabel("IIP / MCPST records")
    axes[0, 1].set_title("Threshold coincidence")

    axes[1, 0].bar(
        [r"Exact $\rho_e$-MST", r"$c_e$-MST", "Unweighted"],
        [int(row["internal_nodes"]) for row in ablation],
        color=["#264653", "#e9c46a", "#e76f51"],
    )
    axes[1, 0].set_ylabel("PFT internal nodes")
    axes[1, 0].set_title("Construction-plan ablation")

    axes[1, 1].plot(
        [int(row["M"]) for row in query],
        [float(row["speedup_vs_raw_replay_x"]) for row in query],
        marker="o",
        color="#e76f51",
        label="Synthetic stumps",
    )
    if gbdt:
        axes[1, 1].plot(
            [int(row["M"]) for row in gbdt],
            [float(row["query_speedup_x"]) for row in gbdt],
            marker="s",
            color="#457b9d",
            label="sklearn GBDT",
        )
    axes[1, 1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_xscale("log", base=2)
    query_ticks = sorted(
        {int(row["M"]) for row in query} | {int(row["M"]) for row in gbdt}
    )
    axes[1, 1].set_xticks(query_ticks)
    axes[1, 1].set_xticklabels([str(value) for value in query_ticks])
    axes[1, 1].set_xlabel("Component trees M")
    axes[1, 1].set_ylabel("Replay / index time")
    axes[1, 1].set_title("Query speedup")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "mcpst_results.png", dpi=180, facecolor="white")
    figure.savefig(output_dir / "mcpst_results.pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        scaling_sizes = (16, 32, 64)
        query_sizes = (32, 64)
        coincidence_sizes = (1, 2, 4)
        coincidence_component_count = 64
        gbdt_sizes = (8, 16)
        query_count = min(args.query_count, 10_000)
        repeats = min(args.repeats, 3)
    else:
        scaling_sizes = (16, 32, 64, 128, 256, 512)
        query_sizes = (32, 64, 128, 256)
        coincidence_sizes = (1, 2, 4, 8, 16)
        coincidence_component_count = 256
        gbdt_sizes = (16, 32, 64)
        query_count = args.query_count
        repeats = args.repeats

    print("[1/5] general-position scaling", flush=True)
    scaling = run_scaling(scaling_sizes, args.seed)
    print("[2/5] coincident-threshold sensitivity", flush=True)
    coincidence = run_coincidence(
        coincidence_sizes,
        args.seed + 100,
        component_count=coincidence_component_count,
    )
    print("[3/5] spanning-plan ablation", flush=True)
    ablation = run_ablation(args.seed + 200)
    print("[4/5] synthetic query benchmark", flush=True)
    query = run_query_benchmark(query_sizes, query_count, repeats, args.seed + 300)
    print("[5/5] sklearn GBDT sanity check", flush=True)
    gbdt = (
        []
        if args.skip_gbdt
        else run_gbdt(gbdt_sizes, query_count, repeats, args.seed + 400)
    )

    write_csv(args.output_dir / "scaling.csv", scaling)
    write_csv(args.output_dir / "coincidence.csv", coincidence)
    write_csv(args.output_dir / "ablation.csv", ablation)
    write_csv(args.output_dir / "query.csv", query)
    write_csv(args.output_dir / "gbdt.csv", gbdt)
    write_summary(
        args.output_dir / "RESULTS.md",
        scaling,
        coincidence,
        ablation,
        query,
        gbdt,
    )
    make_plots(args.output_dir, scaling, coincidence, ablation, query, gbdt)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
