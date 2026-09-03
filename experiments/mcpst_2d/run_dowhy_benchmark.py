#!/usr/bin/env python3
"""Compare MCPST queries with full DoWhy-GCM counterfactual recomputation.

The primary baseline calls ``counterfactual_samples(..., observed_data=unit)``
for every parametric query, so DoWhy repeats abduction, action, and prediction.
Model fitting is outside query time, matching DoWhy's normal serving workflow.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path
from statistics import median
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mcpst_2d.run_experiments import (  # noqa: E402
    GridPointLocator,
    compile_grid,
    grid_centers,
    sklearn_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", default="16,32,64")
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def median_seconds(operation: Callable[[], object], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return median(samples)


def make_interventions(point: np.ndarray) -> dict[str, Callable[[float], float]]:
    x_value = float(point[0])
    y_value = float(point[1])
    return {
        "X1": lambda _value, result=x_value: result,
        "X2": lambda _value, result=y_value: result,
    }


def run_benchmark(
    tree_counts: tuple[int, ...],
    query_count: int,
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], str]:
    try:
        import dowhy
        import dowhy.gcm as gcm
        from dowhy.gcm.ml import SklearnRegressionModel
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "Install requirements-dowhy.txt before running this benchmark."
        ) from exc

    gcm.config.disable_progress_bars()
    rng = np.random.default_rng(seed)
    training_x = rng.uniform(0.0, 1.0, size=(6_000, 2))
    training_y = (
        np.sin(6.0 * training_x[:, 0])
        + 0.7 * np.cos(8.0 * training_x[:, 1])
        + 1.2 * (training_x[:, 0] > training_x[:, 1])
        + rng.normal(0.0, 0.03, len(training_x))
    )
    observed = pd.DataFrame(
        {"X1": training_x[:, 0], "X2": training_x[:, 1], "Y": training_y}
    )
    unit = observed.iloc[[17]].copy()
    points = rng.uniform(0.0, 1.0, size=(query_count, 2))
    interventions = [make_interventions(point) for point in points]

    rows: list[dict[str, object]] = []
    for tree_count in tree_counts:
        graph = nx.DiGraph()
        graph.add_edges_from((("X1", "Y"), ("X2", "Y")))
        causal_model = gcm.InvertibleStructuralCausalModel(graph)
        causal_model.set_causal_mechanism("X1", gcm.EmpiricalDistribution())
        causal_model.set_causal_mechanism("X2", gcm.EmpiricalDistribution())
        causal_model.set_causal_mechanism(
            "Y",
            gcm.AdditiveNoiseModel(
                SklearnRegressionModel(
                    GradientBoostingRegressor(
                        n_estimators=tree_count,
                        max_depth=2,
                        learning_rate=0.08,
                        random_state=seed,
                    )
                )
            ),
        )
        started = time.perf_counter()
        gcm.fit(causal_model, observed)
        fit_seconds = time.perf_counter() - started

        fitted_model = causal_model.causal_mechanism("Y").prediction_model.sklearn_model
        unit_noise = float(
            unit["Y"].iloc[0] - fitted_model.predict(unit[["X1", "X2"]].to_numpy())[0]
        )
        noise_data = pd.DataFrame(
            {
                "X1": unit["X1"].to_numpy(),
                "X2": unit["X2"].to_numpy(),
                "Y": np.asarray([unit_noise]),
            }
        )

        x_thresholds, y_thresholds, estimators = sklearn_thresholds(fitted_model)
        locator = GridPointLocator(x_thresholds, y_thresholds, query_dtype=np.float32)
        centers = grid_centers(locator)
        states = np.column_stack(
            [estimator.apply(centers).astype(np.int32) for estimator in estimators]
        )
        cached_values = fitted_model.predict(centers) + unit_noise
        compilation = compile_grid(locator, states, cached_values)

        correctness_count = min(50, query_count)
        dowhy_values = np.asarray(
            [
                float(
                    gcm.counterfactual_samples(
                        causal_model,
                        interventions[index],
                        observed_data=unit,
                    )["Y"].iloc[0]
                )
                for index in range(correctness_count)
            ]
        )
        index_values = np.asarray(
            [
                compilation.index.query(float(point[0]), float(point[1]))[0]
                for point in points[:correctness_count]
            ]
        )
        max_error = float(np.max(np.abs(dowhy_values - index_values)))
        if max_error > 1e-10:
            raise AssertionError(f"DoWhy and MCPST differ by {max_error}.")

        def fresh_dowhy_queries() -> object:
            result = None
            for intervention in interventions:
                result = gcm.counterfactual_samples(
                    causal_model,
                    intervention,
                    observed_data=unit,
                )
            return result

        def cached_noise_dowhy_queries() -> object:
            result = None
            for intervention in interventions:
                result = gcm.counterfactual_samples(
                    causal_model,
                    intervention,
                    noise_data=noise_data,
                )
            return result

        def scalar_index_queries() -> object:
            result = None
            for point in points:
                result = compilation.index.query(float(point[0]), float(point[1]))
            return result

        fresh_dowhy_queries_warmup = interventions[:3]
        for intervention in fresh_dowhy_queries_warmup:
            gcm.counterfactual_samples(causal_model, intervention, observed_data=unit)
            gcm.counterfactual_samples(
                causal_model, intervention, noise_data=noise_data
            )
        compilation.index.query_batch(points[:3])

        fresh_seconds = median_seconds(fresh_dowhy_queries, repeats)
        cached_noise_seconds = median_seconds(cached_noise_dowhy_queries, repeats)
        scalar_index_seconds = median_seconds(scalar_index_queries, repeats)
        batch_index_seconds = median_seconds(
            lambda: compilation.index.query_batch(points), repeats
        )
        rows.append(
            {
                "dowhy_version": dowhy.__version__,
                "M": tree_count,
                "K_cells": locator.cell_count,
                "queries": query_count,
                "max_abs_error": max_error,
                "fit_s_excluded_from_query": fit_seconds,
                "index_build_s": compilation.total_seconds,
                "dowhy_fresh_s": fresh_seconds,
                "dowhy_cached_noise_s": cached_noise_seconds,
                "mcpst_scalar_s": scalar_index_seconds,
                "mcpst_batch_s": batch_index_seconds,
                "dowhy_fresh_us_per_query": (fresh_seconds * 1e6 / query_count),
                "mcpst_scalar_us_per_query": (scalar_index_seconds * 1e6 / query_count),
                "online_speedup_x": fresh_seconds / scalar_index_seconds,
                "cached_noise_speedup_x": (cached_noise_seconds / scalar_index_seconds),
                "batch_throughput_speedup_x": (fresh_seconds / batch_index_seconds),
            }
        )
    return rows, dowhy.__version__


def write_outputs(
    output_dir: Path, rows: list[dict[str, object]], dowhy_version: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dowhy_query.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table_rows = []
    for row in rows:
        table_rows.append(
            "| "
            f"{row['M']} | {row['K_cells']:,} | "
            f"{row['dowhy_fresh_us_per_query']:.2f} | "
            f"{row['mcpst_scalar_us_per_query']:.3f} | "
            f"{row['online_speedup_x']:.2f}x | "
            f"{row['cached_noise_speedup_x']:.2f}x | "
            f"{row['batch_throughput_speedup_x']:.2f}x | "
            f"{row['max_abs_error']:.1e} |"
        )
    summary = [
        "# MCPST vs DoWhy-GCM",
        "",
        f"- DoWhy: `{dowhy_version}`",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        "- Fitting is excluded from query time for both systems.",
        "- `DoWhy fresh` calls `counterfactual_samples` with `observed_data` for every query,"
        " so each call reruns abduction, action, and prediction.",
        "- `DoWhy cached noise` supplies `noise_data` and skips abduction.",
        "- Online speedup compares separate scalar queries; batch throughput is reported separately.",
        "",
        "| Trees | Cells | DoWhy fresh μs/q | MCPST scalar μs/q | Online speedup | Cached-noise speedup | Batch throughput | Max error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "The online scalar comparison is the conservative headline. The batch number also"
        " benefits from NumPy vectorization and should not replace it in the main claim.",
        "",
    ]
    (output_dir / "DOWHY_RESULTS.md").write_text("\n".join(summary), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency.
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    figure.patch.set_facecolor("white")
    tree_counts = [int(row["M"]) for row in rows]
    axes[0].plot(
        tree_counts,
        [float(row["dowhy_fresh_us_per_query"]) for row in rows],
        marker="o",
        label="DoWhy fresh",
    )
    axes[0].plot(
        tree_counts,
        [
            float(row["dowhy_cached_noise_s"]) * 1e6 / int(row["queries"])
            for row in rows
        ],
        marker="^",
        label="DoWhy cached noise",
    )
    axes[0].plot(
        tree_counts,
        [float(row["mcpst_scalar_us_per_query"]) for row in rows],
        marker="s",
        label="MCPST scalar",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(tree_counts)
    axes[0].set_xlabel("Component trees M")
    axes[0].set_ylabel("Latency (μs/query, log scale)")
    axes[0].set_title("Online counterfactual queries")
    axes[0].legend()

    axes[1].plot(
        tree_counts,
        [float(row["online_speedup_x"]) for row in rows],
        marker="o",
        label="Fresh / scalar",
    )
    axes[1].plot(
        tree_counts,
        [float(row["cached_noise_speedup_x"]) for row in rows],
        marker="^",
        label="Cached noise / scalar",
    )
    axes[1].plot(
        tree_counts,
        [float(row["batch_throughput_speedup_x"]) for row in rows],
        marker="s",
        linestyle="--",
        label="Fresh / batch",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(tree_counts)
    axes[1].set_xlabel("Component trees M")
    axes[1].set_ylabel("Speedup (log scale)")
    axes[1].set_title("MCPST speedup")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "dowhy_results.png", dpi=180, facecolor="white")
    figure.savefig(output_dir / "dowhy_results.pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tree_counts = tuple(int(value) for value in args.trees.split(","))
    if not tree_counts or any(value < 1 for value in tree_counts):
        raise ValueError("Every tree count must be positive.")
    if args.queries < 1 or args.repeats < 1:
        raise ValueError("queries and repeats must be positive.")
    rows, dowhy_version = run_benchmark(
        tree_counts, args.queries, args.repeats, args.seed
    )
    write_outputs(args.output_dir, rows, dowhy_version)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
