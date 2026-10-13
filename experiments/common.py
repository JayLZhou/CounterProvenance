"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import csv
import importlib
import platform
import sys
import time
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from counterprovenance import CellIndex, Slabs  # noqa: E402

RESULTS = ROOT / "results"


def environment() -> dict[str, str]:
    out = {"host": platform.node(), "python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "sklearn", "xgboost", "lightgbm", "dowhy"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = "-"
    return out


def write_csv(name: str, rows: Sequence[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    if not rows:
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def per_call_seconds(fn: Callable[[], object], calls: int, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) / calls)
    return median(samples)


def full_cell_index(bound, **kwargs) -> tuple[CellIndex, Slabs]:
    """One MCPST over the whole overlay (tree slots)."""
    slabs = Slabs(bound)
    items = list(range(len(bound.boxes)))
    cells = CellIndex(
        items=items,
        ranges=[slabs.ranges[i] for i in items],
        slots=[bound.boxes[i].tree for i in items],
        slab_counts=slabs.counts,
        values=[bound.boxes[i].value for i in items],
        offset=bound.offset,
        **kwargs,
    )
    return cells, slabs


def cell_records(cells: CellIndex) -> int:
    s = cells.stats
    return s.pft_nodes + s.pft_entries + 2 * s.faces + s.locator_words + len(cells.items)


def cell_bytes(cells: CellIndex) -> int:
    s = cells.stats
    return 8 * s.pft_nodes + 12 * s.pft_entries + 16 * s.faces + 4 * (s.locator_words + len(cells.items))


def california(data_home: str | None):
    from sklearn.datasets import fetch_california_housing

    ds = fetch_california_housing(data_home=data_home)
    return ds.data.astype(np.float64), ds.target.astype(np.float64), list(ds.feature_names)


def model_zoo(quick: bool = False) -> dict[str, Callable[[], object]]:
    """Name -> unfitted model factory (identity-link regressors)."""
    n = 60 if quick else 300
    zoo: dict[str, Callable[[], object]] = {}
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )

    zoo["sklearn-GBR"] = lambda: GradientBoostingRegressor(n_estimators=n, max_depth=3, learning_rate=0.1, random_state=0)
    zoo["sklearn-HGB"] = lambda: HistGradientBoostingRegressor(max_iter=n, max_depth=3, learning_rate=0.1, random_state=0)
    zoo["sklearn-RF"] = lambda: RandomForestRegressor(
        n_estimators=max(20, n // 3), max_depth=6, random_state=0, n_jobs=8
    )
    try:
        import xgboost as xgb

        zoo["XGBoost-exact"] = lambda: xgb.XGBRegressor(
            n_estimators=n, max_depth=3, learning_rate=0.1, tree_method="exact", random_state=0, n_jobs=8
        )
        zoo["XGBoost-hist"] = lambda: xgb.XGBRegressor(
            n_estimators=n, max_depth=4, learning_rate=0.1, tree_method="hist", random_state=0, n_jobs=8
        )
    except ImportError:
        pass
    try:
        import lightgbm as lgb

        zoo["LightGBM"] = lambda: lgb.LGBMRegressor(
            n_estimators=n, num_leaves=15, learning_rate=0.1, random_state=0, verbose=-1, n_jobs=8
        )
    except ImportError:
        pass
    return zoo


def native_leaves(model, rows: np.ndarray) -> np.ndarray | None:
    """Leaf ids reported by the model itself, shape (n, M), or None."""
    module = type(model).__module__
    name = type(model).__name__
    if module.startswith("xgboost"):
        import xgboost as xgb

        return model.get_booster().predict(xgb.DMatrix(rows), pred_leaf=True).astype(np.int64)
    if module.startswith("lightgbm"):
        return np.asarray(model.predict(rows, pred_leaf=True), dtype=np.int64)
    if name in {"GradientBoostingRegressor", "RandomForestRegressor", "ExtraTreesRegressor"}:
        out = model.apply(rows)
        return out[:, :, 0] if out.ndim == 3 else out
    return None


def value_tolerance(model) -> float:
    """XGBoost accumulates leaf values in float32; everything else in float64."""
    return 5e-5 if type(model).__module__.startswith("xgboost") else 1e-9


def markdown_table(rows: Sequence[dict], columns: Sequence[tuple[str, str]]) -> str:
    def fmt(v):
        if isinstance(v, float):
            if v != v:
                return "-"
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            if abs(v) >= 10:
                return f"{v:.1f}"
            return f"{v:.3g}"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        return str(v)

    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(fmt(r.get(k, "-")) for k, _ in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])
