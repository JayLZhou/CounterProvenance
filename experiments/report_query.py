#!/usr/bin/env python3
"""Summarise results/qbench_*.csv and results/dowhy_*.csv into QUERY_RESULTS.md."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from common import RESULTS, markdown_table

ORDER = ["cal", "yprop", "topo", "uji", "mars"]


def load(prefix: str, exclude: str | None = None) -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob(f"{prefix}*.csv")):
        if exclude and path.name.startswith(exclude):
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                for k, v in row.items():
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                rows.append(row)
    return rows


def geo(values) -> float:
    values = [v for v in values if v and v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def span(values) -> str:
    values = [v for v in values if v and math.isfinite(v)]
    if not values:
        return "-"
    lo, hi = min(values), max(values)
    fmt = (lambda x: f"{x:,.0f}") if lo >= 10 else (lambda x: f"{x:.1f}")
    return f"{fmt(lo)}-{fmt(hi)} (geo {geo(values):,.1f})"


def main() -> None:
    main_rows = [r for r in load("qbench_", exclude="qbench_scale_") if r["trees"] >= 250]
    scale_rows = load("qbench_scale_")
    out = ["# Query benchmark", "",
           "Single-threaded; speedup = baseline time / our time per query; ranges over the models of each row.", ""]

    table = []
    groups = defaultdict(list)
    for r in main_rows:
        groups[(r["dataset"], int(r["d"]))].append(r)
    for (ds, d) in sorted(groups, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1])):
        rs = groups[(ds, d)]
        table.append({
            "dataset": ds, "features": int(rs[0]["features"]), "d": d, "models": len(rs),
            "py": span([r["py_loop_all_us"] / r["py_ours_us"] for r in rs]),
            "py_bound": span([r["py_loop_bound_us"] / r["py_ours_us"] for r in rs]),
            "nb": span([r["nb_loop_all_batch_us"] / r["nb_ours_batch_us"] for r in rs if r.get("nb_ours_batch_us")]),
            "lib": span([r["lib_single_best_us"] / r["nb_ours_call_us"] for r in rs if r.get("nb_ours_call_us")]),
            "lib_batch": span([r["lib_batch_us"] / r["nb_ours_batch_us"] for r in rs if r.get("nb_ours_batch_us")]),
            "slider": span([r["slider_loop_us"] / r["slider_ours_us"] for r in rs]),
            "ours_py_us": span([r["py_ours_us"] for r in rs]),
            "records": span([r["records"] for r in rs]),
            "build_s": span([r["build_s"] for r in rs]),
        })
    out += ["## Speedups (1000 trees; RF 250)", "", markdown_table(table, [
        ("dataset", "data"), ("features", "p"), ("d", "d"), ("models", "models"),
        ("py", "pure Python vs loop (all trees)"), ("py_bound", "vs loop+binding"),
        ("nb", "numba vs numba loop (batch)"), ("lib", "numba call vs library single row"),
        ("lib_batch", "numba batch vs library batch"), ("slider", "slider vs incremental loop"),
    ]), "", "## Our cost", "", markdown_table(table, [
        ("dataset", "data"), ("d", "d"), ("ours_py_us", "pure-Python query (us)"),
        ("records", "records"), ("build_s", "build (s)"),
    ]), ""]

    if scale_rows:
        st = []
        for r in sorted(scale_rows, key=lambda r: (r["dataset"], r["model"], r["d"], r["trees"])):
            st.append({"dataset": r["dataset"], "model": r["model"], "d": int(r["d"]), "trees": int(r["trees"]),
                       "loop": r["py_loop_all_us"], "ours": r["py_ours_us"], "py_x": r["py_loop_all_us"] / r["py_ours_us"],
                       "nb_x": r["nb_loop_all_batch_us"] / r["nb_ours_batch_us"], "records": r["records"]})
        out += ["## Tree count", "", markdown_table(st, [
            ("dataset", "data"), ("model", "model"), ("d", "d"), ("trees", "trees"), ("loop", "loop (us)"),
            ("ours", "ours (us)"), ("py_x", "pure-Python x"), ("nb_x", "numba x"), ("records", "records")]), ""]

    dw = load("dowhy_")
    if dw:
        dt = []
        groups = defaultdict(list)
        for r in dw:
            groups[(r["dataset"], int(r["d"]))].append(r)
        for (ds, d) in sorted(groups, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1])):
            rs = groups[(ds, d)]
            dt.append({"dataset": ds, "d": d, "dowhy_ms": span([r["dowhy_fresh_us"] / 1000 for r in rs]),
                       "py": span([r["dowhy_fresh_us"] / r["ours_py_us"] for r in rs]),
                       "nb": span([r["dowhy_fresh_us"] / r["ours_nb_us"] for r in rs if r.get("ours_nb_us")]),
                       "cached": span([r["dowhy_cached_us"] / r["ours_py_us"] for r in rs])})
        out += ["## DoWhy-GCM", "", markdown_table(dt, [
            ("dataset", "data"), ("d", "d"), ("dowhy_ms", "DoWhy per query (ms)"), ("py", "vs ours, pure Python"),
            ("cached", "DoWhy cached noise vs ours (Python)"), ("nb", "vs ours, numba")]), ""]
    path = RESULTS / "QUERY_RESULTS.md"
    path.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
