#!/usr/bin/env python3
"""Tables and box plots of the main experiment (d = 1, 2, 4, 8).

The baselines (three-step, ChiRho, DoWhy-GCM) come from ``--base`` (``results/scm_full_*.csv`` and
``results/scm_lat/*.npz``), Tessera from ``--ours`` (an ``--ours-only`` run: ``results/scm_full_*.csv``
and ``results/scm_lat_ours/*.npz``).  Rows are matched on (dataset, family, d, unit).
Output in ``--out``: ``summary_by_d.csv``, ``summary_by_dataset.csv``, ``box_by_d.{pdf,png}``,
``box_by_dataset.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = [("ours", "Tessera"), ("three_step", "Three-step"), ("chirho", "ChiRho"), ("dowhy", "DoWhy-GCM")]
COLORS = {"ours": "#2a78d6", "three_step": "#eb6834", "chirho": "#1baf7a", "dowhy": "#eda100"}
KEY = ["dataset", "family", "d", "unit"]


def load(root: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(root / "results" / "scm_full_*.csv"))], ignore_index=True)
    return df[df.status.eq("ok") & df.d.le(8)]


def gmean(x) -> float:
    return float(np.exp(np.log(np.asarray(x, dtype=float)).mean()))


def summary(m: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in m.groupby(by):
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)), settings=len(g))
        r["ours_us"] = g.ours_us.mean()
        for b, _ in METHODS[1:]:
            r[f"{b}_us"] = g[f"{b}_us"].mean()
            r[f"speedup_{b}"] = gmean(g[f"{b}_us"] / g.ours_us)
            r[f"min_speedup_{b}"] = float((g[f"{b}_us"] / g.ours_us).min())
        rows.append(r)
    return pd.DataFrame(rows)


def latencies(m: pd.DataFrame, base: Path, ours: Path, cap: int) -> dict:
    """Per-query latencies (us) pooled per (d, method) and per (dataset, d, method); ``cap`` queries per row."""
    out = {}
    for r in m.itertuples(index=False):
        tag = f"{r.dataset}_{r.family}_d{r.d}_u{r.unit}_d{r.d}"
        zb = np.load(base / "results" / "scm_lat" / f"{tag}.npz")
        zo = np.load(ours / "results" / "scm_lat_ours" / f"{tag}.npz")
        for name, z, k in [("ours", zo, "ours_py")] + [(b, zb, b) for b, _ in METHODS[1:]]:
            v = z[k][:cap]
            out.setdefault((r.d, name), []).append(v)
            out.setdefault((r.dataset, r.d, name), []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def style(ax):
    ax.set_yscale("log")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def boxes(ax, data, positions, names, width):
    bp = ax.boxplot(data, positions=positions, widths=width, showfliers=False, patch_artist=True,
                    whis=(5, 95), medianprops=dict(color="#0b0b0b", linewidth=1.2))
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[n])
        patch.set_edgecolor("#fcfcfb")
        patch.set_linewidth(1.5)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#8a8984")


def legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[n], edgecolor="none") for n, _ in METHODS]
    fig.legend(handles, [lab for _, lab in METHODS], loc="upper center", ncol=4, frameon=False, fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=10_000, help="queries per (setting, unit) in the box plots")
    args = ap.parse_args()
    base, ours, out = Path(args.base), Path(args.ours), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b = load(base)
    o = load(ours)[KEY + ["ours_py_us", "choice"]].rename(columns={"ours_py_us": "ours_us"})
    m = b.merge(o, on=KEY)
    print(f"{len(m)} matched rows, {m.groupby(['dataset', 'family', 'd']).ngroups} settings, "
          f"{m.dataset.nunique()} datasets; wrong answers: three-step {int((m.err_three_step > 0).sum())}, "
          f"chirho {int(m.chirho_wrong.sum())}, dowhy {int(m.dowhy_wrong.sum())}")
    by_d = summary(m, ["d"])
    by_ds = summary(m, ["dataset", "d"])
    by_d.to_csv(out / "summary_by_d.csv", index=False)
    by_ds.to_csv(out / "summary_by_dataset.csv", index=False)
    m.to_csv(out / "merged_rows.csv", index=False)
    pd.set_option("display.width", 250)
    print(by_d.round(1).to_string(index=False))
    lat = latencies(m, base, ours, args.cap)
    dims = sorted(m.d.unique())

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    style(ax)
    for i, d in enumerate(dims):
        pos = [i * 5 + j for j in range(4)]
        boxes(ax, [lat[(d, n)] for n, _ in METHODS], pos, [n for n, _ in METHODS], 0.8)
    ax.set_xticks([i * 5 + 1.5 for i in range(len(dims))], [f"d = {d}" for d in dims])
    ax.set_ylabel("latency per query (µs)", color="#52514e", fontsize=9)
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"box_by_d.{ext}", dpi=200)

    sets = sorted(m.dataset.unique())
    fig, axes = plt.subplots(len(dims), 1, figsize=(10.0, 2.2 * len(dims)), sharex=True)
    for ax, d in zip(np.atleast_1d(axes), dims):
        style(ax)
        for i, ds in enumerate(sets):
            names = [n for n, _ in METHODS if (ds, d, n) in lat]
            boxes(ax, [lat[(ds, d, n)] for n in names], [i * 5 + j for j in range(len(names))], names, 0.8)
        ax.set_ylabel(f"d = {d}  (µs)", color="#52514e", fontsize=9)
    ax.set_xticks([i * 5 + 1.5 for i in range(len(sets))], sets, rotation=20, fontsize=8)
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"box_by_dataset.{ext}", dpi=200)


if __name__ == "__main__":
    main()
