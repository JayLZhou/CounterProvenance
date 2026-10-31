#!/usr/bin/env python3
"""Tables and figures of the main experiment (d = 1, 2, 4, 8).

The baselines (three-step, ChiRho, DoWhy-GCM) come from ``--base`` (``results/scm_full_*.csv`` and
``results/scm_lat/*.npz``), Tessera from ``--ours`` (an ``--ours-only`` run: ``results/scm_full_*.csv``
and ``results/scm_lat_ours/*.npz``).  Rows are matched on (dataset, family, d, unit).
Output in ``--out``:
  summary_by_d.csv, summary_by_dataset.csv, merged_rows.csv
  latency_by_d.{pdf,png}       per-query latency boxes, one panel per d, speedups written above the boxes
  speedup_heatmap.{pdf,png}    geometric-mean speedup per (dataset, d), one panel per baseline
  latency_by_dataset.{pdf,png} per-query latency boxes per dataset, one row per d
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, NullFormatter

METHODS = [("ours", "Tessera"), ("three_step", "Three-step"), ("chirho", "ChiRho"), ("dowhy", "DoWhy-GCM")]
COLORS = {"ours": "#2a78d6", "three_step": "#eb6834", "chirho": "#1baf7a", "dowhy": "#eda100"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4", "#ffffff"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
KEY = ["dataset", "family", "d", "unit"]

plt.rcParams.update({
    "font.family": ["Linux Biolinum O", "Inter", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(root: Path, ok: bool = True) -> pd.DataFrame:
    """All result rows of a run; a unit re-run in its own CSV (``--unit-ids``) replaces the older row."""
    files = sorted(glob.glob(str(root / "results" / "scm_full_*.csv")), key=lambda f: Path(f).stat().st_mtime)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.d.le(8)].drop_duplicates(KEY, keep="last")
    return df[df.status.eq("ok")] if ok else df


def gmean(x) -> float:
    return float(np.exp(np.log(np.asarray(x, dtype=float)).mean()))


def times(x: float) -> str:
    return f"{x / 1000:.1f}k×" if x >= 1000 else f"{x:.0f}×"


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


def time_axis(ax, lo=1, hi=1e6):
    """Log latency axis labelled in us / ms / s, with a light grid at the decades."""
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    decades = [10.0 ** k for k in range(int(np.log10(lo)), int(np.log10(hi)) + 1)]
    names = {1: "1 µs", 10: "10 µs", 100: "100 µs", 1e3: "1 ms", 1e4: "10 ms", 1e5: "100 ms", 1e6: "1 s"}
    ax.yaxis.set_major_locator(FixedLocator(decades))
    ax.set_yticklabels([names.get(v, "") for v in decades])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_boxes(ax, data, positions, names, width=0.62):
    """Light-filled boxes with a colored outline and a strong colored median; whiskers at p5 / p95."""
    bp = ax.boxplot(data, positions=positions, widths=width, showfliers=False, patch_artist=True, whis=(5, 95),
                    showcaps=False)
    for i, n in enumerate(names):
        c = COLORS[n]
        bp["boxes"][i].set(facecolor=matplotlib.colors.to_rgba(c, 0.22), edgecolor=c, linewidth=1.1)
        bp["medians"][i].set(color=c, linewidth=2.2, solid_capstyle="butt")
        for w in bp["whiskers"][2 * i: 2 * i + 2]:
            w.set(color=c, linewidth=1.0)
    return bp


def latency_by_d(lat, dims, by_d, out):
    fig, axes = plt.subplots(1, len(dims), figsize=(7.0, 2.45), sharey=True)
    for j, (ax, d) in enumerate(zip(axes, dims)):
        time_axis(ax, 1, 2e6)
        names = [n for n, _ in METHODS]
        draw_boxes(ax, [lat[(d, n)] for n in names], range(4), names)
        row = by_d[by_d.d == d].iloc[0]
        for i, n in enumerate(names[1:], start=1):
            top = np.percentile(lat[(d, n)], 95)
            ax.text(i, top * 1.35, times(row[f"speedup_{n}"]), ha="center", va="bottom", fontsize=7.5,
                    color=INK2, fontweight="bold")
        med = np.median(lat[(d, "ours")])
        ax.text(0, np.percentile(lat[(d, "ours")], 95) * 1.35, f"{med:.0f} µs" if med >= 10 else f"{med:.1f} µs",
                ha="center", va="bottom", fontsize=7.5, color=COLORS["ours"], fontweight="bold")
        ax.set_xticks(range(4), ["Tessera", "3-step", "ChiRho", "DoWhy"], rotation=0)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(f"d = {d}", pad=6, fontweight="bold")
        if j == 0:
            ax.set_ylabel("latency per query")
    fig.text(0.995, 0.01, "box: p25–p75, line: median, whiskers: p5–p95; labels: geometric-mean speedup of Tessera",
             ha="right", va="bottom", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"latency_by_d.{ext}", dpi=300)
    plt.close(fig)


def speedup_heatmap(by_ds, dims, sets, labels, out):
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    vals = [by_ds[f"speedup_{b}"] for b, _ in METHODS[1:]]
    norm = LogNorm(vmin=max(10.0, min(v.min() for v in vals)), vmax=max(v.max() for v in vals))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.1), sharey=True)
    for ax, (b, lab) in zip(axes, METHODS[1:]):
        M = np.full((len(sets), len(dims)), np.nan)
        for r in by_ds.itertuples(index=False):
            M[sets.index(r.dataset), dims.index(r.d)] = getattr(r, f"speedup_{b}")
        ax.pcolormesh(M, cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2)
        for i in range(len(sets)):
            for j in range(len(dims)):
                if np.isfinite(M[i, j]):
                    dark = norm(M[i, j]) > 0.55
                    ax.text(j + 0.5, i + 0.5, times(M[i, j]), ha="center", va="center", fontsize=7,
                            color="#ffffff" if dark else INK)
        ax.set_xticks(np.arange(len(dims)) + 0.5, [f"d={d}" for d in dims])
        ax.set_yticks(np.arange(len(sets)) + 0.5, labels)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"vs. {lab}", pad=6, fontweight="bold")
    fig.text(0.995, 0.01, "geometric-mean speedup of Tessera over 5 model families × 3 units, 10,000 queries each",
             ha="right", va="bottom", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"speedup_heatmap.{ext}", dpi=300)
    plt.close(fig)


def latency_by_dataset(lat, dims, sets, labels, out):
    fig, axes = plt.subplots(len(dims), 1, figsize=(7.0, 1.45 * len(dims) + 0.4), sharex=True)
    names = [n for n, _ in METHODS]
    for ax, d in zip(axes, dims):
        time_axis(ax, 1, 2e6)
        for i, ds in enumerate(sets):
            have = [n for n in names if (ds, d, n) in lat]
            draw_boxes(ax, [lat[(ds, d, n)] for n in have], [i * 5 + names.index(n) for n in have], have, width=0.75)
        ax.set_ylabel(f"d = {d}", fontweight="bold", color=INK)
        ax.set_xlim(-1, len(sets) * 5 - 1)
        ax.tick_params(axis="x", length=0)
    axes[-1].set_xticks([i * 5 + 1.5 for i in range(len(sets))], [s.replace(" (", "\n(") for s in labels], fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=matplotlib.colors.to_rgba(COLORS[n], 0.22),
                             edgecolor=COLORS[n], linewidth=1.1) for n in names]
    fig.legend(handles, [lab for _, lab in METHODS], loc="upper center", ncol=4, frameon=False, fontsize=8,
               handlelength=1.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"latency_by_dataset.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--data", required=True, help="causal datasets (for row counts in the labels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=10_000, help="queries per (setting, unit) in the box plots")
    args = ap.parse_args()
    base, ours, out = Path(args.base), Path(args.ours), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b = load(base)
    o_all = load(ours, ok=False)
    bad = o_all[~o_all.status.eq("ok")]
    if len(bad):
        print(f"TWIN over budget in {len(bad)} rows:")
        print(bad.groupby(["dataset", "family", "d"]).size().to_string())
    o = o_all[o_all.status.eq("ok")][KEY + ["ours_py_us", "choice"]].rename(columns={"ours_py_us": "ours_us"})
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

    rows = {ds: json.loads((Path(args.data) / ds / "meta.json").read_text())["rows"] for ds in m.dataset.unique()}
    sets = sorted(rows, key=rows.get)  # small to large tables
    fmt = lambda n: f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(n)
    labels = [f"{ds} ({fmt(rows[ds])})" for ds in sets]
    dims = sorted(int(d) for d in m.d.unique())
    lat = latencies(m, base, ours, args.cap)
    latency_by_d(lat, dims, by_d, out)
    speedup_heatmap(by_ds, dims, sets, labels, out)
    latency_by_dataset(lat, dims, sets, labels, out)


if __name__ == "__main__":
    main()
