#!/usr/bin/env python3
"""Accuracy of the outcome's random forest at depth 6 (the main experiment) against depth 10, on every table.

The forest is fitted as in the main experiment (1,000 trees, feature fraction 0.33, up to 50,000 sampled rows) on the
parents of the outcome, and tested on up to 20,000 other rows.  Output: ``results/rf_depth/<dataset>.csv``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from common import RESULTS
from counterprovenance.scm import parents_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--depths", default="6,10")
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    pa = sorted(parents_of(edges)[y])
    rng = np.random.default_rng(0)
    order = rng.permutation(len(df))
    fit = df.iloc[order[:min(50_000, len(df) // 2 if len(df) < 100_000 else 50_000)]]
    test = df.iloc[order[len(fit):len(fit) + 20_000]]
    out = RESULTS / "rf_depth"
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for depth in [int(x) for x in args.depths.split(",")]:
        t0 = time.perf_counter()
        model = RandomForestRegressor(n_estimators=args.trees, max_depth=depth, max_features=0.33, random_state=0,
                                      n_jobs=args.threads).fit(fit[pa].to_numpy(np.float64), fit[y].to_numpy(np.float64))
        fit_s = time.perf_counter() - t0
        pred = model.predict(test[pa].to_numpy(np.float64))
        leaves = int(np.mean([e.tree_.n_leaves for e in model.estimators_]))
        got.append(dict(dataset=args.dataset, depth=depth, parents=len(pa), fit_rows=len(fit), test_rows=len(test),
                        r2=r2_score(test[y], pred), rmse=mean_squared_error(test[y], pred) ** 0.5, mean_leaves=leaves, fit_s=fit_s))
        pd.DataFrame(got).to_csv(out / f"{args.dataset}.csv", index=False)
        print(got[-1], flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
