#!/usr/bin/env python3
"""Quick look: Tessera vs three-step online (no caching), ChiRho and DoWhy-GCM, pure Python.

For each ``dataset:family:d``: fit, bind, index; time ``--queries`` queries for
Tessera and the three-step online, ``--lib-queries`` for ChiRho and DoWhy-GCM; check
every answer against the exact pure-Python propagation.  Output:
``results/quick_baselines.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chirho_baseline import ScmChiRho
from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmOnlinePy, ScmThreeStepPy
from scm_bench import choose_treatments, zoo
from scm_full import dowhy_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--lib-queries", type=int, default=100)
    args = ap.parse_args()
    rows, fitted = [], {}
    for spec in args.settings.split(","):
        ds, fam, d = spec.split(":")
        d = int(d)
        if (ds, fam) not in fitted:
            d0 = Path(args.data) / ds
            meta = json.loads((d0 / "meta.json").read_text())
            df = pd.read_parquet(d0 / "data.parquet")
            df.columns = [str(c) for c in df.columns]
            edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
            outcome = str(meta["target"])
            scm = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]],
                          zoo(1000, 1)[fam], max_rows=50_000, seed=0, n_jobs=4)
            gcm, cg = dowhy_model(scm, edges)
            fitted[(ds, fam)] = (df, outcome, scm, gcm, cg)
        df, outcome, scm, gcm, cg = fitted[(ds, fam)]
        rng = np.random.default_rng(2028)
        unit_row = df.iloc[[int(rng.integers(len(df)))]]
        unit = unit_row.iloc[0].to_dict()
        T = choose_treatments(scm, outcome, d)
        if T is None:
            continue
        start = time.perf_counter()
        bound = bind_scm(scm, unit, T, outcome)
        index = ProvenanceIndex(bound)
        build_s = time.perf_counter() - start
        Q = rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d))
        truth = np.array([ScmOnlinePy(scm, unit, T, outcome).query(t)[0] for t in Q[:1]])  # warm
        ref = ScmOnlinePy(scm, unit, T, outcome)
        truth = np.array([ref.query(t)[0] for t in Q])
        rel = lambda got, k: float(np.max(np.abs(np.asarray(got) - truth[:k]) / np.maximum(1.0, np.abs(truth[:k]))))
        three = ScmThreeStepPy(scm, unit, T, outcome)
        chirho = ScmChiRho(scm, unit, T, outcome)
        k = min(args.lib_queries, len(Q))
        ivs = [{name: (lambda _v, value=float(t[j]): value) for j, name in enumerate(T)} for t in Q[:k]]
        row = dict(dataset=ds, family=fam, d=d, mediators=len(ref.mechs) - 1, boxes=len(bound.boxes), build_s=build_s,
                   err_ours=rel([index.query(t).value for t in Q], len(Q)),
                   err_three=rel([three.query(t)[0] for t in Q], len(Q)),
                   err_chirho=rel([chirho.query(t) for t in Q[:k]], k),
                   err_dowhy=rel([float(gcm.counterfactual_samples(cg, iv, observed_data=unit_row)[outcome].iloc[0])
                                  for iv in ivs], k))
        row["ours_us"] = per_call_seconds(lambda: [index.query(t) for t in Q], len(Q)) * 1e6
        row["three_us"] = per_call_seconds(lambda: [three.query(t) for t in Q], len(Q), repeats=1) * 1e6
        row["chirho_us"] = per_call_seconds(lambda: [chirho.query(t) for t in Q[:k]], k, repeats=1) * 1e6
        row["dowhy_us"] = per_call_seconds(
            lambda: [gcm.counterfactual_samples(cg, iv, observed_data=unit_row) for iv in ivs], k, repeats=1) * 1e6
        rows.append(row)
        write_csv("quick_baselines.csv", rows)
        print(f"{ds} {fam} d={d} med={row['mediators']} | ours {row['ours_us']:.1f}us | three-step "
              f"{row['three_us']:.0f}us ({row['three_us'] / row['ours_us']:.0f}x) | chirho {row['chirho_us'] / 1e3:.1f}ms "
              f"({row['chirho_us'] / row['ours_us']:,.0f}x) | dowhy {row['dowhy_us'] / 1e3:.1f}ms "
              f"({row['dowhy_us'] / row['ours_us']:,.0f}x) | err {row['err_ours']:.0e} {row['err_three']:.0e} "
              f"{row['err_chirho']:.0e} {row['err_dowhy']:.0e}", flush=True)


if __name__ == "__main__":
    main()
