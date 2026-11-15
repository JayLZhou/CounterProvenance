#!/usr/bin/env python3
"""Accuracy of the models of the main experiment: the function of the outcome, refit with the same family and settings on
80% of the rows that the main experiment fits on (up to 50,000 sampled rows, seed 0), R^2 on the other 20%.
Usage: python r2_models.py <dataset> <family> (run from the experiments directory of repo_lean, or repo_lrf for RF).
Output: results/r2/<dataset>_<family>.csv"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from common import RESULTS
from counterprovenance.scm import parents_of
from scm_bench import zoo

ds, fam = sys.argv[1], sys.argv[2]
d0 = Path("/mnt/data/yingli/counterprovenance/data/causal") / ds
meta = json.loads((d0 / "meta.json").read_text())
df = pd.read_parquet(d0 / "data.parquet"); df.columns = [str(c) for c in df.columns]
edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
y = str(meta["target"]); pa = sorted(parents_of(edges)[y], key=lambda p: list(df.columns).index(p))
data = df.sample(n=50_000, random_state=0) if len(df) > 50_000 else df
idx = np.random.default_rng(1).permutation(len(data)); cut = int(0.8 * len(data))
tr, te = data.iloc[idx[:cut]], data.iloc[idx[cut:]]
t0 = time.perf_counter()
m = zoo(1000, 4)[fam](); m.fit(tr[pa].to_numpy(dtype=float), tr[y].to_numpy(dtype=float))
fit_s = time.perf_counter() - t0
r2 = r2_score(te[y].to_numpy(dtype=float), m.predict(te[pa].to_numpy(dtype=float)))
r2_tr = r2_score(tr[y].to_numpy(dtype=float), m.predict(tr[pa].to_numpy(dtype=float)))
out = RESULTS / "r2"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([dict(dataset=ds, family=fam, outcome=y, parents=len(pa), train_rows=len(tr), test_rows=len(te), r2_test=r2, r2_train=r2_tr, fit_s=fit_s)]).to_csv(out / f"{ds}_{fam}.csv", index=False)
print(ds, fam, round(r2, 4), round(r2_tr, 4), round(fit_s, 1))
