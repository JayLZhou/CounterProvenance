"""JDK total_unit_test_time, LightGBM RF of the depth experiment at depth h, with max_bin b: the cells of the trees and the
held-out R^2. Usage: binprobe.py h b"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score
sys.path.insert(0, "/mnt/data/yingli/counterprovenance/repo_lean"); sys.path.insert(0, "/mnt/data/yingli/counterprovenance/repo_lean/experiments")
from counterprovenance.scm import parents_of
from counterprovenance import from_model
h, b = int(sys.argv[1]), int(sys.argv[2])
d0 = Path("/mnt/data/yingli/counterprovenance/data/causal/jdk"); node = "total_unit_test_time"
edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
pa = sorted(parents_of(edges)[node])
df = pd.read_parquet(d0 / "data.parquet"); df.columns = [str(c) for c in df.columns]
data = df.sample(n=50_000, random_state=0) if len(df) > 50_000 else df
idx = np.random.default_rng(1).permutation(len(data)); cut = int(0.8 * len(data))
X, y = data[pa].to_numpy(float), data[node].to_numpy(float)
m = lgb.LGBMRegressor(boosting_type="rf", n_estimators=1000, num_leaves=int(min(2 ** h, 131072)), max_depth=h, max_bin=b, bagging_freq=1,
                      bagging_fraction=0.632, feature_fraction_bynode=0.33, random_state=0, verbose=-1, n_jobs=4)
t0 = time.perf_counter(); m.fit(X[idx[:cut]], y[idx[:cut]]); fit_s = time.perf_counter() - t0
r2 = r2_score(y[idx[cut:]], m.predict(X[idx[cut:]]))
ens = from_model(m)
G = np.array([np.prod([len(set(np.asarray(t.threshold)[np.asarray(t.feature) == a].tolist())) + 1.0 for a in range(len(pa))]) for t in ens.trees])
L = np.mean([int(np.sum(np.asarray(t.feature) < 0)) for t in ens.trees])
print(f"h={h} max_bin={b}: R2 {r2:.4f} | leaves/tree {L:.0f} | cells per tree max {G.max():.3g} mean {G.mean():.3g} | trees over 1e9: {(G > 1e9).sum()} | total {G.sum():.3g} | fit {fit_s:.0f}s", flush=True)
