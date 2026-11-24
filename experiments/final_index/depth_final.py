#!/usr/bin/env python3
"""Effect of the depth: query time of TWIN and ThreeStep against the depth h of the trees, for the five families of the
main experiment with their depth set to h, on one tree ensemble (the outcome of a node from its k parents).
1,000 trees, up to 50,000 rows, 100 units with 10 queries each, all parents intervened, one thread.
An ensemble with a tree over 1e9 blocks or 5e10 blocks in total is reported as infeasible without building."""
import json, os, sys, time, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
warnings.filterwarnings("ignore")
ROOT = "/mnt/data/yingli/counterprovenance"; REPO = ROOT + "/repo_final"; OUT = ROOT + "/depth_final"
sys.path[:0] = [REPO, REPO + "/experiments"]
import numpy as np, pandas as pd
from pathlib import Path
from counterprovenance.scm import fit_scm, parents_of
from counterprovenance.scm_baselines import ScmThreeStepPy
from mech_proto import MechanismIndex


def family(fam, h):
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    import lightgbm as lgb, xgboost as xgb
    leaves = int(min(2 ** h, 131072))
    return {
        "GBR": lambda: GradientBoostingRegressor(n_estimators=1000, max_depth=h, learning_rate=0.05, random_state=0),
        "HGB": lambda: HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=h, max_leaf_nodes=None,
                                                     early_stopping=False, random_state=0),
        "RF": lambda: lgb.LGBMRegressor(boosting_type="rf", n_estimators=1000, num_leaves=leaves, max_depth=h, max_bin=255,
                                        bagging_freq=1, bagging_fraction=0.632, feature_fraction_bynode=0.33, random_state=0,
                                        verbose=-1, n_jobs=1),
        "XGB": lambda: xgb.XGBRegressor(n_estimators=1000, max_depth=h, learning_rate=0.05, tree_method="hist", random_state=0, n_jobs=1),
        "LGBM": lambda: lgb.LGBMRegressor(n_estimators=1000, num_leaves=leaves, max_depth=h, learning_rate=0.05, random_state=0,
                                          verbose=-1, n_jobs=1),
    }[fam]


def depth_of(t):
    left, right = np.asarray(t.left), np.asarray(t.right)
    best = 0; stack = [(0, 0)]
    while stack:
        n, d = stack.pop()
        if left[n] < 0 and right[n] < 0 or t.feature[n] < 0:
            best = max(best, d)
        else:
            stack.append((int(left[n]), d + 1)); stack.append((int(right[n]), d + 1))
    return best


def save(r, out):
    tmp = str(out) + ".tmp"; pd.DataFrame([r]).to_csv(tmp, index=False); os.replace(tmp, out)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


def main(ds, node, fam, h):
    out = Path(OUT) / f"{ds}__{node}__{fam}__{h}.csv"
    d0 = Path(ROOT) / "data/causal" / ds
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    pa = sorted(parents_of(edges)[node])
    df = pd.read_parquet(d0 / "data.parquet"); df.columns = [str(c) for c in df.columns]
    df = df[pa + [node]].copy()
    r = dict(dataset=ds, node=node, family=fam, max_depth=h, k=len(pa), status="ok")
    t0 = time.perf_counter()
    scm = fit_scm(df, [(p, node) for p in pa], node, [pa[0]], family(fam, h), max_rows=50_000, seed=0, n_jobs=1)
    r["fit_s"] = time.perf_counter() - t0
    ens = scm.ensembles[node]
    D = [depth_of(t) for t in ens.trees]
    L = [int(np.sum(np.asarray(t.feature) < 0)) for t in ens.trees]
    r.update(trees=len(ens.trees), real_depth_mean=float(np.mean(D)), real_depth_max=int(max(D)), leaves_mean=float(np.mean(L)))
    G = np.array([np.prod([len(set(np.asarray(t.threshold)[np.asarray(t.feature) == a].tolist())) + 1.0 for a in range(len(pa))]) for t in ens.trees])
    r.update(tree_blocks_mean=float(G.mean()), tree_blocks_max=float(G.max()), tree_blocks_sum=float(G.sum()))
    rng = np.random.default_rng(2028)
    rows = df.iloc[rng.integers(len(df), size=100)].reset_index(drop=True)
    lo, hi = df[pa].quantile(0.01).to_numpy(), df[pa].quantile(0.99).to_numpy()
    Q = [rng.uniform(lo, hi, size=(10, len(pa))) for _ in range(100)]
    del df
    feasible = G.max() <= 1e9
    got = None
    if not feasible:
        r["status"] = "infeasible: a tree over 1e9 grid cells, the limit of the index for one tree"
    else:
        try:
            t0 = time.perf_counter()
            mech = MechanismIndex(scm, rows.iloc[0].to_dict(), [node], 2e7, 1, 2e10, 5e7)
            r["build_s"] = time.perf_counter() - t0
            info = mech.info[node]
            r.update(N=info["boxes"], groups=info["groups"], blocks=info["cells"], index_gb=info["bytes"] / 1e9)
            order = [node]; tw = []; ab = 0.0; got = []
            for u, qs in enumerate(Q):
                unit = rows.iloc[u].to_dict()
                t0 = time.perf_counter(); obs = mech.observed(unit, order); ab += time.perf_counter() - t0
                for t in qs:
                    t0 = time.perf_counter(); v = mech.query(unit, pa, t, order, obs); tw.append(time.perf_counter() - t0); got.append(v)
            r["twin_us"] = float(np.mean(tw) * 1e6 + ab / 1000 * 1e6)
            del mech
        except Exception as e:
            r["status"] = f"twin failed: {type(e).__name__}: {str(e)[:100]}"; got = None
    # ThreeStep is unchanged: its time comes from the earlier run (depth_exp3, same models, units and queries); here it runs
    # only on the first 10 units, to check the answers
    ts = []; base = []
    for u, qs in enumerate(Q[:10]):
        three = ScmThreeStepPy(scm, rows.iloc[u].to_dict(), pa, node)
        for t in qs:
            t0 = time.perf_counter(); v = three.query(t)[0]; ts.append(time.perf_counter() - t0); base.append(float(v))
    prev = Path(ROOT) / "depth_exp3" / f"{ds}__{node}__{fam}__{h}.csv"
    r["three_us"] = float(pd.read_csv(prev).three_us[0]) if prev.exists() else float(np.mean(ts) * 1e6)
    r["three_from"] = "depth_exp3" if prev.exists() else "this run (100 queries)"
    if got is not None:
        r["max_abs_diff"] = float(np.max(np.abs(np.array(got[:len(base)], dtype=float) - np.array(base))))
        r["speedup"] = r["three_us"] / r["twin_us"]
    save(r, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
