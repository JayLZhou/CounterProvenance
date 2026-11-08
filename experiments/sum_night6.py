"""Summaries of the night of 2026-11-08 (arithmetic means; sizes in GB)."""
import glob
import os
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
H, LEAN = W / "repo_hybrid/results", W / "repo_lean/results"
GB = 1e9


def twin(dirs):
    """TWIN rows of the given result folders, later folders overriding earlier ones per (dataset, family, d)."""
    got = {}
    for d in dirs:
        for f in glob.glob(str(d / "*_twin.csv")):
            for r in pd.read_csv(f).to_dict("records"):
                got[(r["dataset"], r["family"], int(r["d"]))] = r
    return got


def builds(dirs):
    got = {}
    for d in dirs:
        for f in glob.glob(str(d / "*_build.csv")):
            t = pd.read_csv(f)
            if "bytes" in t.columns and (t.status == "ok").all():
                got[(t.dataset[0], t.family[0])] = dict(bytes=t.bytes.sum(), build_s=t.build_s.sum(), groups=t.groups.max())
    return got


print("== real memory of the final (lean) index, results/rss")
rows = []
for f in sorted(glob.glob(str(LEAN / "rss/*.csv"))):
    t = pd.read_csv(f)
    rows.append(dict(model=os.path.basename(f)[:-4], formula=t.formula.sum(), real=t.real.clip(lower=0).sum(), build_s=t.build_s.sum(),
                     pft=t.pft_nodes.sum() if "pft_nodes" in t else None, leaves=t.all_leaves.sum() if "all_leaves" in t else None,
                     grid_all_max=t.grid_all.max() if "grid_all" in t else None))
r = pd.DataFrame(rows)
if len(r):
    print(f"models {len(r)}  formula {r.formula.sum() / GB:.1f}  real {r.real.sum() / GB:.1f}  ratio {r.real.sum() / r.formula.sum():.2f}  build h {r.build_s.sum() / 3600:.1f}")
    print(r.sort_values("real", ascending=False).head(6).assign(formula=lambda x: (x.formula / GB).round(2), real=lambda x: (x.real / GB).round(2))[["model", "formula", "real"]].to_string(index=False))
    a = r.dropna(subset=["pft"])
    if len(a):
        nomst = a.formula - 8 * a.pft + 4 * a.leaves
        print(f"== ablation on {len(a)} models: TWIN {a.formula.sum() / GB:.1f} GB, without MST (all leaves per cell) {nomst.sum() / GB:.1f} GB, ratio {nomst.sum() / a.formula.sum():.1f}x;"
              f" per model median {(nomst / a.formula).median():.1f}x max {(nomst / a.formula).max():.1f}x")
    g = r.dropna(subset=["grid_all_max"])
    if len(g):
        print(f"== without grouping: grid of one group over all trees, max over mechanisms: >2^32 on {(g.grid_all_max > 2**32).sum()} of {len(g)} models;"
              f" map alone over 1 TB (4 B per grid cell) on {(4 * g.grid_all_max > 1e12).sum()}")

print("== beta (one group up to 5e7 grid, else greedy groups under beta), on the models with groups")
fin = twin([H / "final"])
fb = builds([H / "final"])
sets = {}
for beta, d in (("2e6", LEAN / "beta_final/2e6"), ("2e7", H / "final"), ("2e8", LEAN / "beta_final/2e8")):
    sets[beta] = (twin([d]), builds([d]))
models = set.intersection(*[set(b) for _, b in sets.values()]) & set(builds([LEAN / "beta_final/2e6"]))
for beta, (t, b) in sets.items():
    ks = [k for k in t if (k[0], k[1]) in models]
    if not ks:
        continue
    print(f"  beta {beta}: models {len(models)} settings {len(ks)} mean us {sum(t[k]['full_us'] for k in ks) / len(ks):.1f}"
          f"  size {sum(b[m]['bytes'] for m in models) / GB:.1f} GB  build h {sum(b[m]['build_s'] for m in models) / 3600:.2f}"
          f"  mean max groups {sum(b[m]['groups'] for m in models) / len(models):.0f}")

print("== kappa (the grid up to which an ensemble is one group), all 75 models; a model not rerun keeps its 5e7 result")
base_t, base_b = twin([H / "final"]), builds([H / "final"])
for kappa, dirs in (("2e6", [H / "hybrid"]), ("1e7", [H / "final_1e7", LEAN / "kappa/1e7"]), ("5e7", [H / "final"]),
                    ("1e8", [H / "one1e8", LEAN / "kappa/1e8"])):
    t, b = dict(base_t), dict(base_b)
    if kappa != "1e8":
        t, b = {}, {}
    t.update(twin(dirs))
    b.update(builds(dirs))
    print(f"  kappa {kappa}: settings {len(t)} models {len(b)} mean us {sum(v['full_us'] for v in t.values()) / max(1, len(t)):.1f}"
          f"  size {sum(v['bytes'] for v in b.values()) / GB:.1f} GB  build h {sum(v['build_s'] for v in b.values()) / 3600:.1f}")

print("== RF depth")
fs = sorted(glob.glob(str(LEAN / "rf_depth/*.csv")))
if fs:
    t = pd.concat([pd.read_csv(f) for f in fs])
    p = t.pivot(index="dataset", columns="depth", values="r2")
    print(p.round(3).to_string())
    print("mean r2", p.mean().round(3).to_dict())

print("== applications, final index")
for f in sorted(glob.glob(str(LEAN / "shard/*_twin_0.csv"))):
    t = pd.read_csv(f)
    old = H / "shard" / os.path.basename(f)
    o = pd.read_csv(old) if old.exists() else None
    cols = [c for c in t.columns if c.endswith("_s") or c.endswith("_us") or c in ("rows", "queries")]
    print(os.path.basename(f)[:-11], t[cols].iloc[0].round(2).to_dict(), "| old", o[cols].iloc[0].round(2).to_dict() if o is not None else None)
