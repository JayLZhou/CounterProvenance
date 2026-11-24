"""Numbers of Section 7 with the RF family replaced by LightGBM RF (mode new) or as in the paper (mode old, to check the definitions).
Definitions copied from sum4.py / sum6.py / exp_overall.tex: arithmetic means; TWIN = full_us; baselines = mean_us."""
import glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
W = Path("/mnt/data/yingli/counterprovenance")
H, LEAN, DEV, LRF = W / "repo_hybrid/results", W / "repo_lean/results", W / "repo_dev2/results", W / "repo_lrf/results"
FIN, FINL = W / "repo_final/results", W / "repo_final_lrf/results"   # the final index: CellForest runs only
NEW = sys.argv[1] == "new"
ORDER = "concrete light_tunnel ailerons pol elevators jdk baf mslr postgres census1990 susy methane higgs airline nyc_taxi".split()
ABBR = "CON LT AIL POL ELE JDK BAF MSLR PG CEN SUSY MET HIG AIR NYC".split()
FAMS = ["GBR", "HGB", "RF", "XGB", "LGBM"]
GB = 1e9


def csvs(pats, rf_pats=None):
    """Rows of the csv files; in mode new, RF rows come only from rf_pats."""
    out = []
    for p in pats:
        for f in glob.glob(str(p)):
            if NEW and is_rf(f):
                continue
            out.append((f, pd.read_csv(f)))
    if NEW and rf_pats:
        for p in rf_pats:
            for f in glob.glob(str(p)):
                out.append((f, pd.read_csv(f)))
    return out


def is_rf(f):
    b = os.path.basename(f)
    return "_RF_" in b or b.endswith("_RF.csv")


def load(pats, rf_pats, key):
    got = {}
    for p in pats:
        for f in glob.glob(str(p)):
            if NEW and is_rf(f):
                continue
            for r in pd.read_csv(f).to_dict("records"):
                got[(r["dataset"], r["family"], int(r["d"]))] = (r, f)
    if NEW:
        for p in rf_pats or []:
            for f in glob.glob(str(p)):
                for r in pd.read_csv(f).to_dict("records"):
                    got[(r["dataset"], r["family"], int(r["d"]))] = (r, f)
    return got


def builds(pats, rf_pats):
    got = {}
    for p in pats:
        for f in glob.glob(str(p)):
            if NEW and is_rf(f):
                continue
            t = pd.read_csv(f)
            if "bytes" in t.columns and (t.status == "ok").all():
                got[(t.dataset[0], t.family[0])] = dict(bytes=t.bytes.sum(), build_s=t.build_s.sum(), groups=t.groups.max(), g=tuple(t.groups))
    if NEW:
        for p in rf_pats or []:
            for f in glob.glob(str(p)):
                t = pd.read_csv(f)
                if "bytes" in t.columns and (t.status == "ok").all():
                    got[(t.dataset[0], t.family[0])] = dict(bytes=t.bytes.sum(), build_s=t.build_s.sum(), groups=t.groups.max(), g=tuple(t.groups))
    return got


# ---------------- Exp.1
tw = load([FIN / "main/*_twin.csv"], [FINL / "main/*_RF_d*_twin.csv"], "full_us")
B = {b: load([DEV / f"main/*_d*_{b}.csv"], [LRF / f"main/*_RF_d*_{b}.csv"], "mean_us") for b in ("three_step", "chirho", "dowhy")}
rows = []
for k, (r, f) in tw.items():
    row = dict(ds=k[0], fam=k[1], d=k[2], twin=r["full_us"], groups=r["groups"], tw_file=f)
    for b, v in B.items():
        if k in v:
            row[b] = v[k][0]["mean_us"]; row[b + "_q"] = v[k][0]["queries"]; row[b + "_file"] = v[k][1]
    rows.append(row)
S = pd.DataFrame(rows)
print("settings", len(S), "per family", S.fam.value_counts().to_dict(), "missing baselines", S[["three_step", "chirho", "dowhy"]].isna().sum().to_dict())
print("queries not 10000:", {b: int((S[b + "_q"] != 10000).sum()) for b in B})
m = S[["twin", "three_step", "chirho", "dowhy"]].mean()
print("EXP1 overall mean us: twin %.1f three %.1f chirho %.1f dowhy %.1f | ratio %.0f %.0f %.0f" % (m.twin, m.three_step, m.chirho, m.dowhy, m.three_step / m.twin, m.chirho / m.twin, m.dowhy / m.twin))
print("EXP1 twin per setting min %.1f max %.1f | min speedup per setting %.0f %.0f %.0f" % (S.twin.min(), S.twin.max(), (S.three_step / S.twin).min(), (S.chirho / S.twin).min(), (S.dowhy / S.twin).min()))
print("EXP1 per table (x, three, chirho, dowhy, twin, label three/twin):")
for i, ds in enumerate(ORDER):
    t = S[S.ds == ds][["three_step", "chirho", "dowhy", "twin"]].mean()
    print(f"  {i + 1} & {t.three_step:.2f} & {t.chirho:.2f} & {t.dowhy:.2f} & {t.twin:.2f}\\\\  % {ABBR[i]}   label {t.three_step / t.twin:,.0f}")
print("EXP1 per family:")
pf = S.groupby("fam")[["twin", "three_step", "chirho", "dowhy", "groups"]].mean()
pf["r_three"], pf["r_chirho"], pf["r_dowhy"] = pf.three_step / pf.twin, pf.chirho / pf.twin, pf.dowhy / pf.twin
print(pf.round(1).to_string())
print("EXP1 examples: AIR GBR %.1f, AIL GBR %.1f" % (S[(S.ds == "airline") & (S.fam == "GBR")].twin.mean(), S[(S.ds == "ailerons") & (S.fam == "GBR")].twin.mean()))
print("EXP1 by d (x, three, chirho, dowhy, twin):")
for i, d in enumerate([1, 2, 4, 8, 16]):
    t = S[S.d == d][["three_step", "chirho", "dowhy", "twin"]].mean()
    print(f"  {i + 1} & {t.three_step:.2f} & {t.chirho:.2f} & {t.dowhy:.2f} & {t.twin:.2f}\\\\  % d={d}")
d16 = S[S.ds.isin(["ailerons", "census1990", "mslr", "pol"])]
print("EXP1 d16 tables twin by d:", d16.groupby("d").twin.mean().round(1).to_dict(), "dowhy by d:", (d16.groupby("d").dowhy.mean() / 1e3).round(0).to_dict(),
      "| twin d<=4 %.1f d=8 %.1f" % (S[S.d <= 4].twin.mean(), S[S.d == 8].twin.mean()))
print("EXP1 per table and d (d_line):")
for i, ds in enumerate(ORDER):
    print(f"  % {ABBR[i]}")
    for d, t in S[S.ds == ds].groupby("d")[["three_step", "chirho", "dowhy", "twin"]].mean().iterrows():
        print(f"  {d} & {t.three_step:.2f} & {t.chirho:.2f} & {t.dowhy:.2f} & {t.twin:.2f}\\\\")
# exactness
mx = {b: 0.0 for b in B}; rel = {b: 0.0 for b in B}; nd = {b: 0 for b in B}; tot = {b: 0 for b in B}; where = {}
for _, r in S.iterrows():
    a = np.load(r.tw_file[:-4] + ".npy")
    for b in B:
        if isinstance(r.get(b + "_file"), str):
            x = np.load(r[b + "_file"][:-4] + ".npy"); n = min(len(a), len(x))
            dd = np.abs(a[:n] - x[:n]); s = dd / np.maximum(1.0, np.abs(x[:n]))
            if n and dd.max() > mx[b]:
                mx[b] = float(dd.max()); where[b] = (r.ds, r.fam, r.d)
            rel[b] = max(rel[b], float(s.max()) if n else 0); nd[b] += int((s > 1e-9).sum()); tot[b] += n
fx = {}
for _, r in S.iterrows():
    a = np.load(r.tw_file[:-4] + ".npy")
    for b in ("chirho", "dowhy"):
        x = np.load(r[b + "_file"][:-4] + ".npy"); n = min(len(a), len(x)); dd = np.abs(a[:n] - x[:n]); s_ = dd / np.maximum(1.0, np.abs(x[:n]))
        e = fx.setdefault((b, r.fam), [0.0, 0, 0]); e[0] = max(e[0], float(dd.max())); e[1] += int((s_ > 1e-9).sum()); e[2] += n
print("EXACT per family (max abs, share rel>1e-9):", {f"{k[0]}-{k[1]}": f"{v[0]:.1e} {v[1] / v[2]:.3f}" for k, v in sorted(fx.items())})
print("EXACT max abs diff:", {b: f"{mx[b]:.2e} at {where.get(b)}" for b in B}, "max rel:", {b: f"{rel[b]:.2e}" for b in B},
      "answers rel diff > 1e-9:", {b: f"{nd[b]} of {tot[b]}" for b in B})
fam_ex = {}
# ---------------- Exp.3
R = []
for f, t in csvs([FIN / "rss/*.csv"], [FINL / "rss/*_RF.csv"]):
    name = os.path.basename(f)[:-4]
    R.append(dict(model=name, fam=name.rsplit("_", 1)[1], real=t.real.clip(lower=0).sum(), formula=t["final"].sum(), build_s=t.build_s.sum(), peak=t.peak.max(),
                  gmax=t.groups.max()))
R = pd.DataFrame(R)
print("EXP3 models %d real %.1f GB formula %.1f GB | median %.2f GB | <10GB %d | largest %s %.1f GB" % (
    len(R), R.real.sum() / GB, R.formula.sum() / GB, R.real.median() / GB, (R.real < 10 * GB).sum(), R.loc[R.real.idxmax(), "model"], R.real.max() / GB))
print("EXP3 build median %.0f s, slowest %s %.2f h, total %.1f h, peak max %.0f GB" % (R.build_s.median(), R.loc[R.build_s.idxmax(), "model"], R.build_s.max() / 3600, R.build_s.sum() / 3600, R.peak.max() / GB))
print("EXP3 per family real GB:", (R.groupby("fam").real.sum() / GB).round(1).to_dict(), "median max groups:", R.groupby("fam").gmax.median().to_dict())
print("EXP3 RF models:", R[R.fam == "RF"].assign(real=lambda x: (x.real / GB).round(2))[["model", "real", "gmax"]].to_string(index=False).replace("\n", "; "))
# ---------------- Exp.4
print("EXP4 (d=4, mean over models):")
for n in (100, 300, 1000, 3000):
    if n == 1000:
        t4 = {k: v for k, v in tw.items() if k[2] == 4}; b4 = {b: {k: v for k, v in B[b].items() if k[2] == 4} for b in B}
    else:
        t4 = {k: v for k, v in load([FIN / f"trees/{n}/*_twin.csv"], [FINL / f"trees/{n}/*_RF_d*_twin.csv"], "full_us").items() if k[2] == 4}
        b4 = {b: {k: v for k, v in load([DEV / f"trees/{n}/*_d4_{b}.csv"], [LRF / f"trees/{n}/*_RF_d4_{b}.csv"], "mean_us").items()} for b in B}
    mt = np.mean([v[0]["full_us"] for v in t4.values()])
    line = f"  {n} TWIN models {len(t4)} mean {mt:.2f}"
    for b in B:
        common = [k for k in t4 if k in b4[b]]
        mb = np.mean([b4[b][k][0]["mean_us"] for k in common]); mtc = np.mean([t4[k][0]["full_us"] for k in common])
        line += f" | {b} n={len(common)} {mb:.2f} x{mb / mtc:,.0f}"
    print(line)
# ---------------- Exp.5
print("EXP5 beta (models with groups):")
sets = {}
for beta, (p, rp) in {"2e6": (FIN / "beta/2e6", FINL / "beta/2e6"), "2e7": (FIN / "main", FINL / "main"), "2e8": (FIN / "beta/2e8", FINL / "beta/2e8")}.items():
    sets[beta] = (load([p / "*_twin.csv"], [rp / "*_RF_d*_twin.csv"], "full_us"), builds([p / "*_build.csv"], [rp / "*_RF_build.csv"]))
models = set.intersection(*[set(b) for _, b in sets.values()])
for beta, (t, b) in sets.items():
    ks = [k for k in t if (k[0], k[1]) in models]
    print(f"  beta {beta}: models {len(models)} (RF {sum(1 for m in models if m[1] == 'RF')}) settings {len(ks)} mean us {np.mean([t[k][0]['full_us'] for k in ks]):.1f}"
          f" size {sum(b[m]['bytes'] for m in models) / GB:.1f} GB build h {sum(b[m]['build_s'] for m in models) / 3600:.2f}")
