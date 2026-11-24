src = open('/mnt/data/yingli/counterprovenance/paper_lrf.py').read()
s = src
s = s.replace('H, LEAN, DEV, LRF = W / "repo_hybrid/results", W / "repo_lean/results", W / "repo_dev2/results", W / "repo_lrf/results"',
              'H, LEAN, DEV, LRF = W / "repo_hybrid/results", W / "repo_lean/results", W / "repo_dev2/results", W / "repo_lrf/results"\n'
              'FIN, FINL = W / "repo_final/results", W / "repo_final_lrf/results"   # the final index: CellForest runs only')
s = s.replace('tw = load([H / "final/*_twin.csv"], [LRF / "main/*_RF_d*_twin.csv"], "full_us")', 'tw = load([FIN / "main/*_twin.csv"], [FINL / "main/*_RF_d*_twin.csv"], "full_us")')
s = s.replace('for f, t in csvs([LEAN / "rss/*.csv"], [LRF / "rss/*_RF.csv"]):', 'for f, t in csvs([FIN / "rss/*.csv"], [FINL / "rss/*_RF.csv"]):')
s = s.replace('R.append(dict(model=name, fam=name.rsplit("_", 1)[1], real=t.real.clip(lower=0).sum(), formula=t.formula.sum(),',
              'R.append(dict(model=name, fam=name.rsplit("_", 1)[1], real=t.real.clip(lower=0).sum(), formula=t["final"].sum(),')
s = s.replace('t4 = {k: v for k, v in load([H / f"trees_pairs/{n}/*_twin.csv"], [LRF / f"trees/{n}/*_RF_d*_twin.csv"], "full_us").items() if k[2] == 4}',
              't4 = {k: v for k, v in load([FIN / f"trees/{n}/*_twin.csv"], [FINL / f"trees/{n}/*_RF_d*_twin.csv"], "full_us").items() if k[2] == 4}')
s = s.replace('for beta, (p, rp) in {"2e6": (LEAN / "beta_final/2e6", LRF / "beta_final/2e6"), "2e7": (H / "final", LRF / "main"), "2e8": (LEAN / "beta_final/2e8", LRF / "beta_final/2e8")}.items():',
              'for beta, (p, rp) in {"2e6": (FIN / "beta/2e6", FINL / "beta/2e6"), "2e7": (FIN / "main", FINL / "main"), "2e8": (FIN / "beta/2e8", FINL / "beta/2e8")}.items():')
s = s.replace('models = set.intersection(*[set(b) for _, b in sets.values()]) & set(builds([LEAN / "beta_final/2e6/*_build.csv"], [LRF / "beta_final/2e6/*_RF_build.csv"]))',
              'models = set.intersection(*[set(b) for _, b in sets.values()])')
s = s[:s.index('print("EXP5 kappa:")')]
assert s.count("FIN") >= 8, s.count("FIN")
open('/mnt/data/yingli/counterprovenance/paper_final.py', 'w').write(s)
print("ok", s.count("FIN"))
