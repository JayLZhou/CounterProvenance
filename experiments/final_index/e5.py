import glob, json, os, re, numpy as np, pandas as pd
W = '/mnt/data/yingli/counterprovenance'
def recs(dirname_pat):
    out = {}
    for d in glob.glob(W + '/repo_final_abl/results/structure_more/' + dirname_pat):
        n = os.path.basename(d); model = re.sub(r"_t\d+_b.*$", "", n)
        for f in glob.glob(d + '/*_groups.csv'):
            if os.path.getsize(f) < 5: continue   # an index without groups of several axes (all constant)
            g = pd.read_csv(f)
            if len(g): out[model] = dict(rec=float(g.changes.sum() + g.ckpt.sum()), table=float(g.celltable_leaves.sum()), groups=len(g))
    return out
# ---- Exp.6 (b): records by number of trees
main = {}
for f in glob.glob(W + '/repo_final_abl/results/structure_final/*_groups.csv'):
    if os.path.getsize(f) < 5: continue
    g = pd.read_csv(f); m = os.path.basename(f)[:-11]; main[m] = dict(rec=float(g.changes.sum() + g.ckpt.sum()), table=float(g.celltable_leaves.sum()))
byM = {1000: main}
for M in (100, 300, 3000):
    byM[M] = recs(f'*_t{M}_b2e+07_p2e+10_*')
common = set.intersection(*[set(v) for v in byM.values()])
rt = {str(M): dict(mst=sum(v[m]['rec'] for m in common), table=sum(v[m]['table'] for m in common)) for M, v in byM.items()}
for M in rt: rt[M]['total'] = rt[M]['table'] / rt[M]['mst']
print('EXP6b models', len(common), {M: round(v['total'], 1) for M, v in rt.items()})
json.dump(rt, open(W + '/tmp/records_trees_final.json', 'w'))
# ---- Exp.5: beta
corr = {}
for tag, pat in (('2e6', '*_t1000_b2e+06_p2e+10_*'), ('2e8', '*_t1000_b2e+08_p2e+10_*')):
    corr[tag] = recs(pat)
corr['2e7'] = main
def builds(sub):
    fs = [f for f in glob.glob(f'{W}/repo_final/results/{sub}/*_build.csv') if not f.endswith('_RF_build.csv')] + glob.glob(f'{W}/repo_final_lrf/results/{sub}/*_RF_build.csv')
    t = pd.concat([pd.read_csv(f) for f in fs]); t = t[t.status == 'ok'] if 'status' in t else t
    return t.groupby(['dataset', 'family']).agg(bytes=('bytes', 'sum'), build=('build_s', 'sum'), groups=('groups', 'sum'))
def twin(sub):
    fs = [f for f in glob.glob(f'{W}/repo_final/results/{sub}/*_twin.csv') if '_RF_' not in os.path.basename(f)] + glob.glob(f'{W}/repo_final_lrf/results/{sub}/*_RF_d*_twin.csv')
    return pd.DataFrame([pd.read_csv(f).iloc[0] for f in fs]).groupby(['dataset', 'family']).full_us.mean()
B = {'2e6': builds('beta/2e6'), '2e7': builds('main'), '2e8': builds('beta/2e8')}
T = {'2e6': twin('beta/2e6'), '2e7': twin('main'), '2e8': twin('beta/2e8')}
for tag in B:
    B[tag]['size'] = [b - 12 * corr[tag].get(f'{d}_{f}', {}).get('rec', np.nan) for (d, f), b in B[tag].bytes.items()]
models = set.intersection(*[set(B[t].dropna().index) for t in B]) & set.intersection(*[set(T[t].index) for t in T])
models = {m for m in models if B['2e7'].loc[m, 'groups'] > len(pd.read_csv(f'{W}/repo_final{"_lrf" if m[1] == "RF" else ""}/results/main/{m[0]}_{m[1]}_build.csv'))}  # models with groups
ds_all = sorted({d for d, _ in models if all((d, f) in models for f in ('GBR', 'HGB', 'LGBM', 'RF', 'XGB'))})
print('EXP5 models with groups and all budgets:', len(models), 'datasets complete:', len(ds_all), ds_all)
for tag in ('2e6', '2e7', '2e8'):
    ms = sorted(models)
    print(tag, 'mean us %.1f' % np.mean([T[tag][m] for m in ms]), 'size GB %.2f' % (sum(B[tag].loc[m, 'size'] for m in ms) / 1e9), 'build h %.2f' % (sum(B[tag].loc[m, 'build'] for m in ms) / 3600), 'groups', int(sum(B[tag].loc[m, 'groups'] for m in ms)))
per = {}
for d in sorted({m[0] for m in models}):
    ms = [m for m in models if m[0] == d]
    per[d] = {tag: dict(us=float(np.mean([T[tag][m] for m in ms])), size=float(sum(B[tag].loc[m, 'size'] for m in ms) / len(ms))) for tag in B}
for d, v in per.items():
    print(d, 'us', [round(v[t]['us'], 1) for t in ('2e6', '2e7', '2e8')], 'size MB', [round(v[t]['size'] / 1e6) for t in ('2e6', '2e7', '2e8')])
json.dump(per, open(W + '/tmp/beta_final.json', 'w'))
