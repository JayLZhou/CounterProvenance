import glob, json, os, numpy as np, pandas as pd
W = '/mnt/data/yingli/counterprovenance'
fs = [f for f in glob.glob(W + '/repo_final/results/main/*_twin.csv') if '_RF_' not in os.path.basename(f)] + glob.glob(W + '/repo_final_lrf/results/main/*_RF_d*_twin.csv')
rows = []
for f in fs:
    r = pd.read_csv(f).iloc[0]; rows.append(dict(ds=r.dataset, fam=r.family, d=int(r.d), us=float(r.full_us), f=f))
t = pd.DataFrame(rows)
out = {"per_model": {f"{ds}|{fam}": float(g.us.mean()) for (ds, fam), g in t.groupby(['ds', 'fam'])},
       "d_line": {ds: {int(d): float(g.us.mean()) for d, g in gd.groupby('d')} for ds, gd in t.groupby('ds')},
       "by_d": {int(d): float(g.groupby('ds').us.mean().mean()) for d, g in t.groupby('d')},
       "overall": float(t.us.mean())}
box = {}
lat0 = np.load(t.f[0][:-4] + '_lat.npy'); out['lat_unit_check'] = [float(lat0.mean()), float(t.us[0])]
for ds, g in t.groupby('ds'):
    v = np.concatenate([np.load(f[:-4] + '_lat.npy') for f in g.f])
    box[ds] = dict(n=int(len(v)), settings=int(len(g)), mean=float(v.mean()), q=[float(x) for x in np.percentile(v, [1, 25, 50, 75, 99])], max=float(v.max()))
out['box'] = box
json.dump(out, open(W + '/tmp/twin_final.json', 'w'))
print('ok', len(t), out['lat_unit_check'])
