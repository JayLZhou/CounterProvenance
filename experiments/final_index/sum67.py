import glob, pandas as pd, numpy as np
W = '/mnt/data/yingli/counterprovenance'
G = pd.concat([pd.read_csv(f) for f in glob.glob(W + '/repo_final_abl/results/structure_final/*_groups.csv')])
S = pd.concat([pd.read_csv(f) for f in glob.glob(W + '/repo_final_abl/results/structure_final/*_sample.csv') if pd.read_csv(f).shape[0]])
print('EXP6 all groups:', len(G), 'models', G.model.nunique(), 'cells %.3e' % G.cells.sum(), 'blocks %.3e' % G.blocks.sum())
rec = G.changes.sum() + G.ckpt.sum()
print('  CellForest leaf records (log changes + checkpoint entries) %.3e, CellTable leaves %.3e, ratio %.1f' % (rec, G.celltable_leaves.sum(), G.celltable_leaves.sum() / rec))
print('  omega %.3e changes/omega %.2f, changes per cell %.2f, celltable leaves per cell %.1f' % (G.omega.sum(), G.changes.sum() / G.omega.sum(), G.changes.sum() / G.cells.sum(), G.celltable_leaves.sum() / G.cells.sum()))
print('  log bytes %.2f GB, celltable bytes %.2f GB' % (G.log_bytes.sum() / 1e9, G.celltable_bytes.sum() / 1e9))
m = G[G.trees > 1]
print('  groups of several trees: %d, mean MST edge weight (omega/(cells-1)) %.2f' % (len(m), m.omega.sum() / (m.cells - 1).clip(lower=1).sum()))
fam = G.assign(fam=G.model.str.rsplit('_', n=1).str[1]).groupby('fam').apply(lambda g: g.celltable_leaves.sum() / (g.changes.sum() + g.ckpt.sum()))
print('  ratio per family', fam.round(1).to_dict())
print('EXP6 sample:', len(S), 'groups', 'cells %.3e' % S.cells.sum())
mst = S.mst_changes.sum() + S.mst_ckpt.sum(); rnd = S.random_changes.sum() + S.random_ckpt.sum()
print('  records MST %.3e random tree %.3e (x%.2f) every edge %.3e (x%.2f) celltable %.3e (x%.1f)' % (mst, rnd, rnd / mst, S.every_edge.sum(), S.every_edge.sum() / S.mst_changes.sum(), S.celltable_leaves.sum(), S.celltable_leaves.sum() / mst))
print('  changes only: MST %.3e random %.3e (x%.2f) every edge (x%.2f)' % (S.mst_changes.sum(), S.random_changes.sum(), S.random_changes.sum() / S.mst_changes.sum(), S.every_edge.sum() / S.mst_changes.sum()))
C = pd.concat([pd.read_csv(f) for f in glob.glob(W + '/repo_final/results/combine/*.csv') + glob.glob(W + '/repo_final_lrf/results/combine/*.csv')])
print('EXP7 models', C.groupby(['dataset', 'family']).ngroups, 'max err', C.max_abs_err.max())
t = C.groupby('combo')[['cf_us', 'base_us']].mean(); t['x'] = t.base_us / t.cf_us
print(t.round(1).to_string())
pt = C.groupby(['dataset', 'combo'])[['cf_us', 'base_us']].mean(); pt['x'] = pt.base_us / pt.cf_us
print('  per table x (min, max):', pt.x.groupby('combo').agg(['min', 'max']).round(0).to_dict())
for ds in [x for x in ('postgres', 'baf') if x in pt.index.get_level_values(0)]:
    print('  ', ds, pt.loc[ds].x.round(0).to_dict())
