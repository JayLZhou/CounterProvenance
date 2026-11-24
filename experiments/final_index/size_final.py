import glob, json, os, pandas as pd
W = '/mnt/data/yingli/counterprovenance'
fs = [f for f in glob.glob(W + '/repo_final/results/main/*_build.csv') if not f.endswith('_RF_build.csv')] + glob.glob(W + '/repo_final_lrf/results/main/*_RF_build.csv')
t = pd.concat([pd.read_csv(f) for f in fs])
m = t.groupby(['dataset', 'family']).agg(bytes=('bytes', 'sum'), groups=('groups', 'max'), blocks=('blocks', 'sum'), cells=('cells', 'sum')).reset_index()
print('models', len(m), 'total GB %.1f' % (m.bytes.sum() / 1e9), 'median GB %.2f' % (m.bytes.median() / 1e9))
big = m.loc[m.bytes.idxmax()]; print('largest model', big.dataset, big.family, '%.1f GB' % (big.bytes / 1e9))
per = m.groupby('dataset').bytes.mean() / 1e9
print('per table mean GB:', per.round(3).sort_values().to_dict())
print('per family GB:', (m.groupby('family').bytes.sum() / 1e9).round(1).to_dict())
json.dump({ds: float(v) for ds, v in per.items()}, open(W + '/tmp/size_final.json', 'w'))
