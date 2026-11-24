"""The parts of the final index size of every model: cells (value and log position), leaves (logs), block directories (maps of
the groups of several trees + cuts), two-level grids, rank rows that locate the groups, item tables."""
import glob, json, os, pickle, re, sys
import numpy as np
W = '/mnt/data/yingli/counterprovenance'
dirs = [d for d in glob.glob(W + '/repo_final/results/indexes/*_t1000_b2e+07_p2e+10_*')] + [d for d in glob.glob(W + '/repo_final_lrf/results/indexes/*_t1000_b2e+07_p2e+10_*')]
tot = {}
for d in sorted(dirs):
    model = re.sub(r"_t\d+_b.*$", "", os.path.basename(d))
    part = dict(cells=0, leaves=0, maps_multi=0, maps_single=0, two=0, rows=0, other=0, dense=0, dense_n=0, blocks=0, total=0)
    for f in glob.glob(d + '/*.pkl'):
        if f.endswith('.tl.pkl'): continue
        index, info = pickle.load(open(f, 'rb')); st = index.stats
        part['cells'] += 16 * st.faces
        part['leaves'] += st.leaf_bytes
        part['rows'] += 4 * st.pack_words
        part['other'] += 4 * (st.locator_words + st.assignments) + 16 * st.segment_nodes + 8 * st.bitmask_words + 8 * st.pft_nodes + 12 * st.pft_entries; part['dense'] += 8 * st.table_cells; part['dense_n'] += st.table_cells; part['blocks'] += st.grid_cells
        two = index._two_level() if index.compiled else False
        single = two.tl.single_map_bytes if two else 0
        part['maps_single'] += single; part['maps_multi'] += st.map_bytes - single
        part['two'] += sum(two.tl.bytes.values()) if two else 0
        del index
    part['total'] = part['cells'] + part['leaves'] + part['rows'] + part['other'] + part['dense'] + part['maps_multi'] + part['two']
    tot[model] = part
    print(model, {k: round(v / 1e9, 3) for k, v in part.items()}, flush=True)
json.dump(tot, open(W + '/tmp/parts_final2.json', 'w'))
s = {k: sum(p[k] for p in tot.values()) / 1e9 for k in next(iter(tot.values()))}
print('TOTAL GB', {k: round(v, 2) for k, v in s.items()})
