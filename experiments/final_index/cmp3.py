import glob, os, pandas as pd
W = '/mnt/data/yingli/counterprovenance'
def tw(pats):
    out = {}
    for p in pats:
        for f in glob.glob(p):
            for r in pd.read_csv(f).to_dict('records'):
                out[(r['dataset'], r['family'], int(r['d']))] = (r['full_us'], r['groups'])
    return out
paper = tw([W + '/repo_hybrid/results/final/*_twin.csv']); paper.update({k: v for k, v in tw([W + '/repo_lrf/results/main/*_RF_d*_twin.csv']).items() if k[1] == 'RF'})
two = tw([W + '/repo_two/results/main/*_twin.csv']); two.update({k: v for k, v in tw([W + '/repo_two_lrf/results/main/*_RF_d*_twin.csv']).items() if k[1] == 'RF'})
fin = tw([W + '/repo_final/results/main/*_twin.csv']); fin.update({k: v for k, v in tw([W + '/repo_final_lrf/results/main/*_RF_d*_twin.csv']).items() if k[1] == 'RF'})
ks = [k for k in fin if k in paper and k in two]
d = pd.DataFrame([dict(ds=k[0], fam=k[1], d=k[2], paper=paper[k][0], two=two[k][0], fin=fin[k][0], gp=paper[k][1], gt=two[k][1], gf=fin[k][1]) for k in ks])
print(len(d), 'settings; mean us paper %.1f two-level(old) %.1f final %.1f' % (d.paper.mean(), d.two.mean(), d.fin.mean()))
m = d.groupby(['ds', 'fam'])[['paper', 'two', 'fin', 'gp', 'gt', 'gf']].mean()
m['r'] = m.fin / m.paper
print(m.sort_values('r', ascending=False).head(15).round(1).to_string())
print('groups changed vs two-level(old):', int((m.gf != m.gt).sum()), 'models')
print(m[m.gf != m.gt].round(0).to_string())
