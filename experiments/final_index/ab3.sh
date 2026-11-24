# Same time, same load: the main experiment of a few models with the paper's index (repo_lean, read without two-level grids),
# the earlier two-level index (repo_two) and the final index (repo_final); each writes to results/ab3.
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES=
for m in "postgres XGB" "light_tunnel XGB" "pol XGB" "jdk XGB" "concrete XGB" "ailerons GBR"; do set -- $m
  for R in repo_lean repo_two repo_final; do
    ( cd $W/$R/experiments && PYTHONPATH=$W/$R:$W/chirho_pkgs CF_TWO_LEVEL=$([ $R = repo_lean ] && echo 0 || echo 1) $W/env/bin/python mech_main.py --data $W/data/causal --dataset $1 --family $2 --backend twin --out ab3 > $W/logs/ab3_${R}_$1_$2.log 2>&1 ) &
  done
  wait
done
$W/env/bin/python - <<'PY'
import glob, pandas as pd
W = '/mnt/data/yingli/counterprovenance'
rows = []
for R in ('repo_lean', 'repo_two', 'repo_final'):
    for f in glob.glob(f'{W}/{R}/results/ab3/*_twin.csv'):
        t = pd.read_csv(f); rows.append(dict(repo=R, model=f.split('/')[-1].rsplit('_d', 1)[0], d=int(t.d[0]), us=t.full_us[0]))
d = pd.DataFrame(rows).pivot_table(index='model', columns='repo', values='us', aggfunc='mean')
print(d.round(1).to_string()); print('means', d.mean().round(1).to_dict())
PY
