#!/bin/bash
# Run scm_full.py with one job per (dataset, family, d), PAR jobs at a time, for the datasets given.
# A dataset gets the d in {1,2,4,8,16} that do not exceed the number of ancestors of its outcome.
# Usage: run_scm_split.sh DATA_DIR PAR dataset1,dataset2,...
DATA=$1; PAR=${2:-60}; SETS=$3
cd "$(dirname "$0")"
[ -d ../../chirho_pkgs ] && export PYTHONPATH=$(cd ../../chirho_pkgs && pwd)
mkdir -p ../../logs/scm_full
../../env/bin/python - "$DATA" "$SETS" <<'PY' | xargs -P "$PAR" -L 1 bash -c 'OMP_NUM_THREADS=1 ../../env/bin/python scm_full.py --data '"$DATA"' --dataset $0 --family $1 --dims $2 --tag $0_$1_d$2 > ../../logs/scm_full/$0_$1_d$2.log 2>&1'
import json, sys
from pathlib import Path
import pandas as pd
data, sets = Path(sys.argv[1]), sys.argv[2].split(",")
for ds in sets:
    meta = json.loads((data / ds / "meta.json").read_text())
    edges = pd.read_csv(data / ds / "graph.csv", dtype=str).itertuples(index=False)
    parents = {}
    for a, b in edges:
        parents.setdefault(b, set()).add(a)
    anc, stack = set(), [str(meta["target"])]
    while stack:
        for p in parents.get(stack.pop(), ()):
            if p not in anc:
                anc.add(p)
                stack.append(p)
    for d in (1, 2, 4, 8, 16):
        if d <= len(anc):
            for fam in ("LGBM", "XGB", "HGB", "RF", "GBR"):
                print(ds, fam, d)
PY
