#!/bin/bash
# Run scm_full.py with one job per (dataset, family, d), PAR jobs at a time, for the datasets given.
# A dataset gets the d in $DIMS (default 1 2 4 8 16) that do not exceed the number of ancestors of its outcome.
# Jobs whose results/scm_full_<tag>.csv holds all 3 units, or that are running, are skipped (resume).
# Usage: [EXTRA="--ours-only"] [LOGDIR=...] [DIMS="1 2 4 8"] run_scm_split.sh DATA_DIR PAR dataset1,dataset2,...
DATA=$1; PAR=${2:-60}; SETS=$3
cd "$(dirname "$0")"
[ -d ../../chirho_pkgs ] && export PYTHONPATH=$(cd ../../chirho_pkgs && pwd)
LOGDIR=${LOGDIR:-../../logs/scm_full}
mkdir -p $LOGDIR
../../env/bin/python - "$DATA" "$SETS" "${DIMS:-1 2 4 8 16}" "$EXTRA" <<'PY' | xargs -P "$PAR" -L 1 bash -c 'OMP_NUM_THREADS=1 ../../env/bin/python scm_full.py --data '"$DATA"' --dataset $0 --family $1 --dims $2 --tag $0_$1_d$2 '"$EXTRA"' > '"$LOGDIR"'/$0_$1_d$2.log 2>&1'
import json, subprocess, sys
from pathlib import Path
import pandas as pd
data, sets, dims, extra = Path(sys.argv[1]), sys.argv[2].split(","), [int(x) for x in sys.argv[3].split()], sys.argv[4]
running = subprocess.run(["pgrep", "-af", "python scm_full.py"], capture_output=True, text=True).stdout.splitlines()
running = {line.split("--tag ")[1].split()[0] for line in running
           if "--tag " in line and ("--ours-only" in line) == ("--ours-only" in extra)}
def done(tag):
    f = Path(f"../results/scm_full_{tag}.csv")
    return f.exists() and len(pd.read_csv(f)) >= 3


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
    for d in dims:
        if d <= len(anc):
            for fam in ("LGBM", "XGB", "HGB", "RF", "GBR"):
                tag = f"{ds}_{fam}_d{d}"
                if tag not in running and not done(tag):
                    print(ds, fam, d)
PY
