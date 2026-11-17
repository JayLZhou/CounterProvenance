#!/usr/bin/env python3
"""Build and save the index of every mechanism that the main experiment uses (every d), for one table and model, into
results/indexes (mech_proto.index_dir), so that later experiments load it instead of building it again."""
import argparse, time
from common import RESULTS
from counterprovenance.scm import relevant_nodes
from mech_main import prepare
from mech_proto import MechanismIndex, index_dir

ap = argparse.ArgumentParser(description=__doc__)
for a, kw in (("--data", dict(required=True)), ("--dataset", dict(required=True)), ("--family", dict(required=True)),
              ("--trees", dict(type=int, default=1000)), ("--units", dict(type=int, default=1000)), ("--out", dict(default="")),
              ("--budget", dict(type=float, default=2e7)), ("--build-threads", dict(type=int, default=8))):
    ap.add_argument(a, **kw)
args = ap.parse_args()
state = prepare(args)
scm, y = state["scm"], state["y"]
nodes = sorted({x for w in state["work"].values() for x in relevant_nodes(scm.parents, w["T"], y)})
t0 = time.perf_counter()
mech = MechanismIndex(scm, state["rows"].iloc[0].to_dict(), nodes, args.budget, args.build_threads,
                      cache=index_dir(RESULTS, args.dataset, args.family, args.trees, args.budget))
print(args.dataset, args.family, len(nodes), "mechanisms", sum(1 for v in mech.info.values() if v.get("loaded")), "loaded,",
      round(time.perf_counter() - t0, 1), "s", flush=True)
