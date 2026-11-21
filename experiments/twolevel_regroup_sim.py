"""Grouping when a group of one tree is counted by its two-level table (as the paper now says): how many groups, and how
many of them single trees, against the cached index.  Greedy algorithm of the library with one change: the size of a group
of one tree.  Nothing is built."""
import argparse, glob, heapq, json, os, pickle, time
import numpy as np, pandas as pd
from counterprovenance.grouping import tree_profiles, nested_groups, Group, _merged
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--family", required=True)
ap.add_argument("--repo", default="/mnt/data/yingli/counterprovenance/repo_lean")
args = ap.parse_args()
build = pd.read_csv(f"{args.repo}/results/main/{args.dataset}_{args.family}_build.csv")
row = build[build.status == "ok"].sort_values("groups", ascending=False).iloc[0]
d0 = glob.glob(f"{args.repo}/results/indexes/{args.dataset}_{args.family}_t1000_b2e+07_*")[0]
index, info = pickle.load(open(os.path.join(d0, str(row.node).replace("/", "_").replace(" ", "_") + ".pkl"), "rb"))
bound, slabs = index.bound, index.slabs
d = bound.d; counts = [int(c) for c in slabs.counts]
full_L, full_U = np.zeros(d, dtype=np.int64), np.array(counts, dtype=np.int64)
def cuts_in(lo, hi, L, U):
    out = []
    for a in range(d):
        c = np.unique(np.concatenate((lo[:, a], hi[:, a]))); out.append(c[(c > L[a]) & (c < U[a])])
    return out
def top_split(lo, hi, L, U):
    best = None
    for a, c in enumerate(cuts_in(lo, hi, L, U)):
        for v in c:
            if not ((lo[:, a] < v) & (hi[:, a] > v)).any():
                left = int((hi[:, a] <= v).sum()); bal = min(left, len(lo) - left)
                if best is None or bal > best[0]: best = (bal, a, int(v))
    return best
def two_level(items):
    lo = np.array([[slabs.ranges[i][a][0] for a in range(d)] for i in items], dtype=np.int64)
    hi = np.array([[slabs.ranges[i][a][1] for a in range(d)] for i in items], dtype=np.int64)
    regions = [(lo, hi, full_L, full_U)]; cuts = [set() for _ in range(d)]; best = None
    for level in range(6):
        cost = int(np.prod([len(c) + 1 for c in cuts])) + sum(int(np.prod([len(c) + 1 for c in cuts_in(*r)])) for r in regions)
        best = cost if best is None else min(best, cost)
        nxt, grew = [], False
        for blo, bhi, bL, bU in regions:
            s = top_split(blo, bhi, bL, bU) if len(blo) > 1 else None
            if s is None: nxt.append((blo, bhi, bL, bU)); continue
            _, a, v = s; cuts[a].add(v); grew = True; m = bhi[:, a] <= v
            U1 = bU.copy(); U1[a] = v; L2 = bL.copy(); L2[a] = v
            nxt += [(blo[m], bhi[m], bL, U1), (blo[~m], bhi[~m], L2, bU)]
        if not grew: break
        regions = nxt
    return best
t0 = time.perf_counter()
profiles = tree_profiles(bound, slabs)
T2 = {p.tree: two_level(p.items) for p in profiles}
cap, budget, start_cap = index.grid_cap, int(index.budget), 1 << 10
def size_of(g): return T2[g.trees[0]] if len(g.trees) == 1 else g.cells
def greedy(profiles, two):
    start = nested_groups([p for p in profiles if p.cells <= cap], min(start_cap, cap))
    alive = dict(enumerate(start)); sizes = {k: (size_of(g) if two else g.cells) for k, g in alive.items()}
    total = sum(sizes.values()); heap = []; keys = list(alive)
    for i, k1 in enumerate(keys):
        g1 = alive[k1]
        for k2 in keys[i + 1:]:
            _, _, size = _merged(g1, alive[k2])
            if size <= cap: heap.append((size - sizes[k1] - sizes[k2], k1, k2))
    heapq.heapify(heap); nxt = len(start)
    while heap:
        delta, k1, k2 = heapq.heappop(heap)
        if k1 not in alive or k2 not in alive or (delta > 0 and total + delta > budget): continue
        axes, cuts, size = _merged(alive[k1], alive[k2])
        g = Group(axes, cuts, alive[k1].items + alive[k2].items, alive[k1].trees + alive[k2].trees)
        total += delta; del alive[k1], alive[k2], sizes[k1], sizes[k2]; k = nxt; nxt += 1
        for k3, g3 in alive.items():
            _, _, s3 = _merged(g, g3)
            if s3 <= cap: heapq.heappush(heap, (s3 - size - sizes[k3], k3, k))
        alive[k] = g; sizes[k] = size
    gs = list(alive.values())
    return dict(groups=len(gs), single=sum(len(g.trees) == 1 for g in gs), entries=int(total), dense_blocks=int(sum(g.cells for g in gs)), largest=max(len(g.trees) for g in gs))
fl = index._fast_leaves
out = dict(dataset=args.dataset, family=args.family, node=str(row.node), k=d, trees=len(profiles), budget=budget,
           cached=dict(groups=len(fl), blocks=int(sum(l.cells.overlay.grid_cells for l in fl))),
           same_rule=greedy(profiles, False), two_level=greedy(profiles, True), sum_two_level_singles=int(sum(T2.values())), sec=time.perf_counter() - t0)
print(json.dumps(out), flush=True)
