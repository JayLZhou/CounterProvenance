"""Two-level grid with a compact layout and a compiled read, against the cached index read the same way.
One cached tree ensemble; nothing of the index is rebuilt or changed.  Every tree: a coarse grid over its top splits whose
blocks name a subtree (1 byte), and per subtree a small grid of leaf numbers (1 byte); the values are kept once per leaf.
Readers timed on the same queries, kernel only (ranks given) and end to end:
  now/numpy   the cached index as the paper reads it;      now/numba   the same arrays read by a compiled loop;
  two/numpy   the two-level grid with numpy;                two/numba   the two-level grid read by a compiled loop."""
import argparse, glob, json, os, pickle, time
from bisect import bisect_left
import numpy as np, pandas as pd, numba

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--family", required=True)
ap.add_argument("--repo", default="/mnt/data/yingli/counterprovenance/repo_lean")
ap.add_argument("--queries", type=int, default=3000); ap.add_argument("--max-top", type=int, default=5)
args = ap.parse_args()
build = pd.read_csv(f"{args.repo}/results/main/{args.dataset}_{args.family}_build.csv")
row = build[build.status == "ok"].sort_values("groups", ascending=False).iloc[0]
d0 = glob.glob(f"{args.repo}/results/indexes/{args.dataset}_{args.family}_t1000_b2e+07_*")[0]
index, info = pickle.load(open(os.path.join(d0, str(row.node).replace("/", "_").replace(" ", "_") + ".pkl"), "rb"))
bound, slabs = index.bound, index.slabs
d, counts, n = bound.d, [int(c) for c in slabs.counts], len(bound.boxes)
tree = np.array([b.tree for b in bound.boxes]); value = np.array([b.value for b in bound.boxes], dtype=np.float64)
LO = np.array([[slabs.ranges[i][a][0] for a in range(d)] for i in range(n)], dtype=np.int64)
HI = np.array([[slabs.ranges[i][a][1] for a in range(d)] for i in range(n)], dtype=np.int64)
S = sum(counts); rowbase = [int(v) for v in np.concatenate(([0], np.cumsum(counts)[:-1]))]
full_L, full_U = np.zeros(d, dtype=np.int64), np.array(counts, dtype=np.int64)
slab_ids = [np.arange(c) for c in counts]

def cuts_in(lo, hi, L, U):
    out = []
    for a in range(d):
        c = np.unique(np.concatenate((lo[:, a], hi[:, a])))
        out.append(c[(c > L[a]) & (c < U[a])])
    return out

def top_split(lo, hi, L, U):
    best = None
    for a, c in enumerate(cuts_in(lo, hi, L, U)):
        for v in c:
            if not ((lo[:, a] < v) & (hi[:, a] > v)).any():
                left = int((hi[:, a] <= v).sum()); bal = min(left, len(lo) - left)
                if best is None or bal > best[0]:
                    best = (bal, a, int(v))
    return best

def plans(lo, hi, ids):
    regions = [(lo, hi, ids, full_L, full_U)]; cuts = [set() for _ in range(d)]
    out = []
    for level in range(args.max_top + 1):
        coarse = int(np.prod([len(c) + 1 for c in cuts]))
        sub = sum(int(np.prod([len(c) + 1 for c in cuts_in(r[0], r[1], r[3], r[4])])) for r in regions)
        out.append((coarse + sub + 2 * S * len(regions), [sorted(c) for c in cuts], list(regions)))   # bytes of the compact layout
        nxt, grew = [], False
        for blo, bhi, bi, bL, bU in regions:
            s = top_split(blo, bhi, bL, bU) if len(blo) > 1 else None
            if s is None:
                nxt.append((blo, bhi, bi, bL, bU)); continue
            _, a, v = s; cuts[a].add(v); grew = True
            m = bhi[:, a] <= v
            U1 = bU.copy(); U1[a] = v; L2 = bL.copy(); L2[a] = v
            nxt += [(blo[m], bhi[m], bi[m], bL, U1), (blo[~m], bhi[~m], bi[~m], L2, bU)]
        if not grew:
            break
        regions = nxt
    return out

t0 = time.perf_counter()
const = float(bound.offset)
ids_, _ = np.unique(tree, return_index=True)
order = np.argsort(tree, kind="stable"); ends = np.r_[np.searchsorted(tree[order], ids_), n]
E_parts, L_parts, W1_cols, W2_cols, base1, base2, subfirst, leafbase, VAL = [], [], [], [], [], [], [], [], []
e_at = l_at = nsub = nleaf = 0; tops = []
for j, m in enumerate(ids_):
    sel = order[ends[j]:ends[j + 1]]
    lo, hi = LO[sel], HI[sel]
    if len(sel) == 1:
        const += float(value[sel[0]]); continue
    ps = plans(lo, hi, np.arange(len(sel)))
    k = int(np.argmin([p[0] for p in ps])); tops.append(k)
    _, ccuts, regions = ps[k]
    dims1 = [len(c) + 1 for c in ccuts]
    E = np.zeros(dims1, dtype=np.int64)
    col1 = np.zeros(S, dtype=np.int64); stride = 1
    for a in reversed(range(d)):
        col1[rowbase[a]:rowbase[a] + counts[a]] = np.searchsorted(np.asarray(ccuts[a], dtype=np.int64), slab_ids[a], side="right") * stride
        stride *= dims1[a]
    subfirst.append(nsub)
    for local, (rlo, rhi, ri, rL, rU) in enumerate(regions):
        sl = tuple(slice(int(np.searchsorted(ccuts[a], rL[a], side="right")), int(np.searchsorted(ccuts[a], rU[a] - 1, side="right")) + 1) for a in range(d))
        E[sl] = local
        rc = cuts_in(rlo, rhi, rL, rU); dims2 = [len(c) + 1 for c in rc]
        G = np.zeros(dims2, dtype=np.int64)
        for b in range(len(rlo)):
            bs = tuple(slice(int(np.searchsorted(rc[a], max(rlo[b, a], rL[a]), side="right")), int(np.searchsorted(rc[a], min(rhi[b, a], rU[a]) - 1, side="right")) + 1) for a in range(d))
            G[bs] = ri[b]
        col2 = np.zeros(S, dtype=np.int64); stride = 1
        for a in reversed(range(d)):
            col2[rowbase[a]:rowbase[a] + counts[a]] = np.searchsorted(rc[a], slab_ids[a], side="right") * stride
            stride *= dims2[a]
        W2_cols.append(col2); base2.append(l_at); L_parts.append(G.ravel()); l_at += G.size; nsub += 1
    W1_cols.append(col1); base1.append(e_at); E_parts.append(E.ravel()); e_at += E.size
    leafbase.append(nleaf); VAL.append(value[sel]); nleaf += len(sel)
def small(a):
    a = np.asarray(a); m = int(a.max()) if a.size else 0
    return np.ascontiguousarray(a.astype(np.uint8 if m < 256 else np.uint16 if m < 65536 else np.uint32))
E1 = small(np.concatenate(E_parts)); L2 = small(np.concatenate(L_parts)); VAL = np.concatenate(VAL)
W1 = small(np.stack(W1_cols, axis=1)); W2 = small(np.stack(W2_cols, axis=1))
base1, base2, subfirst, leafbase = (np.array(v, dtype=np.int64) for v in (base1, base2, subfirst, leafbase))
M = len(tops); build_s = time.perf_counter() - t0
two_bytes = dict(coarse=E1.nbytes, leaf_numbers=L2.nbytes, values=VAL.nbytes, rank1=W1.nbytes, rank2=W2.nbytes, offsets=base1.nbytes + base2.nbytes + subfirst.nbytes + leafbase.nbytes)

th = [list(map(float, t)) for t in slabs.thresholds]; cast = index._cast
def ranks(x):
    return np.array([rowbase[a] + bisect_left(th[a], x[a]) for a in range(d)], dtype=np.int64)

@numba.njit
def two_nb(rs, W1, base1, E1, subfirst, W2, base2, L2, leafbase, VAL, c1, sub, c2):
    m = c1.shape[0]
    for j in range(m):
        c1[j] = base1[j]; c2[j] = 0
    for a in range(rs.shape[0]):
        r = W1[rs[a]]
        for j in range(m): c1[j] += r[j]
    for j in range(m): sub[j] = subfirst[j] + E1[c1[j]]
    for a in range(rs.shape[0]):
        r = W2[rs[a]]
        for j in range(m): c2[j] += r[sub[j]]
    t = 0.0
    for j in range(m): t += VAL[leafbase[j] + L2[base2[sub[j]] + c2[j]]]
    return t
c1b, subb, c2b = (np.empty(M, dtype=np.int64) for _ in range(3))

# level 2 only on the axes that a subtree still splits on
act = np.stack([W2[rowbase[a]:rowbase[a] + counts[a]].max(axis=0) > 0 for a in range(d)], axis=1)          # (subtrees, d)
nact = act.sum(axis=1).astype(np.int64); A = max(int(nact.max()), 1)
axes_of = np.zeros((nsub, A), dtype=np.int64)
for s_ in range(nsub):
    ax = np.flatnonzero(act[s_]); axes_of[s_, :len(ax)] = ax
@numba.njit
def two_nb2(rs, W1, base1, E1, subfirst, W2, base2, L2, leafbase, VAL, nact, axes_of, c1):
    m = c1.shape[0]
    for j in range(m): c1[j] = base1[j]
    for a in range(rs.shape[0]):
        r = W1[rs[a]]
        for j in range(m): c1[j] += r[j]
    t = 0.0
    for j in range(m):
        s = subfirst[j] + E1[c1[j]]
        c = base2[s]
        for q in range(nact[s]):
            c += W2[rs[axes_of[s, q]], s]
        t += VAL[leafbase[j] + L2[c]]
    return t
W1i, W2i, E1i, L2i = W1.astype(np.int64), W2.astype(np.int64), E1.astype(np.int64), L2.astype(np.int64)   # numpy indexes with 8-byte numbers
W1i[rowbase[0]:rowbase[0] + counts[0]] += base1; W2i[rowbase[0]:rowbase[0] + counts[0]] += base2; E1i_ = E1i
tb = np.empty(M, dtype=np.int64); vb = np.empty(M, dtype=np.float64)
def two_np(rs):
    np.add(W1i[rs[0]], W1i[rs[1]], out=c1b) if d > 1 else np.copyto(c1b, W1i[rs[0]])
    for r in rs[2:]: np.add(c1b, W1i[r], out=c1b)
    E1i_.take(c1b, out=subb, mode="clip"); np.add(subb, subfirst, out=subb)
    W2i[rs[0]].take(subb, out=c2b, mode="clip")
    for r in rs[1:]:
        W2i[r].take(subb, out=tb, mode="clip"); np.add(c2b, tb, out=c2b)
    L2i.take(c2b, out=tb, mode="clip"); np.add(tb, leafbase, out=tb)
    VAL.take(tb, out=vb, mode="clip")
    return float(np.add.reduce(vb))

# the cached index, read by numpy as in the paper and by a compiled loop over the same arrays
G = len(index._fast_leaves)
Wn = np.zeros((S, G), dtype=np.int64); used = []
for a, t_, wide in index._nwide:
    Wn[rowbase[a]:rowbase[a] + counts[a]] = wide; used.append(a)
used = np.array(used, dtype=np.int64)
dense = index._ndense is not None
nbase, nfaces = index._nbase.astype(np.int64), (index._nfaces.astype(np.int64) if not dense else np.zeros(G, dtype=np.int64))
m0, m1, m2 = (index._nmaps if not dense else (np.zeros(1, np.uint8), np.zeros(1, np.uint16), np.zeros(1, np.uint32)))
nvalue = index._nvalue if not dense else index._ndense
n1, n12 = (int(index._n1), int(index._n12)) if not dense else (0, 0)
@numba.njit
def now_nb(rs, used, Wn, nbase, m0, m1, m2, nfaces, nvalue, n1, n12, dense, c):
    g = c.shape[0]
    for j in range(g): c[j] = nbase[j]
    for i in range(used.shape[0]):
        r = Wn[rs[used[i]]]
        for j in range(g): c[j] += r[j]
    t = 0.0
    if dense:
        for j in range(g): t += nvalue[c[j]]
        return t
    for j in range(n1): t += nvalue[m0[c[j]] + nfaces[j]]
    for j in range(n1, n12): t += nvalue[m1[c[j]] + nfaces[j]]
    for j in range(n12, g): t += nvalue[m2[c[j]] + nfaces[j]]
    return t
cg = np.empty(G, dtype=np.int64)
def now_np(x):
    c = index._nbase.copy()
    for a, t_, rows_ in index._nwide:
        c += rows_[bisect_left(t_, x[a])]
    return index._const + float(index._read_cells(c, index._nfaces, index._n1, index._n12).sum())

rng = np.random.default_rng(3)
lo_v = np.array([t[0] if t else 0.0 for t in th]); hi_v = np.array([t[-1] if t else 1.0 for t in th])
Q = rng.uniform(lo_v - 0.05 * (hi_v - lo_v), hi_v + 0.05 * (hi_v - lo_v), size=(args.queries, d))
X = [cast(q) for q in Q]; RS = [ranks(x) for x in X]
ref = np.array([index._query_vector(x).value for x in X])
f_two_nb = lambda j: const + two_nb(RS[j], W1, base1, E1, subfirst, W2, base2, L2, leafbase, VAL, c1b, subb, c2b)
f_two_np = lambda j: const + two_np(RS[j])
f_two_nb2 = lambda j: const + two_nb2(RS[j], W1, base1, E1, subfirst, W2, base2, L2, leafbase, VAL, nact, axes_of, c1b)
f_now_nb = lambda j: index._const + now_nb(RS[j], used, Wn, nbase, m0, m1, m2, nfaces, nvalue, n1, n12, dense, cg)
f_now_np = lambda j: now_np(X[j])
errs = {name: float(np.abs(ref - np.array([f(j) for j in range(len(Q))])).max()) for name, f in (("two_numba", f_two_nb), ("two_numba2", f_two_nb2), ("two_numpy", f_two_np), ("now_numba", f_now_nb), ("now_numpy", f_now_np))}
def bench(f):
    best = 1e9
    for _ in range(7):
        t0 = time.perf_counter()
        for j in range(len(Q)): f(j)
        best = min(best, (time.perf_counter() - t0) / len(Q) * 1e6)
    return best
us = dict(now_numpy=bench(f_now_np), two_numpy=bench(f_two_np), now_numba=bench(f_now_nb), two_numba=bench(f_two_nb), two_numba2=bench(f_two_nb2), mean_active_axes=float(nact.mean()),
          locate=bench(lambda j: ranks(cast(Q[j]))), empty_call=bench(lambda j: None))
now_bytes = dict(map=int(index._map_bytes), values=int(nvalue.nbytes), rank=int(sum(w.nbytes for _, _, w in index._nwide)))
print(json.dumps(dict(dataset=args.dataset, family=args.family, node=str(row.node), k=d, trees=M, now_groups=G, subtrees=nsub,
    now_mb=sum(now_bytes.values()) / 1e6, two_mb=sum(two_bytes.values()) / 1e6, now_bytes=now_bytes, two_bytes={k: int(v) for k, v in two_bytes.items()},
    dtypes=dict(E1=str(E1.dtype), L2=str(L2.dtype), W1=str(W1.dtype), W2=str(W2.dtype)), errs=errs, us=us, build_s=build_s)), flush=True)
