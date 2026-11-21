"""Exp.1 with two-level tables: the same groups as the cached indexes, but every group of a single tree is read through
its two-level table (numpy, as the rest of the read).  Same units and queries as the main experiment; the cached index
and the two-level version are timed in the same process, unit by unit, and their answers compared.
Output: results/twolevel/<dataset>_<family>.jsonl (one line per d, and one line with the sizes)."""
import argparse, json, time
from bisect import bisect_left
from pathlib import Path
import numpy as np
from common import RESULTS
from counterprovenance.scm import relevant_nodes
from mech_main import prepare, queries
from mech_proto import MechanismIndex, index_dir

MAX_TOP = 5

def cuts_in(lo, hi, L, U):
    out = []
    for a in range(lo.shape[1]):
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

class TwoLevel:
    """The two-level tables of all groups of one tree of an index (None-like if it has none or is not read by vectors)."""
    def __init__(self, index):
        self.ok = False
        fl = index._fast_leaves
        if getattr(index, "_nbase", None) is None or index._ndense is not None or not getattr(index, "_nwide", None):
            return
        pos = {id(r): i for i, r in enumerate(index.roots) if r is not None}
        bound, slabs = index.bound, index.slabs
        d = bound.d; counts = [int(c) for c in slabs.counts]; S = sum(counts)
        rowbase = [int(v) for v in np.concatenate(([0], np.cumsum(counts)[:-1]))]
        items = [index.groups[pos[id(l)]][1] for l in fl]
        ntree = [len({bound.boxes[i].tree for i in it}) for it in items]
        self.single = np.array([g for g, t in enumerate(ntree) if t == 1], dtype=np.int64)
        self.multi = np.array([g for g, t in enumerate(ntree) if t != 1], dtype=np.int64)
        if not len(self.single):
            return
        full_L, full_U = np.zeros(d, dtype=np.int64), np.array(counts, dtype=np.int64)
        slab_ids = [np.arange(c) for c in counts]
        E_parts, L_parts, W1c, W2c, base1, base2, subfirst, leafbase, VAL, uses = [], [], [], [], [], [], [], [], [], []
        e_at = l_at = nsub = nleaf = 0
        for g in self.single:
            it = items[g]
            lo = np.array([[slabs.ranges[i][a][0] for a in range(d)] for i in it], dtype=np.int64)
            hi = np.array([[slabs.ranges[i][a][1] for a in range(d)] for i in it], dtype=np.int64)
            val = np.array([bound.boxes[i].value for i in it], dtype=np.float64)
            regions = [(lo, hi, np.arange(len(it)), full_L, full_U)]; cuts = [set() for _ in range(d)]; best = None
            for level in range(MAX_TOP + 1):
                coarse = int(np.prod([len(c) + 1 for c in cuts]))
                sub = sum(int(np.prod([len(c) + 1 for c in cuts_in(r[0], r[1], r[3], r[4])])) for r in regions)
                cost = coarse + sub + 2 * S * len(regions)
                if best is None or cost < best[0]:
                    best = (cost, [sorted(c) for c in cuts], list(regions))
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
            _, ccuts, regions = best
            dims1 = [len(c) + 1 for c in ccuts]
            E = np.zeros(dims1, dtype=np.int64); col1 = np.zeros(S, dtype=np.int64); stride = 1
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
                W2c.append(col2); base2.append(l_at); L_parts.append(G.ravel()); l_at += G.size; nsub += 1
            W1c.append(col1); base1.append(e_at); E_parts.append(E.ravel()); e_at += E.size
            leafbase.append(nleaf); VAL.append(val); nleaf += len(it)
            tc = cuts_in(lo, hi, full_L, full_U); uses.append([len(c) > 0 for c in tc])
        self.d, self.S, self.rowbase, self.nsub = d, S, rowbase, nsub
        self.th = [list(map(float, t)) for t in slabs.thresholds]
        self.E1 = np.concatenate(E_parts); self.L2 = np.concatenate(L_parts); self.VAL = np.concatenate(VAL)
        self.W1 = np.ascontiguousarray(np.stack(W1c, axis=1)); self.W2 = np.ascontiguousarray(np.stack(W2c, axis=1))
        self.base1, self.base2, self.subfirst, self.leafbase = (np.array(v, dtype=np.int64) for v in (base1, base2, subfirst, leafbase))
        self.uses = np.array(uses, dtype=bool)
        small = lambda a: 1 if int(a.max()) < 256 else 2 if int(a.max()) < 65536 else 4
        self.bytes = dict(coarse=int(self.E1.size * small(self.E1)), leaf_numbers=int(self.L2.size * small(self.L2)), rank1=int(self.W1.size * small(self.W1)),
                          rank2=int(self.W2.size * small(self.W2)), offsets=int(8 * (3 * len(self.single) + nsub)))
        self.entries = int(self.E1.size + self.L2.size)
        maps_item = [np.dtype(l.cells.overlay.face_of_cell.dtype).itemsize for l in fl]
        self.now_single_blocks = int(sum(fl[g].cells.overlay.grid_cells for g in self.single))
        self.now_single_map_bytes = int(sum(fl[g].cells.overlay.grid_cells * maps_item[g] for g in self.single))
        self.now_single_rank_bytes = int(sum(w.shape[0] for _, _, w in index._nwide) * len(self.single) * 8)
        self.ok = True

    def read(self, rows_all, cols_W1, base1, subfirst, leafbase):
        """All axes: the leaves' values of the trees in ``cols`` (their W1 columns are given)."""
        c1 = base1 + cols_W1[rows_all].sum(axis=0)
        sub = subfirst + self.E1.take(c1)
        c2 = self.base2.take(sub)
        for r in rows_all:
            c2 += self.W2[r].take(sub)
        return float(self.VAL.take(leafbase + self.L2.take(c2)).sum())

class Hybrid:
    """A partial read of an index: groups of several trees by their tables of blocks, groups of one tree by two levels."""
    def __init__(self, index, tl, varying):
        self.index, self.tl, self.cast = index, tl, index._cast
        d = tl.d; vary = sorted(set(varying)); self.vary = vary; self.fixed = [a for a in range(d) if a not in vary]
        p = index.partial(varying); touched = set(p.touched)
        mt = np.array([g for g in tl.multi if g in touched], dtype=np.int64); mu = np.array([g for g in tl.multi if g not in touched], dtype=np.int64)
        self.mt, self.mu = mt, mu
        self.w_mt = np.searchsorted(mt, [index._n1, index._n12]).tolist(); self.w_mu = np.searchsorted(mu, [index._n1, index._n12]).tolist()
        self.f_mt, self.f_mu = index._nfaces[mt], index._nfaces[mu]
        self.rows_fixed = [(a, t, rows) for a, t, rows in index._nwide if a not in vary]
        self.rows_vary = [(a, t, np.ascontiguousarray(rows[:, mt])) for a, t, rows in index._nwide if a in vary]
        ts = np.array([j for j in range(len(tl.single)) if int(tl.single[j]) in touched], dtype=np.int64)
        us = np.array([j for j in range(len(tl.single)) if int(tl.single[j]) not in touched], dtype=np.int64)
        self.ts, self.us = ts, us
        self.W1t = np.ascontiguousarray(tl.W1[:, ts]); self.W1u = np.ascontiguousarray(tl.W1[:, us])
        self.b1t, self.sft, self.lbt = tl.base1[ts], tl.subfirst[ts], tl.leafbase[ts]
        self.b1u, self.sfu, self.lbu = tl.base1[us], tl.subfirst[us], tl.leafbase[us]
        self.touched = len(mt) + len(ts)
        self.base_mt = index._nbase[mt]

    def full(self, x):
        """All groups, all axes: only valid when every axis varies."""
        cast = self.cast(x); index, tl = self.index, self.tl
        v = index._const
        if len(self.mt):
            c = self.base_mt.copy()
            for a, t, rows in self.rows_vary:
                c += rows[bisect_left(t, cast[a])]
            v += float(index._read_cells(c, self.f_mt, *self.w_mt).sum())
        if len(self.ts):
            rs = [tl.rowbase[a] + bisect_left(tl.th[a], cast[a]) for a in range(tl.d)]
            c1 = self.b1t.copy()
            for r in rs:
                c1 += self.W1t[r]
            sub = self.sft + tl.E1.take(c1)
            c2 = tl.base2.take(sub)
            for r in rs:
                c2 += tl.W2[r].take(sub)
            v += float(tl.VAL.take(self.lbt + tl.L2.take(c2)).sum())
        return v

    def bind(self, x):
        cast = self.cast(x); index, tl = self.index, self.tl
        c = index._nbase.copy()
        for a, t, rows in self.rows_fixed:
            c += rows[bisect_left(t, cast[a])]
        const = index._const
        if len(self.mu):
            const += float(index._read_cells(c[self.mu], self.f_mu, *self.w_mu).sum())
        if len(self.us):
            rows_all = [tl.rowbase[a] + bisect_left(tl.th[a], cast[a]) for a in range(tl.d)]
            const += tl.read(rows_all, self.W1u, self.b1u, self.sfu, self.lbu)
        c1f, F2 = self.b1t.copy(), tl.base2.copy()
        for a in self.fixed:
            r = tl.rowbase[a] + bisect_left(tl.th[a], cast[a])
            c1f += self.W1t[r]; F2 += tl.W2[r]
        return c[self.mt], const, c1f, F2

    def value(self, x, b):
        cast = self.cast(x); index, tl = self.index, self.tl
        cm, v, c1f, F2 = b
        if len(self.mt):
            c = cm.copy()
            for a, t, rows in self.rows_vary:
                c += rows[bisect_left(t, cast[a])]
            v = v + float(index._read_cells(c, self.f_mt, *self.w_mt).sum())
        if len(self.ts):
            rs = [tl.rowbase[a] + bisect_left(tl.th[a], cast[a]) for a in self.vary]
            c1 = c1f.copy()
            for r in rs:
                c1 += self.W1t[r]
            sub = self.sft + tl.E1.take(c1)
            c2 = F2.take(sub)
            for r in rs:
                c2 += tl.W2[r].take(sub)
            v = v + float(tl.VAL.take(self.lbt + tl.L2.take(c2)).sum())
        return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--family", required=True)
    ap.add_argument("--trees", type=int, default=1000); ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--n-units", type=int, default=200); ap.add_argument("--per-unit", type=int, default=10)
    args = ap.parse_args()
    out = RESULTS / "twolevel"; out.mkdir(parents=True, exist_ok=True)
    f = open(out / f"{args.dataset}_{args.family}.jsonl", "w")
    state = prepare(args)
    scm, y, rows, work = state["scm"], state["y"], state["rows"], state["work"]
    nodes = sorted({x for w in work.values() for x in relevant_nodes(scm.parents, w["T"], y)})
    cache = index_dir(RESULTS, args.dataset, args.family, args.trees, 2e7)
    if any(not (cache / (str(x).replace("/", "_").replace(" ", "_") + ".pkl")).exists() for x in nodes):
        print(json.dumps(dict(status="no cache")), file=f); return
    mech = MechanismIndex(scm, rows.iloc[0].to_dict(), nodes, 2e7, 1, cache=cache)
    t0 = time.perf_counter()
    TL = {x: TwoLevel(mech.indexes[x]) for x in nodes}
    conv_s = time.perf_counter() - t0
    sizes = dict(kind="sizes", dataset=args.dataset, family=args.family, mechanisms=len(nodes), with_two_level=sum(t.ok for t in TL.values()), convert_s=conv_s,
                 index_bytes=int(sum(v["bytes"] for v in mech.info.values())), groups=int(sum(v["groups"] for v in mech.info.values())),
                 single_groups=int(sum(len(t.single) for t in TL.values() if t.ok)), single_blocks=int(sum(t.now_single_blocks for t in TL.values() if t.ok)),
                 single_map_bytes=int(sum(t.now_single_map_bytes for t in TL.values() if t.ok)), single_rank_bytes=int(sum(t.now_single_rank_bytes for t in TL.values() if t.ok)),
                 two_entries=int(sum(t.entries for t in TL.values() if t.ok)), two_bytes={k: int(sum(t.bytes[k] for t in TL.values() if t.ok)) for k in ("coarse", "leaf_numbers", "rank1", "rank2", "offsets")})
    print(json.dumps(sizes), file=f, flush=True)
    for d, w in work.items():
        T = w["T"]; order = relevant_nodes(scm.parents, T, y); vary = set(T) | set(order)
        plan = mech.plan(T, order)
        plan2 = []
        for x, pa, partial in plan:
            tl = TL[x]
            plan2.append((x, pa, Hybrid(mech.indexes[x], tl, [i for i, p in enumerate(pa) if p in vary]) if tl.ok else partial,
                          Hybrid(mech.indexes[x], tl, list(range(len(pa)))) if tl.ok else None))
        def full_two(unit, t, obs2):
            values = dict(unit)
            for name, v in zip(T, t):
                values[name] = float(v)
            xx = None
            for xx, pa, h, hfull in plan2:
                xv = [values[p] for p in pa]
                values[xx] = unit[xx] + ((hfull.full(xv) if hfull is not None else mech.indexes[xx].query(xv).value) - obs2[xx])
            return values[xx]
        lat_now, lat_two = [], []
        acc = dict(now_abduct=0.0, now_bind=0.0, now_query=0.0, two_abduct=0.0, two_bind=0.0, two_query=0.0); n = 0; worst = 0.0
        for u, Q in enumerate(queries(state, d, args.per_unit)):      # warm-up: both readers touch their arrays
            if u >= 5:
                break
            unit = rows.iloc[u].to_dict(); ob = mech.observed(unit, order)
            for t in Q:
                mech.query(unit, T, t, order, ob); full_two(unit, t, ob)
        for u, Q in enumerate(queries(state, d, args.per_unit)):
            if u >= args.n_units:
                break
            unit = rows.iloc[u].to_dict()
            t0 = time.perf_counter(); observed = mech.observed(unit, order); t1 = time.perf_counter()
            bound = mech.bind(plan, unit, observed); t2 = time.perf_counter()
            now = [mech.query_partial(unit, T, t, bound) for t in Q]; t3 = time.perf_counter()
            t3b = time.perf_counter()
            obs2 = {}
            for x, pa, h, hfull in plan2:
                xv = [unit[p] for p in pa]
                obs2[x] = hfull.value(xv, hfull.bind(xv)) if hfull is not None else mech.indexes[x].query(xv).value
            t4 = time.perf_counter()
            for t in Q:                                               # the full read, as the paper times it
                q0 = time.perf_counter(); fa = mech.query(unit, T, t, order, observed); q1 = time.perf_counter()
                fb = full_two(unit, t, obs2); q2 = time.perf_counter()
                lat_now.append(q1 - q0); lat_two.append(q2 - q1); worst = max(worst, abs(fa - fb))
            t4b = time.perf_counter()
            b2 = [(x, pa, h, h.bind([unit[p] for p in pa]), float(unit[x]), obs2[x]) for x, pa, h, _ in plan2]; t5 = time.perf_counter()
            two = []
            for t in Q:
                values = dict(unit)
                for name, v in zip(T, t):
                    values[name] = float(v)
                xx = None
                for xx, pa, h, b, ux, obs in b2:
                    values[xx] = ux + (h.value([values[p] for p in pa], b) - obs)
                two.append(values[xx])
            t6 = time.perf_counter()
            acc["now_abduct"] += t1 - t0; acc["now_bind"] += t2 - t1; acc["now_query"] += t3 - t2
            acc["two_abduct"] += t4 - t3b; acc["two_bind"] += t5 - t4b; acc["two_query"] += t6 - t5
            n += len(Q); worst = max(worst, float(np.abs(np.array(now) - np.array(two)).max()))
        units = min(args.n_units, len(rows))
        r = dict(kind="time", dataset=args.dataset, family=args.family, d=d, path=len(order), units=units, queries=n, max_abs_diff=worst,
                 read_groups=int(sum(len(pl[2].touched) for pl in plan)), read_single=int(sum(len(h.ts) for _, _, h, hf in plan2 if hf is not None)),
                 **{k + "_us": v / (units if "query" not in k else n) * 1e6 for k, v in acc.items()})
        ln, lt = np.array(lat_now) * 1e6, np.array(lat_two) * 1e6
        r.update(now_full_us=float(ln.mean()), two_full_us=float(lt.mean()), now_full_p50=float(np.median(ln)), two_full_p50=float(np.median(lt)), now_full_p99=float(np.quantile(ln, .99)), two_full_p99=float(np.quantile(lt, .99)))
        r["now_total_us"] = r["now_query_us"] + (r["now_abduct_us"] + r["now_bind_us"]) / args.per_unit
        r["two_total_us"] = r["two_query_us"] + (r["two_abduct_us"] + r["two_bind_us"]) / args.per_unit
        print(json.dumps(r), file=f, flush=True)

if __name__ == "__main__":
    main()
