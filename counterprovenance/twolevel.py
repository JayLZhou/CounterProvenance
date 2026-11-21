"""Two-level tables for the groups of a single tree (Section "Two-level tables for single trees" of the paper).

A tree with L leaves over k parents has up to (cuts per parent + 1)^k blocks, since its grid extends every threshold over
the whole input space.  Its two-level table is a coarse grid over the thresholds of its top levels, whose blocks name a
subtree, and one small grid per subtree with the leaf of each of its blocks.  The top levels are found from the leaf boxes:
a cut that no box of the region straddles is a top split of the tree restricted to the region.

:func:`single_sizes` gives the entries of the two-level table of every tree (the size that the grouping counts for a group
of one tree); :class:`TwoLevel` holds the tables of all single-tree groups of an index, and :class:`Hybrid` reads an index
through them (groups of several trees keep their tables of blocks)."""
from __future__ import annotations

from bisect import bisect_left

import numpy as np

MAX_TOP = 5


def _cuts_in(lo, hi, L, U):
    out = []
    for a in range(lo.shape[1]):
        c = np.unique(np.concatenate((lo[:, a], hi[:, a])))
        out.append(c[(c > L[a]) & (c < U[a])])
    return out


def _top_split(lo, hi, L, U):
    """A cut inside the region that no box straddles, i.e., a top split of the tree restricted to the region; the one with
    the most even numbers of boxes on its two sides.  Per axis: a cut v is straddled by (boxes with lo < v) - (boxes with
    hi <= v) boxes, counted in the sorted bounds."""
    best = None
    n = len(lo)
    for a in range(lo.shape[1]):
        l, h = np.sort(lo[:, a]), np.sort(hi[:, a])
        c = np.unique(np.concatenate((l, h)))
        c = c[(c > L[a]) & (c < U[a])]
        if not len(c):
            continue
        left = np.searchsorted(h, c, side="right")
        free = np.searchsorted(l, c, side="left") == left
        if not free.any():
            continue
        bal = np.where(free, np.minimum(left, n - left), -1)
        i = int(np.argmax(bal))
        if best is None or bal[i] > best[0]:
            best = (int(bal[i]), a, int(c[i]))
    return best


def _boxes(slabs, items):
    d = len(slabs.counts)
    lo = np.array([[slabs.ranges[i][a][0] for a in range(d)] for i in items], dtype=np.int64)
    hi = np.array([[slabs.ranges[i][a][1] for a in range(d)] for i in items], dtype=np.int64)
    return lo, hi


def plan(lo, hi, counts):
    """The top levels (0..MAX_TOP) with the fewest stored numbers: ``(entries, coarse cuts per axis, regions)``; a region is
    ``(lo, hi, box numbers, L, U)``.  The numbers are the entries of the tables plus one per subtree and interval."""
    d, S = lo.shape[1], int(sum(counts))
    full_L, full_U = np.zeros(d, dtype=np.int64), np.array(counts, dtype=np.int64)
    regions = [(lo, hi, np.arange(len(lo)), full_L, full_U)]
    cuts = [set() for _ in range(d)]
    best = None
    for _ in range(MAX_TOP + 1):
        coarse = int(np.prod([len(c) + 1 for c in cuts]))
        sub = sum(int(np.prod([len(c) + 1 for c in _cuts_in(r[0], r[1], r[3], r[4])])) for r in regions)
        cost = coarse + sub + 2 * S * len(regions)
        if best is None or cost < best[0]:
            best = (cost, coarse + sub, [sorted(c) for c in cuts], list(regions))
        nxt, grew = [], False
        for blo, bhi, bi, bL, bU in regions:
            s = _top_split(blo, bhi, bL, bU) if len(blo) > 1 else None
            if s is None:
                nxt.append((blo, bhi, bi, bL, bU))
                continue
            _, a, v = s
            cuts[a].add(v)
            grew = True
            m = bhi[:, a] <= v
            U1 = bU.copy(); U1[a] = v
            L2 = bL.copy(); L2[a] = v
            nxt += [(blo[m], bhi[m], bi[m], bL, U1), (blo[~m], bhi[~m], bi[~m], L2, bU)]
        if not grew:
            break
        regions = nxt
    return best[1], best[2], best[3]


def single_sizes(slabs, profiles) -> dict:
    """Entries of the two-level table of every tree profile (never more than its blocks)."""
    counts = [int(c) for c in slabs.counts]
    out = {}
    for p in profiles:
        lo, hi = _boxes(slabs, p.items)
        out[p.tree] = min(plan(lo, hi, counts)[0], p.cells)
    return out


class TwoLevel:
    """The two-level tables of all groups of one tree of a compiled index (``ok`` is False if it has none, or if the index
    is not read by vectors)."""

    def __init__(self, index):
        self.ok = False
        fl = index._fast_leaves
        if getattr(index, "_nbase", None) is None or index._ndense is not None or not getattr(index, "_nwide", None):
            return
        pos = {id(r): i for i, r in enumerate(index.roots) if r is not None}
        bound, slabs = index.bound, index.slabs
        d = bound.d
        counts = [int(c) for c in slabs.counts]
        S = sum(counts)
        rowbase = [int(v) for v in np.concatenate(([0], np.cumsum(counts)[:-1]))]
        items = [index.groups[pos[id(l)]][1] for l in fl]
        ntree = [len({bound.boxes[i].tree for i in it}) for it in items]
        self.single = np.array([g for g, t in enumerate(ntree) if t == 1], dtype=np.int64)
        self.multi = np.array([g for g, t in enumerate(ntree) if t != 1], dtype=np.int64)
        if not len(self.single):
            return
        slab_ids = [np.arange(c) for c in counts]
        E_parts, L_parts, W1c, W2c, base1, base2, subfirst, leafbase, VAL = [], [], [], [], [], [], [], [], []
        e_at = l_at = nsub = nleaf = 0
        for g in self.single:
            it = items[g]
            lo, hi = _boxes(slabs, it)
            val = np.array([bound.boxes[i].value for i in it], dtype=np.float64)
            _, ccuts, regions = plan(lo, hi, counts)
            dims1 = [len(c) + 1 for c in ccuts]
            E = np.zeros(dims1, dtype=np.int64)
            col1 = np.zeros(S, dtype=np.int64)
            stride = 1
            for a in reversed(range(d)):
                col1[rowbase[a]:rowbase[a] + counts[a]] = np.searchsorted(np.asarray(ccuts[a], dtype=np.int64), slab_ids[a], side="right") * stride
                stride *= dims1[a]
            subfirst.append(nsub)
            for local, (rlo, rhi, ri, rL, rU) in enumerate(regions):
                sl = tuple(slice(int(np.searchsorted(ccuts[a], rL[a], side="right")), int(np.searchsorted(ccuts[a], rU[a] - 1, side="right")) + 1) for a in range(d))
                E[sl] = local
                rc = _cuts_in(rlo, rhi, rL, rU)
                dims2 = [len(c) + 1 for c in rc]
                G = np.zeros(dims2, dtype=np.int64)
                for b in range(len(rlo)):
                    bs = tuple(slice(int(np.searchsorted(rc[a], max(rlo[b, a], rL[a]), side="right")),
                                     int(np.searchsorted(rc[a], min(rhi[b, a], rU[a]) - 1, side="right")) + 1) for a in range(d))
                    G[bs] = ri[b]
                col2 = np.zeros(S, dtype=np.int64)
                stride = 1
                for a in reversed(range(d)):
                    col2[rowbase[a]:rowbase[a] + counts[a]] = np.searchsorted(rc[a], slab_ids[a], side="right") * stride
                    stride *= dims2[a]
                W2c.append(col2); base2.append(l_at); L_parts.append(G.ravel()); l_at += G.size; nsub += 1
            W1c.append(col1); base1.append(e_at); E_parts.append(E.ravel()); e_at += E.size
            leafbase.append(nleaf); VAL.append(val); nleaf += len(it)
        self.d, self.S, self.rowbase, self.nsub = d, S, rowbase, nsub
        self.th = [list(map(float, t)) for t in slabs.thresholds]
        self.E1 = np.concatenate(E_parts); self.L2 = np.concatenate(L_parts); self.VAL = np.concatenate(VAL)
        self.W1 = np.ascontiguousarray(np.stack(W1c, axis=1)); self.W2 = np.ascontiguousarray(np.stack(W2c, axis=1))
        self.base1, self.base2, self.subfirst, self.leafbase = (np.array(v, dtype=np.int64) for v in (base1, base2, subfirst, leafbase))
        width = lambda a: 1 if int(a.max()) < 256 else 2 if int(a.max()) < 65536 else 4
        # the bytes of a compact layout (1-byte subtree and leaf numbers where they fit); the arrays above use 8-byte numbers for numpy
        self.bytes = dict(coarse=int(self.E1.size * width(self.E1)), leaf_numbers=int(self.L2.size * width(self.L2)), rank1=int(self.W1.size * width(self.W1)),
                          rank2=int(self.W2.size * width(self.W2)), offsets=int(8 * (3 * len(self.single) + nsub)))
        self.entries = int(self.E1.size + self.L2.size)
        self.single_blocks = int(sum(fl[g].cells.overlay.grid_cells for g in self.single))
        self.single_map_bytes = int(sum(fl[g].cells.overlay.grid_cells * np.dtype(fl[g].cells.overlay.face_of_cell.dtype).itemsize for g in self.single))
        self.ok = True


class Hybrid:
    """A read of an index in which only the axes in ``varying`` change between the queries of a unit (as ``PartialRead``):
    groups of several trees by their tables of blocks, groups of one tree by their two-level tables."""

    def __init__(self, index, tl: TwoLevel, varying):
        self.index, self.tl, self.cast = index, tl, index._cast
        d = tl.d
        vary = sorted(set(varying))
        self.vary, self.fixed = vary, [a for a in range(d) if a not in vary]
        touched = {g for g, a, _ in index._contrib if a in set(vary)}
        self.touched = sorted(touched)
        mt = np.array([g for g in tl.multi if int(g) in touched], dtype=np.int64)
        mu = np.array([g for g in tl.multi if int(g) not in touched], dtype=np.int64)
        self.mt, self.mu = mt, mu
        self.w_mt = np.searchsorted(mt, [index._n1, index._n12]).tolist()
        self.w_mu = np.searchsorted(mu, [index._n1, index._n12]).tolist()
        self.f_mt, self.f_mu = index._nfaces[mt], index._nfaces[mu]
        self.base_mt = index._nbase[mt]
        self.rows_fixed = [(a, t, rows) for a, t, rows in index._nwide if a not in vary]
        self.rows_vary = [(a, t, np.ascontiguousarray(rows[:, mt])) for a, t, rows in index._nwide if a in vary]
        ts = np.array([j for j in range(len(tl.single)) if int(tl.single[j]) in touched], dtype=np.int64)
        us = np.array([j for j in range(len(tl.single)) if int(tl.single[j]) not in touched], dtype=np.int64)
        self.ts, self.us = ts, us
        self.W1t = np.ascontiguousarray(tl.W1[:, ts]); self.W1u = np.ascontiguousarray(tl.W1[:, us])
        self.b1t, self.sft, self.lbt = tl.base1[ts], tl.subfirst[ts], tl.leafbase[ts]
        self.b1u, self.sfu, self.lbu = tl.base1[us], tl.subfirst[us], tl.leafbase[us]

    def _two(self, rs, W1, b1, sf, lb, c2_base):
        tl = self.tl
        c1 = b1.copy()
        for r in rs:
            c1 += W1[r]
        sub = sf + tl.E1.take(c1)
        c2 = c2_base.take(sub)
        for r in rs:
            c2 += tl.W2[r].take(sub)
        return float(tl.VAL.take(lb + tl.L2.take(c2)).sum())

    def full(self, x) -> float:
        """All groups; only valid when every axis varies."""
        return self.full_cast(self.cast(x))

    def full_cast(self, cast) -> float:
        index, tl = self.index, self.tl
        v = index._const
        if len(self.mt):
            c = self.base_mt.copy()
            for a, t, rows in self.rows_vary:
                c += rows[bisect_left(t, cast[a])]
            v += float(index._read_cells(c, self.f_mt, *self.w_mt).sum())
        if len(self.ts):
            v += self._two([tl.rowbase[a] + bisect_left(tl.th[a], cast[a]) for a in range(tl.d)], self.W1t, self.b1t, self.sft, self.lbt, tl.base2)
        return v

    def batch(self, x, th) -> np.ndarray:
        """The values of many inputs (rows of ``x``, as the model reads them); only valid when every axis varies."""
        index, tl = self.index, self.tl
        n = len(x)
        slab = [np.searchsorted(th[a], x[:, a], side="left") for a in range(tl.d)]
        v = np.full(n, index._const, dtype=np.float64)
        if len(self.mt):
            c = np.repeat(self.base_mt[:, None], n, axis=1)
            for a, _, rows in self.rows_vary:
                c += rows[slab[a]].T
            v += index._read_cells(c, self.f_mt, *self.w_mt).sum(axis=0)
        if len(self.ts):
            c1 = np.repeat(self.b1t[None, :], n, axis=0)
            for a in range(tl.d):
                c1 += self.W1t[tl.rowbase[a] + slab[a]]
            sub = self.sft[None, :] + tl.E1.take(c1)
            c2 = tl.base2.take(sub)
            for a in range(tl.d):
                c2 += tl.W2[(tl.rowbase[a] + slab[a])[:, None], sub]
            v += tl.VAL.take(self.lbt[None, :] + tl.L2.take(c2)).sum(axis=1)
        return v

    def bind(self, x):
        """What the axes outside ``varying`` give for the values ``x`` of a unit."""
        cast = self.cast(x); index, tl = self.index, self.tl
        c = index._nbase.copy()
        for a, t, rows in self.rows_fixed:
            c += rows[bisect_left(t, cast[a])]
        const = index._const
        if len(self.mu):
            const += float(index._read_cells(c[self.mu], self.f_mu, *self.w_mu).sum())
        if len(self.us):
            const += self._two([tl.rowbase[a] + bisect_left(tl.th[a], cast[a]) for a in range(tl.d)], self.W1u, self.b1u, self.sfu, self.lbu, tl.base2)
        c1f, F2 = self.b1t.copy(), tl.base2.copy()
        for a in self.fixed:
            r = tl.rowbase[a] + bisect_left(tl.th[a], cast[a])
            c1f += self.W1t[r]
            F2 += tl.W2[r]
        return c[self.mt], const, c1f, F2

    def value(self, x, b) -> float:
        cast = self.cast(x); index, tl = self.index, self.tl
        cm, v, c1f, F2 = b
        if len(self.mt):
            c = cm.copy()
            for a, t, rows in self.rows_vary:
                c += rows[bisect_left(t, cast[a])]
            v = v + float(index._read_cells(c, self.f_mt, *self.w_mt).sum())
        if len(self.ts):
            v = v + self._two([tl.rowbase[a] + bisect_left(tl.th[a], cast[a]) for a in self.vary], self.W1t, c1f, self.sft, self.lbt, F2)
        return v
