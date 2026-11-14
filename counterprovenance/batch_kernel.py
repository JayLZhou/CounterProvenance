"""A compiled batch read of a ``ProvenanceIndex``: the same reads as ``query_batch``, fused into one loop per row.

``query_batch`` reads many interventions with NumPy: for a model with G groups it adds a (rows x G) matrix of cell
offsets per axis and gathers through it, so that its time is the traffic of these temporaries.  Here a row walks its
G groups once: the cell of a group is its base plus the offsets of the row's slabs on the axes, then one read through
the map (or the dense table).  Libraries evaluate their trees in compiled code; this is TWIN at the same level."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _fused(S, wides, offs, cnt, G, nbase, dense, ndense, n1, n12, map8, map16, map32, faces, nvalue, const, out):
    """Group by group, as ``query_batch``: the reads of all rows stay in the table of one group at a time."""
    R, A = S.shape
    for r in range(R):
        out[r] = 0.0
    for g in range(G):
        base = nbase[g]
        for r in range(R):
            c = base
            for k in range(A):
                c += wides[offs[k] + g * cnt[k] + S[r, k]]
            if dense:
                out[r] += ndense[c]
            else:
                if g < n1:
                    f = np.int64(map8[c])
                elif g < n12:
                    f = np.int64(map16[c])
                else:
                    f = np.int64(map32[c])
                out[r] += nvalue[f + faces[g]]
    for r in range(R):
        out[r] += const


class FusedBatch:
    """Prepared once per index (the flat copy of its rank rows); ``__call__`` reads many interventions."""

    def __init__(self, index) -> None:
        self.index = index
        self.ok = bool(index._fast and index._nbase is not None and not getattr(index, "_baxes", None))
        if not self.ok:
            return
        self.axes = [a for a, _, _ in index._nwide]
        self.th = [np.asarray(t, dtype=np.float64) for t in index.slabs.thresholds]
        G = len(index._nbase)
        self.G = G
        # rank rows transposed, (group, slab): a group reads one contiguous row per axis
        self.wides = np.concatenate([np.ascontiguousarray(np.asarray(w, dtype=np.int64).T).ravel() for _, _, w in index._nwide]) \
            if index._nwide else np.zeros(0, dtype=np.int64)
        self.offs = np.cumsum([0] + [w.size for _, _, w in index._nwide])[:-1].astype(np.int64)
        self.cnt = np.array([w.shape[0] for _, _, w in index._nwide], dtype=np.int64)
        self.dense = index._ndense is not None
        e8, e16, e32 = np.zeros(1, np.uint8), np.zeros(1, np.uint16), np.zeros(1, np.uint32)
        if self.dense:
            self.maps = (e8, e16, e32)
            self.ndense = np.ascontiguousarray(index._ndense, dtype=np.float64)
            self.faces = np.zeros(G, dtype=np.int64)
            self.nvalue = np.zeros(1)
        else:
            m = index._nmaps
            self.maps = (m[0] if len(m[0]) else e8, m[1] if len(m[1]) else e16, m[2] if len(m[2]) else e32)
            self.ndense = np.zeros(1)
            self.faces = np.ascontiguousarray(index._nfaces, dtype=np.int64)
            self.nvalue = np.ascontiguousarray(index._nvalue, dtype=np.float64)
        self.nbase = np.ascontiguousarray(index._nbase, dtype=np.int64)

    def __call__(self, T: np.ndarray) -> np.ndarray:
        if not self.ok:
            return self.index.query_batch(T)
        ix = self.index
        X = ix._read_as_model(np.asarray(T, dtype=np.float64))
        S = np.empty((len(X), len(self.axes)), dtype=np.int64)
        for k, a in enumerate(self.axes):
            S[:, k] = np.searchsorted(self.th[a], X[:, a], side="left")
        out = np.empty(len(X), dtype=np.float64)
        _fused(S, self.wides, self.offs, self.cnt, self.G, self.nbase, self.dense, self.ndense, ix._n1 if not self.dense else 0,
               ix._n12 if not self.dense else 0, self.maps[0], self.maps[1], self.maps[2], self.faces, self.nvalue,
               float(ix._const), out)
        return out
