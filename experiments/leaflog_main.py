#!/usr/bin/env python3
"""The leaves as a log along the MST (LogCellIndex) against the persistent trees (NativeCellIndex), on the cached indexes of
one model: every group of every tree ensemble is built again with the log, from the same items (same groups, same cells).

Per group: cells, trees, blocks; persistent-tree nodes and bytes (8 B per node + 4 B per cell); the changes of the log, the
entries of its checkpoints, its bytes (with the version of every cell); the weight of the MST under the number of changes;
the leaves of sampled cells compared between the two; the time to read the leaves of a cell from each (NumPy and numba),
alternating in the same process.  Output: one JSON line per group in ``--out``/<model>.jsonl and a summary line."""
import argparse, glob, json, os, pickle, sys, time
import numpy as np
from numba import njit
from counterprovenance.native_cells import LogCellIndex, NativeCellIndex


@njit(cache=True)
def walk(root, L, R, H, out):
    """Persistent tree: the item of every slot of the version ``root`` (depth-first)."""
    stack = np.empty(128, np.int64); depth = np.empty(128, np.int64); lo = np.empty(128, np.int64)
    stack[0] = root; depth[0] = 0; lo[0] = 0; sp = 1; width = 1 << H
    while sp > 0:
        sp -= 1
        c = stack[sp]; d = depth[sp]; a = lo[sp]
        if d == H:
            if a < out.shape[0]:
                out[a] = -c - 2
        else:
            half = width >> (d + 1)
            stack[sp] = R[c]; depth[sp] = d + 1; lo[sp] = a + half
            stack[sp + 1] = L[c]; depth[sp + 1] = d + 1; lo[sp + 1] = a
            sp += 2


@njit(cache=True)
def replay(version, slot, item, snap, g, S, direct, out):
    """Log: the item of every slot at ``version`` (the last checkpoint, then the changes in between)."""
    if direct:
        for j in range(g):
            out[j] = np.int64(snap[version * g + j]) - 1
        return
    k = version // S
    for j in range(g):
        out[j] = np.int64(snap[k * g + j]) - 1
    for p in range(k * S, version):
        out[slot[p]] = np.int64(item[p]) - 1


def timed(fn, reps):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="directory of the cached indexes of one model (one .pkl per tree ensemble)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--max-blocks", type=float, default=float("inf"))
    args = ap.parse_args()
    model = os.path.basename(args.dir.rstrip("/")).split("_t")[0]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, model + ".jsonl")
    out = open(path + ".tmp", "w")
    rng = np.random.default_rng(0)
    tot = dict(model=model, groups=0, skipped=0, cells=0, blocks=0, pft_nodes=0, pft_bytes=0, log_bytes=0, changes=0, ckpt=0, omega=0,
               direct=0, mismatches=0, build_s=0.0)
    t00 = time.perf_counter()
    for f in sorted(glob.glob(args.dir + "/*.pkl"), key=os.path.getsize):
        if f.endswith(".tl.pkl"):
            continue
        index, info = pickle.load(open(f, "rb"))
        boxes, slabs = index.bound.boxes, index.slabs
        for gi, leaf in enumerate(getattr(index, "_fast_leaves", None) or []):
            c = leaf.cells
            if not isinstance(c, NativeCellIndex):
                continue
            K, F, g = int(c.overlay.grid_cells), int(c.overlay.n_faces), len(c.slot_trees)
            if K > args.max_blocks:
                tot["skipped"] += 1
                continue
            items, axes = c.items, leaf.axes
            t0 = time.perf_counter()
            new = LogCellIndex(items=items, ranges=[[slabs.ranges[i][a] for a in axes] for i in items], slots=[boxes[i].tree for i in items],
                               slab_counts=[slabs.counts[a] for a in axes], values=[boxes[i].value for i in items], threads=args.threads,
                               spacing_factor=args.spacing)
            build = time.perf_counter() - t0
            assert new.stats.faces == F and new.overlay.grid_cells == K, (F, new.stats.faces)
            nodes = len(c.pft.left)
            L, R, H = np.asarray(c.pft.left, np.int64) if nodes < 5e7 else c.pft.left, None, int(c.pft.height)
            R = np.asarray(c.pft.right, np.int64) if nodes < 5e7 else c.pft.right
            lg = new.log
            cells = rng.choice(F, size=min(F, args.sample), replace=False)
            vo, vn = np.asarray(c.face_version)[cells].astype(np.int64), np.asarray(new.face_version)[cells].astype(np.int64)
            a, b = np.empty(g, np.int64), np.empty(g, np.int64)
            bad = 0
            for x, y in zip(vo, vn):
                walk(x, L, R, H, a); replay(y, lg.slot, lg.item, lg.snap, g, lg.spacing, lg.direct, b)
                bad += int(not np.array_equal(a, b)) + int(not np.array_equal(np.sort(c.pft.items(int(x))), np.sort(lg.items(int(y)))))
            bad += int(not np.allclose(np.asarray(c.face_value)[cells], new.face_value[cells], atol=1e-9, rtol=0))
            n = len(cells)
            reps = max(1, min(20, int(2000 / n)))
            r = dict(old_np=[], new_np=[], old_nb=[], new_nb=[])
            for _ in range(3):      # alternate, keep the best of three
                r["old_np"].append(timed(lambda: [c.pft.items(int(x)) for x in vo], reps) / n)
                r["new_np"].append(timed(lambda: [lg.state(int(y)) for y in vn], reps) / n)
                r["old_nb"].append(timed(lambda: [walk(x, L, R, H, a) for x in vo], reps) / n)
                r["new_nb"].append(timed(lambda: [replay(y, lg.slot, lg.item, lg.snap, g, lg.spacing, lg.direct, b) for y in vn], reps) / n)
            row = dict(node=os.path.basename(f)[:-4], group=gi, trees=g, cells=F, blocks=K, pft_nodes=nodes, pft_bytes=8 * nodes + 4 * F,
                       changes=int(lg.changes), ckpt=int(len(lg.snap)), direct=bool(lg.direct), log_bytes=int(new.leaf_bytes),
                       omega=int(new.stats.tree_weight), bad=bad, build_s=build, **{k: 1e6 * min(v) for k, v in r.items()})
            out.write(json.dumps(row) + "\n"); out.flush()
            tot["groups"] += 1; tot["cells"] += F; tot["blocks"] += K; tot["pft_nodes"] += nodes; tot["pft_bytes"] += row["pft_bytes"]
            tot["log_bytes"] += row["log_bytes"]; tot["changes"] += row["changes"]; tot["ckpt"] += row["ckpt"]; tot["omega"] += row["omega"]
            tot["direct"] += int(lg.direct); tot["mismatches"] += bad; tot["build_s"] += build
            del new, lg, L, R
        del index
    tot["sec"] = time.perf_counter() - t00
    out.close()
    os.replace(path + ".tmp", path)
    print(json.dumps(tot), flush=True)


if __name__ == "__main__":
    main()
