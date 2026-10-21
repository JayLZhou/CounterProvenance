#!/usr/bin/env python3
"""Tune the adaptive unified index: group budget and axis ordering vs persistent K-free."""
from __future__ import annotations
import sys, time, platform
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
import unified_proto as U
from counterprovenance.versioned import CircuitArena

def main():
    print(f"host={platform.node()} python={platform.python_version()}", flush=True)
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    data_home = sys.argv[2] if len(sys.argv) > 2 else None
    X, y, names = U.load_data(data_home)
    rng = np.random.default_rng(2028)
    unit = X[int(rng.integers(len(X)))].copy()
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=0).fit(X, y)
    hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.1, random_state=0).fit(X, y)
    order = list(np.argsort(-gbr.feature_importances_))
    for label, model, kind in (("GBR", gbr, "gbr"), ("HGB", hgb, "hgb")):
        for d in (4, 6, 8):
            S = sorted(order[:d])
            u = U.f32(unit) if kind == "gbr" else unit
            boxes, base = (U.restrict_gbr if kind == "gbr" else U.restrict_hgbr)(model, u, S)
            slabs = U.Slabs(boxes, d)
            lo = np.percentile(X[:, S], 2, axis=0); hi = np.percentile(X[:, S], 98, axis=0)
            pts = rng.uniform(lo, hi, size=(300, d))
            if kind == "gbr":
                pts = U.f32(pts)
            ref = U.model_predict_rows(model, u, S, pts)
            rows = []
            kp = U.KFreePersistent(CircuitArena(), boxes, slabs)
            t0 = time.perf_counter(); pcs = []; err = 0
            for i, p in enumerate(pts):
                ps = kp.query(p); pcs.append(len(ps)); err = max(err, abs(base + sum(v for *_, v in ps) - ref[i]))
            rows.append(("kfree_persistent", kp.total_records, np.mean(pcs), max(pcs), (time.perf_counter()-t0)/len(pts)*1e6, err))
            for budget in (None, 16, 8, 4, 2, 1):
                for ao in ("given", "bycount"):
                    if budget is None and ao == "bycount":
                        pass
                    t0 = time.perf_counter()
                    un = U.UnifiedIndex(boxes, slabs, tau=4096, group_budget=budget, axis_order=ao)
                    b = time.perf_counter() - t0
                    t0 = time.perf_counter(); pcs = []; err = 0
                    for i, p in enumerate(pts):
                        ps = un.query(p); pcs.append(len(ps)); err = max(err, abs(base + sum(v for _, v in ps) - ref[i]))
                    q = (time.perf_counter()-t0)/len(pts)*1e6
                    rows.append((f"unified g={'all' if budget is None else budget} {ao} ({len(un.groups)} groups, {un.leaf_count} leaves)", un.records, np.mean(pcs), max(pcs), q, err))
            print(f"[{label} d={d}] N={len(boxes)} groups(axis-support)={len(U.UnifiedIndex(boxes, slabs, tau=4096).groups)}")
            for name, rec, pm, px, q, e in rows:
                print(f"   {name:<48} records={rec:>9,} pieces mean/max={pm:6.1f}/{px:<4} query={q:6.1f}us max_err={e:.1e}", flush=True)

if __name__ == "__main__":
    main()
