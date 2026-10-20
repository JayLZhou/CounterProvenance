#!/usr/bin/env python3
"""All external baselines in one file, compared at the same implementation level.

For one dataset, the paper zoo, a fixed unit and ``d`` intervened features
(the same unit, features and query points as ``query_bench.py``: the random
stream is replayed), every query ``do(T = t)`` is answered by

* ours, compiled - ``accel.NumbaIndex`` (numba);
* Treelite - the model compiled to C by tl2cgen (gcc -O3), single-threaded;
* QuickScorer - the bitvector traversal of Lucchese et al. over all trees and
  the full feature vector, compiled with numba;
* QuickScorer-B - QuickScorer with the unit bound once: the false nodes of the
  non-intervened features are applied at build time, a query only scans the
  intervened ones (binding is our first step, so this isolates the index);
* DoWhy-GCM - ``counterfactual_samples`` on a star SCM (``--parts dowhy``).

Native code is only compared with native code.  Every method is timed with
one call per query (``*_call_us``, the interactive case; Treelite builds a
one-row DMatrix per call) and with one call for all queries (``*_batch_us``;
Treelite's DMatrix is built outside the timer).  Treelite receives the inputs
as the library reads them (``ensemble.cast``: float32 for scikit-learn and
XGBoost, LightGBM's zero threshold), otherwise it would route points that lie
exactly on a split threshold differently from the library.  Values are checked against
the online loop (ours, QuickScorer) and the library (Treelite, DoWhy).
Settings whose index took longer than ``--max-build`` seconds in
``results/qbench_<dataset>_<model>.csv`` reuse the numba timings of that run
(same unit and points) instead of rebuilding the index.

Output: ``results/baselines_<tag>.csv`` and ``results/baselines_dowhy_<tag>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import math
import os
import tempfile
import time
from collections import Counter

import numpy as np

from common import RESULTS, environment, load_dataset, paper_zoo, per_call_seconds, value_tolerance, write_csv
from counterprovenance import ProvenanceIndex, bind, from_model
from counterprovenance import accel
from query_bench import query_points

from numba import njit

_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)


# ---------------------------------------------------------------------------
# QuickScorer (Lucchese et al., SIGIR 2015), compiled with numba
# ---------------------------------------------------------------------------
@njit(cache=True)
def _qs_scan(x, f_lo, f_hi, thr, n_word0, n_word1, n_mask0, n_mask1, n_mid, bv):
    """Apply the false nodes of every feature in ``f_lo``..``f_hi`` (nodes of a
    feature sorted by threshold) for the feature vector ``x``."""
    for f in range(f_lo.shape[0]):
        v = x[f]
        k = f_lo[f]
        end = f_hi[f]
        while k < end and thr[k] < v:  # test x <= thr is false: drop the left subtree
            w0 = n_word0[k]
            w1 = n_word1[k]
            bv[w0] &= n_mask0[k]
            if w1 != w0:
                for w in range(w0 + 1, w1):
                    bv[w] = n_mid[k]
                bv[w1] &= n_mask1[k]
            k += 1


@njit(cache=True)
def _qs_exit(bv, t_off, t_words, leaf_val, leaf_off, db):
    total = 0.0
    for m in range(t_off.shape[0]):
        base = t_off[m]
        for j in range(t_words[m]):
            w = bv[base + j]
            if w != 0:
                low = w & (~w + np.uint64(1))
                idx = db[np.int64((low * _DEBRUIJN) >> np.uint64(58))]
                total += leaf_val[leaf_off[m] + j * 64 + idx]
                break
    return total


@njit(cache=True)
def _qs_query(t, x0, axes, f32, zt, init, q_lo, q_hi, thr, n_word0, n_word1, n_mask0, n_mask1, n_mid,
              t_off, t_words, leaf_val, leaf_off, db, const, x, bv):
    for i in range(x0.shape[0]):
        x[i] = x0[i]
    for k in range(axes.shape[0]):
        v = t[k]
        if f32 == 1:
            v = np.float64(np.float32(v))
        if -zt <= v <= zt:
            v = 0.0
        x[axes[k]] = v
    for i in range(init.shape[0]):
        bv[i] = init[i]
    _qs_scan(x, q_lo, q_hi, thr, n_word0, n_word1, n_mask0, n_mask1, n_mid, bv)
    return const + _qs_exit(bv, t_off, t_words, leaf_val, leaf_off, db)


@njit(cache=True)
def _qs_batch(T, out, x0, axes, f32, zt, init, q_lo, q_hi, thr, n_word0, n_word1, n_mask0, n_mask1, n_mid,
              t_off, t_words, leaf_val, leaf_off, db, const, x, bv):
    for q in range(T.shape[0]):
        out[q] = _qs_query(T[q], x0, axes, f32, zt, init, q_lo, q_hi, thr, n_word0, n_word1, n_mask0,
                           n_mask1, n_mid, t_off, t_words, leaf_val, leaf_off, db, const, x, bv)


sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))


class QuickScorer:
    """QuickScorer over the trees of ``ensemble``.  With ``bound_unit=True`` the
    false nodes of the non-intervened features are applied once at build time,
    and a query only scans the nodes of the intervened features."""

    def __init__(self, ensemble, unit, axes, noise=0.0, bound_unit=False) -> None:
        p = len(unit)
        trees = ensemble.trees
        t_off, t_words, leaf_off, leaf_val = [], [], [], []
        nodes = [[] for _ in range(p)]  # per feature: (thr, word0, word1, mask0, mask1)
        word = 0
        lv_off = 0
        for tr in trees:
            feat, thr, left, right, val = tr.feature, tr.threshold, tr.left, tr.right, tr.value
            order, span = [], {}

            def visit(node):  # leaves left to right; span[node] = leaf range of the subtree
                if feat[node] < 0:
                    order.append(node)
                    span[node] = (len(order) - 1, len(order))
                    return span[node]
                lo, _ = visit(int(left[node]))
                _, hi = visit(int(right[node]))
                span[node] = (lo, hi)
                return span[node]

            visit(0)
            L = len(order)
            W = (L + 63) // 64
            for node in range(len(feat)):
                if feat[node] < 0:
                    continue
                lo, hi = span[int(left[node])]  # the left subtree's leaves are cleared
                w0, w1 = word + lo // 64, word + (hi - 1) // 64
                b0, b1 = lo % 64, (hi - 1) % 64 + 1
                keep_low = (1 << b0) - 1
                if w0 == w1:
                    clear = ((1 << b1) - 1) & ~keep_low
                    mask0 = mask1 = ((1 << 64) - 1) & ~clear
                else:
                    mask0 = keep_low
                    mask1 = ((1 << 64) - 1) & ~((1 << b1) - 1)
                nodes[int(feat[node])].append((float(thr[node]), w0, w1, mask0, mask1))
            t_off.append(word)
            t_words.append(W)
            leaf_off.append(lv_off)
            vals = np.zeros(W * 64)
            vals[:L] = [val[n] for n in order]
            leaf_val.append(vals)
            word += W
            lv_off += W * 64
        init = np.zeros(word, dtype=np.uint64)
        for m, tr in enumerate(trees):
            L = int(np.sum(tr.feature < 0))
            for j in range(t_words[m]):
                bits = min(64, L - 64 * j)
                init[t_off[m] + j] = np.uint64((1 << bits) - 1) if bits < 64 else np.uint64((1 << 64) - 1)
        thr_all, w0_all, w1_all, m0_all, m1_all, f_lo, f_hi = [], [], [], [], [], [], []
        for f in range(p):
            lst = sorted(nodes[f], key=lambda r: r[0])
            f_lo.append(len(thr_all))
            for r in lst:
                thr_all.append(r[0]); w0_all.append(r[1]); w1_all.append(r[2]); m0_all.append(r[3]); m1_all.append(r[4])
            f_hi.append(len(thr_all))
        self.thr = np.array(thr_all, dtype=np.float64)
        self.w0 = np.array(w0_all, dtype=np.int64)
        self.w1 = np.array(w1_all, dtype=np.int64)
        self.m0 = np.array(m0_all, dtype=np.uint64)
        self.m1 = np.array(m1_all, dtype=np.uint64)
        self.mid = np.zeros(len(thr_all), dtype=np.uint64)  # whole words inside a cleared range
        f_lo = np.array(f_lo, dtype=np.int64)
        f_hi = np.array(f_hi, dtype=np.int64)
        self.t_off = np.array(t_off, dtype=np.int64)
        self.t_words = np.array(t_words, dtype=np.int64)
        self.leaf_off = np.array(leaf_off, dtype=np.int64)
        self.leaf_val = np.concatenate(leaf_val) if leaf_val else np.zeros(0)
        self.db = accel._DB.copy()
        self.f32 = int(ensemble.input_dtype == np.float32)
        self.zt = ensemble.zero_threshold if ensemble.zero_threshold > 0 else -1.0
        self.x0 = ensemble.cast(np.asarray(unit, dtype=float).reshape(1, -1))[0].astype(np.float64)
        self.axes = np.asarray(axes, dtype=np.int64)
        self.const = float(ensemble.base + noise)
        self.x = self.x0.copy()
        self.bv = init.copy()
        if bound_unit:
            # apply the non-intervened features once; queries scan the intervened ones only
            fixed = np.ones(p, dtype=bool)
            fixed[self.axes] = False
            hi_fixed = np.where(fixed, f_hi, f_lo)
            bv = init.copy()
            _qs_scan(self.x0, f_lo, hi_fixed, self.thr, self.w0, self.w1, self.m0, self.m1, self.mid, bv)
            self.init = bv
            self.q_lo = f_lo
            self.q_hi = np.where(fixed, f_lo, f_hi)
        else:
            self.init = init
            self.q_lo, self.q_hi = f_lo, f_hi
        self.nodes = len(thr_all)
        self.words = word

    def _args(self):
        return (self.x0, self.axes, self.f32, self.zt, self.init, self.q_lo, self.q_hi, self.thr, self.w0, self.w1,
                self.m0, self.m1, self.mid, self.t_off, self.t_words, self.leaf_val, self.leaf_off, self.db, self.const,
                self.x, self.bv)

    def query(self, t):
        return _qs_query(t, *self._args())

    def batch(self, T):
        out = np.empty(len(T))
        _qs_batch(np.ascontiguousarray(T, dtype=np.float64), out, *self._args())
        return out


# ---------------------------------------------------------------------------
# Treelite (tl2cgen: model -> C -> shared library)
# ---------------------------------------------------------------------------
def treelite_predictor(model, workdir):
    import treelite
    import tl2cgen

    module = type(model).__module__
    if module.startswith("xgboost"):
        import json

        # XGBoost 3.x adds an (empty, no categorical features) "cats" block that Treelite 4.1 cannot parse
        doc = json.loads(model.get_booster().save_raw("json"))
        gb = doc["learner"]["gradient_booster"]
        for holder in (gb, gb.get("model", {})):
            cats = holder.get("cats")
            if cats is not None:
                assert not any(cats.get(k) for k in ("enc", "feature_segments", "sorted_idx")), "categorical model"
                del holder["cats"]
        # XGBoost 3.x writes base_score as a one-element vector "[2.07E0]"; Treelite 4.1 expects a scalar
        param = doc["learner"]["learner_model_param"]
        bs = param.get("base_score", "")
        if bs.startswith("["):
            values = bs.strip("[]").split(",")
            assert len(values) == 1, "multi-target model"
            param["base_score"] = values[0]
        tl = treelite.frontend.from_xgboost_json(json.dumps(doc))
    elif module.startswith("lightgbm"):
        tl = treelite.frontend.from_lightgbm(model.booster_)
    else:
        tl = treelite.sklearn.import_model(model)
    libpath = os.path.join(workdir, "model.so")
    start = time.perf_counter()
    tl2cgen.export_lib(tl, toolchain="gcc", libpath=libpath, params={"parallel_comp": 32}, nthread=1,
                       options=["-O3"])
    compile_s = time.perf_counter() - start
    predictor = tl2cgen.Predictor(libpath, nthread=1)
    dtype = "float32" if "32" in str(getattr(predictor, "threshold_type", "float64")) else "float64"
    return tl2cgen, predictor, dtype, compile_s


def treelite_predict(tl2cgen, predictor, dmat):
    return np.asarray(predictor.predict(dmat)).reshape(-1)


# ---------------------------------------------------------------------------
def previous_rows(dataset, name, trees):
    path = RESULTS / f"qbench_{dataset}_{name}.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {int(r["d"]): r for r in csv.DictReader(handle) if int(r["trees"]) == trees}


def bench(dataset, X, y, name, model, dims, n_queries, budget, max_build, rng, workdir):
    ensemble = from_model(model)
    usage = Counter(int(f) for t in ensemble.trees for f in t.feature if f >= 0)
    order = [f for f, _ in usage.most_common()] + [f for f in range(X.shape[1]) if f not in usage]
    unit_index = int(rng.integers(len(X)))  # same stream as query_bench.bench
    unit = X[unit_index]
    noise = float(y[unit_index] - model.predict(unit[None, :])[0])
    tol = value_tolerance(model)
    prev = previous_rows(dataset, name, ensemble.tree_count)
    tl2cgen, predictor, dtype, compile_s = treelite_predictor(model, workdir)
    rows = []
    for d in dims:
        if d > len(usage):
            continue
        axes = sorted(order[:d])
        bound = bind(ensemble, unit, axes, noise=noise)
        pts = query_points(X, axes, bound, rng, n_queries)
        n = len(pts)
        for _ in range(300):  # replay the slider steps of query_bench
            int(rng.integers(d))
            int(rng.integers(n))
        T = np.ascontiguousarray(pts, dtype=np.float64)
        full = np.tile(unit, (n, 1))
        full[:, axes] = T
        lib = model.predict(full) + noise
        row = {"dataset": dataset, "features": X.shape[1], "model": name, "trees": ensemble.tree_count, "d": d,
               "queries": n, "treelite_compile_s": compile_s}

        # --- online loop (reference) -----------------------------------
        loop = accel.NumbaLoop(ensemble, unit, axes, noise=noise)
        loop.query(T[0])
        ref = loop.batch(T)
        row["nb_loop_call_us"] = per_call_seconds(lambda: [loop.query(t) for t in T], n) * 1e6
        row["nb_loop_batch_us"] = per_call_seconds(lambda: loop.batch(T), n) * 1e6

        # --- ours ------------------------------------------------------
        p = prev.get(d)
        if p is not None and float(p["build_s"]) > max_build:
            row["ours_source"] = "qbench"
            row["build_s"] = float(p["build_s"])
            row["nb_ours_call_us"] = float(p["nb_ours_call_us"])
            row["nb_ours_batch_us"] = float(p["nb_ours_batch_us"])
        else:
            start = time.perf_counter()
            index = ProvenanceIndex(bound, budget=budget)
            row["build_s"] = time.perf_counter() - start
            nb = accel.NumbaIndex(index)
            nb.query(T[0])
            got = nb.batch(T)
            err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
            if err > 1e-12:
                raise AssertionError(f"{dataset} {name} d={d}: ours differs from the loop by {err}")
            row["ours_source"] = "this run"
            row["nb_ours_call_us"] = per_call_seconds(lambda: [nb.query(t) for t in T], n) * 1e6
            row["nb_ours_batch_us"] = per_call_seconds(lambda: nb.batch(T), n) * 1e6
            del index, nb

        # --- QuickScorer -----------------------------------------------
        for label, bound_unit in (("qs", False), ("qsb", True)):
            start = time.perf_counter()
            qs = QuickScorer(ensemble, unit, axes, noise=noise, bound_unit=bound_unit)
            row[f"{label}_build_s"] = time.perf_counter() - start
            qs.query(T[0])
            got = qs.batch(T)
            err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
            if err > 1e-9:
                raise AssertionError(f"{dataset} {name} d={d}: {label} differs from the loop by {err}")
            row[f"{label}_call_us"] = per_call_seconds(lambda: [qs.query(t) for t in T], n) * 1e6
            row[f"{label}_batch_us"] = per_call_seconds(lambda: qs.batch(T), n) * 1e6
            row[f"{label}_nodes"] = qs.nodes
            row[f"{label}_words"] = qs.words

        # --- Treelite --------------------------------------------------
        # the model's own input semantics (sklearn compares float32 inputs, LightGBM zeroes tiny values)
        F = np.ascontiguousarray(ensemble.cast(full), dtype=np.float32 if dtype == "float32" else np.float64)
        dm = tl2cgen.DMatrix(F, dtype=dtype)
        got = treelite_predict(tl2cgen, predictor, dm) + noise
        rel = np.abs(got - lib) / np.maximum(1.0, np.abs(lib))
        wrong = np.flatnonzero(rel > max(tol, 1e-6))
        # tl2cgen prints thresholds as decimal literals and may round them, so a point that lies
        # exactly on a split threshold can take the other branch; elsewhere Treelite must be exact
        n_uniform = n_queries
        if len(wrong) and wrong.min() < n_uniform:
            raise AssertionError(f"{dataset} {name} d={d}: Treelite differs from the library off thresholds")
        row["treelite_max_rel_err_off_thresholds"] = float(rel[:n_uniform].max())
        row["treelite_wrong_on_thresholds"] = int(len(wrong))
        rows_1 = [F[q:q + 1] for q in range(n)]
        row["treelite_call_us"] = per_call_seconds(
            lambda: [predictor.predict(tl2cgen.DMatrix(r, dtype=dtype)) for r in rows_1], n) * 1e6
        row["treelite_batch_us"] = per_call_seconds(lambda: predictor.predict(dm), n) * 1e6
        rows.append(row)
        print(f"{dataset:5s} {name:12s} d={d:2d} ours({row['ours_source']}) call {row['nb_ours_call_us']:6.2f} "
              f"batch {row['nb_ours_batch_us']:7.3f} | treelite call {row['treelite_call_us']:7.2f} "
              f"batch {row['treelite_batch_us']:7.3f} | qs batch {row['qs_batch_us']:7.3f} "
              f"qsb batch {row['qsb_batch_us']:7.3f} | loop batch {row['nb_loop_batch_us']:7.3f}", flush=True)
    return rows


def run_dowhy(args, X, y, title):
    """DoWhy-GCM on a star SCM; same logic as dowhy_bench.py, with our numba index."""
    import networkx as nx
    import pandas as pd
    from dowhy import gcm
    from dowhy.gcm.ml import SklearnRegressionModel
    from dowhy.graph import get_ordered_predecessors

    gcm.config.disable_progress_bars()
    names = [f"f{i:04d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=names)
    df["Y"] = y
    zoo = paper_zoo(args.trees)
    if args.models:
        zoo = {k: v for k, v in zoo.items() if k in args.models.split(",")}
    rows = []
    for name, factory in zoo.items():
        rng = np.random.default_rng(11)
        unit = df.iloc[[int(rng.integers(len(df)))]].copy()
        scm = gcm.InvertibleStructuralCausalModel(nx.DiGraph([(f, "Y") for f in names]))
        for f in names:
            scm.set_causal_mechanism(f, gcm.EmpiricalDistribution())
        scm.set_causal_mechanism("Y", gcm.AdditiveNoiseModel(SklearnRegressionModel(factory())))
        start = time.perf_counter()
        gcm.fit(scm, df)
        fit_s = time.perf_counter() - start
        model = scm.causal_mechanism("Y").prediction_model.sklearn_model
        assert get_ordered_predecessors(scm.graph, "Y") == names
        x_unit = unit[names].to_numpy()[0]
        noise = float(unit["Y"].iloc[0] - model.predict(x_unit[None, :])[0])
        noise_df = unit.copy()
        noise_df["Y"] = noise
        ensemble = from_model(model)
        usage = Counter(int(f) for t in ensemble.trees for f in t.feature if f >= 0)
        order = [f for f, _ in usage.most_common()]
        prev = previous_rows(args.dataset, name, ensemble.tree_count)
        for d in (int(v) for v in args.dims.split(",")):
            if d > len(order):
                continue
            axes = sorted(order[:d])
            lo = np.percentile(X[:, axes], 2, axis=0)
            hi = np.percentile(X[:, axes], 98, axis=0)
            pts = np.ascontiguousarray(rng.uniform(lo, hi, size=(args.dowhy_queries, d)))
            interventions = [{names[a]: (lambda _v, value=float(t[j]): value) for j, a in enumerate(axes)} for t in pts]
            vals = np.array([float(gcm.counterfactual_samples(scm, iv, observed_data=unit)["Y"].iloc[0])
                             for iv in interventions])
            loop = accel.NumbaLoop(ensemble, x_unit, axes, noise=noise)
            ref = loop.batch(pts)
            err = float(np.max(np.abs(vals - ref) / np.maximum(1.0, np.abs(ref))))
            if err > max(value_tolerance(model), 1e-9):
                raise AssertionError(f"{name} d={d}: DoWhy and the loop differ by {err}")
            k = len(pts)
            fresh = per_call_seconds(
                lambda: [gcm.counterfactual_samples(scm, iv, observed_data=unit) for iv in interventions], k, repeats=1)
            cached = per_call_seconds(
                lambda: [gcm.counterfactual_samples(scm, iv, noise_data=noise_df) for iv in interventions], k, repeats=1)
            row = {"dataset": args.dataset, "features": X.shape[1], "model": name, "trees": ensemble.tree_count,
                   "d": d, "fit_s": fit_s, "dowhy_queries": k, "max_rel_err_vs_loop": err,
                   "dowhy_fresh_us": fresh * 1e6, "dowhy_cached_us": cached * 1e6}
            p = prev.get(d)
            if p is not None:  # same model (deterministic fit), timings of the full workload
                row["nb_ours_call_us"] = float(p["nb_ours_call_us"])
                row["py_ours_us"] = float(p["py_ours_us"])
            rows.append(row)
            print(f"{args.dataset} {name:12s} d={d:2d} DoWhy {row['dowhy_fresh_us']:9.0f}us cached "
                  f"{row['dowhy_cached_us']:9.0f}us err {err:.1e}", flush=True)
            write_csv(f"baselines_dowhy_{args.tag or args.dataset}.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["cal", "yprop", "topo", "uji", "mars"])
    parser.add_argument("--models", default="", help="comma-separated subset of the paper zoo")
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--dims", default="1,2,4,8,16")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--budget", type=int, default=2_000_000)
    parser.add_argument("--max-build", type=float, default=600.0,
                        help="reuse the qbench timings of ours when its index took longer to build")
    parser.add_argument("--dowhy-queries", type=int, default=40)
    parser.add_argument("--parts", default="native,dowhy", help="native (Treelite, QuickScorer, ours) and/or dowhy")
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    print(environment(), flush=True)
    X, y, title = load_dataset(args.dataset, args.data_home)
    print(f"{title}: {X.shape}", flush=True)
    parts = set(args.parts.split(","))
    if "native" in parts:
        dims = [int(v) for v in args.dims.split(",")]
        zoo = paper_zoo(args.trees)
        if args.models:
            zoo = {k: v for k, v in zoo.items() if k in args.models.split(",")}
        rows = []
        for name, factory in zoo.items():
            start = time.perf_counter()
            model = factory().fit(X, y)
            print(f"{name} fitted in {time.perf_counter() - start:.1f}s", flush=True)
            rng = np.random.default_rng(2028)
            with tempfile.TemporaryDirectory() as workdir:
                rows += bench(args.dataset, X, y, name, model, dims, args.queries, args.budget, args.max_build, rng,
                              workdir)
            write_csv(f"baselines_{args.tag or args.dataset}.csv", rows)
    if "dowhy" in parts:
        run_dowhy(args, X, y, title)


if __name__ == "__main__":
    main()
