#!/usr/bin/env python3
"""End-to-end applications that issue many counterfactual queries on one unit, with TWIN embedded or with DoWhy-GCM.

Every application only sees an oracle ``f(Q) -> values`` (one row per intervention), so that it is the same code on
both backends, and both answer a batch at once, in one thread.  The reported time is the run of the application; the
build of TWIN is recorded separately.  DoWhy-GCM gets its most favorable use: the noise is abducted once, and every
call passes the whole batch of the application.
Output: ``results/e2e_<dataset>_<family>_d<d>_u<unit>.csv``, one row per (application, backend)."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import resource
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from common import write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import Infeasible, ancestors, bind_scm, fit_scm, parents_of
from scm_bench import choose_treatments, zoo
from scm_full import dowhy_model


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout


class Oracle:
    """Counts the queries and calls and keeps the latency of the calls with one row."""

    def __init__(self, fn):
        self.fn, self.queries, self.calls, self.single = fn, 0, 0, []

    def __call__(self, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(Q)
        self.queries += len(Q)
        self.calls += 1
        if len(Q) == 1:
            t0 = time.perf_counter()
            y = self.fn(Q)
            self.single.append(time.perf_counter() - t0)
            return y
        return self.fn(Q)


class Twin:
    def __init__(self, scm, unit, T, y, budget, threads):
        t0 = time.perf_counter()
        self.index = ProvenanceIndex(bind_scm(scm, unit, T, y, threads=threads, native=True), grouping="tables",
                                     budget=int(budget), build_threads=threads)
        self.build_s = time.perf_counter() - t0
        st = self.index.stats
        self.info = dict(groups=len(self.index.groups), choice=self.index.choice, grid_cells=st.grid_cells,
                         faces=st.faces, pft_nodes=st.pft_nodes, est_bytes=st.packed_bytes)

    def __call__(self, Q):
        if len(Q) == 1:
            return np.array([self.index.query(Q[0]).value])
        return self.index.query_batch(Q)


class DoWhy:
    def __init__(self, gcm, cg, row, T, y):
        from dowhy.gcm._noise import compute_noise_from_data
        self.gcm, self.cg, self.T, self.y = gcm, cg, T, y
        t0 = time.perf_counter()
        self.noise = compute_noise_from_data(cg, row)
        self.build_s = time.perf_counter() - t0
        self.info = {}

    def __call__(self, Q):
        its = [iter(Q[:, j].tolist()) for j in range(len(self.T))]
        iv = {name: (lambda _v, it=it: next(it)) for name, it in zip(self.T, its)}
        nd = pd.DataFrame(np.repeat(self.noise.to_numpy(), len(Q), axis=0), columns=self.noise.columns)
        return self.gcm.counterfactual_samples(self.cg, iv, noise_data=nd)[self.y].to_numpy()


class Ctx:
    """What an application knows about a unit: the box of the interventions, the factual point and the scales."""

    def __init__(self, df, T, row, y, seed):
        self.T, self.d = T, len(T)
        self.lo, self.hi = df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy()
        self.hi = np.where(self.hi > self.lo, self.hi, self.lo + 1.0)
        self.u = np.clip(row[T].to_numpy(dtype=float)[0], self.lo, self.hi)
        self.scale = np.maximum(df[T].std().to_numpy(), 1e-9)
        self.bg = df[T].sample(min(len(df), 2000), random_state=seed).to_numpy(dtype=float)
        self.seed = seed


# ---- what-if analysis -------------------------------------------------------------------------------------------
def what_if_surface(f, c):
    """A dashboard of what-if surfaces: one per pair of intervened variables, the others stay factual."""
    k = 200 if c.d == 2 else 100
    total = 0.0
    if c.d == 1:
        return float(f(np.linspace(c.lo, c.hi, 40_000)).sum())
    for a, b in itertools.combinations(range(c.d), 2):
        g1, g2 = np.meshgrid(np.linspace(c.lo[a], c.hi[a], k), np.linspace(c.lo[b], c.hi[b], k))
        Q = np.tile(c.u, (k * k, 1))
        Q[:, a], Q[:, b] = g1.ravel(), g2.ravel()
        total += float(f(Q).sum())
    return total


def what_if_slider(f, c, events=2000):
    """A user moves one slider at a time; every move is answered at once, one query."""
    rng = np.random.default_rng(c.seed)
    t = c.u.copy()
    total = 0.0
    for _ in range(events):
        a = rng.integers(c.d)
        t[a] = rng.uniform(c.lo[a], c.hi[a])
        total += float(f(t[None, :])[0])
    return total


def dose_response(f, c, levels=1000):
    """The individual dose-response curve of every intervened variable."""
    total = 0.0
    for a in range(c.d):
        Q = np.tile(c.u, (levels, 1))
        Q[:, a] = np.linspace(c.lo[a], c.hi[a], levels)
        total += float(f(Q).sum())
    return total


# ---- recourse -----------------------------------------------------------------------------------------------------
def _targets(f, c, n=200):
    rng = np.random.default_rng(c.seed)
    y = f(rng.uniform(c.lo, c.hi, size=(n, c.d)))
    return np.quantile(y, [0.6, 0.7, 0.8, 0.9, 0.95])


def recourse_genetic(f, c, restarts=4, cfs=4, max_iter=500):
    """The genetic search of DiCE (population 10 x counterfactuals, at most 500 iterations, stop after 5 stable ones,
    the top 10% survive, the others cross the top half and mutate), on interventions instead of features."""
    total = 0.0
    for r, tau in enumerate(_targets(f, c)):
        for s in range(restarts):
            rng = np.random.default_rng([c.seed, r, s])
            pop = rng.uniform(c.lo, c.hi, size=(10 * cfs, c.d))
            keep = rng.random(pop.shape) < 0.5
            pop = np.where(keep, c.u, pop)

            def loss(P):
                y = f(P)
                dist = np.abs(P - c.u) / c.scale
                return np.maximum(tau - y, 0.0) + 0.2 * dist.sum(axis=1) / c.d + 0.2 * (dist > 0).mean(axis=1)

            best, stable = math.inf, 0
            for _ in range(max_iter):
                ls = loss(pop)
                order = np.argsort(ls, kind="stable")
                pop = pop[order]
                if abs(ls[order[0]] - best) <= 1e-6:
                    stable += 1
                    if stable >= 5:
                        break
                else:
                    stable = 0
                best = float(ls[order[0]])
                n_top = max(1, len(pop) // 10)
                parents = pop[: max(2, len(pop) // 2)]
                a = parents[rng.integers(len(parents), size=len(pop) - n_top)]
                b = parents[rng.integers(len(parents), size=len(pop) - n_top)]
                kids = np.where(rng.random(a.shape) < 0.5, a, b)
                mutate = rng.random(kids.shape) < 0.1
                kids = np.where(mutate, rng.uniform(c.lo, c.hi, size=kids.shape), kids)
                pop = np.vstack((pop[:n_top], kids))
            total += best
    return total


def recourse_minimal(f, c, levels=20, max_size=3):
    """Causal recourse through minimal interventions (MINT): the cheapest action on at most three variables that
    reaches every target, by enumeration of the action sets and a grid of their values."""
    taus = _targets(f, c)
    best = np.full(len(taus), np.inf)
    grids = [np.linspace(c.lo[a], c.hi[a], levels) for a in range(c.d)]
    for size in range(1, min(c.d, max_size) + 1):
        for S in itertools.combinations(range(c.d), size):
            mesh = np.meshgrid(*[grids[a] for a in S], indexing="ij")
            Q = np.tile(c.u, (levels ** size, 1))
            for a, m in zip(S, mesh):
                Q[:, a] = m.ravel()
            y = f(Q)
            cost = (np.abs(Q - c.u) / c.scale).sum(axis=1)
            for i, tau in enumerate(taus):
                ok = y >= tau
                if ok.any():
                    best[i] = min(best[i], float(cost[ok].min()))
    return float(np.where(np.isfinite(best), best, 0.0).sum())


# ---- optimal intervention ------------------------------------------------------------------------------------------
def optimal_evolution(f, c, restarts=3):
    """The best and the worst intervention of the unit, by differential evolution (SciPy) with its vectorized calls."""
    total = 0.0
    bounds = list(zip(c.lo.tolist(), c.hi.tolist()))
    for s in range(restarts):
        for sign in (1.0, -1.0):
            r = differential_evolution(lambda P: sign * f(P.T), bounds, vectorized=True, updating="deferred",
                                       popsize=15, maxiter=100, tol=0.0, polish=False, seed=c.seed + s)
            total += float(r.fun)
    return total


def optimal_candidates(f, c, n=200_000):
    """Scoring a fixed set of candidate interventions, e.g., the doses of a treatment plan."""
    rng = np.random.default_rng(c.seed)
    return float(f(rng.uniform(c.lo, c.hi, size=(n, c.d))).max())


def tipping_points(f, c, budget=6000, steps=30):
    """The smallest change of one variable that moves the outcome by half a standard deviation, by bisection:
    an adaptive search, one query at a time."""
    rng = np.random.default_rng(c.seed)
    y0 = float(f(c.u[None, :])[0])
    spread = float(f(rng.uniform(c.lo, c.hi, size=(200, c.d))).std())
    total, used = 0.0, 0
    while used < budget:
        base = rng.uniform(c.lo, c.hi)
        for a, end in itertools.product(range(c.d), (0, 1)):
            near, far = base[a], (c.lo[a] if end == 0 else c.hi[a])
            for _ in range(steps):
                mid = (near + far) / 2
                t = base.copy()
                t[a] = mid
                if abs(float(f(t[None, :])[0]) - y0) >= spread / 2:
                    far = mid
                else:
                    near = mid
                used += 1
            total += abs(far - base[a])
            if used >= budget:
                break
    return total


# ---- attribution and sensitivity --------------------------------------------------------------------------------
def shapley_attribution(f, c, background=500):
    """Interventional Shapley values of the unit (Janzing et al.): a coalition keeps the unit's values, the others
    take values of a background sample; all coalitions up to d = 10, otherwise 25 permutations as in DoWhy-GCM."""
    bg = c.bg[:background]
    value = {}

    def v(S):
        if S not in value:
            Q = bg.copy()
            if S:
                Q[:, list(S)] = c.u[list(S)]
            value[S] = float(f(Q).mean())
        return value[S]

    phi = np.zeros(c.d)
    if c.d <= 10:
        for S in itertools.chain.from_iterable(itertools.combinations(range(c.d), k) for k in range(c.d + 1)):
            for a in set(range(c.d)) - set(S):
                w = math.factorial(len(S)) * math.factorial(c.d - len(S) - 1) / math.factorial(c.d)
                phi[a] += w * (v(tuple(sorted(S + (a,)))) - v(S))
    else:
        rng = np.random.default_rng(c.seed)
        for _ in range(25):
            S: tuple = ()
            for a in rng.permutation(c.d).tolist():
                new = tuple(sorted(S + (a,)))
                phi[a] += (v(new) - v(S)) / 25
                S = new
    return float(np.abs(phi).sum())


def sobol_sensitivity(f, c, n=1 << 13):
    """Sobol' total-effect indices over the box of the unit (the Saltelli scheme): n (d + 2) queries."""
    rng = np.random.default_rng(c.seed)
    A, B = rng.uniform(c.lo, c.hi, size=(n, c.d)), rng.uniform(c.lo, c.hi, size=(n, c.d))
    ya, yb = f(A), f(B)
    var = float(np.concatenate((ya, yb)).var()) or 1.0
    total = 0.0
    for a in range(c.d):
        M = A.copy()
        M[:, a] = B[:, a]
        total += float(((ya - f(M)) ** 2).mean()) / (2 * var)
    return total


def monte_carlo_risk(f, c, n=200_000):
    """The outcome of the unit under an uncertain intervention: its mean and 5% and 95% quantiles."""
    rng = np.random.default_rng(c.seed)
    Q = np.clip(rng.normal(c.u, (c.hi - c.lo) * 0.1, size=(n, c.d)), c.lo, c.hi)
    y = f(Q)
    return float(y.mean() + np.quantile(y, 0.05) + np.quantile(y, 0.95))


APPS = dict(what_if_surface=what_if_surface, what_if_slider=what_if_slider, dose_response=dose_response,
            recourse_genetic=recourse_genetic, recourse_minimal=recourse_minimal, optimal_evolution=optimal_evolution,
            optimal_candidates=optimal_candidates, tipping_points=tipping_points,
            shapley_attribution=shapley_attribution, sobol_sensitivity=sobol_sensitivity,
            monte_carlo_risk=monte_carlo_risk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--unit", type=int, default=0)
    ap.add_argument("--budgets", default="2e7")
    ap.add_argument("--apps", default=",".join(APPS))
    ap.add_argument("--backends", default="twin,dowhy")
    ap.add_argument("--train-rows", type=int, default=50_000)
    ap.add_argument("--build-threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per application and backend")
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    t0 = time.perf_counter()
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], zoo(1000, 1)[args.family],
                  max_rows=args.train_rows, seed=0, n_jobs=4)
    fit_s = time.perf_counter() - t0
    row = df.iloc[[int(np.random.default_rng(2028).integers(len(df), size=3)[args.unit])]]
    unit = row.iloc[0].to_dict()
    T = choose_treatments(scm, y, args.d)
    if T is None:
        return
    ctx = Ctx(df, T, row, y, 2028 + args.unit)
    tag = f"{args.dataset}_{args.family}_d{args.d}_u{args.unit}"
    backends = {}
    if "twin" in args.backends:
        for b in args.budgets.split(","):
            backends[f"twin_{b}"] = lambda b=b: Twin(scm, unit, T, y, float(b), args.build_threads)
    if "dowhy" in args.backends:
        gcm, cg = dowhy_model(scm, edges)
        backends["dowhy"] = lambda: DoWhy(gcm, cg, row, T, y)
    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    for name, make in backends.items():
        base = dict(dataset=args.dataset, family=args.family, d=args.d, unit=args.unit, backend=name, fit_s=fit_s,
                    mechanisms=len(scm.models), nodes=len(set(sum(map(list, edges), []))))
        try:
            backend = make()
        except (Infeasible, MemoryError) as e:
            rows.append(dict(base, app="-", status=f"infeasible: {str(e)[:60]}"))
            write_csv(f"e2e_{tag}.csv", rows)
            continue
        for app in args.apps.split(","):
            f = Oracle(backend)
            r = dict(base, app=app, build_s=backend.build_s, **backend.info)
            signal.alarm(args.timeout)
            t0 = time.perf_counter()
            try:
                r["output"] = APPS[app](f, ctx)
                r["status"] = "ok"
            except Timeout:
                r["status"] = "timeout"
            finally:
                signal.alarm(0)
            r.update(run_s=time.perf_counter() - t0, queries=f.queries, calls=f.calls)
            if f.single:
                r.update(single_p50_us=float(np.median(f.single)) * 1e6, single_p99_us=float(np.quantile(f.single, 0.99)) * 1e6)
            r["peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
            rows.append(r)
            write_csv(f"e2e_{tag}.csv", rows)
            print(json.dumps({k: r.get(k) for k in ("backend", "app", "status", "build_s", "run_s", "queries", "calls", "output")}), flush=True)


if __name__ == "__main__":
    main()
