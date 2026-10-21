# CounterProvenance

**Exact, compact provenance indexes for repeated counterfactual queries over
tree-ensemble causal models.**

A counterfactual engine such as DoWhy-GCM answers `do(T = t)` for an observed
unit by abduction, action and prediction, and returns a number. Asking again
with a new `t` replays the model even though the fitted model and the unit's
noise have not changed. CounterProvenance compiles the whole intervention
space of one unit once and then answers every `do(T = t)` with

* the **exact counterfactual value** (the model's own leaf selection,
  including points exactly on split thresholds), and
* its **exact provenance**: the multiset of `(tree, leaf)` contributions as an
  element of the free commutative semiring `N[X]`, returned as a handful of
  pre-materialised circuit roots.

It works on fitted **scikit-learn** (GBR, HistGBR, random forests, extra
trees, single trees), **XGBoost** (`exact`, `hist`, ...) and **LightGBM**
regressors, and on any number of intervention variables.

```
pip install -r requirements-models.txt
```

```python
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from counterprovenance import from_model, bind, ProvenanceIndex, autotune

X, y = fetch_california_housing(return_X_y=True)
model = xgb.XGBRegressor(n_estimators=300, max_depth=4).fit(X, y)
unit = X[17]
noise = y[17] - model.predict(unit[None, :])[0]          # abduction (additive noise)

bound = bind(from_model(model), unit, axes=[0, 6, 7], noise=noise)
index = ProvenanceIndex(bound)                            # or: autotune(bound, sample_points).index

answer = index.query([4.5, 34.1, -118.2])                 # do(X0=4.5, X6=34.1, X7=-118.2)
answer.value        # counterfactual Y, equal to model.predict(...) + noise
answer.labels()     # Counter({(tree, leaf): multiplicity, ...}) - one leaf per tree
answer.pieces       # pre-materialised provenance roots (a few to a few dozen)
```

## How it works

```
fitted ensemble ──bind(unit, axes)──► N leaf boxes in R^d
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            │ ProvenanceIndex(tau, group budget)                        │
            │   1. group trees by the intervention axes they split on   │
            │   2. inside a group: overlay small enough (<= tau cells)?  │
            │        yes -> one MCPST leaf                              │
            │        no  -> segment tree on the busiest axis, recurse   │
            │   3. MCPST leaf: exact overlay -> face graph with change  │
            │      labels -> MST construction plan -> persistent factor │
            │      tree (path / node copying) -> one root per face      │
            └───────────────────────────────────────────────────────────┘
query(t): locate t in every visited leaf, return one root per leaf (+ cached values)
```

* **Leaf boxes.** Binding a unit follows every split on a non-intervened
  feature with the unit's own value; each tree becomes a partition of the
  intervention space into axis-aligned boxes, and a query is a
  point-enclosure query over those boxes.
* **Exact overlay.** Box faces label the grid edges they cross with the trees
  whose leaf changes there; faces are the components of the unlabelled-edge
  graph. No cell-by-tree matrix is ever built, and conditional thresholds of
  deep trees do not create spurious cells.
* **MST planning + persistence.** Adjacent faces differ in a few trees. A
  persistent factor tree over the trees stores one directly addressable root
  per face: immutable path copying along the source-augmented MST of the
  exact copy costs, node copying along a covering walk of the face graph, or
  a hybrid that picks the cheaper plan.
* **Groups and the segment tree.** The two parameters move the index between
  one root per query with `Theta(K)` faces (`tau = inf`, one group) and
  `O(log^{d-1} N)` roots per query with `O(N log^{d-1} N)` space independent of
  the overlay size (`tau = 0`). `autotune` picks the point from sample queries.

Exact statements, cost models and proofs: [`docs/DESIGN.md`](docs/DESIGN.md).

## Guarantees (summary)

| | statement | model |
|---|---|---|
| T1 | value and provenance equal the fitted model's own leaf selection, on and off split thresholds | all supported models |
| T2 | 1D: one direct root per region in `O(P + B + K_Delta)` space, `K_Delta = N - M` | explicit change representation |
| T3 | the source-augmented MST under the exact path-copy cost is optimal among adjacency-constrained single-parent immutable plans | fixed factor-tree layout |
| T4 | the MST shortcut walk is a 2-approximation of the cheapest covering walk; node copying realises it with O(1) amortised records per change; in general position every plan needs `K - 1` changes and a Hamiltonian chain attains it | additive per-change cost |
| T5 | `tau = 0`: `O(N log^{d-1} N)` space, `O(log^{d-1} N)` pieces per group, `O(log^d N)` time | pointer machine |

Not claimed: optimal realised allocation of the hybrid (its MST minimises a
surrogate), optimal grouping, bit-level optimality, arbitrary algebraic
factorisation, model families outside the list above.

## Results

Full tables: [`results/RESULTS.md`](results/RESULTS.md) (CSV files alongside). Every
reported index was checked query by query: values against `model.predict`
and DoWhy, provenance against brute force and each library's own leaf ids,
including points exactly on split thresholds. All runs are single-threaded
(`n_jobs=1`, `OMP_NUM_THREADS=1`). Space is counted in structural records;
times are pure Python and are compared only with other Python code.

**Compression against independent per-cell provenance (IIP)** - synthetic
families where every boundary changes one tree:

| workload | IIP records | path copying (rho-MST) | node copying / hybrid | grouped index |
|---|---|---|---|---|
| 1D, 16,384 trees | 268.5 M | 344 K (780x) | 131 K (2,049x) | - |
| 2D, 1,024 stumps, 263 K cells | 270.0 M | 3.43 M (79x) | 1.19 M (227x) | 8,196 records, 2 pieces per query |

**The MST matters on real models** (California housing, d = 2, exact
overlay; `c_e` = trees that change across a face boundary, mean 1.5-3.9,
max 15-25): an MST plan uses 1.2-1.8x fewer persistent records than a random
spanning tree, e.g. LightGBM 91 K vs 133 K (path copying) and 36 K vs 61 K
(hybrid). The exact overlay has 2.0-6.9x fewer faces than the threshold grid.

**Against DoWhy-GCM** (California housing, 8 features, star SCM, autotuned
index, per query):

| intervened features d | DoWhy fresh | index | speed-up vs fresh | vs DoWhy cached noise |
|---|---|---|---|---|
| 2 | 3.1-7.1 ms | 9-10 us | 303-757x | 153-371x |
| 4 | 3.1-7.2 ms | 20-76 us | 44-354x | 22-174x |
| 8 | 3.0-7.2 ms | 193-494 us | 7-37x | 3-18x |

**Hundreds of features** (OpenML topo_2_1 with 261 and yprop_4_1 with 212
features, 300 trees, the d most-used features intervened). DoWhy's cost grows
with the number of nodes; after binding the unit only the trees that split on
an intervened feature remain ("effective trees"), and the rest collapse into
one constant piece:

| d | effective trees | pieces | index | DoWhy fresh | speed-up vs fresh | vs single-row `predict` |
|---|---|---|---|---|---|---|
| 2 | 26-93 / 300 | 3-7 | 7-16 us | 41-57 ms | 2,578-7,814x | 26-307x |
| 8 | 90-177 / 300 | 13-25 | 36-82 us | 38-57 ms | 534-1,460x | 5-53x |
| 16 | 147-238 / 300 | 28-56 | 100-359 us | 41-54 ms | 125-538x | 1.2-21x |

**Limits.** The overlay grows roughly like `N^d`, so a single MCPST is only
feasible for small `d` (here `d <= 2`); beyond that the index trades space for
pieces. With 8 features and `d = 8` it is 7-37x faster than DoWhy but no
faster than a single-row `predict` call; on 200-260 features at `d = 16` it
stays 125-538x faster than DoWhy and 1.2-21x faster than `predict`. Index
build takes 0.05 s (d = 2) to about a minute (d = 16) per unit. Only star
SCMs (the ensemble's inputs are intervened or fixed) are supported; chains of
mechanisms between the intervention and the target are future work.

## Reproducing

All numbers were produced on a Linux server (112 cores, one single-threaded
Python process per experiment, three experiments in parallel) with Python
3.11, numpy 2.4, scikit-learn 1.9, XGBoost 3.2, LightGBM 4.7 and DoWhy 0.14.

```bash
pip install -r requirements-dowhy.txt
python -m unittest discover -s tests          # 27 tests, includes XGBoost/LightGBM if installed
bash experiments/run_all.sh                   # writes results/*.csv and results/RESULTS.md
bash experiments/run_all.sh --quick           # small configuration for a smoke test
```

The California housing data are downloaded by scikit-learn on first use
(`DATA_HOME` selects the cache directory).

## Repository layout

```
counterprovenance/
  semiring.py     hash-consed circuits in N[X] (non-idempotent sum)
  ensemble.py     sklearn / XGBoost / LightGBM compilers, unit binding, slabs
  overlay.py      exact overlay with boundary labels and face adjacency
  persistence.py  persistent factor tree: path copying + node copying
  plan.py         MST, Euler / shortcut / greedy-DFS covering walks
  cellindex.py    MCPST leaf (overlay + plan + materialisation)
  index.py        ProvenanceIndex (groups + adaptive segment tree) and autotune
  synthetic.py    synthetic ensembles for tests and controlled experiments
tests/            exactness and optimality tests (brute force, model leaf APIs)
experiments/      synthetic.py, real_models.py, dowhy_e2e.py, report.py, run_all.sh
results/          CSV files and RESULTS.md
docs/DESIGN.md    design, cost models, theorems and non-claims
docs/archive/     design documents of the v1 prototype
```

The previous prototype (linear SCMs, versioned closures, global-grid MCPST)
is preserved at the git tag `v1-prototype`.

## Data, results, and logs

- `data/` - cached datasets used by the experiments (California housing from scikit-learn; OpenML 416, 422, 46293, 45716), so that runs work without network access.
- `results/` - every CSV of the paper: `qbench_*` (index vs online loop, numba, libraries), `baselines_*` (Treelite, QuickScorer, QuickScorer-B; `experiments/baselines_all.py`), `baselines_dowhy_*` (DoWhy-GCM, 1,000-tree models), and the earlier `e*`/`r*` studies.
- `logs/` - raw stdout of the runs on the experiment server.
- `prototypes/` - the review prototypes and notes behind the design (node copying, K-free index, hybrid plans, unified index).
