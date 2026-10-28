# CounterProvenance

Code of **TWIN: Indexing Tree Ensembles for Fast and Exact Counterfactual Queries**.

A counterfactual query asks, for one observed unit, what the outcome would have been under
`do(T = t)`. In an additive-noise SCM whose mechanisms are tree ensembles (scikit-learn GBR / HGB / RF,
XGBoost, LightGBM), every such query is answered today by abduction, action and prediction: the noise is
abducted, the intervention is applied, and every mechanism between `T` and the outcome is evaluated again,
tree by tree. TWIN compiles the whole intervention space of one unit once, and then answers every
`do(T = t)` with a few table reads, returning

* the **exact counterfactual value** (the libraries' own leaf selection, also on split thresholds), and
* its **provenance**: the leaf of every tree that produced it.

```bash
pip install -r requirements-models.txt          # numpy, scipy, scikit-learn, xgboost, lightgbm
pip install -r requirements-dowhy.txt           # + DoWhy, only for the baselines
```

```python
import pandas as pd
from lightgbm import LGBMRegressor
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import fit_scm, bind_scm

df = pd.read_parquet("data.parquet")                         # columns = nodes of the causal DAG
edges = list(pd.read_csv("graph.csv").itertuples(index=False))   # (source, target) pairs
T, Y = ["A", "B"], "Y"                                       # intervened variables: ancestors of Y
scm = fit_scm(df, edges, outcome=Y, treatments=T,
              factory=lambda: LGBMRegressor(n_estimators=1000, verbose=-1), max_rows=50_000)

unit = df.iloc[17].to_dict()                                 # the observed unit
bound = bind_scm(scm, unit, treatments=T, outcome=Y)         # leaf boxes over t, through the mediators
index = ProvenanceIndex(bound, grouping="auto")              # built once per unit

ans = index.query([40.0, 35.0])          # do(A = 40, B = 35)
ans.value                                # exact counterfactual outcome
ans.labels()                             # provenance: Counter({(tree, leaf): 1, ...})
index.choice                             # "one", "tables" or "masks" (see below)
```

## How it works

```
fitted SCM + unit ──compile──► leaf boxes of the outcome's trees over t ∈ R^d
                                   │
          slab grid of all trees ≤ 2M cells? ──yes──► one cell graph + MST          ("one")
                                   │ no
                     greedy grouping into cell graphs (budget β)
                                   │
            cost model: α·#graphs (+ masks of deep trees)  vs  F + γ·#mask words
                       │                                        │
            several cell graphs, each stored along its MST   leaf bitmasks for every tree
                       ("tables")                                 ("masks")
```

**1. Leaf boxes through mediators.** For a fixed unit, every variable that is not intervened and not a
descendant of `T` keeps its observed value, so every tree of the outcome becomes a partition of the
intervention space into axis-aligned *leaf boxes*. A mediator `Z` between `T` and the outcome is replayed
by its own trees: `Z(t) = u_Z + (f_Z(pa(t)) − f_Z(pa(u)))`, which equals the observed value on the factual
cell. A box of the outcome's tree that tests `Z ≤ θ` is kept whole when `Z` stays on one side of `θ` over
the box. This is checked by a dominance test that bounds `Z` by `u_Z` plus the per-tree minimum and maximum
differences. Only the boxes that straddle `θ` are split at the mediator's internal boundaries
(`scm.bind_scm`).

**2. One cell graph and its MST.** The walls of all leaf boxes cut the space into *cells*; two adjacent
cells differ only in the trees whose leaf changes across their common wall.
* The **cell graph** has one node per cell and one edge per wall, weighted by the number of trees that
  change.
* The index stores the cells along a **minimum spanning tree** of this graph in a persistent factor tree
  (path or node copying). Every cell is then one directly addressable root, and a cell costs only the trees
  that change on its MST edge.
* A query is `d` binary searches plus one table read.

This is used whenever the slab grid of all trees has at most `one_cells = 2·10^6` cells, which in
practice means `d ≤ 2` (`cellindex.py`, `persistence.py`, `plan.py`).

**3. Several cell graphs: greedy grouping.** For larger `d`, the product grid of all trees is too large,
but most trees split on few of the intervened axes.
* *Group sizes.* A group of trees gets its own cell graph over the axes its trees split on. Its size is
  the product, over those axes, of (number of cuts + 1).
* *Problem.* Choose groups that minimise the number of cell graphs (= table reads per query) subject to
  a total size `Σ cells ≤ β`.
* *Hardness.* The problem is NP-hard by a reduction from Product Partition. Hence no polynomial algorithm
  approximates the number of groups within a factor below 3/2 unless P = NP.
* *Greedy* (`grouping.greedy_groups`, Kruskal-like):
  - start from groups of nested axis supports;
  - repeatedly merge the pair whose union adds the fewest cells, while the total stays within `β`
    (default `2·10^6`);
  - trees whose own grid exceeds `grid_cap`, or 64× their number of leaves, are *deep*: a table over
    their grid would be mostly redundant.

  The greedy has no approximation guarantee; it is fast and works well in practice.
* *Storage.* Each group is a cell graph stored along its MST, as in step 2. A single-tree group is
  trivial: one entry per leaf.

**4. Tables or masks: a cost model.** Before building anything, `grouping="auto"` compares two estimated
query costs (in µs):

| choice | per-query cost | answered by |
|---|---|---|
| `tables` | `α · #cell graphs` (+ `F + γ · words` for the deep trees) | one packed read per cell graph |
| `masks`  | `F + γ · Σ_m ⌈leaves_m / 64⌉` | leaf bitmasks of every tree |

* Defaults: `α = 0.15`, `F = 8`, `γ = 0.02`.
* *Leaf bitmasks.* For every axis and slab, one 64-bit field per tree holds the leaves whose box
  contains the slab. The AND over the `d` axes leaves exactly one bit per tree, so the storage is a sum
  over axes rather than a product.
* *Why a pure choice.* Mixing small cell graphs with masks was slower than either pure choice in our
  measurements, so the cost model picks one of the two.

**Exactness.**
* Abduction sums the leaf values exactly (`fsum`). XGBoost's own `predict` accumulates in single
  precision; its deviation is recorded separately as `err_lib`.
* Every experiment checks every query of TWIN against the three-step evaluation, ChiRho and DoWhy-GCM.

## Experiments

**Baselines.** All are pure Python and run single-threaded with the same fitted models:
* **Three-step** (`scm_baselines.ScmThreeStepPy`): abduction, action and prediction on every query,
  without caching the noise;
* **ChiRho** (`experiments/chirho_baseline.py`): a Pyro program under `TwinWorldCounterfactual` and `do`;
* **DoWhy-GCM** (PyWhy): `gcm.counterfactual_samples` over the causal DAG.

**Datasets.** There are 15 real tables, from 1,030 to 101 million rows. Every outcome has ≥ 8 ancestors,
so all `d ∈ {1, 2, 4, 8}` can be evaluated. `experiments/causal_datasets.py` writes `data.parquet`,
`graph.csv` and `meta.json` for each.

| group | datasets |
|---|---|
| published graph | gov_census (fairadapt), light_tunnel (Causal Chambers) |
| partial graph (cause → effect) | postgres, jdk (Gentzel et al.), concrete (Tübingen pairs) |
| graph learned by PC | nyc_taxi (101M), airline (32.6M), higgs (11M), methane (9.2M), susy (5M), census1990 (2.5M), acs_income (1.7M), mslr (1.2M), baf (1M), superconduct (21K) |

For the PC-learned graphs:
* PC (causal-learn) uses Fisher-z tests at level 0.01, conditioning sets of at most 3 variables, and
  10,000 sampled rows; the outcome is forbidden to cause any other variable.
* Conflicting orientations are demoted by the Eades–Lin–Smyth ordering.
* The result is extended to a DAG by Dor–Tarsi, or, when no consistent extension exists, along a
  topological order.

**Setup.**
* *Models.* 1,000 trees per mechanism for all five families (GBR depth 3, HGB and LGBM 31 leaves, XGB
  depth 6, RF depth 10), trained on up to 50,000 sampled rows.
* *Queries.* For each dataset, model and `d` there are 3 random units and 10,000 queries per unit:
  - 7,500 uniform between the 1st and 99th percentiles of the intervened variables;
  - 2,500 with one coordinate moved onto a leaf-box boundary.
* *Intervened variables.* Those the outcome's mechanism splits on most often.

**Results so far** (10 of the 15 datasets, 173 settings; geometric-mean speedup of TWIN; all methods
return the same values):

| d | TWIN, mean per query | vs three-step | vs ChiRho | vs DoWhy-GCM |
|---|---|---|---|---|
| 1 | 5.5 µs | 677× | 3,023× | 9,323× |
| 2 | 6.1 µs | 913× | 3,596× | 8,091× |
| 4 | 42 µs | 233× | 989× | 1,832× |
| 8 | 66 µs | 101× | 601× | 1,060× |

**Reproducing.** Experiments run on a Linux server. The paths below assume the layout
`<W>/{env,data,logs,repo}`.

```bash
python experiments/causal_datasets.py --out ../data/causal              # build the 15 datasets (downloads)
DIMS="1 2 4 8" experiments/run_scm_split.sh ../data/causal 80 <datasets> # baselines (+ TWIN), one job per (dataset, model, d)
EXTRA="--ours-only" LOGDIR=../logs/scm_ours DIMS="1 2 4 8" \
  experiments/run_scm_split.sh ../data/causal 24 <datasets>               # TWIN alone
python experiments/main_report.py --base <baseline repo> --ours <ours repo> \
  --data ../data/causal --out results/main                                # tables and figures
python -m unittest discover -s tests                                      # exactness tests
```

`run_scm_split.sh` has the following properties:
* it skips finished and running jobs, so it can be restarted at any time;
* it hides the GPUs (`CUDA_VISIBLE_DEVICES=`), because XGBoost would otherwise open them and can block on
  the driver of a shared machine;
* it runs every job single-threaded.

## Repository layout

```
counterprovenance/
  ensemble.py       tree compilers (sklearn, XGBoost, LightGBM), unit binding, slabs
  scm.py            additive-noise SCMs on a DAG: fitting, abduction, leaf boxes through mediators
  grouping.py       tree profiles and the greedy grouping under the budget β
  index.py          ProvenanceIndex: one cell graph / greedy cell graphs / leaf masks, cost model, query
  cellindex.py      a cell graph: exact overlay, MST plan, persistent factor tree
  overlay.py        exact overlay of leaf boxes with wall labels and cell adjacency
  plan.py           MST and covering walks of the cell graph
  persistence.py    persistent factor tree (path and node copying)
  accel.py          numba kernels (bitmask AND, packed lookups)
  scm_baselines.py  three-step and online baselines
  semiring.py       provenance circuits in N[X]
experiments/
  causal_datasets.py  the 15 datasets and their DAGs
  scm_full.py         one (dataset, model, d) job: TWIN, three-step, ChiRho, DoWhy-GCM
  chirho_baseline.py  the ChiRho baseline
  run_scm_split.sh    job runner (resumable)
  main_report.py      tables and figures of the main experiment
  *_proto.py, ...     design studies behind the algorithm (grouping, masks, dominance, skyline)
tests/              exactness tests against brute force and the libraries' leaf APIs
docs/DESIGN.md      design notes; docs/archive/ the v1 prototype
```

The previous prototype (linear SCMs, versioned closures, global-grid MCPST) is kept at the git tag
`v1-prototype`.
