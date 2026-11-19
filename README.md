<div align="center">

# 🌲 CellForest

### Indexing Tree Ensembles for Fast and Exact Counterfactual Queries

![Python](https://img.shields.io/badge/python-3.11-blue)
![C++](https://img.shields.io/badge/build-C%2B%2B17-orange)
![Models](https://img.shields.io/badge/models-scikit--learn%20%7C%20XGBoost%20%7C%20LightGBM-green)
![Exact](https://img.shields.io/badge/answers-exact-brightgreen)

**⚡ Microseconds per query · 🎯 exact answers · 🧩 one index for all units and all interventions**

</div>

---

## 🤔 What is this?

A **counterfactual query** asks, for one observed unit (a customer, a patient, a loan applicant):
*what would the outcome have been, had some variables been set to other values?*

On a structural causal model (SCM) whose functions are **tree ensembles** (gradient-boosted trees,
random forests), existing libraries answer every query by *abduction → action → prediction*, and
traverse **every tree again for every query**.

**CellForest** indexes the tree ensembles once, offline, and then answers each query with a few table
reads:

- 🎯 **Exact** — the same selected leaves and the same value as the three steps.
- ⚡ **Fast** — microseconds per query, two to three orders of magnitude faster than existing solutions.
- 🧩 **One index per tree ensemble** — it serves every unit and every intervention; a new unit needs no new index.
- 🍃 **Keeps the leaves** — a query can also return the selected leaf of every tree, e.g., to reweight the trees.

## 🧠 How it works

```
 tree ensemble f_V                                             query do(X = x) on unit u
        │                                                                 │
        ▼                                                                 ▼
 🟦 cells: every tree selects the same leaf                  binary search per parent
        │                                                                 │
        ▼                                                                 ▼
 🕸️ cell graph: neighbors, weighted by changed trees            one read per group
        │                                                                 │
        ▼                                                                 ▼
 🌿 CellTree: changes stored along the MST                        value (+ leaves)
        │
        ▼
 🌲 CellForest: groups of trees, one CellTree each
```

| Step | Idea |
|---|---|
| 🟦 **Cells** | The output of a tree ensemble is constant wherever every tree selects the same leaf. |
| 🕸️ **Cell graph** | Neighboring cells differ in very few trees; an edge is weighted by the trees that change. |
| 🌿 **CellTree** | Store one cell in full and every other cell as its change from a neighbor, along the minimum spanning tree. |
| 🌲 **CellForest** | Split the trees into groups so that the blocks multiply only within a group. Finding the fewest groups is NP-hard, so we merge greedily under a budget β. |

## 🚀 Quick start

### 📥 Install

```bash
pip install -r requirements-models.txt    # numpy, scipy, scikit-learn, xgboost, lightgbm
pip install -r requirements-dowhy.txt     # + DoWhy, networkx, pandas (SCMs and the competitors)
```

The two C++ kernels in `counterprovenance/native/` are compiled with `g++` (C++17) on first use.

### 🌳 One model

```python
from counterprovenance import bind, from_model, ProvenanceIndex

ensemble = from_model(model)                 # a fitted scikit-learn, XGBoost, or LightGBM regressor
bound = bind(ensemble, unit_row, axes=[0, 3], noise=noise)
index = ProvenanceIndex(bound)               # built once

answer = index.query([t0, t3])               # do(X0 = t0, X3 = t3)
answer.value                                 # exact counterfactual value
answer.labels()                              # the selected leaves: (tree, leaf) multiset
```

### 🕸️ A whole SCM

```python
import sys; sys.path.insert(0, "experiments")
import pandas as pd
from lightgbm import LGBMRegressor
from counterprovenance.scm import fit_scm, relevant_nodes
from mech_proto import MechanismIndex

df = pd.read_parquet("data.parquet")                             # columns = variables of the causal DAG
edges = list(pd.read_csv("graph.csv").itertuples(index=False))   # (source, target) pairs
T, Y = ["A", "B"], "Y"                                           # intervened variables and the outcome
scm = fit_scm(df, edges, outcome=Y, treatments=T,
              factory=lambda: LGBMRegressor(n_estimators=1000, verbose=-1), max_rows=50_000)

order = relevant_nodes(scm.parents, T, Y)                        # the mediators and Y, in topological order
forest = MechanismIndex(scm, df.iloc[0].to_dict(), order,        # 🌲 one index per tree ensemble,
                        budget=2e7, threads=1)                   #    built once for all units

unit = df.iloc[17].to_dict()                                     # any observed unit
observed = forest.observed(unit, order)                          # abduction: once per unit
forest.query(unit, T, [40.0, 35.0], order, observed)             # do(A = 40, B = 35)
```

## 📊 Results at a glance

Measured on 15 real tables (1,030 to 101 million rows) with five model families of 1,000 trees
(GBR, HGB, XGBoost, LightGBM, and LightGBM's random forest), in Python, in one thread.

| | CellForest | ThreeStep | CausalAI | ChiRho | DoWhy |
|---|---|---|---|---|---|
| ⏱️ Mean time per query | **25 µs** | 6.9 ms | 10.7 ms | 21 ms | 68 ms |
| 🚀 Speedup of CellForest | — | 155–274× | 390–427× | 450–844× | 1,460–2,707× |

- 💾 **Index**: 7 MB to 5.7 GB per table on average, built in seconds to minutes with one thread.
- 🧮 **Exact**: the same answers as the three steps on all 3.2 million queries.
- 🧰 **Applications** (counterfactual explanations, feature relevance, what-if analysis, counterfactual
  fairness): the queries become two to three orders of magnitude faster, with the application code unchanged.

## 🔬 Reproducing the experiments

```bash
python experiments/causal_datasets.py --out ../data/causal      # 📚 build the 15 tables and their DAGs
python experiments/mech_main.py --data ../data/causal \
       --dataset concrete --family GBR --backend cellforest     # ⚡ CellForest on one table and model
python experiments/mech_main.py --data ../data/causal \
       --dataset concrete --family GBR --backend three_step     # 🐢 a competitor on the same queries
python -m pytest tests                                          # ✅ exactness tests
```

`--backend` is one of `cellforest`, `three_step`, `chirho`, `dowhy`, `causalai`;
`--family` is one of `GBR`, `HGB`, `XGB`, `LGBM`, `RF`.

| Experiment | Scripts |
|---|---|
| 🏁 Query time on all tables (Exp.1–2) | `mech_main.py`, `run_main.py`, `run_latency.py`, `run_causalai.py` |
| 💾 Index size and build time (Exp.3) | `mech_rss.py`, `run_rss.py` |
| 🌳 Number and depth of the trees (Exp.4) | `run_trees.py`, `depth_exp3.py`, `depth_driver3.py` |
| 🎚️ Budget β (Exp.5) | `mech_beta.py`, `beta_probe.py`, `run_beta.py` |
| 🧪 Ablations: CellTable, no grouping (Exp.6) | `mech_structure.py`, `mech_table.py`, `mech_size.py`, `run_none.py` |
| 🍃 Other combinations of the trees (Exp.7) | `combine_bench.py` |
| 🧰 Applications (Exp.8–11) | `e2e_apps.py`, `e2e_table.py`, `e2e_shard.py`, `run_shards.py`, `mint_paths.py` |
| 📈 Figures of the paper | `paperfigs/gen_all.py` |

The `run_*.py` scripts are schedulers for a large shared server: they start the jobs of an experiment
one by one as the memory allows, and can be restarted at any time.

## 🗂️ Repository layout

```
counterprovenance/        📦 the library
  ensemble.py             tree compilers (scikit-learn, XGBoost, LightGBM), leaf boxes
  scm.py                  additive-noise SCMs on a DAG: fitting, abduction, mediators
  grouping.py             the greedy grouping under the budget β
  index.py                ProvenanceIndex: the groups, their cell graphs, and the query
  cellindex.py            a cell graph: cells, MST, persistent tree of the changes
  native/                 C++ kernels (leaf boxes, cell graph and MST)
  scm_baselines.py        ThreeStep and online evaluation in pure Python
experiments/              🔬 one script per experiment (see the table above)
  causal_datasets.py      the 15 tables and their causal DAGs
  workload_datasets.py    the tables of the applications
  chirho_baseline.py      the ChiRho competitor
  causalai_baseline.py    the CausalAI competitor, made exact
  paperfigs/              the figures of the paper
tests/                    ✅ exactness tests against brute force and the libraries' leaf APIs
```

## 📚 Datasets

| Causal graph | Tables |
|---|---|
| ✅ Published | Light Tunnel (Causal Chambers) |
| 🧩 Partial (cause → effect) | PostgreSQL, JDK, Concrete |
| 🤖 Learned by the PC algorithm | NYC Taxi, Airline, Higgs, Methane, SUSY, Census 1990, MSLR, BAF, Elevators, Pol, Ailerons |

## 📝 Citation

```bibtex
@article{cellforest,
  title  = {CellForest: Indexing Tree Ensembles for Fast and Exact Counterfactual Queries},
  author = {Zhou, Yingli and Bonifati, Angela},
  year   = {2026}
}
```

## 💬 Contact

Questions and issues are welcome: please open a GitHub issue. 🙌
