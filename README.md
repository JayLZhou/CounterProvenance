# TESSERA Counterfactual Provenance Index

Repository: <https://github.com/JayLZhou/CounterProvenance>

TESSERA is a research prototype for **exact indexed counterfactual queries over
learned structural causal models**. The current code implements the linear
foundation on arbitrary causal DAGs; the paper design extends the same
provenance semantics and physical index to trees, tree ensembles, and
piecewise-affine neural mechanisms.

The core idea is:

> Do not index model regions or cache individual answers. Index the complete
> provenance-valued response function `Pi[u,Y](t) = Y_{u;do(T=t)}`. Store each
> unit's maximal intervention intervals as pointers to a shared provenance DAG.

The full mechanism class, index design, baselines, and theorem targets are in
[`POSITIONING.md`](POSITIONING.md). The current linear trie is the one-region
special case of that design.

## Why This Exists

Libraries such as DoWhy-GCM can compute counterfactuals by running Pearl's
abduction-action-prediction procedure:

1. recover each unit's residual/noise from observed data;
2. apply an intervention such as `do(T=t')`;
3. replay the SCM forward with the same unit-level residuals.

That is correct, but repeated counterfactual workloads do the same structural
work again and again. TESSERA asks:

> Can we materialize the reusable counterfactual derivation once, then answer
> repeated queries by evaluating stored provenance?

This demo answers that question for linear SCMs over arbitrary cDAGs. The
broader paper asks how to answer `POINT`, `FLIP`, `AGG`, and `EXPLAIN` queries
without fresh SCM replay.

## Three Exact f(.) Classes

TESSERA claims support for exactly three mechanism classes. They share the
internal property **finite guarded-affine and exactly line-intersectable**:

```text
f(x) = a[r]^T x + b[r]  when exact guard g[r](x) holds

optional factorization:
f(x) = sum_m lambda_m * f_m(x)
```

When the counterfactual parents are affine on an intervention interval,
`x(t) = A*t + c`, the local compiler must be able to enumerate every exact
value of `t` at which the active guard changes.

| Class | Representative fitted `f(.)` | Exact local operation | Status |
|---|---|---|---|
| L: Linear | OLS, LinearRegression, Ridge, Lasso, ElasticNet, linear SVR | one affine segment | implemented for generic linear cDAGs |
| T: Tree Ensemble | one regression tree, Random Forest, Extra Trees, GBDT, HistGBDT, numerical XGBoost/LightGBM/CatBoost with raw or identity-link regression output | threshold tracing per tree plus breakpoint multiway merge | first nonlinear target |
| N: ExactLine/PWL Neural | ReLU MLPs and ExactLine-supported piecewise-linear inference graphs | ExactLine-style line partitioning plus provenance lifting | second nonlinear target |

Single trees are the `M=1` case of Class T. Mixed SCMs may use any of L/T/N at
different causal nodes. Class N builds on
[ExactLine](https://papers.nips.cc/paper_files/paper/2019/hash/908075ea2c025c335f4865f7db427062-Abstract.html)
for numeric line partitioning and adds counterfactual provenance and indexing.

KNN, rule lists, splines/GAMs, AdaBoost, sigmoid/tanh/GELU networks, RBF
models, Gaussian processes, polynomial regression, arbitrary black boxes, and
categorical classifiers are outside the submitted three-class claim.

## Unified Index: GPSI

The Guarded Provenance Segment Index has one logical and physical design for
every supported mechanism:

```text
GPSI
|-- local mechanism DAG
|-- shared prefix-factorized provenance DAG
|      reverse trie + hash-consed semiring subcircuits
|-- optional guard-certificate DAG for raw path witnesses
|-- segment directory: (version, unit, treatment, target)
|      -> sorted [(left, right, alpha, beta, template_id, certificate_id)]
|-- unit store: factual and versioned abducted-noise pointers
`-- event indexes for FLIP and optional SUM/AVG queries
```

L produces one segment, T performs threshold tracing and breakpoint merging,
and N performs ExactLine-style partitioning. These are three implementations
of one `COMPILE_LINE` contract; causal composition, semiring provenance,
template sharing, segment coalescing, and query execution remain unchanged.
Prefix factorization is mandatory: for a fixed target, causal paths are reversed
so shared source-to-target suffixes become shared trie prefixes. The trie
compresses derivations; the segment directory indexes their validity.

## Baselines and Theory Targets

The evaluation compares GPSI against:

1. DoWhy-style replay from observed data;
2. cached-noise replay, which removes abduction but still traverses the SCM;
3. global-region materialization;
4. exact numeric per-unit polylines, which match point-lookup complexity but
   omit provenance;
5. exact provenance polylines that duplicate a full expression per segment;
6. tree-ensemble ADD/decision-diagram compilation;
7. unmodified numeric ExactLine for Class N.

The main theoretical advantages are:

- exact closure of L/T/N under causal composition for scalar interventions;
- a Class-T non-Cartesian bound: tree partitions of sizes `b_m` require
  at most `1 + sum_m(b_m - 1)` line intervals, rather than their product;
- output-sensitive construction in realized guard crossings and retained
  segments, with no potential-global-region term;
- exact prefix storage with one node per distinct canonical reversed path
  prefix, instead of one copy per expanded path occurrence;
- a canonical minimum segment quotient after coalescing adjacent intervals
  with identical normalized provenance;
- storage `O(S + D + n*|V|)` instead of `Theta(S*L)` for per-unit full
  expressions, where `S` is the number of realized segments and `D` is the
  shared provenance-DAG size;
- `POINT` in `O(log B_u)` and indexed `FLIP` in `O(log E + k)`.

These are scoped claims. TESSERA does not promise universal compression or
avoid the worst-case exponential number of realized piecewise-affine segments.
It matches, rather than asymptotically beats, numeric polyline point lookup;
the advantage is exact shared provenance plus indexed derived queries.

## Generic Linear cDAG Support

The reusable engine in `counterprovenance/linear.py` supports:

- arbitrary directed acyclic causal graphs;
- additive linear mechanisms;
- non-intervened root variables preserved from the observed unit;
- one or more structural interventions;
- any single target node per compiled query;
- expanded semiring monomials as logical provenance;
- reverse trie storage as physical provenance compression;
- compiled closed-form evaluation for fast repeated queries.

The toy SCM below is just the smallest readable example.

## Toy SCM

The demo uses the following additive linear SCM:

```text
T -> A -> B -> Y
T -> A -> C -> Y
T -> D -> Y
T -> Y
```

with mechanisms:

```text
A = w_TA T + b_A + N_A
B = w_AB A + b_B + N_B
C = w_AC A + b_C + N_C
D = w_TD T + b_D + N_D
Y = w_BY B + w_CY C + w_DY D + w_TY T + b_Y + N_Y
```

For `do(T=t')`, standard counterfactual replay computes:

```text
T_cf = t'
A_cf = w_TA T_cf + b_A + N_A(u)
B_cf = w_AB A_cf + b_B + N_B(u)
C_cf = w_AC A_cf + b_C + N_C(u)
D_cf = w_TD T_cf + b_D + N_D(u)
Y_cf = w_BY B_cf + w_CY C_cf + w_DY D_cf + w_TY T_cf + b_Y + N_Y(u)
```

## Semiring Provenance

TESSERA expands `Y_cf` into semiring monomials. Example terms:

```text
do:T    * theta:T->A * theta:A->B * theta:B->Y
noise:A * theta:A->B * theta:B->Y
bias:B  * theta:B->Y
noise:B * theta:B->Y
```

The valuation is:

```text
do:T        -> intervention value t'
noise:X     -> abducted residual N_X(u)
bias:X      -> learned intercept b_X
theta:X->Z  -> learned coefficient w_XZ
oplus       -> numeric addition
otimes      -> numeric multiplication
```

Under this valuation, the semiring provenance evaluates to the same value as
additive-noise counterfactual replay.

## Trie Compression

The expanded monomials can share propagation fragments. For a fixed target
`Y`, this demo stores them in a **reverse trie** from target back to source.

For example:

```text
theta:B->Y, theta:A->B, theta:T->A, do:T
theta:B->Y, theta:A->B, noise:A
theta:B->Y, noise:B
```

The shared suffix-to-target fragment `A -> B -> Y` is stored once.

Important distinction:

- the raw trie is the compressed provenance store;
- the fast runtime path is a compiled closed-form evaluator derived from that
  provenance store.

For this linear case, the compiled evaluator has the form:

```text
Y_cf(u, t') = constant + alpha * t' + sum_i beta_i * N_i(u)
```

## Current Results

On the toy SCM:

```text
expanded monomials      = 16
expanded token slots    = 39
reverse trie tokens     = 25
storage compression     = 1.56x
```

Correctness:

```text
max |trie(Y_cf) - replay(Y_cf)|     = 1.776e-15
max |compiled(Y_cf) - replay(Y_cf)| = 3.553e-15
max |trie(Y_cf) - dowhy.gcm(Y_cf)|  = 2.665e-15
```

Speed:

```text
n = 10,000, repeats = 5
DoWhy-GCM counterfactual_samples: 0.0160s
compiled TESSERA evaluation:      0.0001s
speedup:                          117.82x
```

Against a hand-written NumPy replay baseline:

```text
n = 1,000,000, repeats = 20
NumPy replay:                 0.3384s
compiled TESSERA evaluation:  0.1062s
speedup:                      3.19x
```

On a random layered cDAG:

```text
layers = (3, 8, 8, 8, 1), nodes = 28, mechanisms = 25
expanded monomials      = 164
expanded token slots    = 572
reverse trie tokens     = 279
storage compression     = 2.05x
compiled speedup        = 7.96x vs NumPy replay
```

These are early sanity numbers, not final system benchmarks. The main point is
that compiled provenance agrees with DoWhy-style counterfactual semantics and
already avoids repeated mechanism replay in the linear case.

## Quick Start

Install the minimal dependency:

```bash
pip install -r requirements.txt
```

Run with only NumPy installed:

```bash
python experiments/tessera_linear_trie/linear_trie_demo.py
```

Run a larger speed check against the lightweight NumPy replay baseline:

```bash
python experiments/tessera_linear_trie/linear_trie_demo.py \
  --n 1000000 \
  --repeats 20 \
  --skip-dowhy \
  --quiet \
  --skip-raw-trie-timing
```

Run a random layered cDAG stress check:

```bash
python experiments/tessera_linear_trie/linear_trie_demo.py \
  --mode layered \
  --layer-sizes 3,8,8,8,1 \
  --edge-prob 0.45 \
  --n 10000 \
  --repeats 10 \
  --skip-dowhy \
  --quiet \
  --skip-raw-trie-timing
```

If DoWhy is installed, compare against real `dowhy.gcm.counterfactual_samples`:

```bash
pip install -r requirements-dowhy.txt

python experiments/tessera_linear_trie/linear_trie_demo.py \
  --n 10000 \
  --repeats 5 \
  --quiet \
  --skip-raw-trie-timing \
  --bench-dowhy
```

Useful flags:

```text
--n                    number of observed units
--mode                 toy or layered
--layer-sizes          comma-separated layer sizes for random layered cDAGs
--edge-prob            edge probability for random layered cDAGs
--t-prime              intervention value for do(T=t')
--repeats              repeated query count for timing
--skip-dowhy           skip optional DoWhy comparison
--bench-dowhy          time real dowhy.gcm.counterfactual_samples
--skip-raw-trie-timing skip slow per-unit Python trie traversal
--quiet                do not print every expanded monomial
```

## Files

```text
README.md                                           this document
POSITIONING.md                                      paper scope, unified index, baselines, and theorem targets
REPO_INTRO.md                                       copy-ready repository introduction
TODO.md                                             next steps
requirements.txt                                    minimal Python dependencies
requirements-dowhy.txt                              optional DoWhy comparison dependencies
counterprovenance/linear.py                          generic linear cDAG provenance engine
experiments/tessera_linear_trie/linear_trie_demo.py main prototype and benchmark
tests/test_linear_general.py                         correctness tests for arbitrary linear cDAGs
```

## What This Prototype Does Not Claim

The checked-in implementation is intentionally narrower than the paper design.

It does not yet handle:

- the guarded segment directory;
- tree, ensemble, or ReLU local compilers;
- FLIP or aggregate event indexes;
- optimized trie execution;
- batched multi-target workload optimization.

Those are the next steps in the broader TESSERA agenda. The prototype only
establishes the one-region affine foundation:

> linear counterfactual replay can be represented as semiring provenance,
> compressed as a propagation trie, and compiled into a fast query plan.

## Research Direction

The repositioned paper is:

**TESSERA: An Exact Provenance Index for Parametric Counterfactual Queries over
Piecewise-Affine Structural Causal Models**

The implementation path follows the unified index rather than creating a
separate data structure for every model family:

1. implement the generic piecewise-affine segment algebra;
2. compile regression trees and additive tree ensembles into that algebra;
3. add the shared reverse-prefix provenance trie/DAG and canonical segment quotient;
4. add FLIP and aggregate event indexes;
5. lift ExactLine pieces to Class-N counterfactual provenance;
6. evaluate mixed SCMs and prove the bounds stated in `POSITIONING.md`.
