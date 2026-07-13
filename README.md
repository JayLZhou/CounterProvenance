# TESSERA Counterfactual Provenance Index

Repository: <https://github.com/JayLZhou/CounterProvenance>

TESSERA is a research prototype for **exact indexed counterfactual queries over
learned structural causal models**. The current code implements the linear
foundation on arbitrary causal DAGs and a model-independent **Versioned
Provenance Closure** core. Tree and ReLU compilers remain future work, but
their arbitrary region-to-region `ON/OFF/ON` patterns no longer require a
staircase assumption in the storage layer.

The core idea is:

> Compile the exact intervention line into a shared semiring template plus an
> ordered sequence of slot environments. Store this one logical closure in
> either a storage-optimal compact layout or a latency-oriented materialized
> circuit layout, while keeping per-unit numeric bindings separate.

The full mechanism class, index design, baselines, and theorem targets are in
[`POSITIONING.md`](POSITIONING.md). The current linear trie is the one-region
special case; `counterprovenance/versioned.py` implements both exact physical
layouts for arbitrary region versions.

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

## Unified Index: Closure-Backed GPSI

For unit `u`, region `j`, let `P[u,j]` be its exact parameterized provenance and
let `h_u` bind arbitrary unit-specific parameters, roots, and abducted noises.
The unified logical object is

```text
C[u,j] = (shared template T[u], slot environment sigma[u,j])
P[u,j] = Eval(T[u], sigma[u,j])
```

It has two exact physical layouts:

- **Closure-Compact** stores each slot's initial root and only its effective
  changes. Its storage is independent of downstream circuit fan-out.
- **VSC-Materialized** resolves every region to a canonical immutable
  `ADD`/`MUL` root. It path-copies affected ancestors and globally hash-conses
  recurring subcircuits across regions and units.

A fragment may follow `ON -> OFF -> ON` repeatedly in either layout. Numeric
values remain outside shared topology in each unit's binding `h_u`.

The Guarded Provenance Segment Index has one logical and physical design for
every supported mechanism:

```text
GPSI
|-- local mechanism DAG
|-- response segment directory
|      sorted boundaries -> environment versions or materialized root IDs
|-- Versioned Provenance Closure
|      shared ZERO/ONE/ATOM/ADD/MUL template
|      Closure-Compact: initial slots + effective change histories
|      VSC-Materialized: affected-ancestor path copying + hash-consing
|-- optional additive frontier optimizer
|      maximal runs -> LCF when supports are structured
|      linear-space stabbing-semigroup directory in compact mode
|-- optional, finer certificate DAG and interval directory
|-- unit store: parameter, factual-root, abducted-noise, and anchor bindings h_u
|-- registered-threshold FLIP event indexes
`-- declared-group SUM/AVG response maps
```

L adds no boundary, T emits leaf changes, and N emits activation changes after
ExactLine partitioning. All become updates to one circuit-template interface.
The support-matrix/LCF representation remains useful at an additive circuit
frontier, but it is no longer the universal logical model: the closure
preserves products directly and therefore does not require expansion into
neural paths.

## Baselines and Theory Targets

The evaluation compares GPSI against:

1. DoWhy-style replay from observed data;
2. cached-noise replay, which removes abduction but still traverses the SCM;
3. exact numeric per-unit segments, which match point-lookup complexity but
   omit provenance;
4. exact segments that duplicate fully expanded provenance paths;
5. global-region materialization on tractable cases;
6. tree-ensemble ADD/decision-diagram compilation;
7. unmodified numeric ExactLine for Class N.

The revised theory spine is:

- exact closure of L/T/N under causal composition for scalar interventions;
- Compact Closure matching build/storage
  `Theta(D_T + C_tot + B_tot + K_delta + H)`, where `K_delta` is the number
  of effective slot changes; the matching word-RAM lower bound allows each
  changed slot root to carry an independent word;
- Materialized VSC construction
  `O(D_T + C_tot + B_tot + A + H)` and storage
  `O(D_T + C_tot + B_tot + G + H)`, where `A` counts affected template nodes
  and `G` counts newly interned canonical nodes;
- canonical VSC is node-instance-optimal within the explicitly defined
  immutable resolved-circuit model, not among all algebraically equivalent
  circuits;
- for a balanced associative block with `C` mutable slots and `k_j` changed
  slots at one boundary, batch path copying has tight worst-case cost
  `Theta(k_j log(1 + C/k_j))`;
- for `B=C=n` and `k_j=O(1)` arbitrary toggles, Independent Interval
  Provenance, Materialized VSC, and Compact Closure use respectively
  `Theta(n^2)`, `O(n log n)`, and `Theta(n)` storage;
- for one slot with fan-out `F` and `B` pairwise distinct versions, Compact
  Closure uses `Theta(B+F)` records while any canonical materialized layout
  needs `Theta(BF)`; resolving an unmaterialized structural circuit requires
  `Omega(F)` output work;
- staircase is only the insert-only `k_j=1` corollary, not a required input
  class; arbitrary deep-ReLU toggles use the same change-sensitive bound;
- for tree ensembles, `k_j=q_j` when `q_j` trees change leaves at a boundary;
  for ExactLine/ReLU, `k_j=q_j` when `q_j` activation signs change;
- a Class-T non-Cartesian bound: tree partitions of sizes `b_m` require
  at most `1 + sum_m(b_m - 1)` line intervals, rather than their product;
- algebraic factorization beyond canonical subcircuit sharing remains a
  secondary optimization: unrestricted MGPF is NP-complete, while LCF targets
  exact/approximate algorithms for structured additive frontiers;
- a canonical minimum segment quotient after coalescing adjacent intervals
  with identical normalized provenance;
- `POINT` in `O(log B_u)` and registered-threshold `FLIP` in
  `O(log E_q + k)`, including tree-induced jumps.

These are scoped claims. TESSERA does not promise universal compression or
avoid the worst-case exponential number of realized piecewise-affine segments.
It matches, rather than asymptotically beats, numeric polyline point lookup;
the advantage is exact shared provenance plus indexed derived queries.
The active definitions are in
[`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md); Closure/VSC and scoped MGPF/LCF
proof drafts are in [`THEORY_PROOFS.md`](THEORY_PROOFS.md).
[`tests/test_lcf_theory.py`](tests/test_lcf_theory.py) exhaustively checks all
laminar families through four regions and all nonempty three-region support
matrices against the theorem bounds and an exact rectangle-partition oracle.

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

The valuation is shown below. The current prototype shares learned coefficients,
while the active paper semantics generalizes each parameter token to a slot and
allows a different binding `h_u` for every unit:

```text
do:T        -> intervention value t'
noise:X     -> abducted residual N_X(u)
bias:X      -> unit binding b_X(u)
theta:X->Z  -> unit binding w_XZ(u)
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

### Arbitrary-Toggle Closure and VSC

`experiments/tessera_vsc/versioned_circuit_demo.py` starts with a random active
set and toggles fragments at every boundary. Individual fragments repeatedly
turn off and back on, so these workloads are deliberately not staircase.

One local five-trial median run with 1,000 handle/root queries produced:

| Components | Regions | Changes/boundary | IIP records | VSC records | Closure records | VSC compression | Closure compression | VSC build speedup | Closure build speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 500 | 2 | 31,352 | 3,541 | 1,755 | 8.85x | 17.86x | 13.99x | 149.81x |
| 128 | 1,000 | 3 | 129,580 | 10,901 | 4,510 | 11.89x | 28.73x | 16.45x | 263.76x |
| 256 | 2,000 | 4 | 508,016 | 31,933 | 11,021 | 15.91x | 46.10x | 23.22x | 429.14x |

Against rebuilding the requested state from its event log, VSC root lookup was
`160.06x`, `424.05x`, and `897.31x` faster; Compact Closure handle lookup was
`66.81x`, `188.44x`, and `464.09x` faster. These are not speedups over IIP:
IIP, VSC, and Closure all have indexed region lookup. Closure returns an exact
`(template,environment version)` handle; VSC returns a directly traversable
resolved root. The primary comparison with IIP is construction and storage.

The 25 checked-in tests also cover mixed `ADD`/`MUL` circuits, non-idempotent
multiplicity, cross-unit structural sharing with different numeric bindings,
and a two-layer ReLU-style expression whose gates follow arbitrary
`ON/OFF/ON` patterns. Compact Closure is checked against Materialized VSC under
both numeric and free-semiring valuations, and its environment record count is
verified to be independent of template fan-out.

### Theorem-Aligned Scaling

The strict-separation workload fixes `B=C=n` and changes exactly one slot per
boundary. The normalized record counts remain stable in the predicted classes:

| n | IIP records | IIP / n^2 | VSC records | VSC / (n log2 n) | Closure records | Closure / n |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3,744 | 0.914 | 538 | 1.401 | 384 | 6.000 |
| 128 | 15,568 | 0.950 | 1,247 | 1.392 | 768 | 6.000 |
| 256 | 67,092 | 1.024 | 2,752 | 1.344 | 1,536 | 6.000 |
| 512 | 261,920 | 0.999 | 6,011 | 1.304 | 3,072 | 6.000 |

This is the experiment corresponding to the `Theta(n^2)`, `O(n log n)`, and
`Theta(n)` theorem. It is separate from the larger arbitrary-toggle table,
whose number of changes per boundary varies by configuration.

The fan-out separation family fixes `B=F=n` and changes one high-fan-out slot
to a fresh atom at every boundary:

| n | Materialized records | Materialized / n^2 | Closure records | Closure / n |
|---:|---:|---:|---:|---:|
| 64 | 8,514 | 2.079 | 450 | 7.031 |
| 128 | 33,410 | 2.039 | 898 | 7.016 |
| 256 | 132,354 | 2.020 | 1,794 | 7.008 |
| 512 | 526,850 | 2.010 | 3,586 | 7.004 |

This second family validates the `Theta(BF)` versus `Theta(B+F)` Pareto
separation. Compact Closure removes fan-out from stored version deltas; an
on-demand canonical structural expansion still pays `Omega(F)` work. This
claim is scoped to the frozen structural-EXPLAIN model: distributive algebraic
factorization may compress this particular polynomial and is a separately
costed layout, not part of the lower bound.

### Linear Foundation

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

Run the general non-staircase Closure/VSC benchmark:

```bash
python experiments/tessera_vsc/versioned_circuit_demo.py

python experiments/tessera_vsc/versioned_circuit_demo.py \
  --components 256 \
  --regions 2000 \
  --changes 4 \
  --queries 1000 \
  --trials 5
```

Run the theorem-aligned storage scaling check:

```bash
python experiments/tessera_vsc/theory_scaling_demo.py
```

Run all correctness and exhaustive theory tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
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
TECHNICAL_DESIGN.md                                 active problem, semantics, algorithms, proofs, and baselines
REPO_INTRO.md                                       copy-ready repository introduction
TODO.md                                             next steps
requirements.txt                                    minimal Python dependencies
requirements-dowhy.txt                              optional DoWhy comparison dependencies
counterprovenance/linear.py                          generic linear cDAG provenance engine
counterprovenance/versioned.py                       compact closures, materialized VSC, and region indexes
experiments/tessera_linear_trie/linear_trie_demo.py main prototype and benchmark
experiments/tessera_vsc/versioned_circuit_demo.py    arbitrary-toggle IIP/Closure/VSC benchmark
experiments/tessera_vsc/theory_scaling_demo.py       theorem-aligned n^2/n-log-n/n storage check
tests/test_linear_general.py                         correctness tests for arbitrary linear cDAGs
tests/test_lcf_theory.py                              exhaustive small-instance factorization verifier
tests/test_versioned_circuit.py                       Closure/VSC exactness, toggles, sharing, and space tests
```

## What This Prototype Does Not Claim

The checked-in implementation is intentionally narrower than the paper design.

It does not yet handle:

- fitted tree, ensemble, or ExactLine/ReLU local compilers;
- semiring-matrix compilation for a complete fitted MLP;
- FLIP or aggregate event indexes;
- the weighted algebraic-factorization optimizer over closure frontiers;
- batched multi-target workload optimization.

The prototype currently establishes three foundations:

1. linear counterfactual replay can be represented as semiring provenance and
   compiled into a fast query plan;
2. Compact Closure stores arbitrary ordered provenance versions as one shared
   template plus exactly one record per effective slot change;
3. Materialized VSC shares exact `ADD` and `MUL` subcircuits without a
   staircase assumption and returns a directly traversable root per region.

## Research Direction

The repositioned paper is:

**TESSERA: An Exact Provenance Index for Parametric Counterfactual Queries over
Piecewise-Affine Structural Causal Models**

The implementation path now follows one closure semantics with two layouts:

1. route Linear compilation through Closure as its one-version case;
2. emit tree leaf changes as exact slot-environment deltas;
3. lift ExactLine activation changes to exact slot deltas without path expansion;
4. add compact columnar histories and byte-accurate layout accounting;
5. add algebraic LCF/min-cut optimization only at suitable additive frontiers;
6. implement FLIP/AGG and evaluate mixed SCM workloads.
