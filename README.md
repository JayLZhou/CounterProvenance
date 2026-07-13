# TESSERA Counterfactual Provenance Index

Repository: <https://github.com/JayLZhou/CounterProvenance>

TESSERA is a research prototype for **exact indexed counterfactual queries over
learned structural causal models**. The current code implements the linear
foundation on arbitrary causal DAGs and a model-independent **Versioned
Semiring Circuit (VSC)** core. Tree and ReLU compilers remain future work, but
their arbitrary region-to-region ON/OFF patterns no longer require a
staircase assumption in the storage layer.

The core idea is:

> Compile the exact intervention line into an ordered sequence of semiring
> circuit versions. Store every distinct `ADD`/`MUL` subcircuit once in a
> global immutable arena, store per-unit bindings separately, and let each
> unit-region keep only a persistent root pointer.

The full mechanism class, index design, baselines, and theorem targets are in
[`POSITIONING.md`](POSITIONING.md). The current linear trie is the one-region
special case; `counterprovenance/versioned.py` implements the general
arbitrary-version storage substrate.

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

## Unified Index: VSC-Backed GPSI

For unit `u`, region `j`, let `P[u,j]` be its exact parameterized provenance and
let `h_u` bind arbitrary unit-specific parameters, roots, and abducted noises.
VSC stores the ordered family

```text
root[u,1], root[u,2], ..., root[u,B_u]
```

inside one global hash-consed arena containing `ZERO`, `ONE`, `ATOM`, `ADD`,
`MUL`, and optional `SWITCH` nodes. At a region boundary, only changed slots
and their affected ancestors are rebuilt. A fragment may follow
`ON -> OFF -> ON` repeatedly; non-adjacent regions and different units still
reuse an identical root whenever canonical structure matches. Numeric values
remain outside the shared topology in each unit's binding `h_u`.

The Guarded Provenance Segment Index has one logical and physical design for
every supported mechanism:

```text
GPSI
|-- local mechanism DAG
|-- response segment directory
|      sorted boundaries -> persistent root IDs
|-- Versioned Semiring Circuit
|      immutable ZERO/ONE/ATOM/ADD/MUL/SWITCH nodes
|      batch slot updates + affected-ancestor path copying
|      global hash-consing across regions and units
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
frontier, but it is no longer the universal logical model: VSC preserves
products directly and therefore does not require expansion into neural paths.

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

The revised theory targets are:

- exact closure of L/T/N under causal composition for scalar interventions;
- general persistent storage
  `O(B_tot + D_0 + sum_(u,j) gamma_(u,j) + sum_u|h_u|)`, where `gamma_(u,j)`
  is the number of new canonical nodes created by a region transition;
- construction
  `O(B_tot + D_0 + sum_(u,j) alpha_(u,j) + sum_u|h_u|)`, where `alpha_(u,j)`
  includes changed slots and the affected template ancestors at that transition and
  `gamma_(u,j) <= alpha_(u,j)`;
- for a balanced associative block with `C` mutable slots and `k_j` changed
  slots at one boundary, batch path copying touches
  `O(k_j log(1 + C/k_j))` template nodes;
- independent interval provenance stores
  `Theta(sum_(u,j)|P[u,j]|)` nodes, while VSC keeps the same logarithmic region
  lookup and returns a compressed root in constant time after lookup;
- staircase is only the insert-only `k_j=1` corollary, not a required input
  class; arbitrary deep-ReLU toggles use the same change-sensitive bound;
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
[`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md); VSC and scoped MGPF/LCF proof
drafts are in [`THEORY_PROOFS.md`](THEORY_PROOFS.md).
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

### Arbitrary-Toggle VSC

`experiments/tessera_vsc/versioned_circuit_demo.py` starts with a random active
set and toggles fragments at every boundary. Individual fragments repeatedly
turn off and back on, so these workloads are deliberately not staircase.

One local three-trial median run with 1,000 root queries produced:

| Components | Regions | Changes/boundary | IIP node occurrences | VSC nodes + roots | Compression | Build speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 500 | 2 | 30,852 | 3,414 | 9.04x | 12.41x |
| 128 | 1,000 | 3 | 128,580 | 10,646 | 12.08x | 16.49x |
| 256 | 2,000 | 4 | 506,016 | 31,422 | 16.10x | 20.05x |

VSC root lookup was `146.89x`, `399.73x`, and `872.39x` faster than rebuilding
the requested state from its event log. This is not presented as a speedup over
IIP: both IIP and VSC already keep one indexed root per region and therefore
have the same lookup complexity. VSC's advantage over IIP is construction and
storage sharing while retaining that lookup time.

The checked-in tests also cover mixed `ADD`/`MUL` circuits, non-idempotent
multiplicity, cross-unit structural sharing with different numeric bindings,
and a two-layer ReLU-style expression whose gates follow arbitrary
`ON/OFF/ON` patterns.

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

Run the general non-staircase VSC benchmark:

```bash
python experiments/tessera_vsc/versioned_circuit_demo.py

python experiments/tessera_vsc/versioned_circuit_demo.py \
  --components 256 \
  --regions 2000 \
  --changes 4 \
  --queries 1000 \
  --trials 3
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
counterprovenance/versioned.py                       persistent semiring circuit and region index
experiments/tessera_linear_trie/linear_trie_demo.py main prototype and benchmark
experiments/tessera_vsc/versioned_circuit_demo.py    arbitrary-toggle VSC benchmark
tests/test_linear_general.py                         correctness tests for arbitrary linear cDAGs
tests/test_lcf_theory.py                              exhaustive small-instance factorization verifier
tests/test_versioned_circuit.py                       VSC, products, toggles, sharing, and space tests
```

## What This Prototype Does Not Claim

The checked-in implementation is intentionally narrower than the paper design.

It does not yet handle:

- fitted tree, ensemble, or ExactLine/ReLU local compilers;
- semiring-matrix compilation for a complete fitted MLP;
- FLIP or aggregate event indexes;
- the weighted algebraic-factorization optimizer over VSC frontiers;
- batched multi-target workload optimization.

The prototype currently establishes two foundations:

1. linear counterfactual replay can be represented as semiring provenance and
   compiled into a fast query plan;
2. arbitrary ordered provenance-circuit versions can share exact `ADD` and
   `MUL` subcircuits without a staircase assumption.

## Research Direction

The repositioned paper is:

**TESSERA: An Exact Provenance Index for Parametric Counterfactual Queries over
Piecewise-Affine Structural Causal Models**

The implementation path now follows one versioned circuit substrate:

1. connect Linear compilation to VSC as its one-version case;
2. emit tree leaf-selector updates into persistent semiring sum trees;
3. lift ExactLine activation masks to semiring selectors without path expansion;
4. formalize change complexity `gamma` and prove the batch path-copy bound;
5. add algebraic LCF/min-cut optimization only at suitable additive frontiers;
6. implement FLIP/AGG and evaluate mixed SCM workloads.
