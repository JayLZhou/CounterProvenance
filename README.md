# TESSERA Linear Trie

Repository: <https://github.com/JayLZhou/CounterProvenance>

TESSERA Linear Trie is a small prototype for **fast repeated counterfactual
queries over learned linear SCMs on arbitrary causal DAGs**.

The core idea is simple:

> A linear additive-noise SCM does not need to replay every mechanism for every
> counterfactual query. Its counterfactual replay can be compiled into semiring
> provenance, stored as a compressed propagation trie, and evaluated through a
> fast closed-form query plan.

This repository is the first sanity check for the larger TESSERA paper idea:
**factorized semiring provenance for counterfactual query processing over
learned structural causal models**.

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

This demo answers that question for linear SCMs over arbitrary cDAGs.

## Exact f(.) Coverage

The broader TESSERA paper classifies structural mechanisms by their exact
region structure, rather than by model family names. A mechanism is in scope
when it can be written as a finite guarded expression:

```text
f(x) = sum_r 1[x in Omega_r] * h_r(x)
```

where each region guard `Omega_r` is exact, and each local expression `h_r` is
either constant or affine.

This makes linear models, trees, tree ensembles, and ReLU-style networks part
of one exact finite-region family:

| Mechanism class | Region structure | Local expression `h_r(x)` | Typical models | Provenance representation | Compression opportunity | Prototype status |
|---|---|---|---|---|---|---|
| Global affine | One global region, `Omega = R^d` | `a^T x + b` | LinearRegression, Ridge, Lasso, OLS additive-noise mechanisms | affine semiring path provenance | shared causal path suffixes; compiled linear form | implemented |
| Global basis-affine | One global region after explicit feature lifting | `a^T phi(x) + b` | polynomial regression, fixed basis expansion | symbolic expression DAG over basis tokens | shared basis subexpressions and causal paths | planned |
| Guarded constant, or piecewise constant | Finite guarded regions, usually one per leaf | `c_r` | decision tree, rule list, lookup table over finite categories | guard provenance plus leaf-value token | shared guard prefixes; collapse subtrees with identical leaf provenance | planned |
| Additive guarded constant ensemble | Sum of many guarded-constant components | `sum_m c_{m,r_m}` without materializing global leaf combinations | random forest, GBDT, XGBoost/LightGBM-style tree ensembles | factorized sum of tree-level guarded provenance | avoids Cartesian-product region explosion across trees | high-priority planned |
| Guarded affine, or piecewise affine | Finite guarded regions, one per activation/sign/segment pattern | `a_r^T x + b_r` | ReLU MLP, leaky-ReLU MLP, maxout, segmented linear models, linear splines | guard provenance plus affine template | shared activation prefixes; shared affine templates; dead-path pruning | planned |
| Additive guarded affine | Additive sum of smaller guarded affine components | `sum_m (a_{m,r}^T x_m + b_{m,r})` | GAM-style splines, additive piecewise-linear models | factorized component-level provenance | avoids full cross-product of component regions | planned |

Smooth mechanisms such as sigmoid/tanh MLPs and arbitrary black-box regressors
can still be evaluated by a causal library, but they are not exact provenance
targets for this paper unless they are replaced by an explicit finite-region
surrogate.

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
REPO_INTRO.md                                       copy-ready repository introduction
TODO.md                                             next steps
requirements.txt                                    minimal Python dependencies
requirements-dowhy.txt                              optional DoWhy comparison dependencies
counterprovenance/linear.py                          generic linear cDAG provenance engine
experiments/tessera_linear_trie/linear_trie_demo.py main prototype and benchmark
tests/test_linear_general.py                         correctness tests for arbitrary linear cDAGs
```

## What This Prototype Does Not Claim

This demo is intentionally narrow.

It does not yet handle:

- nonlinear mechanisms;
- ReLU/tree guarded provenance;
- optimized trie execution;
- batched multi-target workload optimization.

Those are the next steps in the broader TESSERA agenda. This prototype only
establishes the linear foundation:

> linear counterfactual replay can be represented as semiring provenance,
> compressed as a propagation trie, and compiled into a fast query plan.

## Research Direction

The larger paper direction is:

**TESSERA: Factorized Semiring Provenance for Counterfactual Query Processing
over Learned SCMs**

Planned extension path:

1. linear SCMs as the single-region affine baseline;
2. decision trees as guarded constant mechanisms;
3. tree ensembles as additive guarded-constant mechanisms that avoid global
   leaf-combination materialization;
4. ReLU MLPs and piecewise-linear models as guarded affine mechanisms;
5. provenance-equivalence compression and indexing for repeated counterfactual
   workloads.
