# TESSERA Paper Positioning

## One-Sentence Positioning

TESSERA is an exact database index for repeated parametric counterfactual
queries over additive-noise SCMs whose learned mechanisms are drawn from three
classes: linear, numerical tree ensembles, and ExactLine-style piecewise-linear
neural networks.

The indexed object is not a model region and not a cached answer. For a fixed
unit `u`, scalar intervention variable `T`, target `Y`, and model version,
TESSERA indexes the complete provenance-valued response function

```text
Pi[u,Y](t) = Y_{u; do(T=t)}.
```

This response is piecewise affine in `t`. Each maximal interval points to a
shared semiring-provenance template that can reproduce both the numeric answer
and its derivation.

## Core Workload

The theorem-backed index targets:

```text
input:
  an acyclic additive-noise SCM M
  an observed population U
  a fixed scalar treatment T
  one or more targets Y
  a bounded intervention domain [t_min, t_max]

workload:
  POINT(u, Y, t)            -> exact counterfactual value and provenance
  FLIP(Y, threshold, range) -> units whose outcome crosses the threshold
  AGG(group, Y, t)          -> exact SUM/AVG counterfactual response
  EXPLAIN(u, Y, t)          -> active facts, noises, parameters, and causal paths
```

The logical provenance circuit can represent multiple interventions. The
canonical segment index and its strongest complexity guarantees are scoped to
one scalar intervention variable. Exact multi-dimensional point location is a
separate extension, not a headline claim.

## Three Supported Mechanism Classes

The paper claims exact support for exactly three mechanism classes:

1. global affine mechanisms;
2. additive numerical regression-tree ensembles;
3. piecewise-linear neural mechanisms in the ExactLine class.

A single regression tree is the `M=1` special case of the second class. A mixed
SCM may assign any of the three classes to different causal nodes; mixed
composition demonstrates one unified system and is not a fourth class.

### Shared Formal Envelope

For every endogenous node `v`, TESSERA assumes

```text
X_v = f_v(X_pa(v)) + N_v,
```

where `N_v` is the abducted unit-level noise and `f_v` has a finite
guarded-affine decomposition:

```text
f_v(x) = a_r^T x + b_r  when guard g_r(x) is true.
```

The implementation may preserve an additional additive factorization:

```text
f_v(x) = sum_{m=1..M_v} lambda_m * f_{v,m}(x)

f_{v,m}(x) = a_{m,r}^T x + b_{m,r}  when guard g_{m,r}(x) is true.
```

For each component `m`, the guards are finite, mutually exclusive, exhaustive,
and exactly line-intersectable: when the parent response is affine on an
interval,

```text
x(t) = A*t + c,
```

the compiler can enumerate every value of `t` at which the active guard can
change. This envelope explains why the same provenance algebra and segment
index serve all three classes. It is not a claim that every function satisfying
the envelope is implemented.

### Claimed Coverage

| Paper class | Representative fitted `f(.)` | Exact local compiler | Status |
|---|---|---|---|
| L: Linear | OLS, `LinearRegression`, Ridge, Lasso, ElasticNet, linear-kernel SVR | one affine segment; no guard refinement | implemented for generic linear cDAGs |
| T: Tree Ensemble | one regression tree, Extra Tree, random forest, Extra Trees, GBDT, histogram GBDT, numerical XGBoost/LightGBM/CatBoost with raw or identity-link regression output | trace numerical thresholds on each parent segment; merge per-tree breakpoint streams | first nonlinear implementation target |
| N: ExactLine/PWL Neural | ReLU MLPs and inference graphs composed of affine layers with ExactLine-supported piecewise-linear operators such as ReLU and MaxPool | ExactLine-style exact line partitioning, lifted from numeric affine pieces to provenance-annotated pieces | second nonlinear implementation target |

Class T leaf outputs are numeric constants, hence affine expressions with zero
slope. Class N reuses the line-partition primitive introduced by
[ExactLine](https://papers.nips.cc/paper_files/paper/2019/hash/908075ea2c025c335f4865f7db427062-Abstract.html),
but adds SCM abduction semantics, typed provenance propagation, cross-unit/path
factorization, validity indexing, and provenance-native queries.

### One Compiler Contract

Every class implements the same local interface:

```text
COMPILE_LINE(f_v, parent_responses, intervention_domain)
  -> sorted [(left, right, alpha, beta, provenance, certificate)]
```

- L returns one segment by affine substitution.
- T traces each tree independently and merges its sorted breakpoint streams.
- N invokes an ExactLine-style partitioner and lifts every numeric piece to a
  provenance-annotated piece.

All subsequent causal composition, prefix factorization, segment quotienting,
storage, and query processing are class independent.

### Explicitly Outside the Three-Class Claim

The following predictors are not covered by the main exact index:

- sigmoid, tanh, GELU, softplus, or arbitrary smooth neural networks;
- RBF SVR and RBF Gaussian processes;
- polynomial regression and polynomial kernels, which are exact but not affine
  in the original variables;
- KNN, rule lists, splines/GAMs, and weighted-median AdaBoost, even where an
  exact specialized compiler could be designed;
- arbitrary user-defined black-box predictors;
- categorical classifier mechanisms and class-probability semantics;
- vector-valued interventions for the canonical segment-index theorem.

They must not be described as supported merely because they expose sklearn's
`fit` and `predict` interface. Additional exact compilers are future work, not
part of the submitted system's coverage claim.

## Unified Provenance Semantics

Let `n[u,v]` be the noise abducted for unit `u` and node `v`. Counterfactual
provenance is defined recursively in topological order:

```text
Pi[u,v](t) =
  t                                      if v = T,
  observed_root[u,v]                     if v is a non-intervened root,
  f_v(Pi[u,pa(v)](t)) + noise[u,v]       otherwise.
```

For a guarded-affine mechanism, the last line is represented as

```text
sum_r [guard_r(Pi[u,pa(v)](t))]
      * (bias_r + sum_p theta[r,p] * Pi[u,p](t) + noise[u,v]).
```

The provenance vocabulary is shared by every supported model:

```text
facts          observed root or anchor tokens
noises         abducted unit-level noise tokens
parameters     coefficients, biases, leaf values, and network weights
interventions  the symbolic value do(T=t)
guards         split predicates or activation signs
paths          references induced by causal composition
```

The segment interval is the normalized validity guard for the indexed response.
Raw tree paths or activation signatures can be retained in a separate witness
DAG when an audit query requires them; they are not allowed to prevent two
otherwise identical response-provenance templates from being coalesced.

Abduction is a boundary operation. Residual recovery uses subtraction, so the
positive semiring starts after abduction and treats each recovered noise as an
atomic token. A model update therefore versions both the noise materialization
and the segment directory. TESSERA does not claim that retraining only rewrites
a model-level template dictionary.

## One Unified Index

The physical design is the Guarded Provenance Segment Index (GPSI).

```text
GPSI
|
|-- Local Mechanism DAG
|     exact guarded-affine representation of each fitted f_v
|
|-- Shared Prefix-Factorized Provenance DAG
|     reverse trie over canonical target-oriented derivation paths
|     hash-consed ADD/MUL/ATOM/reference nodes across templates and units
|
|-- Guard Certificate DAG
|     optional raw split/activation witnesses for audit queries
|
|-- Segment Directory
|     (model_version, unit, treatment, target)
|       -> sorted [(left, right, alpha, beta, template_id, certificate_id)]
|
|-- Unit Store
|     factual anchor pointer + versioned abducted-noise pointer
|
`-- Secondary Event Indexes
      sorted threshold crossings for FLIP
      optional preaggregated segment maps for SUM/AVG
```

There is no separate linear index, tree index, and neural index. Every
mechanism supplies only a local guard-crossing compiler. Causal composition,
template reduction, segment coalescing, storage, and query processing are
shared.

### Construction

For each unit and target workload, the compiler proceeds in causal topological
order:

```text
COMPILE(M, u, T, domain):
  response[T] <- one segment containing intervention token t

  for v in topological_order(M):
    if v is T:
      continue
    if v is a root:
      response[v] <- one segment containing factual token root[u,v]
      continue

    parent_partition <- merge breakpoints of response[pa(v)]
    candidate <- compose local_mechanism[v] with response[pa(v)]
    refined <- split candidate only at exact guard crossings
    normalized <- intern provenance templates in the shared DAG
    response[v] <- coalesce adjacent response-provenance-equivalent segments

  return response
```

The compiler never enumerates all global model regions. It visits only regions
reached by the one-dimensional intervention trajectory and retains only
maximal intervals with distinct provenance.

### Mandatory Prefix Factorization

Prefix sharing is a required physical invariant of GPSI, not an optional
model-specific optimization. Every provenance derivation is serialized in a
canonical token order. For a fixed target, source-to-target causal paths are
reversed before insertion, so a common suffix near the target becomes a common
trie prefix.

```text
source * theta:e1 * theta:e2 * theta:e3

stored as

theta:e3 -> theta:e2 -> theta:e1 -> source
```

This gives three forms of exact sharing:

- causal propagation prefixes are shared by the reverse provenance trie;
- tree split or neural activation prefixes are shared by local mechanism DAGs;
- identical trie/subcircuit nodes are hash-consed across segment templates and
  units, turning the collection of tries into one shared DAG.

Repeated template-ID sequences may additionally be dictionary encoded, but
every unit-specific numeric breakpoint, slope, and intercept is retained unless
it has an exactly shared representation. A suffix automaton, BST, or ANN is not
the primary abstraction.

A trie is not the whole index: it compresses derivations inside templates,
while the segment directory indexes where each template is valid. These are
orthogonal dimensions of compression.

## Baselines

Let:

```text
n   number of units
C_Q cost of replaying the causal slice for one unit and query
B_u number of maximal provenance segments for unit u
S   sum_u B_u
L   average size of an unfactorized provenance expression per segment
D   number of nodes in the shared prefix-factorized provenance DAG
E   number of materialized threshold-crossing events
k   output size of a reporting query
```

| Baseline | What it stores | Point-query work | Space | Missing capability |
|---|---|---:|---:|---|
| B0: DoWhy-style replay | nothing reusable | `O(n*C_Q)` for all units; may repeat abduction | `O(1)` beyond model/data | no indexed provenance or structural FLIP query |
| B1: cached-noise replay | abducted noises | `O(n*C_Q)` without abduction | `O(n*|V|)` | still traverses every active mechanism per query |
| B2: global-region materialization | one record per global leaf/activation combination | lookup after potentially exponential construction | proportional to materialized global regions | enumerates unreachable and duplicate provenance regions |
| B3: numeric per-unit polyline | boundaries plus numeric slope/intercept | `O(log B_u)` | `Theta(S)` | matches point lookup but cannot return derivation provenance |
| B4: provenance per-unit polyline | every segment with a full affine expression | `O(log B_u)` | `Theta(S*L)` | duplicates templates and causal subexpressions |
| B5: tree-ensemble ADD/decision diagram | tree predicates and numeric terminals | diagram traversal | diagram-dependent | no abducted-noise, causal-path, or unit segment semantics |
| B6: ExactLine | exact numeric affine pieces for one PWL neural input line | `O(log B_u)` after construction | `Theta(S)` numeric pieces | no SCM abduction, typed provenance, cross-unit sharing, or derived indexes |
| TESSERA GPSI | maximal segments plus pointers to a shared provenance DAG | `O(log B_u)` per unit | `O(S + D + n*|V|)` | deliberately supports only L/T/N |

B1 and B3 are the strongest speed baselines. TESSERA does not claim a better
asymptotic point lookup than a numeric polyline; it matches that bound while
retaining exact derivation provenance. B4 is the storage baseline for that
additional capability. B2 exposes the asymptotic benefit of
trajectory-sensitive factorization. B5 is the required Class-T compilation
baseline. B6 is the required Class-N line-partition baseline and isolates the
cost and benefit of provenance lifting.

Per-class comparisons are fixed:

- L: DoWhy/NumPy replay versus the current compiled linear provenance plan;
- T: cached-noise ensemble replay, per-tree line tracing, and ADD compilation;
- N: direct neural replay and unmodified numeric ExactLine versus
  provenance-lifted ExactLine inside GPSI.

## Related-Work Boundary

The novelty cannot be semiring provenance, circuit sharing, or decision-diagram
reduction by itself. Those ideas already have established formulations in
[provenance semirings](https://repository.upenn.edu/entities/publication/f1141264-46ee-4d61-b5ea-4ee75fb8d1be),
[factorized databases](https://ora.ox.ac.uk/objects/uuid%3A9116c474-1e8a-4d63-943e-e0d038454f25),
and [tree-ensemble compilation to algebraic decision diagrams](https://eldorado.tu-dortmund.de/server/api/core/bitstreams/41b47e64-79f2-47fa-baf2-91e0dcf8df6c/content).
Likewise, [ExactLine](https://papers.nips.cc/paper_files/paper/2019/hash/908075ea2c025c335f4865f7db427062-Abstract.html)
already computes exact numeric affine partitions of piecewise-linear neural
networks along one input line. TESSERA treats it as the Class-N local compiler
and baseline, not as the provenance or index contribution.

TESSERA's contribution is their counterfactual-query-specific composition:

1. abduction produces versioned unit-level noise atoms;
2. causal composition produces a provenance-valued parametric response, not
   merely a predictor decision diagram;
3. the scalar intervention line induces a canonical maximal-interval quotient;
4. the index shares derivation templates while retaining unit-specific numeric
   segments;
5. threshold crossings and aggregates become indexed database queries.

## Theoretical Advantages

The paper should prove the following claims. They are stronger and more
specific than a generic statement that prefix sharing saves space.

### Theorem 1: Exact Compositional Closure

For an acyclic additive-noise SCM whose mechanisms are finite
guarded-affine and exactly line-intersectable, fixing a unit and one scalar
intervention variable yields a finite piecewise-affine provenance function at
every descendant node. GPSI construction returns exactly that function.

Proof direction: induction over the causal topological order. Affine
substitution preserves affine expressions on each current interval. A finite
set of exact guard crossings produces a finite refinement. Addition of the
fixed noise token preserves the local form.

### Theorem 2: Additive Non-Cartesian Bound

Suppose an additive mechanism has components `f_1, ..., f_M`. On one parent
interval, component `m` induces `b_m` subintervals along the intervention line.
The exact common refinement for their sum contains at most

```text
1 + sum_m (b_m - 1)
```

intervals. A component-combination baseline can materialize up to

```text
product_m b_m
```

combinations. Thus Class-T tree ensembles admit an exact sum-sized line index
without a Cartesian-product region construction.

This theorem does not claim that the full SCM is polynomial in the worst case;
causal composition can still generate exponentially many realized segments.

Proof direction: component `m` contributes at most `b_m - 1` internal
breakpoints. The common refinement is induced by the union of those breakpoint
sets, whose cardinality is at most their sum. Taking a Cartesian product of
component segment IDs is therefore unnecessary on a line.

### Theorem 3: Output-Sensitive Construction

Let `H` be the number of local guard/circuit nodes inspected on the realized
intervention trajectories, `C` the number of exact crossing events emitted by
local compilers, and `S_all` the number of segments retained across all
materialized intermediate responses. With sorted breakpoint merging and
hash-consing, GPSI can be constructed in

```text
O(H + C*log(C + 1) + S_all + D)
```

time and `O(S_all + D)` working space, in addition to the unit noise store. The
bound contains no term for the number of potential global activation/leaf
regions. If a local compiler emits crossings already sorted, its sorting term
reduces to linear multiway-merge cost.

Any explicit exact segment index needs `Omega(S)` time and space merely to
write its final segment records. GPSI is therefore output-sensitive and
optimal in final segment output size up to local guard inspection, sorting,
and provenance construction.

This theorem is parameterized by the L/T/N local compiler. L has no crossings;
T counts visited split nodes; N inherits ExactLine-style layer/segment work.
The theorem does not hide that `H`, `C`, and `S_all` can be exponential in a
worst-case deep piecewise-linear SCM.

### Theorem 4: Exact Prefix-Factorization Bound

Let `P_occ` be the multiset of canonical target-oriented token-path occurrences
in the materialized provenance templates, let `P_distinct` be its set of
distinct paths, and let

```text
L_expanded = sum_{p in P_occ} |p|.
```

An expanded path store uses `Theta(L_expanded)` token slots. The reverse trie
uses exactly

```text
1 + |Pref(P_distinct)|
```

nodes, where `Pref(P_distinct)` is the set of distinct non-empty path prefixes.
Hence

```text
1 + |Pref(P_distinct)| <= 1 + L_expanded,
```

with a strict inequality whenever a path is repeated or two distinct
derivations share a non-empty reversed causal prefix. For the fixed canonical
token order, every exact trie must distinguish all distinct prefixes, so this
representation is minimum among tries respecting that order. Subsequent
hash-consing can merge additional isomorphic subcircuits without weakening the
bound.

Proof direction: root-to-leaf trie paths are in one-to-one correspondence with
the canonical provenance paths. Each trie node corresponds to exactly one
distinct prefix. Semiring branching preserves addition and edge traversal
preserves multiplication, so trie evaluation is exactly the expanded
provenance polynomial.

### Theorem 5: Canonical Minimum Segment Quotient

Define two adjacent intervals as equivalent when their normalized contributing
provenance templates are identical, including parameter, noise, intervention,
and causal-path tokens. The interval directory itself is the normalized guard.
The union of two adjacent interval guards is again one interval, so repeatedly
coalescing equivalent neighbors produces the unique partition into maximal
connected response-provenance-equivalence classes.

Raw activation signatures or tree-path witnesses may differ across a merged
interval. If requested, they remain in the optional certificate DAG as a finer
audit view; they are not part of the minimum response-index equivalence.

For a fixed unit, treatment, target, model version, and canonical template
normalization, every exact interval index that stores one template per interval
requires at least `B_u` intervals. GPSI stores exactly `B_u`.

This is the main minimality theorem. It is intentionally restricted to the
ordered one-dimensional interval-index class; it does not claim globally
minimum arbitrary arithmetic circuits.

Proof direction: every boundary retained by the reduced partition separates
two unequal response-provenance templates. Any exact interval leaf crossing
that boundary would assign one template to two different responses and is
therefore unsound. Hence every exact interval index must retain every reduced
boundary. Coalescing removes all and only the unnecessary boundaries.

### Theorem 6: Factorized Storage Bound

An exact provenance per-unit polyline baseline uses `Theta(S*L)` expression
slots. A numeric-only polyline already uses `Theta(S)` and is therefore the
matching lower-space speed baseline. GPSI adds exact shared provenance using

```text
O(S + D + n*|V|)
```

space: one boundary, numeric affine payload, and template pointer per maximal
segment, one shared copy of each canonical provenance-DAG node, and one
unit-level noise vector.

Strict provenance compression is workload dependent. In the worst case every
provenance template is unique, and any exact representation needs `Omega(S)`
segment records. The theorem provides a dominance bound and a matching
lower-bound story rather than an invalid universal compression claim.

### Theorem 7: Indexed Query Bounds

With sorted segment boundaries:

```text
POINT(u, Y, t) = O(log B_u)
```

because each segment materializes its numeric `alpha*t + beta` payload. Emitting
the requested provenance additionally costs the size of the returned active
template or witness. With sorted crossing events:

```text
FLIP(range) = O(log E + k).
```

Without a crossing index, replay or per-unit curve evaluation must inspect all
`n` units to discover the same answer. For predeclared groups, an aggregated
piecewise-affine map supports numeric `SUM/AVG(Y(t))` lookup in logarithmic
time after preprocessing.

Proof direction: point lookup is predecessor search over sorted boundaries.
Flip lookup is a range query over sorted crossing events followed by reporting.
For an unindexed collection of per-unit responses, an unseen unit can always
contain the only qualifying crossing, giving an `Omega(n)` inspection lower
bound in the worst case.

## Paper Claims and Non-Claims

### Claims

1. Counterfactual replay over L/T/N and their mixed SCM compositions has an
   exact guarded semiring-provenance semantics.
2. A scalar intervention induces a finite provenance-valued piecewise-affine
   response that can be indexed directly.
3. GPSI factorizes causal paths, additive mechanism components, provenance
   templates, and unit-specific segment boundaries without global-region
   enumeration.
4. Canonical provenance paths are stored once per distinct reversed prefix.
5. The reduced segment directory is canonical and minimum within the exact
   one-dimensional interval-index class.
6. POINT, FLIP, AGG, and EXPLAIN become storage operations rather than fresh
   SCM replays.

### Non-Claims

1. GPSI does not exactly support every sklearn-compatible DoWhy predictor.
2. GPSI does not guarantee compression when every unit and segment has unique
   provenance.
3. GPSI does not avoid worst-case exponential piecewise complexity.
4. GPSI does not make multi-dimensional exact point location cheap.
5. The current repository implements only the one-region affine foundation;
   Class-T and Class-N compilers remain to be built and evaluated.

## Required Experimental Story

The evaluation must separate four effects:

1. **Semantics:** exact agreement with DoWhy-style abduction-action-prediction.
2. **Region avoidance:** realized segments versus potential global regions.
3. **Factorization:** expanded path slots versus distinct trie prefixes, and
   `Theta(S*L)` per-unit polyline storage versus `O(S+D)` GPSI storage.
4. **Indexing:** POINT, FLIP, and AGG latency versus cached-noise replay and
   exact per-unit polyline baselines.

Required mechanism classes:

- L: linear SCMs as the one-segment control;
- T: single regression trees (`M=1`), random forests, and GBDT;
- N: multi-layer ReLU MLPs accepted by the ExactLine-style compiler;
- mixed SCMs whose nodes independently choose L, T, or N.

Required plots:

- realized segment count versus model size and causal depth;
- unique provenance-template count versus population size;
- expanded provenance token slots versus reverse-trie prefixes and DAG nodes;
- index size and build time versus exact per-unit polyline;
- query latency versus number of units and number of repeated intervention
  values;
- FLIP latency versus output selectivity;
- retraining invalidation/rebuild cost, without claiming that only templates
  change.

## Recommended Paper Title

**TESSERA: An Exact Provenance Index for Parametric Counterfactual Queries over
Piecewise-Affine Structural Causal Models**
