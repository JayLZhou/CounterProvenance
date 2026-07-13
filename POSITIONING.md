# TESSERA Paper Positioning

The active design is recorded in [`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md).
Its general logical object is a Versioned Provenance Closure with two exact
physical layouts: storage-optimal Compact Closure and latency-oriented
Materialized VSC. MGPF/LCF remains an additive-frontier optimizer.

## One-Sentence Positioning

TESSERA is an exact database index for repeated parametric counterfactual
queries over additive-noise SCMs whose learned mechanisms are drawn from three
classes: linear, numerical tree ensembles, and ExactLine-style piecewise-linear
neural networks.

The indexed object is not a model region and not a cached answer. For an indexed
population `U`, scalar intervention variable `T`, target `Y`, and model version,
TESSERA jointly indexes the complete provenance-valued response family

```text
Pi[u,Y](t) = Y_{u; do(T=t)}.
```

Each unit response is piecewise affine in `t`. Its exact regions form an
ordered sequence of semiring templates paired with mutable slot environments:

\[
\mathcal C_{u,j}=(T_u,\sigma_{u,j}).
\]

Compact Closure stores each effective slot change once. Materialized VSC
resolves every closure into a canonical immutable root and hash-conses repeated
`ADD/MUL` subcircuits. Per-unit homomorphisms `h_u` bind arbitrary weights,
roots, and noises. Selectors may follow arbitrary `ON/OFF/ON` patterns.

The general physical semantics is:

\[
\Pi_u(t)=h_{u,t}(\operatorname{Eval}(\mathcal C_{u,j(t)})),
\qquad
j(t)=\operatorname{pred}_{\mathcal B_u}(t),
\]

where `B_u` is the sorted region-boundary directory and
`h_{u,t}=ev_t o h_u`. In latency mode,
`R[u,j]=Materialize(C[u,j])` is the global VSC root. At a selected additive
frontier, an optional guarded factorization may still represent roots as
interval-valid payload bundles; it is not required by the general semantics.

## Core Workload

The theorem-backed index targets:

```text
input:
  an acyclic additive-noise SCM M
  an observed population U
  a fixed scalar treatment T
  one or more targets Y
  a bounded intervention domain [t_min, t_max]
  optional registered (target, threshold) FLIP workloads
  optional declared (group, target, SUM|AVG) workloads

workload:
  POINT(u, Y, t)                -> exact value and response provenance
  EXPLAIN(u, Y, t, mode)        -> response derivation and optional audit witness
  FLIP(Y, registered_q, range)  -> threshold-state changes, including tree jumps
  AGG(declared_group, Y, t)     -> exact preprocessed SUM/AVG response
```

The indexed FLIP bound applies only to thresholds registered at build time, and
the indexed AGG bound applies only to declared groups. Arbitrary thresholds or
groups remain valid replay/scan queries but do not receive those bounds.

The logical provenance circuit can represent multiple interventions. The
canonical segment index and its strongest complexity guarantees are scoped to
one scalar intervention variable. Exact multi-dimensional point location is a
separate extension, not a headline claim.

## Three Supported Mechanism Classes

The paper claims exact support for exactly three mechanism classes:

1. affine mechanisms with arbitrary per-unit parameter bindings;
2. additive numerical regression-tree ensembles;
3. piecewise-linear neural mechanisms in the ExactLine class.

A single regression tree is the `M=1` special case of the second class. A mixed
SCM may assign any of the three classes to different causal nodes; mixed
composition demonstrates one unified system and is not a fourth class.

### Shared Formal Envelope

For every endogenous node `v`, TESSERA assumes

```text
X^u_v = f^u_v(X^u_pa(v)) + N^u_v,
```

where `N^u_v` is the abducted unit-level noise and `f^u_v` has a finite
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

Every class implements the same cell-local interface:

```text
COMPILE_CELL(f_v, parent_affine_lines, parent_provenance, interval)
  -> sorted response pieces
     sorted certificate pieces
```

- L returns one response piece per inherited parent cell and adds no boundary.
- T traces each tree independently and merges its sorted breakpoint streams.
- N invokes an ExactLine-style partitioner and lifts every numeric piece to a
  provenance-annotated piece.

All subsequent causal composition, persistent circuit versioning, segment
quotienting, storage, and query processing are class independent.

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

The provenance vocabulary is shared by every supported model. Templates use
unit-bindable slots rather than embedding a unit ID, which permits exact
cross-unit structural sharing:

```text
facts          ROOT_SLOT(v), instantiated for a unit
noises         NOISE_SLOT(v), instantiated for a unit
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
|-- Versioned Provenance Closure
|     shared semiring templates + per-unit region slot environments
|
|-- Closure-Compact Layout
|     initial slot roots + one record per effective change
|
|-- VSC-Materialized Layout
|     affected-ancestor path copying + global resolved-root hash-consing
|
|-- Response Segment Directory
|     (model_version, unit, treatment, target)
|       -> sorted numeric regions + closure handles or VSC root IDs
|
|-- Optional Additive-Frontier Optimizer
|     MGPF/LCF + maximal runs + stabbing-semigroup directory
|
|-- Certificate DAG and Directory
|     optional, possibly finer split/activation witness intervals
|
|-- Unit Store
|     parameter/bias/root/noise/anchor bindings h_u
|
|-- Registered FLIP Event Index
|     (model_version, target, registered_threshold)
|       -> continuous, jump, and plateau threshold-state events
|
`-- Declared Aggregate Directory
      preprocessed piecewise-affine SUM/AVG maps
```

There is no separate linear index, tree index, and neural index. Every
mechanism supplies local boundaries and slot changes. Causal composition,
version construction, segment coalescing, storage, and query processing are
shared. Staircase and laminarity may trigger specialized optimizers, but are
not required by the general index.

### Construction

The local compiler first proceeds independently for each unit in causal
topological order:

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
    for each parent cell:
      response_pieces, certificate_pieces <-
          COMPILE_CELL(local_mechanism[v], parent lines, parent roots, cell)
    normalized <- intern response templates and certificates in separate DAGs
    response[v] <- coalesce adjacent response-provenance-equivalent segments
    certificates[v] <- retain the possibly finer audit partition

  return response
```

The compiler never enumerates all global model regions. It visits only regions
reached by the one-dimensional intervention trajectory and retains only
maximal intervals with distinct provenance.

After each unit is compiled, GPSI orders its exact regions and emits the
effective mutable-slot delta at every boundary. Compact mode appends each delta
to the corresponding slot history and lets the region directory identify the
environment version. Materialized mode additionally instantiates the first
circuit and path-copies only the affected ancestor closure for each later
version. All units use the same immutable arena, so identical subcircuits and
recurring complete roots share automatically.

### Closure Persistence, Circuit Sharing, and Optional Factorization

Numeric parameters are represented by slots and bound by `h_u`. Canonical
operator nodes preserve both semiring addition and multiplication, including
non-idempotent multiplicity. The general sharing unit is therefore a complete
subcircuit rather than an expanded path or additive fragment.

```text
region j:     root_j -> ADD(MUL(...), ...)
region j+1:   root_k -> ADD(shared_subcircuit, changed_subcircuit)
```

Linear path provenance may still use the existing reverse trie because it is a
compact specialization of this arena. Tree ensembles may use a balanced
semiring sum template. ReLU networks retain activation selectors inside mixed
`ADD/MUL` templates, avoiding mandatory expansion into exponentially many
neural paths.

Together the layouts give four forms of exact sharing without support nesting:

- Compact Closure stores every effective region-state delta once, regardless
  of its downstream circuit fan-out;
- unchanged template subcircuits persist between adjacent regions;
- recurring non-adjacent activation/leaf patterns reuse old roots;
- structurally identical nodes are hash-consed across units while bindings stay
  independent.

At a declared additive frontier, population MGPF/LCF may further factor
frontier roots and store their maximal guard runs in a stabbing-semigroup
directory. This optional layer never replaces the full circuit semantics.

## Baselines

Let:

```text
n             number of units
C_Q           cost of replaying the causal slice for one unit and query
B_u           number of maximal response regions for unit u and one target
B_tot         total retained response regions, sum_u B_u
B_c           number of certificate regions
H             total per-unit binding records, sum_u |h_u|
L[u,j]        size of region (u,j)'s independently materialized circuit
D_ind         independent-circuit occurrences, sum_(u,j) L[u,j]
D_T           distinct templates and immutable atom roots
C_tot         total initial mutable-slot records, sum_u C_u
K_delta       total effective slot changes, sum_(u,j) k[u,j]
alpha[u,j]    changed slots plus recomputed ancestors at transition (u,j)
gamma[u,j]    newly interned arena nodes; gamma[u,j] <= alpha[u,j]
A             total affected work, sum_(u,j) alpha[u,j]
G             total newly interned nodes, sum_(u,j) gamma[u,j]
D_c           nodes in the optional certificate DAG
F,K_f,D_f     optional additive-factor roots, guard runs, and factor-DAG nodes
E_q           events for one registered threshold q
k             output size of a reporting query
```

| Baseline | What it stores | Point-query work | Space | Missing capability |
|---|---|---:|---:|---|
| B0: DoWhy-style replay | nothing reusable | `O(n*C_Q)` for all units; may repeat abduction | `O(1)` beyond model/data | no indexed provenance or structural FLIP query |
| B1: cached-noise replay | abducted noises | `O(n*C_Q)` without abduction | `O(n*|V|)` | still traverses every active mechanism per query |
| B2: numeric per-unit segment index | boundaries plus numeric slope/intercept | `O(log B_u)` | `Theta(B_tot)` | matches point lookup but cannot return derivation provenance |
| B3: independent provenance per interval (IIP) | one complete exact circuit and root for every `(unit,region)` | `O(log B_u)`; expansion adds output | `Theta(B_tot+D_ind+H)` | duplicates unchanged subcircuits across regions and units |
| B4: global-region materialization | one record per global leaf/activation combination | lookup after potentially exponential construction | proportional to materialized global regions | enumerates unreachable combinations |
| B5: tree-ensemble ADD/decision diagram | tree predicates and numeric terminals | diagram traversal | diagram-dependent | no abducted-noise, causal-path, or unit segment semantics |
| B6: ExactLine | exact numeric affine pieces for one PWL neural input line | `O(log B_u)` after construction | `Theta(B_tot)` numeric pieces | no SCM abduction, typed provenance, cross-unit sharing, or derived indexes |
| **TESSERA Closure-Compact** | shared templates, region directory, initial slot roots, effective change histories, and unit bindings | **closure handle `O(log B_u)`; expansion resolves historical slots** | **`Theta(D_T+C_tot+B_tot+K_delta+H)`** | storage endpoint; deliberately supports only L/T/N |
| **TESSERA VSC-Materialized** | Closure semantics resolved to one canonical root per region in a global arena | **root `O(log B_u)`; expansion adds output** | **`O(D_T+C_tot+B_tot+G+H)`** | latency endpoint; may use more space than Closure |
| TESSERA + AF | either layout plus optional additive-frontier factors and maximal guard runs | primary lookup plus factor reporting | primary layout plus `O(F+K_f+D_f)` | applies only at declared additive frontiers |
| TESSERA Audit | primary layout plus exact guard witnesses | primary lookup plus audit lookup | primary layout plus `O(B_c+D_c)` | optional extra audit storage |

B1 and B2 are the strongest replay and numeric-lookup baselines. B3 is the
primary provenance-storage baseline. Closure has a matching worst-case storage
bound in the arbitrary word-valued delta model, independent of circuit fan-out.
Both B3 and materialized VSC store an indexed resolved root per region, so VSC
does **not** claim a faster asymptotic root lookup than B3. Its advantage over
B3 is change-sensitive construction/storage when `A << D_ind` or
`G << D_ind`. Against prefix event replay, Closure gives direct historical-slot
access and VSC gives a direct resolved root; that is a separate query comparison.

B4 exposes the benefit of trajectory-sensitive compilation. B5 is the required
Class-T compilation baseline. B6 is the required Class-N line-partition
baseline and isolates the cost and benefit of provenance lifting.

Per-class comparisons are fixed:

- L: DoWhy/NumPy replay versus the current compiled linear provenance plan;
- T: cached-noise ensemble replay, per-tree line tracing, and ADD compilation;
- N: direct neural replay and unmodified numeric ExactLine versus
  provenance-lifted ExactLine inside GPSI.

## Related-Work Boundary

The novelty cannot be semiring provenance, hash-consing, path copying, dynamic
expression trees, or decision-diagram reduction by itself. Those ideas already
have established formulations in
[provenance semirings](https://repository.upenn.edu/entities/publication/f1141264-46ee-4d61-b5ea-4ee75fb8d1be),
[factorized databases](https://ora.ox.ac.uk/objects/uuid%3A9116c474-1e8a-4d63-943e-e0d038454f25),
[persistent data structures](https://www.sciencedirect.com/science/article/pii/0022000089900342),
[dynamic expression trees](https://cs.brown.edu/research/pubs/techreports/reports/CS-90-35.html),
and [tree-ensemble compilation to algebraic decision diagrams](https://eldorado.tu-dortmund.de/server/api/core/bitstreams/41b47e64-79f2-47fa-baf2-91e0dcf8df6c/content).
Likewise, [ExactLine](https://papers.nips.cc/paper_files/paper/2019/hash/908075ea2c025c335f4865f7db427062-Abstract.html)
already computes exact numeric affine partitions of piecewise-linear neural
networks along one input line. TESSERA treats it as the Class-N local compiler
and baseline, not as the provenance or index contribution.

The optional support crossing graph is also not claimed as new. Work on
[nested and 2-nested matrices](https://arxiv.org/abs/1906.11970) uses the same
neither-disjoint-nor-nested conflict relation, under an additional consecutive-
ones requirement. Existing
[co-chordal biclique-partition algorithms](https://doi.org/10.1016/j.dam.2023.05.001)
also establish exact graph subclasses. MGPF/LCF therefore remains a scoped
secondary optimizer until its weighted objective and counterfactual-realizable
hardness are complete.

TESSERA's claimed contribution is the counterfactual-query-specific problem
definition and algorithmic composition, rather than any component in
isolation:

1. abduction produces versioned unit-level noise atoms;
2. causal composition produces a provenance-valued parametric response, not
   merely a predictor decision diagram;
3. exact L/T/N compilers turn the scalar intervention line into a canonical
   ordered sequence of response regions and circuit-slot deltas;
4. Compact Closure stores each effective delta once and reaches the matching
   `Theta(D_T+C_tot+B_tot+K_delta+H)` worst-case storage bound independently of
   downstream circuit fan-out;
5. Materialized VSC applies those deltas to mixed `ADD/MUL` templates,
   preserves unchanged subcircuits, and hash-conses recurring roots across
   non-adjacent regions and units;
6. the two-layout analysis proves a strict `Theta(n^2)` versus `O(n log n)`
   versus `Theta(n)` separation from IIP under sparse arbitrary toggles;
7. a high-fan-out family proves `Theta(BF)` canonical materialization versus
   `Theta(B+F)` Compact Closure, together with the corresponding expansion
   lower bound;
8. optional MGPF/LCF and stabbing-semigroup layouts optimize only selected
   additive frontiers and never define the general logical semantics;
9. registered threshold-state changes, including tree jumps, and declared
   aggregates become indexed database queries.

## Theoretical Advantages

The active definitions are in `TECHNICAL_DESIGN.md`; proof drafts for VSC and
the scoped MGPF/LCF optimizer are in `THEORY_PROOFS.md`.
The revised paper theory has four layers rather than a flat list of storage
bounds.

### Layer 1: Exact Counterfactual Semantics

For an acyclic additive-noise SCM whose nodes use L/T/N, fixing a unit and one
bounded scalar intervention yields a finite, possibly discontinuous,
piecewise-affine guarded provenance response at every relevant node. GPSI
construction returns exactly that response. For a tree ensemble, component
partitions of sizes `b_m` induce at most

```text
1 + sum_m (b_m - 1)
```

line intervals rather than a Cartesian product. Formal provenance equality is
proved first in a free semiring and is then preserved by every per-unit binding
`h_u` and value/lineage homomorphism.

### Layer 2: Two-Layout Optimality and Separation

Compact Closure stores one initial root per mutable slot and one record per
effective slot change:

\[
\boxed{
S_{closure}=T_{closure-build}
=\Theta(D_T+C_{tot}+B_{tot}+K_{delta}+H).
}
\]

This matches a word-RAM worst-case lower bound when every boundary, initial
root, binding, and changed slot root may carry an independent word. It is
independent of a changed guard's downstream fan-out. Region lookup returns an
exact `(template,environment version)` closure in `O(log B_u)`.

Materialized VSC instead resolves each closure. Let `alpha[u,j]` include the
changed slots and affected ancestors, and `gamma[u,j]` be the newly interned
canonical nodes. Then:

\[
T_{VSC}=O(D_T+C_{tot}+B_{tot}+A+H),
\qquad
S_{VSC}=O(D_T+C_{tot}+B_{tot}+G+H).
\]

Within the canonical immutable resolved-circuit model, global hash-consing
stores exactly one record per distinct normalized `(operator,child IDs)` tuple
and is therefore node-instance-optimal. For a balanced `C`-leaf block, the
worst-case batch path union is tight:

\[
\Theta\!\left(k\log\left(1+\frac{C}{k}\right)\right).
\]

Consequently, for `B=C=n` and `O(1)` arbitrary slot changes per boundary:

\[
\boxed{
S_{IIP}=\Theta(n^2),\qquad
S_{VSC}=O(n\log n),\qquad
S_{closure}=\Theta(n).
}
\]

No staircase assumption is used. Closure pays template/slot-resolution work
when expanded; materialized VSC pays additional storage to return an active
resolved root immediately.

The Pareto gap can itself be asymptotic. For one slot `s` feeding `F` distinct
products `s*a_i`, let `B` regions bind `s` to pairwise distinct atoms. Compact
Closure stores `Theta(B+F)` records, while the canonical materialized model
must store `BF` distinct `(MUL,x_j,a_i)` tuples:

\[
S_{closure}=\Theta(B+F),
\qquad
S_{VSC}=\Theta(BF).
\]

For `B=F=n`, this is linear versus quadratic. Conversely, producing an
unmaterialized canonical structural-EXPLAIN circuit requires `Omega(F)` work
because its `F` distinct product tuples are part of the requested output. This
is the formal space-latency boundary between the two layouts under the frozen
canonical structural-EXPLAIN contract. It is not a lower bound against
distributive algebraic rewriting; the constructed polynomial can be factored,
which must be costed as a separate physical layout.

### Layer 3: Optional Algebraic Factorization

For a declared additive frontier, occurrence lifting yields a
`(unit,region)`-by-fragment support matrix. Its exact guarded rectangle
factorization is MGPF. In that restricted representation, the existing draft
establishes an NP-completeness boundary, the rank lower bound
`F* >= rank_R(M)`, laminar optimality, and a produced-cover guarantee
`F_LCF <= ell*F*`.

These results apply only after choosing that additive frontier. They do not
justify expanding a general ReLU circuit into additive paths, and they do not
replace VSC's mixed `ADD/MUL` semantics. Before submission, factor-root count,
payload references, and guard runs must be unified into one weighted storage
objective and the hardness reduction must be realized by a restricted
counterfactual mechanism family.

### Layer 4: Physical Index and Query Bounds

The primary response directory stores sorted starts and either a closure
version or a materialized VSC root per region. It gives:

```text
POINT                    O(log B_u)
Closure handle           O(log B_u)
VSC resolved root        O(log B_u)
VSC expanded EXPLAIN     O(log B_u + output_size)
Closure expansion        template traversal + historical slots + output work
registered FLIP          O(log E_q + k)
declared AGG             O(log B_group)
```

The optional additive-frontier directory stores maximal guard runs in linear
space in its own `K`. Any cache or factor DAG is reported in addition to the
primary-layout bound, never hidden inside it. Materialized VSC matches the
point/root lookup complexity of a numeric polyline and IIP; its primary benefit
there is provenance retention and shared storage, not a stronger lookup
asymptotic.

## Paper Claims and Non-Claims

### Claims

1. Counterfactual replay over L/T/N and their mixed SCM compositions has an
   exact guarded semiring-provenance semantics.
2. A scalar intervention induces a finite provenance-valued piecewise-affine
   response, and exact compilers enumerate only regions reached along that
   intervention line.
3. Versioned Provenance Closures represent every compiled region sequence
   exactly under arbitrary slot toggles while preserving mixed `ADD/MUL`.
4. Closure-Compact has matching worst-case build/storage
   `Theta(D_T+C_tot+B_tot+K_delta+H)` in its stated word-valued update model,
   independent of downstream fan-out.
5. VSC-Materialized has change-sensitive construction/storage, is
   node-instance-optimal in the canonical immutable model, and has a tight
   balanced batch-update bound.
6. For `B=C=n` and constant-size arbitrary deltas, IIP, VSC, and Closure have
   `Theta(n^2)`, `O(n log n)`, and `Theta(n)` storage, respectively.
7. On the high-fan-out family, canonical materialization and Compact Closure
   require `Theta(BF)` and `Theta(B+F)` storage respectively; on-demand
   canonical expansion requires `Omega(F)` output work.
8. Global hash-consing reuses identical subcircuits and recurring roots across
   adjacent regions, non-adjacent regions, and units, while per-unit bindings
   remain independent.
9. The reduced response directory is canonical and minimum within the exact
   one-dimensional interval partition induced by normalized provenance
   equality.
10. POINT and EXPLAIN use primary indexes; registered-threshold FLIP and
   declared-group AGG use workload-aware secondary indexes rather than fresh
   SCM replays.
11. At explicitly declared additive frontiers, MGPF/LCF supplies an additional
   scoped factorization layer with separately reported storage and query cost.

### Non-Claims

1. GPSI does not exactly support every sklearn-compatible DoWhy predictor.
2. Closure's matching lower bound is a worst-case word bound for arbitrary
   word-valued slot roots, not a bit-optimal encoding theorem for binary flips.
3. Canonical VSC is not claimed to be the smallest algebraically equivalent
   circuit; unrestricted rewriting may identify more equivalences.
4. VSC does not guarantee compression when every transition changes a large
   cone and produces unique nodes; `A,G` can match independent materialization.
5. VSC does not beat the asymptotic root-lookup time of an indexed numeric
   polyline or IIP. Query-time speedup over event replay is a separate result.
6. Closure's `O(log B_u)` result returns an exact closure handle, not a fully
   expanded active circuit; expansion pays template and slot-resolution work.
7. The optional `ell` approximation applies to additive factor-root count, not
   arbitrary `ADD/MUL` circuits or serialized bytes; general ReLU need not have
   small `ell`.
8. Reset min-cut optimality applies only to its declared hierarchical reset
   family, not all equivalent circuits or serialized layouts.
9. GPSI does not avoid worst-case exponential piecewise complexity.
10. GPSI does not make multi-dimensional exact point location cheap.
11. GPSI does not provide the indexed FLIP/AGG bounds for arbitrary online
   thresholds or groups.
12. The current repository implements the linear foundation, Compact Closure,
    and Materialized VSC substrates; fitted Class-T and Class-N compilers remain
    to be built and evaluated.

## Required Experimental Story

The evaluation must separate five effects:

1. **Semantics:** exact agreement with DoWhy-style abduction-action-prediction.
2. **Compilation:** realized line regions versus potential global
   leaf/activation combinations, compiler time, and exact boundary agreement.
3. **Persistence:** `B_tot,D_ind,D_T,C_tot,K_delta,A,G,H`, Closure records,
   unique VSC roots, recurring roots, affected nodes, and serialized bytes
   versus IIP.
4. **Querying:** Closure handle lookup, historical-slot resolution, template
   traversal, VSC root lookup, and event replay measured separately. Results
   must not label VSC lookup as a speedup over IIP when both store roots.
5. **Optional factorization:** at declared additive frontiers only, report
   `rank(M),ell,F,F*,K,D_f` and bytes separately; small instances use the exact
   ILP oracle.

Required mechanism classes:

- L: linear SCMs as the one-segment control;
- T: single regression trees (`M=1`), random forests, and GBDT;
- N: multi-layer ReLU MLPs accepted by the ExactLine-style compiler;
- mixed SCMs whose nodes independently choose L, T, or N.

Required plots:

- realized segment count versus model size and causal depth;
- `K_delta/D_ind`, `A/D_ind`, `G/D_ind`, unique-root ratio, index bytes, and
  build time versus changed slots, fan-out, repeated patterns, and population;
- IIP, Closure, and VSC scaling on the theorem's `B=C=n,k=O(1)` family;
- canonical Materialized VSC versus Compact Closure on the theorem's
  `B=F=n` high-fan-out family, including separate on-demand expansion cost;
- Closure/VSC storage and construction for random non-staircase updates, fitted
  tree ensembles, fitted ReLU MLPs, and mixed SCMs;
- Closure handle, VSC root, IIP root, and prefix-event replay lookup, followed
  by separate expanded EXPLAIN costs;
- query latency versus number of units and number of repeated intervention
  values;
- registered-threshold FLIP latency versus output selectivity, including
  continuous and tree-jump events;
- retraining invalidation/rebuild cost, without claiming that only templates
  change.

Optional-factorization plots report support crossing, rank/ILP gaps, guard
runs, factor-DAG nodes, and bytes in a separate ablation. They cannot substitute
for the primary two-layout-versus-IIP result.

## Recommended Paper Title

**TESSERA: An Exact Provenance Index for Parametric Counterfactual Queries over
Piecewise-Affine Structural Causal Models**
