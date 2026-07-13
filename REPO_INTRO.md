# Repository Introduction

## Short GitHub Description

Exact provenance indexing for repeated parametric counterfactual queries over
linear, tree-based, and piecewise-affine structural causal models.

## One-Paragraph Intro

TESSERA studies how to compile repeated counterfactual workloads over learned
additive-noise SCMs into a Guarded Provenance Segment Index. For a fixed unit,
scalar treatment, and target, the index represents the complete response
`Pi[u,Y](t) = Y_{u;do(T=t)}` as maximal intervention intervals pointing to a
shared semiring-provenance DAG. Linear mechanisms produce one interval; tree
thresholds and ReLU activations refine the same index through exact guard
crossings. The checked-in prototype implements the linear foundation over
arbitrary cDAGs and validates it against DoWhy-GCM.

## Paper Positioning

**TESSERA: An Exact Provenance Index for Parametric Counterfactual Queries over
Piecewise-Affine Structural Causal Models**

The submitted system claims exactly three mechanism classes: L, global affine
regressors; T, additive numerical regression-tree ensembles, with a single tree
as `M=1`; and N, ReLU-style piecewise-linear neural mechanisms handled by an
ExactLine-style local compiler. Mixed SCMs may use L, T, and N at different
causal nodes. Other model families are explicitly outside the paper claim.

## Unified Index

The Guarded Provenance Segment Index contains:

- a local guarded-affine mechanism DAG;
- a shared reverse-prefix trie with hash-consed semiring-provenance subcircuits;
- a per-unit sorted segment directory of validity intervals and template IDs;
- versioned factual/noise pointers;
- secondary threshold-crossing indexes for FLIP queries and optional aggregate
  response indexes.

There is no independent tree index or neural index. L/T/N only provide three
implementations of one `COMPILE_LINE` contract; causal composition, prefix
factorization, segment reduction, storage, and query execution are shared.

## Theoretical Spine

The paper targets seven formal results:

1. exact piecewise-affine closure of L/T/N under causal composition;
2. a sum-sized interval bound for Class-T ensembles instead of a Cartesian
   leaf-combination construction;
3. output-sensitive construction in realized guard crossings and segments,
   independent of the potential global-region count;
4. exact prefix factorization with one node per distinct canonical reversed
   path prefix;
5. a canonical minimum segment quotient within exact one-dimensional interval
   indexes;
6. `O(S + D + n*|V|)` factorized storage versus `Theta(S*L)` per-unit full
   expressions;
7. `O(log B_u)` point lookup and `O(log E + k)` indexed flip reporting.

The design explicitly allows worst-case exponential realized segments and does
not claim universal compression.

## Current Linear Foundation

The repository currently demonstrates that:

- linear additive-noise counterfactual replay has semiring provenance;
- arbitrary linear cDAGs, non-intervened roots, and multiple interventions are
  supported;
- expanded propagation paths can be compressed with a reverse trie;
- the provenance can be compiled into a vectorized closed-form evaluator;
- the evaluator matches DoWhy-GCM counterfactual samples to floating-point
  error and accelerates repeated queries.

See `POSITIONING.md` for the formal scope, baseline matrix, theorem statements,
and evaluation plan. See `TODO.md` for the implementation milestones.

## Suggested Tags

```text
causal-inference
counterfactuals
provenance
semiring-provenance
structural-causal-models
dowhy
query-processing
database-systems
piecewise-affine
```
