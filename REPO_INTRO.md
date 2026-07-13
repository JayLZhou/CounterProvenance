# Repository Introduction

## Short GitHub Description

Exact provenance indexing for repeated parametric counterfactual queries over
linear, tree-based, and piecewise-affine structural causal models.

## One-Paragraph Intro

TESSERA studies how to compile repeated counterfactual workloads over learned
additive-noise SCMs into a Versioned Semiring Circuit index. For an indexed
population, scalar treatment, and target, the exact intervention line becomes
an ordered sequence of provenance-circuit versions. A global immutable arena
stores each `ADD`/`MUL` subcircuit once, each region stores a root pointer, and
per-unit bindings retain independent parameters, roots, and abducted noises.
The checked-in prototype implements the linear foundation and a general VSC
core with arbitrary `ON/OFF/ON` selector changes, cross-unit hash-consing, and
indexed region lookup.

## Paper Positioning

**TESSERA: An Exact Provenance Index for Parametric Counterfactual Queries over
Piecewise-Affine Structural Causal Models**

The submitted system claims exactly three mechanism classes: L, affine
regressors with arbitrary per-unit parameter bindings; T, additive numerical regression-tree ensembles, with a single tree
as `M=1`; and N, ReLU-style piecewise-linear neural mechanisms handled by an
ExactLine-style local compiler. Mixed SCMs may use L, T, and N at different
causal nodes. Other model families are explicitly outside the paper claim.

## Unified Index

The Guarded Provenance Segment Index contains:

- a local exact L/T/N mechanism compiler and response-segment directory;
- a global immutable arena of `ZERO/ONE/ATOM/ADD/MUL/SWITCH` nodes;
- persistent region versions built by batch slot updates and affected-ancestor
  path copying;
- one root pointer per `(unit,region)`, globally hash-consed across versions and
  units;
- optional LCF/min-cut and stabbing-semigroup optimization at additive circuit
  frontiers;
- separate response and optional finer certificate interval directories;
- versioned parameter/factual/noise/anchor slot bindings `h_u`;
- registered-threshold FLIP events and declared-group aggregate responses.

There is no independent tree index or neural index. L/T/N emit different local
boundary and selector events into one VSC substrate. Staircase and laminar
patterns are optional optimizable cases, not input assumptions.

## Theoretical Spine

The revised theory has four layers:

1. exact piecewise-affine closure of L/T/N, semiring soundness, and the
   non-Cartesian Class-T breakpoint bound;
2. exact persistent-version semantics and change-sensitive storage
   `O(B_tot + D_0 + sum gamma_u,j + bindings)`;
3. a balanced batch-update bound
   `alpha_u,j = O(k_u,j log(1 + C/k_u,j))`, with
   `gamma_u,j <= alpha_u,j`;
4. secondary algebraic factorization: general hardness, exact structured
   subclasses, and physical interval indexing at additive frontiers.

The design explicitly allows worst-case exponential realized segments and does
not claim arbitrary-circuit minimality, universal compression, or that every
deep-ReLU boundary changes only a small affected cone.

## Current Implementation

The repository currently demonstrates that:

- linear additive-noise counterfactual replay has semiring provenance;
- arbitrary linear cDAGs, non-intervened roots, and multiple interventions are
  supported;
- expanded propagation paths can be compressed with a reverse trie;
- the provenance can be compiled into a vectorized closed-form evaluator;
- the evaluator matches DoWhy-GCM counterfactual samples to floating-point
  error and accelerates repeated queries;
- VSC preserves non-idempotent semiring multiplicity and mixed `ADD`/`MUL`;
- arbitrary non-staircase selector sequences share persistent circuit versions;
- identical non-adjacent and cross-unit roots are hash-consed while numeric
  bindings remain independent;
- the random-toggle benchmark reports independent node occurrences, unique VSC
  nodes, changed slots, affected nodes, and indexed-root query latency.

See `TECHNICAL_DESIGN.md` for the VSC revision and the earlier GPSI/LCF design;
`THEORY_PROOFS.md` contains the VSC construction theorem and scoped MGPF/LCF
proofs. See `TODO.md` for implementation milestones.

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
