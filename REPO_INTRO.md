# Repository Introduction

## Short GitHub Description

Exact provenance indexing for repeated parametric counterfactual queries over
linear, tree-based, and piecewise-affine structural causal models.

## One-Paragraph Intro

TESSERA studies how to compile repeated counterfactual workloads over learned
additive-noise SCMs into a Versioned Provenance Closure index. For an indexed
population, scalar treatment, and target, the exact intervention line becomes
a shared semiring template plus an ordered sequence of slot environments. The
same logical closure has two exact layouts: Closure-Compact stores only initial
slot roots and effective changes, while VSC-Materialized stores canonical
resolved roots through affected-ancestor path copying and global hash-consing.
Per-unit bindings retain independent parameters, roots, and abducted noises.
The checked-in prototype implements the linear foundation and both general
layouts under arbitrary `ON/OFF/ON` changes.

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
- one shared `ZERO/ONE/ATOM/ADD/MUL` semiring template per structural class;
- a Compact Closure layout with one initial value per slot and one record per
  effective region change;
- a Materialized VSC layout with affected-ancestor path copying and one
  canonical resolved root per `(unit,region)`;
- optional LCF/min-cut and stabbing-semigroup optimization at additive circuit
  frontiers;
- separate response and optional finer certificate interval directories;
- versioned parameter/factual/noise/anchor slot bindings `h_u`;
- registered-threshold FLIP events and declared-group aggregate responses.

There is no independent tree index or neural index. L/T/N emit different local
boundary and selector events into one closure contract. Staircase and laminar
patterns are optional optimizable cases, not input assumptions.

## Theoretical Spine

The revised theory has four layers:

1. exact piecewise-affine closure of L/T/N, semiring soundness, and the
   non-Cartesian Class-T breakpoint bound;
2. matching Compact Closure build/storage
   `Theta(D_T + C_tot + B_tot + K_delta + H)`, independent of downstream
   fan-out;
3. canonical-model instance optimality for Materialized VSC, the tight
   balanced batch bound `Theta(k log(1 + C/k))`, and the strict separation
   `IIP=Theta(n^2)`, `VSC=O(n log n)`, `Closure=Theta(n)`, plus a high-fan-out
   canonical `Materialized=Theta(BF)` versus `Closure=Theta(B+F)` separation;
4. secondary algebraic factorization: general hardness, exact structured
   subclasses, and physical interval indexing at additive frontiers.

The design explicitly allows worst-case exponential realized segments and does
not claim arbitrary algebraic-circuit minimality, bit-optimal compression for
binary flips, or that every deep-ReLU boundary changes only a small affected
cone.

## Current Implementation

The repository currently demonstrates that:

- linear additive-noise counterfactual replay has semiring provenance;
- arbitrary linear cDAGs, non-intervened roots, and multiple interventions are
  supported;
- expanded propagation paths can be compressed with a reverse trie;
- the provenance can be compiled into a vectorized closed-form evaluator;
- the evaluator matches DoWhy-GCM counterfactual samples to floating-point
  error and accelerates repeated queries;
- Compact Closure preserves mixed `ADD`/`MUL` semantics with one record per
  effective slot update, independently of template fan-out;
- VSC preserves non-idempotent semiring multiplicity and mixed `ADD`/`MUL`;
- arbitrary non-staircase selector sequences share persistent circuit versions;
- identical non-adjacent and cross-unit roots are hash-consed while numeric
  bindings remain independent;
- the random-toggle benchmark compares independent records, Compact Closure
  records, unique VSC records, changed slots, affected nodes, and separate
  closure-handle/VSC-root query latency.

See `TECHNICAL_DESIGN.md` for the Closure/VSC design and the earlier GPSI/LCF
layer; `THEORY_PROOFS.md` contains the matching Closure bound, the VSC
construction theorems, and scoped MGPF/LCF proofs. See `TODO.md` for
implementation milestones.

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
