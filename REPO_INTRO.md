# Repository Introduction

## Short GitHub Description

Fast repeated counterfactual queries over learned linear SCMs on arbitrary cDAGs using semiring provenance, trie compression, and compiled evaluation.

## One-Paragraph Intro

TESSERA Linear Trie is a prototype for compiling linear-SCM counterfactual replay over arbitrary causal DAGs into reusable semiring provenance. Instead of recomputing abduction-action-prediction for every query, the prototype expands counterfactual derivations into provenance monomials, stores shared propagation fragments in a reverse trie, and derives a fast closed-form evaluator for repeated `do(T=t')` workloads. It validates exactly against DoWhy-GCM's `counterfactual_samples` on fitted linear `AdditiveNoiseModel` mechanisms.

## Longer Project Summary

Modern causal toolchains such as DoWhy-GCM can compute counterfactual samples over fitted structural causal models, but they typically treat each query as a fresh replay: recover residuals, apply an intervention, and propagate through the SCM. This is correct, but inefficient for repeated counterfactual workloads over many units and many intervention values.

TESSERA Linear Trie explores a database-style alternative. For linear additive-noise SCMs on arbitrary cDAGs, the counterfactual response can be expanded into semiring provenance: intervention tokens, root tokens, residual tokens, bias tokens, and parameter tokens connected by addition and multiplication. These expanded monomials share many propagation fragments, so the prototype stores them in a compressed reverse trie and compiles the trie into a vectorized evaluator. The result is numerically equivalent to DoWhy-style replay, while avoiding repeated mechanism traversal at query time.

This repository is a first step toward the broader TESSERA research direction: factorized semiring provenance for counterfactual query processing over learned SCMs.

## Key Claims Demonstrated

- Linear additive-noise counterfactual replay can be represented as semiring provenance.
- The representation supports arbitrary linear causal DAGs, non-intervened roots, and multiple interventions.
- Expanded propagation monomials can be compressed with a reverse trie.
- The compressed provenance can be compiled into a closed-form vectorized evaluator.
- The evaluator matches DoWhy-GCM counterfactual samples up to floating-point error.
- On the toy benchmark, compiled evaluation is substantially faster than repeated DoWhy-GCM calls and faster than a lightweight NumPy replay baseline.

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
```

## Suggested Repo Name Options

```text
tessera-linear-trie
tessera-counterfactual-provenance
linear-scm-provenance-trie
counterfactual-provenance-trie
```
