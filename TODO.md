# TODO

## Prototype

- [x] Validate linear additive-noise counterfactual replay against a semiring provenance trie.
- [x] Compare the trie-derived evaluator against real `dowhy.gcm.counterfactual_samples`.
- [x] Separate raw trie traversal from the compiled fast linear query plan.
- [x] Add a reusable generic linear-cDAG provenance engine.
- [x] Add arbitrary cDAG topological expansion and cycle validation.
- [x] Add support for non-intervened root variables and multiple intervention variables.
- [x] Add a layered random-DAG generator to stress-test trie compression at larger path counts.
- [x] Add unit tests for monomial expansion, trie insertion, and compiled evaluation.
- [ ] Add CSV/JSON benchmark output for reproducible plots.

## System Design

- [x] Replace the demo-only trie with a reusable provenance-store API for linear cDAGs.
- [ ] Add batched support for many targets in one workload.
- [ ] Add incremental update hooks for changed coefficients / retrained mechanisms.
- [ ] Add a vectorized execution backend that evaluates trie/circuit fragments without Python recursion.

## Research Extensions

- [ ] Extend exact guarded provenance to ReLU / piecewise-affine mechanisms.
- [ ] Extend exact guarded provenance to tree and GBDT mechanisms.
- [ ] Add interval provenance for smooth neural mechanisms.
- [ ] Define the formal guard-liftable mechanism class.
- [ ] Write the first paper skeleton around factorized semiring provenance for counterfactual query processing over learned SCMs.
