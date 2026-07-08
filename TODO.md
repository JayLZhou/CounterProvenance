# TODO

## Prototype

- [x] Validate linear additive-noise counterfactual replay against a semiring provenance trie.
- [x] Compare the trie-derived evaluator against real `dowhy.gcm.counterfactual_samples`.
- [x] Separate raw trie traversal from the compiled fast linear query plan.
- [ ] Add a layered random-DAG generator to stress-test trie compression at larger path counts.
- [ ] Add CSV/JSON benchmark output for reproducible plots.
- [ ] Add unit tests for monomial expansion, trie insertion, and compiled evaluation.

## System Design

- [ ] Replace the demo-only trie with a reusable provenance-store API.
- [ ] Add support for multiple intervention variables and multiple targets.
- [ ] Add incremental update hooks for changed coefficients / retrained mechanisms.
- [ ] Add a vectorized execution backend that evaluates trie/circuit fragments without Python recursion.

## Research Extensions

- [ ] Extend exact guarded provenance to ReLU / piecewise-affine mechanisms.
- [ ] Extend exact guarded provenance to tree and GBDT mechanisms.
- [ ] Add interval provenance for smooth neural mechanisms.
- [ ] Define the formal guard-liftable mechanism class.
- [ ] Write the first paper skeleton around factorized semiring provenance for counterfactual query processing over learned SCMs.
