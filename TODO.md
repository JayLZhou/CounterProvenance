# TODO

The implementation order follows the unified Guarded Provenance Segment Index
(GPSI) in `POSITIONING.md`. We do not build one unrelated index per mechanism
family.

## M0: Linear Foundation

- [x] Validate linear additive-noise counterfactual replay against semiring provenance.
- [x] Compare the provenance-derived evaluator against `dowhy.gcm.counterfactual_samples`.
- [x] Separate raw trie traversal from the compiled fast linear query plan.
- [x] Support arbitrary cDAGs, non-intervened roots, and multiple interventions.
- [x] Add a layered random-DAG generator and linear correctness tests.
- [ ] Add CSV/JSON benchmark output for reproducible plots.

## M1: Generic Segment Algebra

- [ ] Add an exact `PiecewiseAffineResponse` type over a bounded scalar intervention domain.
- [ ] Implement sorted partition refinement, affine substitution, addition, and scalar multiplication.
- [ ] Implement exact boundary ownership rules for open/closed threshold endpoints.
- [ ] Normalize provenance templates and coalesce adjacent provenance-equivalent segments.
- [ ] Add property tests against dense replay away from boundaries and explicit tests at boundaries.
- [ ] Verify the one-region affine engine as a degenerate one-segment instance.

Acceptance criteria:

- every operation preserves a disjoint, exhaustive, sorted interval partition;
- coalescing is idempotent and produces maximal equal-template intervals;
- segment evaluation agrees with direct replay to floating-point tolerance.

## M2: Tree and Ensemble Compilers

- [ ] Add a regression-tree local mechanism representation with numeric threshold guards.
- [ ] Trace an affine parent segment through a tree without enumerating unreachable leaves.
- [ ] Compile random forests, Extra Trees, GBDT, and histogram GBDT as additive components.
- [ ] Merge per-tree breakpoint streams without materializing leaf combinations.
- [ ] Preserve leaf values, split predicates, tree IDs, and causal paths as provenance tokens.
- [ ] Add mixed linear/tree SCM correctness tests.

Acceptance criteria:

- exact agreement with sklearn/DoWhy-style replay for every tested intervention value;
- observed ensemble segment count satisfies the additive refinement bound;
- construction visits only leaves reached by the intervention trajectory.

## M3: Shared Prefix-Factorized Provenance DAG

- [ ] Replace per-segment full expressions with canonical template IDs.
- [ ] Define one canonical token order for every provenance derivation path.
- [ ] Insert target-oriented reversed causal paths into a shared prefix trie.
- [ ] Hash-cons `ATOM`, `ADD`, `MUL`, and reference nodes.
- [ ] Store optional raw split/activation witnesses in a separate guard-certificate DAG.
- [ ] Share causal propagation fragments across targets and units.
- [ ] Generalize the implemented linear reverse trie to guarded-affine templates.
- [ ] Dictionary-encode repeated template-ID sequences while retaining unit-specific breakpoints.
- [ ] Add a vectorized template evaluator without Python recursion.

Acceptance criteria:

- storage reports expanded token slots, distinct trie prefixes, `S`, unique templates, DAG nodes, and bytes;
- the DAG and unfactorized expressions evaluate identically;
- no claim of global arithmetic-circuit minimality is made.

## M4: Indexed Queries

- [ ] Implement `POINT(unit, target, t)` with binary search over maximal segments.
- [ ] Materialize exact threshold-crossing events for `FLIP` queries.
- [ ] Implement `FLIP(target, threshold, t_range)` in output-sensitive time.
- [ ] Add optional preaggregated piecewise-affine maps for declared SUM/AVG groups.
- [ ] Implement `EXPLAIN` over the active provenance-template DAG.
- [ ] Add batched support for many targets in one workload.

Acceptance criteria:

- point lookup uses `O(log B_u)` segment search;
- flip reporting uses `O(log E + k)` event search plus output;
- every returned answer can emit its active provenance tokens and validity interval.

## M5: ExactLine/PWL Neural Compiler

- [ ] Integrate or reproduce the ExactLine local line-partition contract.
- [ ] Lift every ExactLine numeric affine piece to typed counterfactual provenance.
- [ ] Add ReLU MLP mechanism representations accepted by the local compiler.
- [ ] Preserve activation certificates without making them part of template equality.
- [ ] Add mixed linear/tree/ReLU SCM tests.
- [ ] Measure realized segments and unique templates versus network width, depth, and population size.

Acceptance criteria:

- exact agreement with direct MLP replay on and between activation boundaries;
- construction is output-sensitive in realized crossings rather than all activation patterns;
- worst-case exponential segment growth is measured and reported, not hidden.

## M6: Versioning and Maintenance

- [ ] Version model parameters, abducted noises, templates, and segment directories together.
- [ ] Add invalidation checks for changed local mechanisms.
- [ ] Rebuild only causal descendants and affected workload targets when sound.
- [ ] Measure full rebuild and partial invalidation costs after retraining.
- [ ] Do not claim factual anchors alone make per-unit indexes invariant to retraining.

## M7: Baselines and Evaluation

- [ ] B0: DoWhy-style replay from observed data.
- [ ] B1: cached-noise vectorized replay.
- [ ] B2: global-region/leaf-combination materialization on tractable small cases.
- [ ] B3: exact numeric per-unit polyline with no derivation provenance.
- [ ] B4: exact provenance polyline with a full expression per segment.
- [ ] B5: tree-ensemble ADD/decision-diagram compilation.
- [ ] B6: unmodified numeric ExactLine for Class N.
- [ ] Report build time, amortization point, bytes, segment count, DAG size, and query latency.
- [ ] Evaluate `POINT`, `FLIP`, `AGG`, and `EXPLAIN` separately.

## M8: Theorems and Paper

- [ ] Prove exact compositional closure for L/T/N and mixed L/T/N SCMs.
- [ ] Prove the Class-T additive non-Cartesian interval bound.
- [ ] Prove output-sensitive construction in visited guards, crossings, and retained segments.
- [ ] Prove the exact prefix-factorization bound and fixed-order trie minimality.
- [ ] Prove canonical minimum segment quotient within the exact 1D interval-index class.
- [ ] Prove the factorized storage bound and matching `Omega(S)` worst-case lower bound.
- [ ] Prove point and flip query bounds.
- [ ] State scalar-intervention, additive-noise, and exact-guard assumptions prominently.
- [ ] Write the paper around one abstraction: guarded provenance response functions indexed by GPSI.
