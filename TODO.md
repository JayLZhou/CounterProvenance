# TODO

[`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md) records the complete earlier
GPSI/LCF design; [`THEORY_PROOFS.md`](THEORY_PROOFS.md) contains its independently
checkable proofs. The current implementation pivot makes Versioned Semiring
Circuits (VSC) the general storage layer and retains MGPF/LCF as an optional
algebraic optimizer at additive frontiers. The documents must now be reconciled
with the checked-in VSC semantics before the paper is frozen again.

## D0: Theory and Algorithm Freeze (Current)

- [x] Remove staircase/laminar structure as a requirement of the general index.
- [x] Define the general indexed object as ordered versions of exact semiring
      circuits with one root per `(unit,region)`.
- [x] Preserve both `ADD` and `MUL`; do not require deep neural provenance to be
      expanded into an additive path-fragment matrix.
- [x] Implement immutable hash-consed circuit nodes and affected-ancestor
      path copying for arbitrary slot toggles.
- [x] Replace the old primary storage theorem with the change-sensitive VSC
      bound `O(B_tot + D_0 + sum gamma_u,j + bindings)`.
- [x] Prove the balanced batch-update lemma
      `O(k_j log(1 + C/k_j))` for associative circuit blocks.
- [ ] State and prove the exact representation model in which canonical VSC is
      optimal, and separate it from harder algebraic circuit minimization.
- [ ] Prove hardness on a counterfactual-realizable restricted circuit family;
      arbitrary support-matrix hardness alone is secondary evidence.
- [ ] Reformulate laminar DP/min-cut against one weighted storage objective,
      rather than optimizing factor count and runs in separate claims.

- [x] Fix the submitted scope to scalar interventions and exactly L/T/N.
- [x] Define the indexed object as the complete population response family over
      cells `(unit,region)`.
- [x] Allow arbitrary per-unit parameter/root/noise bindings through a semiring
      homomorphism `h_u`; sharing is structural and never assumed.
- [x] Separate response-provenance intervals from finer audit certificates.
- [x] Define the restricted ZERO/ATOM/PREFIX/ADD provenance language.
- [x] Replace the global-union-trie claim with designated per-template roots
      and cross-template subtree hash-consing.
- [x] Define FLIP as a registered-threshold state change, including continuous
      crossings, tree jumps, and threshold plateaus.
- [x] Define AGG bounds only for groups declared at build time.
- [x] Assign one explicit role to every baseline B0-B6.
- [x] Define occurrence-lifted MGPF and separate factor roots `F`, guard runs
      `K`, and physical DAG size `D`.
- [x] Define run-aware LCF through the support crossing graph, laminar blocks,
      directed reset min-cuts, LOCAL roots, and non-flattened BUNDLE nodes.
- [x] State the revised theorem chain: hardness, rank lower bound, laminar
      optimality, `ell`-approximation, `K<=min(R_S,R_A)<=R`, dyadic separation,
      and query
      bounds.
- [x] Remove the unverified prefix-DAG global-minimality claim; retain only the
      exact prefix-count lemma and non-increasing hash-consing claim.
- [x] Write appendix-level MGPF/LCF proofs, including occurrence lifting,
      hardness, rank tightness, inherited-payload correctness, reset min-cut,
      approximation, intervalization, and the semigroup mapping.
- [x] Write real-arithmetic L/T/N compiler closure, boundary-completeness,
      endpoint-ownership, tree-ensemble composition, response-quotient, and
      designated-trie proofs.
- [ ] Validate each supported fitted predictor's floating-point branch behavior
      against the formal endpoint policy.
- [x] Add a checked-in exhaustive small-matrix verifier for laminar optimality,
      `F<=ell*F*`, and `K<=min(R_S,R_A)<=R`.
- [x] Extend the verifier to population chains: unit boundaries cannot merge
      runs, and exact factorization survives arbitrary per-unit `h_u` bindings.
- [ ] Complete a dedicated novelty search for laminar biclique partitions and
      laminar-cover approximations.
- [ ] Complete a systematic related-work search before retaining the scoped
      first-system claim.
- [ ] Freeze notation shared by the design document and future paper.
- [ ] Review and accept the complete technical design with the project owner.

Acceptance criteria:

- every query has exact input/output and boundary semantics;
- every asymptotic claim names all stored objects and workload assumptions;
- every theorem has a precise representation class and non-claim;
- no baseline is weaker only because it repeats avoidable abduction;
- implementation can proceed without inventing a new semantic decision.

## M0: Existing Linear Foundation

- [x] Validate linear additive-noise replay against semiring provenance.
- [x] Compare against `dowhy.gcm.counterfactual_samples`.
- [x] Separate raw trie traversal from compiled linear evaluation.
- [x] Support arbitrary cDAGs, fixed roots, and multiple interventions in the
      existing prototype.
- [x] Add random-DAG generation and correctness tests.
- [ ] Preserve the existing implementation unchanged until D0 is accepted.

## M1: Guarded Piecewise-Affine Algebra

- [ ] Add exact interval and endpoint-ownership types for a bounded scalar
      intervention domain.
- [ ] Add independent `ResponseSegment` and `CertificateSegment` types.
- [ ] Implement sorted common refinement, restriction, affine substitution,
      addition, and scalar multiplication.
- [ ] Implement response-template normalization and maximal adjacent-cell
      coalescing.
- [ ] Add direct evaluation and predecessor lookup.
- [ ] Add rational small-case tests and float boundary-validation tests.

Acceptance criteria:

- partitions are sorted, disjoint, and exhaustive;
- tree `<=`/`>` and ReLU-zero ownership are deterministic;
- coalescing is idempotent and never merges distinct response provenance;
- dense replay, exact boundaries, and one-sided checks agree.

## M2: Prefix-Factorized Provenance Core

- [ ] Implement ZERO, ATOM, PREFIX, and canonical ADD nodes.
- [ ] Use parameter, bias, root, noise, and anchor slots with per-unit `h_u` bindings.
- [ ] Give every unique template a designated reverse-trie root.
- [ ] Hash-cons identical labeled subtries across templates and units.
- [ ] Preserve terminal labels and weighted path multiplicities.
- [ ] Add value and lineage valuations.
- [ ] Store DAG nodes and template roots in columnar arrays.
- [ ] Report expanded path tokens, per-template trie nodes, membership
      references, hash-consed DAG nodes, and bytes.

Acceptance criteria:

- DAG evaluation equals logically expanded provenance;
- each template reconstructs only its own weighted path language;
- no uncounted global-union-trie membership structure exists;
- the implementation uses no unrestricted MUL node.

This milestone now describes the specialized Linear/path-prefix layer. General
Tree/ReLU storage is handled by M2.1 and may contain canonical `MUL` nodes.

## M2.1: Versioned Semiring Circuit Core

- [x] Implement immutable `ZERO`, `ONE`, `ATOM`, `ADD`, `MUL`, and `SWITCH`
      nodes with global hash-consing.
- [x] Preserve non-idempotent addition multiplicity.
- [x] Implement fixed circuit templates with mutable selector/fragment slots.
- [x] Implement sequential region versions by batch slot updates and
      affected-ancestor path copying.
- [x] Implement left-closed region directories and indexed root lookup.
- [x] Share identical roots across non-adjacent regions and units while keeping
      unit numeric bindings independent.
- [x] Add exact free-semiring and numeric-homomorphism evaluation.
- [x] Add arbitrary `ON -> OFF -> ON` tests for additive and multiplicative
      ReLU-style circuits.
- [x] Add a random non-staircase benchmark against independent materialization
      and event replay.
- [ ] Add compact columnar node arrays and byte-accurate accounting.
- [ ] Add a persistent semiring-matrix/product-tree specialization for fitted
      multilayer ReLU networks.
- [ ] Add full-persistence/fork APIs for branching unit build schedules.

Acceptance criteria:

- every version polynomial equals independent materialization;
- returning a compressed root takes `O(log B_u)` region lookup plus `O(1)`;
- arbitrary toggles are supported without a laminarity test;
- measured storage reports unique nodes, root references, independent node
  occurrences, changed slots, affected nodes, and newly interned nodes;
- product provenance is never silently converted into an additive-only matrix.

## M2.5: LCF Factorizer and Interval Directory

- [ ] Extract occurrence-weighted canonical fragment supports from population
      cells `(unit,region)` without materializing threshold-expanded copies.
- [ ] Group equal supports and build the support crossing graph.
- [ ] Implement deterministic coloring and report the realized `ell`.
- [ ] Build each laminar inclusion forest and implement LCF inherited-payload
      propagation with LOCAL/BUNDLE nodes.
- [ ] Build each node's multi-chain directed reset graph, with one fixed-zero
      chain start per unit, and solve its exact `s-t` min-cut.
- [ ] Assert exact reconstruction of every population-cell/fragment multiplicity.
- [ ] Emit maximal per-unit guard runs and build linear-space per-unit
      stabbing-semigroup directories.
- [ ] Support compact mode and an explicitly costed region-root cache mode.
- [ ] Add a set-partitioning ILP oracle for small support matrices.
- [ ] Compute `rank(M)` and report `F`, `F*`, `ell`, `R`, `R_S`, `R_A`, `K`,
      `D_0`, `D_b`, and `D_cache`.

Acceptance criteria:

- laminar inputs satisfy `F = rank(M) = F*` against the ILP oracle;
- all checked inputs satisfy `F <= ell*F*` and
  `K <= min(R_S,R_A) <= R`;
- run-aware `K` never exceeds support-only or atom-only `K`, and is strictly
  smaller on the documented adversarial instance;
- compact query roots equal direct `(unit,region)` provenance after the quotient
  map and arbitrary binding `h_u`;
- cache construction and storage are never hidden inside the compact bound;
- exact factorization never relies on subtraction, inverse, or idempotence.

## M3: Unified GPSI Builder and Linear Migration

- [ ] Add the shared `COMPILE_CELL` interface.
- [ ] Add topological parent-partition refinement over arbitrary cDAGs.
- [ ] Implement `LinearCompiler` as a no-new-boundary local compiler.
- [ ] Route the existing linear engine through the new segment and provenance
      structures.
- [ ] Materialize versioned Unit Store and Response Segment Directory.
- [ ] Preserve numerical agreement with the existing compiled evaluator and
      DoWhy replay.

Acceptance criteria:

- all-linear SCMs produce one response segment per unit/target;
- mixed inherited parent boundaries are handled without special-case storage;
- existing linear tests pass through the unified path;
- repeated units share templates when structurally identical while always
  retaining independent bindings; unique templates remain exact singletons.

## M4: Tree and Ensemble Compiler

- [ ] Represent numerical regression trees with exact branch policies.
- [ ] Trace affine parent cells without visiting unreachable branches.
- [ ] Emit response leaf pieces and separate path-certificate pieces.
- [ ] Multiway-merge tree breakpoint streams without leaf Cartesian products.
- [ ] Support one tree, Random Forest, Extra Trees, GBDT, and histogram GBDT
      with additive raw/identity outputs.
- [ ] Add mixed L/T SCM tests, including discontinuous responses.
- [ ] Instrument visited tree nodes and emitted breakpoints.

Acceptance criteria:

- exact agreement with direct fitted-model replay at and between boundaries;
- every observed ensemble partition satisfies the additive breakpoint bound;
- construction never materializes unreachable global leaf combinations;
- jump behavior is visible to downstream FLIP construction.

## M5: Indexed Queries

- [ ] Implement `POINT` over the Response Segment Directory.
- [ ] Implement response and audit modes of `EXPLAIN`.
- [ ] Build registered-threshold FLIP events for interior roots, jumps, and
      plateaus.
- [ ] Implement `FLIP` with sorted event range reporting.
- [ ] Build exact SUM/AVG response maps for declared groups.
- [ ] Implement batched POINT for repeated intervention values.

Acceptance criteria:

- POINT uses `O(log B_u)` predecessor search;
- EXPLAIN accounts for response and certificate output cost;
- registered FLIP uses `O(log E_q+k)` and matches exhaustive state-change
  detection;
- declared AGG agrees with summing direct unit replay.

## M6: ExactLine/PWL Neural Compiler

- [ ] Integrate or reproduce the ExactLine cell-partition contract.
- [ ] Lift numeric pieces to provenance over original weights, biases, parent
      roots, and noise slots.
- [ ] Preserve activation/operator certificates separately from response
      template equality.
- [ ] Add ReLU MLP mechanisms and supported PWL operators.
- [ ] Add mixed L/T/N SCM correctness tests.
- [ ] Measure ExactLine pieces, response quotient segments, certificate
      segments, and unique provenance templates separately.

Acceptance criteria:

- exact agreement with direct network replay on boundaries and cell interiors;
- no opaque effective coefficient replaces original parameter provenance;
- construction reports rather than hides worst-case segment growth.

## M7: Versioning and Maintenance

- [ ] Version parameters, unit bindings, provenance roots, response segments,
      certificates, events, and aggregates together.
- [ ] Recompute changed-node noise and rebuild affected causal slices.
- [ ] Add conservative invalidation before optimized incremental reuse.
- [ ] Measure full rebuild and causal-slice rebuild after retraining.

## M8: Baselines and Evaluation

- [ ] B0: DoWhy replay from observed data.
- [ ] B1: cached-noise vectorized replay.
- [ ] B2: exact numeric per-unit segment index.
- [ ] B3: exact segments with fully expanded provenance paths.
- [ ] B4: global-region/leaf-combination materialization on tractable cases.
- [ ] B5: tree-ensemble ADD/decision-diagram compilation.
- [ ] B6: unmodified numeric ExactLine.
- [ ] Implement all factorization, coalescing, and audit ablations.
- [ ] Report build decomposition, amortization point, bytes, response and
      certificate segments, DAG statistics, and query latency.

## M9: Proofs and Paper

- [ ] Turn every D0 proof sketch into a complete appendix-quality proof.
- [ ] Align theorem symbols with instrumentation emitted by the system.
- [ ] Report negative/worst-case results as explicit non-claims.
- [ ] Write the paper around one object: the guarded counterfactual provenance
      response indexed by GPSI.
- [ ] Keep ExactLine, decision diagrams, provenance semirings, and factorized
      representations as baselines/foundations rather than novelty claims.
