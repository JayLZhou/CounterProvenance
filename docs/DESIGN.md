# Design and guarantees

This document states what the index stores, how it is built, and exactly which
guarantees hold in which cost model. Claims that are empirical are marked as
such.

## 1. Problem

Fix a fitted additive-noise SCM, one observed unit `u`, a target `Y` whose
mechanism is a tree ensemble, and `d` numeric intervention variables. After
abduction, every non-intervened input of the ensemble is a constant, so

```
Y_u(t) = offset_u + sum_m value_m(leaf_m(t)),   t in R^d,
```

where `offset_u` is the base score plus the abducted noise. A query
`do(T = t)` must return the exact value and its exact provenance: the
multiset of `(tree, leaf)` atoms in the free commutative semiring `N[X]`
(addition is not idempotent, there are no inverses).

**Leaf boxes.** Binding the unit follows every split on a non-intervened
feature with the unit's own value. What remains of tree `m` is a partition of
`R^d` into axis-aligned boxes `lo < t <= hi` (reachable leaves only). Let `N`
be the total number of boxes. A query is a point-enclosure query: the answer
is the semiring sum of the payloads of the `M` boxes that contain `t`.

**Overlay.** The common refinement of all boxes partitions `R^d` into `K`
faces; inside a face every tree selects a fixed leaf. Crossing a boundary
between adjacent faces changes the leaves of `c_e` trees. `K` can be as large
as `N^d`.

Supported models (`counterprovenance.ensemble`): scikit-learn
`GradientBoostingRegressor`, `HistGradientBoostingRegressor`,
`RandomForestRegressor`, `ExtraTreesRegressor`, `DecisionTreeRegressor`;
XGBoost `gbtree` regressors (`exact`, `hist`, ...); LightGBM regressors. All
are normalised to `x <= threshold` splits on the model's own input precision;
XGBoost's `x < theta` on float32 becomes `x <= nextafter32(theta, -inf)`.
Identity-link objectives only; categorical splits and LightGBM
`zero_as_missing` are rejected.

## 2. The index

One structure, two parameters (`ProvenanceIndex`):

1. **Groups.** Trees are grouped by the set of intervention axes they split
   on. A group's state space is the overlay of its own trees, typically far
   smaller than the overlay of all trees. `group_budget` merges small groups
   into supersets.
2. **Adaptive segment tree.** Inside a group, if the overlay of the remaining
   axes has at most `tau` grid cells it becomes one MCPST leaf. Otherwise a
   segment tree is built on the axis with most boundaries; every node keeps
   the boxes that span it and recurses on the remaining axes. Two boxes of
   one tree that span the same node are disjoint on a remaining axis, so a
   tree still contributes at most one box per point and stays one slot.
3. **MCPST leaf** (`CellIndex`): exact overlay, construction plan, persistent
   factor tree, one directly addressable root and cached value per face.

A query visits one leaf per segment-tree level per group and returns one
pre-materialised root ("piece") per visited non-empty leaf. `tau = inf` with
one group is a single MCPST (one piece); `tau = 0` recurses to the last axis,
whose leaves are 1D persistent chains. `autotune` builds a grid of
`(tau, group budget)` configurations and picks the smallest index within a
piece budget (or the fewest pieces within a space budget) on sample queries.

### 2.1 Exact overlay

Boxes are mapped to their own compressed grid. Every box face labels the grid
edges it covers with `(slot, item below, item above)`. These labels are the
change sets `Delta_e`; no `K x M` signature matrix is built. Grid cells joined
by an unlabelled edge have equal signatures, so the connected components of
the unlabelled-edge graph are exactly the faces of the non-Cartesian overlay.
Labelled edges between faces form the face adjacency graph. On the six
California-housing models at d = 2 the exact overlay has 2.0-6.9x fewer faces
than the full threshold grid (`results/RESULTS.md`, R3).

### 2.2 Persistent factor tree

A balanced binary tree over the slots; a version assigns an item (or
"inactive") to every slot. Two update mechanisms, mixable per update:

* **Path copying** (immutable): copy the union of root paths of the changed
  slots, exactly `rho_e = |Anc(Delta_e)|` new nodes; any earlier version may
  be the parent.
* **Node copying** (Sarnak-Tarjan, Driscoll-Sarnak-Sleator-Tarjan partial
  persistence): every node owns `capacity` version-stamped modification
  entries and is copied only when they are full. Nodes have in-degree one, so
  this costs O(1) amortised records per changed slot, but versions must form
  one chain.

### 2.3 Construction plans

* **Tree plan** for path copying: the source-augmented minimum spanning tree
  of the face graph under `rho_e`.
* **Walk plan** for node copying: a covering walk of the face graph. We take
  the cheaper of (a) the preorder of the MST with shortcuts (move along a
  direct face edge when it is cheaper than the tree path) and (b) the same
  construction on a greedy depth-first tree that always moves to the cheapest
  unvisited neighbour. On full grids (b) reproduces the boustrophedon chain.
* **Hybrid** (default): compute the exact path-copy cost of the tree plan and
  the surrogate cost of the walk whose steps use path copying when
  `rho <= alpha c` and node copying otherwise (`alpha = 1.5`); materialise the
  cheaper plan.

## 3. Guarantees

Statements are per leaf (per overlay) unless noted. `P` is the payload table,
`B` the number of 1D regions, `K` faces, `K_Delta` effective slot changes.

**T1 (exactness).** For every query point, the value equals `offset_u` plus
the sum of the leaf values selected by the fitted model, and the pieces'
atoms are exactly the multiset of selected `(tree, leaf)` pairs, including on
split thresholds. *Proof idea:* faces have constant signatures (2.1);
every persistent version decodes to its face signature (induction along the
plan); segment-tree canonical decomposition assigns each containing box to
exactly one visited node; groups partition the trees. Verified in the test
suite against brute force and against each model's own leaf ids.

**T2 (1D persistent materialisation).** For a 1D change sequence, node
copying stores a direct root for every region in `O(P + B + K_Delta)` space
(`K_Delta = N - M` for tree ensembles). This is linear in the explicit input
when boundaries and effective updates are stored explicitly and queries may
not re-read the fitted model; it is a representation-model statement, not an
information-theoretic lower bound (the model itself can recompute everything).

**T3 (MST optimality for immutable materialisation).** For a fixed persistent
factor-tree layout, the source-augmented MST under `rho_e` minimises the
number of newly allocated path-copied nodes among all adjacency-constrained
single-parent plans. *Proof:* plans and rooted spanning trees of the
source-augmented graph are in bijection with equal additive cost.

**T4 (covering walks for node copying).** Any covering walk's step set
contains a spanning tree, so its cost under an additive per-step cost is at
least the MST weight; the shortcut walk of the MST costs at most twice the MST
weight. Hence `walk = best of (MST shortcut, greedy DFS)` is a
2-approximation of the cheapest covering walk, and node copying realises it
in O(walk cost) amortised records. In general position (`c_e = 1` on every
face edge) every spanning tree has weight `K - 1`, which lower-bounds every
single-parent plan under the Hamming update cost; a Hamiltonian chain (e.g.
the boustrophedon on a full grid) attains it.

**T5 (chain endpoint).** With `tau = 0` and one group, the index uses
`O(N log^{d-1} N)` space and returns `O(log^{d-1} N)` pre-materialised pieces
per query (one per visited segment-tree node), in `O(log^d N)` time.

**Lower-bound context.** In the arithmetic/semigroup model where pieces are
query-independent pre-materialised aggregates and a query combines at most `t`
of them, Chazelle's lower bound for dominance range searching transfers to
ensembles (map every non-target leaf atom to the semiring zero): a single
piece needs `N^{1 + Omega(1)}` space and near-linear space needs polylog
pieces. The exponents depend on the original theorem's constants; we do not
claim the endpoints match them exactly.

**Not claimed.** Global optimality of the hybrid's realised allocation (its
MST minimises a surrogate); optimality of the axis-support grouping (it is a
heuristic; optimal grouping is open); bit-level optimality; arbitrary
algebraic factorisation of provenance; models outside the list in Section 1.

## 4. Relation to prior work

Persistence by path and node copying (Sarnak-Tarjan 1986; Driscoll, Sarnak,
Sleator, Tarjan 1989), minimum-spanning-tree version storage (Bhattacherjee et
al., VLDB 2015; delta compression by Ouyang et al. 2002), segment trees and
semigroup range searching (Chazelle 1990; Agarwal-Erickson survey), semiring
provenance (Green, Karvounarakis, Tannen 2007) and factorised representations
(Olteanu, Zavodny 2015) are used as components. What is specific here: the
version graph and its change labels are compiled from the counterfactual
geometry of a fitted ensemble; the edge costs are exact persistent-structure
costs; covering walks connect MST planning with partial persistence; and the
grouped adaptive segment tree spans the range from one root per query to
polylog pieces with space independent of `K`.
