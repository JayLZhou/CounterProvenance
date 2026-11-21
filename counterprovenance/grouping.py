"""Grouping trees for the provenance index.

After binding a unit, tree ``m`` only depends on the intervention axes it still
splits on (its *axis support* ``S_m``) and only changes at its own *cuts* (slab
boundaries) on those axes.  A group of trees is answered by one dense table
over the product of the group's cuts, so a group with support ``S`` and cut
sets ``C_a`` costs ``prod_{a in S} (|C_a| + 1)`` cells and one lookup per query.

Choosing the groups is the problem: minimise the number of groups (table reads
per query) subject to ``sum of group cells <= budget``.  It is NP-hard (reduction
from Product Partition), so no polynomial algorithm approximates the number of
groups within a factor below 3/2 unless P = NP.

:func:`greedy_groups` is a Kruskal-like heuristic without an approximation
guarantee: it starts from groups of nested supports and repeatedly merges the
pair of groups whose union adds the fewest cells, until the total would exceed
``budget``; a merge that does not grow the total is always allowed.  Trees whose own table already
exceeds ``cap``, or ``mask_ratio`` times its number of leaf boxes, are returned
separately (they are answered by leaf bitmasks: a table over the grid of a deep
tree's cuts is mostly redundant, since the tree has far fewer leaves than cells).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .ensemble import BoundEnsemble, Slabs


@dataclass
class TreeProfile:
    tree: int
    items: list[int]                      # global box ids of the tree
    axes: frozenset                       # axis support after binding
    cuts: dict = field(default_factory=dict)  # axis -> set of slab cuts

    @property
    def cells(self) -> int:
        return grid_cells(self.axes, self.cuts)


@dataclass
class Group:
    axes: frozenset
    cuts: dict
    items: list[int]
    trees: list[int]

    @property
    def cells(self) -> int:
        return grid_cells(self.axes, self.cuts)


def grid_cells(axes, cuts) -> int:
    return math.prod(len(cuts.get(a, ())) + 1 for a in axes)


def tree_profiles(bound: BoundEnsemble, slabs: Slabs) -> list[TreeProfile]:
    """One profile per tree that still depends on an intervention axis."""
    by_tree: dict[int, TreeProfile] = {}
    for i, box in enumerate(bound.boxes):
        prof = by_tree.get(box.tree)
        if prof is None:
            prof = by_tree[box.tree] = TreeProfile(box.tree, [], frozenset(), {})
        prof.items.append(i)
        for a in range(bound.d):
            lo, hi = slabs.ranges[i][a]
            if lo > 0 or hi < slabs.counts[a]:
                cuts = prof.cuts.setdefault(a, set())
                if lo > 0:
                    cuts.add(lo)
                if hi < slabs.counts[a]:
                    cuts.add(hi)
    out = []
    for prof in by_tree.values():
        prof.axes = frozenset(prof.cuts)
        if prof.axes:
            out.append(prof)
    return sorted(out, key=lambda p: p.tree)


def nested_groups(profiles: list[TreeProfile], cap: int) -> list[Group]:
    """Trees join an existing group whose support contains theirs while the
    group's table stays within ``cap``; otherwise they open a new group."""
    groups: list[Group] = []
    for prof in sorted(profiles, key=lambda p: (-len(p.axes), sorted(p.axes), p.tree)):
        best, best_size = None, None
        for g in groups:
            if prof.axes <= g.axes:
                size = math.prod(len(g.cuts[a] | prof.cuts.get(a, set())) + 1 for a in g.axes)
                if size <= cap and (best is None or size < best_size):
                    best, best_size = g, size
        if best is None:
            groups.append(Group(prof.axes, {a: set(prof.cuts.get(a, ())) for a in prof.axes},
                                list(prof.items), [prof.tree]))
        else:
            for a in best.axes:
                best.cuts[a] |= prof.cuts.get(a, set())
            best.items.extend(prof.items)
            best.trees.append(prof.tree)
    return groups


def _merged(g1: Group, g2: Group) -> tuple[frozenset, dict, int]:
    axes = g1.axes | g2.axes
    cuts = {a: g1.cuts.get(a, set()) | g2.cuts.get(a, set()) for a in axes}
    return axes, cuts, grid_cells(axes, cuts)


def greedy_groups(
    profiles: list[TreeProfile],
    budget: int,
    cap: int,
    start_cap: int = 1 << 10,
    mask_ratio: float = float("inf"),
    single_size: dict | None = None,
) -> tuple[list[Group], list[TreeProfile]]:
    """Return ``(groups, deep)``: table groups whose total size is at most
    ``budget`` (when the nested start fits), and the trees whose own table
    exceeds ``cap``."""
    is_deep = [p.cells > cap or p.cells > mask_ratio * len(p.items) for p in profiles]
    deep = [p for p, flag in zip(profiles, is_deep) if flag]
    shallow = [p for p, flag in zip(profiles, is_deep) if not flag]
    start = nested_groups(shallow, min(start_cap, cap))
    alive = dict(enumerate(start))
    # a group of one tree is stored in two levels (twolevel.py): its size is the entries of that table
    sizes = {k: (single_size[g.trees[0]] if single_size is not None and len(g.trees) == 1 else g.cells) for k, g in alive.items()}
    total = sum(sizes.values())
    heap: list[tuple[int, int, int]] = []
    keys = list(alive)
    for i, k1 in enumerate(keys):
        g1 = alive[k1]
        for k2 in keys[i + 1:]:
            _, _, size = _merged(g1, alive[k2])
            if size <= cap:
                heap.append((size - sizes[k1] - sizes[k2], k1, k2))
    heapq.heapify(heap)
    nxt = len(start)
    while heap:
        delta, k1, k2 = heapq.heappop(heap)
        # a merge that shrinks the total is always taken, also when the trees alone exceed the budget
        if k1 not in alive or k2 not in alive or (delta > 0 and total + delta > budget):
            continue
        axes, cuts, size = _merged(alive[k1], alive[k2])
        g = Group(axes, cuts, alive[k1].items + alive[k2].items, alive[k1].trees + alive[k2].trees)
        total += delta
        del alive[k1], alive[k2], sizes[k1], sizes[k2]
        k = nxt
        nxt += 1
        for k3, g3 in alive.items():
            _, _, s3 = _merged(g, g3)
            if s3 <= cap:
                heapq.heappush(heap, (s3 - size - sizes[k3], k3, k))
        alive[k] = g
        sizes[k] = size
    groups = sorted(alive.values(), key=lambda g: (sorted(g.axes), min(g.trees)))
    return groups, deep
