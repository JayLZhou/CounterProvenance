"""Construction plans over the face adjacency (version) graph.

Two plan shapes are used:

* **spanning tree** (for immutable path copying, which may branch): the
  source-augmented minimum spanning tree under the exact path-copy weight
  ``rho_e = |Anc(Delta_e)|`` minimises the number of newly allocated PFT nodes
  among all adjacency-constrained single-parent plans (one source build per
  connected component; a source edge costs more than any transition).
* **covering walk** (for node copying, which needs one chain of versions): an
  Euler tour of a spanning tree that visits the heaviest branch last and does
  not return from it.  Its cost is at most ``2 W(T) - tail``; since the edges of
  any covering walk contain a spanning tree, the tour of the minimum spanning
  tree is a 2-approximation of the cheapest covering walk under the same
  additive per-step cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .overlay import FaceEdge
from .persistence import ancestor_union_size

WEIGHTS = ("rho", "count", "hybrid", "unit")


def edge_weights(edges: Sequence[FaceEdge], width: int, kind: str, alpha: float = 1.5) -> list[float]:
    if kind not in WEIGHTS:
        raise ValueError(f"unknown weight {kind!r}")
    out: list[float] = []
    for e in edges:
        c = e.count
        if kind == "count":
            out.append(float(c))
        elif kind == "unit":
            out.append(1.0 if c else 0.0)
        else:
            rho = ancestor_union_size(list(e.changes), width) if c else 0
            out.append(float(rho) if kind == "rho" else float(min(rho, alpha * c)))
    return out


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


@dataclass
class SpanningForest:
    n: int
    parent: list[int]
    parent_edge: list[int]
    children: list[list[int]]
    roots: list[int]
    weight: float  # sum of selected transition-edge weights
    edge_weight: list[float] = field(repr=False)


def spanning_forest(
    n: int,
    edges: Sequence[FaceEdge],
    weights: Sequence[float],
    method: str = "mst",
    seed: int = 0,
) -> SpanningForest:
    """``method='mst'``: Kruskal on ``weights``; ``'arbitrary'``: Kruskal on a
    seeded random edge order (a weight-oblivious spanning tree)."""
    order = list(range(len(edges)))
    if method == "mst":
        order.sort(key=lambda i: (weights[i], i))
    elif method == "arbitrary":
        import random

        random.Random(seed).shuffle(order)
    else:
        raise ValueError(f"unknown spanning method {method!r}")
    dsu = _DSU(n)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    total = 0.0
    for i in order:
        e = edges[i]
        if dsu.union(e.a, e.b):
            adjacency[e.a].append((e.b, i))
            adjacency[e.b].append((e.a, i))
            total += weights[i]
    parent = [-2] * n
    parent_edge = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for r in range(n):
        if parent[r] != -2:
            continue
        roots.append(r)
        parent[r] = -1
        stack = [r]
        while stack:
            v = stack.pop()
            for w, i in adjacency[v]:
                if parent[w] == -2:
                    parent[w] = v
                    parent_edge[w] = i
                    children[v].append(w)
                    stack.append(w)
    return SpanningForest(n, parent, parent_edge, children, roots, total, list(weights))


@dataclass
class Walk:
    """A chain of versions: sources start components, steps follow face edges."""

    events: list[tuple[str, int, int]]  # ("source", face, -1) | ("step", face_to, edge index)
    cost: float
    first_visit: list[int]


def euler_walk(forest: SpanningForest) -> Walk:
    """Euler tour of every component; heaviest branch last and never returned from."""
    n = forest.n
    w = forest.edge_weight
    down = [0.0] * n  # heaviest downward path weight
    # post-order without recursion
    for r in forest.roots:
        stack = [(r, False)]
        while stack:
            v, done = stack.pop()
            if done:
                best = 0.0
                for c in forest.children[v]:
                    best = max(best, w[forest.parent_edge[c]] + down[c])
                down[v] = best
                continue
            stack.append((v, True))
            for c in forest.children[v]:
                stack.append((c, False))
    events: list[tuple[str, int, int]] = []
    cost = 0.0
    first = [-1] * n
    for r in forest.roots:
        component: list[tuple[str, int, int]] = [("source", r, -1)]
        last_forward = 0
        ordered = {
            v: sorted(forest.children[v], key=lambda c: (w[forest.parent_edge[c]] + down[c], c))
            for v in _nodes_of(forest, r)
        }
        stack: list[tuple[int, int]] = [(r, 0)]
        while stack:
            v, k = stack[-1]
            kids = ordered[v]
            if k < len(kids):
                stack[-1] = (v, k + 1)
                c = kids[k]
                component.append(("step", c, forest.parent_edge[c]))
                last_forward = len(component)
                stack.append((c, 0))
            else:
                stack.pop()
                if stack:
                    component.append(("step", stack[-1][0], forest.parent_edge[v]))
        component = component[:last_forward] if last_forward else component[:1]
        for kind, face, edge in component:
            if kind == "step":
                cost += w[edge]
        events.extend(component)
    position = 0
    for kind, face, edge in events:
        if first[face] < 0:
            first[face] = position
        position += 1
    return Walk(events, cost, first)


def greedy_dfs_forest(n: int, edges: Sequence[FaceEdge], weights: Sequence[float]) -> SpanningForest:
    """Depth-first tree that always moves to the cheapest unvisited neighbour.

    On grid-like face graphs this produces long boustrophedon-like runs, so the
    shortcut walk over its preorder rarely needs to backtrack."""
    adjacency: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        adjacency[e.a].append((weights[i], e.b, i))
        adjacency[e.b].append((weights[i], e.a, i))
    for lst in adjacency:
        lst.sort()
    parent = [-2] * n
    parent_edge = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    pointer = [0] * n
    total = 0.0
    for r in range(n):
        if parent[r] != -2:
            continue
        roots.append(r)
        parent[r] = -1
        stack = [r]
        while stack:
            v = stack[-1]
            lst = adjacency[v]
            while pointer[v] < len(lst) and parent[lst[pointer[v]][1]] != -2:
                pointer[v] += 1
            if pointer[v] == len(lst):
                stack.pop()
                continue
            w, u, i = lst[pointer[v]]
            parent[u] = v
            parent_edge[u] = i
            children[v].append(u)
            total += w
            stack.append(u)
    return SpanningForest(n, parent, parent_edge, children, roots, total, list(weights))


def shortcut_walk(
    forest: SpanningForest,
    edges: Sequence[FaceEdge],
    weights: Sequence[float],
    heavy_last: bool = True,
) -> Walk:
    """Visit the faces in the forest's preorder; between consecutive faces use a
    direct face edge when one exists and is cheaper than the tree path.

    The cost never exceeds that of the Euler tour, hence at most twice the
    forest weight."""
    n = forest.n
    w = forest.edge_weight
    index = {(e.a, e.b): i for i, e in enumerate(edges)}
    depth_w = [0.0] * n
    down = [0.0] * n
    order_children: list[list[int]] = [list(c) for c in forest.children]
    if heavy_last:
        for r in forest.roots:
            post: list[int] = []
            stack = [r]
            while stack:
                v = stack.pop()
                post.append(v)
                stack.extend(forest.children[v])
            for v in reversed(post):
                best = 0.0
                for c in forest.children[v]:
                    best = max(best, w[forest.parent_edge[c]] + down[c])
                down[v] = best
            for v in post:
                order_children[v] = sorted(forest.children[v], key=lambda c: (w[forest.parent_edge[c]] + down[c], c))
    events: list[tuple[str, int, int]] = []
    cost = 0.0
    for r in forest.roots:
        preorder: list[int] = []
        stack = [r]
        while stack:
            v = stack.pop()
            preorder.append(v)
            for c in reversed(order_children[v]):
                depth_w[c] = depth_w[v] + w[forest.parent_edge[c]]
                stack.append(c)
        events.append(("source", r, -1))
        for u, v in zip(preorder, preorder[1:]):
            p = forest.parent[v]
            tree_cost = depth_w[u] - depth_w[p] + w[forest.parent_edge[v]]
            key = (u, v) if u < v else (v, u)
            direct = index.get(key)
            if direct is not None and weights[direct] <= tree_cost:
                events.append(("step", v, direct))
                cost += weights[direct]
                continue
            x = u
            while x != p:  # ascend to v's parent along tree edges
                events.append(("step", forest.parent[x], forest.parent_edge[x]))
                x = forest.parent[x]
            events.append(("step", v, forest.parent_edge[v]))
            cost += tree_cost
    first = [-1] * n
    for position, (_, face, _) in enumerate(events):
        if first[face] < 0:
            first[face] = position
    return Walk(events, cost, first)


def best_walk(n: int, edges: Sequence[FaceEdge], weights: Sequence[float], method: str = "mst") -> Walk:
    """Cheaper of the MST shortcut walk (2-approximation) and the greedy DFS walk."""
    mst = spanning_forest(n, edges, weights, method=method)
    candidates = [shortcut_walk(mst, edges, weights)]
    if method == "mst":
        candidates.append(shortcut_walk(greedy_dfs_forest(n, edges, weights), edges, weights, heavy_last=False))
    return min(candidates, key=lambda walk: walk.cost)


def _nodes_of(forest: SpanningForest, root: int) -> list[int]:
    out, stack = [], [root]
    while stack:
        v = stack.pop()
        out.append(v)
        stack.extend(forest.children[v])
    return out


def path_walk(order: Sequence[int], edges: Sequence[FaceEdge], weights: Sequence[float]) -> Walk:
    """A Hamiltonian chain (e.g. boustrophedon over a full grid)."""
    index = {(e.a, e.b): i for i, e in enumerate(edges)}
    events: list[tuple[str, int, int]] = [("source", order[0], -1)]
    cost = 0.0
    for prev, face in zip(order, order[1:]):
        key = (prev, face) if prev < face else (face, prev)
        if key not in index:
            raise ValueError(f"faces {prev} and {face} are not adjacent")
        i = index[key]
        events.append(("step", face, i))
        cost += weights[i]
    first = [-1] * (max(order) + 1)
    for position, (_, face, _) in enumerate(events):
        if first[face] < 0:
            first[face] = position
    return Walk(events, cost, first)


def boustrophedon(shape: Sequence[int]) -> list[int]:
    """Row-major flat indices visited so that consecutive cells are grid-adjacent."""
    if not shape:
        return [0]
    # build from the innermost axis outwards: the sub-order is reversed on
    # every other index of the enclosing axis, so block boundaries stay adjacent
    order = list(range(shape[-1]))
    stride = shape[-1]
    for k in range(len(shape) - 2, -1, -1):
        reverse = order[::-1]
        merged: list[int] = []
        for i in range(shape[k]):
            base = i * stride
            merged.extend(base + c for c in (order if i % 2 == 0 else reverse))
        order = merged
        stride *= shape[k]
    return order
