"""Hash-consed circuits in the free commutative provenance semiring N[X].

Addition is *not* idempotent: ``a + a`` keeps multiplicity two.  Children of
commutative operators are sorted only to canonicalise commutativity; duplicate
children are retained.  The arena is shared by every index in this package so
that equal subcircuits (tree leaves, aggregates) are stored once.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

NodeId = int
Monomial = tuple[str, ...]
Polynomial = Counter  # Counter[Monomial, int]


@dataclass(frozen=True, slots=True)
class CircuitNode:
    op: str  # "zero" | "one" | "atom" | "add" | "mul"
    children: tuple[NodeId, ...] = ()
    label: Hashable | None = None


class CircuitArena:
    """Append-only, hash-consed semiring circuit nodes."""

    def __init__(self) -> None:
        self._nodes: list[CircuitNode] = []
        self._interned: dict[CircuitNode, NodeId] = {}
        self.zero = self._intern(CircuitNode("zero"))
        self.one = self._intern(CircuitNode("one"))

    # -- construction -----------------------------------------------------
    def _intern(self, node: CircuitNode) -> NodeId:
        existing = self._interned.get(node)
        if existing is not None:
            return existing
        node_id = len(self._nodes)
        self._nodes.append(node)
        self._interned[node] = node_id
        return node_id

    def atom(self, label: Hashable) -> NodeId:
        return self._intern(CircuitNode("atom", label=label))

    def add(self, *children: NodeId) -> NodeId:
        kept = [child for child in children if child != self.zero]
        if not kept:
            return self.zero
        if len(kept) == 1:
            return kept[0]
        return self._intern(CircuitNode("add", tuple(sorted(kept))))

    def add_all(self, children: Iterable[NodeId]) -> NodeId:
        """Balanced binary sum (keeps circuits shallow for large sums)."""
        level = [child for child in children if child != self.zero]
        if not level:
            return self.zero
        while len(level) > 1:
            level = [
                self.add(*level[i : i + 2]) for i in range(0, len(level), 2)
            ]
        return level[0]

    def mul(self, *children: NodeId) -> NodeId:
        if any(child == self.zero for child in children):
            return self.zero
        kept = [child for child in children if child != self.one]
        if not kept:
            return self.one
        if len(kept) == 1:
            return kept[0]
        return self._intern(CircuitNode("mul", tuple(sorted(kept))))

    # -- inspection -------------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def node(self, node_id: NodeId) -> CircuitNode:
        return self._nodes[node_id]

    def atoms(self, root: NodeId) -> Counter:
        """Multiset of atom labels under ``root`` for sum-only circuits."""
        out: Counter = Counter()
        stack = [root]
        while stack:
            node = self._nodes[stack.pop()]
            if node.op == "atom":
                out[node.label] += 1
            elif node.op == "add":
                stack.extend(node.children)
            elif node.op == "zero":
                continue
            else:
                raise ValueError(f"atoms() expects a sum circuit, found {node.op!r}")
        return out

    def evaluate(self, root: NodeId, valuation: Mapping[Hashable, float]) -> float:
        """Evaluate under a numeric semiring homomorphism (atoms -> reals)."""
        memo: dict[NodeId, float] = {}

        def visit(node_id: NodeId) -> float:
            if node_id in memo:
                return memo[node_id]
            node = self._nodes[node_id]
            if node.op == "zero":
                value = 0.0
            elif node.op == "one":
                value = 1.0
            elif node.op == "atom":
                value = float(valuation[node.label])
            elif node.op == "add":
                value = sum(visit(child) for child in node.children)
            else:
                value = 1.0
                for child in node.children:
                    value *= visit(child)
            memo[node_id] = value
            return value

        return visit(root)

    def polynomial(self, root: NodeId) -> Polynomial:
        """Expand into N[X] (for tests on small circuits only)."""
        memo: dict[NodeId, Counter] = {}

        def visit(node_id: NodeId) -> Counter:
            if node_id in memo:
                return memo[node_id]
            node = self._nodes[node_id]
            if node.op == "zero":
                result: Counter = Counter()
            elif node.op == "one":
                result = Counter({(): 1})
            elif node.op == "atom":
                result = Counter({(str(node.label),): 1})
            elif node.op == "add":
                result = Counter()
                for child in node.children:
                    result.update(visit(child))
            else:
                result = Counter({(): 1})
                for child in node.children:
                    product: Counter = Counter()
                    for left, lc in result.items():
                        for right, rc in visit(child).items():
                            product[tuple(sorted(left + right))] += lc * rc
                    result = product
            memo[node_id] = result
            return result

        return Counter(visit(root))
