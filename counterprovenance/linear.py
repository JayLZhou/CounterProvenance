"""Generic linear-SCM counterfactual provenance.

This module implements the linear foundation for TESSERA:

* arbitrary directed acyclic causal graphs;
* additive linear mechanisms;
* one or more structural interventions;
* expanded semiring monomials as logical provenance;
* reverse trie storage for compressed suffix-to-target fragments;
* compiled closed-form evaluation for repeated counterfactual queries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


Token = str
Monomial = Tuple[Token, Tuple[Token, ...]]


@dataclass(frozen=True)
class LinearMechanism:
    """A fitted linear mechanism ``child = bias + sum_j w_j parent_j + noise``."""

    child: str
    parents: Tuple[str, ...]
    weights: Mapping[str, float]
    bias: float = 0.0

    def __post_init__(self) -> None:
        missing = set(self.parents) - set(self.weights)
        extra = set(self.weights) - set(self.parents)
        if missing or extra:
            raise ValueError(
                f"Mechanism {self.child!r} has parent/weight mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )


@dataclass(frozen=True)
class LinearSCM:
    """A linear additive-noise SCM over an arbitrary cDAG."""

    mechanisms: Mapping[str, LinearMechanism]

    def __post_init__(self) -> None:
        for child, mechanism in self.mechanisms.items():
            if child != mechanism.child:
                raise ValueError(f"Mechanism key {child!r} does not match child {mechanism.child!r}")
            if child in mechanism.parents:
                raise ValueError(f"Self loop detected at {child!r}")
        # Trigger validation once.
        self.topological_order()

    @property
    def nodes(self) -> Tuple[str, ...]:
        result = set(self.mechanisms)
        for mechanism in self.mechanisms.values():
            result.update(mechanism.parents)
        return tuple(sorted(result))

    @property
    def roots(self) -> Tuple[str, ...]:
        children = set(self.mechanisms)
        return tuple(node for node in self.topological_order() if node not in children)

    def parents_of(self, node: str) -> Tuple[str, ...]:
        if node not in self.mechanisms:
            return tuple()
        return self.mechanisms[node].parents

    def topological_order(self) -> Tuple[str, ...]:
        nodes = set(self.mechanisms)
        adjacency: Dict[str, List[str]] = {}
        indegree: Dict[str, int] = {}

        for mechanism in self.mechanisms.values():
            nodes.update(mechanism.parents)

        for node in nodes:
            adjacency[node] = []
            indegree[node] = 0

        for child, mechanism in self.mechanisms.items():
            for parent in mechanism.parents:
                adjacency[parent].append(child)
                indegree[child] += 1

        ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        order: List[str] = []

        while ready:
            node = ready.popleft()
            order.append(node)
            for child in sorted(adjacency[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

        if len(order) != len(nodes):
            raise ValueError("The linear SCM graph must be acyclic.")

        return tuple(order)


@dataclass
class TrieNode:
    children: Dict[Token, "TrieNode"] = field(default_factory=dict)


class ReverseProvenanceTrie:
    """Compressed storage for expanded semiring monomial paths.

    Monomials are logical source-to-target paths:

        source_token * theta:e1 * ... * theta:ek

    The trie stores them in reverse order to share suffix-to-target fragments.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self.num_paths = 0

    def insert(self, source_token: Token, edge_tokens_source_to_target: Sequence[Token]) -> None:
        node = self.root
        for token in reversed(edge_tokens_source_to_target):
            node = node.children.setdefault(token, TrieNode())
        node.children.setdefault(source_token, TrieNode())
        self.num_paths += 1

    def evaluate(self, valuation: Mapping[Token, float]) -> float:
        return self._evaluate_node(self.root, valuation)

    def _evaluate_node(self, node: TrieNode, valuation: Mapping[Token, float]) -> float:
        if not node.children:
            return 1.0

        total = 0.0
        for token, child in node.children.items():
            total += valuation[token] * self._evaluate_node(child, valuation)
        return total

    def node_count(self) -> int:
        def count(node: TrieNode) -> int:
            return 1 + sum(count(child) for child in node.children.values())

        return count(self.root)

    def edge_count(self) -> int:
        def count(node: TrieNode) -> int:
            return len(node.children) + sum(count(child) for child in node.children.values())

        return count(self.root)


@dataclass(frozen=True)
class CompiledLinearForm:
    constant: float
    intervention_coefs: Mapping[str, float]
    root_coefs: Mapping[str, float]
    noise_coefs: Mapping[str, float]


def edge_token(parent: str, child: str) -> Token:
    return f"theta:{parent}->{child}"


def noise_token(node: str) -> Token:
    return f"noise:{node}"


def bias_token(node: str) -> Token:
    return f"bias:{node}"


def root_token(node: str) -> Token:
    return f"root:{node}"


def intervention_token(node: str) -> Token:
    return f"do:{node}"


def combine_paths(paths: Iterable[Monomial], edge: Token) -> List[Monomial]:
    return [(source, path + (edge,)) for source, path in paths]


def expand_linear_monomials(
    scm: LinearSCM,
    interventions: Iterable[str],
) -> Dict[str, List[Monomial]]:
    """Expand linear counterfactual provenance for all nodes.

    Intervened nodes are represented by ``do:<node>`` tokens. Non-intervened
    roots are represented by ``root:<node>`` tokens, matching DoWhy-GCM's
    observed-data counterfactual semantics for root nodes.
    """

    intervention_set = set(interventions)
    monomials: Dict[str, List[Monomial]] = {}

    for node in scm.topological_order():
        if node in intervention_set:
            monomials[node] = [(intervention_token(node), tuple())]
            continue

        if node not in scm.mechanisms:
            monomials[node] = [(root_token(node), tuple())]
            continue

        mechanism = scm.mechanisms[node]
        parts: List[Monomial] = []
        for parent in mechanism.parents:
            parts.extend(combine_paths(monomials[parent], edge_token(parent, node)))

        parts.append((bias_token(node), tuple()))
        parts.append((noise_token(node), tuple()))
        monomials[node] = parts

    return monomials


def build_trie_for_target(monomials: Sequence[Monomial]) -> ReverseProvenanceTrie:
    trie = ReverseProvenanceTrie()
    for source, path in monomials:
        trie.insert(source, path)
    return trie


def token_product(path: Sequence[Token], parameter_values: Mapping[Token, float]) -> float:
    product = 1.0
    for token in path:
        product *= parameter_values[token]
    return product


def base_parameter_valuation(scm: LinearSCM) -> Dict[Token, float]:
    valuation: Dict[Token, float] = {}
    for child, mechanism in scm.mechanisms.items():
        valuation[bias_token(child)] = mechanism.bias
        for parent, weight in mechanism.weights.items():
            valuation[edge_token(parent, child)] = float(weight)
    return valuation


def compile_linear_form(
    monomials: Sequence[Monomial],
    parameter_values: Mapping[Token, float],
) -> CompiledLinearForm:
    """Compile monomial provenance into vectorized linear coefficients."""

    constant = 0.0
    intervention_coefs: Dict[str, float] = {}
    root_coefs: Dict[str, float] = {}
    noise_coefs: Dict[str, float] = {}

    for source, path in monomials:
        coef = token_product(path, parameter_values)
        if source.startswith("do:"):
            node = source.removeprefix("do:")
            intervention_coefs[node] = intervention_coefs.get(node, 0.0) + coef
        elif source.startswith("root:"):
            node = source.removeprefix("root:")
            root_coefs[node] = root_coefs.get(node, 0.0) + coef
        elif source.startswith("bias:"):
            constant += parameter_values[source] * coef
        elif source.startswith("noise:"):
            node = source.removeprefix("noise:")
            noise_coefs[node] = noise_coefs.get(node, 0.0) + coef
        else:
            raise ValueError(f"Unsupported source token {source!r}")

    return CompiledLinearForm(constant, intervention_coefs, root_coefs, noise_coefs)


def _as_vector(value: float | np.ndarray, n: int) -> np.ndarray:
    if np.isscalar(value):
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"Expected vector of shape {(n,)}, got {arr.shape}")
    return arr


def evaluate_compiled_form(
    compiled: CompiledLinearForm,
    root_values: Mapping[str, np.ndarray],
    noises: Mapping[str, np.ndarray],
    interventions: Mapping[str, float | np.ndarray],
    n: int | None = None,
) -> np.ndarray:
    """Evaluate a compiled counterfactual form for all units."""

    if n is None:
        if root_values:
            n = len(next(iter(root_values.values())))
        elif noises:
            n = len(next(iter(noises.values())))
        elif interventions:
            first = next(iter(interventions.values()))
            n = 1 if np.isscalar(first) else len(np.asarray(first))
        else:
            n = 1

    result = np.full(n, compiled.constant)
    for node, coef in compiled.intervention_coefs.items():
        result += coef * _as_vector(interventions[node], n)
    for node, coef in compiled.root_coefs.items():
        result += coef * root_values[node]
    for node, coef in compiled.noise_coefs.items():
        result += coef * noises[node]
    return result


def unit_valuation(
    parameter_values: Mapping[Token, float],
    root_values: Mapping[str, np.ndarray],
    noises: Mapping[str, np.ndarray],
    interventions: Mapping[str, float | np.ndarray],
    unit_idx: int,
) -> Dict[Token, float]:
    valuation = dict(parameter_values)
    for node, value in interventions.items():
        if np.isscalar(value):
            valuation[intervention_token(node)] = float(value)
        else:
            valuation[intervention_token(node)] = float(np.asarray(value)[unit_idx])
    for node, values in root_values.items():
        valuation[root_token(node)] = float(values[unit_idx])
    for node, values in noises.items():
        valuation[noise_token(node)] = float(values[unit_idx])
    return valuation


def vectorized_trie_eval(
    trie: ReverseProvenanceTrie,
    parameter_values: Mapping[Token, float],
    root_values: Mapping[str, np.ndarray],
    noises: Mapping[str, np.ndarray],
    interventions: Mapping[str, float | np.ndarray],
    n: int,
) -> np.ndarray:
    """Slow reference evaluator over the physical trie."""

    results = np.empty(n)
    for idx in range(n):
        results[idx] = trie.evaluate(unit_valuation(parameter_values, root_values, noises, interventions, idx))
    return results


def path_storage_tokens(monomials: Sequence[Monomial]) -> int:
    return sum(len(path) + 1 for _, path in monomials)


def simulate_linear_scm(
    scm: LinearSCM,
    n: int,
    seed: int,
    noise_scales: Mapping[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    """Draw observed data from a linear SCM."""

    rng = np.random.default_rng(seed)
    data: Dict[str, np.ndarray] = {}
    scales = dict(noise_scales or {})

    for root in scm.roots:
        data[root] = rng.normal(0.0, 1.0, size=n)

    for node in scm.topological_order():
        if node not in scm.mechanisms:
            continue
        mechanism = scm.mechanisms[node]
        predicted = np.full(n, mechanism.bias)
        for parent in mechanism.parents:
            predicted += mechanism.weights[parent] * data[parent]
        scale = scales.get(node, 0.3)
        data[node] = predicted + rng.normal(0.0, scale, size=n)

    return data


def root_values_from_data(scm: LinearSCM, data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {root: data[root] for root in scm.roots}


def abduct_noises(
    scm: LinearSCM,
    data: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Recover additive residuals for all non-root mechanisms."""

    noises: Dict[str, np.ndarray] = {}
    for node in scm.topological_order():
        if node not in scm.mechanisms:
            continue
        mechanism = scm.mechanisms[node]
        predicted = np.full_like(data[node], mechanism.bias)
        for parent in mechanism.parents:
            predicted += mechanism.weights[parent] * data[parent]
        noises[node] = data[node] - predicted
    return noises


def counterfactual_replay(
    scm: LinearSCM,
    root_values: Mapping[str, np.ndarray],
    noises: Mapping[str, np.ndarray],
    interventions: Mapping[str, float | np.ndarray],
) -> Dict[str, np.ndarray]:
    """DoWhy-style additive-noise counterfactual replay."""

    if root_values:
        n = len(next(iter(root_values.values())))
    elif noises:
        n = len(next(iter(noises.values())))
    else:
        n = 1

    cf: Dict[str, np.ndarray] = {}
    for node in scm.topological_order():
        if node in interventions:
            cf[node] = _as_vector(interventions[node], n)
            continue

        if node not in scm.mechanisms:
            cf[node] = root_values[node]
            continue

        mechanism = scm.mechanisms[node]
        predicted = np.full(n, mechanism.bias)
        for parent in mechanism.parents:
            predicted += mechanism.weights[parent] * cf[parent]
        cf[node] = predicted + noises[node]

    return cf


def make_layered_linear_scm(
    layer_sizes: Sequence[int],
    edge_prob: float,
    seed: int,
    weight_scale: float = 1.0,
) -> Tuple[LinearSCM, Tuple[str, ...], Tuple[str, ...]]:
    """Create a random layered linear cDAG for compression stress tests.

    Edges go from earlier layers to later layers. Every non-root node is forced
    to have at least one parent, so the graph remains connected enough for a
    target query.
    """

    if len(layer_sizes) < 2:
        raise ValueError("Need at least a root layer and one non-root layer.")

    rng = np.random.default_rng(seed)
    layers: List[List[str]] = []
    for layer_idx, size in enumerate(layer_sizes):
        if size <= 0:
            raise ValueError("Layer sizes must be positive.")
        layers.append([f"L{layer_idx}_{node_idx}" for node_idx in range(size)])

    mechanisms: Dict[str, LinearMechanism] = {}
    earlier_nodes: List[str] = list(layers[0])

    for layer_idx in range(1, len(layers)):
        for child in layers[layer_idx]:
            parents = [node for node in earlier_nodes if rng.random() < edge_prob]
            if not parents:
                parents = [str(rng.choice(earlier_nodes))]
            parents = sorted(set(parents))
            weights = {parent: float(rng.normal(0.0, weight_scale)) for parent in parents}
            mechanisms[child] = LinearMechanism(
                child=child,
                parents=tuple(parents),
                weights=weights,
                bias=float(rng.normal(0.0, 0.2)),
            )
        earlier_nodes.extend(layers[layer_idx])

    return LinearSCM(mechanisms), tuple(layers[0]), tuple(layers[-1])
