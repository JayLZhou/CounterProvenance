#!/usr/bin/env python3
"""Linear-SCM counterfactual provenance trie demo.

This is a deliberately small TESSERA prototype:

* Semantics: expanded semiring monomials for a linear SCM counterfactual.
* Physical representation: a reverse trie that shares suffix-to-target path
  fragments instead of storing every monomial independently.
* Validation: compare trie evaluation against a DoWhy-style
  abduction-action-prediction replay.

The script does not require DoWhy. If dowhy is installed, the output notes that
the environment has it, but the comparison below uses a transparent
DoWhy-style replay with the same fitted linear coefficients so the demo stays
deterministic and easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


Token = str
Monomial = Tuple[Token, Tuple[Token, ...]]


@dataclass(frozen=True)
class LinearMechanism:
    """A fitted linear mechanism X_child = sum_j w_j X_j + noise."""

    child: str
    parents: Tuple[str, ...]
    weights: Mapping[str, float]
    bias: float = 0.0


@dataclass
class TrieNode:
    children: Dict[Token, "TrieNode"] = field(default_factory=dict)


class ReverseProvenanceTrie:
    """Compressed physical storage for expanded semiring monomial paths.

    Each expanded monomial is represented as:

        source_token * edge_token_1 * ... * edge_token_k

    where edges are listed from source to target. For a fixed target, we insert
    the reversed sequence:

        edge_token_k, ..., edge_token_1, source_token

    This shares suffix-to-target fragments such as A->B->Y across both
    intervention and noise contributions.
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


def edge_token(parent: str, child: str) -> Token:
    return f"theta:{parent}->{child}"


def noise_token(node: str) -> Token:
    return f"noise:{node}"


def bias_token(node: str) -> Token:
    return f"bias:{node}"


def intervention_token(node: str) -> Token:
    return f"do:{node}"


def build_demo_scm() -> Dict[str, LinearMechanism]:
    """A small SCM with shared suffix-to-target fragments.

    Graph:

        T -> A -> B -> Y
        T -> A -> C -> Y
        T -> D -> Y
        T -> Y

    The A->B->Y suffix is shared by T's contribution through A and A's own
    noise contribution. The same holds for A->C->Y.
    """

    return {
        "A": LinearMechanism("A", ("T",), {"T": 1.35}, bias=0.40),
        "B": LinearMechanism("B", ("A",), {"A": -0.70}, bias=-0.20),
        "C": LinearMechanism("C", ("A",), {"A": 0.45}, bias=0.10),
        "D": LinearMechanism("D", ("T",), {"T": 0.80}, bias=-0.15),
        "Y": LinearMechanism(
            "Y",
            ("B", "C", "D", "T"),
            {"B": 1.10, "C": -1.60, "D": 0.55, "T": 0.25},
            bias=0.05,
        ),
    }


def topological_nodes() -> Tuple[str, ...]:
    return ("T", "A", "B", "C", "D", "Y")


def simulate_units(n: int, seed: int, mechanisms: Mapping[str, LinearMechanism]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    data: Dict[str, np.ndarray] = {}
    data["T"] = rng.normal(0.0, 1.0, size=n)

    noise_scale = {
        "A": 0.35,
        "B": 0.25,
        "C": 0.25,
        "D": 0.30,
        "Y": 0.40,
    }
    for node in topological_nodes()[1:]:
        mechanism = mechanisms[node]
        predicted = np.full(n, mechanism.bias)
        for parent in mechanism.parents:
            predicted += mechanism.weights[parent] * data[parent]
        data[node] = predicted + rng.normal(0.0, noise_scale[node], size=n)

    return data


def abduct_noises(
    data: Mapping[str, np.ndarray],
    mechanisms: Mapping[str, LinearMechanism],
) -> Dict[str, np.ndarray]:
    noises: Dict[str, np.ndarray] = {}
    for node in topological_nodes()[1:]:
        mechanism = mechanisms[node]
        predicted = np.full_like(data[node], mechanism.bias)
        for parent in mechanism.parents:
            predicted += mechanism.weights[parent] * data[parent]
        noises[node] = data[node] - predicted
    return noises


def dowhy_style_counterfactual_replay(
    mechanisms: Mapping[str, LinearMechanism],
    noises: Mapping[str, np.ndarray],
    t_prime: float,
) -> Dict[str, np.ndarray]:
    """Transparent abduction-action-prediction replay.

    This mirrors the DoWhy-GCM counterfactual contract for additive-noise
    mechanisms: keep unit-level noises fixed, set T to t_prime, and propagate
    forward in topological order.
    """

    n = len(next(iter(noises.values())))
    cf: Dict[str, np.ndarray] = {"T": np.full(n, t_prime)}
    for node in topological_nodes()[1:]:
        mechanism = mechanisms[node]
        predicted = np.full(n, mechanism.bias)
        for parent in mechanism.parents:
            predicted += mechanism.weights[parent] * cf[parent]
        cf[node] = predicted + noises[node]
    return cf


def combine_paths(
    left: Iterable[Tuple[Token, Tuple[Token, ...]]],
    edge: Token,
) -> List[Tuple[Token, Tuple[Token, ...]]]:
    return [(source, path + (edge,)) for source, path in left]


def expanded_linear_monomials(
    mechanisms: Mapping[str, LinearMechanism],
    intervention: str,
    target: str,
) -> Dict[str, List[Monomial]]:
    """Build expanded semiring monomials for every node.

    A monomial is (source_token, edge_tokens_source_to_target).
    Numeric coefficients live in valuation[edge_token]; unit-specific residuals
    live in valuation[noise_token(node)].
    """

    monomials: Dict[str, List[Monomial]] = {intervention: [(intervention_token(intervention), tuple())]}

    for node in topological_nodes()[1:]:
        parts: List[Monomial] = []
        mechanism = mechanisms[node]
        for parent in mechanism.parents:
            parent_paths = monomials[parent]
            parts.extend(combine_paths(parent_paths, edge_token(parent, node)))
        parts.append((bias_token(node), tuple()))
        parts.append((noise_token(node), tuple()))
        monomials[node] = parts

    if target not in monomials:
        raise KeyError(f"Unknown target {target}")

    return monomials


def build_trie_for_target(monomials: Sequence[Monomial]) -> ReverseProvenanceTrie:
    trie = ReverseProvenanceTrie()
    for source, path in monomials:
        trie.insert(source, path)
    return trie


def base_parameter_valuation(mechanisms: Mapping[str, LinearMechanism]) -> Dict[Token, float]:
    valuation: Dict[Token, float] = {}
    for child, mechanism in mechanisms.items():
        valuation[bias_token(child)] = mechanism.bias
        for parent, weight in mechanism.weights.items():
            valuation[edge_token(parent, child)] = weight
    return valuation


def unit_valuation(
    parameter_values: Mapping[Token, float],
    noises: Mapping[str, np.ndarray],
    unit_idx: int,
    intervention: str,
    t_prime: float,
) -> Dict[Token, float]:
    valuation = dict(parameter_values)
    valuation[intervention_token(intervention)] = t_prime
    for node, values in noises.items():
        valuation[noise_token(node)] = float(values[unit_idx])
    return valuation


def vectorized_trie_eval(
    trie: ReverseProvenanceTrie,
    parameter_values: Mapping[Token, float],
    noises: Mapping[str, np.ndarray],
    intervention: str,
    t_prime: float,
    n: int,
) -> np.ndarray:
    results = np.empty(n)
    for idx in range(n):
        results[idx] = trie.evaluate(unit_valuation(parameter_values, noises, idx, intervention, t_prime))
    return results


def token_product(path: Sequence[Token], parameter_values: Mapping[Token, float]) -> float:
    product = 1.0
    for token in path:
        product *= parameter_values[token]
    return product


def compile_linear_closed_form(
    monomials: Sequence[Monomial],
    parameter_values: Mapping[Token, float],
    intervention: str,
) -> Tuple[float, float, Dict[str, float]]:
    """Compile monomial provenance into vectorized linear coefficients.

    For a fixed target in a linear SCM, the expanded provenance has the form:

        constant + do_coef * t_prime + sum_i noise_coef[i] * noise_i(u)

    This is the fast evaluation path. The trie is the compressed provenance
    store; this compiled closed form is the query plan we derive from it.
    """

    constant = 0.0
    do_coef = 0.0
    noise_coefs: Dict[str, float] = {}
    do_token = intervention_token(intervention)

    for source, path in monomials:
        coef = token_product(path, parameter_values)
        if source == do_token:
            do_coef += coef
        elif source.startswith("bias:"):
            constant += parameter_values[source] * coef
        elif source.startswith("noise:"):
            node = source.removeprefix("noise:")
            noise_coefs[node] = noise_coefs.get(node, 0.0) + coef
        else:
            raise ValueError(f"Unsupported source token {source!r}")

    return constant, do_coef, noise_coefs


def compiled_linear_eval(
    constant: float,
    do_coef: float,
    noise_coefs: Mapping[str, float],
    noises: Mapping[str, np.ndarray],
    t_prime: float,
) -> np.ndarray:
    n = len(next(iter(noises.values())))
    result = np.full(n, constant + do_coef * t_prime)
    for node, coef in noise_coefs.items():
        result += coef * noises[node]
    return result


def path_storage_tokens(monomials: Sequence[Monomial]) -> int:
    return sum(len(path) + 1 for _, path in monomials)


def import_dowhy_status() -> str:
    try:
        import dowhy  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent.
        return f"not installed ({exc.__class__.__name__}: {exc})"

    version = getattr(dowhy, "__version__", "unknown")
    return f"installed (version={version})"


def dowhy_gcm_counterfactual_if_available(
    data: Mapping[str, np.ndarray],
    t_prime: float,
    repeats: int,
    bench: bool,
) -> Tuple[str, float | None]:
    """Fit DoWhy-GCM linear ANMs and compare them with trie evaluation.

    This is optional because dowhy is not a lightweight dependency. The trie is
    rebuilt from DoWhy's fitted linear coefficients and residuals, then compared
    against dowhy.gcm.counterfactual_samples on the same observed rows.
    """

    try:
        import networkx as nx  # type: ignore
        import pandas as pd  # type: ignore
        import dowhy.gcm as gcm  # type: ignore
        from dowhy.gcm.ml import create_linear_regressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent.
        return f"skipped ({exc.__class__.__name__}: {exc})", None

    base_mechanisms = build_demo_scm()
    graph = nx.DiGraph()
    graph.add_nodes_from(topological_nodes())
    for child in topological_nodes()[1:]:
        for parent in base_mechanisms[child].parents:
            graph.add_edge(parent, child)

    causal_model = gcm.InvertibleStructuralCausalModel(graph)
    causal_model.set_causal_mechanism("T", gcm.EmpiricalDistribution())
    for node in topological_nodes()[1:]:
        causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(create_linear_regressor()))

    observed = pd.DataFrame({node: data[node] for node in topological_nodes()})
    gcm.fit(causal_model, observed)
    dowhy_cf = gcm.counterfactual_samples(causal_model, {"T": lambda _: t_prime}, observed_data=observed)

    fitted_mechanisms: Dict[str, LinearMechanism] = {}
    for node in topological_nodes()[1:]:
        parent_order = base_mechanisms[node].parents
        sklearn_model = causal_model.causal_mechanism(node).prediction_model.sklearn_model
        coefs = np.asarray(sklearn_model.coef_, dtype=float).reshape(-1)
        fitted_mechanisms[node] = LinearMechanism(
            node,
            parent_order,
            {parent: float(coef) for parent, coef in zip(parent_order, coefs)},
            bias=float(sklearn_model.intercept_),
        )

    fitted_noises = abduct_noises(data, fitted_mechanisms)
    monomials = expanded_linear_monomials(fitted_mechanisms, intervention="T", target="Y")["Y"]
    trie = build_trie_for_target(monomials)
    parameter_values = base_parameter_valuation(fitted_mechanisms)
    trie_values = vectorized_trie_eval(trie, parameter_values, fitted_noises, "T", t_prime, len(observed))
    max_abs_error = float(np.max(np.abs(trie_values - dowhy_cf["Y"].to_numpy())))

    if bench:
        constant, do_coef, noise_coefs = compile_linear_closed_form(monomials, parameter_values, "T")

        start = time.perf_counter()
        for _ in range(repeats):
            _ = gcm.counterfactual_samples(causal_model, {"T": lambda _: t_prime}, observed_data=observed)
        dowhy_seconds = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(repeats):
            _ = compiled_linear_eval(constant, do_coef, noise_coefs, fitted_noises, t_prime)
        compiled_seconds = time.perf_counter() - start

        speedup = dowhy_seconds / compiled_seconds if compiled_seconds > 0 else float("inf")
        return (
            "ran dowhy.gcm with fitted linear AdditiveNoiseModel; "
            f"dowhy query {dowhy_seconds:.4f}s vs compiled {compiled_seconds:.4f}s "
            f"for {repeats} repeats ({speedup:.2f}x)",
            max_abs_error,
        )

    return "ran dowhy.gcm with fitted linear AdditiveNoiseModel", max_abs_error


def run_demo(
    n: int,
    seed: int,
    t_prime: float,
    repeats: int,
    skip_dowhy: bool,
    bench_dowhy: bool,
    quiet: bool,
    skip_raw_trie_timing: bool,
) -> None:
    mechanisms = build_demo_scm()
    data = simulate_units(n=n, seed=seed, mechanisms=mechanisms)
    noises = abduct_noises(data, mechanisms)
    cf = dowhy_style_counterfactual_replay(mechanisms, noises, t_prime)

    monomials_by_node = expanded_linear_monomials(mechanisms, intervention="T", target="Y")
    target_monomials = monomials_by_node["Y"]
    trie = build_trie_for_target(target_monomials)
    parameter_values = base_parameter_valuation(mechanisms)

    trie_values = vectorized_trie_eval(trie, parameter_values, noises, "T", t_prime, n)
    max_abs_error = float(np.max(np.abs(trie_values - cf["Y"])))
    constant, do_coef, noise_coefs = compile_linear_closed_form(target_monomials, parameter_values, "T")
    compiled_values = compiled_linear_eval(constant, do_coef, noise_coefs, noises, t_prime)
    compiled_max_abs_error = float(np.max(np.abs(compiled_values - cf["Y"])))

    expanded_tokens = path_storage_tokens(target_monomials)
    trie_edges = trie.edge_count()

    start = time.perf_counter()
    for _ in range(repeats):
        _ = dowhy_style_counterfactual_replay(mechanisms, noises, t_prime)["Y"]
    replay_seconds = time.perf_counter() - start

    trie_seconds = None
    if not skip_raw_trie_timing:
        start = time.perf_counter()
        for _ in range(repeats):
            _ = vectorized_trie_eval(trie, parameter_values, noises, "T", t_prime, n)
        trie_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        _ = compiled_linear_eval(constant, do_coef, noise_coefs, noises, t_prime)
    compiled_seconds = time.perf_counter() - start

    print("=== TESSERA linear provenance trie demo ===")
    print(f"DoWhy import status: {import_dowhy_status()}")
    print()
    print("SCM graph: T->A->B->Y, T->A->C->Y, T->D->Y, T->Y")
    print(f"units: {n:,}, intervention: do(T={t_prime})")
    print()
    print("Validation against DoWhy-style additive-noise replay")
    print(f"max |trie(Y_cf) - replay(Y_cf)| = {max_abs_error:.3e}")
    print(f"max |compiled(Y_cf) - replay(Y_cf)| = {compiled_max_abs_error:.3e}")
    if not skip_dowhy:
        dowhy_message, dowhy_error = dowhy_gcm_counterfactual_if_available(data, t_prime, repeats, bench_dowhy)
        print(f"DoWhy-GCM comparison: {dowhy_message}")
        if dowhy_error is not None:
            print(f"max |trie(Y_cf) - dowhy.gcm(Y_cf)| = {dowhy_error:.3e}")
    print()
    if not quiet:
        print("Expanded semiring monomials for Y")
        for source, path in target_monomials:
            tokens = " * ".join((source, *path))
            print(f"  {tokens}")
        print()
    print("Storage sketch")
    print(f"expanded monomials: {len(target_monomials)}")
    print(f"expanded token slots: {expanded_tokens}")
    print(f"reverse trie nodes: {trie.node_count()}  (including root)")
    print(f"reverse trie edges/tokens: {trie_edges}")
    print(f"token compression vs expanded slots: {expanded_tokens / max(trie_edges, 1):.2f}x")
    print()
    print("Tiny timing loop")
    print(f"replay baseline: {replay_seconds:.4f}s for {repeats} repeats")
    if trie_seconds is None:
        print("raw trie eval:   skipped")
    else:
        print(f"raw trie eval:   {trie_seconds:.4f}s for {repeats} repeats")
    print(f"compiled eval:   {compiled_seconds:.4f}s for {repeats} repeats")
    if compiled_seconds > 0:
        print(f"compiled speedup vs replay: {replay_seconds / compiled_seconds:.2f}x")
    print()
    print("Note: raw Python trie traversal validates storage semantics; compiled eval is the fast linear query plan.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="Number of observed units.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--t-prime", type=float, default=1.75, help="Intervention value do(T=t_prime).")
    parser.add_argument("--repeats", type=int, default=20, help="Tiny timing-loop repeats.")
    parser.add_argument("--skip-dowhy", action="store_true", help="Skip optional real dowhy.gcm comparison.")
    parser.add_argument("--bench-dowhy", action="store_true", help="Also time real dowhy.gcm.counterfactual_samples.")
    parser.add_argument("--skip-raw-trie-timing", action="store_true", help="Skip slow per-unit Python trie timing.")
    parser.add_argument("--quiet", action="store_true", help="Do not print every expanded monomial.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(
        n=args.n,
        seed=args.seed,
        t_prime=args.t_prime,
        repeats=args.repeats,
        skip_dowhy=args.skip_dowhy,
        bench_dowhy=args.bench_dowhy,
        quiet=args.quiet,
        skip_raw_trie_timing=args.skip_raw_trie_timing,
    )


if __name__ == "__main__":
    main()
