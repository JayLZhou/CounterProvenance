#!/usr/bin/env python3
"""Generic linear-cDAG counterfactual provenance trie demo.

This demo exercises the reusable ``counterprovenance`` linear engine:

* arbitrary linear additive-noise cDAGs;
* one or more structural interventions;
* expanded semiring monomials as logical provenance;
* reverse trie compression for suffix-to-target propagation fragments;
* compiled closed-form evaluation for fast repeated queries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Mapping, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterprovenance import (
    LinearMechanism,
    LinearSCM,
    abduct_noises,
    base_parameter_valuation,
    build_trie_for_target,
    compile_linear_form,
    counterfactual_replay,
    evaluate_compiled_form,
    expand_linear_monomials,
    make_layered_linear_scm,
    path_storage_tokens,
    root_values_from_data,
    simulate_linear_scm,
    vectorized_trie_eval,
)


def build_toy_scm() -> LinearSCM:
    return LinearSCM(
        {
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
    )


def import_dowhy_status() -> str:
    try:
        import dowhy  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent.
        return f"not installed ({exc.__class__.__name__}: {exc})"

    version = getattr(dowhy, "__version__", "unknown")
    return f"installed (version={version})"


def dowhy_gcm_counterfactual_if_available(
    data: Mapping[str, np.ndarray],
    scm: LinearSCM,
    target: str,
    interventions: Mapping[str, float],
    repeats: int,
    bench: bool,
) -> Tuple[str, float | None]:
    """Fit DoWhy-GCM linear ANMs and compare them with compiled provenance."""

    try:
        import networkx as nx  # type: ignore
        import pandas as pd  # type: ignore
        import dowhy.gcm as gcm  # type: ignore
        from dowhy.gcm.ml import create_linear_regressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent.
        return f"skipped ({exc.__class__.__name__}: {exc})", None

    graph = nx.DiGraph()
    graph.add_nodes_from(scm.topological_order())
    for child, mechanism in scm.mechanisms.items():
        for parent in mechanism.parents:
            graph.add_edge(parent, child)

    causal_model = gcm.InvertibleStructuralCausalModel(graph)
    for root in scm.roots:
        causal_model.set_causal_mechanism(root, gcm.EmpiricalDistribution())
    for node in scm.topological_order():
        if node in scm.mechanisms:
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(create_linear_regressor()))

    observed = pd.DataFrame({node: data[node] for node in scm.topological_order()})
    gcm.fit(causal_model, observed)
    dowhy_cf = gcm.counterfactual_samples(
        causal_model,
        {node: (lambda value: lambda _: value)(value) for node, value in interventions.items()},
        observed_data=observed,
    )

    fitted_mechanisms: Dict[str, LinearMechanism] = {}
    for node in scm.topological_order():
        if node not in scm.mechanisms:
            continue
        parent_order = scm.parents_of(node)
        sklearn_model = causal_model.causal_mechanism(node).prediction_model.sklearn_model
        coefs = np.asarray(sklearn_model.coef_, dtype=float).reshape(-1)
        fitted_mechanisms[node] = LinearMechanism(
            node,
            parent_order,
            {parent: float(coef) for parent, coef in zip(parent_order, coefs)},
            bias=float(sklearn_model.intercept_),
        )

    fitted_scm = LinearSCM(fitted_mechanisms)
    root_values = root_values_from_data(fitted_scm, data)
    fitted_noises = abduct_noises(fitted_scm, data)
    monomials = expand_linear_monomials(fitted_scm, interventions)[target]
    parameter_values = base_parameter_valuation(fitted_scm)
    compiled = compile_linear_form(monomials, parameter_values)
    compiled_values = evaluate_compiled_form(compiled, root_values, fitted_noises, interventions)
    max_abs_error = float(np.max(np.abs(compiled_values - dowhy_cf[target].to_numpy())))

    if bench:
        start = time.perf_counter()
        for _ in range(repeats):
            _ = gcm.counterfactual_samples(
                causal_model,
                {node: (lambda value: lambda _: value)(value) for node, value in interventions.items()},
                observed_data=observed,
            )
        dowhy_seconds = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(repeats):
            _ = evaluate_compiled_form(compiled, root_values, fitted_noises, interventions)
        compiled_seconds = time.perf_counter() - start
        speedup = dowhy_seconds / compiled_seconds if compiled_seconds > 0 else float("inf")
        return (
            "ran dowhy.gcm linear AdditiveNoiseModel; "
            f"dowhy query {dowhy_seconds:.4f}s vs compiled {compiled_seconds:.4f}s "
            f"for {repeats} repeats ({speedup:.2f}x)",
            max_abs_error,
        )

    return "ran dowhy.gcm linear AdditiveNoiseModel", max_abs_error


def parse_layer_sizes(value: str) -> Tuple[int, ...]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("Need at least two layers, e.g. 2,6,6,1.")
    return parts


def make_scm(args: argparse.Namespace) -> Tuple[LinearSCM, str, Dict[str, float], str]:
    if args.mode == "toy":
        scm = build_toy_scm()
        return scm, "Y", {"T": args.t_prime}, "T->A->B/Y toy cDAG"

    scm, roots, targets = make_layered_linear_scm(
        layer_sizes=args.layer_sizes,
        edge_prob=args.edge_prob,
        seed=args.seed,
        weight_scale=args.weight_scale,
    )
    intervention_node = roots[0]
    target = targets[-1]
    return (
        scm,
        target,
        {intervention_node: args.t_prime},
        f"random layered cDAG layers={args.layer_sizes}, do({intervention_node}={args.t_prime}), target={target}",
    )


def run_demo(args: argparse.Namespace) -> None:
    scm, target, interventions, graph_label = make_scm(args)
    data = simulate_linear_scm(scm, n=args.n, seed=args.seed)
    root_values = root_values_from_data(scm, data)
    noises = abduct_noises(scm, data)
    cf = counterfactual_replay(scm, root_values, noises, interventions)

    monomials_by_node = expand_linear_monomials(scm, interventions)
    target_monomials = monomials_by_node[target]
    trie = build_trie_for_target(target_monomials)
    parameter_values = base_parameter_valuation(scm)
    compiled = compile_linear_form(target_monomials, parameter_values)

    trie_values = vectorized_trie_eval(trie, parameter_values, root_values, noises, interventions, args.n)
    compiled_values = evaluate_compiled_form(compiled, root_values, noises, interventions, args.n)
    trie_error = float(np.max(np.abs(trie_values - cf[target])))
    compiled_error = float(np.max(np.abs(compiled_values - cf[target])))

    expanded_tokens = path_storage_tokens(target_monomials)
    trie_edges = trie.edge_count()

    start = time.perf_counter()
    for _ in range(args.repeats):
        _ = counterfactual_replay(scm, root_values, noises, interventions)[target]
    replay_seconds = time.perf_counter() - start

    trie_seconds = None
    if not args.skip_raw_trie_timing:
        start = time.perf_counter()
        for _ in range(args.repeats):
            _ = vectorized_trie_eval(trie, parameter_values, root_values, noises, interventions, args.n)
        trie_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(args.repeats):
        _ = evaluate_compiled_form(compiled, root_values, noises, interventions, args.n)
    compiled_seconds = time.perf_counter() - start

    print("=== TESSERA generic linear cDAG provenance trie demo ===")
    print(f"DoWhy import status: {import_dowhy_status()}")
    print()
    print(f"graph: {graph_label}")
    print(f"nodes: {len(scm.nodes)}, mechanisms: {len(scm.mechanisms)}, roots: {len(scm.roots)}")
    print(f"units: {args.n:,}, target: {target}, interventions: {interventions}")
    print()
    print("Validation against additive-noise counterfactual replay")
    print(f"max |trie(target_cf) - replay(target_cf)| = {trie_error:.3e}")
    print(f"max |compiled(target_cf) - replay(target_cf)| = {compiled_error:.3e}")

    if not args.skip_dowhy:
        dowhy_message, dowhy_error = dowhy_gcm_counterfactual_if_available(
            data,
            scm,
            target,
            interventions,
            args.repeats,
            args.bench_dowhy,
        )
        print(f"DoWhy-GCM comparison: {dowhy_message}")
        if dowhy_error is not None:
            print(f"max |compiled(target_cf) - dowhy.gcm(target_cf)| = {dowhy_error:.3e}")

    print()
    if not args.quiet:
        print(f"Expanded semiring monomials for {target}")
        for source, path in target_monomials:
            print("  " + " * ".join((source, *path)))
        print()

    print("Storage sketch")
    print(f"expanded monomials: {len(target_monomials)}")
    print(f"expanded token slots: {expanded_tokens}")
    print(f"reverse trie nodes: {trie.node_count()}  (including root)")
    print(f"reverse trie edges/tokens: {trie_edges}")
    print(f"token compression vs expanded slots: {expanded_tokens / max(trie_edges, 1):.2f}x")
    print()
    print("Tiny timing loop")
    print(f"replay baseline: {replay_seconds:.4f}s for {args.repeats} repeats")
    if trie_seconds is None:
        print("raw trie eval:   skipped")
    else:
        print(f"raw trie eval:   {trie_seconds:.4f}s for {args.repeats} repeats")
    print(f"compiled eval:   {compiled_seconds:.4f}s for {args.repeats} repeats")
    if compiled_seconds > 0:
        print(f"compiled speedup vs replay: {replay_seconds / compiled_seconds:.2f}x")
    print()
    print("Note: raw trie traversal validates storage semantics; compiled eval is the fast linear query plan.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "layered"), default="toy")
    parser.add_argument("--layer-sizes", type=parse_layer_sizes, default=(2, 5, 5, 1))
    parser.add_argument("--edge-prob", type=float, default=0.55)
    parser.add_argument("--weight-scale", type=float, default=0.8)
    parser.add_argument("--n", type=int, default=1000, help="Number of observed units.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--t-prime", type=float, default=1.75, help="Intervention value.")
    parser.add_argument("--repeats", type=int, default=20, help="Repeated query count for timing.")
    parser.add_argument("--skip-dowhy", action="store_true", help="Skip optional real dowhy.gcm comparison.")
    parser.add_argument("--bench-dowhy", action="store_true", help="Also time real dowhy.gcm.counterfactual_samples.")
    parser.add_argument("--skip-raw-trie-timing", action="store_true", help="Skip slow per-unit Python trie timing.")
    parser.add_argument("--quiet", action="store_true", help="Do not print every expanded monomial.")
    return parser.parse_args()


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()
