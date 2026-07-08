from __future__ import annotations

import unittest

import numpy as np

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
    root_values_from_data,
    simulate_linear_scm,
    vectorized_trie_eval,
)


class LinearGeneralTests(unittest.TestCase):
    def assert_matches_replay(
        self,
        scm: LinearSCM,
        target: str,
        interventions: dict[str, float],
        n: int = 250,
    ) -> None:
        data = simulate_linear_scm(scm, n=n, seed=11)
        roots = root_values_from_data(scm, data)
        noises = abduct_noises(scm, data)
        replay = counterfactual_replay(scm, roots, noises, interventions)[target]
        monomials = expand_linear_monomials(scm, interventions)[target]
        params = base_parameter_valuation(scm)
        trie = build_trie_for_target(monomials)
        trie_values = vectorized_trie_eval(trie, params, roots, noises, interventions, n)
        compiled = compile_linear_form(monomials, params)
        compiled_values = evaluate_compiled_form(compiled, roots, noises, interventions, n)

        np.testing.assert_allclose(trie_values, replay, rtol=0, atol=1e-12)
        np.testing.assert_allclose(compiled_values, replay, rtol=0, atol=1e-12)

    def test_chain_scm(self) -> None:
        scm = LinearSCM(
            {
                "M": LinearMechanism("M", ("T",), {"T": 2.0}, bias=0.3),
                "Y": LinearMechanism("Y", ("M",), {"M": -1.5}, bias=0.1),
            }
        )
        self.assert_matches_replay(scm, "Y", {"T": 1.25})

    def test_non_intervened_root_is_preserved(self) -> None:
        scm = LinearSCM(
            {
                "M": LinearMechanism("M", ("T", "Z"), {"T": 1.2, "Z": -0.4}, bias=-0.1),
                "Y": LinearMechanism("Y", ("M", "Z"), {"M": 0.7, "Z": 0.9}, bias=0.2),
            }
        )
        self.assert_matches_replay(scm, "Y", {"T": 2.0})

    def test_multiple_interventions(self) -> None:
        scm = LinearSCM(
            {
                "A": LinearMechanism("A", ("T",), {"T": 1.0}, bias=0.0),
                "B": LinearMechanism("B", ("A",), {"A": 2.0}, bias=0.5),
                "Y": LinearMechanism("Y", ("A", "B"), {"A": -0.3, "B": 0.8}, bias=-0.2),
            }
        )
        self.assert_matches_replay(scm, "Y", {"T": 1.5, "B": -0.7})

    def test_cycle_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LinearSCM(
                {
                    "A": LinearMechanism("A", ("B",), {"B": 1.0}),
                    "B": LinearMechanism("B", ("A",), {"A": 1.0}),
                }
            )


if __name__ == "__main__":
    unittest.main()
