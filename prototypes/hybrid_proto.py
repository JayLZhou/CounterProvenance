#!/usr/bin/env python3
"""Per-edge hybrid of path copying and node copying along the MCPST version tree.

For every construction edge the materialiser chooses path copying (fresh
immutable ancestor union, cost rho_e = |Anc(Delta_e)|) when that is cheaper
than node copying (about 2 entries per changed slot plus occasional copies),
otherwise node copying.  Both kinds of nodes coexist in one persistent arena,
so any mixture is exact.  The spanning plan uses the hybrid weight
min(rho_e, alpha * c_e).
"""

from __future__ import annotations

import sys
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "mcpst_2d"))

from counterprovenance.mcpst import (  # noqa: E402
    BalancedPFTLayout,
    CellTransition,
    build_spanning_plan,
    grid_adjacency,
    independent_pft_nodes,
    materialize_plan,
    transitions_from_states,
)
from run_experiments import make_stump_grid  # noqa: E402
from nodecopy_proto import (  # noqa: E402
    NodeCopyPFT,
    SnakePlan,
    chain_adjacency,
    payload_ref,
    preorder_numbering,
    random_toggle_states,
)


class HybridPFT(NodeCopyPFT):
    def path_copy(self, version: int, root_of: list[int], pre: int, updates: dict[int, int]) -> int:
        positions = sorted(updates)
        created_before = len(self.nodes)

        def copy(x: int, left: int, right: int, lo: int, hi: int) -> int:
            if lo >= hi:
                return x
            if right - left == 1:
                return updates[positions[lo]]
            mid = (left + right) // 2
            split = bisect_left(positions, mid, lo, hi)
            xl = self.read(x, 0, pre)
            xr = self.read(x, 1, pre)
            nl = copy(xl, left, mid, lo, split)
            nr = copy(xr, mid, right, split, hi)
            return self._new(nl, nr)

        root_of[version] = copy(root_of[version], 0, self.layout.padded_leaf_count, 0, len(positions))
        return len(self.nodes) - created_before


def materialize(states, layout, plan, mode: str, alpha: float, capacity: int = 2):
    """mode in {'path', 'node', 'hybrid'}; returns (arena, records, edges_path, edges_node)."""
    pre, post, order = preorder_numbering(plan.parents)
    arena = HybridPFT(layout, capacity)
    root_of = [0] * len(plan.parents)
    V = len(plan.parents)
    e_path = e_node = 0
    for v in order:
        parent = plan.parents[v]
        if parent == -1:
            root_of[v] = arena.build([payload_ref(m, states[v, m]) for m in range(layout.component_count)])
            continue
        root_of[v] = root_of[parent]
        changed = plan.parent_changes[v]
        if not changed:
            continue
        rho = layout.transition_cost(changed)
        use_path = mode == "path" or (mode == "hybrid" and rho <= alpha * len(changed))
        if use_path:
            e_path += 1
            arena.path_copy(v, root_of, pre[v], {m: payload_ref(m, states[v, m]) for m in changed})
        else:
            e_node += 1
            for m in changed:
                arena.update(v, root_of, pre[v], post[v], V, m, payload_ref(m, states[v, m]))
    # exactness: decode every version (sampled on big instances)
    if V * layout.component_count <= 40_000_000:
        to_check = range(V)
    else:
        to_check = np.random.default_rng(0).choice(V, size=4000, replace=False).tolist()
    for v in to_check:
        if arena.decode(root_of[v], pre[v]) != [payload_ref(m, states[v, m]) for m in range(layout.component_count)]:
            raise AssertionError(f"version {v} decoded incorrectly ({mode})")
    return arena, arena.records, e_path, e_node


def hybrid_transitions(transitions, alpha: float):
    return tuple(
        CellTransition(t.left, t.right, t.changed, min(t.weight, int(np.ceil(alpha * len(t.changed)))) if t.changed else 0)
        for t in transitions
    )


def run_case(name: str, states: np.ndarray, adjacency, grid=None, alpha: float = 2.5) -> None:
    """Separate the two questions: (a) which materialiser, (b) which spanning plan.

    Columns:  path@rho  = path copying along the rho_e-MST (current MCPST)
              node@cnt  = node copying along the c_e-MST
              hyb@rho / hyb@cnt / hyb@hyb / hyb@unw = hybrid materialiser along the
              rho_e-MST, the c_e-MST, the MST under the hybrid weight, and an
              arbitrary (unweighted-Kruskal) spanning tree.
    """
    layout = BalancedPFTLayout(states.shape[1])
    K, M = states.shape
    transitions = transitions_from_states(states, adjacency, layout)
    plans = {
        "rho": build_spanning_plan(K, transitions, layout, objective="pft"),
        "cnt": build_spanning_plan(K, transitions, layout, objective="changed"),
        "hyb": build_spanning_plan(K, hybrid_transitions(transitions, alpha), layout, objective="pft"),
        "unw": build_spanning_plan(K, transitions, layout, objective="unweighted"),
    }
    _, path_rho, _, _ = materialize(states, layout, plans["rho"], "path", alpha)
    _, node_cnt, _, _ = materialize(states, layout, plans["cnt"], "node", alpha)
    hyb = {}
    mix = {}
    for key, plan in plans.items():
        _, rec, ep, en = materialize(states, layout, plan, "hybrid", alpha)
        hyb[key] = rec
        mix[key] = (ep, en)
    best_single = min(path_rho, node_cnt)
    line = (
        f"{name:<24} M={M:>4} K={K:>6} IIP={independent_pft_nodes(K, layout):>11,} | "
        f"path@rho={path_rho:>9,} node@cnt={node_cnt:>9,} | "
        f"hyb@rho={hyb['rho']:>9,} hyb@cnt={hyb['cnt']:>9,} hyb@hyb={hyb['hyb']:>9,} hyb@unw={hyb['unw']:>9,} "
        f"(hyb-MST edges: path {mix['hyb'][0]:,} / node {mix['hyb'][1]:,}) | "
        f"hyb@hyb/min(path,node)={hyb['hyb']/best_single:.3f}  unw/hyb-MST={hyb['unw']/hyb['hyb']:.3f}"
    )
    if grid is not None:
        plan_snake = SnakePlan(states, *grid)
        _, s_path, _, _ = materialize(states, layout, plan_snake, "path", alpha)
        _, s_node, _, _ = materialize(states, layout, plan_snake, "node", alpha)
        _, s_hyb, sp, sn = materialize(states, layout, plan_snake, "hybrid", 1.6)
        line += f" | snake chain: path={s_path:,} node={s_node:,} hybrid={s_hyb:,}"
    print(line, flush=True)


def main() -> None:
    import platform
    print(f"host={platform.node()} python={platform.python_version()} numpy={np.__version__}", flush=True)
    seed = 2028
    print("== 2D general position ==")
    for M in (64, 256, 512):
        w = make_stump_grid(M // 2, M - M // 2, 1, seed + M)
        run_case(f"2D general M={M}", w.states, grid_adjacency(w.locator.x_cells, w.locator.y_cells), grid=(w.locator.x_cells, w.locator.y_cells))
    print("== 2D coincident thresholds (M=256) ==")
    for kappa in (1, 2, 4, 8, 16):
        g = 128 // kappa
        w = make_stump_grid(g, g, kappa, seed + 100 + kappa)
        run_case(f"2D kappa={kappa}", w.states, grid_adjacency(w.locator.x_cells, w.locator.y_cells), grid=(w.locator.x_cells, w.locator.y_cells))
    print("== 2D ablation workload (c_e=4, scattered vs contiguous) ==")
    w = make_stump_grid(8, 8, 4, seed + 200, order="y_scattered")
    run_case("2D ablation", w.states, grid_adjacency(w.locator.x_cells, w.locator.y_cells), grid=(w.locator.x_cells, w.locator.y_cells))
    print("== 2D random dense selector grids ==")
    rng = np.random.default_rng(7)
    st = rng.integers(0, 4, size=(900, 64), dtype=np.int64)
    run_case("2D random 30x30 M=64", st, grid_adjacency(30, 30), grid=(30, 30))
    st = rng.integers(0, 3, size=(36, 16), dtype=np.int64)
    run_case("2D random 6x6 M=16", st, grid_adjacency(6, 6), grid=(6, 6))
    print("== 1D random toggles (C=256, B=2000) ==")
    for k in (1, 4, 16, 64):
        st = random_toggle_states(256, 2000, k, seed + k)
        run_case(f"1D toggles k={k}", st, chain_adjacency(2000))


if __name__ == "__main__":
    main()
