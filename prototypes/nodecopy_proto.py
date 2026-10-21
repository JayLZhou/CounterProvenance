#!/usr/bin/env python3
"""Node copying (DSST-style persistence) versus path copying on the MCPST workloads.

The persistent factor tree keeps the same balanced slot layout as
`counterprovenance.mcpst.BalancedPFTLayout`.  Instead of copying every
ancestor of a changed slot (path copying, cost |Anc(Delta)|), each node owns a
small number of version-stamped modification entries; a node is copied only
when its entries are exhausted (limited node copying, Sarnak-Tarjan 1986 /
Driscoll-Sarnak-Sleator-Tarjan 1989).  The version tree is the same spanning
plan that MCPST uses; versions are linearised by preorder so that full
persistence works with enter/exit entries.

Records = internal nodes + modification entries.  Every cell root is decoded
and compared with the exact selector matrix.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "mcpst_2d"))

from counterprovenance.mcpst import (  # noqa: E402
    BalancedPFTLayout,
    build_spanning_plan,
    grid_adjacency,
    independent_pft_nodes,
    materialize_plan,
    transitions_from_states,
)
from run_experiments import make_stump_grid  # noqa: E402


@dataclass
class Entry:
    pos: int
    field: int
    value: int
    seq: int
    version: int  # creating version for enter entries, -1 for restore entries


class Node:
    __slots__ = ("fields", "entries")

    def __init__(self, left: int, right: int) -> None:
        self.fields = [left, right]
        self.entries: list[Entry] = []


class NodeCopyPFT:
    """Fully persistent balanced factor tree with limited node copying."""

    def __init__(self, layout: BalancedPFTLayout, capacity: int) -> None:
        self.layout = layout
        self.capacity = capacity
        self.nodes: list[Node] = []
        self.entries = 0
        self.copies = 0
        self.seq = 0

    # -- primitive persistence operations ---------------------------------
    def _new(self, left: int, right: int) -> int:
        self.nodes.append(Node(left, right))
        return len(self.nodes) - 1

    def read(self, x: int, field: int, pos: int) -> int:
        node = self.nodes[x]
        value = node.fields[field]
        best = None
        for entry in node.entries:
            if entry.field == field and entry.pos <= pos:
                key = (entry.pos, entry.seq)
                if best is None or key > best:
                    best = key
                    value = entry.value
        return value

    def _add_entry(self, x: int, pos: int, field: int, value: int, version: int) -> None:
        self.seq += 1
        self.nodes[x].entries.append(Entry(pos, field, value, self.seq, version))
        self.entries += 1

    def build(self, payload_refs: list[int]) -> int:
        level = list(payload_refs)
        level.extend([-1] * (self.layout.padded_leaf_count - len(level)))
        while len(level) > 1:
            level = [self._new(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        return level[0]

    def set_field(
        self,
        path: list[tuple[int, int]],
        idx: int,
        value: int,
        pre: int,
        post: int,
        version_count: int,
        root_of: list[int],
        version: int,
    ) -> None:
        x, field = path[idx][0], path[idx][1]
        node = self.nodes[x]
        for entry in node.entries:  # same-version overwrite of this version's own enter entry
            if entry.version == version and entry.pos == pre and entry.field == field:
                entry.value = value
                return
        old = self.read(x, field, pre)
        need_restore = post < version_count and not any(
            e.pos == post and e.field == field for e in node.entries
        )
        needed = 2 if need_restore else 1
        if len(node.entries) + needed <= self.capacity:
            self._add_entry(x, pre, field, value, version)
            if need_restore:
                self._add_entry(x, post, field, old, -1)
            return
        # Node is full: copy it with its values as of this version.
        self.copies += 1
        left = self.read(x, 0, pre)
        right = self.read(x, 1, pre)
        copy = self._new(left, right)
        self.nodes[copy].fields[field] = value
        if idx == 0:
            root_of[version] = copy
            return
        self.set_field(path, idx - 1, copy, pre, post, version_count, root_of, version)

    def update(
        self,
        version: int,
        root_of: list[int],
        pre: int,
        post: int,
        version_count: int,
        slot: int,
        payload_ref: int,
    ) -> None:
        height = self.layout.height
        path: list[tuple[int, int]] = []
        current = root_of[version]
        for depth in range(height):
            direction = (slot >> (height - depth - 1)) & 1
            path.append((current, direction))
            if depth + 1 < height:
                current = self.read(current, direction, pre)
        self.set_field(path, height - 1, payload_ref, pre, post, version_count, root_of, version)

    def decode(self, root: int, pos: int) -> list[int]:
        out: list[int] = []

        def visit(x: int, width: int) -> None:
            if width == 1:
                out.append(x)
                return
            visit(self.read(x, 0, pos), width // 2)
            visit(self.read(x, 1, pos), width // 2)

        visit(root, self.layout.padded_leaf_count)
        return out[: self.layout.component_count]

    @property
    def records(self) -> int:
        return len(self.nodes) + self.entries


def preorder_numbering(parents: tuple[int, ...]) -> tuple[list[int], list[int], list[int]]:
    n = len(parents)
    children: list[list[int]] = [[] for _ in range(n)]
    roots = []
    for v, p in enumerate(parents):
        if p == -1:
            roots.append(v)
        else:
            children[p].append(v)
    pre = [0] * n
    post = [0] * n
    order: list[int] = []
    counter = 0
    for r in roots:
        stack = [(r, 0)]
        while stack:
            v, state = stack.pop()
            if state == 0:
                pre[v] = counter
                counter += 1
                order.append(v)
                stack.append((v, 1))
                for c in reversed(children[v]):
                    stack.append((c, 0))
            else:
                post[v] = counter
    return pre, post, order


def payload_ref(component: int, state: int) -> int:
    return -(2 + component * 64 + int(state))  # distinct negative ref per (slot, state)


VERIFY_FULL_LIMIT = 40_000_000  # verify every cell while K*M is below this, else sample


def node_copy_materialize(states: np.ndarray, layout: BalancedPFTLayout, plan, capacity: int):
    pre, post, order = preorder_numbering(plan.parents)
    arena = NodeCopyPFT(layout, capacity)
    root_of = [0] * len(plan.parents)
    version_count = len(plan.parents)
    for v in order:
        parent = plan.parents[v]
        if parent == -1:
            root_of[v] = arena.build([payload_ref(m, states[v, m]) for m in range(layout.component_count)])
            continue
        root_of[v] = root_of[parent]
        for m in plan.parent_changes[v]:
            arena.update(v, root_of, pre[v], post[v], version_count, m, payload_ref(m, states[v, m]))
    # exactness check: decode every version (or a random sample on very large instances)
    if version_count * layout.component_count <= VERIFY_FULL_LIMIT:
        to_check = range(version_count)
    else:
        to_check = np.random.default_rng(0).choice(version_count, size=4000, replace=False).tolist()
    for v in to_check:
        decoded = arena.decode(root_of[v], pre[v])
        expected = [payload_ref(m, states[v, m]) for m in range(layout.component_count)]
        if decoded != expected:
            raise AssertionError(f"version {v} decoded incorrectly")
    return arena


NODE_BYTES = 8     # two int32 child refs (same packing as run_packed_scaling.py)
ENTRY_BYTES = 12   # int32 position + int32 field/flag + int32 value


def run_case(name: str, states: np.ndarray, adjacency, capacity: int = 2, grid=None) -> dict:
    layout = BalancedPFTLayout(states.shape[1])
    transitions = transitions_from_states(states, adjacency, layout)
    K, M = states.shape
    row = {"case": name, "M": M, "K": K, "iip_nodes": independent_pft_nodes(K, layout)}
    for objective in ("pft", "changed"):
        plan = build_spanning_plan(K, transitions, layout, objective=objective)
        t0 = time.perf_counter()
        mat = materialize_plan(states, layout, plan)
        t_path = time.perf_counter() - t0
        t0 = time.perf_counter()
        nc = node_copy_materialize(states, layout, plan, capacity)
        t_nc = time.perf_counter() - t0
        sum_c = sum(len(c) for c in plan.parent_changes)
        row[f"{objective}:sum_c_e"] = sum_c
        row[f"{objective}:path_nodes"] = mat.internal_nodes
        row[f"{objective}:nc_nodes"] = len(nc.nodes)
        row[f"{objective}:nc_entries"] = nc.entries
        row[f"{objective}:nc_records"] = nc.records
        row[f"{objective}:t_path"] = t_path
        row[f"{objective}:t_nc"] = t_nc
        row[f"{objective}:path_bytes"] = mat.internal_nodes * NODE_BYTES
        row[f"{objective}:nc_bytes"] = len(nc.nodes) * NODE_BYTES + nc.entries * ENTRY_BYTES
    if grid is not None:
        plan = SnakePlan(states, *grid)
        nc = node_copy_materialize(states, layout, plan, capacity)
        row["snake:nc_nodes"] = len(nc.nodes)
        row["snake:nc_entries"] = nc.entries
        row["snake:nc_records"] = nc.records
        row["snake:nc_bytes"] = len(nc.nodes) * NODE_BYTES + nc.entries * ENTRY_BYTES
        row["snake:sum_c_e"] = sum(len(c) for c in plan.parent_changes)
    return row


def chain_states(n: int, seed: int) -> np.ndarray:
    """Theorem-aligned 1D family: B=C=n, exactly one slot changes per boundary."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n, n), dtype=np.int64)
    current = np.zeros(n, dtype=np.int64)
    for j in range(1, n):
        slot = int(rng.integers(0, n))
        current[slot] += 1
        states[j] = current
    return states


def random_toggle_states(n_slots: int, n_versions: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = np.zeros((n_versions, n_slots), dtype=np.int64)
    current = np.zeros(n_slots, dtype=np.int64)
    for j in range(1, n_versions):
        for slot in rng.choice(n_slots, size=k, replace=False):
            current[slot] += 1
        states[j] = current
    return states


def chain_adjacency(n: int):
    return tuple((i, i + 1) for i in range(n - 1))


def snake_order(x_cells: int, y_cells: int) -> list[int]:
    order = []
    for y in range(y_cells):
        xs = range(x_cells) if y % 2 == 0 else range(x_cells - 1, -1, -1)
        order.extend(y * x_cells + x for x in xs)
    return order


class SnakePlan:
    """A Hamiltonian-path construction plan (chain of versions) over a grid."""

    def __init__(self, states: np.ndarray, x_cells: int, y_cells: int) -> None:
        order = snake_order(x_cells, y_cells)
        n = len(order)
        parents = [-1] * n
        changes: list[tuple[int, ...]] = [()] * n
        for a, b in zip(order, order[1:]):
            parents[b] = a
            changes[b] = tuple(int(i) for i in np.flatnonzero(states[a] != states[b]))
        self.parents = tuple(parents)
        self.parent_changes = tuple(changes)
        self.traversal = tuple(order)


def fmt(row: dict) -> str:
    return (
        f"{row['case']:<28} M={row['M']:>5} K={row['K']:>8} IIP={row['iip_nodes']:>11,} | "
        f"rho-MST: path={row['pft:path_nodes']:>10,} nc={row['pft:nc_records']:>9,} "
        f"(nodes {row['pft:nc_nodes']:,}+entries {row['pft:nc_entries']:,}; sum c_e={row['pft:sum_c_e']:,}) | "
        f"c-MST: path={row['changed:path_nodes']:>10,} nc={row['changed:nc_records']:>9,} "
        f"(sum c_e={row['changed:sum_c_e']:,}) | "
        f"path/nc={row['pft:path_nodes']/min(row['pft:nc_records'],row['changed:nc_records']):.2f}x "
        f"IIP/nc={row['iip_nodes']/min(row['pft:nc_records'],row['changed:nc_records']):.1f}x "
        f"| bytes path={row['pft:path_bytes']/2**20:.2f}MiB nc={row['pft:nc_bytes']/2**20:.2f}MiB"
        + (f" | snake(partial): records={row['snake:nc_records']:,} (nodes {row['snake:nc_nodes']:,}+entries {row['snake:nc_entries']:,}) "
           f"bytes={row['snake:nc_bytes']/2**20:.2f}MiB path/snake={row['pft:path_nodes']/row['snake:nc_records']:.2f}x"
           if 'snake:nc_records' in row else "")
    )


def main() -> None:
    import platform
    print(f"host={platform.node()} python={platform.python_version()} numpy={np.__version__}")
    seed = 2028
    rows = []
    print("== 1D theorem-aligned family (B=C=n, k=1) ==")
    for n in (64, 128, 256, 512, 1024, 2048):
        st = chain_states(n, seed + n)
        rows.append(run_case(f"1D chain n={n}", st, chain_adjacency(n)))
        print(fmt(rows[-1]), flush=True)
    print("== 1D random toggles (C=256, k changes per boundary) ==")
    for k in (1, 2, 4, 8, 16):
        st = random_toggle_states(256, 2000, k, seed + k)
        rows.append(run_case(f"1D toggles k={k}", st, chain_adjacency(2000)))
        print(fmt(rows[-1]), flush=True)
    print("== 2D general-position stump grids (RQ1 workload) ==")
    for M in (16, 32, 64, 128, 256, 512, 1024):
        t0 = time.perf_counter()
        w = make_stump_grid(M // 2, M - M // 2, 1, seed + M)
        rows.append(run_case(f"2D general M={M}", w.states, grid_adjacency(w.locator.x_cells, w.locator.y_cells), grid=(w.locator.x_cells, w.locator.y_cells)))
        print(fmt(rows[-1]) + f" | wall={time.perf_counter()-t0:.1f}s", flush=True)
    print("== 2D coincident thresholds (RQ2 workload, M=256) ==")
    for kappa in (1, 2, 4, 8, 16):
        g = 128 // kappa
        w = make_stump_grid(g, g, kappa, seed + 100 + kappa)
        rows.append(run_case(f"2D kappa={kappa}", w.states, grid_adjacency(w.locator.x_cells, w.locator.y_cells), grid=(w.locator.x_cells, w.locator.y_cells)))
        print(fmt(rows[-1]), flush=True)
    print("== 2D spanning-plan ablation workload (RQ3, c_e=4 everywhere) ==")
    w = make_stump_grid(8, 8, 4, seed + 200, order="y_scattered")
    rows.append(run_case("2D ablation", w.states, grid_adjacency(w.locator.x_cells, w.locator.y_cells), grid=(w.locator.x_cells, w.locator.y_cells)))
    print(fmt(rows[-1]), flush=True)
    print("== 2D random selector grids (arbitrary ON/OFF/ON, 6x5 and 30x30) ==")
    rng = np.random.default_rng(7)
    st = rng.integers(0, 4, size=(900, 64), dtype=np.int64)
    rows.append(run_case("2D random 30x30 M=64", st, grid_adjacency(30, 30), grid=(30, 30)))
    st = rng.integers(0, 3, size=(36, 16), dtype=np.int64)
    rows.append(run_case("2D random 6x6 M=16", st, grid_adjacency(6, 6), grid=(6, 6)))
    print(fmt(rows[-1]), flush=True)
    print("== capacity sensitivity on 2D general M=256 ==")
    w = make_stump_grid(128, 128, 1, seed + 256)
    for cap in (1, 2, 3, 4, 8):
        layout = BalancedPFTLayout(256)
        tr = transitions_from_states(w.states, grid_adjacency(w.locator.x_cells, w.locator.y_cells), layout)
        plan = build_spanning_plan(len(w.states), tr, layout, objective="changed")
        nc = node_copy_materialize(w.states, layout, plan, cap)
        print(f"capacity={cap}: nodes={len(nc.nodes):,} entries={nc.entries:,} copies={nc.copies:,} records={nc.records:,}")


if __name__ == "__main__":
    main()
