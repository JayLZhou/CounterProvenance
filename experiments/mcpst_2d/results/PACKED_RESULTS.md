# Packed MCPST Scaling

- Python: `3.13.5`
- NumPy: `2.1.3`
- Numba: `0.61.0`
- MCPST arrays are actually allocated and populated.
- Baseline bytes are exact counts under the same packed schema; very large baselines are not allocated to avoid exhausting machine memory.
- Schema: each internal node has two int32 refs; each MCPST root is int32; cached values and thresholds are float64; each q=1 payload is two int32 fields.
- Strong IIP is allowed to derive each independent root offset implicitly and therefore is not charged a root directory.
- This is a general-position specialized builder: its snake path is already an optimal spanning tree because every transition has the same one-component cost. Build time excludes cell discovery, graph construction, and generic MST planning.

| M | Cells K | MCPST actual | Strong IIP | Full duplicate q=1 | vs IIP | vs full | Build (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16,641 | 1.21 MiB | 32.51 MiB | 65.01 MiB | 26.78x | 53.55x | 0.000 |
| 512 | 66,049 | 5.31 MiB | 258.02 MiB | 516.01 MiB | 48.62x | 97.24x | 0.002 |
| 1,024 | 263,169 | 23.12 MiB | 2.01 GiB | 4.02 GiB | 88.92x | 177.85x | 0.007 |
| 2,048 | 1,050,625 | 100.26 MiB | 16.03 GiB | 32.06 GiB | 163.74x | 327.48x | 0.032 |

Every row verifies 4,096 random (cell, component) payload lookups against the closed-form stump state. The packed builder excludes one-time Numba JIT time.
