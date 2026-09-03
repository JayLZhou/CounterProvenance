# MCPST vs DoWhy-GCM

- DoWhy: `0.14`
- Python: `3.13.5`
- Platform: `macOS-14.6-arm64-arm-64bit-Mach-O`
- Fitting is excluded from query time for both systems.
- `DoWhy fresh` calls `counterfactual_samples` with `observed_data` for every query, so each call reruns abduction, action, and prediction.
- `DoWhy cached noise` supplies `noise_data` and skips abduction.
- Online speedup compares separate scalar queries; batch throughput is reported separately.

| Trees | Cells | DoWhy fresh μs/q | MCPST scalar μs/q | Online speedup | Cached-noise speedup | Batch throughput | Max error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 400 | 587.07 | 1.802 | 325.74x | 161.58x | 7247.79x | 0.0e+00 |
| 32 | 1,610 | 591.54 | 1.831 | 322.98x | 161.90x | 6754.14x | 0.0e+00 |
| 64 | 5,607 | 600.64 | 1.866 | 321.92x | 159.44x | 6056.93x | 0.0e+00 |

The online scalar comparison is the conservative headline. The batch number also benefits from NumPy vectorization and should not replace it in the main claim.
