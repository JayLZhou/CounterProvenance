# Runs of the final index (2026-11-22 to 2026-11-24)

The paper's experiments with the final CellForest index: leaves stored as a log along the MST with checkpoints
(`LogCellIndex`), two-level grids for the groups of one tree (`twolevel.py`), and one group only while blocks x trees <= 2e10.
The baselines (ThreeStep, ChiRho, DoWhy, CausalAI) and the applications' own evaluations were not rerun: their results are the
paper's earlier runs. Every batch ran in a systemd user scope with `MemoryMax=1000G` (or a share of it) and `OOMPolicy=continue`,
with `memguard2.sh` as a second guard; paths refer to the experiment server.

| Script | What it runs |
|---|---|
| `start_final.sh`, `build_sched.py` | rebuild of the 75 indexes, builds scheduled by an estimate of their memory |
| `run_final.sh`, `run_final_b.sh` | Exp.1-3 (main), Exp.4 (trees), Exp.7 (combinations), Exp.5 (beta); `_b`: timings again after the two-level layout was restored |
| `chain_final.sh`, `run_final2.sh`, `run_final3.sh` | Exp.6 (`final_structure.py`) and Exp.7; applications, one-thread build time and depth |
| `beta2e8_run.sh`, `apps2_run.sh`, `paths_run.sh`, `nogroup_final.sh` | reruns: beta 2e8, the applications with mediators (`--sensitive` / `--actionable`), MINT mediators, indexes without grouping (Figure 19) |
| `param_sched.py` | appendix: one-group limit and pre-merge cap |
| `setup_final2.sh`, `depth_patch.py`, `depth_final.py`, `depth_driver.py` | the depth experiment on the final index (from `depth_exp3`; ThreeStep times reused) |
| `paper_final.py` (made by `mk_paper_final.py`), `e4.py`, `e5.py`, `sum67.py`, `twin_json.py`, `size_final.py`, `parts_final2.py`, `cmp3.py`, `ab3.sh` | summaries: Exp.1-5 numbers, Exp.6/7, figure data, sizes (the log counted once: 12 bytes per record subtracted from runs before commit 4e48a75), size by part, comparisons with the earlier indexes |
