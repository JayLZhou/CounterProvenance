# MCPST 2D Experiments

This directory evaluates the minimum-change persistent factor-tree index under
the exact physical model used by the paper design.

Run the full benchmark from the repository root:

```bash
pip install -r requirements-mcpst.txt
python experiments/mcpst_2d/run_experiments.py
```

Fast correctness/debug run:

```bash
python experiments/mcpst_2d/run_experiments.py --quick
```

Scale a packed implicit general-position index to more than one million cells:

```bash
python experiments/mcpst_2d/run_packed_scaling.py
```

Persist the million-cell `M=2048` structure as memory-mappable arrays:

```bash
python experiments/mcpst_2d/run_packed_scaling.py \
  --sizes 2048 \
  --persist-size 2048
```

The artifact is written to `artifacts/packed_m2048/` and ignored by Git because
its packed arrays occupy about 100 MiB.

Compare online MCPST queries with complete DoWhy-GCM abduction-action-
prediction recomputation:

```bash
pip install -r requirements-dowhy.txt
python experiments/mcpst_2d/run_dowhy_benchmark.py
```

The script writes raw CSV files, one combined PNG/PDF figure, and a Chinese
`RESULTS.md` interpretation to `experiments/mcpst_2d/results/`.
The DoWhy benchmark additionally writes `dowhy_query.csv` and
`DOWHY_RESULTS.md`.
The packed benchmark writes `packed_scaling.csv`, `PACKED_RESULTS.md`, and a
PNG/PDF figure.

The comparison is deliberately conservative: IIP and MCPST share the same
payload records, use the same balanced factor-tree layout, store the same
per-cell root directory, and use the same point locator. The storage difference
is only independent root materialization versus MCPST path copying.
`RESULTS.md` also reports a genuinely fully duplicated baseline that copies
every active leaf payload in every cell.
