# MCPST 二维实验结果

## 结论摘要

- 在 general-position stump workload 上，最大规模 $M=512$、$K=66,049$ 时，MCPST 的总记录数相对 IIP 压缩 **51.08x**。
- 若连每个 cell 的 active leaf payload 也完全复制，同一配置的压缩为 **102.16x**；前一个数字使用的是更强、更保守的 IIP。
- 对 100,000 个查询，$M=256$ 时相对 raw ensemble replay 加速 **8.37x**；IIP 与 MCPST 使用相同 point-location/root-directory，因此二者查询复杂度相同。
- 构造消融中，change-count MST 的内部节点开销是精确 PFT copy-cost MST 的 **1.83x**。
- 所有 synthetic 与 sklearn GBDT 随机点、阈值等号点均通过 exactness 校验。

## 实验环境

- Python: `3.13.5`
- NumPy: `2.1.3`
- Platform: `macOS-14.6-arm64-arm-64bit-Mach-O`
- 时间为 2 次 warm-up 后多次运行的 median；空间为结构记录数，不是 Python 对象字节数。

## General-Position Scaling

| M | K | IIP records | Full duplicate | MCPST records | vs IIP | vs full | Build (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 81 | 1,328 | 2,592 | 448 | 2.964 | 5.786 | 0.001 |
| 32 | 289 | 9,312 | 18,496 | 1,824 | 5.105 | 10.14 | 0.004 |
| 64 | 1,089 | 69,824 | 139,392 | 7,808 | 8.943 | 17.85 | 0.012 |
| 128 | 4,225 | 541,056 | 1,081,600 | 34,176 | 15.83 | 31.65 | 0.115 |
| 256 | 16,641 | 4,260,608 | 8,520,192 | 150,528 | 28.30 | 56.60 | 0.387 |
| 512 | 66,049 | 33,818,112 | 67,634,176 | 662,016 | 51.08 | 102.16 | 1.537 |

这里每个 cell boundary 仅改变一个 component，实测内部节点数严格等于 $b+(K-1)\log_2 M$，不是拟合曲线。两种方法都计入共享 payload 和每-cell root pointer。

## Fully Duplicated Payload Sensitivity

| Payload nodes q | Full duplicate | MCPST | Compression |
|---:|---:|---:|---:|
| 1 | 67,634,176 | 662,016 | 102.16 |
| 4 | 169,085,440 | 665,088 | 254.23 |
| 16 | 574,890,496 | 677,376 | 848.70 |

这里 q 是一个 active tree-leaf provenance payload 的平均 circuit-node 数。q=1 对应当前 leaf-ID 原型；q>1 是 final provenance payload 的参数化空间预测，不是尚未实现的实测字节数。
在 general-position、$q=1$、$M$ 为 2 的幂且 $K$ 足够大时，strong IIP 与 full-duplicate 的压缩率分别近似 $M/(\log_2 M+1)$ 和 $2M/(\log_2 M+1)$。最坏情况下若每条边几乎改变全部 components，MCPST 会退化到与 IIP 同阶，因此不存在对所有模型都固定的压缩倍数。

## Coincident Thresholds

| κ | M | K | MCPST nodes | Compression |
|---:|---:|---:|---:|---:|
| 1 | 256 | 16,641 | 133,375 | 28.30 |
| 2 | 256 | 4,225 | 34,047 | 27.90 |
| 4 | 256 | 1,089 | 10,047 | 23.98 |
| 8 | 256 | 289 | 3,711 | 16.51 |
| 16 | 256 | 81 | 1,775 | 8.973 |

本组固定 $M=256$，将多个 components 合并到同一 threshold，使 κ 增加且 cell 数相应减少，用来检验 copy-cost 对 coincident selectors 的敏感性。

## Spanning-Plan Ablation

| Strategy | Internal nodes | vs IIP | vs MCPST cost |
|---:|---:|---:|---:|
| MCPST (exact PFT weight) | 695 | 5.876 | 1.000 |
| Change-count MST | 1,271 | 3.589 | 1.829 |
| Unweighted MST | 1,271 | 3.589 | 1.829 |

该 workload 让 horizontal boundaries 改变连续 leaves，vertical boundaries 改变分散 leaves；两者 $c_e$ 相同但 $\rho_e$ 不同。因此只看 changed count 无法选出最省 path-copy 的 construction tree。

## Query

| M | Queries | Replay μs/q | Index μs/q | Speedup |
|---:|---:|---:|---:|---:|
| 32 | 100,000 | 0.100 | 0.059 | 1.701 |
| 64 | 100,000 | 0.174 | 0.090 | 1.935 |
| 128 | 100,000 | 0.429 | 0.114 | 3.775 |
| 256 | 100,000 | 0.889 | 0.106 | 8.372 |

MCPST 的查询为 predecessor point location 后直接读取 root/value。它加速的是从头 replay；相对已完整物化所有 cells 的 IIP，查询时间预期持平，优势在空间与构建写入量。

## sklearn GBDT Sanity Check

| Trees | Global-grid K | Zero-change edges | Compression | Query speedup | Max error |
|---:|---:|---:|---:|---:|---:|
| 16 | 442 | 181 | 4.491 | 2.069 | 0.000 |
| 32 | 1,550 | 596 | 7.379 | 3.549 | 0.000 |
| 64 | 4,956 | 2,617 | 14.46 | 5.685 | 0.000 |

这里使用所有 sklearn split thresholds 的 global-grid refinement。它是 exact 但不保证 cell decomposition 最小，因此这是端到端正确性与趋势验证，不应被写成最优 partition 结果。

## DoWhy-GCM Comparison

完整 abduction-action-prediction 重算使用独立脚本运行；见 `DOWHY_RESULTS.md` 和 `dowhy_query.csv`。主表应使用 online scalar speedup，不要用包含 NumPy vectorization 的 batch throughput 代替。

## Packed Million-Cell Scaling

去除 Python object overhead 的 packed general-position builder 使用独立脚本；见 `PACKED_RESULTS.md` 和 `packed_scaling.csv`。该实验实际分配 MCPST arrays，并在同一字宽下精确计算 strong IIP 与 fully duplicated baseline bytes。

## 解释边界

1. 记录数比较采用相同 balanced PFT layout，并让 IIP 也共享 payload，因此不是靠弱化 baseline 得到压缩率。
2. MCPST 的最优性只针对 adjacency-constrained、single-parent、fixed-layout、无跨分支全局 internal-node hash-consing 的构造模型。
3. 本实验验证二维 exact tree ensembles；高维 cell enumeration 仍是后续工作。
