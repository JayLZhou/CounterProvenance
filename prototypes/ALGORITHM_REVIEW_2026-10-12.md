# CounterProvenance 代码理解、1D/2D 压缩极限与多维推广分析

> 日期：2026-10-12
> 对象：`https://github.com/JayLZhou/CounterProvenance`（commit `f68821a`）与本目录论文稿
> 配套原型：`review_prototype/nodecopy_proto.py`（结果见 `review_prototype/nodecopy_results.txt`，
> 由 hit server3 = `yujia-server3`（3haoji-A100）上的 `hammer` conda 环境运行）
> 服务器工作区：`/data1/yujia/Yingli/CounterProvenance_review/`（仓库副本 + 脚本 + 日志）
> 运行：`/data1/yujia/envs/hammer/bin/python nodecopy_proto.py <repo path>`

## 0. 结论先行

1. **代码现状。** 仓库实现了三条互不相连的线：(a) 1D 线性版本序列上的
   Versioned Provenance Closure（`versioned.py`：fat-node 式 Compact 布局 +
   path-copying 式 VSC 布局）；(b) 1D 静态 stabbing 索引（`tree.py`：
   interval tree + prefix/suffix 聚合 + fractional cascading）；(c) 2D MCPST
   （`mcpst.py`：全阈值笛卡尔网格 + 4 邻接 + PFT path copying + $\rho_e$-MST）。
   论文正文第 3–5 节只覆盖 (a)，第 2 节的 $d$ 维 cell 叙述对应 (c) 的设计文档。
2. **1D 还能压吗？** 空间上 Compact 的 $\Theta(N)$ word 已是该模型下界，
   不能再降（$N$ = 所有树在干预线上的 leaf-interval 总数）。但有两处可以实质提升：
   * **Materialized 布局从 $\Theta(N\log M)$ 降到 $\Theta(N)$**：把 path copying
     换成 limited **node copying**（Sarnak–Tarjan 1986 / DSST 1989）。每次有效
     slot 变化摊还 $O(1)$ 条记录，遍历每个节点 $O(1)$ 开销，仍然是“每个 region
     一个可直接遍历的 root”。实测 sparse-change family 上比 VSC 少 3.3x（$n=64$）
     到 5.5x（$n=2048$）记录，且随 $n$ 增大而扩大（对应 $\log n$ 因子）。论文里 IIP $\Theta(n^2)$ /
     VSC $\Theta(n\log n)$ / Compact $\Theta(n)$ 的三分离，中间项可以变成 $\Theta(n)$。
   * **Compact 的完整展开从 $O(M\log B)$ 降到 $O(M)$**：Compact 的 per-slot
     history 本质是 change sequence 的 inverted list，`root_at` 就是 rank 查询；
     用 wavelet tree 替换后单 slot $O(\log C)$、全部 slot 一次 DFS $O(C)$，
     history 空间从 $K_\Delta$ words 降到 $K_\Delta\log C$ bits。
3. **2D 还能压吗？** 能，而且结论相同：沿同一棵 MST 版本树做 **full-persistence
   node copying**，每条边代价从 $\rho_e=|\mathrm{Anc}(\Delta_e)|\approx c_e(1+\log\frac{M}{c_e})$
   降为约 $c_e$ 量级。**修正（联合审稿后，见 §8）**：这个 $O(c_e)$ 摊还界只对链状
   计划（partial persistence，Sarnak–Tarjan）成立；一般版本树上我原型的 enter/exit
   复制方案代价依赖树形，星形树上会退化到 path copying 量级，严格的 $O(1)$ 摊还需要
   DSST 的 node splitting（未实现）。因此 general position 下 $O(P+K+M)$ 对 snake 链
   成立（网格总有 Hamiltonian 链），对一般 MST 树只是实测结论。
   $\sum_T c_e\ge K-1$ 是 single-parent、Hamming 更新代价模型内所有方案的下界，
   所以 general position 下邻接生成树在该模型内全局最优（回答了设计文档 §18.1 的
   担忧，但不推广到实际记录数或其他表示）。实测
   $M=512$：path copying 594,943 节点 → node copying 197,885 记录（MST 版本树）
   或 99,364 记录（snake 路径），3.0x / 6.0x；$M=1024$（$K=263{,}169$）：
   2,632,703 → 788,989 / 395,477，3.3x / 6.7x。但要诚实指出：当一条边同时变化的
   slot 很多且在 PFT 布局上连续（$\kappa\ge 4$），path copying 的 ancestor union
   每个变化只需 ~1.2 节点，常数反而更好；两者都是 $\Theta(c_e)$，逐边取
   $\min$ 的混合方案同时支配两者（实测见 §4.6）。**MST 保留**：变的只是边权
   $\min\{\rho_e,\alpha c_e\}$；在边代价异质的 RQ3 workload 上混合权 MST 比任意
   生成树省 1.57x，并同时优于两种单一机制。
4. **剩下的 $\Omega(K)$ 不是信息论下界。** cell 方法必须给每个 cell 一个 root
   指针，所以 $K$ 项无法避免；但 $K$ 个 cell 由 $N$ 个 leaf box 决定，信息量只有
   $O(N)$。多维的"更通用算法"因此不是继续优化 cell 图，而是换成**不枚举 cell**
   的 semigroup point-enclosure 索引：$d$ 层 segment tree，最后一层就是
   `tree.py` 的 interval tree。空间 $O(N\log^{d-1}N)$，查询返回 $O(\log^{d}N)$ 个
   hash-consed 片段（$d=2$ 时 $O(N\log N)$ 空间、$O(\log^2 N)$ 片段），与 $K$ 无关。
   Chazelle (1990) 的 arithmetic-model 下界（经 dominance→ensemble 的归约引理，待写）
   说明：在“piece 为预物化聚合、查询只做 $\le t$ 次组合”的模型里，近线性空间下
   polylog 片段不可避免，单片段需要 $N^{1+\Omega(1)}$ 空间（指数常数依赖原定理，
   **不能写成 $N^2$**）。所以“每 cell 一个 root”的方案在高维要付多项式级空间，
   但“与 $\Theta(K)$ 精确匹配”不是已证结论。
   推荐架构：K-free 索引做基底，热 cell 的 root 按需 node-copy 物化并缓存。
   实测（§5.2.1，server3，全部精确）：2D $M=2048$ 时 K-free 索引 $8.1\times10^4$
   条记录 vs cell 方案（即使换成 node copying）$3.7\times10^6$；3D/4D 下
   $K_{grid}\ge10^9$ 时 cell 方案不可行而 K-free 仍是 $10^5$ 量级、每查询 20–70 个片段。
5. **对论文定理的影响**：第 6 节逐条列出哪些加强、哪些需改写、哪些 claim 会失效。
   §8 是与 Codex 的两轮联合审稿及其导致的降级；§9 是三项加强的实测：真实数据
   （California housing，GBR/HGB 各 300 棵）上 cell 方案只在 $d=2$ 可行，K-free 与
   统一索引在 $d=8$ 仍是 $10^5$ 级记录且全部精确；持久化末维变体实现并验证；
   对比式归因作为 provenance 的应用用例。
   最需要注意的是 RQ3（$\rho_e$-MST 比 $c_e$-MST 省 45%）在 node copying 下不再
   成立，“exact PFT copy cost 是算法内容”这一差异化论点会弱化。

---

## 1. 代码理解

### 1.1 `linear.py`（Class L）

展开线性 SCM 的 source-to-target 单项式，反向 trie 共享后缀，编译成向量化线性
形式并与 DoWhy-GCM 对拍。与压缩问题无关，是一个 region 的特例。

### 1.2 `versioned.py`（1D Versioned Provenance Closure，两种布局）

* `CircuitArena`：hash-consed 半环电路（`zero/one/atom/add/mul/switch`），
  `add` 不做幂等去重，children 排序只为交换律规范化。
* `CircuitTemplate` + `VersionedSlotEnvironment` = **Compact 布局**。每个 slot
  一个 `(versions[], values[])`，`append_version` 只记有效变化，`root_at`
  用 `bisect_right` 做前驱查找。这正是 DSST 的 **fat node** 方法。
  空间 $\Theta(C+K_\Delta)$；单 slot $O(\log(h_s+1))$；`CircuitClosure.materialize`
  / `evaluate` 对每个 slot 各做一次 bisect，完整解析 $O(|T|+\sum_s\log(h_s+1))$，
  最坏 $O(C\log B)$。
* `CircuitVersionBuilder` = **VSC-Materialized 布局**。按线性版本顺序，对变化
  slot 的 inclusive ancestor closure 自底向上重新 intern，这是 **path copying**。
  balanced 模板上每次更新 $O(k\log(1+C/k))$ 个新节点；全局 hash-consing 可能
  复用。版本只能是线性序列（partial persistence），没有版本树。
* 两个目录类 `VersionedClosureIndex` / `VersionedSemiringIndex`：按 `starts`
  二分定位 region，$O(\log B_u)$。

### 1.3 `tree.py`（1D 静态 stabbing 索引）

输入每棵树在干预线上的 leaf intervals（$N$ 个，按树构成划分）。
centered interval tree：每个 interval 挂在唯一节点；节点按 left 排序存 prefix
聚合 root、按 right 排序存 suffix 聚合 root（只用 $\oplus$，不需减法）；
catalog 用“子节点 catalog 每隔一个采样”做 fractional cascading，根上一次
二分、以下每层 $O(1)$。

* 空间 $O(N)$ 记录（interval、树节点、catalog、rank、aggregate、cascade link
  各 $O(N)$），arena 新增 $O(N)$ 个 `add` 节点。
* 查询 $O(\log N)$，返回 $O(\log N)$ 个 factor roots（`TreeStabbingClosure`），
  `materialize()` 再合成一个 root（每次 $O(\log N)$ 个新 arena 节点，未缓存）。
* 它与 `versioned.py` 彼此独立，没有共用“version/closure”接口。

### 1.4 `mcpst.py`（2D MCPST）

* `GridPointLocator`：**全阈值笛卡尔网格**，$K=(B_x+1)(B_y+1)$，两次
  `searchsorted`，等号归左/下 cell。不是论文第 2 节强调的 exact non-Cartesian
  overlay；`RESULTS.md` 也承认“global-threshold grid 不保证最小 cell 分解”。
  sklearn 深度 2 GBDT 上 36–53% 的邻接边是 zero-change，就是网格多切出来的。
* `transitions_from_states`：对每条邻接边比较完整 $K\times M$ 状态矩阵，
  构建时间和内存都是 $\Theta(KM)$，与 IIP 同阶。设计文档 §5.3 说 $\Delta_e$
  应由 boundary label 直接产生，代码没有做到；这也是为什么百万 cell 实验要
  单独写 `run_packed_scaling.py`（snake 路径、隐式状态）。
* `BalancedPFTLayout.transition_cost` = $|\mathrm{Anc}(\Delta)|$（逐层取唯一
  parent）；`PFTArena.batch_update` 按 changed positions 的区间划分递归复制，
  实际分配数与预测逐边相等（测试覆盖）。
* `build_spanning_plan`：source-augmented Kruskal，候选边 $|E|+K$，三种目标
  （`pft`/`changed`/`unweighted`），之后 BFS 生成 rooted plan。
* `materialize_plan`：沿 plan 做 path copying；payload 单独共享，不计入 PFT 节点。
* 查询：两次二分 + 数组读 root/值，$O(\log B_x+\log B_y)+O(1)$。

### 1.5 实验口径（已核对）

RQ1 的 $(M-1)+(K-1)\log_2 M$ 与我复算完全一致；IIP 记录 = payload + K + K·b。
40 个单元测试全部通过。

---

## 2. 统一抽象

固定 unit 后，Class T 的索引问题是：

* $M$ 个 slot（component tree），$N=\sum_m b_m$ 个 leaf box（1D 时是 interval），
  overlay 得到 $K$ 个 cell，cell $q$ 的签名 $\sigma(q)\in\prod_m\mathcal L_m$；
* 相邻 cell 的变化集 $\Delta_e$，$c_e=|\Delta_e|$，构造树 $T$（1D 是链，2D 是 MST）；
* 查询：$\mathbf t\mapsto$ cell $\mapsto$（数值，provenance root），root 展开应恰好
  得到 $M$ 个 payload 的 multiset（$\oplus$ 可交换、不幂等、无逆元）。

四种物理布局正好对应四种 persistence 策略：

| 布局 | persistence 策略 | 空间 | root 访问 | 完整展开 |
|---|---|---|---|---|
| IIP | 不共享 | $\Theta(KM)$ | $O(1)$ | $O(M)$ |
| Compact | fat node | $\Theta(M+K+\sum_T c_e)$ | $O(1)$ handle | $O(M\log B)$（代码）→ $O(M)$（wavelet tree） |
| VSC / MCPST | path copying | $O(M+K+\sum_T c_e(1+\log\frac{M}{c_e}))$ | $O(1)$ | $O(M)$ |
| **node copying**（缺失） | limited node copying | $\Theta(M+K+\sum_T c_e)$ | $O(1)$ | $O(M)$，每节点 $O(1)$ |

论文把 Compact 与 VSC 描述为 space–latency Pareto 两端；node copying 说明在
“允许节点带版本戳”的模型下，这个 Pareto 基本坍缩成一个点。DSST 论文本身就证明
fat node（Compact）与 node copying 是同一问题的两种解。

---

## 3. 1D：还能压多少

### 3.1 下界

Compact 的 $\Theta(D_T+C+B+K_\Delta+H)$ 是**表示模型内**的线性界：在显式存储
boundary 与每条有效更新、且查询不回读模型的表示下，它等于显式输入大小（§8 A7：
不能称信息论或 word-RAM 下界，因为模型本身可重算一切）。Class T 中
$K_\Delta=\sum_m(b_m-1)=N-M$ 与阈值是否重合无关，所以 1D 索引在该表示模型内就是
$\Theta(N)$，**在此模型内不能再降**。实数 boundary 上的前驱查找 $O(\log B)$ 也无法
在比较模型下改进。

### 3.2 可以改进的三处

**(a) Materialized 布局：path copying → node copying。**
balanced 模板是 in-degree 1 的树，每个内部节点带一个（partial persistence）或常数
个（full persistence）modification entry `(version, field, value)`；entry 用完才
复制节点并向父节点递归。摊还每次 slot 变化 $O(1)$ 记录，访问时每节点一次比较。
结果是：$\Theta(N)$ 空间 **且** 每个 region 有可直接遍历的 root，展开 $O(M)$。

实测（`review_prototype`，记录 = 内部节点 + entry；path copying 记录 = 节点）：

| B=C=n, k=1 | IIP | VSC path copying | node copying | VSC/nc |
|---:|---:|---:|---:|---:|
| 64 | 4,032 | 441 | 135 | 3.27x |
| 128 | 16,256 | 1,016 | 276 | 3.68x |
| 256 | 65,280 | 2,295 | 553 | 4.15x |
| 512 | 261,632 | 5,110 | 1,103 | 4.63x |
| 1024 | 1,047,552 | 11,253 | 2,223 | 5.06x |
| 2048 | 4,192,256 | 24,564 | 4,442 | 5.53x |

| C=256, B=2000, k/boundary | path copying | node copying | 比值 |
|---:|---:|---:|---:|
| 1 | 16,247 | 3,126 | 5.20x |
| 2 | 28,274 | 6,122 | 4.62x |
| 4 | 48,972 | 12,109 | 4.04x |
| 8 | 82,729 | 24,117 | 3.43x |
| 16 | 135,743 | 48,104 | 2.82x |

node copying 每次变化约 1.5 条记录，与 $k$ 无关；path copying 为 $\log(1+C/k)$ 级。

与论文模型的关系：node-copy entry 等价于一个 selector 为“version $\ge v$”的
`SWITCH` 节点，arena 已有 `switch` 和 `selectors` 参数，语义上兼容；代价是这些
节点不再是 canonical immutable tuple，跨 unit / 非相邻 region 的 hash-consing
只能作用于不变子树。

**(b) Compact 布局的解析代价：bisect → rank。**
把所有有效变化按 region 顺序拼成序列 $S\in[C]^{K_\Delta}$，再记录每个 region 的
前缀长度。slot $s$ 在 region $j$ 的值 = `values_s[rank_s(S, P_j)]`。wavelet tree
给出 $O(\log C)$ 单 slot 查询、$O(C)$ 一次得到全部 rank（遍历 $2C-1$ 个 wavelet
节点各做一次 $O(1)$ rank），位置部分只占 $K_\Delta\log C(1+o(1))$ bits。
Class T 下这恰好是“tree $m$ 在 cell $j$ 选第几个 leaf”；Class N 下 gate 状态 =
初值 XOR rank 的奇偶。values 表仍是 $N$ 个 payload 指针，所以 word 级别仍是
$\Theta(N)$，提升的是展开时间与 bit 级空间。论文已声明不主张 bit-optimal，
这一条可作为实现优化而非定理。

**(c) 不可变 canonical root 的真正代价尚未刻画。**
Theorem“path-copying bound 紧”只对“固定模板 + path copying”成立，不是所有不可变
DAG 的下界：单调（staircase）family 用 prefix 链 + suffix 链就能以 $O(N)$ 不可变
节点给每个 region 一个 root。任意交错下不可变单 root 的最优空间是开放的；
node copying 用版本戳绕开了这个问题，论文若保留“canonical materialized model”，
应把它明确为可选模型而非主结论。

---

## 4. 2D：还能压多少

### 4.1 path copying → full-persistence node copying

版本树就是 MST 构造树。按 preorder 编号后，每次变化写入 enter entry
`(pre[v], new)` 与 exit entry `(post[v], old)`，节点满则复制并递归到父节点
（DSST full persistence 在静态版本树上的特例）。每条边摊还 $O(c_e)$ 记录。

| general position, $M$ | $K$ | IIP 节点 | path copying ($\rho_e$-MST) | node copy（MST 版本树） | node copy（snake 链） | path/snake |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 81 | 1,215 | 335 | 229 | 129 | 2.60x |
| 32 | 289 | 8,959 | 1,471 | 845 | 445 | 3.31x |
| 64 | 1,089 | 68,607 | 6,591 | 3,229 | 1,667 | 3.95x |
| 128 | 4,225 | 536,575 | 29,695 | 12,605 | 6,388 | 4.65x |
| 256 | 16,641 | 4,243,455 | 133,375 | 49,789 | 25,121 | 5.31x |
| 512 | 66,049 | 33,751,039 | 594,943 | 197,885 | 99,364 | 5.99x |
| 1024 | 263,169 | 269,221,887 | 2,632,703 | 788,989 | 395,477 | 6.66x |

按 `run_packed_scaling.py` 的字宽（节点 8 B，entry 12 B）：$M=512$ 内部结构
4.54 MiB → 2.01 MiB（MST）/ 1.01 MiB（snake）；$M=1024$：20.09 MiB → 8.02 MiB /
4.02 MiB。外推 $M=2048$（高度 11）：92 MiB 内部节点 → 约 17 MiB，整体 100 MiB
→ 约 30 MiB；这一行是外推，不是实测。所有实测均在 hit server3 上完成，
$M=1024$ 与 $n\ge1024$ 的正确性检查为 4,000 个随机 cell 抽样解码，其余为全量解码。

两点细节：

* **版本树形状开始有影响。** path copying 只看边权，node copying 的 full
  persistence 每次变化多写一条 exit entry，链（snake）只需 enter entry，所以 snake
  比 MST 版本树再省约 2x。网格有 Hamiltonian path；一般 planar dual graph 不保证，
  但 DFS 序 + exit entry 总是可用。
* **capacity $\ge 2$ 即可**，1 会退化成 path copying（132,606 ≈ 原值），
  再大收益可忽略（实测 49,789 → 49,186）。

### 4.2 什么时候 path copying 反而更好

| $M=256$, $\kappa$ | $K$ | path copying | node copy（MST） | node copy（snake） |
|---:|---:|---:|---:|---:|
| 1 | 16,641 | 133,375 | 49,789 | 25,121 |
| 2 | 4,225 | 34,047 | 25,341 | 12,780 |
| 4 | 1,089 | 10,047 | 13,117 | 6,676 |
| 8 | 289 | 3,711 | 7,005 | 3,571 |
| 16 | 81 | 1,775 | 3,949 | 2,087 |

RQ2 的重合 slot 是连续编号，16 个连续 leaf 的 ancestor union 只有 19 个节点
（每变化 1.2 节点），而 node copying 每变化 1.5–3 条记录。RQ3 的 ablation
workload（$c_e=4$）同样是 695 vs 911/521；随机密集变化网格（$c_e\approx 9/16$）
path copying 也更好。理论上 $\rho_e\in[c_e/2,\;c_e(1+\log\frac{M}{c_e})]$，两者
都是 $\Theta(c_e)$，差别只在常数与 $\log$ 因子出现的区域（稀疏变化）。

**建议的最终算法是逐边混合**：边权取
$w_e=\min\{\rho_e,\;\alpha c_e\}$（$\alpha\approx 2$–$3$），物化时对该边选对应策略。
path copying 产生的新节点 entry 为空，与 node copying 可以自由混用，正确性不受
影响。这样既保留 RQ2/RQ3 的优势，又在 general position 去掉 $\log M$。

### 4.3 general position 下的全局最优性

任意两个不同 cell 至少一个 slot 不同，所以任何 single-parent 方案都有
$\sum_T c_e\ge K-1$；general position 下每条邻接边 $c_e=1$，邻接图上任意生成树都
达到这个下界。因此在 **single-parent、Hamming 更新代价（每次 slot 变化计 1）** 的
模型内，**MCPST 不只是 adjacency-constrained optimal，而是全局最优**，不需要
MST（snake 即可，`run_packed_scaling.py` 已这么做）。这个引理不涉及实际记录数、
root 目录、点定位或任何 DAG 因子化表示；超出该模型就不成立。重合阈值时全局最优与邻接
MST 的差距 $\le\kappa$。这比设计文档 §18.1 现在的保守表述强。

### 4.4 exact overlay 替代笛卡尔网格

深度 $\ge2$ 的树上，网格把条件阈值延伸到整个域。sklearn 64 树：$K_{grid}=4956$，
zero-change 边 2617 条，说明可合并的 cell 很多。做法：对所有 reachable leaf box
的边做扫描线，产生 cell、邻接与 boundary label（$\Delta_e$ 直接来自产生该边的树），
点定位用 Sarnak–Tarjan persistent search tree（$O(K)$ 空间、$O(\log K)$ 查询，
和 node copying 是同一套 persistence 机制）。同时把构建从 $\Theta(KM)$ 降到
$O(N\log N+K)$，消除现在 `transitions_from_states` 的稠密矩阵。

### 4.5 MST 的角色（MST 要留着，变的只是边权）

上面所有改动都**不去掉 source-augmented MST**：它仍然是在 boundary-labeled
version graph 上选择 single-parent 构造方案的步骤，Lemma 4 / Theorem 1 的证明只
用到“每条边的代价可加、且不依赖其他边”，对任何边权函数都成立。变化的只有
边权本身：

| 物化机制 | 边权 $w_e$ | 性质 |
|---|---|---|
| path copying（现状） | $\rho_e=\lvert\mathrm{Anc}(\Delta_e)\rvert$ | 精确、确定、与其他边无关 → MST 的模型内最优性成立 |
| node copying | 链上 $\approx 1.5c_e$（已证 $O(1)$ 摊还）；一般版本树上依赖树形，星形树最坏约 $3c_e\log M$ | 不是独立可加的边权 |
| **逐边混合（建议）** | $\min\{\rho_e,\ \alpha c_e\}$ | **surrogate**：MST 对 surrogate 最优，不对实际分配最优 |

**修正（§8）**：node copying 的代价依赖最终版本树的形状与节点 entry 占用情况，
不能写成选边前固定的可加边权，所以“混合权 MST 模型内最优”不成立；exact
model-optimality 只保留给 immutable path copying（Lemma 4 / Theorem 1 原样成立）。
混合方案应表述为“MST 最小化一个静态 surrogate 上界，实测上等于或优于两种单一
机制”。要恢复一般版本树上的 $O(1)$ 摊还，需要 DSST 的 node splitting 与版本序
维护，这是一个独立的、尚未实现的数据结构工作。

**MST 什么时候真正起作用。** 当且仅当 $c_e$（或 $\rho_e$）在邻接图上不是常数：
阈值重合（histogram GBDT、量化特征、重复阈值）、deep tree 的 exact overlay
（不同 face 由不同数量的树共享）、以及 cell 合并后不再是网格的邻接图。general
position 下每条边代价相同，任何生成树都是 $K-1$，MST 退化为任意生成树；这不是
损失而是 §4.3 的全局最优引理，`run_packed_scaling.py` 用 snake 路径时已经默认了
这一点。

**关于 novelty。** source-augmented MST 做版本存储来自 dataset versioning
（Bhattacherjee et al., VLDB 2015），设计文档 §17.5 已明确不 claim MST 本身。
可以 claim、而且在新代价模型下更强的是：
(i) 版本图与边标签 $\Delta_e$ 由反事实几何编译产生，而非外部给定的数据版本；
(ii) 边权是**两种 persistence 机制之间可证明的逐边选择**（精确 ancestor-union
代价 vs 摊还 node-copy 代价），不是黑盒 delta 大小，这保留了 RQ3 “边权定义是算法
内容”的论点，并把 $\rho_e$-vs-$c_e$ 的 1.83x 升级为“混合权 vs 单一机制 vs 任意
生成树”的三方对照；
(iii) general-position 全局最优引理与 $\kappa$ 间隙界；
(iv) 与 K-free 索引的分层组合（第 5 节）：MST/cell 层是低延迟物化层，K-free 层
保证空间与 $K$ 无关。

实测对照见 §4.6。

### 4.6 混合权与生成树的实测（`review_prototype/hybrid_proto.py`，hit server3）

把两个问题分开测：(a) 用哪种物化机制；(b) 沿哪棵生成树。列含义：
`path@ρ` = 现状（path copying 沿 $\rho_e$-MST）；`node@c` = node copying 沿
$c_e$-MST；`hyb@X` = 逐边混合物化（$\alpha=2.5$）沿 X 树，X ∈ {ρ-MST, c-MST,
混合权 MST, 任意生成树(unweighted Kruskal)}；`snake` = 沿 Hamiltonian 链
（partial persistence，无 exit entry）。所有版本均逐 cell 解码验证。

| workload | path@ρ | node@c | hyb@ρ-MST | hyb@混合MST | hyb@任意树 | 混合MST 选边 (path/node) | snake: path / node / hyb |
|---|---:|---:|---:|---:|---:|---|---|
| 2D general $M=512$ | 594,943 | 197,885 | 197,885 | 197,885 | 197,885 | 0 / 66,048 | 594,943 / 99,364 / 99,364 |
| 2D $\kappa=2$ | 34,047 | 25,341 | 25,341 | 25,341 | 25,341 | 0 / 4,224 | 34,047 / 12,780 / 12,780 |
| 2D $\kappa=4$ | 10,047 | 13,117 | 10,047 | 10,047 | 10,047 | 1,088 / 0 | 10,047 / 6,676 / 6,676 |
| 2D $\kappa=16$ | 1,775 | 3,949 | 1,775 | 1,775 | 1,775 | 80 / 0 | 1,775 / 2,087 / 1,775 |
| **2D ablation（RQ3，$c_e=4$ 异质）** | 695 | 973 | 599 | **599** | 942 | 72 / 8 | 695 / 521 / 521 |
| 2D random 30×30 dense | 53,904 | 126,159 | 53,904 | 53,904 | 54,772 | 899 / 0 | 54,856 / 64,689 / 54,856 |
| 1D toggles $k=16$ | 135,743 | 48,104 | 48,104 | 48,104 | 48,104 | 0 / 1,999 | 链 |
| 1D toggles $k=64$（$\alpha=2.5$） | 319,533 | 192,019 | 239,346 | 239,346 | 239,346 | 1,111 / 888 | 链 |

结论：

1. **混合物化在所有 2D case 上都等于或优于两种单一机制**：稀疏变化时自动选
   node copying（去掉 $\log M$），聚簇大批量变化时自动选 path copying，异质时
   逐边混用（ablation：599 < min(695, 973)）。
2. **MST 在边代价异质时仍然起作用**：ablation 上混合权 MST 比任意生成树省 1.57x，
   随机密集网格上省 1.6%–5.2%；在 general position 与均匀重合网格上所有生成树
   相同，这正是 §4.3 的全局最优引理，不是 MST 失效。RQ3 的写法应改为“混合权 MST
   vs 任意生成树 vs 单一机制”。
3. **$\alpha$ 要与版本树形状匹配**：full persistence 每次变化 2 条 entry
   （$\alpha\approx2.5$），链（partial persistence）只要 1 条（$\alpha\approx1.5$）。
   1D $k=64$ 一行用 $\alpha=2.5$ 误选了 path copying，比纯 node copying 差 25%；
   换 $\alpha=1.5$ 复测后混合 = 192,019 = 纯 node copying（全部 1,999 条边选 node）。
   反过来，在 ablation 的分叉版本树上用 $\alpha=1.5$ 会全选 node copying，退化为
   973（比 $\alpha=2.5$ 的 599 差 1.4x）。所以 $\alpha$ 必须按“该边的目标版本是否
   需要 exit entry”（$post[v]<V$）逐边取值，而不是全局常数；这在构造计划已知后
   是可以精确判定的。
4. **版本树的形状成了新的优化维度**：node copying 下链状计划省掉全部 exit entry，
   ablation 上 snake 链 + node copying（521）甚至优于混合权 MST（599）。
   因此 node copying 下的计划优化是“最小化 $\sum_e w_e$ + 尽量链状”，
   精确版本是 Hamiltonian-path 型问题（NP-hard），可用 MST 的 DFS 序 /
   最长链启发式或网格 snake 近似；这是 MST 之上可以加的新内容，而不是替代 MST。

### 4.7 2D 之后剩下什么

$O(P+K+M+\sum_T c_e)$ 中，$P$ 与 $M$ 是模型大小，$\sum_T c_e$ 在 general
position 是 $K-1$，所以剩下的全是 $\Theta(K)$：root 目录与数值缓存。它来自
“每个 cell 一个 root”的设计，不是信息论下界；去掉它需要换索引结构，见第 5 节。

---

## 5. 多维：更通用、更优的算法

### 5.1 cell 方法的 $d$ 维形式（back-end 不变，但要补两件事）

* **Compact 布局推广到版本树。** 论文的 Compact 只定义在线性 region 序列上。
  用构造树的 preorder/exit 编号，每次变化写两条记录（enter/exit），slot 查询仍是
  按位置的前驱查找：空间 $\Theta(C+2\sum_T c_e)$，访问 $O(\log h_s)$。
  这就是 DSST fat-node full persistence 的静态版本，任意固定 $d$ 都一样。
* **物化布局用 node copying。** 同上，$\Theta(M+K+\sum_T c_e)$。
* **代价模型改为 $c_e$**，MST 退化为 Hamming 距离最小生成树；非邻接 cell 的
  零代价边（签名相同）可用沿扫描线增量维护的 rolling hash 以 $O(c_e)$/边发现。
* 不变的问题：$K\le\prod_j(B_j+1)$，exact overlay 最坏也是 $O(N^d)$；点定位在
  $d$ 维 box arrangement 上需要 $O(K)$ 级结构。Theorem 4（fixed-$d$）仍成立，
  但只是 dimension independence。

### 5.2 不枚举 cell 的通用索引（推荐作为高维主方案）

把问题看成 **$N$ 个 axis-aligned leaf box 上的 semigroup point-enclosure
（stabbing-sum）查询**：答案 = 包含 $\mathbf t$ 的所有 box 的 payload 之
$\oplus$，每棵树恰好贡献一个 box，因此多重性自动正确。

结构：第 1 维 segment tree，每个 box 挂到 $O(\log N)$ 个 canonical 节点；节点递归
存 $(d-1)$ 维结构；最后一维就是 `tree.py` 的 interval tree（prefix/suffix 聚合
root 在 arena 中 hash-consed，并各缓存一个 unit-bound 数值）。

* 空间 $O(N\log^{d-1}N)$ 记录；$d=2$ 是 $O(N\log N)$。
* 查询 $O(\log^{d}N)$ 时间，返回 $O(\log^{d}N)$ 个片段 root；数值 = 这些片段
  缓存数值之和；单 root 按需 `add` 合成、hash-cons、按 cell 缓存。
* 片段两两不交（segment tree 的 canonical 分解 + interval tree 每个 interval 只挂
  一处），所以不幂等 $\oplus$ 下精确。
* 与 $K$ 无关。对照：2D stump $M=2048$ 时 $K=1{,}050{,}625$（packed 100 MiB），
  而 $N\log N\approx 4096\times 12\approx 5\times10^4$ 记录。

下界侧：Chazelle (1990, arithmetic model) 对 $d$ 维 dominance/orthogonal range
给出空间 $m$ 时 $\Omega\big((\log N/\log(2m/N))^{d-1}\big)$ 次半环加法。
$d$ 维 dominance 可嵌入 tree ensemble（每个点对应一棵深度 $d$、只有一个非零 leaf
的树；把其余 leaf 的原子映到零元是自由半环上的同态，预物化 piece 经同态后仍是
piece），所以在“piece 为预物化、与查询无关的聚合，查询只做 $\le t$ 次组合”的
模型里：近线性空间下 polylog 片段不可避免；单片段需要 $m=N^{1+\Omega(1)}$，
$t$ 片段需要 $N^{1+\Omega(1/t)}$。**修正（§8）**：指数中的常数来自原定理，
不能写成 $N^2$ 或 $N^{1+1/t}$，所以 cell 方案的 $\Theta(K)$ 与下界只是“同为多项式
级”，不是精确匹配；中间点也没有已知的匹配构造。设计空间仍然清楚：cell 方法是
“1 片段 / $\Theta(K)$ 空间”端点，K-free 索引是“polylog 片段 / 近线性空间”端点，
后者与下界只差 $\mathrm{polyloglog}$ 因子。归约引理需要单独写出并钉死 piece 模型。

关于片段数：原型（末维 interval tree）是 $O(\log^d N)$ 片段。把末维换成按
elementary slab 物化的 1D 持久化 factor tree（节点 $v$ 的 $n_v$ 个区间划出
$\le 2n_v$ 个 slab，相邻 slab 只差 $O(1)$ 个区间，沿 slab 链做 partial-persistence
node copying，空间 $O(n_v)$，每个 slab 一个预物化 root）可得每个 canonical 节点
1 个 piece，总计 $O(\log^{d-1}N)$ 片段、$O(N\log^{d-1}N)$ 空间，查询时间
$O(\log^{d-1}N\cdot\log N)$。这是设计与证明草图，**未实现、未实测**。

只要数值不要 provenance 时，减法是允许的：2D stabbing-sum 经容斥化为 dominance
sum，persistent BST 扫描线给出 $O(N)$ 空间、$O(\log N)$ 查询；$d$ 维
$O(N\log^{d-2}N)$ 空间、$O(\log^{d-1}N)$ 查询。所以数值查询在任何维度都不需要
cell；半环结构只为 provenance 服务。

### 5.2.1 实测（`review_prototype/kfree_proto.py`，hit server3）

实现了上述 $d$ 层结构（前 $d-1$ 维 segment tree，末维 centred interval tree +
prefix/suffix 聚合，聚合 root 在 `CircuitArena` 中 hash-cons，并缓存数值）。
每个 case 对随机点与恰好落在阈值上的点做校验：数值与直接求值一致（误差
$\le 1.4\times10^{-13}$），抽样查询展开后的 (tree, leaf) multiset 与直接求值完全一致，
且每棵树恰好一个 leaf。记录口径：assignment + 2 个排序键 + 聚合 arena 节点 +
数值缓存 + 树节点。对照列“cells(grid)”按第 4 节的实测系数估算：node copying
$\approx 3.5K$、path copying $\approx K(2+\log_2 M)$、IIP $\approx K(M+1)$。

| 场景 | $M$ | $N$ | $K_{grid}$ | K-free 记录 | 片段数 mean/max | cells: node-copy | cells: path-copy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D stumps | 512 | 1,024 | 65,985 | 16,924 | 10.3 / 13 | 2.3e5 | 7.3e5 |
| 2D stumps | 2,048 | 4,096 | 1,050,589 | 80,742 | 12.4 / 16 | 3.7e6 | 1.4e7 |
| 2D stumps | 4,096 | 8,192 | 4,198,232 | 174,202 | 13.3 / 17 | 1.5e7 | 5.9e7 |
| 2D depth-2 | 256 | 1,024 | 148,216 ($K_{exact}$=101,946) | 21,887 | 17.7 / 27 | 5.2e5 | 1.5e6 |
| 2D depth-2 | 1,024 | 4,096 | 2,362,368 | 110,497 | 22.9 / 35 | 8.3e6 | 2.8e7 |
| 2D depth-4 | 256 | 4,096 | 3,689,085 | 126,933 | 31.4 / 39 | 1.3e7 | 3.7e7 |
| 3D stumps | 3,000 | 6,000 | 1.0e9 | 145,388 | 22.3 / 26 | 3.5e9 | 1.4e10 |
| 3D depth-2 | 1,000 | 4,000 | 1.0e9 | 224,824 | 67.3 / 85 | 3.5e9 | 1.2e10 |
| 3D depth-3 | 300 | 2,400 | 3.4e8 | 189,190 | 68.9 / 90 | 1.2e9 | 3.8e9 |
| 4D stumps | 2,000 | 4,000 | 6.3e10 | 96,327 | 29.3 / 32 | 2.2e11 | 8.2e11 |
| sklearn GBDT depth-2 | 128 | 512 | 13,780 ($K_{exact}$=6,574) | 9,338 | 10.4 / 18 | 4.8e4 | 1.2e5 |

读法：

* 2D $M=2048$ 正是 packed 实验的百万 cell 规模：cell 方案即使换成 node copying
  也要约 $3.7\times10^6$ 条记录（加 $2K$ 的 root/值目录），K-free 索引只要
  $8.1\times10^4$ 条，约 45x；相对现有 path copying 约 170x。
* $d=3,4$ 时 $K_{grid}$ 达到 $10^9$–$10^{10}$，cell 方案不可行；K-free 索引仍是
  $10^5$ 量级，查询 20–70 个片段。片段数随深度（box 数 $N$）与维度缓慢增长，
  与 $O(\log^d N)$ 一致，远小于 $M$。
* sklearn 深度 2 GBDT 上与 `model.predict` 完全一致（含 float32 量化语义）；
  exact overlay 比全阈值网格少 1.6–2.1x cell，深度 4 合成树上是 3.6x。
* 构建时间 0.01–0.3 s（纯 Python），因为从不枚举 cell。
* 代价是每次查询 10–90 次指针读取与加法，而不是 cell 方案的 1 次；对 provenance
  handle 来说这是 output-insensitive 的常数，对数值查询仍远低于 $M\cdot D$ 次树遍历。

### 5.3 推荐架构

```
K-free stabbing 索引（基底，O(N log^{d-1} N)，与 K 无关）
    └── 热 cell 缓存：按需 node-copy 物化单 root，hash-cons，LRU
低维且 K 可承受时：exact overlay + c_e-MST/snake + node copying（全量物化）
```

两层共享同一个 payload arena 与同一个 unit binding $h_u$，语义不变。

### 5.4 一个介于两者之间的简单选项

把 $M$ 棵树分成大小 $g$ 的组，每组做 exact overlay（$K_g\le(gb)^d$）并物化组内
root，查询取 $M/g$ 个组 root 之和。空间 $\sum_g K_g$、片段 $M/g$，$g$ 可调。
stump 场景下被 segment tree 支配，但实现简单，深树上可作为工程备选。

---

## 6. 对论文定理与实验的具体建议

| 位置 | 现状 | 建议 |
|---|---|---|
| Thm. path-copying bound, Cor. sparse separation | VSC $O(n\log n)$ | 增加 node-copying 布局：$\Theta(D_T+C+B+K_\Delta+H)$ 空间、每节点 $O(1)$ 遍历；分离变为 IIP $\Theta(n^2)$ vs materialized $\Theta(n)$ |
| Fan-out separation $\Theta(BF)$ vs $\Theta(B+F)$ | 成立 | 保留。node copying 在 in-degree $F$ 下每次变化也要 $\Theta(F)$，所以该分离在版本戳模型下依然成立，可顺便说明 |
| Compact “resolve slot $O(\log(h_s+1))$” | per-slot bisect | 改为 wavelet tree：$O(\log C)$ 单 slot、$O(C)$ 全解析；bits 级空间 |
| MCPST Theorem 1/2, RQ3 | $\rho_e$-MST，exact copy cost 是算法内容 | **保留 MST 与 Theorem 1（immutable path copying 内精确最优）**；混合权 $\min\{\rho_e,\alpha c_e\}$ 作为 surrogate 的经验改进，不写进最优性定理（§8 A4）；general position 下补 Hamming 模型内的全局最优引理；RQ3 改为三方对照：surrogate-MST 599 vs 任意生成树 942 vs 纯 path 695 / 纯 node 973（§4.6）；链状计划作为经验观察 |
| §2 exact overlay vs 代码网格 | 文本与实现不一致 | 实现扫描线 overlay + persistent point location；报告 $K_{exact}/K_{grid}$（实测：sklearn 深度 2 为 1.6–2.1x，合成深度 4 为 3.6x） |
| 多维（新） | 仅 Theorem 4 的 dimension independence | 加 K-free point-enclosure 索引定理 $O(N\log^{d-1}N)$ 空间 / $O(\log^d N)$ 片段，实测 2D $M=2048$ 比 cell 方案少 45x，3D/4D 下 $K_{grid}\ge10^9$ 时仍只需 $10^5$ 记录（§5.2.1） |
| Theorem 4（fixed-$d$） | dimension independence | 补 Compact 的版本树推广（enter/exit），再补 K-free 索引定理与 Chazelle 下界讨论 |
| 构建复杂度 | $\Theta(KM)$ 稠密矩阵 | boundary label 产生 $\Delta_e$，整数权计数排序，$O(N\log N+K\,\alpha(K))$ |
| 实验 | 记录数 | 加 node copying vs path copying 的 bytes；$M=2048$ 上 K-free 索引 vs 全量 cell |

审稿风险要提前写明：node copying、segment tree、persistent point location 都是
教科书结构，novelty 必须落在 (i) 从 fitted 模型编译出 cell/boundary label 的
compiler，(ii) 不幂等半环 provenance + certificate 的语义，(iii) 两类索引之间的
space–pieces 权衡与下界论证，而不是 persistence 本身。仓库的 non-claims 列表已经
有这个方向，建议保持。

---

## 8. 联合审稿（Claude + Codex，2026-10-13）与修正

Codex（gpt-5.6）以 SIGMOD 审稿人身份对统一方案做了对抗式审稿，我逐条核对后的裁定
与已做的修正如下。原则：凡是“受限模型内的结论被写成一般下界/全局最优”的地方一律
降级；凡是原型未实现的设计一律标注。

| 编号 | Codex 意见 | 我的核对 | 修正 |
|---|---|---|---|
| A1/A2 | Chazelle 归约与 $N^2$、$N^{1+1/t}$ 曲线过强 | 同意指数不能写死：原定理只给 $N^{1+\Omega(1)}$、$N^{1+\Omega(1/t)}$；归约本身（映到零元的同态）我认为成立，但 piece 模型必须钉死 | §0、§5.2 已改；“端点匹配”“最优曲线”撤回，改为“同为多项式级 / 差 polyloglog” |
| A3 | enter/exit 的 full-persistence node copying 不是 DSST 定理实例 | 同意。星形版本树上每个孩子都复制整条路径，退化到 path copying 量级（最坏约 $3c_e\log M$）；$O(1)$ 摊还只在链上已证 | §0、§4.5 已改；一般版本树上的 $O(1)$ 摊还列为需要 DSST node splitting 的未完成工作 |
| A4 | 混合权 MST 的“模型内最优”不成立 | 同意。node copying 代价依赖树形与 entry 占用，不是独立可加边权 | §4.5 已改为 surrogate；exact 最优性只保留给 immutable path copying |
| A5/A6 | general-position 全局最优与 $\kappa$ 界只在 Hamming 更新模型内 | 同意 | §0、§4.3 已加限定 |
| A7 | $\Theta(P+N)$ 不是信息论/word 下界 | 同意（模型可重算一切）；改为“显式 boundary + 有效更新、查询不回读模型的表示模型内线性最优” | §3.1 的措辞按此理解；论文 Compact-closure bound 与 compression law 需同步降级 |
| A8 | 分组 NP-hardness 不可采信 | 同意，撤回；$\Sigma K_G\le$ 组数$\cdot (N+1)^D$ 作为粗上界保留 | 不再写 NP-hard，写 open |
| A9 | K-free 片段应为 $O(\log^d N)$ | 部分同意：原型确实是 $\log^d$；$\log^{d-1}$ 需要“末维按 slab 物化的持久化 1D 结构”，是设计不是实测 | §5.2 已区分 |
| C1 | 片段数要乘组数、定义 per-query frontier | 同意 | 统一索引的 $t$ 定义为 $\sum_G t_G$ |
| C2–C5 | 查询成本分项、$\hat K$ 校准、lazy cache 独立、hash-consing 与 node copying 不叠加 | 同意 | 写入实验契约 |
| C6 | exact overlay 未实现，现有 2D 数字来自网格 | 同意（本文 §1.4、§4.4 已指出） | 投稿前必须实现 |
| B | novelty 只能收缩为“几何编译的版本图 + 受限模型内精确物化 + 可验证 tradeoff” | 同意 | 补引 Olteanu–Závodný、Agarwal–Erickson、Bahar ADD、QuickScorer |

第二轮（我提出四条反驳，Codex 逐条裁定）：

* R1 归约：同意“映到零元的半环同态”使归约成立，前提是 piece 模型钉死为
  query-independent 的预物化聚合、查询至多组合 $t$ 个；指数只能写
  $N^{1+\Omega(1)}$ / $N^{1+\Omega(1/t)}$。还需证明每棵 point-tree 的命中叶恰贡献
  对应 $w_i$ 且空间映射不增。
* R2 node copying：同意收缩后的表述；论文措辞用 “we do not implement or analyze
  the additional machinery required for full persistence”，不要写成 DSST 不支持。
* R3 K-free 持久化末维：同意 $O(\log^{d-1}N)$ 个预物化 piece、$O(N\log^{d-1}N)$ 空间，
  但查询时间保守写 $O(\log^d N)$（fractional cascading 未证明前不进主定理）；
  “版本戳只影响常数”要求 bounded degree、常数 modification capacity、root/slab
  目录计入 $O(n_v)$。
* R4 1D 线性最优：同意作为 representation-model statement，措辞 “When boundaries and
  effective updates are stored explicitly and queries may not reread the fitted model,
  the representation uses linear space in the explicit input size”；不称
  information-theoretic 或 word-RAM lower bound。

Codex 建议的三条主定理措辞（可直接用）：

1. *Immutable MCPST optimality.* For a fixed persistent factor-tree layout, MCPST's
   source-augmented minimum spanning tree minimizes the number of newly allocated
   immutable path-copied nodes among all adjacency-constrained single-parent
   materialization plans.
2. *K-free enclosure index.* For $N$ axis-aligned leaf boxes in fixed dimension $d$,
   there is an exact non-idempotent-semigroup point-enclosure index using
   $O(N\log^{d-1}N)$ space that returns $O(\log^{d-1}N)$ query-independent
   pre-materialized aggregate roots in $O(\log^{d}N)$ time.
3. *One-dimensional persistent materialization.* For a one-dimensional change sequence
   with $B$ regions and $K_\Delta$ effective slot updates, a bounded-degree partially
   persistent factor tree stores direct roots in $O(P+B+K_\Delta)$ space while
   preserving exact non-idempotent provenance multiplicities.

修正后仍然成立、可以进论文的声称：

1. 1D：PFT + node copying 沿链，$\Theta(P+N)$（显式表示模型内线性最优），每 region
   直接 root，遍历每节点 $O(1)$（Sarnak–Tarjan）。实测比 path copying 3.3–5.5x。
2. 2D：exact overlay + $\rho_e$ 边权 + source-augmented MST + immutable path copying
   在 adjacency-constrained、single-parent 模型内最优（原 Theorem 1）；Hamming 模型内
   general position 任意生成树最优、$\kappa$ 近似。node copying/混合为经验改进
   （链状计划 6–6.7x，MST 树 3x），surrogate 权的 MST 在异质 workload 上比任意
   生成树省 1.57x。
3. 多维：K-free 索引 $O(N\log^{d-1}N)$ 空间、$O(\log^d N)$ 片段（已实现、已验证 1D–4D）；
   $O(\log^{d-1}N)$ 片段的持久化末维变体为设计。
4. 下界：在钉死的 arithmetic 模型下，单片段需 $N^{1+\Omega(1)}$，近线性空间需
   polylog 片段（归约引理待写）。
5. 分组：按轴支撑分组的空间 $\sum_G K_G\le$ 组数$\cdot(N+1)^D$，stump 时 $\le N$；
   最优分组复杂度 open。

## 9. 三项加强（2026-10-13）：真实案例、高维对比与自动配置、持久化末维定理

目标是把审稿结论从 weak accept 推向 accept 的三件事。代码
`review_prototype/unified_proto.py`，全部在 server3 上用真实数据运行。

### 9.1 设置

* 数据：California housing（20,640 行、8 个特征）。模型：sklearn
  `GradientBoostingRegressor`（300 棵、深度 3，精确切分，接近 general position）与
  `HistGradientBoostingRegressor`（300 轮、深度 3、255 箱，阈值大量重合）。
* 固定 unit：随机一行。干预集合 $S$ 取重要性最高的 $d$ 个特征，$d\in\{2,3,4,6,8\}$。
  unit binding：每棵树中非干预特征上的 split 按该 unit 的取值走单边，
  得到只含干预轴的 reachable leaf box（即论文 §2 的 fixed-context 编译）。
* 四种索引：cells（全阈值网格 + snake node copying，$K_{grid}\le 3\times10^5$ 才建）、
  K-free（末维 interval tree）、**K-free 持久化末维**（新，每个 canonical 节点 1 个
  预物化 root）、**统一索引**（按轴支撑分组 + $\hat K\le\tau$ 的自适应递归 + 叶级
  按 box slot 的 snake node copying）。
* 正确性：每个设置 400 个随机 what-if 点与 `model.predict` 逐点比较（GBR 按 sklearn
  的 float32 输入语义取整），抽样 30 点做 leaf multiset 比对。
* 延迟基线：逐行 `model.predict`（what-if 的真实重放方式）。

### 9.2 定理：持久化末维的 K-free 索引

**定理（K-free enclosure index, persistent last axis）.** 给定固定维度 $d$ 中的
$N$ 个轴对齐 leaf box，存在精确的不幂等半环 point-enclosure 索引，空间
$O(N\log^{d-1}N)$，每次查询返回 $O(\log^{d-1}N)$ 个与查询无关的预物化聚合 root，
查询时间 $O(\log^{d}N)$。

**证明草图.** 前 $d-1$ 维建 segment tree，每个 box 在每一维被拆到 $O(\log N)$ 个
canonical 节点，故总 assignment 数 $A=O(N\log^{d-1}N)$。对每个叶级节点 $v$（它持有
$n_v$ 个 box 在末维上的区间），区间端点把末维切成 $\le 2n_v$ 个 elementary slab；
相邻 slab 之间只有在该端点开始或结束的区间改变活跃状态，所以沿 slab 链的有效
更新总数 $\le 2n_v$。以这 $n_v$ 个区间为 slot 建 balanced persistent factor tree，
沿链做 bounded-degree partial-persistence node copying（Sarnak–Tarjan），空间
$O(n_v)$ 摊还，并为每个 slab 记录 root 与数值缓存。总空间
$\sum_v O(n_v)=O(A)$。查询沿每一维的 segment tree 走到 $O(\log N)$ 个节点，
共 $O(\log^{d-1}N)$ 个叶级节点，在每个节点上二分定位 slab 并取 1 个 root，因此
片段数 $O(\log^{d-1}N)$、时间 $O(\log^{d-1}N\cdot\log N)$。精确性：canonical 分解
使每个 box 在查询路径上恰出现一次，slab 内活跃集合恰为包含查询点的区间，
不幂等 $\oplus$ 下多重性正确。$\square$

限定（Codex R3）：版本戳只影响常数需要 bounded degree 与常数 entry 容量；slab 目录
计入 $O(n_v)$；fractional cascading 可把时间降到 $O(\log^{d-1}N)$ 但未证明不入定理；
一个 piece 的展开代价是该节点的 slot 数 $n_v$ 而非活跃数，要 output-sensitive 展开
需换成活跃集合的持久化平衡树（同样 node copying，未实现）。

### 9.3 实测（`review_prototype/unified_results.txt`，server3）

固定 unit #11670，GBR $R^2=0.847$、HGB $R^2=0.843$。所有设置、所有索引的 400 个
what-if 点与 `model.predict` 误差 $\le 1.5\times10^{-14}$，抽样 multiset 全部一致。
记录数为结构记录（节点 + entry + 目录），查询为纯 Python 单点延迟。

**GBR（精确切分）**

| $d$ | $N$ | $K_{grid}$ | cells 记录 / 片段 | K-free(itree) 记录 / 片段 | **K-free(持久化末维)** 记录 / 片段 | **统一索引 τ=4096** 记录 / 片段 | 查询 μs（持久化 / 统一 / 逐行 predict） |
|---:|---:|---:|---|---|---|---|---|
| 2 | 745 | 10,680 | 51,694 / 1 | 10,670 / 11.1 | 3,994 / 7.9 | 15,679 / 3.0 | 5.6 / 5.3 / 278 |
| 3 | 966 | 1.7e6 | 不可行 | 26,792 / 31.3 | 13,754 / 25.8 | 33,070 / 12.4 | 17 / 18 / 277 |
| 4 | 1,520 | 4.3e8 | 不可行 | 74,558 / 58.2 | 44,583 / 50.3 | 80,106 / 35.2 | 41 / 49 / 283 |
| 6 | 2,032 | 3.4e12 | 不可行 | 191,083 / 112.9 | 126,915 / 106.1 | 114,495 / 118.8 | 121 / 176 / 271 |
| 8 | 2,376 | 3.4e16 | 不可行 | 310,787 / 144.1 | 216,548 / 137.5 | 163,880 / 181.6 | 205 / 288 / 275 |

**HGB（255 箱，阈值重合：每个阈值平均被 3–5 个 box 共用，最多 43）**

| $d$ | $N$ | $K_{grid}$ | cells | K-free(itree) | K-free(持久化) | 统一索引 | 查询 μs（持久化 / 逐行 predict） |
|---:|---:|---:|---|---|---|---|---|
| 2 | 767 | 6,320 | 36,735 / 1 | 9,676 / 10.3 | 3,465 / 7.2 | 11,790 / 3.0 | 5.3 / 7,065 |
| 3 | 958 | 6.4e5 | 不可行 | 21,967 / 31.0 | 10,711 / 25.3 | 23,830 / 11.9 | 16 / 7,284 |
| 4 | 1,533 | 1.2e8 | 不可行 | 60,797 / 59.7 | 34,699 / 51.4 | 61,295 / 39.5 | 40 / 7,489 |
| 6 | 2,040 | 4.3e11 | 不可行 | 157,938 / 118.6 | 102,704 / 110.0 | 118,418 / 115.3 | 123 / 7,742 |
| 8 | 2,358 | 1.2e15 | 不可行 | 246,034 / 148.2 | 169,922 / 140.5 | 182,870 / 168.6 | 214 / 7,644 |

读法：

1. **cell 方案只在 $d=2$ 可行**；$d=3$ 起 $K_{grid}$ 从 $10^6$ 到 $10^{16}$，而三种
   K 无关的索引在 $d=8$ 仍只有 $1.6$–$3.1\times10^5$ 条记录、构建 0.5–1 s。这就是
   “建不出来 vs 建得出来”的直接证据。
2. **持久化末维**比 interval-tree 末维记录少 1.4–2.7x、片段少 4–6 个；高维时片段主要
   来自前几维的 segment tree，所以相对减少有限，与 $O(\log^{d-1}N)$ vs $O(\log^d N)$
   的差一个 $\log$ 一致。
3. **统一索引**在 $d\le4$ 用 1.5–2.4x 的空间换到少 2.6x 的片段（3 vs 8，12 vs 26，
   35 vs 50）；$d\ge6$ 时反而记录更少但片段更多，因为按轴支撑分出的组太多
   （深度 3 的树在 8 个轴上有几十种支撑组合，每组至少 1 个片段）。改进方向明确：
   把树数少的组合并到 $\hat K$ 仍 $\le\tau$ 的邻近组，或给组数设预算。
4. **自动配置的旋钮有效且单调**（GBR，$d=4$）：τ 从 256 到 65,536，记录
   31k→223k，片段 76→31，叶级节点 1,602→440。
5. **阈值重合不伤 K-free**：HGB 的 K-free 记录与片段与 GBR 同量级甚至更少，
   而 cell 方案在 $d=2$ 的 HGB 上同样要 3.7 万条记录。
6. **延迟**：对 GBR 逐行 predict（真实 what-if 重放）约 275 μs，索引 5–200 μs；
   对 HGB 逐行 predict 约 7 ms，索引 5–214 μs，即 35–1400x。纯 Python 实现，
   相对数字可信，绝对数字不可写进论文。
7. **provenance 用例（对比式归因）**：$t_1\to t_2$ 的预测差 $-0.2705$ 被精确分解到
   300 棵树中换叶的 79 棵上，逐树 delta 之和与模型差值精确相等，并给出贡献最大的
   5 棵树；这是“返回 provenance root”在应用上的直接用途（exact contrastive
   explanation / audit），数值上无法从 `predict` 得到。

### 9.3.1 自动配置的改进：组数预算 $g$ 与轴顺序（`unified_tune.py`，`unified_tune_results.txt`）

§9.3 第 3 点的问题（$d\ge6$ 时组太多）用两个开关修正：组数预算 $g$（贪心把树最少
的组并入支撑集合是其超集的组，没有超集则与轴重叠最多的组并成并集，直到组数
$\le g$）和轴顺序（组内先对阈值多的轴建 segment tree）。τ 固定 4096。

| 模型 / $d$ | K-free 持久化（记录 / 片段） | $g$=∞（按轴支撑） | $g$=16 | $g$=8 bycount | $g$=4 bycount | $g$=1 bycount |
|---|---|---|---|---|---|---|
| GBR 4 | 44,583 / 50.6 | 80,106 / 35.4 | 80,106 / 35.4 | 71,849 / 30.1 | 61,491 / 32.9 | 84,555 / 27.2 |
| GBR 6 | 126,915 / 105.7 | 114,495 / 119.0 | 134,003 / 95.4 | 180,305 / 84.1 | 214,007 / 77.0 | 321,195 / 54.6 |
| GBR 8 | 216,548 / 137.6 | 163,880 / 181.5 | 193,911 / 149.4 | 304,154 / 118.5 | 401,880 / 100.4 | 506,016 / 88.5 |
| HGB 6 | 102,704 / 110.7 | 118,418 / 115.5 | 133,779 / 91.5 | 173,262 / 74.3 | 206,289 / 60.9 | 264,987 / 48.8 |
| HGB 8 | 169,922 / 140.4 | 182,870 / 168.0 | 220,576 / 138.4 | 268,293 / 117.7 | 353,858 / 103.4 | 486,441 / 75.5 |

全部精确（误差 $\le 1.4\times10^{-14}$）。结论：

1. $g$ 是一个单调、可预测的旋钮：$d=8$ 时片段从 181 降到 88（GBR）/ 168 降到 76
   （HGB），记录升到 2.3–2.9x；$d=4$ 时 $g=4$–8 只多 1.4–1.6x 记录就把片段从 51 降到
   30–33。轴顺序 bycount 在 $g$ 小时再省 10–15% 片段。
2. 统一索引不会在记录数和片段数上**同时**优于 K-free（$d\ge6$），这是权衡曲线决定的，
   不是实现缺陷：K-free 是空间端点，$g$ 和 τ 把索引沿曲线向“少片段”方向移动。
3. 因此自动配置应是 **build-time 搜索**而不是单一阈值：构建只需 0.1–2 s，可对
   $g\in\{\infty,16,8,4,2,1\}$、τ 若干档各建一次，用采样查询测片段数，选满足片段
   预算 $t$ 的最小空间配置（或满足空间预算的最少片段配置）。这与物理设计调优同一
   思路，并且对任何模型/维度都给出位于实测 Pareto 前沿上的点。

### 9.4 这三项对审稿结论的影响

* 真实案例 + 高维对比：回应“问题是否真实、高维怎么办、索引是不是 case by case”，
  统一索引在 $d=2..8$ 一套代码、一个 τ 跑通，并且自动在 $d=2$ 退化为三组 1D/2D。
* 持久化末维：是可写成定理并已验证的新算法结果（§9.2），片段 $O(\log^{d-1}N)$。
* 仍缺的：exact overlay 替代网格（cell 层）、组数预算、C++ 实现与字节级测量、
  多 unit 实验。

## 7. 原型说明

`review_prototype/nodecopy_proto.py`：

* 复用仓库的 `BalancedPFTLayout`、`transitions_from_states`、
  `build_spanning_plan`、`materialize_plan` 与 `make_stump_grid`，所以 path
  copying 数字与 `RESULTS.md` 一致；
* `NodeCopyPFT` 实现静态版本树上的 limited node copying（preorder/exit entry，
  capacity 可调），`decode()` 对每个 cell 解码并与状态矩阵逐项比对；
* `SnakePlan` 给出链式（partial persistence）变体；
* 记录口径：内部节点 1 条，entry 1 条；字节口径节点 8 B、entry 12 B。

可直接扩展的下一步：(1) 逐边混合策略；(2) 2D 扫描线 exact overlay；
(3) 2D segment-tree stabbing 索引并与 cell 方案比空间/片段数。
