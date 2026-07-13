# TESSERA 理论附录草稿

状态：VSC 修订进行中。下文原有 L/T/N compiler semantics 与 MGPF/LCF 定理仍按
各自声明的表示类成立，但 MGPF/LCF 已降为 additive frontier optimizer。general
storage theorem 改为本节的 Versioned Semiring Circuit；restricted hardness、完整
weighted factorization objective 和 fitted-MLP compiler bound 仍是投稿前 obligations。

## 0. Versioned Semiring Circuit

### 定义 0.1：Canonical Circuit Template

固定一个 finite acyclic template `T=(V,E)`。leaf 分为 immutable atoms 与 mutable
slots；internal node 标签属于 `{ADD,MUL}`。所有 region versions 使用同一 topology，
但 slot `s` 在 version `(u,j)` 中绑定到某个 canonical semiring root
`sigma[u,j](s)`，包括 `ZERO`。

全局 arena 对 `(operator, canonical child IDs, payload)` 做 hash-cons。`ADD` 保留
重复 child，因此不假设 idempotence。令 `R[u,j]` 是按照 template 拓扑序实例化后
的 root ID。

### 引理 0.1：Version Update Exactness

设 `R[u,j-1]` 已精确表示 version `j-1`。把所有发生变化的 slots 更新为
`sigma[u,j]`，并按拓扑序重新 intern 它们的 ancestor closure；其余 template node
IDs 保持不变。得到的 root `R[u,j]` 等于从头实例化 `sigma[u,j]` 所得到的 semiring
circuit。

**证明。** 对 template 拓扑序归纳。immutable leaves 与未改变 slots 显然保持正确，
改变 slots 被直接赋为新 binding。考虑 internal node `v`。若 `v` 不在 changed-slot
ancestor closure 中，则所有 descendant slots 均未改变，旧 node ID 仍表示相同
subcircuit。若 `v` 在 closure 中，算法在所有 children 已正确后，用相同 `ADD` 或
`MUL` operator 重新 intern children roots，因此得到从头实例化 `v` 的 canonical
root。归纳到 template root 即得结论。证明不使用 slot changes 的单调性，所以允许
任意 `ON/OFF/ON` 序列。\(\square\)

### 引理 0.2：Balanced Batch Path Bound

设一个 associative block 被组织为包含 `C` 个 leaves 的 height-balanced binary
operator tree。一个 version boundary 改变 `k` 个不同 leaves。批量更新需要访问的
changed leaves 及其 internal ancestor closure 的节点数为：

\[
O\!\left(k\log\left(1+\frac{C}{k}\right)\right).
\tag{V1}
\]

**证明。** 先令 `C` 为 2 的幂，tree height `h=log_2 C`；非 2 的幂只改变常数。
在 depth `d`，`k` 条 leaf-to-root paths 至多访问 `min(2^d,k)` 个不同 nodes。因此
总访问量至多：

\[
\sum_{d=0}^{h}\min(2^d,k).
\]

令 `d_0=floor(log_2 k)`。前 `d_0` 层之和为 `O(k)`；余下
`h-d_0` 层每层至多 `k` 个 nodes，总计
`O(k(h-log k))=O(k log(C/k))`。两部分合并即为式 (V1)。当 `k=C` 时右式按
`log(1+C/k)` 写法仍为 `Theta(C)`。\(\square\)

### 定理 0.1：VSC Construction, Storage, and Root Lookup

记：

```text
B_tot        population exact-region 总数
D_0          初始 templates、atoms 与第一 versions 的 canonical node 数
alpha[u,j]   transition 中 changed slots 加上被重新计算的 ancestor nodes
gamma[u,j]   其中新产生的 canonical arena nodes；gamma[u,j] <= alpha[u,j]
H_u          unit binding 大小
```

顺序构建所有 unit versions 的空间为：

\[
O\!\left(B_{tot}+D_0+\sum_{u,j}\gamma[u,j]+\sum_u|H_u|\right)
\]

构建时间为：

\[
O\!\left(B_{tot}+D_0+\sum_{u,j}\alpha[u,j]+\sum_u|H_u|\right)
\]

个 template/arena operations，再加 boundary sorting。给每个 unit 保存 sorted
starts 与 roots 后，查询 `(u,t)` 在 `O(log B_u)` 时间返回 compressed root；完整
展开另需 `O(output)`。

对 balanced associative block，若 transition `(u,j)` 改变 `k[u,j]` 个 slots，
引理 0.2 给出：

\[
\alpha[u,j]
\le
O\!\left(k[u,j]\log\left(1+\frac{C}{k[u,j]}\right)\right),
\]

且 `gamma[u,j] <= alpha[u,j]`；global hash-cons 只能进一步减少实际新节点数。

**证明。** 引理 0.1 给出每个 root 的 exactness。arena 初始存 `D_0` nodes；每个
transition 访问 `alpha[u,j]` 个 template nodes，至多追加 `gamma[u,j]` 个 immutable
nodes，并追加一个 root reference；unit bindings 独立计数。sorted predecessor
search 给出 `O(log B_u)`。balanced specialization 直接应用引理 0.2，每个访问 node
至多产生一个新 canonical node。
\(\square\)

### 当前 Lower-Bound 边界

任意 exact index 至少需要区分 `B_tot` 个可独立变化的 region boundaries，并编码
所有可独立辨识的 slot changes，因此存在 families 需要
`Omega(B_tot + sum k[u,j])` information。当前实现相对该 event bound 至多有
balanced-tree logarithmic overhead。投稿前仍需选择明确 machine/representation
model，证明 matching lower bound；不能把“canonical node 每种存一次”包装成无条件
global algebraic optimality。

完整 algebraic rewriting 可能把 syntactically different circuits 合并。该问题与
后文 MGPF 相关，但后文任意 support-matrix hardness 还不足以证明一个受限 fitted
Tree/ReLU family 上的 VSC minimization hardness。

## 1. 固定表示类

固定一个 `(model_version, treatment, target)` 和 finite indexed population
`U`。unit `u` 的 reduced response partition 有 ordered regions
`[B_u]={1,...,B_u}`。定义 population-cell universe：

\[
\Omega=\{(u,j):u\in\mathcal U,\ j\in[B_u]\},
\qquad B_{\mathrm{tot}}=|\Omega|=\sum_u B_u.
\]

同时固定 canonical parameterized fragment dictionary。本文不允许优化器临时
发明任意新的 arithmetic subexpression，也不假设不同 units 共享数值参数或完整
structural template。

令原始 region-fragment multiplicity matrix 为：

\[
W\in\mathbb N^{B_{\mathrm{tot}}\times p},
\qquad
\widetilde P_{u,j}=\bigoplus_{f=1}^{p}W_{(u,j),f}\otimes H_f,
\qquad P_{u,j}=h_u(\widetilde P_{u,j}).
\]

### 定义 1：Occurrence Lifting

对每个原始 fragment `H_f` 和
\(c\in\{1,\ldots,\max_{\omega\in\Omega} W_{\omega,f}\}\) 创建 lifted symbol
\(F_{(f,c)}\)：

\[
M_{(u,j),(f,c)}=[W_{(u,j),f}\ge c].
\]

因此 `M` 是 binary matrix。令：

\[
\widetilde{\mathcal K}=\mathbb N[\{F_{(f,c)}\}_{f,c}]
\]

为 occurrence-lifted free semiring，并定义 quotient homomorphism：

\[
\pi(F_{(f,c)})=H_f.
\]

实现不物化所有 copies；它把相同 fragment 的 multiplicity layers 保存为
weighted support records。lifting 只用于给 hardness、rank 和 partition
correctness 一个无歧义的自由生成元空间。

### 引理 1：Occurrence Lifting Correctness

对每个 population cell `(u,j)`：

\[
\pi\left(
\bigoplus_{f,c}M_{(u,j),(f,c)}\otimes F_{(f,c)}
\right)
=
\bigoplus_f W_{(u,j),f}\otimes H_f.
\]

**证明。** 固定 `(u,j),f`。恰有 `W[(u,j),f]` 个整数 `c` 满足
\(1\le c\le W_{(u,j),f}\)，所以 quotient 后 \(H_f\) 的 multiplicity 恰为
\(W_{(u,j),f}\)。对所有 \(f\) 求和即得结论。\(\square\)

### 定义 2：Unit-Binding Homomorphism

令 parameterized semiring 的 generators 包含 `PARAM_SLOT(s)`、`BIAS_SLOT(s)`、
`ROOT_SLOT(v)`、`NOISE_SLOT(v)` 和 `TAU(T)`。每个 unit `u` 定义 semiring
homomorphism `h_u`，把这些 slots 映射到该 unit 的参数、factual roots、abducted
noises 和 intervention indeterminate，并保持 `0,1,+,*`。后续 numeric valuation
再把 `TAU(T)` 映射到查询值 `t`。

不同 units 的 `h_u` 可以完全不同。若某个 unit 使用不同 topology，其独有
fragments 在 dictionary 中有独立 columns；只有 canonical identity 相同的
parameterized fragments 才共享。

## 2. L/T/N 编译闭包

固定有界 intervention domain \(D\subset\mathbb R\) 和 deterministic endpoint
policy。一个 normalized guarded-provenance response 是有限序列：

\[
\mathcal R_v=
\{(I_j,a_j,b_j,P_j)\}_{j=1}^{B_v},
\]

其中 \(I_j\) 是按顺序排列、两两不交且恰好覆盖 \(D\) 的 connected intervals；
在 \(I_j\) 上，\(X_v(t)=a_jt+b_j\)，并且 value homomorphism 满足
\(h_{u,t}(P_j)=a_jt+b_j\)。允许相邻 pieces 在边界处跳变。

### 引理 A：Endpoint Ownership

采用以下 convention：tree predicate 的 equality 归 `<=` branch；ReLU 的零点
归 inactive branch，即 `z<=0` 输出 `ZERO`；Max/MaxPool ties 按固定 input order
选择第一个 maximizer。每次 split 都把 predicate 与其 complement 配对。因此
local compiler 输出的 intervals 两两不交、穷尽输入 cell，并且每个边界点有且
仅有一个 owner。

**证明。** 一个 deterministic predicate \(g\) 与 \(\neg g\) partition 当前
cell。上述 conventions 只决定 equality 属于哪一侧，不改变 union。递归应用
该事实，所有叶 cells 仍然 disjoint 且 exhaustive。\(\square\)

### 引理 B：Finite Common Refinement

有限个 normalized parent responses 的所有 boundaries 排序去重后，得到唯一的
maximal common refinement。它是有限 partition，并且每个 refined cell 内所有
parent values 和 parent provenance roots 均固定为 affine pieces。

**证明。** 有限集合的有限 union 仍有限。相邻 union boundaries 之间不含任何
parent boundary，所以每个 parent annotation 固定；引理 A 处理 boundary points。
任何同时细化所有 parent partitions 的 interval partition 都必须在每个不同的
parent boundary 处分裂，因此 maximal common refinement 唯一。\(\square\)

### 引理 C：Local Compiler Soundness and Boundary Completeness

给定一个 refined parent cell \(I\)，其 parent vector 为
\(x(t)=At+c\)。L、T、N 三类 local compiler 都返回有限、disjoint、exhaustive
的 pieces；每个 piece 上的 affine value 与 provenance 均等于 fitted mechanism
的精确输出，并且所有可能改变 active affine rule 的 interior boundaries 都被
枚举。

**证明。** 分三类。

**L。** 对 \(f(x)=w^{\mathsf T}x+b\)，

\[
f(x(t))+n_v=(w^{\mathsf T}A)t+(w^{\mathsf T}c+b+n_v).
\]

它在 \(I\) 上只有一个 affine rule，不产生新 boundary。对应 provenance 是
`BIAS_SLOT`、`NOISE_SLOT` 与所有 `PREFIX(PARAM_SLOT(w_i),P_i)` 的 `ADD`，value homomorphism
直接给出上式。

**T。** 在 numerical regression tree 的任意 reachable node，predicate
\(x_k(t)\le\theta\) 是一元 affine inequality。若 \(A_k=0\)，它在当前 interval
恒真或恒假；否则唯一候选 crossing 是
\(t=(\theta-c_k)/A_k\)。把当前 interval 与该 half-line 及其 complement 相交，
得到至多两个 intervals。有限树上的递归因此终止；引理 A 保证 nonempty leaf
intervals 精确 partition \(I\)。每个 leaf value 是 constant，故是 slope 为零的
affine piece。Additive ensemble 对各 tree partitions 做 common refinement 并
相加，所以仍 finite 且 exact。Tree predicates 进入 certificate；leaf、ensemble
weight 和 noise 进入 response provenance。

**N。** 对 feed-forward ExactLine/PWL inference DAG 按 operator order 归纳。
Affine operator 不增加 boundary。对每个当前 affine preactivation
\(z(t)=at+b\)，ReLU 只可能在 \(a\ne0\) 时的唯一零点改变 rule；compiler 在该
零点切分并分别返回 `ZERO` 或输入 provenance。Max/MaxPool 在当前 cell 内只会
在有限个 pairwise affine equalities 处改变 maximizer，排序这些 crossings 后
应用固定 tie policy。每个其他被接纳的 ExactLine operator 必须提供同样的
finite exact-crossing contract；否则不属于 Class N。有限 inference DAG 上反复
细化仍然有限。Affine provenance propagation 使用 `ADD/PREFIX`，selection
operator 选择对应 input root 或 `ZERO`，所以 value 与 fitted graph 一致。

在三类中，任意 guard 在两个连续候选 crossings 之间都不能改变 truth value；
故枚举不仅 sound，而且 boundary-complete。\(\square\)

### 定理 1：L/T/N Global Closure and Exactness

设 SCM 是有限 acyclic additive-noise graph，所有相关 mechanisms 属于 L/T/N。
固定 unit、一个 bounded scalar intervention 和 deterministic branch semantics。
则每个 relevant node 都有 finite、possibly discontinuous、piecewise-affine
guarded-provenance response；topological compiler 返回的 response 在所有
\(t\in D\) 上等于 Pearl abduction-action-prediction 的结果。

**证明。** 对 causal topological order 归纳。Intervened root 是 \(t\)，未干预
root 是 unit factual constant，均为单 piece response。对非 root node，归纳
假设给出所有 parent responses；引理 B 构造 finite common refinement，引理 C
在每个 refined cell 上精确编译 local mechanism，最后加上固定 abducted-noise
atom。有限个 finite outputs 的拼接仍 finite。每一步都实现该 node 的 structural
equation，因此归纳终点等于使用同一组 abducted noises 的 counterfactual replay。
Tree jumps 被 interval endpoints 保留，证明不要求 continuity。\(\square\)

### 定理 2：Tree-Ensemble Non-Cartesian Bound

在一个 connected parent cell 上，若 additive ensemble 的第 \(r\) 棵树沿
intervention line 产生 \(b_r\ge1\) 个 nonempty intervals，则 ensemble common
refinement 至多包含：

\[
1+\sum_{r=1}^{T}(b_r-1)
\]

个 intervals，而不需要枚举 \(\prod_r b_r\) 个 leaf combinations。

**证明。** 第 \(r\) 个 ordered partition 有恰好 \(b_r-1\) 个 interior
boundaries。common refinement 的 interior boundaries 是这些集合的 union，其
大小至多为 \(\sum_r(b_r-1)\)。一个 connected interval 被 \(q\) 个不同 interior
points 切成 \(q+1\) pieces，故得上界。逐 tree breakpoint merge 直接构造该
refinement；leaf Cartesian product 不是必要步骤。\(\square\)

### 定理 3：Parameterized Provenance-DAG and Unit-Binding Soundness

对 `ZERO/ATOM/PREFIX/ADD/BUNDLE` DAG 的任意 semiring valuation，DAG root 的
值等于其逻辑展开 polynomial。若 guarded factors exact-partition occurrence-
lifted population matrix 的 `1` cells，则 factor roots 在每个 cell `(u,j)` 上的和
等于原 parameterized provenance。应用任意 unit binding `h_u` 后：

\[
h_u\!\left(\bigoplus_f M_{(u,j),f}F_f\right)
=h_u\!\left(
\bigoplus_{q:(u,j)\in G_q}\bigoplus_{f\in C_q}F_f
\right).
\]

因此 arbitrary per-unit weights、biases、roots 和 noises 不改变 exactness；该等式
再经任意 value/lineage homomorphism 后仍保持。

**证明。** 第一部分对 DAG 拓扑序归纳：`ZERO/ATOM` 是 base cases，`PREFIX`
实现一次 semiring multiplication，`ADD` 与语义相同的 `BUNDLE` 实现 finite
semiring addition。第二部分固定 matrix cell；exact partition 使每个 lifted
symbol 在对应 population cell 中出现恰好一次，其他 cells 中零次，所以两边 formal
polynomial 的每个 coefficient 相同。Semiring homomorphism 保持两种运算，
因此 quotient、`h_u` 和查询 valuation 均保持等式。\(\square\)

## 3. MGPF

### 定义 3：Guarded Factor

一个 factor 是：

\[
q=(G_q,C_q),
\quad G_q\subseteq\Omega,
\quad C_q\subseteq[m].
\]

它对应 binary rank-one matrix：

\[
R_q=\mathbf 1_{G_q}\mathbf 1_{C_q}^{\mathsf T}.
\]

factor family `Q` exact，当且仅当：

\[
M=\sum_{q\in\mathcal Q}R_q
\tag{1}
\]

在 ordinary integer arithmetic 下逐 entry 成立。由于 `M` 为 binary，式
(1) 表示每个 `1` cell 被恰好一个 rectangle 覆盖，所有 `0` cells 不被覆盖。

定义：

\[
F(\mathcal Q)=|\mathcal Q|.
\]

这里的 population run cost 定义为：

\[
\operatorname{runs}_{\mathcal U}(G)
=\sum_{u\in\mathcal U}\operatorname{runs}_u
\bigl(\{j:(u,j)\in G\}\bigr),
\qquad
K(\mathcal Q)=\sum_q\operatorname{runs}_{\mathcal U}(G_q).
\]

该定义明确禁止跨 unit chain 合并相邻 bit positions。

若两个 factors 有相同 payload set \(C_q\)，exactness 保证其 guards 不相交，
因此可把 guards 取 union 并合并为一个 factor。故 factor-root optimum 中可以
假设 payload sets 互不相同。

### 引理 2：Lifted Factorization Soundness

若 `Q` 满足式 (1)，则对每个 population cell `(u,j)`：

\[
\bigoplus_f M_{(u,j),f}\otimes F_f
=
\bigoplus_{q:(u,j)\in G_q}\bigoplus_{f\in C_q}F_f
\]

在 \(\widetilde{\mathcal K}\) 中成立。应用 \(\pi\) 后，它也在原 provenance
semiring 中成立；再
复合 `h_u` 以及任何 value 或 lineage homomorphism，等式仍成立。

**证明。** 固定 `(u,j),f`。式 (1) 保证恰有一个 `q` 同时满足
\((u,j)\in G_q\) 与 \(f\in C_q\)，当且仅当 \(M_{(u,j),f}=1\)。因此两边每个自由生成元的
multiplicity 相同。free-semiring polynomial equality 随之成立。semiring
homomorphism 保持 \(\oplus\) 和 \(\otimes\)，所以后两步成立。\(\square\)

## 4. 一般困难性与 Lower Bound

### 定理 4：MGPF-ROOT NP-Complete

Decision problem `MGPF-ROOT(M,k)` 问是否存在 exact `Q` 使
`F(Q)<=k`。该问题 NP-complete。

**证明。** 它属于 NP，因为可在 polynomial time 检查所有 rectangle 是否只
覆盖 `1` cells，并检查每个 `1` cell 的覆盖 multiplicity 是否为 1。

从 Minimum Biclique Edge Partition 归约。给定 bipartite graph
`G=(U,V,E)`，令 `M` 为其 adjacency matrix。一个 all-one rectangle
`G_q x C_q` 正好对应 `G` 的一个 biclique；式 (1) 正好要求这些 bicliques
partition \(E\)。映射保持 factor 数，且双向显然成立。任意 row subset 都可在
population domain 中表示：让 graph 的每个 row 成为一个 `B_u=1` 的 unit。
所有 row subsets 都是 singleton unit-runs 的 union，所以 multi-chain order
不影响该归约。Minimum Biclique Edge Partition 的困难性见
<https://doi.org/10.1587/transinf.E93.D.290>。\(\square\)

**范围。** 该证明针对 factorizer 接受的一般 support matrices。它不声称某个
更窄的固定 ReLU architecture 仍然 NP-hard。单棵树的 laminar subclass 将在
第 5 节精确求解。

### 定理 5：Real-Rank Lower Bound

任意 exact factorization 满足：

\[
F(\mathcal Q)\ge\operatorname{rank}_{\mathbb R}(M).
\]

**证明。** 每个 `R_q` 是两个 vectors 的 outer product，因此 real rank 至多
为 1。由 rank subadditivity：

\[
\operatorname{rank}_{\mathbb R}(M)
=\operatorname{rank}_{\mathbb R}\left(\sum_qR_q\right)
\le\sum_q\operatorname{rank}_{\mathbb R}(R_q)
\le F(\mathcal Q).
\]

\(\square\)

rank lower bound 不总是 tight，但可在无法运行 ILP 的大实例上给出 certified
gap `F/rank(M)`。

## 5. Laminar Exact Factorization

对一个 lifted column `f` 定义 support：

\[
S_f=\{(u,j)\in\Omega:M_{(u,j),f}=1\}.
\]

删除 empty-support columns；它们不出现在任何 region polynomial 中。相同
supports 的 columns 组成 `LOCAL(v)`。distinct nonempty supports 形成 family
\(\mathcal S\)。若任意 \(A,B\in\mathcal S\) 都满足 disjoint、
\(A\subseteq B\) 或 \(B\subseteq A\)，则 \(\mathcal S\) laminar。

把 distinct supports 建成 inclusion forest。节点 `v` 的 children 是 maximal
proper support subsets，并定义 residual atom：

\[
A_v=S_v\setminus\bigcup_{c\in child(v)}S_c.
\tag{2}
\]

children 两两 disjoint。

### 引理 3：Laminar Atom Rank

对 laminar support matrix：

\[
\operatorname{rank}_{\mathbb R}(M)
=|\{v:A_v\ne\varnothing\}|.
\tag{3}
\]

**证明。** 所有 nonempty \(A_v\) 两两 disjoint，所以 vectors
\(\mathbf 1_{A_v}\) 非零且
线性独立。每个 support 是 subtree residual atoms 的 disjoint union：

\[
\mathbf 1_{S_v}
=\sum_{u\in subtree(v):A_u\ne\varnothing}\mathbf 1_{A_u}.
\]

所以 support-column span 包含于 atom span。反向，由式 (2)：

\[
\mathbf 1_{A_v}
=\mathbf 1_{S_v}-\sum_{c\in child(v)}\mathbf 1_{S_c}.
\]

所以每个 atom vector 也在 support-column span 内。两者 span 相同，而
nonempty atom vectors 独立，故 rank 等于其数量。duplicate support columns
不改变 column span。\(\square\)

### 算法 1：RUN-AWARE-LCF-BLOCK

`BUNDLE(x,y)` 的语义是 \(x\oplus y\)，但物理上不 flatten `x` 的 children。

```text
RUN_AWARE_LCF_BLOCK(block):
  group equal supports
  build the inclusion forest
  for each root v:
    FACTOR(v, ZERO)

FACTOR(v, inherited):
  bundle = BUNDLE(inherited, LOCAL(v))
  A_v = S_v minus union(S_c for c in child(v))

  if A_v is empty:
    for c in child(v):
      FACTOR(c, bundle)
    return

  J = MIN_RUN_RESET(A_v, child(v))
  G_v = A_v union union(S_c for c in J)
  emit (guard=G_v, payload=bundle)

  for c in child(v):
    if c in J:
      FACTOR(c, ZERO)
    else:
      FACTOR(c, bundle)

MIN_RUN_RESET(A_v, children):
  return argmin_{J subseteq children}
         runs_U(A_v union union(S_c for c in J))
```

`MIN_RUN_RESET` 不枚举 \(2^{|child(v)|}\) 个 subsets。对每个 child \(c\) 建一个
跨所有 unit chains 共享的 binary variable \(x_c\)。对 unit `u` 的 ordered
region `j`，定义：

\[
z_{u,j}=
\begin{cases}
1,&(u,j)\in A_v,\\
x_c,&(u,j)\in S_c,\\
0,&(u,j)\notin S_v,
\end{cases}
\qquad z_{u,0}=0\quad\forall u.
\]

候选 guard 的 run 数为：

\[
\operatorname{runs}_{\mathcal U}(G_v)
=\sum_{u\in\mathcal U}\sum_{j=1}^{B_u}
z_{u,j}(1-z_{u,j-1}).
\tag{4}
\]

把 fixed `1` 映射到 source、fixed `0` 映射到 sink，并为每个 unit 内的相邻 pair
加一条 unit-capacity directed edge \(z_{u,j}\to z_{u,j-1}\)。若 variable 取 1
表示位于 source side，则该 edge 跨 cut 当且仅当
\(z_{u,j}=1,z_{u,j-1}=0\)。每条 unit chain 都独立从 fixed zero 开始，因此不会
跨 units 合并 runs。minimum \(s\)-\(t\) cut 的值恰为式 (4)，source side 的
child variables 就是同时最小化所有 unit-chain runs 总和的最优 \(J\)。

这一步只优化一个明确的 **hierarchical reset family**：当前 factor 必须包含
\(A_v\)，可选择吸收若干完整 immediate-child supports；被吸收的 children 从
`ZERO` 继续，其余 children 继承 `bundle`。本文不声称该 family 包含所有可能的
rank-optimal rectangle partitions。旧算法是 \(J=child(v)\)，pure-atom variant
是 \(J=\varnothing\)。

### 引理 4：Inherited-Payload Invariant

调用 `FACTOR(v,H)` 时，假设 `H` 中每个 lifted fragment 都应在 `S_v` 上恰好
贡献一次。算法输出后：

1. `H` 与 `LOCAL(v)` 在 `S_v` 上各贡献一次；
2. subtree descendants 的 fragments 在各自 supports 上各贡献一次；
3. 其他 cells 上没有贡献。

**证明。** 对 inclusion-forest height 归纳。

若 \(A_v\) nonempty，算法在
\(G_v=A_v\uplus\bigcup_{c\in J}S_c\) 上 emit bundle。对 \(c\in J\)，bundle
已在整个 \(S_c\) 上贡献一次，所以 child 从 `ZERO` 开始；对 \(c\notin J\)，
child 继承 bundle。Residual atom 与所有 child supports 两两 disjoint，并共同
partition \(S_v\)，故 bundle 在 \(S_v\) 上恰好贡献一次。每个 child recursion
仍处理自己的 LOCAL 和 descendants。

若 \(A_v\) empty，则 children 两两 disjoint且 union 为 \(S_v\)。算法把 bundle
传给每个 child；由归纳假设，该 payload 在每个 child support 上恰好出现一次，
所以在整个 \(S_v\) 上也恰好出现一次。其他 cells 不在任何 emitted guard 中。
\(\square\)

### 定理 6a：Laminar Factor-Root Optimality

RUN-AWARE-LCF-BLOCK 精确，并且：

\[
F_{LCF}
=F^*
=\operatorname{rank}_{\mathbb R}(M)
=|\{v:A_v\ne\varnothing\}|.
\]

**证明。** 由引理 4，算法 exact。算法仅当 `A_v` nonempty 时 emit 一个
factor，所以 factor 数等于 nonempty atoms 数。由引理 3，该数量等于 matrix
rank；由定理 5，任何 exact factorization 至少需要这么多 factors。因此 LCF
达到全局 optimum。\(\square\)

### 定理 6b：Run-Aware Interval Bound and Reset-Family Optimality

令：

\[
R=\sum_{v\in\mathcal S}\operatorname{runs}_{\mathcal U}(S_v).
\]

再令 \(\mathcal V_+=\{v:A_v\ne\varnothing\}\)，并定义：

\[
\kappa_v=
\min_{J\subseteq child(v)}
\operatorname{runs}_{\mathcal U}\left(
A_v\cup\bigcup_{c\in J}S_c
\right),
\]

\[
R_S=\sum_{v\in\mathcal V_+}\operatorname{runs}_{\mathcal U}(S_v),
\qquad
R_A=\sum_{v\in\mathcal V_+}\operatorname{runs}_{\mathcal U}(A_v).
\]

Run-aware LCF 输出满足：

\[
K_{LCF}=\sum_{v\in\mathcal V_+}\kappa_v
\le \min(R_S,R_A)
\le R.
\tag{5}
\]

而且它在上述 hierarchical reset family 中使 \(K\) 全局最小。

**证明。** 式 (4) 的 directed min-cut 精确求得每个 \(\kappa_v\)。每个 nonempty
atom node 恰好 emit 一个 factor，因此总 run cost 是各 node guard costs 的和。
一个 child 从 `ZERO` 还是 inherited bundle 开始，只改变 payload pointer，不改变
该 child 可选的 guards 或 run costs，所以各 node 的 reset choices 相互独立；
逐 node min-cut 因而给出 reset family 内的 global minimum。

取 \(J=child(v)\) 得到 \(G_v=S_v\)，取 \(J=\varnothing\) 得到 \(G_v=A_v\)。
故 \(\kappa_v\le\min\{\operatorname{runs}_{\mathcal U}(S_v),
\operatorname{runs}_{\mathcal U}(A_v)\}\)，求和得到前两个不等式；最后
\(R_S\le R\)。
\(\square\)

### 定理 6c：Laminar Bundle-DAG Bound

设 `d` 为 distinct supports 数，`m` 为 lifted occurrence-support layers 数。
除 canonical fragment DAG 外，LCF 需要：

```text
O(d) BUNDLE nodes
O(m) LOCAL membership references
```

**证明。** 每个 inclusion node 计算一次
`BUNDLE(inherited,LOCAL(v))`，所以 BUNDLE nodes 至多 `d`。每个 lifted
column 恰好属于一个 equal-support LOCAL group，所以总 memberships 为 `m`。
Non-flattening 保证 inherited subtree 通过一个 pointer 共享，而不复制其全部
children。Min-cut graphs 只在 construction 时存在，不进入 factor DAG。
\(\square\)

## 6. Laminar-Cover Approximation

定义 support crossing graph \(H_\times\)：vertices 为 distinct supports；若
\(A\cap B\ne\varnothing\) 且二者互不包含，则连边。proper coloring 的每个
color class 都是 laminar family。

### 算法 2：LCF

1. 对 equal supports 分组。
2. proper-color \(H_\times\)，得到 blocks \(M_1,\ldots,M_\ell\)。
3. 对每个 block 运行 RUN-AWARE-LCF-BLOCK。
4. 合并输出，并只执行对 \(F,K,D\) 均不增加的 exact rewrites。

### 定理 7：Data-Dependent Approximation

若算法实际产生 `ell` 个 blocks，则：

\[
F_{LCF}\le\ell F^*,
\qquad
K_{LCF}\le\min(R_S,R_A)\le R,
\]

其中 \(R_S,R_A,R\) 在各 color blocks 的 inclusion forests 上求和。

**证明。** 对 block `b`，定理 6a 给出 `F_b=F_b^*`。取 global optimum
factorization `Q*`，把每个 factor payload 限制到 block `b` 的 columns，并
删除 empty factors。guards 不变，且每个 block cell 仍恰好被覆盖一次，所以
得到 `M_b` 的合法 factorization，factor 数至多 `F*`。因此
`F_b^*<=F*`。求和：

\[
F_{LCF}=\sum_{b=1}^{\ell}F_b^*
\le\ell F^*.
\]

每个 distinct support 只属于一个 block。由定理 6b 逐 block 求和，并使用
\(\sum_b\min(x_b,y_b)\le\min(\sum_bx_b,\sum_by_b)\)，得到
\(K_{LCF}\le\min(R_S,R_A)\le R\)。coordinate-wise non-increasing rewrites
保持这些 bounds。\(\square\)

### 推论 1：三类机制与 Population Heterogeneity

- 对一个 fixed linear unit，`B_u=1`，所有 present fragments 的 supports equal，
  因而 `ell=1`。
- 对一个 fixed unit 的单棵 regression tree，node/path guards nested 或 disjoint，
  因而该 per-unit family 有 `ell=1`。
- 若所有 units 共享同一个 structural template，上述 laminar 性可以延伸到对应
  population supports。
- 对 arbitrary per-unit templates，即使 mechanism 是 Linear 或 Tree，两个
  fragment supports 也可能因不同 unit incidence 而 crossing；ReLU 更无 constant
  bound。算法始终 exact，但 worst-case `ell=d`。

Greedy coloring 不需要逼近 chromatic number。定理 7 使用实际输出的 `ell`。

## 7. Canonical Response Quotient and Prefix Storage

### 定理 8：Canonical Response-Segment Quotient

在 primitive response partition 上，若且唯若两个相邻 cells 的 normalized
provenance root、\((a,b)\) 和 endpoint-compatible response semantics 相同，
将它们合并。最终 maximal equivalent runs 唯一，并在“每个 interval 只允许一
个 response annotation”的 exact one-dimensional index class 中使用最少的
intervals。

**证明。** 等价 annotation 在相邻 cells 上定义一条 path 的 edge relation；
其 maximal connected components 唯一。每个 component 可安全合并，因为同一
affine line 和同一 provenance root 在 union 上有效。反之，任何 exact single-
annotation interval 都不能跨过两个 non-equivalent adjacent cells，否则至少有
一个 annotation field 错误。因此每个 maximal component 至少需要一个 index
interval，而该构造恰用一个。Certificate annotation 不在此等价关系中，所以
更细 audit partition 不影响该最优性。\(\square\)

### 安全引理：Designated Reverse-Trie Size

对一个 designated template root \(r\)，令 \(\operatorname{Pref}(L_r)\) 为其
finite target-oriented path language 的所有 nonempty prefixes。未做 cross-root
hash-consing 的 reverse trie 恰有

\[
1+|\operatorname{Pref}(L_r)|
\]

个 nodes；bottom-up hash-consing 只识别相同 labeled subtries，因此不增加总
nodes，并保持每个 designated root 的 path language。

**证明。** Trie root 对应 empty prefix，其他 node 与一个 distinct nonempty
prefix 一一对应。Hash-consing 是对等价 subtrees 取 quotient；designated root
pointers 保留后，可达 labeled paths 不变且 quotient 不会增加 equivalence
classes。该引理不声称 trie 在 arbitrary circuits 或 multi-root automata 中全局
最小。\(\square\)

## 8. Guard Intervalization

### 引理 5：Canonical Maximal Runs

固定 factor guard \(G\subseteq\Omega\)。对每个 unit `u`，其 projection
\(G_u=\{j:(u,j)\in G\}\) 在该 unit region order 上的 maximal connected
components 是唯一的 minimum exact interval decomposition。全体 unit projections
的 union 是 `G` 的唯一 minimum exact multi-chain interval decomposition。

**证明。** 对每条 unit chain 独立应用一维结论：每个 connected component 至少
需要一个 interval；一个 interval 不能跨过 `G_u` 外的 region，也不能跨 unit。
maximal components 本身给出可行 decomposition。对 units 求和，lower bound
可达且 decomposition 唯一。\(\square\)

### 定理 9：Dyadic Separation

固定 factors 后，stabbing directory 存 `K` 个 maximal unit-guard runs。standard
per-unit dyadic segment-tree decomposition 最坏使用 `O(K log B_max)` references，
而存在单个 interval 已产生 `Theta(log B_max)` 对 1 的 separation。

**证明。** unit `u` 的每个 interval canonical dyadic decomposition 使用
`O(log B_u)` nodes，所以总数 `O(K log B_max)`。令某个 `B_u=2^h`，regions 为
`0,...,B_u-1`。interval `[1,B_u-1]` 分解为 sizes
`1,2,4,...,B_u/2` 的 `h` 个 dyadic nodes，所以需要 `Theta(log B_u)` references；
maximal-run directory 只存一个 interval。对独立 index keys 复制该构造，得到
total \(\Theta(K\log B_{max})\) family。\(\square\)

## 9. Semigroup Index Mapping

### 定理 10：Stabbing-Semigroup Correctness and Bound

每个 maximal guard run `(u,I)` 的 weight 是 factor payload polynomial `Q_q`。
population cell `(u,j)` 的 stabbing query 为：

\[
\bigoplus_{(u,I):j\in I}\omega(u,I)
=\bigoplus_{q:(u,j)\in G_q}Q_q
=\widetilde P_{u,j}.
\]

第一步是 physical interval expansion，第二步来自每个 factor guard 的 run
partition，第三步来自引理 2。\((\mathbb N[Z],\oplus)\) 是 commutative
monoid，因此不需要 subtraction、inverse 或 idempotence。故采用已知的
linear-space stabbing-semigroup structure 后，所有 directories 共使用 \(O(K)\)
space，unit `u` 的 cold aggregate-root query 使用 \(O(\log K_u)\) time。之后应用
`h_u` 得到 exact unit provenance/value。

该 stabbing-semigroup result 及其 dynamic pointer-machine bounds 见：
<https://doi.org/10.1137/10078791X>。TESSERA 使用该 result，但不把它作为自身
novelty。

## 10. Proof Obligations Before Submission

以下内容尚未因本文而自动完成：

1. 把 real-arithmetic endpoint policy 映射到每种 stored fitted predictor 的
   floating-point branch behavior，并以 boundary tests 验证。
2. 对 Class N 中除 affine/ReLU/Max 以外的每个 operator 单独实例化
   exact-crossing contract；未实例化者不得列入支持表。
3. 系统检索 laminar-neighborhood biclique partition 与 support-family coloring
   literature，确认定理 6a 和定理 7 的 novelty 表述。
4. 检查 weighted multiplicity implementation 是否与 occurrence-lifting quotient
   完全一致。
5. 用 checked-in exhaustive reference verifier 对拍生产版 directed min-cut 与
   factorizer；当前 verifier 已覆盖小矩阵上的 exactness、factor bound 和 run bound。
6. 在 paper 中严格区分 factor-root approximation、guard-run bound、DAG size
   bound 与 serialized bytes。

在这些 obligations 完成前，可以声称 theorem design 已闭合，但不能声称所有
appendix proofs 与 related-work novelty audit 已完成。
