# TESSERA 技术设计

状态：VSC 修订进行中。本文档原有主体记录了 population MGPF/LCF 方案；该方案
仍可作为 additive frontier 的二级 algebraic optimizer，但不再是 general storage
layer。当前 general index 是 Versioned Semiring Circuit (VSC)。后续章节中把
support matrix 写成所有机制的统一逻辑表示、或把 LCF 写成主索引的段落，均应按
本节修订理解，等待逐节重写。

# 第零部分：VSC 修订

## 0.1 为什么修改主索引

旧表示

\[
P_{u,j}=\bigoplus_f M_{(u,j),f}F_f
\]

要求 region provenance 已经被切成 additive fragments。它适合 Linear paths、
tree contributions 和某些 ReLU frontier，但一般深层 ReLU provenance 同时包含
`ADD` 与 `MUL`。若先展开成 neural paths，逻辑 polynomial 可能指数大；若不展开，
support matrix 并没有完整表示 circuit sharing。

因此 general object 改为有序的完整 semiring circuit versions：

\[
P_{u,1},P_{u,2},\ldots,P_{u,B_u}.
\]

Staircase、laminar、prefix/suffix 都只是这组 versions 可能具有的特殊结构，不再是
索引正确性或可构建性的前提。

## 0.2 Versioned Semiring Circuit

全局 immutable arena 支持：

```text
ZERO
ONE
ATOM(label)
ADD(children)
MUL(children)
SWITCH(selector, on, off)
```

节点以 `(operator, child_ids, payload)` 为 key 全局 hash-cons。`ADD` 不执行
idempotent 去重；相同 child 出现两次仍表示 multiplicity 2。每个 unit 保留自己的
parameter/root/noise binding `h_u`，所以共享 root 不等于共享数值。

每个机制 family 编译出一个固定 circuit template，其中 mechanism fragments、leaf
choices 或 activation gates 是 mutable slots。对 region boundary `j -> j+1`，compiler
输出变更集合 `Delta[u,j]`。VSC 只更新这些 slots，并沿 template parent links 重建受
影响 ancestors；未改变的 node IDs 原样继承。每个 region 最终只保存：

```text
(left_boundary, numeric affine payload, provenance_root_id)
```

不同 regions 或 units 若产生相同 canonical subcircuit，会自动复用同一 node ID。
这同时覆盖相邻 prefix sharing、非相邻 pattern recurrence 和 cross-unit structural
sharing。

## 0.3 构建算法

```text
BUILD_VSC(unit, exact_regions, template):
    state = INSTANTIATE(template, exact_regions[0].slot_values)
    roots = [state.root]

    for boundary j in increasing order:
        changed_slots = DIFF(exact_regions[j-1], exact_regions[j])
        affected = ANCESTOR_CLOSURE(template, changed_slots)

        for node in TOPOLOGICAL_ORDER(affected):
            state[node] = INTERN(op[node], state[children[node]])

        roots.append(state.root)

    return RegionDirectory(boundaries, roots)
```

实现不需要构造完整 region-fragment matrix，也不需要先判断 supports 是否
staircase/laminar。

## 0.4 一般复杂度参数

记：

```text
B_tot       所有 units 的 exact response regions 总数
D_0         初始 canonical circuit/template 节点数
alpha[u,j]  boundary j 上 changed slots 加被重新计算的 template ancestors 数
gamma[u,j]  其中真正新建的 canonical arena nodes 数；gamma[u,j] <= alpha[u,j]
H_u         unit binding h_u 的大小
```

当前可直接证明的实现上界为：

\[
S_{VSC}
=O\!\left(
B_{tot}+D_0+\sum_{u,j}\gamma_{u,j}+\sum_u|H_u|
\right).
\]

\[
T_{build}
=O\!\left(
B_{tot}+D_0+\sum_{u,j}\alpha_{u,j}+\sum_u|H_u|
\right)
\]

再加 boundary sorting 与 hash-table expected-time cost。区分 `alpha` 和 `gamma`
很重要：一个被重新计算的 node 可能被 hash-cons 到早先已经存在的 node ID，因此
增加构建工作但不增加最终索引空间。

region lookup 二分 boundaries 后返回一个 root pointer：

\[
T_{root}=O(\log B_u),
\qquad
T_{expanded}=O(\log B_u+|output|).
\]

对含 `C` 个 mutable leaves 的 balanced associative block，若一次 boundary 同时改变
`k_j` 个 leaves，它们到 root 的 inclusive path union 大小满足
`THEORY_PROOFS.md` 引理 0.2 的
bound：

\[
\alpha_j
=O\!\left(k_j\log\left(1+\frac{C}{k_j}\right)\right).
\]

同时 `gamma_j <= alpha_j`，所以该式也给出新节点数的上界。

一般 CDAG 使用实际 affected canonical cone 作为 `alpha[u,j]`。最坏情况下
`alpha[u,j]=Theta(D_0)` 且所有结果都可能是新节点，因此 VSC 不声称 universal
compression。

## 0.5 三类机制映射

```text
Linear:
    one fixed version; intervention/root/noise remain binding slots

Tree Ensemble:
    balanced semiring SUM template; a threshold replaces selected leaf roots

ReLU/PWL:
    activation selectors are mutable slots inside ADD/MUL circuit templates
    arbitrary OFF/ON/OFF sequences are legal
```

完整 fitted MLP 的下一步实现是 semiring selector matrix/product tree。固定 activation
pattern `j` 时：

\[
P_{u,j}
=\widetilde W_L\widetilde D_{u,j,L-1}\widetilde W_{L-1}\cdots
\widetilde D_{u,j,1}\widetilde W_1\widetilde X_u,
\]

其中 `W` globally shared，`D` 是 persistent activation-selector versions；bias 用
homogeneous coordinate 表示。该结构保留 matrix product circuit，不展开 neural
paths。

## 0.6 MGPF/LCF 的新位置

在一个明确选定的 additive frontier 上，仍可把 frontier roots 写成 support matrix，
并运行 occurrence lifting、MGPF、LCF、reset min-cut 和 stabbing-semigroup index。
这些结果继续有效于它们已经声明的表示类，但不能替代 general `ADD/MUL` circuit
semantics。

新的两级关系是：

```text
general exact layer:       VSC canonical persistent sharing
optional algebraic layer:  weighted guarded factorization at additive frontiers
```

后续理论需要把 factor count、payload references 和 guard runs 合并到同一个 weighted
storage objective，并证明 restricted counterfactual-realizable hardness。任意 binary
matrix 的 biclique-partition hardness 只作为一般边界，不再承担主要 novelty。

# 第一部分：三类函数的反事实响应

## 1. 共同问题：我们究竟要计算什么

给定一个 acyclic additive-noise SCM。为避免把 cross-unit sharing 偷写成假设，本文直接使用最一般的 unit-indexed 记号：

```text
X^u_v = f^u_v(X^u_pa(v)) + N^u_v
```

标准 fitted SCM 中通常有 `f^u_v=f_v`；本文的算法与证明不依赖这个相等关系。不同 units 可以有不同参数、不同 breakpoints，甚至不同 canonical templates；能够共享的结构由索引发现，不能共享的结构保留为 singleton。对于每个已观测 unit 和非 root 节点，先从 factual observation 中恢复 noise：

```text
n[u,v] = x[u,v] - f^u_v(x[u,pa(v)])
```

之后固定这些 noises，对 treatment 执行标量干预 `do(T=t)`。我们要计算的不是某一个固定 `t` 对应的答案，而是 target `Y` 在整个有界干预域 `D=[t_min,t_max]` 上的函数：

```text
Pi[u,Y](t) = Y_{u;do(T=t)}
```

普通 counterfactual replay 每收到一个 `(u,t)`，都会沿该 unit 的 causal DAG 重新执行一遍 prediction。TESSERA 的目标是一次性推导整个 indexed population：

\[
\mathcal P=\{\Pi_{u,Y}(t):u\in\mathcal U,\ t\in D_u\},
\]

以后重复查询只做 unit-local interval lookup、factor stabbing 和 provenance evaluation。这个定义同时包含两种共享：一个 unit 内不同 intervals 的共享，以及不同 units 之间的共享。

下面分别从三类 `f_v(.)` 推导这条曲线。

## 2. Linear Mechanism：整条曲线是一条直线

### 2.1 一般推导

考虑节点 `v` 对 unit `u` 的 linear mechanism。权重和截距明确允许随 unit 改变：

```text
X^u_v = c[u,v] + sum_{p in pa_u(v)} w[u,v,p] * X^u_p + N^u_v
```

对于 unit `u`，abduction 后的 `n[u,v]` 已经固定。假设当前每个 parent 的反事实响应是：

```text
Pi[u,p](t) = a_p*t + b_p
```

代入结构方程：

```text
Pi[u,v](t)
  = c[u,v] + n[u,v] + sum_p w[u,v,p] * Pi[u,p](t)

  = c[u,v] + n[u,v] + sum_p w[u,v,p] * (a_p*t + b_p)

  = (sum_p w[u,v,p]*a_p) * t
    + (c[u,v] + n[u,v] + sum_p w[u,v,p]*b_p)
```

因此：

```text
alpha[u,v] = sum_p w[u,v,p]*a_p
beta[u,v]  = c[u,v] + n[u,v] + sum_p w[u,v,p]*b_p

Pi[u,v](t) = alpha[u,v]*t + beta[u,v]
```

Linear mechanism 不产生新的 breakpoint。只要 parents 在当前区间内是 affine，child 在该区间内也一定是 affine。

数值不同不妨碍结构共享。令 `THETA(v,p)`、`BIAS(v)`、`NOISE(v)` 是 parameterized provenance slots，并为每个 unit 定义 semiring homomorphism：

\[
h_u(\Theta_{v,p})=w[u,v,p],\qquad
h_u(B_v)=c[u,v],\qquad
h_u(N_v)=n[u,v].
\]

全局 Prefix/semiring DAG 保存 slots 和运算结构；unit store 只保存 `h_u` 的 bindings。若两个 units 的结构不同，它们拥有不同 designated roots，但仍可 hash-cons 恰好相同的子结构。算法从不要求两个 units 共享完整 path、activation plan 或数值参数。

### 2.2 一个完整的 Linear SCM 例子

考虑：

```text
X = T + R + N_X
M = 2*X + N_M
Y = 3*M + N_Y
```

`R` 是不被干预的 root。对于某个 unit `u`，假设：

```text
R_u   = 2
n[u,X] = 1
n[u,M] = -2
n[u,Y] = 5
```

执行 `do(T=t)`：

```text
Pi[u,X](t)
  = t + 2 + 1
  = t + 3

Pi[u,M](t)
  = 2*(t+3) - 2
  = 2*t + 4

Pi[u,Y](t)
  = 3*(2*t+4) + 5
  = 6*t + 17
```

所以这个 unit 的整个 target response 只需要一个记录：

```text
(D, alpha=6, beta=17)
```

例如：

```text
POINT(u,Y,1) = 6*1 + 17 = 23
POINT(u,Y,4) = 6*4 + 17 = 41
```

不需要重新执行 `X -> M -> Y`。

### 2.3 Linear Provenance 记录什么

只保存 `6*t+17` 可以快速计算数值，但不能解释 `6` 和 `17` 从哪里来。因此同时保存：

```text
P_X = ADD(
        TAU(T),
        ROOT_SLOT(R),
        NOISE_SLOT(X)
      )

P_M = ADD(
        PREFIX(PARAM_SLOT(M,X), P_X),
        NOISE_SLOT(M)
      )

P_Y = ADD(
        PREFIX(PARAM_SLOT(Y,M), P_M),
        NOISE_SLOT(Y)
      )
```

对本例绑定 `h_u(PARAM_SLOT(M,X))=2`、`h_u(PARAM_SLOT(Y,M))=3`，再把 `TAU(T)` 绑定到查询值 `t`，就能重新得到 `6*t+17`。另一个 unit 可以把同一结构绑定到完全不同的权重；这个 provenance 还保留了：

- intervention 经过哪些 causal edges 到达 `Y`；
- 哪些 factual roots 和 abducted noises 参与结果；
- 每条路径上使用了哪些 model parameters。

因此 Linear 的最终对象是：

```text
一个全局 interval D
+ 一条 numeric line alpha*t+beta
+ 一个 provenance root P_Y
```

## 3. Tree Ensemble：整条曲线是多段常数

### 3.1 先看一棵 Tree

假设 parent response 已经推导为：

```text
Pi[u,X](t) = t + 3,  t in [0,4]
```

节点 `Y` 的 regression tree 为：

```text
if X <= 5:
    leaf_left  = 10
else:
    leaf_right = 20

Y = Tree(X) + N_Y
```

并且该 unit 的 `n[u,Y]=3`。

Tree split 在反事实世界中变成：

```text
t + 3 <= 5
```

唯一 crossing 为：

```text
t_star = 2
```

因此：

```text
Pi[u,Y](t) =
  13,  0 <= t <= 2
  23,  2 <  t <= 4
```

于是具体查询为：

```text
POINT(u,Y,1) = 13
POINT(u,Y,2) = 13
POINT(u,Y,3) = 23
```

对于 predicate `Y>=18`，状态变化发生在 boundary `t=2`：`t=2` 时仍为 false，任意紧邻右侧的 `t>2` 都为 true。这里没有满足 `Y(t)=18` 的点，这正是 Tree jump 与连续 affine crossing 的区别。

GPSI 要保存两条记录：

```text
([0,2], alpha=0, beta=13, provenance=P_left)
((2,4], alpha=0, beta=23, provenance=P_right)
```

Tree response 是 piecewise constant，而 constant 就是 slope 为 0 的 affine function。

### 3.2 Value Provenance 与 Path Certificate

左区间的 arithmetic provenance 为：

```text
P_left = ADD(
           NOISE_SLOT(Y),
           LEAF(tree_1, left)
         )
```

右区间为：

```text
P_right = ADD(
            NOISE_SLOT(Y),
            LEAF(tree_1, right)
          )
```

但是 parent `X` 没有出现在 arithmetic expression 中。它只决定系统选择哪个 leaf。因此还要保存独立的 certificate：

```text
C_left  = PRED(P_X, X <= 5)
C_right = PRED(P_X, X > 5)
```

这就是为什么 response provenance 和 guard certificate 不能混成一个结构：

- `P_left/P_right` 回答“结果的值由什么组成”；
- `C_left/C_right` 回答“为什么当前 leaf 有效”。

### 3.3 一般 Regression Tree 推导

一棵 tree `m` 可以写成：

```text
Tree_m(x)
  = sum_{leaf l}
      1[path_guard[m,l](x)] * leaf_value[m,l]
```

在一个 parent interval 内，每个输入维度都是：

```text
x_k(t) = a_k*t + b_k
```

Tree node 的 split `x_k <= theta` 变成：

```text
a_k*t + b_k <= theta
```

当 `a_k=0` 时，整个 interval 只会进入一个 branch。当 `a_k!=0` 时，最多只有一个 crossing：

```text
t_star = (theta-b_k)/a_k
```

因此 compiler 只需要沿 intervention trajectory 访问实际可达的 branches，而不需要遍历全部 leaves。

### 3.4 Tree Ensemble 推导

对于 additive tree ensemble：

```text
f_v(x) = base_v + sum_{m=1..M} eta_m * Tree_m(x)
```

每棵 tree 独立产生一个有序 breakpoint stream。系统取这些 breakpoint 的并集。在合并后的每个 interval `I_j` 上，每棵 tree 都选择固定 leaf `l_m(j)`：

```text
Pi[u,v](t)
  = n[u,v] + base_v
    + sum_m eta_m * leaf_value[m,l_m(j)]
  = beta_j
```

因此：

```text
alpha_j = 0
P_j = ADD(
        NOISE_SLOT(v),
        BIAS(base_v),
        PREFIX(PARAM_SLOT(eta_m), LEAF(m,l_m(j))) for every m
      )
```

如果 tree `m` 产生 `b_m` 个 intervals，所有 trees 的 common refinement 最多为：

```text
1 + sum_m (b_m-1)
```

而不是 `product_m b_m`。这就是 Tree Ensemble compiler 的第一项明确理论优势：只合并 line breakpoints，不枚举 leaf combinations。

### 3.5 Tree 的特殊问题：Jump

上面的例子在 `t=2` 从 13 跳到 23。它不经过 18，却会让 predicate `Y>=18` 从 false 变成 true。

所以 FLIP 不能只求：

```text
alpha*t+beta = threshold
```

还必须比较每个 segment boundary 两侧的 threshold state。这一点会直接影响后面的 event index 设计。

## 4. ReLU MLP：整条曲线是多段直线

### 4.1 一个可以手算的 MLP

考虑标量输入的一个隐藏层网络：

```text
f(x) = 2*ReLU(x-1) - ReLU(-x) + 1
Y = f(X) + N_Y
```

假设：

```text
Pi[u,X](t) = t
n[u,Y] = 3
D = [-2,2]
```

那么：

```text
Pi[u,Y](t)
  = 2*ReLU(t-1) - ReLU(-t) + 4
```

两个 ReLU 的状态分别在 `t=1` 和 `t=0` 改变。

当 `t<0`：

```text
ReLU(t-1) = 0
ReLU(-t)  = -t
Pi[u,Y](t) = t + 4
```

当 `0<=t<=1`：

```text
ReLU(t-1) = 0
ReLU(-t)  = 0
Pi[u,Y](t) = 4
```

当 `t>1`：

```text
ReLU(t-1) = t-1
ReLU(-t)  = 0
Pi[u,Y](t) = 2*t + 2
```

最终得到：

```text
Pi[u,Y](t) =
  1*t + 4,  -2 <= t < 0
  0*t + 4,  0 <= t <= 1
  2*t + 2,  1 < t <= 2
```

例如：

```text
POINT(u,Y,-1)  = 3
POINT(u,Y,0.5) = 4
POINT(u,Y,1.5) = 5
```

这与 Tree 的区别是：

- Tree 每个区间内通常是 constant；
- ReLU MLP 每个区间内可以有非零 slope；
- 两者都只需要保存 `(interval,alpha,beta)`。

### 4.2 一般 ReLU MLP 推导

在当前 parent interval 内，network input 是：

```text
x(t) = A*t + b
```

对于第 `l` 层，在某个固定 activation interval `I_j` 上，ReLU mask `D_j^(l)` 不变：

```text
A_j^(0) = A
B_j^(0) = b

A_j^(l) = D_j^(l) * W_l * A_j^(l-1)

B_j^(l)
  = D_j^(l) * (W_l * B_j^(l-1) + b_l)
```

最后一层输出仍然是：

```text
Pi[u,v](t) = alpha_j*t + beta_j
```

ExactLine 的职责是找出所有 activation/operator crossings，并返回这些 affine intervals。

### 4.3 MLP Provenance 不能只存 Effective Weight

如果只存最终的 `alpha_j,beta_j`，或者只存一个 effective linear layer，就会丢失原始 network 中哪些 weights、biases 和 active neurons 参与了结果。

因此在固定 activation mask 下，同时传播 symbolic provenance：

```text
P^(0) = parent provenance vector

Z^(l)[k]
  = ADD(
      BIAS(l,k),
      PREFIX(PARAM_SLOT(W_l[k,q]), P^(l-1)[q]) for every q
    )

P^(l)[k] =
  Z^(l)[k]  if gate (l,k) is active
  ZERO      otherwise
```

Activation mask 单独进入 certificate：

```text
C_j = AND(
        GATE(layer, neuron, active_or_inactive)
      )
```

所以 ExactLine 提供 numeric partition；TESSERA 负责 SCM abduction semantics、原始参数 provenance、cross-unit sharing 和 index。

# 第二部分：三类函数如何统一

## 5. 统一结论

三类机制的推导结果为：

| 机制 | 新区间来自哪里 | 每个区间内的函数 |
|---|---|---|
| Linear | 不产生新区间 | affine |
| Tree Ensemble | numerical threshold crossing | constant，即 affine 的特殊情况 |
| ReLU/PWL Neural | activation/operator crossing | affine |

因此，在一个标量 intervention line 上，三类机制都输出同一种记录：

```text
ResponsePiece(
    interval=I_j,
    alpha=alpha_j,
    beta=beta_j,
    provenance_root=P_j
)

CertificatePiece(
    interval=J_k,
    certificate_root=C_k
)
```

统一 compiler contract 为：

```text
COMPILE_CELL(
    mechanism,
    parent_affine_lines,
    parent_provenance_roots,
    interval,
    boundary_policy
) -> (
    sorted response pieces,
    sorted certificate pieces
)
```

三类 mechanism 的区别只存在于 `COMPILE_CELL` 内部。输出后全部进入同一个 causal compiler、provenance store 和 query engine。

### 5.1 先统一逻辑响应

对 unit `u`，compiler 产生 intervals `I[u,j]` 和该 interval 上的 active path/rule
occurrences `P[u,j]`。令每条 path `pi` 从一个 typed source atom 开始，包括
`TAU/ROOT/NOISE/BIAS/LEAF` 之一，之后带上沿传播路径出现的 parameter slots。
把 path 上全部 labels 统一记为 `Lambda`，则：

\[
\widetilde P_{u,j}
=\bigoplus_{\pi\in\mathcal P_{u,j}}
  \bigotimes_{\lambda\in\pi}\Lambda_\lambda,
\qquad
P_{u,j}=h_u(\widetilde P_{u,j}).
\]

这里的 path-expanded 写法只是逻辑语义；物理实现保存 hash-consed
Prefix/semiring DAG，不枚举所有 paths。长度为零的传播对应一个单独 typed atom；
例如 target 自身的 noise。上游 noise 或 bias 则与后续 edge parameters 构成完整
monomial，因此不会遗漏 response intercept 的 provenance。

于是未压缩的完整响应是：

\[
\Pi_u(t)
=\bigoplus_{j=1}^{B_u}[t\in I_{u,j}]\otimes P_{u,j}.
\tag{U1}
\]

Linear、Tree Ensemble 和 ReLU/PWL 的区别只在 `I[u,j]` 与 `P[u,j]` 如何产生；式 (U1) 对三类完全相同。

### 5.2 再统一 population factorization

令 `Q_q` 是全局 payload dictionary 中的 parameterized provenance root：

\[
Q_q
=\bigoplus_{\pi\in C_q}\bigotimes_{\lambda\in\pi}\Lambda_\lambda.
\]

令 `G[q,u]` 是 unit `u` 上需要 payload `q` 的 region set，并把它唯一分解为 maximal interval runs `I[q,u,k]`。最终统一表示是：

\[
\boxed{
\Pi_u(t)=
h_{u,t}\!\left[
\bigoplus_{q\in\mathcal Q}
\left(
\bigoplus_{k=1}^{r_{q,u}}[t\in I_{q,u,k}]
\right)
\otimes Q_q
\right]
}
\tag{U2}
\]

其中 `h_u` 绑定 unit-specific slots 并保留 intervention indeterminate `tau`，
`ev_t(tau)=t`，而 $h_{u,t}=ev_t\circ h_u$；对已经求值的 guards，`h_{u,t}` 保持其
`0/1` 值。`Q_q` 压缩“推导内容”，`I[q,u,k]` 压缩“该内容对这个 unit 在哪里有效”，
`h_{u,t}` 保留任意 per-unit 参数、root、noise 和 anchor。三类机制只是式 (U2) 的三个实例：

| 机制 | `B_u` / guards | `h_{u,t}(Q_q)` 的数值形态 |
|---|---|---|
| Linear | `B_u=1`，guard 为整个 `D_u` | affine |
| Tree Ensemble | threshold/leaf intervals | constant，亦即 slope 为 0 的 affine |
| ReLU/PWL | ExactLine activation/operator intervals | affine |

这个统一不假设不同 units 有相同权重或相同 rules。若没有任何共享，每个 `G[q,u]` 可以退化为 singleton，表示仍然 exact；若存在重复 causal prefixes、payloads 或 interval supports，三层结构才产生相应压缩。

## 6. 为什么 Mixed SCM 仍然闭合

按 causal topological order 处理 SCM。假设当前所有 parents 都已经是 finite piecewise-affine responses。

1. 先合并 parent boundaries，得到一组 parent cells。
2. 在每个 cell 内，所有 parents 都是 affine lines。
3. Linear 不增加边界并输出 affine。
4. Tree 在 threshold crossings 处切分并输出 constant。
5. ReLU MLP 在 activation crossings 处切分并输出 affine。
6. 把所有输出 cells 拼接起来，child 仍然是 finite piecewise-affine response。

因此 L/T/N 可以出现在同一个 causal DAG 的不同节点中，而不需要改变全局算法。

## 7. 统一的 Guarded Provenance Response

对 unit `u` 和节点 `v`，最终 response 表示为：

```text
GPResponse[u,v]
  = [(I_j, alpha_j, beta_j, P_j)]_{j=1..B[u,v]}
```

查询 `t` 时，只找到拥有 `t` 的一个 interval `I_j`：

```text
Pi[u,v](t) = alpha_j*t + beta_j
```

同时：

```text
h_value[u,t](P_j) = alpha_j*t + beta_j
```

这里：

- `alpha_j,beta_j` 是快速数值缓存；
- `P_j` 是 arithmetic response provenance；
- 独立的 `C_k` 是 tree path 或 activation certificate；
- response intervals 可以比 certificate intervals 更粗。

这就是三类函数真正统一的地方：不是它们内部模型相同，而是它们沿 scalar intervention line 的输出类型相同。

# 第三部分：为什么以及如何存储

## 8. 四种直接方案为什么不够

### 8.1 每次 Replay

什么都不存。每次查询重新：

```text
abduction -> intervention -> causal propagation
```

优点是简单；缺点是每个 unit、每个 `t` 都重复遍历 SCM。

### 8.2 只存 Numeric Segments

存储：

```text
(interval, alpha, beta)
```

POINT 已经可以达到 `O(log B_u)`，但无法回答：

- 哪些 noises 和 parameters 贡献了结果；
- 干预通过哪些 causal paths 传播；
- Tree 为什么进入当前 leaf；
- MLP 哪些 activation pattern 有效。

这是 TESSERA 必须击败或匹配的最强 speed baseline，而不是弱 baseline。

### 8.3 每个 Segment 展开并存完整 Provenance

每个 unit、每个 segment 都保存完整 path expression。这样可以解释结果，但同一条 causal path、同一个 subtree 和同一个 template 会被重复很多次。

### 8.4 物化所有 Global Regions

对 Tree Ensemble 枚举所有 leaf combinations，或者对 MLP 枚举所有 activation patterns。绝大多数 global regions 并不会与当前 unit 的 intervention line 相交，而且组合数可能指数增长。

TESSERA 只物化实际 intervention trajectory 穿过的 intervals。

## 9. GPSI 到底存什么

GPSI 分为 numeric response、canonical provenance fragments、cross-population
factors、physical interval directory 和 audit/secondary indexes。后面的
factorization 不改变第 7 节的逻辑 `GPResponse`；它只改变多个 unit-region roots
如何共享存储。

### 9.1 Unit Store

```text
(model_version, unit)
  -> PARAM_SLOT / BIAS_SLOT bindings
  -> ROOT_SLOT bindings
  -> NOISE_SLOT bindings
  -> ANCHOR_SLOT bindings
```

例如：

```text
unit u:
  ROOT_SLOT(R)  -> 2
  NOISE_SLOT(X) -> 1
  NOISE_SLOT(M) -> -2
  NOISE_SLOT(Y) -> 5
```

Template 中只出现 slot，不出现 unit ID。因此结构相同的 units 可以共享同一个
template；结构不同者保留不同 root，算法不假设共享一定存在。

### 9.2 Response Segment Directory

Key 为：

```text
(model_version, unit, treatment, target)
```

Value 是有序数组：

```text
[(left, right, endpoint_flags, alpha, beta, region_id, cached_root?)]
```

前面的例子分别存成：

```text
Linear:
  [(D.left, D.right, D.flags, 6, 17, r0, root0?)]

Tree:
  [(0, 2, "[]", 0, 13, r0, root0?),
   (2, 4, "(]", 0, 23, r1, root1?)]

MLP:
  [(-2, 0, "[)", 1, 4, r0, root0?),
   ( 0, 1, "[]", 0, 4, r1, root1?),
   ( 1, 2, "(]", 2, 2, r2, root2?)]
```

POINT 只对 boundary array 做 binary search。

### 9.3 Provenance Fragment DAG

逻辑上的 segment provenance 首先被分解为 canonical fragment roots。Fragment
DAG 保存：

```text
ZERO
ATOM(kind, slot_or_parameter_id)
PREFIX(parameter_slot, child_id)
ADD(sorted child_ids with multiplicities)
```

为什么不需要通用 `MUL`：

- 在 affine cell 内，每个 child response 只会乘一个 parameter；
- 多层传播形成 parameter chain；
- 不会把两个独立 parent responses 相乘；
- 因此 `PREFIX(parameter,child)` 足够表示每条 causal path。

Cross-population factor roots、per-unit guard runs 和
`(unit,region) -> aggregate_root` cache 在第 11 节定义。`cached_root` 可以缺省；
缺省时 EXPLAIN 使用该 unit 的 stabbing-semigroup directory 重建当前 region 的
aggregate root。

### 9.4 Certificate DAG 与 Directory

Certificate 使用独立 Boolean nodes：

```text
TRUE
PRED(parent_provenance_root, feature, <=|>, threshold)
GATE(layer, neuron_or_operator, active_choice)
AND(sorted child_ids)
```

Certificate Directory 可能比 Response Directory 更细。

例如两个相邻 MLP activation patterns 最终计算出完全相同的 provenance 和 affine line：

```text
response:
  [I_1 union I_2 -> one template]

certificate:
  [I_1 -> activation pattern A,
   I_2 -> activation pattern B]
```

这样 POINT 不会因为 audit detail 被过度切分，EXPLAIN(audit) 仍然可以返回精确 witness。

### 9.5 Secondary Indexes

```text
Registered FLIP Event Index
Declared Aggregate Directory
```

它们只为 build time 声明的 workload 构建，不属于任意 threshold/group 的通用索引。

## 10. Prefix Provenance 如何压缩

回到 Linear 例子：

```text
X = T + R + N_X
M = 2*X + N_M
Y = 3*M + N_Y
```

先用 parameter slots 完全展开 `P_Y`，得到以下贡献路径：

```text
THETA(Y,M) * THETA(M,X) * TAU(T)
THETA(Y,M) * THETA(M,X) * ROOT_SLOT(R)
THETA(Y,M) * THETA(M,X) * NOISE_SLOT(X)
THETA(Y,M) * NOISE_SLOT(M)
NOISE_SLOT(Y)
```

按照 target-to-source 顺序存储：

```text
THETA(Y,M) -> THETA(M,X) -> TAU(T)
THETA(Y,M) -> THETA(M,X) -> ROOT_SLOT(R)
THETA(Y,M) -> THETA(M,X) -> NOISE_SLOT(X)
THETA(Y,M) -> NOISE_SLOT(M)
NOISE_SLOT(Y)
```

Reverse trie 为：

```text
template_root
|-- PARAM_SLOT(Y,M)
|    |-- PARAM_SLOT(M,X)
|    |    |-- TAU(T)
|    |    |-- ROOT_SLOT(R)
|    |    `-- NOISE_SLOT(X)
|    `-- NOISE_SLOT(M)
`-- NOISE_SLOT(Y)
```

公共 target-side propagation `THETA(Y,M) -> THETA(M,X)` 只存一次。这正对应多条 source-to-target causal paths 共享靠近 target 的 suffix。unit `u` 查询时才通过 `h_u` 把 slots 绑定成 `3` 和 `2`；unit `u'` 可以绑定成任意其他权重而复用同一结构。

Prefix 层内部有三种共享：

1. 同一个 template 内，共享 reverse path prefixes；
2. 多个 population cells 指向同一个 parameterized template root；
3. 不同 template roots 之间，对完全相同的 labeled subtries 做 bottom-up hash-consing。

每个 unique template 都必须保留 designated root。不能把所有 paths 放进一个 global union trie 后忘记 template membership，否则无法知道某个 segment 应该计算哪些 paths。

## 11. Response Segment 如何压缩

局部 compiler 可能产生很多 primitive cells。对于相邻 cells，如果以下内容完全相同：

```text
canonical provenance root
alpha, beta
endpoint-compatible response semantics
```

则把它们合并成一个 maximal response interval。

Tree path 或 activation pattern 是否相同，不参与 response coalescing；这些差异由 certificate directory 保存。

完整索引有三个正交压缩层：

```text
Prefix DAG
  压缩每个 parameterized payload 内部的 causal path prefixes

Segment quotient
  先删除单个 unit 内相邻且完整 response annotation 相同的 primitive cells

Population guarded factorization
  共同压缩不同 units、不同 intervals 中重复出现的 provenance fragments
```

下面把第三层写成精确的 population factorization 问题。

### 11.1 从 Population Cells 到 Support Matrix

固定 `(version, treatment, target)` 和 indexed population `U`。unit `u` 有
`B_u` 个有序 response regions。factorizer 的 row universe 是：

\[
\Omega=\{(u,j):u\in\mathcal U,\ j\in[B_u]\},
\qquad B_{\mathrm{tot}}=|\Omega|=\sum_u B_u.
\]

不同 units 的 region chains 彼此不相邻；只有同一个 unit 内的 `j` 和 `j+1`
才可能合成一个 interval run。把所有 population-cell roots 展开到 canonical、
parameterized fragment occurrences：

```text
F_1, ..., F_m
```

这里的 fragment 可以是一个 canonical causal-prefix subtree、parameter/noise/root slot、
leaf contribution 或已经 intern 的 additive sub-DAG。为每个逻辑 occurrence
分配独立 symbol `F_f`；多个 symbols 可以通过 quotient map 指向同一个底层
fragment root。定义 multiplicity matrix：

\[
\widetilde P_{u,j}=\bigoplus_{f=1}^{m} M_{(u,j),f}\otimes F_f,
\qquad M_{(u,j),f}\in\mathbb N,
\qquad P_{u,j}=h_u(\widetilde P_{u,j}).
\]

数值 weights、biases、factual roots 和 abducted noises 不进入 matrix identity；
它们由 `h_u` 在查询时绑定。因此两条结构相同但权重完全不同的 unit paths
仍可指向同一个 parameterized fragment。结构不同的 fragments 只是得到不同 columns。

理论上可把每个 integer column 做 threshold expansion：

\[
M_{(u,j),f}
=\sum_{c=1}^{\max_{(u',j')\in\Omega} M_{(u',j'),f}}
[M_{(u,j),f}\ge c],
\]

得到 binary incidence matrix `M in {0,1}^{B_tot x m'}`。证明工作在 occurrence-lifted
free semiring `N[F_1,...,F_m]`；quotient homomorphism `pi` 再把属于同一
canonical fragment 的 occurrences 映回同一个底层 DAG root。因而 lifted
exactness 会推出原 provenance semiring 中的 exactness。

这些 occurrence copies 只用于证明；实现保存 weighted multiplicity，不做可能
很大的物理展开。后文仍把 binary columns 简写为 `F_1,...,F_m`，并定义 support：

\[
S_f=\{(u,j)\in\Omega:M_{(u,j),f}=1\}.
\]

相同 support 的 columns 先组成一个 `LOCAL(S)` payload，因此 crossing graph
只为 distinct supports 建 vertex。

### 11.2 Minimum Guarded Provenance Factorization

一个 guarded factor 是：

\[
q=(G_q,C_q),
\qquad G_q\subseteq\Omega,\quad C_q\subseteq[m].
\]

`G_q` 是 factor 有效的 population-cell set。投影到 unit `u`：

\[
G_{q,u}=\{j:(u,j)\in G_q\};
\]

物理上把每个 `G[q,u]` 分别表示为 maximal contiguous runs，绝不跨 unit 合并。
`C_q` 对应一个 shared provenance payload root：

\[
Q_q=\bigoplus_{f\in C_q}F_f.
\]

factor family `Q` 精确，当且仅当：

\[
M=\sum_{q\in\mathcal Q}
\mathbf 1_{G_q}\mathbf 1_{C_q}^{\mathsf T}
\]

在 ordinary integer arithmetic 下逐 entry 成立。等价地，所有 `1` cells 被
rectangles `G_q x C_q` 恰好 partition 一次。

这里不能把 `partition` 偷换成 `cover`。provenance 使用
$\mathbb N[Z]$，$a\oplus a$ 的 multiplicity 是 2，不等于 `a`；重复覆盖会改变
provenance polynomial。

定义：

```text
F(Q) = number of guarded factors / factor roots
K(Q) = sum_q sum_u runs_u(G[q,u]), physical weighted intervals
D(Q) = factor-payload DAG nodes and edges
```

给定第 11.1 节冻结的 canonical fragment dictionary，主理论问题
`MGPF-ROOT` 最小化 `F(Q)`；物理布局随后最小化 `K(Q)` 和 `D(Q)`。这三个量
必须分开报告，不能把 factor-root approximation 误写成 arbitrary DAG-byte
approximation，也不能把它表述为同时优化 fragment dictionary。

### 11.3 Run-Aware Laminar-Cover Factorization (LCF)

两个 supports `A,B` 在以下情况 crossing：

\[
A\cap B\ne\varnothing,\qquad A\nsubseteq B,\qquad B\nsubseteq A.
\]

Support Crossing Graph 的 vertices 是 distinct supports，crossing pairs 之间连
边。任意 proper coloring 都把 supports 分成 `ell` 个 laminar blocks：同一
block 内任意两个 supports 必然 disjoint、equal 或 nested。

对一个 laminar block，先把 distinct supports 建成 inclusion forest。节点 `v`
保存：

```text
S_v        support set
LOCAL(v)   所有 support 恰好等于 S_v 的 fragment roots
child(v)   maximal proper support subsets
A_v        S_v minus union_{c in child(v)} S_c
```

`BUNDLE(x,y)` 在代数上只是 $x\oplus y$，物理上是 non-flattened binary ADD
node，使 inherited payload 可以被多个 children 共享。最初的 support-cut 规则
保持 factor-root optimal，却不总是给出较少的 guard runs。最终算法增加一个
run-aware reset optimizer：

```text
RUN_AWARE_LCF_BLOCK(block):
  group equal supports and build the inclusion forest
  for each forest root v:
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
  emit factor (guard=G_v, payload=bundle)

  for c in child(v):
    if c in J:
      FACTOR(c, ZERO)
    else:
      FACTOR(c, bundle)

MIN_RUN_RESET(A_v, children):
  return argmin_{J subseteq children}
         runs(A_v union union(S_c for c in J))
```

`J` 是由当前 factor 完整吸收并 reset inherited payload 的 child subtrees。若
`J=child(v)`，guard 是旧算法的 `S_v`；若 `J=empty`，guard 是 residual atom
`A_v`。中间选择可以填平某些 interval gaps，同时避免吸收会制造新 run 的
child support。

这个 subset optimization 可精确归约为 directed min-cut。为每个 child `c`
建一个在所有 units 间共享的 binary variable $x_c$。对 unit `u` 的 ordered
region `j`，令：

\[
z_{u,j}=
\begin{cases}
1,&(u,j)\in A_v,\\
x_c,&(u,j)\in S_c,\\
0,&(u,j)\notin S_v,
\end{cases}
\qquad z_{u,0}=0\quad\forall u.
\]

则：

\[
\operatorname{runs}_{\mathcal U}(G_v)
=\sum_{u\in\mathcal U}\sum_{j=1}^{B_u}
z_{u,j}(1-z_{u,j-1}).
\]

把 fixed `1`/`0` 分别映射到 source/sink，并为每个 adjacent pair 加 directed
unit edge $z_{u,j}\to z_{u,j-1}$。每个 unit 都有独立的 fixed-zero chain start，
所以最后一个 region of `u` 不会与第一个 region of `u+1` 伪合并。当 1 表示
source side 时，cut capacity恰好等于上式；min-cut 的 source-side child variables
给出同时最小化所有 unit-chain runs 总和的最优 `J`。重复 label-pair edges 可
聚合为一条 weighted edge。Min-cut graph 只在 build time 存在。

下面保留一个 single-unit 例子；population 版本只是把相同目标在每个 unit chain
上的 run starts 相加：

```text
B = 5
S_v  = {1,2,3,5}
A_v  = {1,3}
S_c1 = {2}
S_c2 = {5}
```

旧 support cut `S_v` 有 2 runs，pure atom `A_v` 也有 2 runs。Run-aware LCF
选择 `J={c1}`，emit `G_v={1,2,3}`，只需 1 run；`c1` 从 ZERO 继续，`c2`
继承 bundle。factor 数不变。

LCF 的完整流程是：

1. 建 support crossing graph；
2. 使用 deterministic greedy coloring 或用户指定 partition 得到 `ell` blocks；
3. 每个 block 运行 `RUN_AWARE_LCF_BLOCK`；
4. 对输出应用对 `F`、`K`、`D` 均不增加的 exact rewrites；
5. 把每个 guard 转成 maximal runs 并建立 stabbing-semigroup directory。

Run-aware min-cut 在 hierarchical reset factorization family 内使 `K` 最小。
它不声称覆盖所有 rank-optimal rectangle partitions，因此不能写成 unrestricted
global lexicographic `(F,K)` optimality。

### 11.4 Exact ILP Oracle

小实例枚举 candidate bicliques `q=(G_q,C_q)`，求解 set-partitioning ILP：

\[
\min\sum_q x_q,
\qquad
\sum_{q:((u,j),f)\in G_q\times C_q}x_q=1
\quad\forall ((u,j),f):M_{(u,j),f}=1,
\qquad x_q\in\{0,1\}.
\]

它只作为小实例 `OPT` oracle，不进入生产查询路径。实验必须报告
`F_LCF/OPT`，而不能只报告相对弱 baseline 的压缩率。

### 11.5 从 Factors 到 Interval Index

对 factor guard `G_q`，分别计算每个 unit projection `G[q,u]` 在该 unit region
order 上唯一的 maximal contiguous decomposition。每个 run 产生一个 weighted
interval：

```text
(unit, left_region, right_region, payload_root=Q_q)
```

查询 population cell `(u,j)` 返回：

\[
P_{u,j}=h_u\!\left(\bigoplus_{q:(u,j)\in G_q}Q_q\right).
\]

这正是 stabbing-semigroup query；$(\mathbb N[Z],\oplus)$ 是 commutative monoid，
因此不需要 subtraction、inverse 或 idempotence。Latency mode 可以再物化
`(unit,region_id) -> aggregate_root` cache，用 `O(B_tot)` pointers 和显式计量的
`D_cache` aggregate DAG 换掉 cold EXPLAIN 中的 stabbing lookup；compact mode
不构建该 cache。

## 12. GPSI 物理布局

```text
GPSI
|
|-- Model Catalog
|     causal DAG、L/T/N mechanisms、parameters、model version
|
|-- Unit Store
|     parameter、factual-root、abducted-noise 和 anchor slot bindings h_u
|
|-- Provenance Fragment DAG
|     multi-root ZERO/ATOM/PREFIX/ADD DAG
|
|-- LCF Factor Bundle DAG
|     LOCAL roots and non-flattened BUNDLE(base,delta) nodes
|
|-- Factor Interval Directory
|     per-unit linear-space stabbing-semigroup indexes over weighted guard runs
|
|-- Response Segment Directory
|     (version, unit, treatment, target)
|       -> sorted numeric segments and optional region-root cache
|
|-- Certificate DAG and Directory
|     更细的 tree-path / activation witnesses
|
|-- Registered FLIP Event Index
|     (version, target, threshold)
|       -> sorted state-change events
|
`-- Declared Aggregate Directory
      (version, group, target, SUM|AVG)
        -> aggregate piecewise-affine response
```

# 第四部分：查询与完整算法

## 13. 查询算法

### 13.1 POINT

```text
POINT(u,Y,t):
  segments = ResponseDirectory[(version,u,T,Y)]
  j = predecessor_search(segments.boundaries, t)
  return segments[j].alpha*t + segments[j].beta,
         segments[j].region_id,
         segments[j].cached_root?
```

复杂度：

```text
O(log B_u)
```

### 13.2 EXPLAIN

```text
EXPLAIN(response, cold):
  (u,j) = POINT(...).unit_and_region_id
  root = StabbingSemigroup[u].query(j)
  traverse root

EXPLAIN(response, cached):
  root = POINT(...).cached_root
  traverse root

EXPLAIN(audit):
  run response EXPLAIN
  certificate predecessor lookup
  return response lineage + guard/control witness
```

复杂度：

```text
cold response EXPLAIN    O(log B_u + log K_u + A_u,j)
cached response EXPLAIN  O(log B_u + A_u,j)
audit EXPLAIN            add O(log B_c + certificate_output)
```

`A_u,j` 是 population cell `(u,j)` 上实际遍历或 materialize 的 compressed
provenance 大小。若调用者只需要 opaque root ID，不展开 lineage，则去掉 `A_u,j`。

### 13.3 Registered FLIP

对预注册 threshold `q` 定义：

```text
state_u(t) = 1[Pi[u,Y](t) >= q]
```

构建时生成三类 event：

1. affine segment 内部的 continuous crossing；
2. Tree 或 mixed SCM boundary 上的 jump；
3. 进入或离开 threshold plateau。

按 `t` 排序后：

```text
FLIP(Y,q,[a,b]) = lower_bound(a) + sequential reporting
```

复杂度：

```text
O(log E_q + k)
```

这个保证只适用于 build time 注册的 threshold。

### 13.4 Declared AGG

对预声明 group，合并所有 member response boundaries，在每个公共 interval 上累加 slopes 和 intercepts：

```text
alpha_group = sum_u alpha_u
beta_group  = sum_u beta_u
```

AVG 再除以 group size。物化后的查询复杂度：

```text
O(log B_group)
```

## 14. 全局构建算法

### 14.1 Local Compiler Contract

```text
COMPILE_CELL(
    mechanism,
    parent_affine_lines,
    parent_provenance_roots,
    interval,
    boundary_policy
) -> response_pieces, certificate_pieces
```

### 14.2 Tree Tracing

```text
TRACE_TREE(node, I, parent_lines):
  if node is leaf:
    emit leaf response and accumulated path certificate
    return

  transform x_k <= theta into a_k*t+b_k <= theta

  if no crossing lies in I:
    recurse into the only reachable child
  else:
    split I at t_star
    recurse into both reachable child intervals
```

每棵 tree 输出有序 stream；ensemble 使用 multiway merge。

### 14.3 Neural Line Compilation

```text
COMPILE_NEURAL_CELL(network, x(t), I):
  pieces = ExactLine(network, x(t), I)

  for each piece with fixed activation pattern:
    compute alpha, beta
    propagate original weight/bias provenance
    emit response piece
    emit activation certificate piece
```

### 14.4 Causal Topological Compilation

```text
COMPILE_UNIT(M, u, T, Targets, D):
  response[T] = [(D, 1, 0, TAU(T))]

  for v in relevant topological order:
    if v is fixed root:
      response[v] = [(D, 0, x[u,v], ROOT_SLOT(v))]
      continue

    parent_cells =
      common_refinement(response[p] for p in pa(v))

    for I in parent_cells:
      parent_lines, parent_roots = restrict parents to I
      local_response, local_certificates =
        COMPILE_CELL(f_v, parent_lines, parent_roots, I)
      append results

    response[v] =
      intern templates and coalesce response-equivalent neighbors

    certificates[v] =
      intern certificates without response-level coalescing

  materialize declared targets
```

系统只处理 `ancestors(Targets)` 中受 intervention 影响的 causal slice，以及计算这些节点所需的 fixed parents。

### 14.5 Cross-Population Factorization 与物理构建

每个 declared target 的所有 units 完成 logical `GPResponse` 后联合执行：

```text
BUILD_GPSI(population_responses):
  cells = [(u,j,response[u].root[j]) for every unit u and region j]
  fragments, supports = EXTRACT_CANONICAL_FRAGMENTS(cells)
  support_groups = GROUP_EQUAL_SUPPORTS(fragments, supports)

  crossing_graph = BUILD_CROSSING_GRAPH(support_groups)
  blocks = DETERMINISTIC_COLOR(crossing_graph)

  factors = union(RUN_AWARE_LCF_BLOCK(block) for block in blocks)
  factors = EXACT_NONINCREASING_REWRITES(factors)

  runs = MAXIMAL_GUARD_RUNS_PER_UNIT(factors)
  interval_indexes = BUILD_PER_UNIT_STABBING_SEMIGROUP(runs)

  if root_cache_enabled:
    root_cache = MATERIALIZE_POPULATION_CELL_ROOTS(interval_indexes, B_u for all u)

  return population_responses.numeric_segments, factors, interval_indexes, root_cache
```

`DETERMINISTIC_COLOR` 不声称求 minimum coloring。理论 approximation ratio 使用
算法实际产生的 block 数 `ell_hat`；实验同时报告 lower bound、greedy
`ell_hat` 和 natural mechanism partition，例如 tree ensemble 的 per-tree blocks。

# 第五部分：问题范围、Baselines 与理论

## 15. 正式问题范围

### 15.1 构建输入

```text
M_mu       model version mu 下的 acyclic additive-noise SCM
U          observed numerical units
T          one scalar intervention variable
Targets    numerical targets
D          bounded domain [t_min,t_max]
Q_flip     registered (target,threshold) pairs
Q_agg      declared (group,target,SUM|AVG) workloads
```

不同 causal nodes 可以分别使用 L/T/N。Mixed SCM 不是第四类机制。

### 15.2 Exactness 边界

形式理论使用 real arithmetic 和 deterministic branch semantics。实现相对于 stored fitted predictor、`<=`/`>` branch policy 和 ReLU-at-zero convention 保持 model-exact，不通过 sampling 近似 intervention domain。

浮点实现必须在 boundary 及其两侧直接验证模型输出。论文不能把 IEEE floating-point arithmetic 描述成 symbolic real arithmetic。

### 15.3 非支持范围

- vector-valued intervention 和一般多维 point location；
- sigmoid、tanh、GELU、softplus 和 smooth networks；
- RBF、Gaussian process、polynomial mechanism 和 black box；
- KNN、rule list、spline/GAM、weighted-median AdaBoost；
- categorical classifier semantics 和 probability link；
- cyclic SCM、non-additive 或 non-invertible abduction。

## 16. Baselines

| ID | Baseline | 它检验什么 |
|---|---|---|
| B0 | DoWhy `counterfactual_samples` from observed data | 端到端 replay，包括重复 abduction |
| B1 | cached-noise vectorized SCM replay | 最强公平 replay baseline |
| B2 | exact numeric per-unit segment index | POINT speed；不含 provenance |
| B3 | Independent Interval Provenance (IIP)：每个 `(unit,region)` 独立保存完整 circuit | VSC persistence 与 interning 的 storage/build baseline |
| B4 | global region / leaf Cartesian materialization | trajectory-sensitive construction |
| B5 | tree-ensemble ADD/decision-diagram compilation | Class-T symbolic compilation |
| B6 | unmodified numeric ExactLine | Class-N partition |

必须实现以下 ablations：

```text
GPSI without response coalescing
VSC with one arena per unit and no cross-unit hash-consing
IIP with one independent circuit per region and no persistence
event log with no materialized region roots
VSC-Core without certificates
VSC-Audit with certificates
VSC plus optional additive-frontier factorization
```

固定比较关系：

- replay latency 与 B1 比，numeric POINT 与 B2 比；
- provenance storage/build 与 B3 比；VSC 与 B3 的 indexed-root lookup 渐近相同；
- region avoidance 与 B4 比；
- Tree compiler 与 B5 比；
- Neural partition 与 B6 比。

TESSERA 不声称 POINT 渐近快于 B2。它匹配 B2 的 lookup complexity，同时提供 exact provenance 和 secondary indexes。

## 17. Provenance 形式语义

令 `Z` 包含：

```text
ROOT_SLOT(v)
NOISE_SLOT(v)
PARAM_SLOT(s)
BIAS_SLOT(s)
LEAF(m,l)
```

`tau` 是独立 intervention indeterminate，物理上表示为 `TAU(T)`。

General VSC semantics 使用：

```text
K = N[Z]
```

作为 free commutative parameterized provenance semiring。Signed parameter 和 noise 是 slots，其 unit-specific numeric valuation 可以为负。每个 unit 有一个 binding homomorphism `h_u`；它把 slots 映射到该 unit 的参数 tokens 或 numeric values，并保持 `ZERO/ONE/ATOM/ADD/MUL`。固定一个 response interval 后：

```text
P(tau) in K[tau]
degree_tau(P) <= 1
```

系统提供：

```text
h_value[u,t]
h_response_lineage
h_audit_lineage
```

`h_value` 重建 numeric response；`h_response_lineage` 返回 arithmetic contributors；`h_audit_lineage` 再加入 certificate control dependencies。

一般 logical response 直接由 VSC root 给出：

\[
P_{u,j}=h_u(R_{u,j}),
\qquad
\Pi_u(t)=h_{u,t}(R_{u,j(t)}).
\]

该定义保留完整 `ADD/MUL` circuit，不要求 path expansion，也不要求 support
staircase/laminar。

只有当优化器选择一个明确的 additive frontier 时，才令 row universe
`Omega={(u,j): u in U, j in [B_u]}`。把 occurrence-lifted parameterized fragment
symbols 记为 `F_1,...,F_m`，binary support matrix 记为 `M`。逻辑 response
与 factorized response 分别是：

\[
\widetilde P_{u,j}=\bigoplus_f M_{(u,j),f}\otimes F_f,
\qquad P_{u,j}=h_u(\widetilde P_{u,j}),
\]

\[
\widehat P_{u,j}
=h_u\!\left(\bigoplus_{q:(u,j)\in G_q}
\left(\bigoplus_{f\in C_q}F_f\right)
\right).
\]

MGPF exactness 先在 occurrence-lifted free semiring 中要求 formal-polynomial
equality `P_tilde[u,j] = P_hat_tilde[u,j]`，不是只要求某个 numeric valuation 上恰好相等。逐
occurrence multiplicity 相同等价于 lifted polynomial 相同。quotient
homomorphism `pi` 将 occurrence symbols 映射到原 `K[tau]` fragment
polynomials；再复合任意 unit binding `h_u` 和 value/lineage homomorphism，
等式仍然保持。这一步是允许 arbitrary per-unit weights 的关键。

VSC 主定理使用：

```text
B_tot   population regions，sum_u B_u
D_0     atoms、templates 与 first-version canonical nodes
alpha   changed slots 与 affected ancestors 的 transition 总工作
gamma   transition 中真正新 intern 的 nodes，gamma <= alpha
H       sum_u |h_u|
```

后续 MGPF/LCF secondary theorems 另使用：

```text
B_u     unit u 的 ordered response regions
B_tot   population cells，sum_u B_u
m       logical occurrence-support layers after weighted multiplicity compression
d       distinct nonempty supports after equal-support grouping
F*      minimum number of exact guarded factor roots
F       LCF output factor roots
R       sum of maximal-run counts of the d distinct supports
K       sum_q sum_u runs_u(G[q,u])
ell     number of laminar blocks produced by support-graph coloring
D_f     canonical additive-frontier fragment/radix DAG size
D_b     LOCAL/BUNDLE factor DAG size
D_cache optional materialized region-root cache DAG size
A_u,j   materialized provenance output size at population cell (u,j)
```

## 18. 定理体系

### 定理 1：L/T/N 精确组合闭包

对于所有相关 nodes 使用 L/T/N 的 acyclic additive-noise SCM，固定 unit 和有界 scalar intervention 后，每个 relevant node 都得到 finite、possibly discontinuous、piecewise-affine guarded provenance response。Global compiler 精确返回该 response。

证明方向：沿 causal topological order 归纳，并证明每个 local compiler 的 boundary soundness 和 completeness。

### 定理 2：Tree Ensemble 非笛卡尔积界

在一个 parent cell 上，如果 tree `m` 产生 `b_m` 个 intervals，则 common refinement 最多为：

```text
1 + sum_m(b_m-1)
```

而 leaf-combination construction 可能产生 `product_m b_m` 个组合。完整 causal composition 仍可能指数增长。

### 定理 3：VSC Exactness、Construction 与 Storage

给定 exact region sequence、固定 circuit template 和每个 boundary 的 slot delta，
VSC 对任意 `ON/OFF/ON` 更新序列返回与每个 region 从头 materialize 完全相同的
free-semiring circuit。令 `alpha[u,j]` 是 changed slots 与 affected ancestor closure
大小，`gamma[u,j]` 是其中真正新 intern 的 arena nodes，则：

\[
T_{build}=O\!\left(B_{tot}+D_0+\sum_{u,j}\alpha[u,j]+H\right),
\]

\[
S_{VSC}=O\!\left(B_{tot}+D_0+\sum_{u,j}\gamma[u,j]+H\right),
\qquad \gamma[u,j]\le\alpha[u,j].
\]

对 balanced associative block，若 `C` 个 leaves 中有 `k` 个改变，则 inclusive
ancestor-path union 为 `O(k log(1+C/k))`。证明见 `THEORY_PROOFS.md` 第 0 节。
一般 circuit 可有 `alpha=Theta(D_0)`，所以本定理不声称 universal compression 或
arithmetic-circuit global optimality。

### 定理 3A：Additive-Frontier Provenance DAG 与 Unit Binding 正确性

在任意 semiring valuation 下，ZERO/ATOM/PREFIX/ADD/BUNDLE DAG 的计算结果等于
逻辑展开的 parameterized provenance polynomial。若 guarded factors partition
population matrix `M` 的 `1` cells，则 `P_tilde[u,j] = P_hat_tilde[u,j]` 对每个
population cell 成立。对任意 per-unit 参数、root、noise bindings 定义的 semiring
homomorphism `h_u`，都有

\[
h_u(\widetilde P_{u,j})=h_u(\widehat{\widetilde P}_{u,j}).
\]

因此即使所有 units 的 weights 都不同，value、response lineage 和 factor lineage
仍然精确。

证明分三步：先对 DAG nodes 做结构归纳；再使用每个 matrix entry 恰好被一个
factor rectangle 覆盖，将双重 $\oplus$ 重排回 `P_tilde[u,j]`；最后应用 quotient
map `pi`、unit binding `h_u` 和查询 homomorphism。certificate correctness
独立地对 Boolean DAG 做结构归纳。

### 安全引理：Prefix-DAG Size

令 `Paths(r)` 为 designated root `r` 的 weighted target-oriented path set。
独立 reverse trie 节点数严格等于：

```text
1 + |Pref(Paths(r))|
```

Bottom-up hash-consing 不增加节点数。本文不再声称它在 arbitrary arithmetic
circuits 或所有 weighted multi-root automata 中全局最小；这避免把未核实的
representation-class claim 当成主定理。

### 定理 4：MGPF-ROOT NP-Complete

`MGPF-ROOT` 的 decision version 问：给定 binary support matrix `M` 与整数
`k`，是否存在至多 `k` 个 guarded factors 精确 partition 所有 `1` cells。

给定 bipartite graph `G=(U,V,E)`，取 `M` 为其 adjacency matrix。每个 factor
`G_q x C_q` 正好是一个 biclique；由于 `N[Z]` 非幂等，exactness 要求每条边
出现一次，正好得到 Minimum Biclique Edge Partition。归约可以令每个 graph row
成为一个只有一个 region 的 unit，因此任意 row subset 都可表示为 singleton
unit-runs 的 union；multi-unit interval order 不消除该
困难性。解可在 polynomial time 验证，所以问题是 NP-complete。

该结论只针对 factor-root objective；相关 biclique-partition hardness 与
inapproximability 结果见：
<https://doi.org/10.1587/transinf.E93.D.290>。

### 定理 5：通用 Rank Lower Bound

任何 exact factorization 都满足：

\[
F^*\ge \operatorname{rank}_{\mathbb R}(M).
\]

因为每个 rectangle matrix
`1_{G_q} 1_{C_q}^T` 在 real field 上 rank 至多为 1，而 `F` 个 rank-one
matrices 的和 rank 至多为 `F`。该 lower bound 可在线性代数时间内计算，供
大实例报告 certified gap；它可能不 tight，因此不能替代小实例 ILP oracle。

### 定理 6：Laminar Exact Optimality 与 Run-Aware Interval Bound

若 distinct column supports 构成 laminar family，则
`RUN_AWARE_LCF_BLOCK`：

1. 精确重建 `M`；
2. 使用全局最少的 factor roots；
3. 满足：

\[
F=F^*=\operatorname{rank}_{\mathbb R}(M)
=|\{v:A_v\ne\varnothing\}|;
\]

4. 若

\[
\kappa_v=\min_{J\subseteq child(v)}
\operatorname{runs}_{\mathcal U}\left(A_v\cup\bigcup_{c\in J}S_c\right),
\]

   则

\[
K=\sum_{v:A_v\ne\varnothing}\kappa_v
\le \min(R_S,R_A)\le R,
\]

   其中 $R_S=\sum_{v:A_v\ne\varnothing}runs_{\mathcal U}(S_v)$，
   $R_A=\sum_{v:A_v\ne\varnothing}runs_{\mathcal U}(A_v)$；
5. 在 hierarchical reset factorization family 内，上式的 `K` 最小；
6. 除已有 fragment DAG 外，只需 `O(d)` 个 BUNDLE nodes 和 `O(m)` 个
   LOCAL membership references。

证明要点：

- **Exactness invariant。** `FACTOR(v,inherited)` 的 inherited payload 必须在
  `S_v` 上恰好贡献一次。若 `A_v` nonempty，factor guard 是
  $A_v\uplus\bigcup_{c\in J}S_c$；selected children 从 ZERO 开始，其余 children
  继承 bundle。若 `A_v` empty，children disjointly partition `S_v`，全部继承
  bundle。三种 disjoint 部分合起来恰好覆盖 `S_v`。
- **Rank tightness。** Nonempty residual atoms `A_v` 两两 disjoint，所以其
  indicator vectors 线性独立。每个 support 是 subtree atoms 的 disjoint sum；
  反向由
  `1_{A_v}=1_{S_v}-sum_c 1_{S_c}` 做 tree Möbius inversion。因此 matrix rank
  等于 nonempty atoms 数。LCF 正好每个 nonempty atom 对应一个 emitted
  factor，达到定理 5 lower bound。
- **Run-aware optimality。** 对每个 child variable $x_c$，令 unit `u` 的 ordered
  position label $z_{u,j}$ 等于 residual 上的 1、outside-support 上的 0 或对应的
  $x_c$，且每条 unit chain 独立设置 $z_{u,0}=0$。`runs_U` 是
  $\sum_u\sum_j z_{u,j}(1-z_{u,j-1})$，等于 directed edges
  $z_{u,j}\to z_{u,j-1}$ 的 cut capacity。故一次 min-cut 精确求得 $\kappa_v$。各 node
  的 guard cost 与 inherited payload 内容无关，因此 node-wise optima 可相加。
  `J=all` 与 `J=empty` 分别给出 $S_v$ 和 $A_v$ 两个可行解，故
  $K\le\min(R_S,R_A)\le R$。
- **DAG safety。** 每个 support 只构建一个 `LOCAL(v)` 和一个
  `BUNDLE(inherited,LOCAL(v))`；BUNDLE 不 flatten inherited subtree，min-cut
  graph 不进入 persistent index。

### 定理 7：Laminar-Cover Approximation

设 support crossing graph 的一个 proper coloring 产生 `ell` 个 laminar
blocks。对每个 block 运行 `RUN_AWARE_LCF_BLOCK`，合并其输出，则：

\[
F_{LCF}\le \ell F^*,
\qquad K_{LCF}\le\min(R_S,R_A)\le R,
\qquad D_b=O(d+m).
\]

证明：第 `b` 个 block 的 LCF 输出等于该 submatrix 的 optimum `F_b^*`。把
任意 global optimum factors 的 payload 限制到 block `b` 并删除 empty
factors，得到至多 `F^*` 个 factors 的合法 submatrix factorization，因此
`F_b^* <= F^*`。对 `ell` 个 blocks 求和即得 approximation bound。`K` 使用
定理 6 以及 $\sum_b\min(x_b,y_b)\le\min(\sum_bx_b,\sum_by_b)$；`D_b` 由
定理 6 逐 block 相加得到。

直接推论必须区分 single-unit 与 arbitrary population：

- 对一个 linear unit，`B_u=1`，所有 nonempty supports equal，因而 `ell=1`；
- 对一个 fixed unit 的单棵 Tree，node/path guards nested 或 disjoint，因而
  per-unit support family 为 laminar；
- 当所有 units 共享同一 structural template 时，Linear/Tree 的上述 laminar 性
  可以延伸到 population；
- 对 arbitrary per-unit templates，Linear、Tree 和 ReLU 的 population supports
  都可能 crossing。算法始终 exact，定理 7 只保证实际 coloring 的
  `F<=ell*F*`，最坏 `ell=d`。论文必须报告 realized `ell`，不能从机制类别
  推导未经假设的 constant ratio。

Greedy coloring 不声称逼近 minimum laminar-cover number。定理 ratio 使用算法
实际产生的 `ell`，因此不依赖 optimal coloring。

### 定理 8：Canonical Response-Segment Quotient

相邻 primitive cells 在以下内容相同时 response-equivalent：

```text
canonical provenance root
alpha, beta
endpoint-compatible response semantics
```

Maximal equivalent runs 构成唯一 reduced response partition，并且在每个 interval 只存一个 response annotation 的精确一维索引类中 interval 数最少。

Certificate equality 不参与该等价关系。

### 定理 9：Canonical Guard Runs 与 Dyadic Separation

固定 factor family 后，每个 guard `G_q` 在每条 unit chain 上的 maximal
contiguous runs 是唯一的 minimum exact interval decomposition。因此 physical
interval count 恰为：

\[
K=\sum_q\sum_u \operatorname{runs}_u(G_{q,u}).
\]

Dyadic segment-tree layout 会把 unit `u` 的每个 run 分成 `O(log B_u)` 个
canonical nodes，总 references 为 `O(K log B_max)`。当 `B_u=2^h` 时，region
interval `[1,B_u-1]` 需要 `Theta(log B_u)` 个 dyadic nodes，而 stabbing directory
只存一个 interval；因此存在严格的 `Theta(log B_max)` worst-case
directory-space separation。

该 separation 针对 interval-directory references，不把 provenance DAG bytes
混入同一个 bound。

### 定理 10：Semigroup Index 与 Query Bounds

将每个 guard run 赋权为 factor payload root 后，cold response EXPLAIN 是
commutative-semigroup stabbing query。linear-space stabbing-semigroup structure
提供：

```text
directory space          O(K)
cold factor-root query   O(log K_u)
```

该 data-structure result 来自：<https://doi.org/10.1137/10078791X>；TESSERA
不把它声明为新颖性。它适用是因为 provenance addition 构成 commutative
monoid，不要求 inverse。

加上 sorted response/certificate directories：

```text
POINT                    O(log B_u)
cold response EXPLAIN    O(log B_u + log K_u + A_u,j)
cached response EXPLAIN  O(log B_u + A_u,j)
audit EXPLAIN            add O(log B_c + certificate_output)
registered FLIP          O(log E_q + k)
declared AGG             O(log B_group)
```

在 comparison model 中，POINT 已匹配 predecessor lower bound；若输出完整
lineage，任何算法还必须支付 `Omega(A_u,j)`。因此 cached EXPLAIN 在该模型下
output-optimal。word-RAM predecessor bounds 不在此 claim 内。

### 定理 11：Optional Additive-Frontier Construction 与 Storage

本定理只计算在 VSC 之外启用 MGPF/LCF 与 stabbing directory 的附加成本；general
construction/storage 由定理 3 给出。

定义：

```text
T_abduct   total abduction cost
K_local    realized local compiler cost
R_merge    merged boundary records
Delta      maximum merge fan-in
T_VSC      定理 3 的 primary VSC version-construction cost
S_r,S_c    response/certificate segments
N_1        nonzero support incidences before equal-support grouping
X          support crossing-graph edges
T_rel      support pair classification and laminar-forest construction time
T_color    deterministic coloring time
T_reset    total time of run-aware directed min-cut instances
D_f,D_b,D_c fragment, bundle, and certificate DAG sizes
```

在 expected constant-time hash interning 下，完整 build time 为：

```text
T_VSC + O(T_abduct + K_local + R_merge*log(Delta+1)
  + N_1 + T_rel + T_color + T_reset
  + K*log(K+1) + D_f + D_b + D_c + S_r + S_c)
```

bitset 实现同时分类 equal/disjoint/nested/crossing support pairs，并从 containment
relations 建 laminar inclusion forests。朴素实现为
`T_rel=O(d^2*ceil(B_tot/w))` word operations；稀疏 support 实现可按实际
incidences/intersections 降低。编译部分不包含 potential global-region Cartesian term，
但 `B_tot,N_1,d` 在 adversarial model 下仍可能指数增长。

第 `v` 个 reset graph 有 `|child(v)|+2` 个 variable/terminal vertices 和至多
`B_tot` 条未聚合 transition edges；每个 unit chain 单独起始，重复 ordered
label pairs 聚合后可能更小。
`T_reset` 显式包含这些 graphs 的构建和 max-flow/min-cut 时间，本文不把它隐藏
在 `O(d+m)` persistent-DAG bound 中。

不计 fitted model 和 observed input，storage 为：

```text
VSC + additive-frontier directory (cold):
  S_VSC + O(S_r + D_f + D_b + K)

VSC + additive-frontier directory (cached):
  S_VSC + O(S_r + D_f + D_b + D_cache + K)

VSC-Audit:
  add O(S_c + D_c + E + A)
```

`E` 是 registered FLIP events，`A` 是 declared aggregate segments。默认
region-root cache 再使用 `O(S_r)` pointers；aggregate ADD nodes/edges 明确计为
`D_cache`，不假装它们免费。Numeric segment baseline 已需要 `Theta(S_r)`，
所以不声称 universal strict total-byte compression。

### 18.1 明确的 Non-Claims

1. Canonical VSC 不保证是最小 algebraically equivalent circuit；它只对固定
   template topology 做 exact persistence 与 syntactic interning。
2. VSC 不保证 universal compression；一般 transition 可有
   `alpha,gamma=Theta(D_0)`。
3. VSC 与 IIP 都保存 indexed region roots，因此不声称比 IIP 有更好的 POINT/root
   lookup 渐近复杂度。
4. `ell`-approximation 只针对 guarded factor-root count `F`，不针对 arbitrary
   arithmetic-circuit minimum 或最终 serialized bytes。
5. MGPF 假定 canonical fragment dictionary 已冻结；它不联合搜索任意新的
   algebraic subexpressions。
6. NP-completeness 针对 factorizer 接受的一般 support matrices；单棵 Tree 的
   laminar subclass 已由定理 6 polynomial-time 精确解决。本文不额外声称某个
   更窄的固定 ReLU architecture 仍然 NP-hard。
7. `K<=min(R_S,R_A)<=R` 针对 LCF emitted guards 与两个 reset-family
   endpoints；它不是相对任意 joint `(F,K,D)` optimum 的 approximation ratio。
8. Run-aware min-cut 只在 hierarchical reset factorization family 内使 `K`
   最小；general rank-optimal rectangle partitions 可能位于该 family 之外。
9. Prefix hash-consing 不声称 arbitrary arithmetic-circuit minimality。
10. Stabbing-semigroup complexity 是采用的已知 data-structure guarantee，不是
   TESSERA 自己的 novelty。
11. 任意 ReLU network 的 `ell`、response regions 和 fragment incidences 都可能
   很大；exactness 不意味着 polynomial worst-case compilation。
12. FLIP bound 只适用于 registered thresholds，AGG bound 只适用于 declared
   groups。

## 19. 算法优势总结（VSC 主线）

下面只比较 response/provenance index；所有方法共同需要的 fitted SCM、observed
data 和 abduction bindings 不重复计入。记：

```text
B_u          unit u 的 response regions 数量
B_tot        sum_u B_u
L[u,j]       region (u,j) 独立 materialize 时的 circuit size
D_ind        sum_(u,j) L[u,j]
D_0          atoms、templates 与 first versions 的 canonical nodes
alpha[u,j]   transition (u,j) 的 changed slots 加被重新计算的 ancestors
gamma[u,j]   其中真正新 intern 的 arena nodes，gamma <= alpha
A,G          分别为 alpha 与 gamma 在全部 transitions 上的总和
H            sum_u |h_u|
Q_u,j        EXPLAIN 实际输出的 provenance 大小
```

完全不压缩的 Independent Interval Provenance (IIP) 保存：

\[
S_{IIP}=\Theta(B_{tot}+D_{ind}+H).
\]

VSC 保存：

\[
S_{VSC}=O(B_{tot}+D_0+G+H),
\qquad
T_{build}=O(B_{tot}+D_0+A+H).
\]

这两个式子不能只凭符号声称 VSC 总是更小。理论优势是 change-sensitive：当相邻
regions 只改变小 affected cone，`A << D_ind`；当更新结果复用旧 subcircuits 或旧
patterns，`G << A`。最坏情况下 `A,G=Theta(D_ind)`，不保证压缩。

| Index | Build time | Space | POINT/root | EXPLAIN | 核心判断 |
|---|---:|---:|---:|---:|---|
| cached-noise replay | `O(T_abduct)` | `O(n|V|)` | `Theta(C_replay)` | `Theta(C_replay+Q_u,j)` | 每次仍传播 SCM |
| numeric segment array | `O(B_tot)` | `Theta(B_tot)` | `O(log B_u)` | 不支持 provenance | 最强纯数值 lookup baseline |
| IIP | `Theta(D_ind)` | `Theta(B_tot+D_ind+H)` | `O(log B_u)` | `O(log B_u+Q_u,j)` | 每个 region 单独保存完整 circuit |
| event log + first circuit | `O(D_0+sum k[u,j])` | `O(D_0+sum k[u,j])` | 需 replay prefix events | replay 后再输出 | 空间轻，但把 version reconstruction 推迟到查询 |
| **TESSERA VSC-Core** | **`O(B_tot+D_0+A+H)`** | **`O(B_tot+D_0+G+H)`** | **`O(log B_u)`** | **`O(log B_u+Q_u,j)`** | **任意 slot toggles；mixed `ADD/MUL`；跨相邻、非相邻和跨 unit 共享** |
| VSC + optional additive factorization | Core 加 factorization cost | Core 加 `O(F+K+D_f)` | 同 Core | factor reporting 另计 | 只优化声明的 additive frontier |

必须公平解释两种速度比较：

1. VSC 和 IIP 都保存 region root，POINT/root lookup 的渐近复杂度相同；VSC 对 IIP
   的主优势是 storage 与 construction。
2. VSC 对 event log 的 query 优势来自预先保存每个 persistent root；不能把该数字
   写成“VSC 比 IIP 查询快”。

对 balanced associative block，`k` 个 changed leaves 的 ancestor-path union 是
`O(k log(1+C/k))`。这是 Tree ensemble sums 和 MLP 内部 balanced product/sum blocks
的可证明 specialization；一般 circuit 使用实测 `alpha`。

功能层面的优势如下：

| 步骤 | 对比对象 | 优势 |
|---|---|---|
| 一次 abduction | replay | online query 不再恢复 residual |
| trajectory compilation | global-region materialization | 不枚举 intervention line 未到达的组合 |
| 三类 exact compiler | 三套独立 index | 统一输出 regions、slot deltas 与 certificates |
| persistent VSC | IIP | 只重建 affected ancestors，并全局复用 canonical subcircuits |
| region root directory | event replay | query 不重放 boundary updates |
| response/certificate 分离 | 不分离的 guard segments | POINT 保持 response-minimal，audit 仍精确 |
| optional MGPF/LCF | 仅 hash-consing | 在 additive frontier 进一步做代数 factorization |
| registered event index | unit scan | FLIP output-sensitive reporting |
| declared aggregate map | replay-and-sum | preprocessing 后 logarithmic lookup |

论文最醒目的物理设计应写成：

```text
GPSI = exact L/T/N trajectory compiler
     + Versioned Semiring Circuit arena
     + per-unit boundary -> persistent-root directory
     + independent unit bindings h_u
     + optional additive-frontier optimizer
     + provenance-native secondary query indexes
```

## 20. Versioning 与 Maintenance

所有数据按 model version 建 key。若 mechanism `f_v` 改变：

1. 为 indexed units 重新计算 `n[u,v]`；
2. 从 `v` 到受影响 declared targets 重建 causal slice；
3. 创建新的 template、response、certificate、event 和 aggregate roots；
4. 只有 canonical identity 在版本策略下仍有效时才复用 immutable nodes。

第一版采用 conservative causal-slice rebuild。Fine-grained incremental patching 属于 future work。

## 21. 实验契约

### 21.1 Exactness

- 与直接 fitted-model abduction-action-prediction 对比；
- 测试 boundaries、one-sided limits、tree jumps、zero slopes、ReLU zeros；
- 报告 absolute/relative errors 和 branch mismatches。

### 21.2 Construction

- 分开报告 abduction、local compilation、region diff、affected-ancestor
  recomputation、arena interning、region-directory build 和 total build time；
- 报告 realized segments 与 potential global regions；
- 报告每个 transition 的 `k[u,j]`、`alpha[u,j]`、`gamma[u,j]`；
- 报告相对于 cached-noise replay 的 amortization query count。

### 21.3 Compression

- 报告 response/certificate segments；
- 主实验报告 `B_tot,D_ind,D_0,A,G,H`、unique roots、recurring roots、arena nodes
  和 serialized bytes；
- 对比 IIP、只存 event log、仅 adjacent path-copy、global hash-consed VSC；
- 分别改变 transition sparsity、pattern recurrence 和 cross-unit template reuse；
- 改变 population、causal path multiplicity、tree depth/count、MLP width/depth；
- 包含无共享的 worst-case workload。

Optional additive-frontier ablation 另行报告 `m,d,ell,F,F*,rank(M),K,D_f`、
reset max-flow 时间和 serialized bytes。它不能替代 VSC 对 IIP 的主实验。

### 21.4 Queries

- 分别测试 POINT、response EXPLAIN、audit EXPLAIN、registered FLIP、declared AGG；
- 使用第 16 节指定的对应 baseline；
- 改变 units、intervention values、selectivity 和 repeated-query count。

### 21.5 Theory-Experiment Alignment

| Theory claim | 必须对应的实验 |
|---|---|
| VSC exactness | 每个 version 的 free-semiring polynomial 与 independent materialization 相等 |
| change-sensitive construction | 每个 transition 记录 `k,alpha,gamma`，验证 `gamma<=alpha` 并测量 `A/D_ind` |
| balanced batch bound | 改变 `C,k`，报告 visited ancestors 与 `k log(1+C/k)` 的比例 |
| global interning | 构造非相邻重复 patterns 与跨 unit 同构 templates，验证 root ID reuse 和独立 `h_u` |
| worst-case non-claim | 构造大 affected cone 且无重复 node 的 workload，展示压缩可消失 |
| query bounds | 分离 boundary lookup、event replay、compressed-root return 和 output traversal；IIP lookup 单列 |
| optional LCF claims | 仅在 additive-frontier ablation 中以 ILP/rank/run 指标验证，不推广到 general VSC |

## 22. Related-Work 边界

本设计使用但不声称以下工作的 novelty：

- DoWhy counterfactual replay：
  <https://www.pywhy.org/dowhy/v0.14/user_guide/causal_tasks/what_if/counterfactuals.html>
- partial persistence and path copying：
  <https://www.sciencedirect.com/science/article/pii/0022000089900342>
- dynamic expression trees：
  <https://cs.brown.edu/research/pubs/techreports/reports/CS-90-35.html>
- provenance semirings：
  <https://doi.org/10.1145/1265530.1265535>
- biclique edge partition hardness/inapproximability：
  <https://doi.org/10.1587/transinf.E93.D.290>
- biclique cover/partition parameterized complexity：
  <https://doi.org/10.4230/LIPIcs.IPEC.2016.11>
- nested/2-nested matrices and the disjoint-or-nested conflict graph：
  <https://arxiv.org/abs/1906.11970>
- biclique partitions on co-chordal exact subclasses：
  <https://doi.org/10.1016/j.dam.2023.05.001>
- optimal dynamic stabbing-semigroup index：
  <https://doi.org/10.1137/10078791X>
- ExactLine：
  <https://papers.nips.cc/paper_files/paper/2019/hash/908075ea2c025c335f4865f7db427062-Abstract.html>
- tree-ensemble decision diagrams：
  <https://eldorado.tu-dortmund.de/server/api/core/bitstreams/41b47e64-79f2-47fa-baf2-91e0dcf8df6c/content>
- FACET：
  <https://doi.org/10.1145/3626729>
- CFDB：
  <https://doi.org/10.1145/3514221.3520162>

Hash-consing、path copying、dynamic expression trees 和 support crossing graph 本身
都不是 novelty claim。TESSERA 的候选贡献必须落在如下完整链条：unit-level SCM
abduction；L/T/N 的 exact intervention-line compiler；provenance-valued region deltas；
任意切换的 mixed `ADD/MUL` persistent versions；change-sensitive construction/storage；
以及 provenance-native queries。MGPF/LCF 只在 additive frontier 上提供 secondary
algebraic compression。投稿前仍需系统核对 dynamic circuit maintenance、persistent
expression DAG、tree-ensemble DD 与 provenance index 的交叉文献。

候选 scoped first claim：

> To the best of our knowledge, TESSERA is the first system to materialize,
> factorize, and index exact semiring provenance for unit-level parametric
> counterfactual responses over learned structural causal models.

投稿前必须完成系统性 related-work search 才能保留该表述。

## 23. 设计冻结检查表

- [x] VSC 已替换 LCF 成为 general storage layer；
- [x] arbitrary `ON/OFF/ON` 与 mixed `ADD/MUL` exactness 已定义并有实现测试；
- [x] `alpha` construction 与 `gamma` storage bounds 已分离；
- [x] IIP 与 event-log baselines 已分开，未混用 query speedup；
- [ ] canonical VSC 的明确 representation-model lower bound 已完成；
- [ ] fitted Tree compiler 已接入 VSC；
- [ ] fitted ReLU/ExactLine compiler 与 semiring product tree 已接入 VSC；
- [ ] 三类函数的反事实推导及数值例子已确认；
- [ ] 统一 GPResponse 和 COMPILE_CELL contract 已确认；
- [ ] response/certificate 分离已确认；
- [ ] per-template reverse trie 与 subtree hash-consing 已确认；
- [ ] POINT、EXPLAIN、registered-FLIP、declared-AGG 语义已确认；
- [ ] B0-B6 的职责和公平性已确认；
- [ ] MGPF、occurrence lifting 和 factor-root objective 已确认；
- [ ] LCF exactness、rank tightness、reset min-cut、`ell`-approximation 与
      `K<=min(R_S,R_A)<=R` proofs 已确认；
- [ ] Prefix-DAG 只保留 size lemma，不再包含未核实的 global minimality；
- [ ] biclique-partition、laminar-family 与 semigroup-index related work 已系统核对；
- [ ] exactness、floating point 和 worst-case 边界已确认；
- [ ] 全部设计通过后再开始 nonlinear implementation。
