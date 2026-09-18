# 物理 Lambertian γ 判定与新谱界紧度（M13 / M15）

状态：experimental — NOT part of the published results。日期：2026-09-18。

本文件是本次限定新增的数学说明。完整可 `input` 的中文证明正文位于 `C:/Users/35702/publication/mode-resolved-calibration/docs/theory/section_physical_gamma.tex`。新能力隔离于 `experiments`，没有新增 `src` API，没有改变历史 `gamma_lower_bound` 的行为，没有执行 Git 命令。

## 1. 结论与边界

| 项目 | 本次结论 | 不应延伸的结论 |
|---|---|---|
| M13：物理子类 | 给出全部参数显式、有理数可验证、由 `LightBlocks` 实际装配的三灯反例。固定 κ=10、固定 α，γ 可趋零 | 不声称同质对角先验这一更窄子类也已否定；本例允许逐灯异质、强度—方向相关先验 |
| M15：全迹保证 | 新保证为 `max(b2, 4*bfull/(1+bfull)^2)`，其中 b2 留出两个非零候选，因此不弱于已证留一界 b1 | 不允许把 Kantorovich 项的 bfull 换成 b1；也不适用于原坐标下任意任务权重 |
| M15：紧度 | 二维、两个非零候选、固定基线条件数 χ 时，`inf γ=1/χ`；给出二次收敛参数族 | 此族更新范数发散，不解决同时固定 κ、m、χ、更新规模的完整最坏值 |
| 现实对照 | 11 对象×2 level，全 22 设计点重算，新有效保证较旧有效谱界提高至少约 3.92–4 倍 | 没有对真实 48 候选格穷举，未测真实 γ；旧 α 表达式不是保证 |

## 2. 物理反例的完整参数

所有量与 `NominalScene` 的固定法向、加性反照率 Jacobian 对齐：

\[
s_{kp}=(n_p^T\ell_k)_+,\quad
B_{\phi,k}(p,:)=[\rho_p s_{kp},\rho_p h_{kp}n_p^Tt_{1k},\rho_p h_{kp}n_p^Tt_{2k}].
\]

本例名义反照率 ρ=(1,1)，在此点加性反照率与 log-albedo 一阶 Jacobian 数值相同。**没有忽略 shading 和方向导数的耦合**。

令

\[
0<q<1,\quad \varepsilon=\frac{2q}{1+q^2},\quad c=\frac{1-q^2}{1+q^2},\quad c^2+\varepsilon^2=1.
\]

- 像素数 P=2，独立标定灯数 L=3，κ=10。
- 法向 `n1=(3/5,4/5,0)`、`n2=(-c,ε,0)`。
- 灯方向 `l0=ex`、`l+=ey`、`l-=ey`。
- 切基 `t1=l×ez`、`t2=l×t1`，与代码在这些方向的约定一致。
- 权重 `w0=(25/9,1)`、`w+=w-=(25/16,1)`，全部严格正。
- shading `s0=(3/5,0)`、`s+=s-=(4/5,ε)`；灯 0 的第二像素严格背向，不位于 Lambertian 截断折点。
- Jacobian

\[
B_{\phi,0}=\begin{pmatrix}3/5&-4/5&0\\0&0&0\end{pmatrix},\qquad
B_{\phi,\pm}=\begin{pmatrix}4/5&3/5&0\\\varepsilon&-c&0\end{pmatrix}.
\]

定义

\[
H=\begin{pmatrix}1&3/4\\\varepsilon&-c\end{pmatrix},\quad
S_\pm=\frac1{20}\begin{pmatrix}101&\pm99\\\pm99&101\end{pmatrix},
\]

\[
\Lambda_0=\operatorname{diag}(25/9,25/9,1),\qquad
\Lambda_\pm=(H^TS_\pm^{-1}H)\oplus 1.
\]

`S±` 特征值为 10、1/10，`det H=-c-3ε/4≠0`，所以全部精度正定且有限。有理 q 保证几何、权重、先验全部有理。

### 精确装配，不绕过 LightBlocks

`A_k=diag(sqrt(w_k)*s_k)`、`B_k=sqrt(w_k)*B_phi[k]`。

灯 ± 的白化 B 前两列等于 H，白化观测 Jacobian 为 `D=diag(1,ε)`。由矩阵恒等式

\[
I-H(H^TH+tH^TS_\pm^{-1}H)^{-1}H^T=t(tI+S_\pm)^{-1},
\]

其信息贡献为 `D t(tI+S±)^-1 D`。在 t=1 时两灯的和等于 `D²`；灯 0 贡献 `diag(t/(1+t),0)`。因此

\[
A=\operatorname{diag}(3/2,\varepsilon^2),\qquad W_0=\operatorname{diag}(9/22,0),
\]

\[
W_\pm=\begin{pmatrix}99/404&\pm729\varepsilon/4444\\
\pm729\varepsilon/4444&99\varepsilon^2/404\end{pmatrix}.
\]

`physical_lambertian_family` 从 n、l 构建 Bφ，再构建 `LightBlocks`，不直接把这些 2×2 结果冒充物理构造。`physical_exact_certificate` 用 SymPy 独立从有理物理参数重做全部块乘积和求逆，并断言结果等于上述式子；测试再将两条路径逐一对拍所有 8 个子集。

白化总更新

\[
A^{-1/2}\sum W_k A^{-1/2}=\operatorname{diag}(666/1111,99/202)\prec9I,
\]

显式满足共享精度结构约束。

### 固定 α 和 κ 的二次退化

\[
\alpha=\frac{165}{808}+\frac{3\sqrt{172105}}{8888}
=0.34423561987863397\ldots
\]

不依赖 ε。灯 0 的 α 为 3/11，而两个混合灯给出上式。

\[
d(A,W_0)=\frac17,\qquad
\gamma\le R(\varepsilon)=
\frac{3025408256\varepsilon^2}{77(30614089\varepsilon^2+531441)}
\le\frac{3025408256}{40920957}\varepsilon^2\longrightarrow0.
\]

因此这不是仅在“一般共享线性模型”中的否定，而是在明确物理 Lambertian Jacobian 子类中的否定。固定 κ 和固定 α 均不能提供正统一保证；固定条件数的附加约束不在本否定之内，因为 `χ(A)=3/(2ε²)` 发散。

当 q=1/100 时，ε=200/10001。精确穷举 27 个三元组（包含 S=T）给

\[
\gamma=\frac{121016330240000}{4187205554180957}
=0.0289014543647527077010839874981057461776344593\ldots,
\]

最小见证为 `x=0,S=∅,T={+}` 或 `{−}`。分母边际为 `598172222025851/121016330240000`。精确不等式 `γ<3/100<1/2<1/(1+α)` 已足以判否，无需依赖浮点容差。

此例新旧谱保证的实际差距：

| 量 | 数值 |
|---|---:|
| 原有效 leave-one b1 | 0.00018565135387649704 |
| leave-two b2 | 0.00020948191104678112 |
| Kantorovich full / 组合新界 | 0.0006665361789779346 |
| γ / 原有效界 | 155.675968751508 |
| γ / 组合新界 | 43.360668597269765 |

字段前缀均为 `mathematics.physical_counterexample`，前三行为 `.bounds.leave_one_spectral`、`.bounds.leave_two_spectral`、`.bounds.improved`，后两行 `.gamma_over_leave_one`、`.gamma_over_improved`。

### 可见性、重复灯、秩及已排除路线

- `v_camera=(ε/(2c),1,δ)`（δ>0）与两个 n 和所有 l 点积严格正；共同刚体旋转到该相机 z 轴即可使 `n_z,l_z>0`。几何点积、切向导数和信息块不变。
- 两个同向灯是独立 nuisance 的物理光源/曝光，不违反模型。有限反例的严格不等式也在足够小的合法方向扰动下保持。
- 第三 nuisance 列为零，但先验有限正定；这只是合法观测秩亏。耦合实际秩为 2，不是 3L；未受影响保留率模式数为 P−2=0。不能强制 `P−3L` 取等号。
- 一般线性反例的混合像素观测行与任意 nuisance-free 补块，不是每灯对角 A，不能直接搬入。
- 若只保留 scalar logI nuisance，`u_kp=w_kp ρ_p s_kp²≥0`。对角基线要求 `sum_k u_kp u_kq/(M0k+λk)=0`，每项都非负，所以每项都为零，所有更新也对角，γ=1。故“只用强度且保持精确对角基线”的路线已严格排除。本构造使用真实方向导数及相关先验的交叉项抵消，避开这个阻塞。

## 3. 新谱界证明

令非零候选集为 U+，m=|U+|，`a0=λmin(A)`，并记

\[
b_{\rm full}=\frac{a_0}{\lambda_{\max}(A+\sum_{i\in U_+}W_i)},\quad
b_1=\frac{a_0}{\max_x\lambda_{\max}(A+\sum_{i\ne x}W_i)},\quad
b_2=\frac{a_0}{\max_{x\ne y}\lambda_{\max}(A+\sum_{i\ne x,y}W_i)}.
\]

m<2 时 γ=1；m≥2 时

\[
\boxed{\gamma\ge\max\{b_2,4b_{\rm full}/(1+b_{\rm full})^2\}\ge b_1.}
\]

### 3.1 留二项

S=T 比值为 1。对于严格 S⊂T，删去零更新后选 y∈T\S，则 `M(S)≤M(U+\{x,y})`。已证成对界

`d(M,W)/d(N,W) ≥ λmin(N)/λmax(M)`

给出 b2。显然 b2≥b1，所以新的组合保证不弱于现有一般界。

### 3.2 Kantorovich 项：没有逆平方保序错误

若 `0<X≤Y` 且 `m0 I≤X≤M0 I`，则

\[
X^2\preceq(m_0+M_0)X-m_0M_0I
\preceq(m_0+M_0)Y-m_0M_0I
\preceq\frac{(m_0+M_0)^2}{4m_0M_0}Y^2.
\]

首步用 X 的谱，末步将 Y 的标量多项式配成平方。对于边际积分令

`X=(N+tW)^−1`、`Y=(M+tW)^−1`、t∈[0,1]。

以总矩阵和基线给出共同特征值范围，并使用

\[
d(P,W)=\int_0^1\operatorname{tr}[W(P+tW)^{-2}]\,dt,
\]

即得所述 bound。路径经过 tWx，**必须使用完整集合的 bfull**。无证明的 `4*b1/(1+b1)^2` 未入实现、结果或结论。

在结构 `M(U+)≤κA` 下还得到

\[
\gamma\ge\frac{4\kappa\chi}{(\kappa\chi+1)^2}.
\]

这是附加固定条件数的有效保证，不是其最坏常数已经证明最优。

## 4. 精确紧度：限定为两候选、固定 χ

取固定 0<a<1、h→0+：

\[
A=\operatorname{diag}(1,a),\quad
W_0=h^{-4}e_1e_1^T,\quad W_1=h^{-4}(1,h)(1,h)^T.
\]

m=2 意味着 b2=a。两个严格三元组的比值由有理逆矩阵计算得到

\[
R_0=a+(2a^2-a+1)h^2+O(h^4),\qquad
R_1=a+(2a^2-1+a^{-1})h^2+O(h^4).
\]

差的二阶系数为 `(a−1)²/a>0`；足够小 h 时 `γ=R0`。因此

\[
\inf_{\dim=2,m=2,\operatorname{cond}(A)=\chi}\gamma=\frac1\chi,
\quad
\gamma-\frac1\chi=(2a^2-a+1)h^2+O(h^4).
\]

旧 b1=`a h⁴+O(h⁶)`，新的组合界最终恰为 a。a=1 端点为 γ≡1，不能套用前述非恒定展开。

取 a=1/4 的实际有理检查：

| h | 精确有理 γ 的小数 | 原 b1 | 新界 | `(γ−1/4)/h²` |
|---|---:|---:|---:|---:|
| 1/4 | 0.30367613670622046 | 0.0009158985446156755 | 0.25 | 0.8588181872995274 |
| 1/8 | 0.26361030119223805 | 6.0081877704733e-05 | 0.25 | 0.8710592763032369 |
| 1/16 | 0.2534141456618505 | 3.799796499457e-06 | 0.25 | 0.8740212894337241 |
| 1/32 | 0.25085425363791825 | 2.3818574884578243e-07 | 0.25 | 0.8747557252282833 |
| 1/64 | 0.25021360814366733 | 1.4897523215419863e-08 | 0.25 | 0.8749389564612788 |

字段：`mathematics.tightness_two_candidates.rows[]` 内 `.h_exact`、`.gamma_exact`、`.gamma`、`.bounds.leave_one_spectral`、`.bounds.improved`、`.scaled_gap`；二阶极限 `.scaled_gap_limit=7/8`。

物理族的新界本身也为 Θ(ε²)，加上 M13 的 O(ε²) 上界得到物理 γ 的阶紧度；见证比值 / 新保证的极限为 `672018808864/15154394409≈44.344814495846606`。这只证明该物理族的阶与这个见证的差距，不声称该常数最优。

## 5. 低秩现实计算及 11 设计对照

对 `E=A+sum W` 求一次 top-r 特征子空间 Q。对于 `E−VVᵀ`，定义

`aQ=λmax(Qᵀ(E−VVᵀ)Q)`、`cQ=λ_(r+1)(E)`、`bQ=||QᵀVVᵀQperp||`。

由分块 Rayleigh 商

\[
a_Q\le\lambda_{\max}(E-VV^T)
\le\min\{\lambda_{\max}(E),(a_Q+c_Q+\sqrt{(a_Q-c_Q)^2+4b_Q^2})/2\}.
\]

令 Z=QᵀV，则 `bQ²=λmax(Z(VᵀV−ZᵀZ)Zᵀ)`。所以每个移除集合只做小 Gram 运算，不做逐候选 P×P 分解，也不错误地从总谱减去低秩非零最小特征值。

浮点端点加 safety pad；文中精确算术不等式与普通 LAPACK 数值明确区分，未宣称区间算术。b1/b2 的下端返回保证，上端用于报告精确谱公式被包围的位置。若 denominator 下端被 pad 截到 0，上端界直接为 1，避免除零。小矩阵路径重新从 A 累加 leave-one/two，避免 h⁻⁴ 大更新相减消去基线。

现实计算复用原加载器、corrected 噪声和 `default_rng([20260910,obj_idx])`：每场景 1200 像素、48 活跃灯，2 个 level；94 零占位完全省略不会改变 γ。旧表的 α 只用于配置一致性检查。

以下“原界”列为有效 b1 包围的下端，不是被否定的旧 α 表达式。

| 对象 | 原界，level 0.1 | 新界，level 0.1 | 原界，level 0.5 | 新界，level 0.5 |
|---|---:|---:|---:|---:|
| obj_03_pumpkin | 0.0018088882 | 0.0072094469 | 7.8667718e-05 | 0.00031462137 |
| obj_04_dolphin | 2.7896996e-05 | 0.00011158176 | 1.1273426e-06 | 4.5093604e-06 |
| obj_07_pumpkin2 | 9.1011378e-06 | 3.6403889e-05 | 3.6886246e-07 | 1.4754488e-06 |
| obj_09_ball | 0.00079505525 | 0.0031751701 | 3.3422731e-05 | 0.00013368199 |
| obj_10_pumpkin3 | 1.3524644e-05 | 5.4097111e-05 | 5.7225822e-07 | 2.2890303e-06 |
| obj_11_pine | 0.010568103 | 0.041392902 | 0.00051498017 | 0.0020578007 |
| obj_13_mushroom | 0.0026254993 | 0.010447068 | 0.00010916166 | 0.00043655132 |
| obj_16_friends_cup | 0.0059576388 | 0.023549125 | 0.00026125676 | 0.0010444812 |
| obj_17_pumpkin5 | 0.00080721014 | 0.0032236342 | 3.5009885e-05 | 0.00014002974 |
| obj_18_fabric_hat | 0.0062570976 | 0.024718096 | 0.00029246982 | 0.0011691953 |
| obj_19_cylinder | 3.2732057e-05 | 0.00013091966 | 1.3181081e-06 | 5.2724187e-06 |

行定位：`real_designs.rows[object=...,level=...]`；原列 `.bounds.leave_one_spectral_interval[0]`，新列 `.bounds.improved`。有证明的改进倍数使用旧界包围的**上端**作分母，字段 `.improvement_factor_vs_valid_leave_one_at_least`。

汇总数值与字段：

- 对象 11、设计点 22：`real_designs.cohort_size`、`.design_points`。
- 改进倍数最小 3.9167767388200643、最大 3.999912235212594：`real_designs.summary.improvement_factor_min/max`。
- 旧 b1 包围最大相对宽度 9.227368960942055e-05：`.summary.leave_one_bracket_relative_width_max`。
- 新界范围 `[1.4754487656372166e-6, 0.04139290196762592]`：`.summary.new_lower_bound_min/max`。
- α 与旧表六位小数的最大差 4.849694582031994e-07：`.summary.alpha_archive_abs_difference_max`，小于最后一位的半单位，配置对拍通过。
- 历史结果 hash 前后一致：`.historical_artifact_sha256_before/after`；core/runner 当前 hash 与 `provenance.core_sha256/runner_sha256` 一致。

## 6. 测试、复现与文件索引

最后实际运行：**34 passed in 1.53s**；完整 22 设计重算 **26.43 秒**。机器性能数字是本次运行记录，不是算法复杂度保证。

测试覆盖：有理物理成员资格、全部 8 子集的 LightBlocks 装配、27 个精确比值、固定 α 和二次率、共同开放半球、PSD/SPD 与零更新守卫、随机小规模穷举、inverse-square Kantorovich 比较、符号紧族系数、h=1e-5 的基线消差回归、top-r 包围以及 pad 下端为 0 的除零回归。

新增实验产物另记录 40 个随机实例、1839 个三元组、0 违反：`random_exhaustive.instances/triples/violations`。随机穷举只验证实现和寻找反例，不构成紧度证据。

Git Bash 中可从任意目录执行（不写 pytest cache 或 bytecode）：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -o addopts='' -q C:/Users/35702/publication/mode-resolved-calibration/tests/test_gamma_extensions.py
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python C:/Users/35702/publication/mode-resolved-calibration/experiments/gamma_extensions.py --real
```

限定新增的六个文件：

1. `C:/Users/35702/publication/mode-resolved-calibration/experiments/theory_gamma.py`：实验核心。
2. `C:/Users/35702/publication/mode-resolved-calibration/experiments/gamma_extensions.py`：唯一结果路径的入口。
3. `C:/Users/35702/publication/mode-resolved-calibration/tests/test_gamma_extensions.py`：新增测试。
4. `C:/Users/35702/publication/mode-resolved-calibration/docs/theory/physical_gamma_and_tightness.md`：本说明。
5. `C:/Users/35702/publication/mode-resolved-calibration/docs/theory/section_physical_gamma.tex`：无导言的中文证明正文。
6. `C:/Users/35702/publication/mode-resolved-calibration/results/theory_extension_20260918/gamma_extensions.json`：全部新实验和精确证据。

## 7. 保留的未证范围

- 更窄的共享对角、方向各向同性先验 Lambertian 子类的完整 γ 判定。
- 同时固定 κ、m、χ 与更新范数的最优 γ；本次只给有效正界，不声称达到最优。
- 新 Kantorovich 成对常数的最优性。
- 任意 rank-deficient 任务的统一正界：事实上有反例；`M=I2,N=I2+11ᵀ/2,W=diag(1,0),H=(0,1)` 给任务边际 0 与 1/28。
- 对 `Q=HᵀH>0` 可变换 `Mtilde=Q^-1/2 M Q^-1/2` 后重新计算证书，但不能复用未加权原谱系数。
- 真实 22 点的准确 γ 未求，不把保证增强等同于重建收益或决策收益。
