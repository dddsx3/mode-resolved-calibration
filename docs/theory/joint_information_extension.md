# M17：联合反照率—法向信息模型及保留的结构结论

本节是对六项任务中⑤的模型级判定与小规模独立验证。证明不依赖实证数值；数值实验只验证实现。实现隔离于 `experiments/theory_joint.py`，没有替换历史估计器或冻结结果。

## 1. 先校正已有代码的参数含义

`experiments/openillumination_validation.py` 中 `A_k=diag(sqrt(w_k)*s_hat[k])` 是对 **加性反照率扰动** `δρ` 的导数；光强列则为 `sqrt(w_k)*ρ*s_hat[k]`。因此历史信息变量不是文档曾称的 log-albedo。令 `Dρ=diag(ρ)`，在 `ρ_p>0` 时有

\[
δρ=D_ρ δβ,
\quad A_β=A_ρD_ρ,
\quad F_β=D_ρF_ρD_ρ,
\quad H_β=H_ρD_ρ.
\]

只有同时变换信息和任务，`tr(H F^{-1}H^T)` 才不变。未加权全迹本身不是任意重新参数化下的不变量。这里不重算或重新命名历史产物内的字段，只纠正文档解释。

另一处必须澄清的事实是：`src/calibinfo/estimators/joint_map.py` 接收固定 `n_gt`，联合的是反照率与灯，而非反照率与场景法向。它不是本节所需的联合场景估计器，且未调用其任何 `diagnose_trace` 路线。

## 2. 物理模型、坐标与完整 Jacobian

固定一个不在阴影边界上的名义场景，观察

\[
y_{kp}=\rho_p I_k [n_p^T\ell_k]_+ + \epsilon_{kp},
\qquad \rho_p,I_k>0,\quad \|n_p\|=\|\ell_k\|=1.
\]

用 `diagnostics.tangent_basis` 对每个单位向量构造同一规则的正交切基 `T_p`、`E_k`。局部参数采用

\[
x_p=(\delta\beta_p,\eta_p)\in\mathbb R^3,
\quad c_k=(\delta\iota_k,\zeta_k)\in\mathbb R^3,
\]

其中 `β=log ρ`、`ι=log I`，法向与灯向图为

\[
n_p(\eta)=\frac{n_p+T_p\eta}{\sqrt{1+\|\eta\|^2}},
\qquad
\ell_k(\zeta)=\frac{\ell_k+E_k\zeta}{\sqrt{1+\|\zeta\|^2}}.
\]

令 `h_kp=1{n_p^T l_k>0}`、`s_kp=[n_p^T l_k]_+`，则名义点单行的非零导数为

\[
A_{kp,p}=\sqrt{w_{kp}}\rho_pI_k
  [s_{kp},\;h_{kp}\ell_k^TT_p],
\]

\[
B_{kp,k}=\sqrt{w_{kp}}\rho_pI_k
  [s_{kp},\;h_{kp}n_p^TE_k].
\]

`A` 的其他像素块及 `B` 的其他灯块为零。若 `n·l=0`，正部函数不可微，不能将任意一侧 Jacobian 当作本定理；实现明确拒绝这类名义点。有限非线性实验另记录可见性翻转数。

因此场景有 **3P 个内禀自由度**，不是无约束 `4P`。`(log ρ,n∈R³)` 的 `4P` 是带 P 条单位长度约束的环境坐标。

## 3. Schur 信息、Gram 结构与低秩路线

先用现有 `whitening.whiten_system` 白化，使传感器噪声为 `N(0,σ²I)`。每灯 nuisance 独立，惩罚为 `c_k^T t_k Λ0k c_k`。定义

\[
D=A^TA=\operatorname{blockdiag}(D_1,\ldots,D_P),
\quad U_k=A_k^TB_k,\quad M_{0k}=B_k^TB_k.
\]

`D_p` 是 `3×3` Gram 块，而不是标量。以现有 `schur.delta_f` 的变分定义为准，正定 nuisance 块时有

\[
F(t)=D-\sum_k U_k(M_{0k}+t_k\Lambda_{0k})^{-1}U_k^T.
\]

所有项必须源自同一 `A,B`；不能独立拼出 `D,U,M0`。**此普通逆表达式要求每个 nuisance 块可逆**；PSD 惩罚可能留下 nuisance 核，即便场景已取商也不保证块可逆。此时采用变分定义，或在 nuisance 支撑商上用广义逆实现。实验 constructor、梯度与匹配 Gaussian 模拟限于有限 SPD 先验；flat 情形另用既有 `schur.delta_f(...,0)` 和固定商空间核验，不冒称 constructor 支持全部 PSD。

数值实现以逐灯 SPD 消元计算拟合 nuisance，再由“残差平方加先验平方”的非负分解装配信息，避免全局 SVD 相对截断使一盏极高精度灯抹去其他灯。复用低秩例程前，对每灯 `M0,Λ0` 同除该块尺度，`U` 同除尺度平方根；这不改变信息。若归一后旧低秩阈值仍会截断正谱，则明确拒绝该低秩路径，改用稠密 SPD 路径，不静默改模型。旧文档中的 `u_k u_k^T=F∞M0k` 并非一般矩阵恒等式，在通常维数下右端甚至不可乘。共同 Gram 半正定性与下面的变分形式才是正确结构。

固定可辨识基 `Q`，令 `D_Q=Q^TDQ≻0`、`T=QD_Q^{-1/2}`。则

\[
R=T^TFT=I-VV^T,
\quad V=[T^TU_k K_k^{1/2}]_k.
\]

`#(ρ=1)=dim(Q)-rank(V)`，且 `rank(V)≤min(dim(Q),3L)`。只有 `rank(V)=3L` 时才有精确的 `dim(Q)-3L` 个单位模式；不能将这个有条件等式用于 `dim(Q)<3L` 或退化灯组。

低秩计算把 `T^TU_k` 送入现有 `LightBlocks`，其基线设为单位对角，而任务同时设为 `H T`。于是

\[
J_H=\operatorname{tr}[H Q(Q^TFQ)^{-1}Q^T H^T]
=\operatorname{tr}[(HT)R^{-1}(HT)^T].
\]

这允许直接复用 `woodbury_quad_risk`。**只白化信息而不变换 H 会改变任务，尤其不能借此把加权全迹当成未加权全迹。**

## 4. 可辨识性、尺度与更大的规范群

### 4.1 Proper 先验：无需人为删除已经被先验抬升的方向

若全部 `Λ0k≻0`，则

\[
F(t)=A^T(I+B\Lambda(t)^{-1}B^T)^{-1}A,
\qquad \ker F(t)=\ker A.
\]

对每个像素，`[n_p,T_p]` 为可逆正交矩阵，故

\[
\operatorname{rank}D_p
=\dim\operatorname{span}\{\ell_k:w_{kp}>0,\ n_p^T\ell_k>0\}.
\]

因此每个像素至少有三条**线性独立**的正权重亮灯方向，是 `rank A=3P` 的充要条件；三条灯数本身不够。平行灯只有一维、共面但非平行灯至多二维、完全背光像素零维。正的反照率与光强是这一等价关系的前提。

图像中的尺度共变关系在本坐标下仍为 `Aa=B cbar`，其中 `a_p=(1,0,0)`、`cbar_k=(1,0,0)`。物理不变方向在全 Jacobian 中是 `(a,-cbar)`。本实现直接调用现有 `information/gauge.py::gauge_response`，核对其谱响应与联合 Schur 的 `a^TFa`，没有另造与原规范定义冲突的标量规则。Proper 标定先验使绝对尺度具有统计参考，不应再删除该方向并称其“不可辨识”。

### 4.2 Flat/半正定惩罚：先定商空间，后求逆

对 `Λ0⪰0`，由

\[
x^TF(t)x=\min_c\{\|Ax-Bc\|^2+c^T\Lambda(t)c\}
\]

得到精确核公式

\[
\ker F(t)=\{x:\exists c,\ Ax=Bc,\ \Lambda_0c=0\}.
\]

所有 `t_k>0` 时先验核不变，故该场景核不随分配改变。用现有 `metrics.spectral_criteria.identifiable_subspace` 在参考 `F(1)` 上构造一次 `Q`，然后始终在该固定商空间内装配、计算任务和证书。`gauge_response` 本身只算信息响应，并不返回投影基；不能把它误说成通用去规范器。

完全平坦 `Λ=0` 时还有精确维数式

\[
\operatorname{rank}F(0)
=\operatorname{rank}[A\ B]-\operatorname{rank}B.
\]

完全观测、全亮、无积分性/深度约束时，记 `g_p=ρ_pn_p`、`v_k=I_k l_k`，模型就是 `g_p^Tv_k`。若两边三维因子都满秩，则

\[
g_p\mapsto Gg_p,\qquad v_k\mapsto G^{-T}v_k,
\qquad G\in GL(3)
\]

形成九维局部规范群，而不只是一个尺度或三个旋转。对任意 `3×3` 矩阵 `E`，显式无穷小生成元为

\[
\delta\beta_p=n_p^TEn_p,
\quad \eta_p=T_p^TEn_p,
\quad \delta\iota_k=-\ell_k^TE^T\ell_k,
\quad \zeta_k=-E_k^TE^T\ell_k.
\]

代入 Jacobian 逐行得零。完整秩三矩阵分解的全部一阶核恰由这九个生成元构成：若 `δGV^T+GδV^T=0`，左乘 `G` 列空间的正交补，再用 V 满列秩，即知 `δG=GE^T`；代回知 `δV=-VE`。故满秩完全观测下全 Jacobian 秩为 `3P+3L−9`、`rank B=3L`，剖面场景可辨识维数为 **3P−9**。缺失观测、退化法向或灯向时应使用一般秩公式，不能机械减九。单位球坐标只是去掉冗余，并不消灭 `GL(3)` 的物理分解歧义。

奇异 **proper covariance** 的零方向表示硬已知，不等于平坦惩罚；仍须用仓库协方差因子路线，禁止用伪逆精度互换两种语义。

## 5. 上限与凸分配证书：在固定模型下仍成立

### 命题 M17-A（保留上限）

对同一固定 `A,B`、`Λ0k⪰0`，以及 `1≤t_k≤κ`，在上节固定正定商空间内，任意固定线性任务 H 且 `J_H(1)>0` 满足

\[
F(1)\preceq F(t)\preceq\kappa F(1),
\qquad 0\le V_H(t)=1-J_H(t)/J_H(1)\le1-1/\kappa.
\]

**证明。** 每个固定 c 都满足

\[
\|Ax-Bc\|^2+\sum_k c_k^T\Lambda_{0k}c_k
\le\|Ax-Bc\|^2+\sum_k t_k c_k^T\Lambda_{0k}c_k
\le\kappa\big(\|Ax-Bc\|^2+\sum_k c_k^T\Lambda_{0k}c_k\big).
\]

分别取最小值得 Loewner 链；限制到固定商空间后逆序，最后以 `H^TH⪰0` 取迹。H 秩亏不影响此结论，但恒零任务的相对价值 `0/0` 不定义。证毕。

### 命题 M17-B（保留凸性与证书）

固定 `A,B,Λ0,H,Q` 时，`F(t)` 关于 t 算子凹且递增（逐方向的仿射函数族下确界）；`tr(H_Q F_Q(t)^{-1}H_Q^T)` 是凸函数。若 `Λ0k≻0`、`F_Q(t)≻0`，其梯度为

\[
\partial_{t_k}J_H=-\operatorname{tr}\{K_k\Lambda_{0k}K_k
 (U_k^TQZ)(U_k^TQZ)^T\},
\quad Z=F_Q(t)^{-1}H_Q^T.
\]

对盒约束及线性预算集合，复用 `CertificateProblem.lmo` 得顶点 `s*`，则

\[
g(t)=\nabla J_H(t)^T(t-s^*)\ge0,
\quad J_H(t)-g(t)\le J^*_{\rm relax}\le J^*_{\rm discrete}.
\]

这既不依赖 γ 候选界，也不要求任务是未加权全迹。改变场景线性化、可见集、权重、任务或商空间后，原证书不再自动适用。非线性最优误差不是该凸优化的目标，不能把局部信息证书改叫非线性全局最优证书。

## 6. 与匹配剖面估计器的协方差连接

令 proper nuisance `c~N(0,σ²Λ^{-1})` 与 `ε~N(0,σ²I)` 独立，匹配线性联合最小二乘为

\[
\min_{x,c}\|y-Ax-Bc\|^2+c^T\Lambda c.
\]

剖面解等于匹配边缘 GLS，因此在固定商空间中

\[
\operatorname{Cov}(\hat x-x)=\sigma^2QF_Q^{-1}Q^T.
\]

这一恒等式要求**噪声和 nuisance 都按相应系综重抽**。如果固定 c、只重抽传感器噪声，剖面估计的 frequentist 协方差一般为夹心矩阵而非 `σ²F^{-1}`，本实验不混用。

非线性测试在上述单位球图内同时估计 `logρ`、法向、灯强和灯向；未使用历史固定法向 `joint_map`。在 **rank(A)=3P（或已限制到正定的固定场景商空间）** 与 proper 先验共同成立时，名义点增广 Hessian 才正定。再加平滑且在同一亮区的前提，小噪声局部解的一阶影响函数为上述 GLS；这说明局部渐近对应，不声称任意扰动半径的精确恒等式。

## 7. 已执行的独立验证

预设配置：`configs/joint_information_extension_20260918.json`；原始随机种子、样本量、方差比接受区间均在运行前写定。结果为 `results/theory_extension_20260918/joint_information_extension.json`。

- `dimension_and_gauge`：六像素、十灯；内禀场景十八维。完全 flat 时可辨识九维，proper 先验时十八维。九个生成元残差不超过 `6.67e-16`；用现有 gauge 响应对拍误差 `1.51e-14`。
- `dimension_and_gauge.jacobian_max_abs_error`：中央差分最大绝对误差 `1.685e-10`。
- `rows[0/1].linear`：精度倍率一与十，各两万次匹配线性重抽。全部坐标方差比在 **[0.97527,1.01051]**。
- `rows[0/1].nonlinear`：各一千次完整局部非线性剖面拟合，全部样本保留，全部优化器报告成功，无阴影翻转；全部坐标方差比在 **[0.89094,1.10484]**，在预设 **[0.8,1.25]** 内。成功状态不是严格数值最优证书，另报 `max_optimality` 和 `max_nfev`。
- `rows[*].tasks`：三类任务的块低秩与稠密风险相对差异不超过 `1.25e-15`。
- `ceiling.task_values`：全场景、log-albedo、法向切空间任务的价值分别约 `0.24016,0.26883,0.23643`，均低于九成上限；这是此合成场景的例值，不外推为实物平均。
- `certificate`：枚举四十五个“两灯精化”离散点，全部不低于 Frank–Wolfe 切线下界；最小余量约 `0.76861`。
- `tests/test_joint_information_extension.py`：九项测试通过，另覆盖平行灯的秩退化、固定核、凹性、任务梯度、共享 Gram 与协方差恒等式。

**完成边界：** 模型与结构命题、小噪声独立合成协方差验证已完成；尚未对真实对象做联合信息分配及法向表现比较，未用本节结果宣称已有任何非线性重建优势。
