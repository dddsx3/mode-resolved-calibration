# M14（预留）：gauge 两项近似的严格余项与可计算证书

状态：**experimental；complete_scoped**。这是独立研究扩展，不改变已发布方法或历史结果。完整解析证明在同目录 `section_two_term_remainder.tex`，该文件是可直接 `\input` 的正文 section，仅要求主文档加载 amsmath、amssymb、amsthm，不含预导言或本地绝对路径。M14 注册与主文整合由协调代理负责。

## 1. 交付范围及关键结论

精确 gauge **不**自动推出小误差。对固定白化线性模型、满列秩场景 Jacobian、正定 nuisance 协方差及单位任务方向，可以证明

\[
J_a(t)=\underbrace{\frac1{tq}+r}_{j_t}
       +\underbrace{\delta_D+e^T(tI+S_\perp)^{-1}e}_{\mathcal R_t\ge0}.
\]

- `δ_D` 是空间信息非均匀性余项；`e` 是先验加权的 gauge 残差。两个量不因 `Aa=Bc` 自动消失。
- 固定二倍谱箱给出 **不读取实际风险或实际误差** 的端点界 `L_t ≤ R_t ≤ U_t`，并证明 `t≥1` 时 `R_t ≤ U_t ≤ 2R_t`。
- 两端点共同的谱权重给出 `V−Vpred` 的有符号区间，不把两端点余项当作独立量。
- **没有**普遍的 `V` 界/实测 ≤10 定理：非零端点余项可在 `V` 中精确抵消，分母为零或任意小。本文给出可实现模型及一阶反例族；≤10 对 `V` 只是本次全量数值验收结论。
- direction-only 失效既有完整解析 Lambertian 例子，也在 C9 原有最坏点被设计侧区间明确排除 0.05 精度；不是只在高 corruption 时给出 trivial 小误差。

本研究实现按 `CONTRIBUTING.md` 的隔离要求放在 `experiments/theory_remainder.py`，**没有新增或修改 `src/calibinfo` 模块**。

## 2. 定义与完整证明索引

### 2.1 模型与坐标

令 `A∈R^(n×p)` 满列秩、`B∈R^(n×m)`，`Σ0≻0`，并令

\[
D=A^TA,\quad U=A^TB,\quad M=B^TB,\quad \Lambda_0=\Sigma_0^{-1},
\qquad F(t)=D-U(M+t\Lambda_0)^{-1}U^T.
\]

`a` 是 **Euclidean unit** 任务向量，`Aa=Bc` 为指定 gauge。实数据严格沿用现有 Jacobian：`A_k=diag(sqrt(w_k)*s_k)`、`B_phi[k,:,0]=rho*s_k`，所以 `a=rho/||rho||`、每个 active 灯的 `c_k=e0/||rho||`。这里不把 `A` 偷换为 log-albedo Jacobian。

\[
q=c^T\Lambda_0c>0,\qquad r=(a^TDa)^{-1}=\|Aa\|^{-2}.
\]

`c` 若不唯一，本定理仍针对**指定** `c`；不偷偷重选成最小 prior-norm gauge。实际实验取上述强度 gauge。

### 2.2 观测投影分解（TeX Proposition M14.1）

\[
P_A=AD^{-1}A^T,\quad T=B^T(I-P_A)B=M-U^TD^{-1}U,\quad g=U^TD^{-1}a.
\]

`T≽0`、`Tc=0`、`c^Tg=1`。取任意 `Σ0=CC^T`（实现使用 Cholesky），定义

\[
h=C^{-1}c/\sqrt q,\quad S=C^TTC,\quad v=C^Tg,\quad
e_0=v-h/\sqrt q=C^Tg-C^{-1}c/q.
\]

`||h||=1`、`Sh=0`、`h^Te0=0`。`Q` 为 `h⊥` 的正交基，`S_perp=Q^TSQ`、`e=Q^Te0`。仅去掉指定 gauge，**保留其他零模式**。定义

\[
\delta_D=a^TD^{-1}a-r.
\]

完整证明先对正定联合二次型 `||Ax+By||²+t y^TΛ0y` 作块消元，得到

\[
F(t)^{-1}=D^{-1}+D^{-1}U(t\Lambda_0+T)^{-1}U^TD^{-1},
\]

再利用 `h⊕h⊥` 的不变分解抽出 `1/(tq)`；Cauchy–Schwarz 给 `δ_D≥0`。TeX 逐步证明这些恒等式及等号条件：`J=j` 当且仅当 `Da=(a^TDa)a` 且 `g=Λ0c/q`。

两个可测的正和表达式为

\[
\delta_D=\frac{a^T(D-\mu I)D^{-1}(D-\mu I)a}{\mu^2},\quad \mu=a^TDa,
\]

\[
\|e\|^2=\left(g-\Lambda_0c/q\right)^T\Sigma_0\left(g-\Lambda_0c/q\right)
       =g^T\Sigma_0g-1/q.
\]

实现使用前述正和/残差范数，不用两个近等量相减计算主余项。对当前对角 `D`，`δ_D=Σ_i a_i²(d_i−μ)²/(μ²d_i)`。

标量 Schur 有效信息也得到精确展开。若 `s_a=1/J_a`、`s_pred=1/j_t`，则

\[
s_{\mathrm{pred}}-s_a=\mathcal R_t/[j_t(j_t+\mathcal R_t)].
\]

因此可以控制任务方向在消元后的信息误差；**不声称**这一标量结果控制全矩阵 `F` 的每个模式。

### 2.3 非循环端点界（TeX Proposition M14.2）

`S_perp` 的特征值 `λ_i≥0`、谱权重 `w_i=(z_i^Te)²` 给精确余项 `δ_D+Σ_i w_i/(t+λ_i)`。证书不直接把此和当上界，而把谱压缩为固定箱

\[
[\ell_b,u_b]=[2^b-1,2^{b+1}-1],\qquad b=\lfloor\log_2(1+\lambda_i)\rfloor,
\quad m_b=\sum_{i\in b}w_i.
\]

证书只接受 `q,r,δ_D,(ell_b,u_b,m_b)`，其上下界为

\[
L_t=\delta_D+\sum_b\frac{m_b}{t+u_b},\qquad
U_t=\delta_D+\sum_b\frac{m_b}{t+\ell_b}.
\]

逐项单调性给 enclosure；`u_b=2 ell_b+1` 及 `t≥1` 给 `(t+λ_i)/(t+ell_b)≤2`，所以 `U_t≤2R_t`。这是一条模型无关的**解析证明**，不是用真实 44 cells 推断的规律。单测还固定谱箱/权重，仅移动箱内真特征值，验证实际误差改变而证书不变。公共 API 也接受一般谱箱，此时只有每个正质量箱的 `(t+upper)/(t+lower)≤2` 才报告二倍保证；对 `[0,100]` 等一般箱，单独 `t≥1` 不充分。

该证书需要 nuisance 大小的谱分解，计算量不一定小于精确 low-rank risk。其价值是显式、可审计的充分条件和误差 enclosure；不声称算法加速。

### 2.4 联合端点的 `V` 界（TeX Proposition M14.3）

\[
\beta=j_\kappa/j_1=(1/\kappa+qr)/(1+qr),\quad V_{\rm pred}=1-\beta,
\]

\[
V-V_{\rm pred}=-N/(j_1+\mathcal R_1),\quad
N=(1-\beta)\delta_D+\sum_iw_i f(\lambda_i),
\]

\[
f(x)=\frac1{\kappa+x}-\frac\beta{1+x}
    =\frac{(1-\beta)(x-\kappa qr)}{(1+x)(\kappa+x)}.
\]

每箱内 `f` 的最小值在端点，最大值在端点或唯一极大点

\[
x_*=(\kappa\sqrt\beta-1)/(1-\sqrt\beta).
\]

取箱内核函数极值和正权重得到 `[N_-,N_+]`；分母在正区间 `[j1+L1,j1+U1]`。四个角点的商给出有符号 correction 区间及绝对界 `B_V`。TeX 完整证明导数符号、极值和区间除法。

**失效边界不是仅 upper bound 变大**：若 correction 全区间高于 `ε` 或低于 `−ε`，则证书严格排除误差 ≤`ε`。如果区间跨过该范围，则明确为 inconclusive。

反过来，`q=r=1,δ=0,S_perp=10,e=1,κ=10` 时端点余项非零而 `V−Vpred=0`。可实现于 `A=(1,0)^T`、`B=((1,1),(0,sqrt(10)))`、`Σ0=I`、`a=1,c=(1,0)^T`。改变 `10` 为 `10+u` 给 `V−Vpred=−9u/9200+O(u²)`，谱箱 `[7,15]` 的界仍非零，故不存在普遍的 value 相对紧度常数。

## 3. 可测充分条件与物理参数

以下条件全部使用名义场景/先验，不使用实际误差。

- `U_t≤η j_t ⇒ 0≤(J−j_t)/j_t≤η`；相对真实 `J` 的界为 `U_t/(j_t+U_t)`。
- 若 `S_perp≽λ_* I, λ_*≥0`，则 `R_t≤δ_D+||e||²/(t+λ_*)`。`λ_*=0` 总合法，但忽略谱结构可很松。
- 若任务支持上 `|d_i/μ−1|≤ξ<1`，则 `δ_D≤r ξ²/(1−ξ)`。它与前项相加得到物理充分条件。实场景不保证 `ξ<1`；不满足时仍可使用精确正和及谱箱。
- `B_V≤ε` 认证 value 绝对误差。与 C9 一致，0.05 用作已有精度诊断，不用于调整谱箱或 tightness 判据。

记 `Rrho=||rho||`、active 灯数为 `L`，定义光照 leverage

\[
\ell_k=\sum_p a_p^2 w_{kp}s_{kp}^2/d_p,\quad \sum_k\ell_k=1,
\quad g_{k0}=R_\rho\ell_k.
\]

未耦合的逐灯标准差 `s_Ik`（logI）、`s_Dk`（弧度/切向轴）给

\[
\pi_k=\frac{s_{Ik}^{-2}}{\sum_j s_{Ij}^{-2}},\quad
q=\frac{\sum_ks_{Ik}^{-2}}{R_\rho^2},\quad
\|e\|^2=R_\rho^2\sum_ks_{Ik}^2(\ell_k-\pi_k)^2
       +\sum_ks_{Dk}^2(g_{k1}^2+g_{k2}^2).
\]

这说明仅把方向标准差调小，并不能消除光照 leverage 与异质先验的错配。均匀强度先验时 `π_k=1/L`。阴影、法向切向响应、权重和纹理通过 `ell,g,D,T` 进入，不可忽略。

C9 参数族保持原定义：`s_Ik=sI exp(het_sigma*z_k)`、`s_Dk=radians(sig_dir_deg) exp(het_sigma*z_k)`，固定 `family_seed=20260915`，**先生成全部142灯再取 active 索引**。通道相关块的两个交叉项为 `rho_c*sI_k*sD_k/sqrt(2)`。因此

\[
q=\frac1{R_\rho^2}\sum_k[s_{Ik}^2(1-\rho_c^2)]^{-1},\quad
\|e\|^2=\sum_k(g_k-\Sigma_k^{-1}c_k/q)^T\Sigma_k(g_k-\Sigma_k^{-1}c_k/q).
\]

完整物理翻译、相关矩阵的 `1±|rho_c|` 谱界及证明见 TeX。实现记录 `rows[].physical` 中的物理 q、正和残差、leverage 及 degree/logI 分布。

**角度单位**：`CorruptionFamily` 的 `sig_dir_deg` 是两个切向坐标各自的标准差。其一阶二轴 angular RMS 为 `sqrt(2)*sig_dir_deg`（异质时逐灯乘 multiplier）。不把它偷换成其他非线性采样器的径向旋转 RMS；产物同时记录 per-axis 和 two-axis RMS。

## 4. direction-only 的解析反例与奇异极限纪律

正定控制 `sI→0`、方向标准差固定时，`q→∞`、`j_t→r`、`Vpred→0`，但 `e` 的方向项不必消失，并可随 `t` 保留显著风险增益。

严格 `sI=0` 下的完整 `Σ0` 是奇异的，此时 **不能将 `pinv(Σ0)` 当成 precision 来算 q**：零方差表示已知量/无限精度，不是零精度。应取 `sI↓0` 的正定极限，或移除已知坐标后重建模型；后者不再自动具有原 intensity gauge。本次 C9 控制完全沿用其 finite-PD `sig_logI=1e-6`，不假装测试了精确零方差。

合法 Lambertian 单灯单像素反例：

- `rho=1, w=10000`；`n=(4/5,0,3/5), d=(0,0,1)`，切向基 `(1,0,0),(0,1,0)`，无遮挡。
- `A=[60], B=[60,80,0], a=1, c=(1,0,0)`，精确 `Aa=Bc`。
- `Σ0=diag(epsilon²,1/10000,1/10000)`，方向 per-axis 标准差 `0.01 rad`。
- 精确 `J=1/3600+epsilon²/t+4/(22500t)`，而 `j=1/3600+epsilon²/t`。
- `κ=10,epsilon↓0` 时 `V→72/205` 而 `Vpred→0`。

恒等式与极限由解析推导和 SymPy 有理检查分别给出。固定小角度但提高正噪声精度 `w`，风险相对余项极限为 `w*u²*sD²`（`u` 是切向响应），可任意大。因此没有只凭 gauge 或 degree/logI 比的几何无关小误差定理。

所有通道同时关闭也不保证 `J≈r`：其极限为 `a^TD^-1 a=r+δ_D`；空间非均匀项仍存在。

## 5. 真实数据数值验收（不是证明）

产物：`results/theory_extension_20260918/two_term_remainder.json`。

原始文件来自 `D:/data/OpenIllumination/OLAT` 与 `D:/data/OpenIllumination_meta`。runner 参数 `--data-root` 应给 **`D:/data/OpenIllumination`**，因为历史 loader 会自行附加 `/OLAT`。使用原 loader、48个 active 灯、1200 像素、`default_rng([20260910,obj_idx])`、corrected noise-fit，不运行历史脚本的写产物入口。

覆盖 44 nominal cells 两个端点，再加完整 45点×11对象 C9 网格；共 **539条记录、1078个端点记录，包含11条重复anchor，实际528个唯一 object-parameter conditions**。全量 C9 不是为满足验收而选择的几个有利设定；其中包括全部 88 个 direction-only cells。重复来自 nominal level=0.5 与 C9 anchor 的相同场景/先验，不能把合并计数当作独立样本数。计数和键定义在 `protocol.overlap_scope`。C9 tags 也有重叠，不能相加计算唯一 cells。

### 5.1 界/实测比分布

全部比值使用未舍入的实际误差。零误差比值应记 `null`，不加 denominator floor；本次没有零误差。

| 集合与量 | 数量 | min | median | p95 | max | ≤10 |
|---|---:|---:|---:|---:|---:|---:|
| nominal 风险端点 | 88 | 1.114744 | 1.353238 | 1.546959 | 1.652930 | 88/88 |
| nominal value | 44 | 1.155955 | 1.298290 | 1.435463 | 1.483561 | 44/44 |
| C9 风险端点 | 990 | 1.000000 | 1.317729 | 1.535014 | 1.724254 | 990/990 |
| C9 value | 495 | 1.000002 | 1.317539 | 2.480009 | 4.146191 | 495/495 |

字段分别为 `summary.{nominal,family}.{endpoint_bound_over_observed,value_bound_over_observed}.{min,median,p95,max}`；计数为 `*_tightness_le_10_count`。所有端点和 value 区间均无违反，对应 `*_bound_violations=0`。

最难的 nominal level=0.1 单独报告，避免仅依赖高 corruption 端：

- 22端点比值 median=1.292228，max=1.454136；全部 ≤10。
- 11个 value 比值 median=1.224734，max=1.305742；全部 ≤10。
- value 误差 max=0.0385394150，最大证书上界=0.0451016409。
- 风险端点相对余项本身并非都小，`t=κ` 时实测最大约 0.4951；“界紧”不应写成“近似普遍高精度”。

字段前缀：`summary.nominal_by_level["0.1"]`。相对实测余项可由 `rows[].endpoints.tkappa.observed_remainder / prediction` 原样计算。

异质/耦合验证沿用 C9 tags：

| tag | cells | 风险比 max | value 比 median / max |
|---|---:|---:|---:|
| `het_sweep` | 33 | 1.602804 | 1.310244 / 1.558183 |
| `rho_sweep` | 22 | 1.577073 | 1.319491 / 2.207632 |
| `tier2` | 55 | 1.724254 | 1.386295 / 2.808454 |
| `dir_only` | 88 | 1.562689 | 2.049264 / 3.057997 |

字段：`summary.family_by_tag[tag]` 的相同 distribution 子字典。

### 5.2 direction-only 机制是否被界看见

最坏点 `family/obj_19_cylinder/35` 使用 `sig_logI=1e-6, sig_dir_deg=0.1, het_sigma=0, rho_c=0`：

- `Vpred=3.7540780993e-7`，实际 `V=0.4641371108`。
- 实际 signed correction `0.4641367354`。
- **不读取此 actual error 的证书区间** `[0.2566364348,0.9046182854]`。
- `B_V/observed=1.949034`；下界本身就超过 0.05，明确排除小误差。
- `t=1` 风险余项/预测的上界约 6.835，`t=κ` 约 2.230，方向通道的遗漏并未被 gauge 对齐掩盖。

精确字段：`summary.direction_only_worst.value.{prediction,observed,observed_correction,correction_lower,correction_upper,bound_over_observed,certificate_rules_out_0_05_accuracy}`，风险字段在该字典同级 `endpoints`。

88个 direction-only cells 中实际误差 >0.05 的有42个，证书能严格排除0.05精度的有32个；其余10个是 **inconclusive**，不谎称全部检测。全 C9 家族相应计数为83与61。字段为 `summary.family_by_tag.dir_only` 或 `summary.family` 下的 `value_error_exceeds_0_05_count` 与 `value_certificate_rules_out_0_05_accuracy_count`。

### 5.3 冻结、数值精度与局限

- 75份历史结果读取前后逐文件哈希相同，总哈希均为 `85c33124fab53bcd55dd463f7e9ce4fe81715b1941649301e8dffac532a92e23`。
- nominal endpoints 最大 frozen replay 相对差 `4.8141e-10`，字段 `summary.nominal.nominal_frozen_replay_relative_error.max`。
- 精确余项公式与独立 low-rank 风险最大相对差 `5.7167e-9`，字段 `summary.overall.identity_risk_relative_error.max`。
- 25个投影 Gram 小负特征值按明确数值守卫记录为 roundoff zero；最大绝对值 `2.8251e-11`、最大相对量 `7.4060e-18`。每 cell 的 magnitude、count、scale 位于 `rows[].diagnostics`。较大 PSD 违例直接报错，绝不添加 ridge。
- prior PSD 守卫采用母 Gram 误差经合同传播的尺度：`T_hat≽−τ||M||_F I ⇒ Q^TC^TT_hat CQ≽−τ||M||_F λmax(Σ0)I`。不能改用可能趋于零的 `||C^TTC||` 作相对守卫，否则合法 `B=AW,T=0` 会因 Gram 相减 roundoff 被误拒绝。两种实际尺度及接受阈值分别记录为 `spectrum_scale`、`spectrum_parent_gram_propagated_scale`、`spectrum_psd_guard_tolerance`，并有随机和确定性 exact-T0 回归。
- 严格证明属于实数精确算术。浮点执行 **不是 directed-rounding interval proof**；预设 check tolerance `2e-11+2e-7|J|` 只用于数值一致性，既不修改证书，也不放宽 ≤10 紧度阈值。
- 尚无严格 floating-point enclosure；也不证明非线性/重建端点结论、跨队列排序或几何无关的 angle/intensity 比界。这些属于具体理论边界，不是缺数据。所需真实数据齐备，本次没有未完成的数据验收。

`family_e_diag.py` 的 `S_pred/S_real` 定义已核对：它们是策略预测排序与真实重建排序的跨臂敏感性。本任务不把这些排序量重新命名为风险余项指标，也不据本结果重述真值端脱钩结论。

## 6. 实现契约、复现与测试

### 6.1 文件

新增且仅新增以下六个目标文件（实现位置依协调代理指示从 `src` 改为 `experiments`）：

1. `experiments/theory_remainder.py`：纯设计侧数学模块。
2. `experiments/two_term_remainder.py`：可复现实验入口，输出路径固定为本次唯一授权 JSON。
3. `tests/test_two_term_remainder.py`：数学/契约/冻结回归测试。
4. `docs/theory/two_term_remainder.md`：本说明。
5. `docs/theory/section_two_term_remainder.tex`：独立可 input 正文及完整证明。
6. `results/theory_extension_20260918/two_term_remainder.json`：本次全部输入指纹与数值证据。

无修改旧 methods、历史脚本、既有 results、README、claims、CHANGELOG；这些集成文件由主代理独立处理。没有提交或推送操作。

### 6.2 Python API

- `prepare_geometry(finf_diag, cross, nuisance_gram, a, c) -> GaugeGeometry`：对角 `D` 的 `T,g,r,δ_D`；检查 unit、Gram gauge、PSD，拒绝静默修复无效模型。
- `analyze_prior(geometry, sigma0) -> GaugeSpectrum`：全正定 prior 白化，删除指定 gauge，保留附加零模式；输出残差谱及 roundoff diagnostics。
- `compress_spectrum(spectrum) -> BinnedRemainder`：固定 dyadic interval + 非负 mass，丢弃箱内精确特征值。
- `endpoint_certificate(binned, t)`：只接受箱摘要，返回余项上下界、风险区间、相对上界和粗残差界。
- `value_certificate(binned, kappa)`：同一箱摘要对应的 signed correction/value interval、absolute bound、kernel diagnostics。
- `GaugeSpectrum.remainder_for_validation(t)`：**明确只作恒等式对拍**，不被证书函数调用。

### 6.3 结果字典字段

顶层包括 `gate, reserved_claim, analysis_status, published_status, mathematics, protocol, acceptance, summary, scenes, rows, manifest`。

每个 `rows[]` 包括：

- 身份与参数：`cell_id, cohort, object, parameters, tags`。
- 分解：`q, r, qr, delta_diagonal, diagnostics, physical`。
- 可独立重算证书的全部输入：`spectral_bins[].{lower,upper,mass,multiplicity}`，无需读取真误差。
- `endpoints.{t1,tkappa}`：`prediction, remainder_lower, remainder_upper, risk_lower, risk_upper, relative_to_prediction_upper, relative_to_actual_upper, observed_risk, frozen_risk, recomputed_risk, observed_remainder, observed_absolute_error, identity_remainder_validation_only, identity_risk_relative_error, frozen_replay_relative_error, bound_over_observed, bounds_hold, tightness_le_10` 等。
- `value`：`prediction, correction_lower, correction_upper, absolute_error_upper, value_lower, value_upper, numerator_lower, numerator_upper, denominator_lower, denominator_upper, observed, recomputed, observed_correction, observed_absolute_error, bound_over_observed, bounds_hold, tightness_le_10, certificate_rules_out_0_05_accuracy` 等。
- `source_fields`：nominal 使用冻结 JSON 原始风险字段；family 的冻结文件只保存 value，两个 risk 端点为新增 independent lowrank evaluation。

汇总 `summary.{overall,nominal,family,nominal_by_level,family_by_tag,direction_only_worst}` 保存比例分布、精度/失效计数和 replay error。`scenes[object]` 保存逐PNG哈希、灯位文件哈希、48灯/1200像素索引和 geometry diagnostics。

`manifest` 保存 `frozen_results_sha256`（75份逐文件）、`frozen_tree_sha256_before/after`、`changed_frozen_paths`（空）、`source_sha256`、软件版本、路径、种子及 reproduction command。`acceptance` 区分已完成验收与 `not_claimed` 的理论边界。

### 6.4 复现命令

可从任何工作目录运行，不需要修改历史 YAML。

```text
python -B C:/Users/35702/publication/mode-resolved-calibration/experiments/two_term_remainder.py --data-root D:/data/OpenIllumination --data-meta D:/data/OpenIllumination_meta
```

入口对75份冻结结果的任务起点总哈希作 fail-fast 检查，写文件前再验一次；唯一写入为新增 `two_term_remainder.json`。运行计时位于 `manifest.elapsed_s`，不将其当作性能比较。原有数据和结果完全只读。

快速数学/产物测试（不会重建大数据场景）：

```text
python -B -m pytest -o addopts='' -q -p no:cacheprovider C:/Users/35702/publication/mode-resolved-calibration/tests/test_two_term_remainder.py
```

测试覆盖独立 dense Fisher 恒等式对拍、SymPy精确物理反例、先验换坐标、异质耦合 q、附加零模式、exact-T0数值尺度、一般谱箱flag、箱内移动非循环性、核函数内部极值、非法输入、44 frozen cells全端点、完整C9网格、纯箱重算证书及75份历史文件哈希。

相关回归范围还包括 `test_goal_oriented.py`、`test_lowrank_identity.py`、`test_corruption_family_sensitivity.py`、`test_corruption_family_degeneracy.py`。本说明不固化测试数量；最终通过数以主代理统一验收日志为准。不将随机数值测试当成数学证明。TeX 静态检查已确认本地交叉引用完整、label无重名、无预导言/绝对路径；研究补充PDF的合并编译由主代理统一进行。
