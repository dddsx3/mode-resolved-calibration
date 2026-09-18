# 匹配系综与信息风险到重建排序的条件桥接

状态：experimental — NOT part of the published results。C13（匹配系综）与 M16（排序条件）仅预留，由主代理整合注册编号。

## 文件与范围

- 核心 `experiments/theory_ensemble.py`，入口 `experiments/matched_ensemble_bridge.py`，测试 `tests/test_ensemble_bridge.py`。依 CONTRIBUTING 的实验隔离约定，不新增 `src` 模块。
- 可直接 input 的中文正文：`docs/theory/section_ensemble_bridge.tex`，无导言、无 document 环境。
- 首跑前固定配置：`configs/matched_ensemble_bridge_20260918.json`，SHA256 `bfb61839b002c02a896a64be5cebca03098db2c38aa0e682cb5bb0f8fb46368f`。
- 新产物仅在 `results/theory_extension_20260918/`；不修改历史 results、ZIP 幅值快照、README、claims、CHANGELOG 或 gamma 接口。
- S1/S3 即便使用真实图像得到的名义几何，生成的观测仍然是 **synthetic**。真正的固定影像诊断与这些合成系综分开标记。

## 1. 只读研究结论

### Monte Carlo 与名义 real pipeline 不是同一估计器

`experiments/monte_carlo_validation.py` 的 joint ensemble 同时抽样 nuisance 与测量噪声，用匹配边缘 GLS；fixed-delta-c 控制只有条件方差，不能替代边缘协方差恒等式。

`experiments/openillumination_validation.py:NominalScene` 使用固定图像估计名义 rho、n、noise weights；实际 Jacobian `A_k=diag(s_k)` 对应 **线性 albedo 增量**。B_phi 的 log 参数是灯强度，不是场景 log-albedo。固定 n 的模型不包含法向联合信息；`joint_map.py` 也是固定 n_gt 的 rho+灯联合，不是法向联合。

历史幅值 numerator 是固定真实图像上对假设灯光作 estimator-side corruption，然后 plug-in 逐像素重估；denominator 是展平原始残差池的 iid 重抽样。二者不是匹配 marginal GLS + ideal GLS 的噪声系综。

ZIP `amplitude_rerun_20260917` 的四臂实现已经修正 per-seed gauge，并保存两侧投影的 matched linear prediction，但仍明确只匹配坐标，没有匹配残差分母。只读来源为：

- ZIP `amplitude_rerun_20260917/amplitude_comparison.json:variants.D.new_ratio_matched_prediction.pooled.median = 53.74419400160423`。
- 同文件 `variants.D.new_ratio_original_prediction.pooled.median = 16.768833829304768`。
- 同文件 `interpretation` 明确两种比值不能单独验证或否证完整匹配定理。

还需独立指出一个具体不匹配：历史 `rotate_dirs` / `apply_scaled_corruption` 以随机切向轴和一个高斯角度旋转；小角度位移在切向平面均匀转向，每个切向分量方差为 `sigma_rad²/2`，而 `sigma_phi_diag` 宣告的是每个切向分量 `sigma_rad²`。本任务保留历史函数不动，S1/S3 使用真正按每轴 covariance 生成的 tangent Gaussian。

### 端点不能默认为同一个 H

沿用 `woodbury_quad_risk` 的任务**算子**约定 `J_H=tr(H DeltaF^-1 H.T)`：

| 端点 | H/额外条件 | 能否直接用原全迹排序 |
|---|---|---|
| 全参数平方误差 | H=I | 匹配无偏线性时可以用于期望 |
| gauge-aligned albedo MSE | H=P_rho/sqrt(P) | 不同于 H=I，需重算 J_H |
| 固定模式能量 | H=W.T P_rho，按具体均值规范缩放 | 不同于前两者，跨方案必须固定 |
| mean angular drift | 原 pipeline 一步 normal refit 后平均角度 | 不是固定 n 的 albedo quadratic risk；不能令 H=I 代替 |
| 真正 GT 法向误差 | 与 nominal drift 的 reference 也不同 | 另需 GT/端点/联合参数化模型 |

只读历史端点分歧的准确字段是 `results/openillumination/decision_quality.json:spearman_pred_vs_realized`（ang median +0.916667，MSE median -0.575），不可混成 `spearman_informed_only` 的同名端点。

`family_e_diag` 的 `s_pred/s_real/fit_anc/fit_het/cross_ah/cross_ha` 复用为描述性分解；原脚本“NOT sigma misspecification”的强因果推断不作为本定理的一部分。预测 rank 不变不能排除预测数值偏差、误差几何错误或端点错配。

## 2. C13：严格匹配恒等式

设 `y=A x+B phi+eps`，`Cov(eps)=sigma² Sigma`、`Cov(phi)=Sigma_phi`，独立且均值零。`Sigma` 正定，`Sigma_phi` 为 proper PSD，采用支持因子 `L_phi L_phi.T`。

只对白化 Sigma，令 `Aw=Wy A, Bw=Wy B, C=Bw L_phi/sigma`，`M=(I+C C.T)^-1`，`DeltaF=Aw.T M Aw`，`F_inf=Aw.T Aw`。固定可辨识空间上两者正定。

匹配 GLS `K=DeltaF^-1 Aw.T M Wy` 给出 `KA=I` 和 `Cov(Ky-x)=sigma² DeltaF^-1`；零 nuisance 的 ideal GLS 给出 `sigma² F_inf^-1`。证明在 tex 中完整给出。

**单位纪律**：满秩 covariance 对应 `Lambda=sigma² inverse(Sigma_phi)`，不是遗漏 sigma² 的精度。若 covariance 奇异，零方差=硬支撑约束，不可取 `sigma² pinv(Sigma_phi)` 后当无约束 profiling。已知答案 A=1、B=[1,1]、Sigma_phi=diag(1,0) 给正确 DeltaF=1/2，伪精度却为0。实现支持全零 covariance 和任意小物理量级 covariance，不用 `max(scale,1)` 删掉小方差。

以 `P=P_rho`，`W=F_inf^(1/2) U`，`q_j=P W_j`，matched prediction 是 `v_j/b_j`，其中 `v_j=q_j.T sigma² DeltaF^-1 q_j`、`b_j=q_j.T sigma² F_inf^-1 q_j`。P 必须同时作用预测 covariance 两侧和每个样本。投影后一般不是 `1/retention_j`。零模式报告 invalid，不截分母。

## 3. S1 实际运行：已 PASS

**首跑前** 固定 seed、8192 repeats、batch256 和全部 mode gates `[0.8,1.25]`，没有按结果调整。配置 mtime UTC 2026-09-17T18:27:32.892178；首跑 UTC 2026-09-17T18:39:21.219116（本地日期9月18日）。

对象 OI dolphin、ball，各24/142灯、96 masked pixels，无排除；levels .1/.35；另三组 dense controls 覆盖非对角测量 Sigma、奇异/零 Sigma_phi 和 sigma .35/1.7/.6。所有观测重新合成并通过 GLS 增益，不是直接从预测 covariance 抽取误差。

产物 `matched_ensemble_bridge_s1.json` 的字段：

| 字段 | 结果 |
|---|---:|
| `n_cases` / `n_mode_checks` | 7 / 35 |
| `all_pass` | true |
| `numerator_ratio_summary.minimum/median/maximum` | 0.9732858129 / 1.0031776302 / 1.0239557683 |
| `baseline_ratio_summary.minimum/maximum` | 0.9694119793 / 1.0373898366 |
| `degradation_ratio_summary.minimum/median/maximum` | 0.9511950650 / 1.0025943178 / 1.0562648184 |
| `maximum_numerical_identity_error` | 9.4040903334e-13 |

这证明的是**在匹配生成、估计器、理想基线、投影、坐标同时成立时，本实现闭合**；不把它叫作真实重建优势。

## 4. S2/S3 实际运行与精确乘法归因

新 checkpoint `matched_ensemble_bridge_nested.json` 完成，所有 S1 replay variance / S3 paired linear samples 核对通过。

- S1 主比值 `R1=vhat_linear/v_pred`，用解析理想 denominator；另独立抽 ideal baseline 只是检查。
- S2 只替换 denominator。原始残差池不中心化，不做 df 修正，pool population variance 是 `ddof=0`。条件 covariance 为 `tau² K0 K0.T`，所以 `bboot_pop_j=tau² ||K0.T q_j||²`。
- 精确 `R2=R1*(b_ideal/bboot_pop)*(bboot_pop/bboot_emp)`，分离残差池机制与有限重抽样波动。
- S3 只以有限 Lambertian forward delta 替 B_phi phi，保持同 phi、同 eps、同 fixed marginal GLS、同 modes、同 bootstrap samples。精确 `R3=R2*(vhat_nonlin/vhat_linear)`。
- 可乘的是逐模式 factors 或它们的几何均值，**不是中位数的乘积**。centered variance、empirical mean、second moment 分开。

| case | S2 emp degradation / prediction 范围 | S3/S2 nonlinear factor 范围 |
|---|---:|---:|
| dolphin .1 | .00284669–.0190723 | 1.000583–1.002448 |
| dolphin .35 | .00288791–.0197673 | 1.028304–1.139806 |
| ball .1 | .662407–1.118721 | 1.000052–1.001245 |
| ball .35 | .611935–1.089822 | 1.001855–1.018319 |

字段为 `matched_ensemble_bridge_nested.json:cases[i].factors.s2` 与 `cases[i].factors.s2_to_s3_nonlinearity_factor`，各5个 modes，case_id 明示对象/level。

**范围限制**：此分母效应常常把比值向下推，不能把53.744当作它的普遍结果。S3 与历史固定影像 estimator-side fit 不同构；历史半径的 estimator/endpoint 也不同，level小于历史crossing不是严格半径证书。没有重放历史全66 cells，不能声称解释掉历史幅值。

## 5. M16：期望风险、可测半径、pairwise margin

### 精确期望

对任何有限二阶矩误差，`E ||H e||²=tr(H Cov(e) H.T)+||H E e||²`。matched unbiased linear 时为 `sigma² J_H`。注意 sigma 跨方案变化时排序的对象是 **sigma² J_H**，不是裸 J_H。

### 可测总半径

实际误差 `e_i=e_linear_i+r_i`；预测 covariance C_i，真实线性 covariance Ctilde_i，bias b_i。同一H下用可证上界

- `c_i >= ||H(Ctilde_i-C_i)H.T||_*`；
- `beta_i >= ||H b_i||`；
- `d_i² >= E||H r_i||²`；
- `ell_i >= |E endpoint_i-E||H e_i||²|`；
- `s_i >= |empirical endpoint mean-E endpoint_i|` 在同时概率事件上。

则 `eta_exp_i=c_i+beta_i²+2 sqrt(p_i+c_i+beta_i²) d_i+d_i²+ell_i`，`eta_i=eta_exp_i+s_i`。证明用迹的核范数界与 Cauchy–Schwarz，不要求余项独立。

**可测的含义与字段映射**：

| 项 | 可判定方法 | 产物/实现 |
|---|---|---|
| 输入协方差与 GLS 一致 | 固定生成 covariance、gain，直接 sandwich | `model_metadata`, `covariance_identity`; bridge `matched_conditions` |
| H/坐标/gauge 相同 | 保存算子哈希、固定每cell全部plans | `H_operator_sha256`, `projection.score_sha256` |
| covariance error | 可知 V 的 `K V K.T`、或外部带置信界的 covariance estimate | `covariance_risk_radius`；matched case精确0 |
| bias | 可知均值模型；独立有效区间 | matched零均值由设定；样本mean单独诊断 |
| nonlinear d | 解析矩界/认证Hessian与高斯tail，或有效独立上置信界 | S1精确0；S3只报paired sample RMS，不冒充population bound |
| endpoint ell | 显式same H则0；若G不同需bound | `endpoint_operator_radius`；angle未认证 |
| sampling s | 零均值高斯二次型的已知covariance集中界 | `gaussian_quadratic_mean_radius` |
| pairwise margin | `gap > radius_i+radius_j` | `certificates.*.pairs[*].slack` |

零均值高斯误差下 `T=H C H.T`，`t=log(2K/alpha)`，
`s=2||T||_F sqrt(t/N)+2||T||op t/N` 给K个risk同时至少1-alpha事件，方案之间可配对、不要求互相独立。该式针对 **mean squared loss**，不是 ddof=1 sample variance；非高斯real/boot不能直接套。

若只有配对观测残差，样本 Cauchy–Schwarz 给实际 nonlinear empirical risk 与 linear empirical risk 差的上界 `2 sqrt(mean linear loss)*dhat+dhat²`。结合已认证线性高斯 mean 的误差 s，可用 `s+2sqrt(p+s)*dhat+dhat²` 认证非线性**样本均值**；这仍不认证其总体期望。

### 排序与Spearman

`p_j-p_i>eta_i+eta_j` 蕴含实际（或empirical）risk_i<risk_j。所有相邻预测对严格分离足以得到所有对和Spearman=1。只有部分对时，区间 `[p_i-eta_i,p_i+eta_i]` 给每方案可行rank区间 `[l_i,u_i]`；令预测rank r_i，`D_i=max(r_i-l_i,u_i-r_i)`，无ties时有

`Spearman >= max(-1, 1-6 sum(D_i²)/(K(K²-1)))`。

只有这个下界≥预注册rho0才作阈值保证；ties时明确不应用无ties公式；不把没有margin当作“理论失败”。

### 反例

1. `p_i=1+i epsilon`, `true risk_i=1+(K-1-i)epsilon`：同H、无偏、线性、covariance error任意小，但Spearman=-1。缺精确匹配和充分margin；有理数写出误差，epsilon取预注册三值。
2. 完全matched Gaussian但finite N：各平方均值在正半轴有正密度，任意逆序open set概率非零；近ties完全逆序概率趋1/K!。不能无margin保证经验Spearman阈值。
3. 两计划 covariance diag(1,4)、diag(2,1)，H=[1,0]风险1<2，但G=[0,1]风险4>1。仅端点一致性失败，即可逆序，噪声匹配并不挽救错误H。

## 6. 独立验证：已完成，不放大适用范围

DiLiGenT ballPNG/bearPNG，各16/96灯、64像素，从原 canonical loader 加载。全部对象/selection由首跑前config固定，无排除或重采样。GT只用于绝对sanity，逐灯 `gray / light_intensities[R]` 与底层像素直接核对的最大误差均为0。名义 GT normal mean error 分别 3.12703°、7.19969°，不与实际诊断的 nominal angular drift 混淆。

四档全灯precision ladder与四个等预算4/16活跃灯子集，anchor/het两family。het固定标准差轮廓，非按结果拟合。每plan4096 matched synthetic，另256 fixed-real-image历史arm_metrics；32种plan risks各跑相应样本。无universe-random comparison，不声称优于random。全灯ladder对比不同精度，不是同预算灯选择优势实验。

主结果 `matched_ensemble_bridge.json:bridge`：

| 字段 | 结果 |
|---|---:|
| `n_gaussian_plan_risks` | 32 |
| `gaussian_risk_ratio_summary.minimum/median/maximum` | .9940441722 / .9993871367 / 1.0100414861 |
| `all_gaussian_risks_inside_simultaneous_radius` | true |
| `n_certificate_cells` | 8 |
| `n_target_certified`（rho0=.7） | 2 |
| `n_all_pairs_certified` | 0 |
| `all_certified_pairs_observed_agree` | true |
| `fixed_real_certificate_claim` | false |

每个family×cell的证书与实测rho（`bridge.cells[*].certificates.<family>.gaussian_empirical`）：

| 独立对象/cell | family | 已认证对/6 | Spearman可证下界 | 实测Spearman |
|---|---|---:|---:|---:|
| ball precision ladder | anchor | 4 | .7 | 1 |
| ball precision ladder | het | 5 | .8 | 1 |
| ball equal budget | anchor | 0 | -1 | .4 |
| ball equal budget | het | 0 | -1 | 1 |
| bear precision ladder | anchor | 0 | -1 | 1 |
| bear precision ladder | het | 0 | -1 | 1 |
| bear equal budget | anchor | 0 | -1 | -.4 |
| bear equal budget | het | 0 | -1 | 1 |

“未认证”不是假，实际为1也不允许事后把radius缩小来给证书；实际为-.4也不允许重抽MC改善。只有ball的两条precision ladder下界达到预注册.7。独立 scalar Gaussian 正对照（不冒充数据集迁移）在预注册 risks `[1,2,4,8,16,32]` 上15/15 pairs均认证、实测rho1，见 `counterexamples.independent_separated_synthetic_control.certificate`。

### 原 S_pred/S_real 的独立复用

以下是 **4个object×plan-cell各先算rank再取中位数**，不是合并32个plan散点。定义保持 `family_e_diag`；匹配population row的perfect fit是解析恒等式，并不算第二次独立实验。准确字段为 `bridge.family_e_diag_summary.<endpoint>.<stat>.median`：

| endpoint | S_pred | S_real | fit_anc | fit_het | cross_ah | cross_ha |
|---|---:|---:|---:|---:|---:|---:|
| `matched_population_quadratic_risk` | .9 | .9 | 1 | 1 | .9 | .9 |
| `matched_empirical_quadratic_loss` | .9 | .6 | .7 | 1 | .9 | .6 |
| `fixed_real_mse_aligned` | .9 | .8 | -.8 | -.6 | -.7 | -.7 |
| `fixed_real_ang_mean_deg` | .9 | .9 | .3 | .2 | .5 | .3 |
| `fixed_real_dual_mean` | .9 | .6 | -.8 | .1 | -.1 | -.7 |

实际影像 albedo MSE 虽然使用同 H=P_rho/sqrt(P)，其固定图像 estimator-side plug-in ensemble 不是预测的 marginal GLS；因而该失败不能被读为M16反例，更不能说“已经解决真实重建排序”。角度与dual端点另外列出，不能拿角度正号为aligned MSE背书。

### 未改阈值的数值审计

bear 的旧corrected noise拟合自身带负intercept，历史floor将观测Sigma跨度放大，DeltaF condition约2.35e7。首轮在新GLS与原 `woodbury_quad_risk` 的1e-9 float对拍处停止。**保留5个原float检查false，不放宽1e-9**：从同一冻结物理数组独立用60位mpmath直接组装Schur，证新GLS风险相对误差4.41e-14–7.03e-14，而旧float Woodbury误差1.30e-9–5.28e-9。其余27对原float检查通过。结果在每plan `numerical_audit` 中，新增审计不改变seed、样本、对象或任何历史接口。

### 换条件后的关系范围

| 变更 | 数学关系何时保留 | 本次证据/不能继承的结论 |
|---|---|---|
| 换名义场景 | 固定可辨识空间PD、same H和matching重建后恒等式仍在 | OI两设计与独立DiLiGenT两设计已检验；不能继承旧margin |
| 换Sigma_phi家族 | 支持因子、实际生成与GLS同步改变 | anchor/het风险闭合；S_pred/S_real并不恒等于1 |
| 改残差bootstrap分母 | numerator恒等式不变，degradation多乘已算因子 | S2明确向上/向下皆可；不认证旧53.744 |
| 引入有限物理扰动 | 同估计器下需余项/尾项；paired sample bound另列 | S3 finite-forward effect实测，不等于population radius保证 |
| 换端点H→G/angle | 必须有endpoint discrepancy界或新的合适J_G | real角度、MSE、dual不可互换 |
| 换数据集/real estimator | conditional theorem不依赖数据集名，但条件需重新验证 | 两个DiLiGenT真实fixed-image diagnostic不具GLS风险证书，不做全队列或真实重标定优势声称 |

## 7. 复现与测试

```sh
python experiments/matched_ensemble_bridge.py --stage s1
python experiments/matched_ensemble_bridge.py --stage all
python -m pytest tests/test_ensemble_bridge.py -q
```

入口使用独有的新输出并拒绝覆盖已有产物；`--stage all` 要求先有同config SHA的S1 PASS。复现应在不含这些新产物的干净checkout执行，不得删除历史results来腾位置。当前新增的三个文件可直接审阅，主结果通过SHA引用S1且包含S2/S3副本。

测试覆盖proper singular/zero/tiny covariance、sigma² scaling、稠密与逐灯Schur、双侧投影及零模式、实际观测抽样、residual pool总体方差、S1→S2→S3逐项乘法、有限forward导数、quadratic bias项、余项半径、严格margin/ties、穷举部分rank bound、无margin反例、独立artifact及60位数值审计。ZIP gauge helper在主代理导入前为明确skip，不替换现有实现；其余测试通过。

正式理论在tex，经验条件和字段在JSON；任一real端点缺可测上界时就标 **not certified**，不造Spearman保证。只有条件定理、当前抽样实验和明确反例；没有理论向历史真实效果的无条件外推。
