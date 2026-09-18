[English](README.md) · 简体中文

# mode-resolved-calibration（模式级标定敏感度分析）

[![CI](https://github.com/dddsx3/mode-resolved-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/dddsx3/mode-resolved-calibration/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**calibinfo** 是一个分析"带标定不确定度的线性化逆问题"的 Python 库。它不把估计器的信息量压成一个标量（trace、对数行列式、E-最优性），而是追踪 Fisher 信息中**最脆弱的可辨识方向**：随着标定不确定度增大，每个脆弱方向上还剩多少可用信息（retention 谱），以及把标定预算花在哪些灯/分量上最有效（凸程序 + 认证下界）。

典型场景：光度立体、多光照重建等依赖辐射标定的逆问题——只要问题能局部线性化成 `y = A x + B δc + ε`（δc 是标定/干扰参数），就能用这套工具。

**图 1 · 脆弱方向、它们的谱、以及预算能买到什么。** 左：真实物体的逐像素
Fisher 信息——蓝色区域信息最薄；中：保留谱把 tracked 方向按脆弱程度排序；
右：精化少量灯恰好抬升最弱模式。

![首屏图](docs/img/hero.png)

![两问导览](examples/calibration_tour.png)

## 回答两个问题（四层流水线）

当前研究方向把**标定不确定度当作可设计的资源**：更好的标定值多少钱、
预算该花在哪、这个分配离全局最优有多远。程序分四层，每层有自己的对象、
问题与证据：

| 层 | 对象 | 回答的问题 | 证据 |
|---|---|---|---|
| **诊断 Diagnosis** | `R = F∞^{-1/2} ΔF F∞^{-1/2}` | 哪些可辨识方向被标定不确定度伤害？ | `docs/methods.md` §3（matched GLS）；当前 B1 `variants.D` 排序控制见下，不是 matched prediction 验证 |
| **估值 Valuation** | `V(B) = 1 − J*(B)/J₀` | 更好的标定值多少钱？ | `results/certification/certified_gaps_levels.json`（level 曲线） |
| **决策 Decision** | `t*(B) = argmin J_A(t)` | 预算具体投到哪里？ | `results/certification/certified_gaps.json`（greedy 前缀） |
| **认证 Certification** | `J_A(t) − J* ≤ g_FW(t)` | 离最优还有多远？ | `results/certification/certified_gaps.json`（FW 间隙） |

1. **诊断**：哪些可辨识参数方向最脆弱？它们如何随标定精度恢复？
2. **决策**：只够重新标定 k 个分量时，能换来多少信息？选哪几个有区别吗？（用凸优化对偶间隙给出**全局认证**的回答，而不是只报一个策略排名）

**面向任务的价值（标定是为了什么任务？）**：价值层可扩展到下游任务——
对任务算子 `H`（参数估计的线性泛函），`J_H = tr(H ΔF⁻¹Hᵀ)` 给同一标定
状态按任务定价（低秩 push-through 路线；`methods.md` §10）。在 11 个留出
物体上，同一均匀精化在 level 0.5 处对全参数 A-opt 泛函买回中位 62.87%
动态范围（= P-CERT 认证头条，新路线交叉复现），而对均值-反照率任务是
89.76%（level 0.1 处：7.82% vs 87.93%）；两个任务下的逐灯增益排序不一致
（中位 Spearman 0.936；44 个物体×level 单元中 11 个 top-3 完全不同）——
`results/goal_oriented/goal_orientation.json`。均值类任务逼近普适上界并非
巧合：ρ-加权均值泛函与灯强 nuisance 分量**精确 gauge 对齐**，闭式
`V = (1−1/κ)/(1+q·r)` 以中位 0.0008 的偏差预测其价值曲线
（`gauge_mechanism_rho_mean`；`methods.md` §10）。

## 安装

```bash
pip install -e .            # 核心（numpy、scipy）
pip install -e .[dev]       # + pytest
pip install -e .[examples]  # + matplotlib（示例/图）
```

需要 Python ≥ 3.10。

## 快速上手

```python
import numpy as np
from calibinfo.information.schur import delta_f
from calibinfo.information.retention import retention_spectrum

# 线性化模型  y = A x + B dc + eps,  dc ~ N(0, Sigma_c)
A = ...                      # (m, n) 关注参数的设计矩阵
B = ...                      # (m, q) 标定/干扰参数的设计矩阵
Lam = ...                    # 剖分精度 Lambda = sigma^2 * Sigma_c^{-1}
                             # （要求 Sigma_c > 0；奇异 Sigma_c 走
                             #  delta_f_marginal / 因子路线 C = B L）

DeltaF, M, diag = delta_f(A, B, Lam)       # Schur 补信息
spec = retention_spectrum(DeltaF, A.T @ A) # 逐模式保留率
# spec["rho"] 升序为可辨识子空间 range(F∞) 上的广义保留特征值；
# spec["modes"] 是对应特征向量——库所追踪的脆弱方向
```

## 方法要点

1. **有效信息（Schur 补）**：`ΔF(Λ) = Aᵀ[I − B(BᵀB+Λ)⁻¹Bᵀ]A`，单一实现
   （SVD/lstsq 路径，秩亏不裸解）。
2. **保留谱（广义特征值）**：`R(Λ) = F∞^{-1/2} ΔF F∞^{-1/2} ∈ [0, I]`，
   维数 `r = rank(F∞)`（不是干扰维数 q）；`ρ_j` 是广义 Rayleigh 量，
   不是两个矩阵普通特征值之比。
3. **协方差定理**：matched GLS 下 `F∞^{1/2} Cov(x̂) F∞^{1/2}/σ² = R⁻¹`，
   即 `1/ρ_j` 是归一化对偶误差坐标 `z_j = u_jᵀ F∞^{1/2}(x̂ − x)` 的方差
   放大倍数。
4. **预算分配是凸程序**：ΔF 对逐灯精度乘子联合算子凹 ⇒ 标准 OED 泛函
   凸 ⇒ Frank–Wolfe 对偶间隙给出全局认证下界；低秩结构
   `R = I − VVᵀ` 恰有 `P − rank(V)` 个 ρ≡1，仅在 `rank(V)=3L` 时为
   `P−3L`。其余模式由 `VᵀV` 的正特征值给出，不能重复计入其零模式。
5. **奇异协方差**：零协方差 ≠ 零精度——奇异 Σ_c 走因子分解/边缘化路线，
   禁止用伪逆精度替换（库内有已知答案反例）。
6. **γ 保证范围（M9/M12）**：旧 alpha-only 候选已 **refuted（反例否定）**，
   不是尚待补证；旧真实物体 γ ≥ 0.635 的保证解释已撤回。
   `gamma_lower_bound(alpha)` 的历史数值行为及回归测试完全保留，但不是
   有效证书。M12 是已证 **未加权全迹、固定 SPD 参数空间、PSD 更新** 的
   leave-one-out 谱界，API 为
   `calibinfo.allocation.alpha_bound.spectral_gamma_lower_bound(A, updates)`；
   不能直接外推到任意任务权重、奇异基线或变化的可辨识子空间，也没有在此
   报告替代旧值的真实物体谱界读数（methods.md §9）。

完整推导见 [docs/methods.md](docs/methods.md)。冻结光度管线的
`A_k = diag(s_hat_k)` 对应 **additive albedo（加性反照率）**，不是
log-albedo。均值/对比度表只作用于这个逐像素标量参数；单位法向的反照率–
法向联合模型具有内禀 **3P** 维（每像素一个反照率、两个法向切向坐标），
不是无约束 4P 模型。联合模型扩展见 methods，不把旧标量表当作联合实测结果。
导入证据与后续物理／联合模型推导的区分见
[理论续推索引](docs/theory/README.md)。

## 示例

`examples/` 下四个自包含脚本（自带合成数据，克隆即跑）：

| 脚本 | 演示内容 | 输出 |
|---|---|---|
| `ex1_retention_gauge.py` | 保留谱 + gauge 响应闭式 | `retention_gauge.png` |
| `ex2_mode_vs_scalar.py` | 脆弱模式判据 vs trace/logdet 标量 | `mode_vs_scalar.png` |
| `ex3_allocation_demo.py` | 迷你标定分配实验（多策略贪心） | `allocation_demo.png` |
| `ex4_calibration_tour.py` | 两问导览：脆弱方向 + 预算价值的认证回答 | `calibration_tour.png` |

## 基准与认证结果（11 个留出 OpenIllumination 物体）

结果文件与复现命令的完整索引见 [docs/claims.md](docs/claims.md)
（声明 → 证据文件 → 复现命令 → 验收测试），复现指南见
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。摘要：

- **幅值比较（B2；同投影，不是同系综）**：修正 D 臂的经验退化量与
  同投影线性预测之比，中位 53.744194（5–95% [0.489414, 820.342156]），
  绑定新导入 [`amplitude_comparison.json`](results/theory_extension_20260918/imported_20260917/amplitude_comparison.json)
  的 `variants.D.new_ratio_matched_prediction.pooled`。预测仅匹配逐 seed
  gauge 投影及 dual 坐标；经验分母仍是 **residual bootstrap（残差重抽样）**，
  不是 matched GLS 的标定极限系综。因此比值偏离一既不能验证、也不能反驳
  matched-GLS 方差定理；合成 matched MC 参照仍为 1.0045。

  **历史幅值摘要（superseded）**：此前修正 D 臂中位 15.98、原 A 臂中位
  201.1 保留在 `results/magnitude/directional_amplitude_summary.json`，
  不是当前 B2 读数，也不能作为定理失效的证据。

- **历史有效域图（B8；旧投影管线）**：冻结 validity map 的各曲率箱
  Spearman 中位 ≥ 0.8、Kendall 符号一致率中位 ≥ 0.7，仅是历史读数，
  **不能保证当前逐 seed gauge 投影管线的排序稳健性**；旧“线性化失效
  只打幅值、不打方向”的推广已被 superseded
  （`results/magnitude/validity_map.json`）。
- **认证动态范围**：全部 masked 像素、全部 142 灯下，重标所有 48 个
  有效灯可带来 **27.3–89.4%（中位 60.06%）** 的 tr ΔF⁻¹ 改善，逐物体
  差异大——所以按实例认证；J_A-greedy 距凸下界仅 **0.002–0.005%**；
  动态范围是标定不确定度操作点 `level` 的**函数**而非单个数:同一泛函在 11 档
  level 网格上重采样(见 `results/certification/certified_gaps_levels.json`),
  中位 D 从 level=0.025 的 0.7% 单调上升到 level=8.0 的 89.85%(37 倍跨度);
  P-CERT 操作点 level=0.5 处为中位 62.87%。曲线在大 level 处逼近普适上界
  1 − 1/κ = 90%(docs/methods.md §7,level=64 表即此上界被取到);
  通道分解:标定预算价值几乎全部由**强度**通道承载——仅强度通道的 D 曲线
  在每档 level 都与 joint 复现(level=0.5 处中位差 <0.05pp,见
  `results/openillumination/channel_decomposition.json`);方向通道在
  每档 level/物体上的贡献都 ≤2%(最大 1.68%,obj_19_cylinder @ level 0.2),
  且其随 level 的剖面是非单调的帽形——被追踪的脆弱**方向**提供诊断,而非
  预算价值的量级;
  **Σ_φ 参数族敏感性(P-SIGMA-FAMILY)**:把单线参数化(logI/方向方差比被
  钉死在 ~3283)换成预注册三轴族——独立通道比(σ_dir 至 25°)、逐灯异质
  性(het 至 1.0)、通道耦合(|ρ| ≤ 1,PSD 有效域)——区分了稳健结论与参数化特定
  结论(`results/openillumination/corruption_family_sensitivity.json`、
  `results/magnitude/linearization_radius_family.json`):**强度通道主导
  稳健**(操作点方向份额 0.07%,σ_dir 扫到 25° 也从未达到 2% 阈值,11/11
  物体右删失);**普适上界在整个族上成立**(定理,运行时守卫:495 行最大
  违反 −0.0028);**两项定律恰在价值所在的区域有效**(中位
  |dV| 0.0076;强度承载区最差 0.048 仍处于冻结包络,爆点 0.464 全部
  落在 D 已排除的方向独占区);**线性化半径依赖通道**
  (joint 与仅强度:1.0/1.5,冻结值逐位复现;仅方向在整个 [0.05, 8] 网格
  从未离开一阶标度;het=1.0 使半径缩小 5 倍——逐灯标定异质性
  让名义线性化比同总方差的任何均匀尺度腐蚀都提前约 5 倍失效,
  这是本轮全新的物理发现)。族机制在退化锚点上对冻结
  产物逐位复现(264/264 通道分解行 + 逐点锚点)。**知情序有效性(E)参数化特定,且非 σ 误设伪影**:控制臂精确复现
  冻结子集(88/125 = 0.704);方向更重的腐蚀使预测序**更**可信(0.889),
  但逐灯异质性把它破坏到低于随机(0.353,逐格 Spearman 中位 −0.78);
  零成本交叉诊断排除 σ 误设解释——把真实逐灯 σ 交给预测器其排序只动
  0.05(S_pred = 0.949),而实测收益排序被完全重排(S_real = −0.325):
  失效源于真值端脱钩,逐灯排序在异质性下失去决策有效性,与 σ 规格无关
  (`results/openillumination/decision_quality_family.json` +
  `corruption_family_e_diag.json`)。

  **物理锚点(P-BALL-ANCHOR)**:在 DiLiGenT ballPNG(96 灯,GT 法向)上跑
  标准球光度立体重标定,把 Σ_φ 锚到**实测**误差——方向 RMS 2.96°、
  log 强度 std 0.0159——真实流程的方向方差是其强度方差的 ~10.6 倍,
  与 joint 参数化强制的 3283 **相反**。在该实测锚点上方向通道承载
  预算价值的中位 **36%**(预注册规则判 D-flipped):强度主导结论
  参数化稳健但**并非经验普适**——更好标定的价值取决于标定流程的
  误差实际落在哪个通道(一个**通道条件性判据**,两个区域都有实证
  实例:合成族扫描 + 真实实测锚点;S1 扫描固定的 σ_logI ∈ {0.05…1.0}
  正是强度主导的支撑来源——实测锚点低于扫描下限 3.1 倍;翻转的
  物体间跨度 4.8%–71.4%)
  (`results/openillumination/ball_anchor.json`)。

  **第二数据集(P-DILIGENT-QUEUE,loader 已修正)**:在 DiLiGenT(10 物体/
  96 灯,独立采集系统)上:**线性化包络精确迁移**(radius_2x 中位 1.0,
  全物体 [0.75, 1.5];radius_10x 10/10 全为 1.5);**强度主导的通道
  分裂同样迁移**(方向 max D 0.20%,OI 为 1.68%;强度通道复现 joint
  误差 <0.05pp)。仍保持条件性的:球锚点方向份额在同一实测剖面下按
  队列/物体不同(OI 中位 36%,区间 4.8–71.4%;DiLiGenT 中位 ≈0,
  cat 22%/pot1 29% 为高值物体)。
  **机制(P-ANCHOR-MECHANISM)**:物体级方向份额与一个**不经 D 功能量**
  计算的场景统计量单调相关——弱模式子空间上方向/强度 nuisance 能量比
  (pooled Spearman −0.899;OI 内 −0.927、DiLiGenT 内 −1.000):队列差异
  是场景几何的后果,而非数据集本身的属性。
  *更正说明*:此前段落报的"通道分裂
  不迁移"及较大的 pot1/pot2 锚点份额,是 loader 丢失 DiLiGenT 逐灯光强
  归一化的伪影;已修复、门禁化、撤回(字段路径+日期:
  `results/diligent/provenance/loader_normalization_fix.json`,
  2026-09-16)。

  **基线对照(P-BASELINE v1.1,active 集控制)**:Drbohlav–Chantler
  式良构配置选择(方向球面贪心最远点采样——纯几何),候选池限制在
  照亮物体的灯内,对照 active 集内置换的随机基线(C8 规定的框架):
  **OpenIllumination 上几何有增益但替代不了模型**(dc05_active
  5.24°/4.01° vs randomA48 5.78–5.99°/4.49–5.25°,e_opt 4.52°/2.89°);
  **DiLiGenT 上模型的逐灯排序本身不迁移**(a_opt 输给 active 随机,
  偏差 +0.24°/+0.47°,dc05_active 也仅打平)——排序优势与论文所有
  其他发现一样是条件性的。v1 全域池 DC05 的随机带表现是可见性混淆
  (预算 36–86% 落在照不到物体的灯上,重合度已逐物体记录;88 行 OI
  锚点与冻结 E 臂逐位一致;`results/baseline/baseline_comparison.json`)。
- **靶向干预无可检出增量**：预注册三臂对照发现，模式靶向臂与标量 OED
  靶向臂统计不可区分——模式分解提供诊断洞察（哪些方向脆弱），但在简单
  标量准则之上没有增量分配价值（诚实负结果）；
- **负结果**：E-opt 增益违反次模性（γ_min = 0.704）；固定水平 severity
  分支经修正管线复测后已撤回（符号翻转，且该跨物体对比是标量化恒等式的
  产物）；策略排序差异全部落在认证 epsilon 内。
- **保留序控制（B1；当前逐 seed gauge 重算）**：单元内 Spearman
  R_A = 0.55（object-cluster bootstrap 95% CI [-0.1, 0.7]，43/66 单元、
  7/11 物体为正），对应新导入
  [`mf0_factorial_summary.json`](results/theory_extension_20260918/imported_20260917/mf0_factorial_summary.json)
  的 `variants.D`：`gauge_mode=per_seed`、`prediction_field=pred_deg`。
  这是**原预测排序控制**，不是 matched prediction 验证；CI 跨零，不能据此
  宣称当前管线的方向验证稳健，更不验证幅值 `1/ρ_j`。

  **历史 B1（superseded）**：R_A = 0.90（CI [0.7, 0.95]，65/66 单元、
  11/11 物体为正）保留在旧 `results/openillumination/correctness/mf0_factorial_summary.json`。
  其与 mode-index 的精确等价（66/66 单元、偏差 0.0）是场景内构造恒等式，
  不是当前管线的样本外证据。

## 复现与完整性
**入口**:`./reproduce.sh` 分阶段复现整条链——`--stage synthetic`
(无需原始数据)、`--stage diag`(对已提交产物的零成本诊断)、
`--stage oi` / `--stage dq`(原始数据实验,长跑在 `--long` 后)、
`--stage figures`。每步打印其产物的 `gate` 名便于与 `results/` 对账;
数据缺失时显式 SKIP。`--list` 打印 runner 表。


- `checksums.sha256` 钉住**每一个**提交文件的哈希（CI 强制校验）；
- 每个认证分析在运行前提交 config、带 manifest（git SHA、种子、
  口径）；
- 复现命令、证据文件与验收测试的对应表见
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)；
- 冻结 `legacy` / `corrected` A/B/C/D factorial 保留为历史复现锚。
  **B1/B2 当前读数来自另行导入的逐 seed gauge 修正**，来源由
  `results/theory_extension_20260918/imported_20260917/manifest.json` 固定；
  旧 machinery 对拍通过不表示新经验值与旧摘要相等。其他冻结分析仅保持
  各自记录的口径和范围，详见 `docs/EXPERIMENTS.md` §11。

## 贡献

Bug 报告与 PR 欢迎——见 `CONTRIBUTING.md`。已知答案测试与声明门
（`tests/test_claims_gate.py`）在每次改动时运行。

## 引用 / 许可

软件引用见 `CITATION.cff`（研究论文撰写中）。许可见 `LICENSE`。
