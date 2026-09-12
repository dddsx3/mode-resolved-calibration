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

## 回答两个问题

1. **诊断**：哪些可辨识参数方向最脆弱？它们如何随标定精度恢复？
2. **决策**：只够重新标定 k 个分量时，能换来多少信息？选哪几个有区别吗？（用凸优化对偶间隙给出**全局认证**的回答，而不是只报一个策略排名）

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
   `R = I − VVᵀ`（恰 P−3L 个 ρ≡1）让全分辨率计算可行。
5. **奇异协方差**：零协方差 ≠ 零精度——奇异 Σ_c 走因子分解/边缘化路线，
   禁止用伪逆精度替换（库内有已知答案反例）。

完整推导见 [docs/methods.md](docs/methods.md)。

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

- **方向性验证**：细胞内 Spearman R_A = 0.90（bootstrap 95% CI
  [0.7, 0.95]，65/66 细胞、11/11 物体为正）——验证的是**方向**（理论标出的
  弱方向确实是经验上更脆的方向），不验证幅值；该细胞内统计量与按模式序号
  排序构造性等价的说明见 [docs/methods.md](docs/methods.md)；
- **幅值有效域**：真实数据 emp/pred 中位 201.1（合成 matched MC 为
  1.0045）——线性化理论在真实光度数据上的可证伪有效域；
- **认证动态范围**：全部 masked 像素、全部 142 灯下，重标所有 48 个
  有效灯可带来 **27.3–89.4%（中位 60.06%）** 的 tr ΔF⁻¹ 改善，逐物体
  差异大——所以按实例认证；J_A-greedy 距凸下界仅 **0.002–0.005%**；
- **靶向干预无可检出增量**：预注册三臂对照发现，模式靶向臂与标量 OED
  靶向臂统计不可区分——模式分解提供诊断洞察（哪些方向脆弱），但在简单
  标量准则之上没有增量分配价值（诚实负结果）；
- **负结果**：E-opt 增益违反次模性（γ_min = 0.704）；固定水平 severity
  分支经修正管线复测后已撤回（符号翻转，且该跨物体对比是标量化恒等式的
  产物）；策略排序差异全部落在认证 epsilon 内。

## 复现与完整性

- `checksums.sha256` 钉住**每一个**提交文件的哈希（CI 强制校验）；
- 每个认证分析在运行前提交 config、带 manifest（git SHA、种子、
  口径）；
- 复现命令、证据文件与验收测试的对应表见
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)；
- 流水线有两个口径：`legacy`（逐位复现原始基准产物）与 `corrected`
  （文档化的修正口径，报告数字所用）；A/B/C/D factorial 验证两者关系
  （原始口径逐位复现 + 修正口径重测所有头条）。

## 贡献

Bug 报告与 PR 欢迎——见 `CONTRIBUTING.md`。已知答案测试与声明门
（`tests/test_claims_gate.py`）在每次改动时运行。

## 引用 / 许可

软件引用见 `CITATION.cff`（研究论文撰写中）。许可见 `LICENSE`。
