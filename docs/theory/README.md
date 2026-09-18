# 剩余推导与外推：数学增补及证据入口

本轮从本地仓库与 20260917 证据包继续，不假定能访问未提供的上一会话。仓库起点为 `22ee2e24e1ac7d0e411cad65c8e18dc6aa94f5f2`，工作位于分支 `theory/remaining-derivations-20260918`；2026-09-19 已落地提交并推送远端。

## 1. 读者入口

- **完整数学增补**：`remaining_derivations.tex` 及同名 PDF。中文导读、物理与联合模型正文，加上可直接引用的英文余项证明；模块可分别并入论文，不强制改动现有主稿。
- **①物理 γ 反例与④谱界紧度**：`physical_gamma_and_tightness.md`，正文 `section_physical_gamma.tex`。
- **②两项定律余项**：`two_term_remainder.md`，正文 `section_two_term_remainder.tex`。
- **③匹配系综与⑥排序桥接**：`matched_ensemble_and_bridge.md`，正文 `section_ensemble_bridge.tex`。
- **⑤联合反照率—法向模型**：`joint_information_extension.md`，正文 `section_joint_extension.tex`。
- **结论注册**：`../claims.md` 中 M9 的否证状态、M12–M17、C13 与 C14；B1/B2/B8 同步到新旧证据的正确范围。论文主张↔注册表编号的完整映射见 `../paper_binding.md`。

## 2. 本轮真正推进了什么

**物理负结果。** 新构造不再只写任意二维矩阵，而是从单位法向、灯向、着色、正权重、方向切基与逐灯正定先验装配 LightBlocks。在固定精化倍率与固定 α 下仍有 γ 趋零族。先验允许通道耦合与逐灯变化，不能扩大成对所有更窄先验子类的否定。

**近似升级为主项加余项。** 对精确 gauge 的单位任务，严格得到

`J_a(t)=1/(tq)+r+δ_D+eᵀ(tI+S_perp)⁻¹e`。

余项非负，固定谱箱可给不输入实测误差的区间。44 个名义条件和完整 C9 家族 495 条记录都被检查；两组有 11 个公共锚点，所以共 539 条验证记录、528 个不同的对象—参数条件，并不是 539 个独立样本。Direction-only 的大偏差可以由设计侧区间预测，价值差的界／真误差倍数却不具有普适常数，因为端点误差可以抵消。

**匹配理论与真实影像边界分开。** S1 五项匹配闭合以后才做 S2 基线替换和 S3 有限前向扰动；不把历史 53.744 单独当成定理失败证据。排序命题加入协方差、偏差、非线性与采样的误差界及设计对之间的裕量，拒绝无条件 Spearman 保证。

**谱界改进有有限范围的紧度定理。** 留二与完整集合条件数增强原界；二维、两候选、固定基线条件数 χ 的下确界为 1/χ，并有参数族和收敛阶。但该族不固定更新幅度或 κ，联合约束下的精确最坏值仍未求出。

**联合模型保持结构上限与证书。** 场景内禀维数为 3P。Proper 标定先验与平坦惩罚的可辨识性不同；全亮完整观测的未标定因子分解通常有九维 GL(3) 规范，而非只删尺度。在固定模型、固定商空间和固定 H 下，价值上限和凸分配证书保留。独立合成线性与小噪声非线性剖面估计协方差已经验证，不宣称实物非线性优势。

## 3. 与旧仓库不一致之处如何处理

1. 导入前仓库仍保留旧 α-only 说法。已按证据包导入 `spectral_gamma_lower_bound` 与对应回归测试；历史 `gamma_lower_bound` 的名称、数值、输入和警告行为不变，只明确其候选式已被否证。
2. B1 新值来自逐样本投影、原预测排序控制；B2 新值只匹配坐标，不匹配完整生成与估计系综。旧 JSON 和快照原样保留。导入统计副本位于新的 `results/theory_extension_20260918/imported_20260917/`，标注为 **导入证据，而非本轮新测量**。
3. 根据实际导数，固定法向代码的场景变量是加性反照率，log-albedo 必须作合同变换并同步变换任务。没有因此重算历史结果。
4. 模式数的精确结论是 `P-rank(V)`，不是无条件 `P-3L`；单位 retention 模式是标定不敏感的可辨识模式。低秩模块只修正说明，数值实现未变。
5. 旧上限证明中不合法的矩阵等式与减法推理，换为共享模型的变分证明；上限结论本身未被推翻。

## 4. 可执行复现入口

在仓库根目录安装开发与复现依赖后执行。SymPy 只为精确符号核验登记在开发依赖，不是核心库运行依赖。

```bash
python -m pip install -e ".[dev,reproduce]"
python -m pytest -o addopts= -q
python experiments/gamma_extensions.py
python experiments/two_term_remainder.py --data-root D:/data/OpenIllumination --data-meta D:/data/OpenIllumination_meta
python experiments/matched_ensemble_bridge.py
python experiments/joint_information_extension.py
```

各脚本只写本轮新增目录，但重跑会覆盖本轮同名结果，若要保留此次记录，先使用各入口的 `--out` 或输出目录选项另存。原图像仍由本地数据路径读取，未下载或上传。Gamma 与余项的真实设计复算使用历史名义选择；匹配桥接使用其配置中事先固定的小子集，不能把小子集结果改称整个冻结队列的全面验证。

源码排版使用已经存在的 XeLaTeX 和中文字体，无新增大型安装。可从本目录执行：

```bash
xelatex -no-shell-escape -disable-installer -interaction=nonstopmode -halt-on-error remaining_derivations.tex
xelatex -no-shell-escape -disable-installer -interaction=nonstopmode -halt-on-error remaining_derivations.tex
```

## 5. 数字与证据字段

所有路径均以仓库根为基准；精确数值看 JSON，正文按适当精度显示。

| 事项 | 产物 | 字段定位 |
|---|---|---|
| 当前 B1 原预测排序控制 | `results/theory_extension_20260918/imported_20260917/mf0_factorial_summary.json` | `variants.D.{RA,RA_ci95,n_pos_cells,n_pos_objects,n_objects}`、`protocol.n_cells` |
| 当前 B2 同坐标幅值 | `results/theory_extension_20260918/imported_20260917/amplitude_comparison.json` | `variants.D.new_ratio_matched_prediction.pooled`、`bootstrap` |
| 物理反例与谱界 | `results/theory_extension_20260918/gamma_extensions.json` | 具体 schema 见该项推导说明 |
| 余项全量验证 | `results/theory_extension_20260918/two_term_remainder.json` | `summary.nominal`、`summary.family`、`summary.direction_only_worst`；细项 `rows[].endpoints` 与 `rows[].value` |
| 匹配系综与桥接 | `results/theory_extension_20260918/matched_ensemble_bridge.json` | 具体 schema 见该项推导说明；阶段结果另存 S1 与 nested 文件 |
| 联合协方差 | `results/theory_extension_20260918/joint_information_extension.json` | `rows[].linear`、`rows[].nonlinear`、`acceptance` |
| 联合规范维数、上限与离散证书 | 同上 | `dimension_and_gauge`、`ceiling`、`certificate` |

## 6. 未越过的边界

- 未解决所有狭窄物理先验子类的 γ 最坏值。
- 未解决同时固定 κ、更新规模、条件数与几何的联合紧度问题。
- 未把有限条件上的余项界紧度升级为价值差的普适相对紧度。
- 未宣称 S1–S3 已完整归因历史 53.744。
- 未从二次信息成本直接推出真实角度误差或无条件跨系统排序。
- 未运行真实对象的联合反照率—法向分配，也未宣称大扰动非线性全局证书。

各项的“已证”“已验证”“边界未闭合”分别记录；不以随机未违反、测试通过或精美排版替代数学证明。
