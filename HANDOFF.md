# HANDOFF · 会话交接（2026-09-09 · P1 断代式清理执行中断点）

> **当前任务**：《P1_断代式清理执行条例_v1.0.md》（D 盘根）——把旧研究仓库
> `dddsx3/Multi-Illumination-Inverse-Rendering` 断代为一个全新的论文复现仓库
> `mode-resolved-calibration`。**科学内容零改动**（Branch B 已闭口，任务书 v3.0 + 执行条例 v1.0 是上位文件）。
> **中断原因**：执行会话上下文溢出（反复 400 报错），在新仓库 S2b→S3 验收交界处交接。
> **交接时状态**：S0/S1 完成、S2a 完成、S2b 约 90%、S2c 未开始、S3 未开始、S4 未开始。

---

## 1. 30 秒接手须知

- **旧仓库**（只读！）：`D:\MIR_Archive_20260829\Multi-Illumination-Inverse-Rendering`，HEAD `9adb84a`（工作树多 1 个已提交的封存 commit `ab8404b`：README 横幅 + checksums-archive.sha256）。tag `research-archive-2026-09` 已推 GitHub。
- **备份**（G0 已过，四项保险齐）：
  - `D:\MIR_Archive_20260829\archive_backup\Multi-Illumination-Inverse-Rendering.git`（mirror clone，含全部 tag）；
  - `D:\MIR_Archive_20260829\checksums-archive.sha256`（2877 文件，text 模式，抽验 30/30 OK）；
  - GitHub tag 本身。
- **新仓库（工作区）**：`C:\Users\35702\publication\mode-resolved-calibration`，fresh git init，**尚未 git commit**，**尚未 push**（T2.4：push 前须过 G1–G5 + 24h 冷静期——冷静期由用户裁决）。
- **上位文件**（D 盘根）：《CI04R_最终收口级任务书_v3.0.md》《CI04R执行后审计与专家提交文档.md》《P1_断代式清理执行条例_v1.0.md》。断代条例第二章（保险总纲）效力最高，冲突即 STOP。

## 2. 断点精确位置：S2b 适配的 2 个测试失败待修

新仓库 pytest 当前 **2 failed**（其余全绿），原因与修法已定位：

1. **`tests/test_datasets_diligent.py::test_load_object_matches_legacy_exp8r`**：
   该测试 `sys.path.insert` 旧仓库的 `critical_experiments/` 做口径对账——新仓库无此目录。
   **修法**：删除该测试函数（口径对账是旧仓库阶段的验收，迁移后无意义），或改跳过。
2. **`tests/test_io_manifest.py::test_git_sha_present_in_repo`**：
   断言 `git rev-parse HEAD` 非 None——新仓库还没做过 git commit（无 HEAD）。
   **修法**：在 S2 收尾先做首次 git commit（见 §4 顺序），测试自然转绿；或 skipif 无 git。

修完这两个 = G2 门（pytest 全绿）过。

## 3. 已完成清单（不要重做）

- ✅ **S0 保险（G0 门过）**：tag `research-archive-2026-09` 已推 GitHub；mirror clone 在
  `D:\MIR_Archive_20260829\archive_backup\`；双份 checksums（旧仓库根 + MIR_Archive 根，2877 文件）；
  三批抽验 30/30 OK。
- ✅ **S1 封存**：旧仓库 README 顶部横幅已 commit+push（`ab8404b`）。
  **留给用户**：GitHub 网页 Settings→Archive（可逆）+ About 栏描述修改。
- ✅ **S2a 复制**：新仓库 `C:\Users\35702\publication\mode-resolved-calibration` 已建 76 文件；
  同名直迁 26 文件 sha256 对账 0 漂移。
- ✅ **S2b 大部分**：
  - src 全量 + `ci04r_predictors.py`→`spectral_criteria.py` 改名；
  - tests 16 个（2 个依赖旧管线的已按表 A 删除：test_permutation_invariance/test_stability_guards；
    `_redteam_reference.py`→`_reference_impl.py`，其引用已改）；
  - experiments 6 个 R 改名（`ci04r_final_closure.py`→`openillumination_severity.py` **已删 branch
    裁决逻辑与 PASS-A/FAIL-A 字段**，输出改 `validation_summary.json`；语法已验证）；
  - configs 7 个 R 改名（`configs/{synthetic,gauge_spectrum,monte_carlo,nonlinear,openillumination,diligent,diligent_ablation}.yaml`）；
  - results/（frozen 14 文件按表分目录 + ci04r 三 CSV 改名 mode_ranking/level_severity/predictor_comparison）；
  - data/manifests/（openillumination_dev_manifest + splits×2）；
  - **去治理措辞已完成 6 轮**（"红队/任务书/CLAIMS_REGISTRY/branch_decision/rt_f12/CI04R"等高危串在
    py/yaml/sh 中已清零；grep 验证）；
  - `pyproject.toml` 需要把项目 name 改为 `mode-resolved-calibration`（上轮 replace 未生效，**待办**）。

## 4. 剩余工作（按序执行）

1. **修 2 个失败测试**（§2）；
2. `pyproject.toml` 项目 name 修正 → 重新 `pip install -e .` → pytest 全绿（G2）；
3. **S2c 新文档**（条例 §5，全部新写）：
   - `README.md`（专家六节：Title/Overview/Mathematical core 仅 4 式/Key results 仅 2 图/Reproduce 三命令/Data+Layout+Citation；**禁一切状态字/阶段/裁决/事故语言**）；
   - `docs/EXPERIMENTS.md`（9 份 CI memo 压缩为纯客观 protocol：每实验 Dataset→sampling→corruption→estimator→metric→bootstrap→seed→output；OpenIllumination 节须含 11 对象名单/6 levels/20 seeds/5 modes/within-cell Spearman 定义/fixed-level severity/object-cluster bootstrap/B=10000/seed=20260908；**不写任务书史/AI 决策史**）；
   - `docs/REPRODUCIBILITY.md`（Paper item → Script → Source data 三列）；
   - `docs/DATA.md`（OpenIllumination/DiLiGenT 官方下载指引 + 不含原始数据声明）；
   - `CITATION.cff`；
   - `checksums.sha256`（新仓库文件清单）；
   - `tests/test_reproduction.py`（读 results/** 断言附录 C N1–N9 逐位复现——N4 阈值 ≤1e-14）；
4. **S3 验收 G2–G5**：
   - G3 数字门 N1–N9（条例 §7.B 清单，全部须逐位复现）；
   - G4 泄漏门：全文件 grep 模式族（条例 T2.8 词族）0 命中（results 重命名文件与 EXPERIMENTS.md 客观语言豁免）+ `git log --all` 无旧历史；
   - G5 清洁室：新 venv → install → pytest → reproduce → figures/tables 从零再生。
5. **S4 发布预备**：本地 commit + tag `pub-stage-1..5`；**push/公开/旧仓库 Archive 按钮均为用户真人操作**（24h 冷静期）。
6. 全程禁改：frozen 数值、统计定义、seed、公式；禁 notebook 唯一路径。

## 5. 关键路径速查

| 项 | 值 |
|---|---|
| 旧仓库（只读） | `D:\MIR_Archive_20260829\Multi-Illumination-Inverse-Rendering`（HEAD ab8404b） |
| mirror 备份 | `D:\MIR_Archive_20260829\archive_backup\Multi-Illumination-Inverse-Rendering.git` |
| 新仓库工作区 | `C:\Users\35702\publication\mode-resolved-calibration`（git init 过，未 commit） |
| 数据（不打包） | `D:\data\OpenIllumination`、`D:\data\DiLiGenT\pmsData` |
| git push 代理 | `https_proxy=http://127.0.0.1:63770`（会轮换，失败查注册表 ProxyServer） |
| 上位文件 | D 盘根：断代条例 v1.0 > CI04R 任务书 v3.0 > 主控计划书 v1.0 > 宪法 v1.0 |
| Branch B 锁定句 | `CLAIMS_REGISTRY.yaml::claims_lock_20260909`（旧仓库）+ 手稿 §VIII 已入稿 |

## 6. 已知坑（本会话踩过）

1. **Git Bash 路径**：新仓库在 `/c/Users/...`，旧在 `/d/...`——shell cwd 常被重置为
   `D:\MIR_Archive_20260829`，**每条命令显式 cd**；git 命令必须在仓库目录内执行否则
   "not a git repository"。
2. **Python 写中文文件**：heredoc 里 `\n` 会被 bash 吃掉——写含 `\n` 字面量的文件用
   Write 工具或 `chr(10)` 拼接，禁 heredoc 直写。
3. **grep 中文显示**：Git Bash 控制台 GBK 假象（乱码≠内容坏）——权威判断用 Python
   `Path.read_text(encoding="utf-8")` 扫描。
4. **pytest 收集**：新仓库 `pyproject.toml` 的 testpaths 若指向 `tests/unit` 需改为 `tests`（结构已扁平化）。
5. **分支裁决逻辑已删**：`openillumination_severity.py` 无 branch/PASS-A 字段——
   不要从旧仓库重新拷贝覆盖；若需要旧版对照，去 mirror 备份看。

## 7. 验收后交付物清单（S4 前检查）

- [ ] G2–G5 全过记录（写 `docs/REPRODUCIBILITY.md` 附对账数字）；
- [ ] 新仓库 `git log --all` 只有新历史；
- [ ] 文件数 ≤170（当前 76+文档 ≈ 85）；
- [ ] 旧仓库 GitHub Archive 按钮（用户）；
- [ ] 新仓库 GitHub remote 由用户创建（建议名 `mode-resolved-calibration`，先 private）。

*交接完成。按条例 T1.2 默认拒绝原则执行剩余步骤；未覆盖情况 STOP 登记。*
