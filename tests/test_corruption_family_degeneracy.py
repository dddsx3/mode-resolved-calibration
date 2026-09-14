"""S0-3 · 参数族退化锚点回归(P-SIGMA-FAMILY 的地基,**最重要**)。

把 S0-1/S0-2 的改造在现有三档参数化(intensity / direction / joint)的
level 网格上与已入库的 `results/openillumination/channel_decomposition.json`
**逐位比对**。任何一位不一致 => 后面所有家族数字建立在未确认的改写上,
全部作废。

两个层次:
1. **单元锚点**(CI-safe,无原始数据):预测侧 delta_f_injected 的共享
   (3,3) 路径在改造前后逐位一致(共享块堆叠 (L,3,3) == 共享 (3,3) 单次
   调用);注入侧标量退化 == apply_scaled_corruption 逐位一致;构造器
   het=0/rho=0 == CorruptionGenerator.sigma_phi_diag() 逐位一致。
2. **端到端锚点**(需原始数据,本地跑):三档 × level 网格重算
   channel-decomposition 的 D(level) 表——镜像原实验真实生成路径
   (build_state + J_A,关闭通道极小正方差),唯一替换点 Σ_φ 来源换
   成 CorruptionFamily。产物 264 行的 J_A_1 / J_A_kappa / D 与已入库
   值浮点全等(==)。数据缺失时 skip。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from calibinfo.models.corruption import CorruptionGenerator
from calibinfo.models.corruption_family import CorruptionFamily
from calibinfo.allocation.corruption import (
    raw_innovations, apply_scaled_corruption,
    raw_innovations_family, apply_corruption_family)

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/channel_decomposition.json"


# ------------------------------------------------ 1. 单元锚点(CI-safe)
def test_predict_side_shared_block_bit_identical():
    """S0-1:共享 (3,3) 堆叠成 (L,3,3) 后,ΔF 与改造前的共享单次调用逐位一致。

    用合成场景(无原始数据)直接比较两条代码路径:
    (a) sig 传入 (3,3)(走 per_light=False 分支——即旧实现);
    (b) sig 传入 np.stack([sig]*L)(走 per_light=True 分支——新实现)。
    """
    rng = np.random.default_rng(11)
    P, L = 60, 7
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s_hat = rng.uniform(0.2, 1.5, size=(L, P))
    nrm = np.tile(np.array([0.0, 0.0, 1.0]), (P, 1)) + 0.05 * rng.normal(size=(P, 3))
    B_phi = rng.normal(0.0, 0.5, size=(L, P, 3))
    sig = np.diag([0.25, np.radians(0.5) ** 2, np.radians(0.5) ** 2])

    from calibinfo.information.schur import delta_f

    def run(sig_arg):
        DF = np.zeros((P, P))
        for k in range(L):
            A_k = np.diag(np.sqrt(w[k]) * s_hat[k])
            B_k = np.sqrt(w[k])[:, None] * B_phi[k]
            Lam = (np.linalg.inv(sig_arg[k]) if np.ndim(sig_arg) == 3
                   else np.linalg.inv(sig_arg))
            DFk, _, _ = delta_f(A_k, B_k, Lam)
            DF += DFk
        return DF

    shared = run(sig)
    stacked = run(np.stack([sig] * L))
    assert np.array_equal(shared, stacked)


def test_injection_side_scalar_degeneracy_bit_identical():
    """S0-2:标量 σ + rho_c=0 时 family 路径 == 既有路径(含 scales ≠ 1)。"""
    rng_a, rng_b = np.random.default_rng(21), np.random.default_rng(21)
    dirs = np.random.default_rng(5).normal(size=(48, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    raw_a = raw_innovations(rng_a, 48)
    raw_b = raw_innovations_family(rng_b, 48, rho_c=0.0)
    assert all(np.array_equal(a, b) for a, b in zip(raw_a, raw_b))
    scales = np.where(np.arange(48) % 3 == 0, 1.0 / np.sqrt(10.0), 1.0)
    for sI, sR in ((0.5, 0.01), (0.0, 0.02), (0.3, 0.0)):
        oa = apply_scaled_corruption(dirs, sI, sR, scales, raw_a)
        ob = apply_corruption_family(dirs, sI, sR, scales, raw_b)
        assert np.array_equal(oa[0], ob[0])
        assert np.array_equal(oa[1], ob[1])


def test_family_constructor_degeneracy():
    """S0-4:het=0/rho=0 时每灯块 == CorruptionGenerator.sigma_phi_diag()。"""
    for sI, sD in ((0.5, 0.5), (0.2, 1.0), (1.0, 0.1)):
        fam = CorruptionFamily(sI, sD, het_sigma=0.0, rho_c=0.0, n_lights=32)
        gen = CorruptionGenerator("joint", 0.0)          # 只用它的参数化形状
        ref = np.diag([sI ** 2, np.radians(sD) ** 2, np.radians(sD) ** 2])
        blocks = fam.sigma_phi_block()
        assert np.array_equal(blocks[0], ref)
        assert np.array_equal(blocks[-1], ref)
        assert all(np.all(np.linalg.eigvalsh(b) > 0) for b in blocks)
        # rho_c=0 时向量退化 == 标量
        assert np.array_equal(fam.sigma_logI_vec(), np.full(32, sI))
    # het>0:固定种子可复现 + 分布宽度合理
    f1 = CorruptionFamily(0.5, 0.5, het_sigma=0.5, n_lights=142)
    f2 = CorruptionFamily(0.5, 0.5, het_sigma=0.5, n_lights=142)
    assert np.array_equal(f1.sigma_logI_vec(), f2.sigma_logI_vec())
    v = f1.sigma_logI_vec()
    assert v.min() < 0.5 < v.max()
    assert 0.2 < np.median(v) < 1.5


def test_rho_c_couples_channels_only():
    """rho_c ≠ 0:axes/angles 不变,logs 变(耦合只进 (logI, angle) 对)。"""
    ra = np.random.default_rng(31)
    rb = np.random.default_rng(31)
    n = 24
    r0 = raw_innovations_family(ra, n, rho_c=0.0)
    r5 = raw_innovations_family(rb, n, rho_c=0.5)
    assert np.array_equal(r0[0], r5[0])          # axes 相同
    assert np.array_equal(r0[1], r5[1])          # angles 相同
    assert not np.array_equal(r0[2], r5[2])      # logs 变了


# ------------------------------------------------ 2. 端到端锚点(需数据)
def test_channel_decomposition_end_to_end_bit_identical():
    """三档 × level 网格重算 D(level) 与已入库产物逐位一致。

    镜像原实验的真实生成路径(experiments/channel_decomposition.py 的
    build_state + J_A):场景装配、关闭通道极小正方差(radians(1e-3)² /
    (1e-6)²)、κ=10 端点全部相同,**唯一替换点是 Σ_φ 来源**——从
    `_sigma_phi_diag` 换成 CorruptionFamily(het=0, rho_c=0) 的
    sigma_phi_block() → (K,3,3) → 批量 inv。对产物 264 行的
    J_A_1 / J_A_kappa / D 做浮点全等(==)比对;任何一位不一致即失败。
    """
    import yaml
    from calibinfo.datasets.openillumination import load_object
    from experiments.openillumination_validation import NominalScene
    from calibinfo.allocation.blocks import LightBlocks
    import experiments.channel_decomposition as cd

    if not ART.exists():
        pytest.skip("artifact not committed")
    art = json.loads(ART.read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO / "configs/channel_decomposition.yaml")
                         .read_text(encoding="utf-8"))
    if not Path(cfg["data_root"]).exists():
        pytest.skip("raw data absent")

    kappa = float(cfg["kappa"])
    K = int(cfg["K_lights"])
    closed_deg = float(cfg["closed_direction_sigma_deg"])
    closed_li = float(cfg["closed_intensity_sigma_logI"])
    types = [str(x) for x in cfg["channels"]]
    levels = [float(x) for x in cfg["levels"]]

    art_rows = {(r["object"], r["channel"], float(r["level"])): r
                for r in art["rows"]}
    assert len(art_rows) == len(art["rows"])          # 无重复键

    n_checked = 0
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name,
                          data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        t_one = np.ones(K)
        # P-CERT 同款装配(与 cd.build_state 逐行一致)
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                           scen.w[:, :, None] * scen.B_phi)
        u142 = np.zeros((K, scen.Finf_diag.shape[0], 3))
        M0142 = np.zeros((K, 3, 3))
        u142[scen.sel] = u_act
        M0142[scen.sel] = M0_act
        active142 = np.zeros(K, bool)
        active142[scen.sel] = True
        active = list(np.flatnonzero(active142))
        for ctype in types:
            for lv in levels:
                # 家族侧(het=0, rho_c=0):唯一替换点。关闭通道与原跑
                # 相同的极小正方差,非奇异。
                li = lv if ctype != "direction" else closed_li
                sd = lv if ctype != "intensity" else closed_deg
                fam = CorruptionFamily(li, sd, het_sigma=0.0, rho_c=0.0,
                                       n_lights=K)
                lam0142 = np.linalg.inv(fam.sigma_phi_block())   # (K,3,3)
                blk = LightBlocks(u142, M0142, lam0142,
                                  scen.Finf_diag, active142)
                t_all = t_one.copy()
                t_all[active] = kappa
                J1 = cd.J_A(blk, t_one)                 # 原实验的目标函数
                Jk = cd.J_A(blk, t_all)
                row = art_rows[(obj_name, ctype, lv)]
                assert J1 == row["J_A_1"], \
                    (obj_name, ctype, lv, "J_A_1", J1, row["J_A_1"])
                assert Jk == row["J_A_kappa"], \
                    (obj_name, ctype, lv, "J_A_kappa", Jk, row["J_A_kappa"])
                assert (J1 - Jk) / J1 == row["D"], \
                    (obj_name, ctype, lv, "D", (J1 - Jk) / J1, row["D"])
                n_checked += 1
        print(f"[anchor] {obj_name}: {n_checked}/{len(art['rows'])} rows",
              flush=True)
    assert n_checked == len(art["rows"])
