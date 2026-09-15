"""S0-3 · 层 1(CI-safe,无原始数据):Σ_φ 换源精确性断言。

验收报告 §5 的结构性修复:S0-3 的端到端逐位锚点需要原始数据 +
生产 numpy,**任何他人环境都会失败**(实测:numpy 2.5.3 下 1/24 复现
失败,最坏 3.1e-11 相对偏差——浮点条件数放大,不是代码回归)。
而"S0-3 想证明的东西"——Σ_φ 从 CorruptionGenerator 换源到
CorruptionFamily 是**精确的**——本身不依赖场景装配,可以在无数据
环境下逐位钉死:

全部 24 个 (通道, level) 组合(3 通道 × 冻结 8 档 level,含关闭通道
极小方差约定):
  Σ:  channel_decomposition._sigma_phi_diag(ctype, lv, 1e-3, 1e-6)
      == CorruptionFamily(li, sd, het=0, rho=0).sigma_phi_block()[0]
      (np.array_equal,逐位)
  Λ₀: np.linalg.inv(上述两者) 逐位相同

端到端(J_A 端点)的逐位比对保留在
tests/test_corruption_family_degeneracy.py 的层 2(本机,rtol=1e-9):
逐位只在生产环境(numpy 2.4.1,见 REPRODUCIBILITY)成立。
"""

import numpy as np

from calibinfo.models.corruption_family import CorruptionFamily

LEVELS = [0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0]
CLOSED_DEG = 0.001       # frozen closed-channel convention (both files)
CLOSED_LI = 1e-06


def _cd_sigma(ctype, lv):
    """channel_decomposition._sigma_phi_diag 的逐算子镜像
    (rad = radians(lv if ctype != 'intensity' else closed_deg);
     li  = lv if ctype != 'direction' else closed_li)。"""
    rad = np.radians(lv if ctype != "intensity" else CLOSED_DEG)
    li = lv if ctype != "direction" else CLOSED_LI
    return np.diag([li ** 2, rad ** 2, rad ** 2])


def test_sigma_source_exchange_bit_exact_all_24():
    """Σ 与 Λ₀:24 组合全部逐位相同(无数据,CI 可跑)。"""
    for ctype in ("joint", "intensity", "direction"):
        for lv in LEVELS:
            cd_sig = _cd_sigma(ctype, lv)
            li = lv if ctype != "direction" else CLOSED_LI
            sd = lv if ctype != "intensity" else CLOSED_DEG
            fam_sig = CorruptionFamily(li, sd, het_sigma=0.0, rho_c=0.0,
                                       n_lights=2).sigma_phi_block()[0]
            assert np.array_equal(cd_sig, fam_sig), (ctype, lv, "Sigma")
            assert np.array_equal(np.linalg.inv(cd_sig),
                                  np.linalg.inv(fam_sig)), (ctype, lv,
                                                            "Lambda0")


def test_lambda0_batched_equals_single_bit_exact():
    """批量 (K,3,3) inv == 单次 (3,3) inv(S1 用批量路径的精确性)。"""
    K = 142
    for ctype in ("joint", "direction"):
        lv = 0.5
        cd_sig = _cd_sigma(ctype, lv)
        fam = CorruptionFamily(cd_sig[0, 0] ** 0.5,
                               np.degrees(cd_sig[1, 1] ** 0.5),
                               het_sigma=0.0, rho_c=0.0, n_lights=K)
        blocks = fam.sigma_phi_block()
        single = np.linalg.inv(blocks[0])
        batched = np.linalg.inv(blocks)
        assert np.array_equal(batched,
                              np.broadcast_to(single, (K, 3, 3))), ctype
