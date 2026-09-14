"""S0-4 · 三轴 Σ_φ 参数族构造器(家族敏感性分析,P-SIGMA-FAMILY)。

把 CorruptionGenerator 的单线参数化(intensity/direction/joint,强度/方向
方差比被 joint 钉死在 ~3283)扩展为三轴:
  - **通道比**(sig_logI vs sig_dir_deg 独立设定)——解耦 D 的隐患;
  - **逐灯异质性**(het_sigma):σ_k = σ̄·exp(s·z_k),z_k~N(0,1),固定种子;
  - **通道耦合**(rho_c):Σ_φ 的 (log I, 角度幅值) 相关系数。

退化锚点(het=0, rho_c=0, 单位通道比):每灯块 == CorruptionGenerator 的
sigma_phi_diag()(tests/test_corruption_family_degeneracy.py 钉死)。
"""

from __future__ import annotations

import numpy as np


class CorruptionFamily:
    """三轴 Σ_φ 参数族:通道比 × 异质性 × 耦合,固定种子可复现。"""

    def __init__(self, sig_logI, sig_dir_deg, het_sigma=0.0, rho_c=0.0,
                 seed=20260915, n_lights=142):
        self.sig_logI = float(sig_logI)
        self.sig_dir_deg = float(sig_dir_deg)
        self.het_sigma = float(het_sigma)
        self.rho_c = float(rho_c)
        self.n_lights = int(n_lights)
        # 逐灯乘子(对数正态,固定种子)
        rng = np.random.default_rng(seed)
        z = rng.normal(size=self.n_lights)
        self._mult = np.exp(self.het_sigma * z)

    def sigma_logI_vec(self):
        """(n,) 逐灯 log-强度标准差。"""
        return self.sig_logI * self._mult

    def sigma_rad_vec(self):
        """(n,) 逐灯方向标准差(弧度)。"""
        return np.radians(self.sig_dir_deg) * self._mult

    def _base_block(self):
        """共享的 (3,3) 单位块(通道比 + 耦合)。"""
        sI = self.sig_logI
        sR = np.radians(self.sig_dir_deg)
        r = float(np.clip(self.rho_c, -0.999, 0.999))
        # Σ_φ:对角 (sI², sR², sR²);对角项与 channel_decomposition.
        # _sigma_phi_diag 逐算子一致(li**2 / radians(sd)**2),保证退化
        # 锚点逐位复现(勿改成乘法形式)。
        # 耦合(修正版):非对角 Σ_01 = Σ_02 = r·sI·sR/√2,对角不动 ——
        # Corr(logI, 角度幅度 (θ+ψ)/√2) = r 精确成立,且 |r| ≤ 1 时块
        # 恒 PSD(任意通道比)。
        # (v1 bug:原秩一加法 r·sI·sR·eeᵀ 会改对角,大通道比下把
        #  Σ_22 = sR² + r·sI·sR/2 加成负值 → Σ_φ 非 PSD → Λ0 不定 →
        #  t=κ 处 Schur 检查崩溃;S1 首跑在 obj_11_pine 暴露。)
        S = np.diag([sI ** 2, sR ** 2, sR ** 2])
        if r != 0.0:
            c = r * sI * sR / np.sqrt(2.0)
            S[0, 1] = S[1, 0] = c
            S[0, 2] = S[2, 0] = c
        w = np.linalg.eigvalsh(0.5 * (S + S.T))
        if w.min() <= 0.0:
            raise ValueError(
                f"CorruptionFamily: Sigma_phi not PD (min eig {w.min():.3e}) "
                f"for sig_logI={sI}, sig_dir_deg={self.sig_dir_deg}, "
                f"rho_c={self.rho_c}")
        return S

    def sigma_phi_block(self):
        """(L,3,3) 逐灯 Σ_φ(het 缩放整块;全部 PD)。

        het=0 且 rho_c=0 时每块 == diag(sI², sR², sR²) ==
        CorruptionGenerator(sigma 对应档).sigma_phi_diag()。
        """
        base = self._base_block()
        if self.het_sigma == 0.0:
            return np.stack([base] * self.n_lights)
        return base[None, :, :] * (self._mult ** 2)[:, None, None]

    def summary(self):
        """实测 σ 分布摘要(供产物登记)。"""
        vI = self.sigma_logI_vec()
        vR = np.degrees(self.sigma_rad_vec())
        return dict(
            sig_logI_mean=float(vI.mean()),
            sig_logI_p16_p84=[float(np.percentile(vI, 16)),
                              float(np.percentile(vI, 84))],
            sig_dir_deg_mean=float(vR.mean()),
            sig_dir_deg_p16_p84=[float(np.percentile(vR, 16)),
                                 float(np.percentile(vR, 84))],
            ratio_variance_logI_over_dir=(
                float((self.sig_logI ** 2)
                      / max(np.radians(self.sig_dir_deg) ** 2, 1e-300))))
