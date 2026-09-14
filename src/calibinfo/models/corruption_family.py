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
        # Σ_φ:对角 (sI², sR², sR²);耦合在 (log I, θ-幅值) 之间。
        # 角度有两个自由度,耦合作用在抽取侧(raw_innovations_family 的
        # angles 总幅值);矩阵侧用秩一形式:非对角 = r·sI·sR·e₁e₂ᵀ。
        # 对角项与 channel_decomposition._sigma_phi_diag 逐算子一致(li**2 /
        # radians(sd)**2),保证退化锚点逐位复现(勿改成乘法形式)。
        S = np.diag([sI ** 2, sR ** 2, sR ** 2])
        if r != 0.0:
            e = np.zeros((3, 2))
            e[0, 0] = 1.0
            e[1:, 1] = 1.0 / np.sqrt(2.0)     # 角度幅值方向(θ,ψ 等权)
            S = S + r * sI * sR * (e @ e.T)
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
