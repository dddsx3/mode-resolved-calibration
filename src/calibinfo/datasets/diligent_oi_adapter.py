"""DILIGENT-QUEUE · DiLiGenT → OpenIllumination 对象格式适配层(缺口 2)。

把 DiLiGenT pmsData 的单对象包装成 NominalScene 期望的 dict 格式:
    images (96,H,W,3) float[0,1]、mask (H,W) bool、
    light_directions (96,3) 单位向量。

口径(与 src/calibinfo/datasets/diligent.py 的 B3/exp8r 血统一致,
除了**不除 GT 强度**——这里要的是 nominal 场景,不是已标定残差):
  - 灰度 = PIL convert("L") / 255;
  - mask>128;
  - light_directions.txt 直接读取(已单位化)。
注意:DiLiGenT 光向是"从表面指向灯"的约定(与 OI 的 −pos/‖pos‖ 相同
朝向口径——两者都是指向灯的单位向量);ball_anchor 的 GT 比对已经
按此口径验证(夹角余弦直接点积)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_diligent_as_oi(d):
    """DiLiGenT 单对象目录 → OI 格式 dict(NominalScene 直接可用)。"""
    d = Path(d)
    dirs = np.loadtxt(d / "light_directions.txt")            # (96,3) 单位
    mask = np.array(Image.open(d / "mask.png").convert("L")) > 128
    K = dirs.shape[0]
    imgs = np.stack([np.array(Image.open(str(d / f"{i:03d}.png"))
                              .convert("L")).astype(float) / 255.0
                     for i in range(1, K + 1)])              # (K,H,W)
    return dict(images=imgs[..., None].repeat(3, axis=-1),   # (K,H,W,3)
                mask=mask,
                light_directions=dirs,
                meta=dict(dataset="DiLiGenT", object=d.name, n_lights=K))
