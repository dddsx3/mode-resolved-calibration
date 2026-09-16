"""P-DILIGENT-LOADER · 两加载器一致性门禁(缺陷报告 2026-09-16 判据 #4)。

背景:`diligent.py`(B3/exp8r 血统,sanity 支在用)与
`diligent_oi_adapter.py`(P-DILIGENT-QUEUE / P-BASELINE 的 DQ 侧在用)
是同一数据的**两条加载路径**。适配层 v1 丢了逐灯光强归一化,使名义
重建相对 GT 偏 15.6–26.3°(正确归一化后 2.56–6.34°),而**相对端点**
(腐蚀 vs 名义,双侧同源同偏差)对它天然失明——只有绝对检查/门禁能抓。

本门禁把"两加载器不得分叉"变成机器断言(数据缺失时 skip):
同一对象的 I(掩码内、逐灯)必须满足
    adapter.I == diligent.I_norm
逐位(两者共用同一归一化算式;像素访问顺序一致)。
另断言适配层 meta 记录了归一化约定名(防第三个加载器再分叉)。
"""

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = Path("D:/data/DiLiGenT/pmsData")


def _available():
    return DATA.exists() and (DATA / "ballPNG" / "mask.png").exists()


@pytest.mark.skipif(not _available(), reason="DiLiGenT raw data absent")
def test_adapter_matches_diligent_loader():
    from calibinfo.datasets.diligent import load_object
    from calibinfo.datasets.diligent_oi_adapter import load_diligent_as_oi

    d = DATA / "ballPNG"
    ref = load_object(str(d))              # (dirs, normals_gt, I_norm, mask)
    ada = load_diligent_as_oi(d)
    I_ada = ada["images"][..., 0][:, ada["mask"]]      # (K,P) 掩码内
    assert I_ada.shape == ref["I_norm"].shape
    assert np.array_equal(I_ada, ref["I_norm"]), (
        "the two DiLiGenT loaders diverged: adapter must apply the SAME "
        "per-light intensity normalization as diligent.py:43 "
        "(`gray / light_intensities[:, 0]`)")
    assert np.array_equal(ada["light_directions"], ref["dirs"])


@pytest.mark.skipif(not _available(), reason="DiLiGenT raw data absent")
def test_adapter_records_the_convention():
    from calibinfo.datasets.diligent_oi_adapter import load_diligent_as_oi
    ada = load_diligent_as_oi(DATA / "ballPNG")
    assert "normalization" in ada["meta"]
    assert "light_intensities" in ada["meta"]["normalization"]
