"""P-DILIGENT-QUEUE artifact gates (TCI gap 2: second dataset).

CI-safe: reads only results/diligent/diligent_queue.json. Pins:

1. Grid completeness: 10 objects x 3 channels x 8 levels = 240 rows.
2. Outcome consistency (v1.1 reading, loader-corrected): the outcome is
   recomputed from the stored tables and must match; the measured
   outcome is transfer-confirmed -- the intensity-dominance channel
   split transfers (direction max 0.20%, intensity reproduces joint to
   <0.05 pp at every level) and the radius median is 1.0. The
   pre-correction "channel split does not transfer" reading (direction
   max 51.197%, pot1/pot2 ~86% anchor shares) was a loader artifact
   (missing per-light intensity normalization) and is withdrawn --
   see results/diligent/provenance/loader_normalization_fix.json.
3. The radius table: median 1.0 (identical to OI), all objects in
   [0.75, 1.5]; radius_10x = 1.5 on 10/10 objects.
4. The direction-channel table: max 0.203% -- same side as OI
   (intensity-dominant on both cohorts).
5. The ball-anchor share is reported as the C10/C12 anchor test
   (object-conditional by construction): median ~0, cat 22.1%/pot1
   29.5% the high objects; degenerate denominators annotated
   (harvest, reading).
6. Manifest integrity.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/diligent/diligent_queue.json"
CFG = REPO / "configs/diligent_queue.yaml"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_grid_completeness():
    art = _load()
    assert art["gate"] == "P-DILIGENT-QUEUE"
    assert art["n_objects"] == 10
    rows = art["channel_decomposition"]["rows"]
    assert len(rows) == 10 * 3 * 8
    assert {r["channel"] for r in rows} == {"joint", "intensity",
                                            "direction"}


def test_outcome_consistency():
    """v1.1 判读(N3):结局必须与修正后的通道分裂迁移判据作用于存档
    表的结果一致——两队列同为强度主导(direction max ≤5%,intensity
    复现 joint <0.5pp)且半径中位在 OI 值 4 倍内 ⇒ transfer-confirmed;
    旧份额判据(≥2%)保留为参照,不是读数基础。"""
    art = _load()
    cd = art["channel_decomposition"]
    byc = cd["by_channel"]
    med_share = art["ball_anchor_share"]["median"]
    r2 = art["linearization_radius"]["radius_2x"][
        "median_over_crossed_subset"]
    radius_ok = r2 is not None and 0.25 <= r2 <= 4.0
    direction_max = cd["direction_max_pct"] / 100.0
    eq = all(abs(byc["joint"][lv]["median_pct"]
                 - byc["intensity"][lv]["median_pct"]) / 100.0 < 5e-3
             for lv in byc["joint"])
    expected = ("transfer-confirmed" if (eq and direction_max <= 0.05
                                         and radius_ok)
                else "transfer-partial (radius transfers; channel split "
                     "does not)" if radius_ok else "transfer-failed")
    assert art["outcome"] == expected
    # 通道分裂迁移:方向 max 0.203%,intensity/joint 逐 level <0.05pp
    assert cd["direction_max_pct"] == 0.203
    for lv in byc["joint"]:
        assert abs(byc["joint"][lv]["median_pct"]
                   - byc["intensity"][lv]["median_pct"]) < 0.05
    assert radius_ok
    # 锚点份额(独立字段,按物体条件性):中位 ≈0
    assert round(med_share, 6) == -5.2e-05


def test_radius_transfers():
    """半径跨系统复现(loader 修正后加固:10x 全 10/10 = 1.5)。"""
    art = _load()
    per_obj = art["linearization_radius"]["radius_2x"]["per_object"]
    assert len(per_obj) == 10
    assert all(0.75 <= v <= 1.5 for v in per_obj.values())
    assert art["linearization_radius"]["radius_2x"][
        "median_over_crossed_subset"] == 1.0
    assert art["linearization_radius"]["radius_10x"][
        "median_over_crossed_subset"] == 1.5
    assert all(v == 1.5 for v in
               art["linearization_radius"]["radius_10x"]
               ["per_object"].values())


def test_direction_channel_table():
    """loader 修正后:DiLiGenT 方向通道很小(与 OI 同侧)——
    修正前的 51.2% 是丢光强归一化的伪影(见 P-DILIGENT-LOADER-FIX)。"""
    art = _load()
    cd = art["channel_decomposition"]
    assert cd["direction_max_pct"] == 0.203
    # 强度通道复现 joint(与 OI 相同的强度主导)
    byc = cd["by_channel"]
    for lv in byc["joint"]:
        assert abs(byc["joint"][lv]["median_pct"]
                   - byc["intensity"][lv]["median_pct"]) < 0.05


def test_ball_anchor_share_split():
    """loader 修正后:锚点份额按物体分裂但整体近零(cat 22.1%、pot1 29.5%,
    其余 ~0);修正前的 pot1/pot2 ~86% 是伪影。"""
    art = _load()
    sh = art["ball_anchor_share"]["per_object"]
    assert round(sh["catPNG"], 4) == 0.2208
    assert round(sh["pot1PNG"], 4) == 0.2946
    assert abs(sh["pot2PNG"]) < 0.01
    assert abs(sh["ballPNG"]) < 0.001
    assert abs(sh["bearPNG"]) < 0.001
    assert art["ball_anchor_share"]["degenerate_objects"] == [
        "harvestPNG", "readingPNG"]
    assert round(art["ball_anchor_share"]["median"], 6) == -5.2e-05


def test_manifest_integrity():
    art = _load()
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        CFG.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
