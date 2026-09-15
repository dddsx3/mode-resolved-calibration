"""P-DILIGENT-QUEUE artifact gates (TCI gap 2: second dataset).

CI-safe: reads only results/diligent/diligent_queue.json. Pins:

1. Grid completeness: 10 objects x 3 channels x 8 levels = 240 rows.
2. Outcome consistency: transfer-partial (radius transfers, share does
   not transfer as a dataset-level flip -- it splits by object).
3. The radius table: median 1.0 (identical to OI), all objects in
   [0.75, 1.5] -- the linearization envelope transfers.
4. The direction-channel table: max 51.197% (vs OI's 1.68%) -- the
   direction channel carries far more value on DiLiGenT.
5. Degenerate-share annotation: negative/absurd shares occur exactly
   where D(anchor) ~ 0 (documented in the artifact note); the pot1/pot2
   86% values are the genuine direction-coupled cases.
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
    art = _load()
    med_share = art["ball_anchor_share"]["median"]
    r2 = art["linearization_radius"]["radius_2x"][
        "median_over_crossed_subset"]
    share_ok = med_share >= 0.02
    radius_ok = r2 is not None and 0.25 <= r2 <= 4.0
    expected = ("transfer-confirmed" if (share_ok and radius_ok)
                else "transfer-partial" if (share_ok or radius_ok)
                else "transfer-failed")
    assert art["outcome"] == expected
    # 实测:半径过、份额不过 → partial
    assert art["outcome"] == "transfer-partial"
    assert radius_ok and not share_ok
    assert round(med_share, 4) == -0.0067
    assert r2 == 1.0


def test_radius_transfers():
    art = _load()
    per_obj = art["linearization_radius"]["radius_2x"]["per_object"]
    assert len(per_obj) == 10
    assert all(0.75 <= v <= 1.5 for v in per_obj.values())
    assert art["linearization_radius"]["radius_2x"][
        "median_over_crossed_subset"] == 1.0
    assert art["linearization_radius"]["radius_10x"][
        "median_over_crossed_subset"] == 1.5


def test_direction_channel_table():
    art = _load()
    cd = art["channel_decomposition"]
    assert cd["direction_max_pct"] == 51.197
    # 方向通道在 DiLiGenT 上远大于 OI 的 1.68%(通道条件性判据的
    # 第二数据集实例)
    assert cd["direction_max_pct"] > 30.0


def test_ball_anchor_share_split():
    art = _load()
    sh = art["ball_anchor_share"]["per_object"]
    # 真方向耦合:pot1/pot2 ~86%
    assert round(sh["pot1PNG"], 4) == 0.8559     # 0.855874
    assert round(sh["pot2PNG"], 4) == 0.8608     # 0.860849
    # 近零对象
    assert abs(sh["ballPNG"]) < 0.03
    assert abs(sh["bearPNG"]) < 0.03
    assert abs(sh["buddhaPNG"]) < 0.01
    # 退化(D(anchor)≈0,分母无意义)对象在 artifact 登记
    assert art["ball_anchor_share"]["degenerate_objects"] == ["readingPNG"]
    assert "degenerate_note" in art["ball_anchor_share"]
    # harvest 的 -40% 介于两者之间(D(anchor) 有限但小)——分母量级断言,
    # 防误读为"负方向贡献"
    d_harvest = [r["D"] for r in art["channel_decomposition"]["rows"]
                 if r["object"] == "harvestPNG" and r["channel"] == "joint"
                 and r["level"] == 0.05][0]
    assert 0.5 < d_harvest < 0.6


def test_manifest_integrity():
    art = _load()
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        CFG.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
