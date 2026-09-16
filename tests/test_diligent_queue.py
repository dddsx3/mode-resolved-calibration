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
    assert round(med_share, 6) == -5.2e-05
    assert r2 == 1.0


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
