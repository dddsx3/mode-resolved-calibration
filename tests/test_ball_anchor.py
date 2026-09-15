"""P-BALL-ANCHOR artifact gates (TCI gap 1: physical anchoring).

CI-safe: reads only results/openillumination/ball_anchor.json. Pins:

1. Sanity guards: RMS direction error <= 5 deg, RMS relative intensity
   error <= 10% (the preregistered implementation-correctness guards).
2. The measured anchor values (sig_logI 0.0159, sig_dir_deg 2.9644) and
   the per-channel variance ratio (direction ~12x intensity -- the
   OPPOSITE of the joint parameterization's forced 3283).
3. Outcome consistency: D-flipped with median direction share 36.0%
   (>= 2% rule), per-object range 4.75%..71.40%.
4. Manifest integrity (config sha256 vs committed config).
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/ball_anchor.json"
CFG = REPO / "configs/ball_anchor.yaml"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_ball_anchor_guards():
    art = _load()
    assert art["gate"] == "P-BALL-ANCHOR"
    g = art["gt_comparison"]
    assert g["rms_direction_error_deg"] <= 5.0
    assert g["max_direction_error_deg"] <= 10.0
    assert g["rms_relative_intensity_error"] <= 0.10


def test_ball_anchor_values():
    art = _load()
    a = art["measured_anchor"]
    assert round(a["sig_logI"], 4) == 0.0159
    assert round(a["sig_dir_deg"], 4) == 2.9644
    # 方向方差 ~12x 强度方差(与 joint 参数化的 3283 相反;同单位:logI
    # 与弧度方向)
    import numpy as np
    ratio = (np.radians(a["sig_dir_deg"]) / a["sig_logI"]) ** 2
    assert 10.0 < ratio < 15.0
    # 96 灯逐灯表
    assert len(art["per_light"]["direction_error_deg"]) == 96
    assert len(art["per_light"]["logI_deviation"]) == 96


def test_ball_anchor_outcome_consistency():
    art = _load()
    sh = art["direction_share_at_anchor"]
    med = sh["median"]
    expected = "D-confirmed" if med < 0.02 else "D-flipped"
    assert art["outcome"] == expected
    assert med == 0.360024
    vals = sorted(sh["per_object"].values())
    assert round(vals[0], 4) == 0.0475          # obj_04_dolphin
    assert round(vals[-1], 4) == 0.714          # obj_11_pine
    # 中位自洽(11 对象 -> 第 6 顺序统计量)
    import numpy as np
    assert abs(med - float(np.median(list(sh["per_object"].values())))) < 1e-6


def test_ball_anchor_manifest():
    art = _load()
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        CFG.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
    # 族落位字段存在且最近点是 S1 网格的合法点
    fp = art["family_placement"]
    assert fp["nearest_grid_point"]["sig_logI"] == 0.05
    assert fp["nearest_grid_point"]["sig_dir_deg"] == 1.0
