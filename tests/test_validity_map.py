"""P-VALIDITY-MAP · artifact pin tests (E2/M2).

The validity map decomposes "where is the Fisher linearization valid"
into ordering endpoints (Spearman / Kendall sign agreement — robust
across curvature regimes) vs the magnitude ratio (degrades with the
curvature metric). Read-only join of the corrected arm-D cells and the
linearization-radius curvature; no re-run.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/magnitude/validity_map.json"


def test_validity_map_structure_and_findings():
    if not ART.exists():
        pytest.skip("artifact not committed")
    j = json.loads(ART.read_text(encoding="utf-8"))
    assert j["analysis_status"] == "validity_map_v1"
    assert j["n_cells"] == 66 and len(j["cells"]) == 66
    assert len(j["levels"]) == 6
    # 逐 cell 四端点都在合法域
    for c in j["cells"]:
        assert -1.0 <= c["spearman"] <= 1.0
        assert -1.0 <= c["kendall_sign_agreement"] <= 1.0
        assert c["ratio_median"] > 0
        assert c["excess_over_scaling"] > 0
    # 核心发现:排序/符号效度对曲率稳健(每箱 Spearman >= 0.8, Kendall >= 0.7),
    # 幅值比随曲率单调恶化(首箱 < 末箱,量级增长)
    bins = j["by_excess_bin"]
    assert len(bins) == 4
    for b in bins:
        assert b["spearman_median"] >= 0.8
        assert b["kendall_median"] >= 0.7
    assert bins[0]["ratio_median_of_medians"] < bins[-1]["ratio_median_of_medians"]
    assert bins[-1]["ratio_median_of_medians"] / bins[0]["ratio_median_of_medians"] > 5
    # manifest:来源产物 hash 有记录
    assert j["manifest"]["rows_sha256"] and j["manifest"]["radius_sha256"]
    assert j["manifest"]["git_sha"]
