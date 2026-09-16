"""P-DQ-FEASIBLE artifact gates (acceptance 9515039 section 4, P1).

CI-safe: reads only results/openillumination/decision_quality_feasible.json
and the committed DQ artifact it derives from. Pins:

1. Degeneracy: k = 57/85/114 are flagged degenerate (informed and
   active48-random prefixes select the same lights; endpoints coincide
   bit-exactly in all 88 cells) and infeasible (k > |active| = 48).
2. The full-grid dAUC reproduces the frozen artifact's bootstrap_dAUC
   (vs_randomA48, ang_mean_deg) within rounding (store rounding is
   1e-6; recomputation runs at full precision).
3. The feasible-interval dAUC values (the corrected headline numbers).
4. The dilution ratios (2.0-2.8x) and the feasible A48/U ratio
   (0.27-0.51, i.e. NOT an order of magnitude).
5. Source-artifact sha256 anchor.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/decision_quality_feasible.json"
SRC = REPO / "results/openillumination/decision_quality.json"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_degenerate_budgets_flagged():
    art = _load()
    assert art["n_active"] == 48
    deg = {k: v["degenerate"] for k, v in art["degenerate_budgets"].items()}
    assert deg == {"14": False, "28": False, "57": True, "85": True,
                   "114": True}
    for k in ("57", "85", "114"):
        d = art["degenerate_budgets"][k]
        assert d["n_cells"] == 88 and d["n_total"] == 88
        assert d["feasible"] is False
    for k in ("14", "28"):
        assert art["degenerate_budgets"][k]["feasible"] is True


def test_full_grid_matches_frozen():
    art = _load()
    frz = json.loads(SRC.read_text(encoding="utf-8"))["bootstrap_dAUC"]
    for key, v in art["full_grid_dAUC_vs_randomA48"].items():
        u, rg = key.split("|")
        fk = f"{u}|{rg}|ang_mean_deg|vs_randomA48"
        assert abs(v["median_dAUC"] - frz[fk]["median_dAUC"]) < 5e-6, key


def test_feasible_values_pinned():
    art = _load()
    fe = art["feasible_dAUC_vs_randomA48"]
    pins = {"mode_aware|10": -0.712039, "e_opt|10": -1.150885,
            "a_opt|10": -1.14416, "d_opt|10": -0.400189,
            "mode_aware|100": -0.694817, "e_opt|100": -1.404892,
            "a_opt|100": -1.221627, "d_opt|100": -0.384316}
    for key, want in pins.items():
        assert fe[key]["median_dAUC"] == want, key
        # 可行区间每个 cell 的 CI 都排除 0
        assert fe[key]["ci95"][1] < 0.0, key
        assert fe[key]["n_objects"] == 44          # 44 (object, level) 单元


def test_dilution_and_ratio():
    art = _load()
    ratios = [v["ratio"] for v in art["dilution"].values()]
    assert min(ratios) >= 2.0 and max(ratios) <= 2.9
    r = art["feasible_ratio_active48_over_universeRandom"]
    assert min(r.values()) >= 0.25 and max(r.values()) <= 0.55
    # "order of magnitude" 被证伪:8 个 cell 的比值全部 < 0.55
    assert all(v < 0.6 for v in r.values())
    assert "diluted" in art["reading"]


def test_source_anchor():
    art = _load()
    assert art["gate"] == "P-DQ-FEASIBLE"
    assert art["source_manifest_sha256"] == hashlib.sha256(
        SRC.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
