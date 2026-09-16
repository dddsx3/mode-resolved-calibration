"""P-ABLATION-FEASIBLE artifact gates (acceptance 665d01e section 4, N-1).

CI-safe: reads only results/openillumination/active_set_ablation_feasible.json
and its committed source. Pins:

1. Budget-axis degeneracy marking: k=48 = |active| is flagged as
   ordering-sub-term degenerate (11/11 rows with mode_ordering_gap == 0
   -- definitionally, both prefixes select all lights), while the
   dilution term stays valid there and would only degenerate for k>48.
2. The two calibers of the C8 ratio: with k=48 ~1364x (full grid), without
   ~1001x -- both order-of-magnitude.
3. Source-artifact sha256 anchor.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/active_set_ablation_feasible.json"
SRC = REPO / "results/openillumination/active_set_ablation.json"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_degeneracy_marking():
    art = _load()
    assert art["gate"] == "P-ABLATION-FEASIBLE"
    assert art["n_active"] == 48
    d = art["degenerate_budgets"]
    # k<48:正常
    for k in ("5", "10", "14", "28"):
        assert d[k]["ordering_subterm_degenerate"] is False
        assert d[k]["dilution_term_degenerate"] is False
    # k=48:ordering 子项定义性零(11/11),dilution 仍有效
    k48 = d["48"]
    assert k48["n_rows"] == 11
    assert k48["n_ordering_gap_zero"] == 11
    assert k48["ordering_subterm_degenerate"] is True
    assert k48["dilution_term_degenerate"] is False
    assert "definitionally" in k48["note"]


def test_two_calibers():
    art = _load()
    w = art["ratio_with_k48"]
    wo = art["ratio_without_k48"]
    assert round(w["median_ordering_gap"], 10) == 4.623e-05 or \
        abs(w["median_ordering_gap"] - 4.623e-05) < 1e-9
    assert round(w["as_multiple"], 1) == 1364.3
    assert round(wo["as_multiple"], 1) == 1001.2
    # 两种口径都是数量级差(结论不变)
    assert w["as_multiple"] > 1000 and wo["as_multiple"] > 1000
    assert "k=48" in art["caliber_note"]


def test_source_anchor():
    art = _load()
    assert art["source_manifest_sha256"] == hashlib.sha256(
        SRC.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
