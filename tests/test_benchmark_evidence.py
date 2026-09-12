"""Benchmark-evidence recomputation gate (N10-N12).

Independently recomputes the three post-adjudication allocation headline
numbers from the frozen artifacts under results/ (post tag
manuscript-evidence-v1 these artifacts and this file may not change --
see the manuscript-evidence drift gate in .github/workflows/ci.yml):

  N10  post-hoc paired policy comparison: mode-aware minus each classical
       baseline (E/A/D-opt greedy) has positive median and CI excluding 0
       in both regimes -- classical achieves modestly lower AUC;
  N11  random_active48 attribution control: with random restricted to the
       48 Fisher-active lights, the mode-aware advantage disappears
       (Delta ~ +0.014/+0.019, CI spanning 0, 3/11 objects);
  N12  active-set effect: random48 - random_full = -0.198/-0.342 with CIs
       excluding 0, 11/11 improved.

Self-contained: reads only committed CSV/JSON artifacts; no experiment
pipeline imports. N1-N9 live in tests/test_reproduction.py (frozen at
tag science-closed); N10-N12 here (frozen at tag manuscript-evidence-v1).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
ALLOC = R / "openillumination/allocation"
BUDGETS = (0.1, 0.2, 0.4, 0.6, 0.8)


def _allocation_uos():
    """{(object, regime, policy, budget): E_osb} from the frozen uos_table."""
    out = {}
    with open(ALLOC / "uos_table.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["object"], int(r["regime"]), r["policy"],
                 float(r["budget"]))] = float(r["E_osb"])
    return out


def _auc95(table, obj, regime, policy):
    """Normalized trapezoidal AUC of the budget curve (preregistered endpoint)."""
    es = np.array([table[(obj, regime, policy, b)] for b in BUDGETS])
    bs = np.array(BUDGETS)
    return float(np.sum((es[:-1] + es[1:]) / 2.0 * np.diff(bs)) / (0.8 - 0.1))


def _paired_boot(d, seed=20260910, n_boot=10000):
    """Median + percentile CI of a paired object-level difference with the
    frozen bootstrap (shared resamples, same draw order as a5_analyze)."""
    diffs = np.array(list(d.values()))
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = [float(np.median(diffs[rng.integers(0, n, n)]))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.median(diffs)), float(lo), float(hi)


def test_N10_pairwise_classical_lower_auc():
    """N10: mode-aware minus each classical baseline: positive medians
    (+0.019..+0.027) and CIs excluding 0, both regimes; cross-checked
    against the committed allocation_policy_pairwise.csv."""
    table = _allocation_uos()
    objs = sorted({o for (o, r, _p, _b) in table})
    assert len(objs) == 11
    med_all = []
    with open(ALLOC / "allocation_policy_pairwise.csv", newline="",
              encoding="utf-8") as f:
        pw = {(r["regime"], r["comparison"]): r
              for r in csv.DictReader(f)}
    for regime in (10, 100):
        for pol in ("e_opt", "a_opt", "d_opt"):
            d = {o: _auc95(table, o, regime, "mode_aware")
                 - _auc95(table, o, regime, pol) for o in objs}
            med, lo, hi = _paired_boot(d)
            assert med > 0 and lo > 0            # classical better, stable
            med_all.append(med)
            row = pw[(str(regime), f"mode_aware_minus_{pol}")]
            assert row["analysis_status"] == "posthoc_paired_comparison"
            assert float(row["median_delta"]) == pytest.approx(med, abs=1e-5)
            assert float(row["ci_lo"]) == pytest.approx(lo, abs=1e-5)
            assert float(row["ci_hi"]) == pytest.approx(hi, abs=1e-5)
    assert 0.0185 <= min(med_all) <= max(med_all) <= 0.0275


def test_N11_random48_control_ci_spans_zero():
    """N11: with random restricted to the 48 Fisher-active lights, the
    mode-aware advantage disappears (Delta ~ +0.014/+0.019, CI spanning 0)."""
    s = json.loads((ALLOC / "allocation_random48_summary.json").read_text(
        encoding="utf-8"))
    assert s["analysis_status"] == "posthoc_attribution_control"
    for regime, med_ref in (("10", 0.0141), ("100", 0.0187)):
        v = s["regimes"][regime]["mode_minus_random_active48"]
        assert v["median"] == pytest.approx(med_ref, abs=5e-4)
        assert v["ci95"][0] < 0 < v["ci95"][1]
        assert v["improved_negative"] == 3


def test_N12_active_set_attribution():
    """N12: the active-set effect itself (random48 - random_full) is large
    and significant: medians -0.198 / -0.342, CIs excluding 0, 11/11."""
    s = json.loads((ALLOC / "allocation_random48_summary.json").read_text(
        encoding="utf-8"))
    for regime, med_ref in (("10", -0.198), ("100", -0.342)):
        v = s["regimes"][regime]["random_active48_minus_random_full"]
        assert v["median"] == pytest.approx(med_ref, abs=5e-4)
        assert v["ci95"][1] < 0
        assert v["improved_negative"] == 11
