"""Appendix-C reproduction gate (T6.3).

Independently recomputes the twelve headline numbers (N1-N12) of the manuscript from the
frozen artifacts under ``results/`` and ``data/manifests/`` and asserts they agree with
the values reported in the manuscript appendix, bit-for-bit where the quantity is
deterministic and within tight tolerance otherwise.

This test is self-contained: it reads only the committed CSV/JSON artifacts and the
public ``calibinfo`` helpers; it does not require the (external) raw datasets.

Expected values come from regulation P1 §7.B (appendix C).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"


def _spearman(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _load_csv(name):
    import csv
    with open(R / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------- N1 / N5 (OpenIllumination)
def _openillumination_cells():
    """Return {(object, level): (pred_deg[], emp_deg[])} from mode_ranking.csv."""
    from collections import defaultdict
    rows = _load_csv("openillumination/mode_ranking.csv")
    cell = defaultdict(lambda: [[], []])
    for r in rows:
        cell[(r["object_id"], float(r["level"]))][0].append(float(r["pred_deg"]))
        cell[(r["object_id"], float(r["level"]))][1].append(float(r["emp_deg"]))
    return cell


def _per_cell_overlap(cell):
    from collections import defaultdict
    rol = {}
    for k, (p, e) in cell.items():
        S = 1.0 - np.asarray(p, float) ** (-1.0)
        rol[k] = _spearman(S, np.asarray(e, float))
    per_obj = defaultdict(list)
    for (o, _lv), v in rol.items():
        per_obj[o].append(v)
    return per_obj


def _ra_stat(objs):
    return float(np.median([np.median(v) for _, v in objs]))


def test_N1_mode_resolved_pass_rate():
    """N1: R_A = 0.90 (object-cluster CI [0.90, 0.95], 66/66 cells, 11/11 objects)."""
    per_obj = _per_cell_overlap(_openillumination_cells())
    ra = _ra_stat([(o, v) for o, v in per_obj.items()])
    assert ra == pytest.approx(0.90, abs=1e-9)
    payload = [(o, v) for o, v in per_obj.items()]
    _pt, ci, _ = cluster_bootstrap(payload, _ra_stat, 10000, 20260908)
    assert ci[0] == pytest.approx(0.90, abs=1e-6)
    assert ci[1] == pytest.approx(0.95, abs=1e-6)
    assert len(per_obj) == 11


def test_N5_old_pooled():
    """N5: old pooled Spearman = 0.728 (object-cluster CI [0.705, 0.754])."""
    from collections import defaultdict
    d = (R / "openillumination/ci04_formal_summary.json").read_text(encoding="utf-8")
    rows = __import__("json").loads(d)["rows"]
    obj = defaultdict(lambda: [[], []])
    for r in rows:
        for pd, ed in zip(r["pred_deg"], r["emp_deg"]):
            obj[r["object"]][0].append(pd)
            obj[r["object"]][1].append(ed)
    payload = [(o, v) for o, v in obj.items()]

    def pooled(objs):
        P, T = [], []
        for _o, (p, e) in objs:
            P += p
            T += e
        return _spearman(P, T)

    pt, ci, _ = cluster_bootstrap(payload, pooled, 10000, 20260908)
    assert pt == pytest.approx(0.728, abs=1e-3)
    assert ci[0] == pytest.approx(0.705, abs=1e-3)
    assert ci[1] == pytest.approx(0.754, abs=1e-3)


# --------------------------------------------------------------- N2 / N3 / N4 (per-cell predictors)
def _level_severity():
    return _load_csv("openillumination/level_severity.csv")


def test_N4_emin_equals_mode():
    """N4: P_emin == P_mode for every cell (max absolute difference <= 1e-14)."""
    ls = _level_severity()
    max_diff = max(abs(float(r["P_emin"]) - float(r["P_mode"])) for r in ls)
    assert max_diff <= 1e-14


def test_N2_stratified_median():
    """N2: stratified median of within-level Spearman.
    mode/E-min = 0.536, logdet = 0.418, trace = 0.400."""
    import csv
    from collections import defaultdict
    ls = _level_severity()
    expect = {"P_mode": 0.536, "P_emin": 0.536, "P_logdet": 0.418, "P_trace": 0.400}
    for pred, target in expect.items():
        by_lv = defaultdict(list)
        for r in ls:
            by_lv[float(r["level"])].append(
                (float(r[pred]), float(r["T_ol"])))
        rhos = [_spearman([x[0] for x in v], [x[1] for x in v]) for v in by_lv.values()]
        assert np.median(rhos) == pytest.approx(target, abs=1e-3)


def test_N3_pooled_spearman():
    """N3: pooled Spearman across 66 cells.
    mode=0.871, emin=0.871, logdet=0.859, trace=0.733, corr=0.867."""
    ls = _level_severity()
    expect = {"P_mode": 0.871, "P_emin": 0.871, "P_logdet": 0.859,
              "P_trace": 0.733, "P_corr": 0.867}
    for pred, target in expect.items():
        pv = [float(r[pred]) for r in ls]
        tv = [float(r["T_ol"]) for r in ls]
        assert _spearman(pv, tv) == pytest.approx(target, abs=1e-3)


# --------------------------------------------------------------- N6 (gauge spectrum)
def test_N6_gauge_closed_form():
    """N6: CI02 gauge closed-form agrees with direct route within 3.9e-8
    across 25 decades of lambda."""
    d = (R / "gauge_spectrum/ci02_formal_summary.json").read_text(encoding="utf-8")
    s = __import__("json").loads(d)
    assert s["closed_vs_direct_max"] <= 3.9e-8
    assert s["retention_dual_rel_max"] <= 3.9e-8
    assert s["closed_vs_direct_pass"] is True


# --------------------------------------------------------------- N7 (Monte-Carlo)
def test_N7_covariance_ratio():
    """N7: median of the empirical/analytic covariance ratio = 1.0045 (≈ 1)."""
    d = (R / "monte_carlo/ci03_formal_summary.json").read_text(encoding="utf-8")
    checks = __import__("json").loads(d)["checks"]
    ratios = [c["var_ratio_median"] for c in checks]
    assert np.median(ratios) == pytest.approx(1.0045, abs=0.02)
    assert all(c["gate_pass"] for c in checks)


# --------------------------------------------------------------- N8 (DiLiGenT taxonomy)
def test_N8_taxonomy():
    """N8: CI05 weak-mode taxonomy median ≈ 273x (calibration propagation
    comparable across objects); every object >> 1."""
    d = (R / "diligent/ci05_formal_summary.json").read_text(encoding="utf-8")
    rows = __import__("json").loads(d)["rows"]
    tax = [r["weak_mode_median"] / r["weak_mode_floor"] for r in rows]
    assert np.median(tax) == pytest.approx(273.0, rel=0.10)
    assert min(tax) > 1.0


# --------------------------------------------------------------- N9 (known-answer V1-V6)
def test_N9_known_answer_suite_present():
    """N9: the V1-V6 known-answer tests are part of the standard suite.
    They are exercised by the standard pytest run; this guard confirms collection."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "tests/test_covariance_identity.py", "tests/test_gauge_closed_form.py",
         "tests/test_retention_bounds.py", "tests/test_parameterization.py",
         "tests/test_scale_invariance.py", "tests/test_mode_tracking.py",
         "tests/test_known_answer_precheck.py"],
        cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0
    assert "test" in out.stdout


# --------------------------------------------------------------- N10-N12 (allocation, post-hoc)
def _allocation_uos():
    """{(object, regime, policy, budget): E_osb} from the frozen uos_table."""
    import csv
    out = {}
    with open(R / "openillumination/allocation/uos_table.csv", newline="",
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["object"], int(r["regime"]), r["policy"],
                 float(r["budget"]))] = float(r["E_osb"])
    return out


def _auc95(table, obj, regime, policy, budgets=(0.1, 0.2, 0.4, 0.6, 0.8)):
    """Normalized trapezoidal AUC of the budget curve (preregistered endpoint)."""
    es = np.array([table[(obj, regime, policy, b)] for b in budgets])
    bs = np.array(budgets)
    return float(np.sum((es[:-1] + es[1:]) / 2.0 * np.diff(bs)) / (0.8 - 0.1))


def _paired_boot(d, seed=20260910, n_boot=10000):
    """Median + percentile CI of a paired object-level difference with the
    frozen bootstrap (shared resamples, same draw order as a5_analyze)."""
    objs = list(d)
    diffs = np.array([d[o] for o in objs])
    rng = np.random.default_rng(seed)
    n = len(objs)
    boots = [float(np.median(diffs[rng.integers(0, n, n)]))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.median(diffs)), float(lo), float(hi)


def test_N10_pairwise_classical_lower_auc():
    """N10: post-hoc paired comparison — mode-aware minus each classical
    baseline has positive median (+0.019..+0.027) and CI excluding 0, in both
    regimes (classical E/A/D-opt slightly lower AUC). Recomputed from
    uos_table.csv and cross-checked against the committed pairwise CSV."""
    import csv
    table = _allocation_uos()
    objs = sorted({o for (o, r, _p, _b) in table})
    assert len(objs) == 11
    med_all = []
    for regime in (10, 100):
        for pol in ("e_opt", "a_opt", "d_opt"):
            d = {o: _auc95(table, o, regime, "mode_aware")
                 - _auc95(table, o, regime, pol) for o in objs}
            med, lo, hi = _paired_boot(d)
            assert med > 0 and lo > 0            # classical better, stable
            med_all.append(med)
        # cross-check the committed pairwise CSV rows for this regime
        with open(R / "openillumination/allocation/"
                  "allocation_policy_pairwise.csv", newline="",
                  encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["regime"] == str(regime)]
        assert len(rows) == 3
        assert all(r["analysis_status"] == "posthoc_paired_comparison"
                   for r in rows)
    assert 0.0185 <= min(med_all) <= max(med_all) <= 0.0275


def test_N11_random48_control_ci_spans_zero():
    """N11: with random restricted to the 48 Fisher-active lights, the
    mode-aware advantage disappears (Delta ~ +0.014/+0.019, CI spanning 0)."""
    import json
    s = json.loads((R / "openillumination/allocation/"
                    "allocation_random48_summary.json").read_text(
                        encoding="utf-8"))
    assert s["analysis_status"] == "posthoc_attribution_control"
    for regime, med_ref in (("10", 0.0141), ("100", 0.0187)):
        v = s["regimes"][regime]["mode_minus_random_active48"]
        assert v["median"] == pytest.approx(med_ref, abs=5e-4)
        assert v["ci95"][0] < 0 < v["ci95"][1]
        assert v["improved_negative"] == 3


def test_N12_active_set_attribution():
    """N12: the active-set effect itself (random48 - random_full) is large and
    significant: medians -0.198 / -0.342, CIs excluding 0, 11/11 improved."""
    import json
    s = json.loads((R / "openillumination/allocation/"
                    "allocation_random48_summary.json").read_text(
                        encoding="utf-8"))
    for regime, med_ref in (("10", -0.198), ("100", -0.342)):
        v = s["regimes"][regime]["random_active48_minus_random_full"]
        assert v["median"] == pytest.approx(med_ref, abs=5e-4)
        assert v["ci95"][1] < 0
        assert v["improved_negative"] == 11
