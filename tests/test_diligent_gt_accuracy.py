"""P-DILIGENT-GT-ACCURACY artifact gates (absolute endpoint, second dataset).

CI-safe: reads only results/diligent/diligent_gt_accuracy.json. Pins:

1. Protocol: 10 DiLiGenT objects, 96 lights with 48 analysis lights,
   budgets {0, 5, 14, 28, 48}, policies {mode_aware, active_random},
   kappa=10, level=0.5; the endpoint is the full-mask mean angular error
   to DiLiGenT Normal_gt.
2. Headline curve: mode_aware cohort median 18.9681 deg (k=0) falls to
   16.4146 deg (k=48) against the exact-calibration nominal floor
   16.0175 deg; the two policies coincide at k=0 and k=48.
3. Paired policy differences: object-median mode_aware-minus-active_random
   is +0.0203 / -0.0372 / +0.3706 deg at k=5/14/28 (negative favors
   mode-aware) -- the sign changes at intermediate budgets, so no
   universal ordering advantage is claimed; both medians are 0 at k=0
   and k=48.
4. Predesignated bearPNG error map: the mean over the 10 paired
   corruption seeds reproduces 13.0746 deg (k=0) and 11.3461 deg (k=28).
5. Table completeness: 2000 rows (mode_aware 100 and active_random 300
   per budget over 10 objects), per_object 100 rows, nominal_per_object
   and object_metadata cover the cohort.
6. Provenance: loader_sha256 pins src/calibinfo/datasets/diligent.py
   byte-for-byte; raw images are not redistributed; the scope disclaimer
   is present.
"""

import hashlib
import json
import statistics
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/diligent/diligent_gt_accuracy.json"
LOADER = REPO / "src/calibinfo/datasets/diligent.py"

TOL = 5e-5


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_protocol_pin():
    art = _load()
    assert art["analysis_status"] == "diligent_gt_accuracy_v1"
    p = art["protocol"]
    assert p["n_objects"] == 10 and len(p["cohort"]) == 10
    assert p["n_lights_total"] == 96 and p["n_analysis_lights"] == 48
    assert p["budgets"] == [0, 5, 14, 28, 48]
    assert set(p["policies"]) == {"mode_aware", "active_random"}
    assert p["kappa"] == 10.0 and p["level"] == 0.5


def test_headline_curve():
    art = _load()
    cc = art["summary"]["cohort_curve"]
    assert cc["mode_aware"]["0"]["median_deg"] == pytest.approx(
        18.968076281977822, abs=TOL)
    assert cc["mode_aware"]["48"]["median_deg"] == pytest.approx(
        16.414571656443428, abs=TOL)
    assert cc["exact_calibration_floor"]["median_deg"] == pytest.approx(
        16.017471280957125, abs=TOL)
    # the two policies share the same selection at k=0 and k=48
    for k in ("0", "48"):
        assert cc["mode_aware"][k]["median_deg"] == pytest.approx(
            cc["active_random"][k]["median_deg"], abs=TOL)


def test_policy_differences_change_sign():
    art = _load()
    ppd = art["summary"]["paired_policy_difference"]
    assert ppd["0"]["median_mode_minus_random_deg"] == pytest.approx(0.0, abs=TOL)
    assert ppd["48"]["median_mode_minus_random_deg"] == pytest.approx(0.0, abs=TOL)
    assert ppd["5"]["median_mode_minus_random_deg"] == pytest.approx(
        0.020333159559511316, abs=TOL)
    assert ppd["14"]["median_mode_minus_random_deg"] == pytest.approx(
        -0.03723615338211772, abs=TOL)
    assert ppd["28"]["median_mode_minus_random_deg"] == pytest.approx(
        0.3706111997118109, abs=TOL)


def test_bearpng_map_values():
    art = _load()
    assert art["protocol"]["map"]["object"] == "bearPNG"
    assert art["protocol"]["map"]["budget"] == 28
    by_budget = {}
    for r in art["rows"]:
        if (r["object"] == "bearPNG" and r["policy"] == "mode_aware"
                and r["budget"] in (0, 28)):
            by_budget.setdefault(r["budget"], []).append(r["mean_deg"])
    assert len(by_budget[0]) == 10 and len(by_budget[28]) == 10
    # per-pixel mean over the 10 paired corruption seeds (figure panels b/c)
    assert statistics.mean(by_budget[0]) == pytest.approx(13.0746, abs=5e-4)
    assert statistics.mean(by_budget[28]) == pytest.approx(11.3461, abs=5e-4)


def test_table_completeness():
    art = _load()
    rows = art["rows"]
    assert len(rows) == 2000
    counts = {}
    required = {"object", "policy", "budget", "seed",
                "mean_deg", "median_deg", "p95_deg"}
    allowed = required | {"random_index"}   # active_random rows carry it
    for r in rows:
        key = (r["policy"], r["budget"])
        counts[key] = counts.get(key, 0) + 1
        assert required <= set(r) <= allowed
    for b in (0, 5, 14, 28, 48):
        assert counts[("mode_aware", b)] == 100
        assert counts[("active_random", b)] == 300
    cohort = set(art["protocol"]["cohort"])
    assert len(art["per_object"]) == 100
    assert {r["object"] for r in art["per_object"]} == cohort
    assert set(art["nominal_per_object"]) == cohort
    assert set(art["object_metadata"]) == cohort


def test_provenance_pins_loader():
    art = _load()
    prov = art["provenance"]
    assert prov["loader_sha256"] == hashlib.sha256(
        LOADER.read_bytes()).hexdigest()
    assert prov["raw_images_redistributed"] is False
    assert "dataset_mirror_commit" in prov and "script_sha256" in prov
    assert "does not" in art["scope"]
