"""P-OBJECT-CLUSTER-BOOTSTRAP artifact gates (decision-quality contrasts).

CI-safe: reads only results/openillumination/object_cluster_bootstrap.json
and the committed source artifact decision_quality.json. Pins:

1. Provenance chain: source_sha256 equals the committed
   results/openillumination/decision_quality.json byte hash; protocol
   fields: 11 objects x 4 levels = 44 effects, B=10000, seed 20260910,
   budgets [14, 28], regime (kappa) 10.
2. Statistic identity: every recorded median equals the plain median of
   the stored 44 object-level effects.
3. Full regeneration: every recorded 95% interval is regenerated EXACTLY
   by the committed src/calibinfo/metrics/cluster_bootstrap.py module
   (objects resampled with replacement, B=10000, seed 20260910) applied
   to the stored effects — the artifact is a deterministic function of
   committed inputs.
4. Manuscript binding: the six macros bound to this record
   (mode_aware.active48 / universe medians and CIs) match the stored
   values at the manuscript's display precision (-0.712, [-1.046,
   -0.424], -1.790, [-1.980, -1.460]; negative favors the informed
   allocation).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/object_cluster_bootstrap.json"
SOURCE = REPO / "results/openillumination/decision_quality.json"

B, SEED = 10000, 20260910


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_provenance_chain_and_protocol():
    art = _load()
    assert art["source"] == "remote_repo/results/openillumination/decision_quality.json"
    assert art["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert art["n_objects"] == 11 and art["n_levels"] == 4
    assert art["n_records"] == 44
    assert art["B"] == B and art["seed"] == SEED
    assert art["budgets"] == [14, 28] and art["regime"] == 10
    assert art["levels"] == [0.2, 0.5, 1.0, 2.0]
    assert len(art["objects"]) == 11
    assert set(art["results"]) == {"mode_aware", "e_opt", "a_opt", "d_opt"}
    for recs in art["results"].values():
        assert set(recs) == {"active48", "universe"}


def test_statistic_identity():
    art = _load()
    for recs in art["results"].values():
        for rec in recs.values():
            effects = np.asarray(rec["object_level_effects"], dtype=float)
            assert effects.shape == (11, 4)
            assert float(np.median(effects)) == pytest.approx(rec["median"],
                                                              abs=1e-12)


def test_intervals_regenerate_from_committed_module():
    art = _load()

    def _stat(rows):
        return float(np.median(
            np.concatenate([np.asarray(p, dtype=float) for _, p in rows])))

    for recs in art["results"].values():
        for rec in recs.values():
            objs = [(f"obj_{i}", row) for i, row in
                    enumerate(rec["object_level_effects"])]
            point, (lo, hi), _ = cluster_bootstrap(objs, _stat, B=B, seed=SEED)
            assert point == pytest.approx(rec["median"], abs=1e-12)
            assert lo == pytest.approx(rec["ci95"][0], abs=1e-9)
            assert hi == pytest.approx(rec["ci95"][1], abs=1e-9)


def test_manuscript_binding_mode_aware():
    art = _load()
    a48 = art["results"]["mode_aware"]["active48"]
    uni = art["results"]["mode_aware"]["universe"]
    assert a48["median"] == pytest.approx(-0.712, abs=5e-4)
    assert a48["ci95"][0] == pytest.approx(-1.046, abs=5e-4)
    assert a48["ci95"][1] == pytest.approx(-0.424, abs=5e-4)
    assert uni["median"] == pytest.approx(-1.790, abs=5e-4)
    assert uni["ci95"][0] == pytest.approx(-1.980, abs=5e-4)
    assert uni["ci95"][1] == pytest.approx(-1.460, abs=5e-4)
    # negative favors the informed allocation
    assert a48["median"] < 0 and uni["median"] < 0
