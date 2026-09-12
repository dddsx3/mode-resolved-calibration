"""Manifest integrity: config sha256 recorded inside result manifests must
match the committed config file (or be explicitly registered as drift).

Audit A7 changed two certification manifests from silently-stale
config_sha256 to an explicit `config_sha256_at_run` + `config_drift_note`
pair; this gate keeps the rest of the ledger honest and prevents any future
config edit from silently desyncing a result manifest."""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PAIRS = [("configs/certified_gaps.yaml", "results/certification/certified_gaps.json"),
         ("configs/lowrank_fullres.yaml", "results/certification/lowrank_fullres.json"),
         ("configs/certificate_concentration.yaml",
          "results/certification/certificate_concentration.json"),
         ("configs/linearization_radius.yaml",
          "results/magnitude/linearization_radius.json"),
         ("configs/allocation_mode_tail.yaml",
          "results/mode_tail/allocation_mode_tail.json")]


def test_config_hash_matches_manifest():
    bad = []
    for cfg, res in PAIRS:
        a = hashlib.sha256((REPO / cfg).read_bytes()).hexdigest()
        m = json.loads((REPO / res).read_text(encoding="utf-8")).get("manifest", {})
        b = m.get("config_sha256")
        if b is not None and a != b:
            # a config edit after the run must be registered explicitly
            drift_ok = (m.get("config_sha256_at_run") == b
                        and m.get("config_drift_note"))
            if not drift_ok:
                bad.append((cfg, res, a[:12], b[:12]))
    assert not bad, f"config drift without registration: {bad}"