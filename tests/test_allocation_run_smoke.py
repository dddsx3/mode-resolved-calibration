"""run() dry-run smoke tests (bug-fix checklist acceptance).

A minimal fake object (no raw data) is injected through a monkeypatched loader;
the level/seed grid is reduced for speed. These tests lock the three historical
crash points (NameError / IndexError / KeyError), the 142-light selection
universe, budget fairness in the actual run path, the flat budget tail
(k >= 48 selects every analysis light), and the frozen preregistration file.
"""

import csv
import json

import numpy as np
import pytest
import yaml

import experiments.openillumination_allocation as alloc

REPO = alloc.REPO


def _fake_object(seed=0, n_lights=142, side=8, n_mask=40):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_lights, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mask = np.zeros((side, side), bool)
    mask.ravel()[rng.choice(side * side, n_mask, replace=False)] = True
    # physically consistent images: every light lights every masked pixel
    # (s_hat_true = |n_true . d| > 0 for all lights), so all 48 analysis lights
    # carry Fisher presence — matching real OpenIllumination scenes
    n_true = rng.normal(size=(n_mask, 3))
    n_true /= np.linalg.norm(n_true, axis=1, keepdims=True)
    rho_true = rng.uniform(0.5, 1.0, size=n_mask)
    s_true = np.clip(n_true @ dirs.T, 0.05, None).T           # (n_lights, n_mask)
    imgs2d = s_true * rho_true[None, :]
    images = np.zeros((n_lights, side, side, 3))
    flat = images[..., 0].reshape(n_lights, side * side)
    flat[:, mask.ravel()] = imgs2d
    return {"images": images, "mask": mask, "light_directions": dirs}


def _temp_config(tmp_path, n_objects=1):
    cfg = yaml.safe_load(
        (REPO / "configs/openillumination_allocation.yaml").read_text(encoding="utf-8"))
    cfg["cohort"] = [f"fake_obj_{i}" for i in range(n_objects)]
    cfg["n_objects"] = n_objects
    p = tmp_path / "smoke_config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture()
def smoke_run(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc, "load_object",
                        lambda root, name, data_meta=None: _fake_object(0))
    monkeypatch.setattr(alloc, "LEVELS", [0.2])
    monkeypatch.setattr(alloc, "SEEDS_PER_LEVEL", 2)
    out = tmp_path / "out"
    summary = alloc.run(config_path=_temp_config(tmp_path), out_dir=out)
    return summary, out


def test_run_smoke_completes_and_grades(smoke_run):
    """The three historical crash points (NameError / IndexError / KeyError) stay
    fixed: run() completes end-to-end on the 142-light universe."""
    summary, out = smoke_run
    assert set(summary["regimes"]) == {10, 100}
    for regime, blk in summary["regimes"].items():
        assert blk["grade"] in ("strong", "neutral", "negative")
        assert len(blk["objects"]) == 1
        for p, r in blk["policy_deltas"].items():
            assert "per_object" in r


def test_selection_universe_is_142(smoke_run):
    """Every ordering covers the full capture light set (universe-size contract)."""
    _summary, out = smoke_run
    orders = json.loads((out / "selection_orders.json").read_text(encoding="utf-8"))
    assert orders
    for key, ordr in orders.items():
        assert len(ordr) == 142, key
        assert sorted(ordr) == list(range(142)), key


def test_budget_prefix_active_counts(tmp_path, monkeypatch):
    """Mechanical budget fairness in the actual run path: every ordering devotes
    exactly k = round(b*142) prefix slots to improved lights; deterministic
    policies additionally front-load the Fisher-active analysis lights, so their
    prefix contains min(k, 48) active lights (random orderings contain at most
    that many — no argmax guidance is their defining property)."""
    monkeypatch.setattr(alloc, "load_object",
                        lambda root, name, data_meta=None: _fake_object(0))
    monkeypatch.setattr(alloc, "LEVELS", [0.2])
    monkeypatch.setattr(alloc, "SEEDS_PER_LEVEL", 1)
    out = tmp_path / "out"
    alloc.run(config_path=_temp_config(tmp_path), out_dir=out)
    orders = json.loads((out / "selection_orders.json").read_text(encoding="utf-8"))
    sel = np.sort(np.random.default_rng([20260910, 0]).choice(142, 48, replace=False))
    sel_set = {int(i) for i in sel}
    assert len(sel_set) == 48
    for key, ordr in orders.items():
        assert len(ordr) == 142 and sorted(ordr) == list(range(142))
        for b, k in zip(alloc.BUDGETS, alloc.BUDGET_COUNTS):
            prefix = ordr[:k]
            n_active = sum(1 for pos in prefix if pos in sel_set)
            if key.split("|")[-1].startswith("random"):
                assert n_active <= min(k, 48), (key, b, n_active)
            else:
                assert n_active == min(k, 48), (key, b, n_active)


def test_budget_flat_tail_and_direction(smoke_run):
    """Deterministic policies: once k >= 48 the prefix contains every
    Fisher-active analysis light, so E_osb is bitwise flat across budgets
    0.4/0.6/0.8 (frozen design: only 48 lights carry Fisher presence), and more
    budget never increases the error (E(0.4) <= E(0.1))."""
    _summary, out = smoke_run
    table = {}
    with open(out / "uos_table.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            table[(r["object"], int(r["regime"]), r["policy"], float(r["budget"]))] = \
                float(r["E_osb"])
    assert table
    for (o, regime, p, _b) in list(table):
        if p.startswith("random"):
            continue
        e01 = table[(o, regime, p, 0.1)]
        e04 = table[(o, regime, p, 0.4)]
        e06 = table[(o, regime, p, 0.6)]
        e08 = table[(o, regime, p, 0.8)]
        assert e04 == e06 == e08, (o, regime, p, e04, e06, e08)
        assert e04 <= e01


def test_per_run_provenance_written(smoke_run):
    """Policy sequence and per-run errors are saved with the improved-light count
    (deterministic policies: exactly min(k, 48); random: at most min(k, 48))."""
    _summary, out = smoke_run
    with open(out / "per_run_errors.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        if r["policy"].startswith("random"):
            assert int(r["n_improved"]) <= min(round(float(r["budget"]) * 142), 48)
        else:
            assert int(r["n_improved"]) == min(round(float(r["budget"]) * 142), 48)
        assert float(r["E_run"]) >= 0.0


def test_frozen_prereg_config_unchanged():
    """The preregistration file keeps its frozen content (cohort, budgets, seeds)."""
    cfg = yaml.safe_load(
        (REPO / "configs/openillumination_allocation.yaml").read_text(encoding="utf-8"))
    assert cfg["n_objects"] == 11 and len(cfg["cohort"]) == 11
    assert cfg["cohort"][0] == "obj_03_pumpkin"
    assert cfg["cohort"][-1] == "obj_19_cylinder"
    assert cfg["budget_counts"] == [14, 28, 57, 85, 114]
    assert cfg["K_lights"] == 142 and cfg["analysis_lights"] == 48
    assert cfg["seeds"]["bootstrap"] == 20260910
    assert cfg["positioning"].strip().startswith(
        "preregistered retrospective actionability evaluation")
