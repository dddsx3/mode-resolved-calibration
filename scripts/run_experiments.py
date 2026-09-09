#!/usr/bin/env python3
"""Unified experiment entrypoint.

Usage: python scripts/run_experiments.py --experiment synthetic --config configs/synthetic.yaml
Pipeline: config load -> run manifest (git/config/seed/dataset checksums) -> experiment
runner -> results/ summaries.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from calibinfo.io.manifest import write_manifest  # noqa: E402

RUNNERS = {
    "synthetic": ("experiments.numerical_identities", "run"),
    "gauge_spectrum": ("experiments.gauge_spectrum", "run"),
    "monte_carlo": ("experiments.monte_carlo_validation", "run"),
    "nonlinear": ("experiments.nonlinear_validity", "run"),
    "openillumination": ("experiments.openillumination_validation", "run"),
    "diligent": ("experiments.diligent_sanity", "run_sanity"),
    "diligent_ablation": ("experiments.diligent_sanity", "run_ablation"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", default=str(ROOT / "results" / "raw"))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run_dir = Path(args.output_root) / f"{args.experiment}_{config.get('run_name', 'run')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(run_dir / "manifest.json", config=config,
                   experiment=args.experiment, seed=config.get("seed"),
                   repo_root=str(ROOT))

    mod_name, fn_name = RUNNERS[args.experiment]
    result = getattr(importlib.import_module(mod_name), fn_name)(config, run_dir)

    frozen = Path(args.output_root) / args.experiment
    frozen.mkdir(parents=True, exist_ok=True)
    stem = f"{args.experiment}_{config.get('run_name', 'run')}"
    # results/ holds summaries + manifests only (provenance goes beside the summary)
    (frozen / f"{stem}_manifest.json").write_text(
        (run_dir / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    summary_path = frozen / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[run] {args.experiment} done -> {run_dir}")
    print(f"[run] summary -> {summary_path}")


if __name__ == "__main__":
    main()
