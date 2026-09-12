#!/usr/bin/env bash
# reproduce_paper.sh · full reproduction chain: install -> tests -> experiments -> figures/tables
# Dependencies: python>=3.10, numpy, scipy, pyyaml, matplotlib, pillow
# Raw data is external: OpenIllumination (D:/data/OpenIllumination) and
# DiLiGenT (D:/data/DiLiGenT/pmsData), see docs/DATA.md. Steps that need raw data are
# skipped automatically when the data directory is absent.
set -euo pipefail
cd "$(dirname "$0")"

echo "[reproduce] 1/4 install"
python -m pip install -e . --quiet

echo "[reproduce] 2/4 tests (unit + known-answer + independent headline recomputation)"
python -m pytest -q

echo "[reproduce] 3/4 synthetic experiments (CPU full grid)"
python scripts/run_experiments.py --experiment synthetic --config configs/synthetic.yaml
python scripts/run_experiments.py --experiment gauge_spectrum --config configs/gauge_spectrum.yaml
python scripts/run_experiments.py --experiment monte_carlo --config configs/monte_carlo.yaml
python scripts/run_experiments.py --experiment nonlinear --config configs/nonlinear.yaml

if [ -d "D:/data/OpenIllumination" ]; then
    python scripts/run_experiments.py --experiment openillumination --config configs/openillumination.yaml
else
    echo "[reproduce] skip openillumination (no local data; contract = results/openillumination/*_manifest.json)"
fi

if [ -d "D:/data/DiLiGenT/pmsData" ]; then
    python scripts/run_experiments.py --experiment diligent --config configs/diligent.yaml
    python scripts/run_experiments.py --experiment diligent_ablation --config configs/diligent_ablation.yaml
else
    echo "[reproduce] skip diligent (no local data; contract = results/diligent/*_manifest.json)"
fi

echo "[reproduce] 4/4 figures from results/"
for i in $(seq 1 10); do
    python scripts/make_figures.py --figure "$i" || \
        { echo "[reproduce] FAIL: Fig.$i"; exit 1; }
done

echo "[reproduce] DONE"