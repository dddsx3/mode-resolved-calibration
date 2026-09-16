#!/usr/bin/env bash
# reproduce_paper.sh · staged reproduction chain: install -> tests -> experiments
# (by stage) -> figures. Every step prints the GATE name of the artifact it
# produces so a runner can cross-check against results/ (acceptance 665d01e
# section 5: the previous version covered only the early synthetic era and
# was referenced by no document).
#
# Usage:
#   ./reproduce_paper.sh                      # all stages (skips stages whose
#                                             # raw data is absent)
#   ./reproduce_paper.sh --stage synthetic    # no raw data needed
#   ./reproduce_paper.sh --stage diag         # zero-cost diagnostics on the
#                                             # COMMITTED artifacts (seconds)
#   ./reproduce_paper.sh --stage oi           # OpenIllumination experiments
#   ./reproduce_paper.sh --stage dq           # DiLiGenT experiments
#   ./reproduce_paper.sh --stage figures
#   ./reproduce_paper.sh --list               # print the runner table
#   ./reproduce_paper.sh --stage oi --long    # also run the multi-hour
#                                             # experiments (skipped by default)
#
# Raw data is external: OpenIllumination (D:/data/OpenIllumination) and
# DiLiGenT (D:/data/DiLiGenT/pmsData), see docs/DATA.md. Steps whose data
# directory is absent are SKIPPED with an explicit message (never silently).
# Dependencies: python>=3.10, numpy, scipy, pyyaml, matplotlib, pillow.
set -uo pipefail
cd "$(dirname "$0")"

STAGE="all"
LONG=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --long) LONG=1; shift ;;
        --list) STAGE="list"; shift ;;
        *) echo "[reproduce] unknown argument: $1"; exit 2 ;;
    esac
done

PASSED=0; FAILED=0; SKIPPED=0
FAILED_STEPS=()

gate_of() {  # prints the gate + analysis_status of an artifact, or MISSING
    python - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print(f"    MISSING artifact: {sys.argv[1]}"); sys.exit(1)
print(f"    artifact {sys.argv[1]}  gate={d.get('gate', '<none>')}"
      f"  status={d.get('analysis_status', '?')}")
PY
}

run_step() {  # label, artifact, requires(oi|dq|oidq|none), long(0|1), command...
    local label="$1" artifact="$2" requires="$3" is_long="$4"; shift 4
    if [ "$STAGE" != "all" ] && [ "$STAGE" != "$_CURRENT_STAGE" ]; then return; fi
    if [ "$is_long" = "1" ] && [ "$LONG" = "0" ]; then
        echo "[reproduce] SKIP (long; use --long) $label"; SKIPPED=$((SKIPPED+1)); return
    fi
    if { [ "$requires" = "oi" ] || [ "$requires" = "oidq" ]; } && [ ! -d "D:/data/OpenIllumination" ]; then
        echo "[reproduce] SKIP (no OpenIllumination data) $label"; SKIPPED=$((SKIPPED+1)); return
    fi
    if { [ "$requires" = "dq" ] || [ "$requires" = "oidq" ]; } && [ ! -d "D:/data/DiLiGenT/pmsData" ]; then
        echo "[reproduce] SKIP (no DiLiGenT data) $label"; SKIPPED=$((SKIPPED+1)); return
    fi
    echo "[reproduce] $label"
    if "$@"; then
        if gate_of "$artifact"; then PASSED=$((PASSED+1)); else
            echo "[reproduce] FAIL (artifact missing): $label"
            FAILED=$((FAILED+1)); FAILED_STEPS+=("$label"); fi
    else
        echo "[reproduce] FAIL (command): $label"
        FAILED=$((FAILED+1)); FAILED_STEPS+=("$label")
    fi
}

RUNNERS_TABLE() {  # stage|label|artifact|requires|long
    cat <<'TABLE'
synthetic|synthetic full grid|results/synthetic/ci01_formal_summary.json|none|0
synthetic|gauge spectrum|results/gauge_spectrum/ci02_formal_summary.json|none|0
synthetic|Monte-Carlo variance|results/monte_carlo/ci03_formal_summary.json|none|0
synthetic|nonlinear validity|results/nonlinear/ci03nl_nl_formal_summary.json|none|0
diag|feasible-budget DQ diagnostic|results/openillumination/decision_quality_feasible.json|none|0
diag|ablation-budget degeneracy|results/openillumination/active_set_ablation_feasible.json|none|0
diag|E-arm cross diagnosis|results/openillumination/corruption_family_e_diag.json|none|0
diag|baseline v1->v1.1 provenance|results/baseline/provenance/v1_v1_1_reuse.json|none|0
oi|anchor-share mechanism|results/openillumination/anchor_mechanism.json|oi|0
oi|certified dynamic range|results/certification/certified_gaps.json|oi|0
oi|directional amplitude|results/magnitude/directional_amplitude_summary.json|oi|0
oi|full-resolution confirmation|results/certification/lowrank_fullres.json|oi|0
oi|allocation rank invariance|results/openillumination/correctness/allocation_rank_check.json|oi|0
oi|mode-tail allocation|results/mode_tail/allocation_mode_tail.json|oi|0
oi|certificate concentration|results/certification/certificate_concentration.json|oi|0
oi|factorial correctness rerun|results/openillumination/correctness/mf0_factorial_summary.json|oi|0
oi|Sigma_phi family sensitivity|results/openillumination/corruption_family_sensitivity.json|oi|0
oi|linearization radius (v2)|results/magnitude/linearization_radius.json|oi|0
oi|linearization radius (family)|results/magnitude/linearization_radius_family.json|oi|0
oi|decision quality (full grid)|results/openillumination/decision_quality.json|oi|1
oi|decision quality (family)|results/openillumination/decision_quality_family.json|oi|1
dq|DiLiGenT sanity|results/diligent/ci05_formal_summary.json|dq|0
dq|DiLiGenT ablation|results/diligent_ablation/ci05abl_ablation_summary.json|dq|0
dq|DiLiGenT queue (transfer)|results/diligent/diligent_queue.json|dq|0
dq|ball anchor (sphere calibration)|results/openillumination/ball_anchor.json|dq|0
dq|baseline comparison (OI+DiLiGenT)|results/baseline/baseline_comparison.json|oidq|0
TABLE
}

if [ "$STAGE" = "list" ]; then
    echo "stage | label | artifact | requires | long"
    RUNNERS_TABLE
    exit 0
fi

echo "[reproduce] stage=$STAGE long=$LONG"
echo "[reproduce] 1/3 install"
python -m pip install -e . --quiet

echo "[reproduce] 2/3 tests (unit + gates + independent headline recomputation)"
if python -m pytest -q; then PASSED=$((PASSED+1)); else
    echo "[reproduce] FAIL (pytest)"; FAILED=$((FAILED+1)); FAILED_STEPS+=("pytest")
fi

echo "[reproduce] 3/3 experiments + figures"

_CURRENT_STAGE="synthetic"
run_step "synthetic full grid" results/synthetic/ci01_formal_summary.json none 0 \
    python scripts/run_experiments.py --experiment synthetic --config configs/synthetic.yaml
run_step "gauge spectrum" results/gauge_spectrum/ci02_formal_summary.json none 0 \
    python scripts/run_experiments.py --experiment gauge_spectrum --config configs/gauge_spectrum.yaml
run_step "Monte-Carlo variance" results/monte_carlo/ci03_formal_summary.json none 0 \
    python scripts/run_experiments.py --experiment monte_carlo --config configs/monte_carlo.yaml
run_step "nonlinear validity" results/nonlinear/ci03nl_nl_formal_summary.json none 0 \
    python scripts/run_experiments.py --experiment nonlinear --config configs/nonlinear.yaml

_CURRENT_STAGE="diag"
run_step "feasible-budget DQ diagnostic" results/openillumination/decision_quality_feasible.json none 0 \
    python experiments/decision_quality_feasible.py
run_step "ablation-budget degeneracy" results/openillumination/active_set_ablation_feasible.json none 0 \
    python experiments/active_set_ablation_feasible.py
run_step "E-arm cross diagnosis" results/openillumination/corruption_family_e_diag.json none 0 \
    python experiments/family_e_diag.py
run_step "baseline v1->v1.1 provenance" results/baseline/provenance/v1_v1_1_reuse.json none 0 \
    python experiments/baseline_provenance.py
run_step "submodularity recomputation" results/submodularity/submodularity_search.json none 0 \
    python experiments/submodularity_search.py --out /tmp/submod.json

_CURRENT_STAGE="oi"
run_step "anchor-share mechanism" results/openillumination/anchor_mechanism.json oi 0 \
    python experiments/anchor_mechanism.py
run_step "certified dynamic range" results/certification/certified_gaps.json oi 0 \
    python experiments/certified_gaps.py
run_step "directional amplitude" results/magnitude/directional_amplitude_summary.json oi 0 \
    python experiments/directional_amplitude.py
run_step "full-resolution confirmation" results/certification/lowrank_fullres.json oi 0 \
    python experiments/lowrank_fullres.py
run_step "allocation rank invariance" results/openillumination/correctness/allocation_rank_check.json oi 0 \
    python experiments/allocation_rank_check.py
run_step "mode-tail allocation" results/mode_tail/allocation_mode_tail.json oi 0 \
    python experiments/allocation_mode_tail.py
run_step "certificate concentration" results/certification/certificate_concentration.json oi 0 \
    python experiments/certificate_concentration.py
run_step "factorial correctness rerun" results/openillumination/correctness/mf0_factorial_summary.json oi 0 \
    python experiments/openillumination_factorial.py
run_step "Sigma_phi family sensitivity" results/openillumination/corruption_family_sensitivity.json oi 0 \
    python experiments/corruption_family_sensitivity.py
run_step "linearization radius (v2)" results/magnitude/linearization_radius.json oi 0 \
    python experiments/linearization_radius.py
run_step "linearization radius (family)" results/magnitude/linearization_radius_family.json oi 0 \
    python experiments/linearization_radius_family.py
run_step "decision quality (full grid)" results/openillumination/decision_quality.json oi 1 \
    python experiments/decision_quality.py
run_step "decision quality (family)" results/openillumination/decision_quality_family.json oi 1 \
    python experiments/decision_quality_family.py

_CURRENT_STAGE="dq"
run_step "DiLiGenT sanity" results/diligent/ci05_formal_summary.json dq 0 \
    python scripts/run_experiments.py --experiment diligent --config configs/diligent.yaml
run_step "DiLiGenT ablation" results/diligent_ablation/ci05abl_ablation_summary.json dq 0 \
    python scripts/run_experiments.py --experiment diligent_ablation --config configs/diligent_ablation.yaml
run_step "DiLiGenT queue (transfer)" results/diligent/diligent_queue.json dq 0 \
    python experiments/diligent_queue.py
run_step "ball anchor (sphere calibration)" results/openillumination/ball_anchor.json dq 0 \
    python experiments/ball_anchor.py
run_step "baseline comparison (OI+DiLiGenT)" results/baseline/baseline_comparison.json oidq 0 \
    python experiments/baseline_comparison.py

_CURRENT_STAGE="figures"
if [ "$STAGE" = "all" ] || [ "$STAGE" = "figures" ]; then
    echo "[reproduce] figures"
    FIGOK=1
    for i in $(seq 1 10); do
        python scripts/make_figures.py --figure "$i" || { echo "[reproduce] FAIL: Fig.$i"; FIGOK=0; }
    done
    if [ "$FIGOK" = "1" ]; then PASSED=$((PASSED+1)); else
        FAILED=$((FAILED+1)); FAILED_STEPS+=("figures"); fi
fi

echo "[reproduce] SUMMARY: passed=$PASSED failed=$FAILED skipped=$SKIPPED"
if [ "$FAILED" -gt 0 ]; then
    printf '[reproduce] FAILED steps: %s\n' "${FAILED_STEPS[*]}"
    exit 1
fi
echo "[reproduce] DONE"
