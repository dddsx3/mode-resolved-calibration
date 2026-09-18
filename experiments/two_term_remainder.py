# status: experimental — NOT part of the published results
"""M14 research runner: frozen 44-cell endpoints plus the full C9 Sigma family.

No historical script is executed as a program and no frozen result is written.
All 45 already-defined C9 grid points are included (495 cells, including all
88 direction-only controls), without selecting points by observed error.

Run from any directory, for example:
  python -B C:/Users/35702/publication/mode-resolved-calibration/experiments/two_term_remainder.py

The output path is intentionally fixed to the one authorized new artifact.
Numerical observations are computed by the unchanged published low-rank risk
routine, independently of the binned certificate.  The dyadic bins are fixed
by a factor-two endpoint theorem, not fitted to the observed error.  This is
not claimed to be a prospectively preregistered study: one object was explored
before implementation; the user's <=10 tightness criterion is unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
from scipy.linalg import block_diag
import scipy
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]

from calibinfo.allocation.blocks import LightBlocks
from calibinfo.datasets.openillumination import load_object
from calibinfo.information.lowrank import woodbury_quad_risk
from calibinfo.models.corruption_family import CorruptionFamily
from experiments.alpha_bound import build_state
from experiments.corruption_family_sensitivity import build_grid
from experiments.openillumination_validation import NominalScene
from experiments.theory_remainder import (
    analyze_prior, compress_spectrum, endpoint_certificate, prepare_geometry,
    value_certificate,
)

OUT = REPO / 'results/theory_extension_20260918/two_term_remainder.json'
GOAL = REPO / 'results/goal_oriented/goal_orientation.json'
FAMILY = REPO / 'results/openillumination/corruption_family_sensitivity.json'
EXPECTED_FROZEN_COUNT = 75
EXPECTED_FROZEN_TREE = '85c33124fab53bcd55dd463f7e9ce4fe81715b1941649301e8dffac532a92e23'
# Fixed before the all-object run; NOT tuned against any observed residual.
TIGHTNESS_THRESHOLD = 10.0
NUMERICAL_RTOL = 2e-7
NUMERICAL_ATOL = 2e-11
REPLAY_RTOL = 2e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frozen_snapshot():
    return {p.relative_to(REPO).as_posix(): sha(p)
            for p in sorted((REPO / 'results').rglob('*'))
            if p.is_file() and 'theory_extension_20260918' not in p.parts}


def distribution(values):
    arr = np.asarray(list(values), dtype=float)
    if not len(arr):
        return dict(n=0, min=None, p05=None, median=None, p95=None, max=None)
    return dict(n=len(arr), min=float(arr.min()), p05=float(np.percentile(arr, 5)),
                median=float(np.median(arr)), p95=float(np.percentile(arr, 95)),
                max=float(arr.max()))


def ratio(bound, observed):
    # Do not floor a tiny observed error to manufacture a tightness pass.
    return float(bound / observed) if observed > 0 else None


def tolerance(scale):
    return NUMERICAL_ATOL + NUMERICAL_RTOL * abs(scale)


def scene_geometry(scen, blk):
    idx = np.flatnonzero(blk.active)
    U = np.concatenate([blk.u[k] for k in idx], axis=1)
    M = block_diag(*blk.M0[idx])
    norm = float(np.linalg.norm(scen.rho))
    a = scen.rho / norm
    c = np.tile([1.0 / norm, 0.0, 0.0], len(idx))
    geometry = prepare_geometry(blk.finf, U, M, a, c)
    Aa = np.sqrt(scen.w) * scen.s_hat * a[None, :]
    Bc = np.sqrt(scen.w) * scen.B_phi[:, :, 0] / norm
    gauge = float(np.linalg.norm(Aa - Bc) / np.linalg.norm(Aa))
    return geometry, a, gauge


def physical_diagnostics(geometry, scen, fam, active_idx, sigma_active, spectrum):
    g = geometry.g.reshape(-1, 3)
    R2 = float(scen.rho @ scen.rho)
    sI = fam.sigma_logI_vec()[active_idx]
    sD = fam.sigma_rad_vec()[active_idx]
    q_formula = float(np.sum(1.0 / (sI**2 * (1.0 - fam.rho_c**2))) / R2)
    # Direct physical quadratic form; this diagnostic is not the certificate.
    g_sigma_g = float(np.einsum('ki,kij,kj->', g, sigma_active, g))
    prior_target = np.linalg.solve(sigma_active, geometry.c.reshape(-1, 3)[..., None])[..., 0]
    prior_target /= spectrum.q
    er = g - prior_target
    residual_physical = float(np.einsum('ki,kij,kj->', er, sigma_active, er))
    return dict(
        sigma_logI_active=distribution(sI),
        sigma_dir_per_axis_deg_active=distribution(np.degrees(sD)),
        sigma_dir_two_axis_rms_deg_active=distribution(np.sqrt(2.0) * np.degrees(sD)),
        het_sigma=fam.het_sigma, rho_c=fam.rho_c,
        q_from_physical_parameters=q_formula,
        prior_target_residual_squared=residual_physical,
        prior_residual_gSigmaG_minus_inverse_q=g_sigma_g - 1.0 / spectrum.q,
        uniform_intensity_leverage_spread=float(np.linalg.norm(g[:, 0] / np.sqrt(R2)
                                                                 - 1.0 / len(g))),
        direction_leverage_squared=float(np.sum(g[:, 1:]**2)),
    )


def evaluate_cell(scen, blk, geometry, a, parameters, kappa, family_seed,
                  frozen_row, cohort, cell_id):
    idx = np.flatnonzero(blk.active)
    fam = CorruptionFamily(parameters['sig_logI'], parameters['sig_dir_deg'],
                           het_sigma=parameters['het_sigma'], rho_c=parameters['rho_c'],
                           seed=family_seed, n_lights=blk.L)
    sig_all = fam.sigma_phi_block()
    sig = sig_all[idx]
    lam = np.linalg.inv(sig_all)
    # All certificate construction precedes any observed risk calculation.
    spectrum = analyze_prior(geometry, block_diag(*sig))
    compressed = compress_spectrum(spectrum)
    certs = {key: endpoint_certificate(compressed, t)
             for key, t in (('t1', 1.0), ('tkappa', kappa))}
    vc = value_certificate(compressed, kappa)
    endpoints = {}
    actual = {}
    for key, t, frozen_key in (('t1', 1.0, 'J_rho_mean_1'),
                               ('tkappa', kappa, 'J_rho_mean_kappa')):
        ts = np.ones(blk.L)
        ts[idx] = t
        computed = woodbury_quad_risk(blk.finf, blk.u, blk.M0, lam,
                                      blk.active, ts, a)
        frozen_j = float(frozen_row[frozen_key]) if cohort == 'nominal' else None
        observed_j = frozen_j if frozen_j is not None else computed
        cert = certs[key]
        observed_rem = observed_j - cert['prediction']
        identity_rem = spectrum.remainder_for_validation(t)
        bound = cert['remainder_upper']
        tol = tolerance(observed_j)
        endpoints[key] = dict(
            **cert, observed_risk=observed_j, recomputed_risk=computed,
            frozen_risk=frozen_j,
            observed_remainder=observed_rem,
            observed_absolute_error=abs(observed_rem),
            identity_remainder_validation_only=identity_rem,
            identity_risk_relative_error=abs(cert['prediction'] + identity_rem - computed) / computed,
            frozen_replay_relative_error=(abs(computed - frozen_j) / frozen_j
                                          if frozen_j is not None else None),
            bound_over_observed=ratio(bound, abs(observed_rem)),
            numerical_check_tolerance=tol,
            bounds_hold=bool(cert['remainder_lower'] - tol <= observed_rem <= bound + tol),
            absolute_bound_holds=bool(abs(observed_rem) <= bound + tol),
            tightness_le_10=bool(abs(observed_rem) > 0 and bound / abs(observed_rem) <= TIGHTNESS_THRESHOLD),
        )
        actual[key] = observed_j
    observed_v = float(frozen_row['V_rho_mean'] if cohort == 'nominal' else frozen_row['V_meas'])
    recomputed_v = 1.0 - endpoints['tkappa']['recomputed_risk'] / endpoints['t1']['recomputed_risk']
    observed_correction = observed_v - vc['prediction']
    tol_v = tolerance(1.0)
    value = dict(
        **vc, observed=observed_v, recomputed=recomputed_v,
        observed_correction=observed_correction,
        observed_absolute_error=abs(observed_correction),
        frozen_replay_absolute_error=abs(recomputed_v - observed_v),
        bound_over_observed=ratio(vc['absolute_error_upper'], abs(observed_correction)),
        bounds_hold=bool(vc['correction_lower'] - tol_v <= observed_correction
                         <= vc['correction_upper'] + tol_v),
        tightness_le_10=bool(abs(observed_correction) > 0
                             and vc['absolute_error_upper'] / abs(observed_correction) <= TIGHTNESS_THRESHOLD),
        error_exceeds_0_05=bool(abs(observed_correction) > 0.05),
        certificate_rules_out_0_05_accuracy=bool(vc['correction_lower'] > 0.05
                                                or vc['correction_upper'] < -0.05),
        certified_absolute_accuracy_0_05=bool(vc['absolute_error_upper'] <= 0.05),
    )
    return dict(
        cell_id=cell_id, cohort=cohort, object=frozen_row['object'],
        parameters=parameters, tags=parameters.get('tags', ['nominal']),
        q=spectrum.q, r=spectrum.r, qr=spectrum.q * spectrum.r,
        delta_diagonal=spectrum.delta_diagonal,
        diagnostics=spectrum.diagnostics,
        physical=physical_diagnostics(geometry, scen, fam, idx, sig, spectrum),
        spectral_bins=[asdict(b) for b in compressed.bins],
        endpoints=endpoints, value=value,
        source_fields=(dict(t1='rows[].J_rho_mean_1', tkappa='rows[].J_rho_mean_kappa',
                            value='rows[].V_rho_mean') if cohort == 'nominal'
                       else dict(endpoints='new independent lowrank evaluations; not stored in frozen family',
                                 value='rows[].V_meas')),
    )


def summarize(rows):
    eps = [e for row in rows for e in row['endpoints'].values()]
    vs = [row['value'] for row in rows]
    return dict(
        n_cells=len(rows), n_endpoints=len(eps),
        endpoint_bound_over_observed=distribution(e['bound_over_observed'] for e in eps
                                                   if e['bound_over_observed'] is not None),
        endpoint_tightness_le_10_count=sum(e['tightness_le_10'] for e in eps),
        endpoint_undefined_ratio_count=sum(e['bound_over_observed'] is None for e in eps),
        endpoint_bound_violations=sum(not e['bounds_hold'] for e in eps),
        endpoint_absolute_bound_violations=sum(not e['absolute_bound_holds'] for e in eps),
        endpoint_relative_error_upper=distribution(e['relative_to_prediction_upper'] for e in eps),
        endpoint_observed_remainder=distribution(e['observed_remainder'] for e in eps),
        identity_risk_relative_error=distribution(e['identity_risk_relative_error'] for e in eps),
        nominal_frozen_replay_relative_error=distribution(e['frozen_replay_relative_error'] for e in eps
                                                         if e['frozen_replay_relative_error'] is not None),
        value_bound_over_observed=distribution(v['bound_over_observed'] for v in vs
                                                if v['bound_over_observed'] is not None),
        value_tightness_le_10_count=sum(v['tightness_le_10'] for v in vs),
        value_undefined_ratio_count=sum(v['bound_over_observed'] is None for v in vs),
        value_bound_violations=sum(not v['bounds_hold'] for v in vs),
        value_observed_absolute_error=distribution(v['observed_absolute_error'] for v in vs),
        value_absolute_error_upper=distribution(v['absolute_error_upper'] for v in vs),
        value_frozen_replay_absolute_error=distribution(v['frozen_replay_absolute_error'] for v in vs),
        value_error_exceeds_0_05_count=sum(v['error_exceeds_0_05'] for v in vs),
        value_certificate_rules_out_0_05_accuracy_count=sum(v['certificate_rules_out_0_05_accuracy'] for v in vs),
        value_certified_absolute_accuracy_0_05_count=sum(v['certified_absolute_accuracy_0_05'] for v in vs),
        prior_residual_squared=distribution(row['diagnostics']['prior_residual_squared'] for row in rows),
        negative_roundoff_eigenvalue_count=sum(row['diagnostics']['spectrum_negative_roundoff_count'] for row in rows),
    )


def data_fingerprint(root, meta, name, scen):
    files = {f'OLAT/{name}/Lights/{i:03d}/com_masked_thumbnail/A1.png':
             sha(root / 'OLAT' / name / 'Lights' / f'{i:03d}' / 'com_masked_thumbnail/A1.png')
             for i in range(142)}
    light_path = root / 'light_pos.npy'
    if not light_path.exists():
        light_path = meta / 'light_pos.npy'
    return dict(
        png_files_sha256=files, png_tree_sha256=canonical_hash(files),
        lighting_path=str(light_path.resolve()), lighting_sha256=sha(light_path),
        active_light_indices=scen.sel.tolist(), selected_masked_pixel_indices=scen.pidx.tolist(),
        n_pixels=int(len(scen.rho)), n_active_lights=int(len(scen.sel)),
        noise_fit_a=float(scen.a), noise_fit_b=float(scen.b),
        finf_min=float(scen.Finf_diag.min()), finf_max=float(scen.Finf_diag.max()),
    )


def run(data_root=None, data_meta=None):
    start = time.time()
    before = frozen_snapshot()
    if len(before) != EXPECTED_FROZEN_COUNT or canonical_hash(before) != EXPECTED_FROZEN_TREE:
        raise RuntimeError('the 75-file frozen baseline differs from the task-start fingerprint')
    cfg_path = REPO / 'configs/goal_orientation.yaml'
    fam_cfg_path = REPO / 'configs/corruption_family_sensitivity.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    fcfg = yaml.safe_load(fam_cfg_path.read_text(encoding='utf-8'))
    goal = json.loads(GOAL.read_text(encoding='utf-8'))
    frozen_family = json.loads(FAMILY.read_text(encoding='utf-8'))
    root = Path(data_root or cfg['data_root']).resolve()
    meta = Path(data_meta or cfg['data_meta']).resolve()
    kappa, K = float(cfg['kappa']), int(cfg['K_lights'])
    grid = build_grid(fcfg)
    if grid != frozen_family['grid'] or cfg['cohort'] != fcfg['cohort']:
        raise RuntimeError('family/cohort definitions differ from frozen source')
    nominal_lookup = {(r['object'], float(r['level'])): r for r in goal['rows']}
    family_lookup = {(r['object'], r['grid_idx']): r for r in frozen_family['rows']}
    rows, scenes = [], {}
    for obj_idx, name in enumerate(cfg['cohort']):
        obj = load_object(root, name, data_meta=meta)
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg['noise_fit_convention'])
        del obj
        blk = build_state(scen, 0.5, kappa, K)
        geometry, a, gauge = scene_geometry(scen, blk)
        scenes[name] = dict(**data_fingerprint(root, meta, name, scen),
                            observation_gauge_relative_residual=gauge,
                            geometry=geometry.diagnostics)
        for level in cfg['levels']:
            level = float(level)
            parameters = dict(sig_logI=level, sig_dir_deg=level, het_sigma=0.0,
                              rho_c=0.0, tags=['nominal'], level=level)
            rows.append(evaluate_cell(scen, blk, geometry, a, parameters, kappa,
                                       int(fcfg['family_seed']), nominal_lookup[(name, level)],
                                       'nominal', f'nominal/{name}/{level:g}'))
        for gi, point in enumerate(grid):
            rows.append(evaluate_cell(scen, blk, geometry, a, point, kappa,
                                       int(fcfg['family_seed']), family_lookup[(name, gi)],
                                       'family', f'family/{name}/{gi}'))
        print(f'[M14] {name}: 4 nominal + {len(grid)} C9 cells ({time.time()-start:.1f}s)', flush=True)
    nominal = [r for r in rows if r['cohort'] == 'nominal']
    family = [r for r in rows if r['cohort'] == 'family']
    direction = [r for r in family if 'dir_only' in r['tags']]
    worst_dir = max(direction, key=lambda r: r['value']['observed_absolute_error'])
    overall = summarize(rows)
    nominal_summary = summarize(nominal)
    condition_keys = {(r['object'], *(r['parameters'][key] for key in
                                     ('sig_logI', 'sig_dir_deg', 'het_sigma', 'rho_c')))
                      for r in rows}
    overlap_scope = dict(
        n_records=len(rows), n_distinct_object_parameter_conditions=len(condition_keys),
        n_repeated_anchor_records=len(rows) - len(condition_keys),
        definition='condition key=(object,sig_logI,sig_dir_deg,het_sigma,rho_c); same seeds and kappa',
        note='539 records contain 11 repeated nominal-level-0.5/C9-anchor conditions: 528 distinct conditions, not independent samples; family tags overlap too',
    )
    after = frozen_snapshot()
    changed = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
    if changed:
        raise RuntimeError(f'frozen results changed during run: {changed}')
    nominal_ok = (nominal_summary['endpoint_bound_violations'] == 0
                  and nominal_summary['value_bound_violations'] == 0
                  and nominal_summary['endpoint_tightness_le_10_count'] == 88
                  and nominal_summary['value_tightness_le_10_count'] == 44)
    replay_ok = (nominal_summary['nominal_frozen_replay_relative_error']['max'] <= REPLAY_RTOL
                 and overall['value_frozen_replay_absolute_error']['max'] <= REPLAY_RTOL)
    all_bounds_ok = overall['endpoint_bound_violations'] == 0 and overall['value_bound_violations'] == 0
    passed = nominal_ok and replay_ok and all_bounds_ok and worst_dir['value']['certificate_rules_out_0_05_accuracy']
    source_paths = [cfg_path, fam_cfg_path, GOAL, FAMILY,
                    REPO / 'experiments/family_e_diag.py',
                    REPO / 'experiments/goal_orientation.py',
                    REPO / 'experiments/corruption_family_sensitivity.py',
                    REPO / 'experiments/openillumination_validation.py',
                    REPO / 'src/calibinfo/models/corruption_family.py',
                    REPO / 'src/calibinfo/information/lowrank.py',
                    REPO / 'experiments/theory_remainder.py', Path(__file__).resolve()]
    out = dict(
        gate='P-TWO-TERM-REMAINDER', reserved_claim='M14',
        analysis_status='complete_scoped' if passed else 'partial',
        published_status='experimental; M14 reserved for coordinator integration',
        mathematics=dict(
            identity='J_a(t)=1/(t*q)+r+delta_D+e^T(t*I+S_perp)^(-1)e',
            assumptions=['whitened fixed-normal linear model', 'D=A^T A positive definite',
                         'Sigma0 positive definite, t>0', 'unit a, exact Aa=Bc'],
            residual='delta_D=a^T D^-1 a-r>=0; e=C^T g-C^-1 c/q, Sigma0=C C^T',
            non_circularity='certificate sees q,r,delta_D and fixed spectral bins/masses only; never observed J or error',
            bin_rule='[2^j-1,2^(j+1)-1], j=floor(log2(1+lambda)); no data-fitted bin width',
            endpoint_theorem='R_t <= U_t <= 2 R_t for t>=1 in exact arithmetic',
            value_scope='valid joint-endpoint enclosure; no universal factor-10 theorem for V (spectral cancellation can make true error zero)',
            exact_direction_only='Sigma_logI=0 is outside the finite-PD theorem; tested controls use the existing C9 1e-6 convention; an exact limit counterexample is proved separately',
        ),
        protocol=dict(
            kappa=kappa, nominal_levels=cfg['levels'], cohort=cfg['cohort'],
            family_grid=grid, family_seed=int(fcfg['family_seed']),
            scene_rng_spec=cfg['scene_rng_spec'], noise_fit_convention=cfg['noise_fit_convention'],
            tightness_threshold=TIGHTNESS_THRESHOLD,
            numerical_rtol=NUMERICAL_RTOL, numerical_atol=NUMERICAL_ATOL,
            frozen_replay_rtol=REPLAY_RTOL,
            floating_point_scope='analytic certificate evaluated in double precision; tolerances only guard numerical replay, not tune the bound',
            psd_guard_scale='PSD_RTOL*||M||_F*lambda_max(Sigma0): propagation of the parent-Gram accepted lower bound through C^T T C, not relative scaling by a possibly zero T',
            overlap_scope=overlap_scope,
            study_design='all frozen nominal cells and all existing C9 grid points; one-object exploratory check before implementation; not claimed prospectively preregistered',
            family_e_diag_scope='read for S_pred/S_real definitions; these are reconstruction ranking diagnostics and are NOT repurposed as risk-error or certificate metrics here',
        ),
        acceptance=dict(
            nominal_44_cells_88_endpoints_verified=nominal_ok,
            frozen_replay_verified=replay_ok, all_tested_bounds_hold=all_bounds_ok,
            heterogeneous_and_coupled_covered=True,
            worst_direction_only_small_accuracy_excluded=worst_dir['value']['certificate_rules_out_0_05_accuracy'],
            historical_results_unchanged=not changed,
            not_claimed=['universal V bound/observed <=10', 'gauge identity alone implies small remainder',
                         'geometry-free degree/logI-ratio accuracy law', 'nonlinear reconstruction-error certificate',
                         'rigorous directed-rounding floating-point enclosure'],
        ),
        summary=dict(overall=overall, nominal=nominal_summary, family=summarize(family),
                     nominal_by_level={str(lv): summarize([r for r in nominal if r['parameters']['level'] == lv])
                                       for lv in cfg['levels']},
                     family_by_tag={tag: summarize([r for r in family if tag in r['tags']])
                                    for tag in sorted({tag for r in family for tag in r['tags']})},
                     direction_only_worst=dict(cell_id=worst_dir['cell_id'], object=worst_dir['object'],
                                               parameters=worst_dir['parameters'], value=worst_dir['value'],
                                               endpoints=worst_dir['endpoints'],
                                               prior_residual_squared=worst_dir['diagnostics']['prior_residual_squared'])),
        scenes=scenes, rows=rows,
        manifest=dict(
            data_root=str(root), data_meta=str(meta),
            frozen_results_count_before=len(before), frozen_results_count_after=len(after),
            frozen_tree_sha256_before=canonical_hash(before), frozen_tree_sha256_after=canonical_hash(after),
            frozen_results_sha256=before, changed_frozen_paths=changed,
            source_sha256={p.relative_to(REPO).as_posix(): sha(p) for p in source_paths},
            git_sha_at_run=subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                          capture_output=True, text=True, check=True).stdout.strip(),
            numpy_version=np.__version__, scipy_version=scipy.__version__,
            python_version=sys.version.split()[0], elapsed_s=round(time.time() - start, 3),
            reproduction_command=f'python -B "{Path(__file__).resolve()}" --data-root "{root}" --data-meta "{meta}"',
        ),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print('[M14] status:', out['analysis_status'], flush=True)
    print('[M14] nominal:', json.dumps(nominal_summary, ensure_ascii=True), flush=True)
    print('[M14] direction-only worst:', json.dumps(out['summary']['direction_only_worst'], ensure_ascii=True), flush=True)
    print(f'[M14] wrote {OUT}', flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--data-meta')
    args = parser.parse_args()
    run(args.data_root, args.data_meta)
