# status: experimental — NOT part of the published results
"""P-JOINT-INFORMATION: independent joint-scene covariance and geometry checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]

from calibinfo.information.schur import delta_f
from calibinfo.metrics.spectral_criteria import identifiable_subspace
from experiments.theory_joint import synthetic_joint_scene


def _relative(a, b):
    return float(np.linalg.norm(a-b) / np.linalg.norm(b))


def _cov_report(errors, predicted, limits):
    empirical = np.cov(errors, rowvar=False, ddof=1)
    ratio = np.diag(empirical) / np.diag(predicted)
    return {
        'n_samples': len(errors), 'variance_ratios': ratio.tolist(),
        'min_ratio': float(ratio.min()), 'max_ratio': float(ratio.max()),
        'median_ratio': float(np.median(ratio)),
        'full_covariance_relative_frobenius_error': _relative(empirical, predicted),
        'mean_squared_bias_over_predicted_variance':
            (np.mean(errors, axis=0)**2 / np.diag(predicted)).tolist(),
        'within_preregistered_interval': bool(np.all(
            (ratio >= limits[0]) & (ratio <= limits[1]))),
    }


def run(config_path, out_path):
    started = time.monotonic()
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if cfg['uniform_precision_multipliers'] != [1.0, 10.0]:
        raise ValueError("this registered experiment requires multiplier endpoints [1,10]")
    model = synthetic_joint_scene()
    rng = np.random.default_rng(cfg['seed'])
    sigma = cfg['sigma']
    limits = cfg['variance_ratio_interval']
    nscene = 3*model.P
    z0 = np.zeros(3*(model.P+model.L))
    _, rawA, rawB = model.physical(z0[:nscene], z0[nscene:], jacobian=True)
    J = np.c_[rawA, rawB]
    numerical = np.empty_like(J)
    step = cfg['jacobian_step']
    for j in range(len(z0)):
        dz = np.zeros_like(z0)
        dz[j] = step
        numerical[:, j] = ((model.physical(dz[:nscene], dz[nscene:])
                            - model.physical(-dz[:nscene], -dz[nscene:]))
                           / (2*step)).ravel()
    gauge_x, gauge_c = model.gauge_generators()
    F0, _, _ = delta_f(model.A, model.B, 0)
    quotient_rank, quotient, _ = identifiable_subspace(F0)
    scale = model.scale_gauge_report()
    zero_gauge = model.A @ gauge_x + model.B @ gauge_c
    constraints = {
        'P': model.P, 'L': model.L, 'intrinsic_scene_dimension': nscene,
        'ambient_rho_and_normal_dimension': 4*model.P,
        'rank_A': int(np.linalg.matrix_rank(model.A)),
        'rank_B': int(np.linalg.matrix_rank(model.B)),
        'rank_full_unpenalized_jacobian': int(np.linalg.matrix_rank(np.c_[model.A, model.B])),
        'uncalibrated_gl3_gauge_dimension': int(np.linalg.matrix_rank(np.r_[gauge_x, gauge_c])),
        'uncalibrated_scene_gauge_dimension': int(np.linalg.matrix_rank(gauge_x)),
        'flat_prior_identifiable_scene_dimension': quotient_rank,
        'proper_prior_identifiable_scene_dimension': model.fixed_basis()[1]['rank'],
        'gauge_max_abs_residual': float(np.max(np.abs(zero_gauge))),
        'quotient_gauge_orthogonality_norm': float(np.linalg.norm(quotient.T @ gauge_x)),
        'scale_alignment_residual': scale['alignment_residual'],
        'existing_gauge_response_max_abs_error': scale['gauge_response_max_abs_error'],
        'jacobian_max_abs_error': float(np.max(np.abs(numerical-J))),
        'jacobian_relative_frobenius_error': _relative(numerical, J),
        'minimum_lit_cosine': float(np.min(model.directions @ model.normals.T)),
    }
    rows = []
    for precision in cfg['uniform_precision_multipliers']:
        t = np.full(model.L, precision)
        Lambda = model.precision(t)
        F = model.fisher(t)
        Sigma_c = sigma**2 * np.linalg.inv(Lambda)
        sensor = rng.normal(size=(cfg['linear_samples'], len(model.A))) * sigma
        nuisance = rng.multivariate_normal(np.zeros(3*model.L), Sigma_c,
                                           size=cfg['linear_samples'])
        observations = sensor + nuisance @ model.B.T
        estimate = model.linear_profile(observations, t, sigma)
        predicted = sigma**2 * np.linalg.inv(F)
        linear = _cov_report(estimate, predicted, limits)
        nonlinear_estimates, statuses = [], []
        visibility_flips = 0
        for i in range(cfg['nonlinear_samples']):
            c = rng.multivariate_normal(np.zeros(3*model.L), Sigma_c)
            physical = model.physical(np.zeros(nscene), c)
            visibility_flips += int(np.sum((physical > 0) != model.visibility))
            noisy = physical + rng.normal(size=physical.shape) * sigma / np.sqrt(model.weights)
            xhat, status = model.nonlinear_profile(noisy, t, sigma)
            nonlinear_estimates.append(xhat)
            statuses.append(status)
        nonlinear = _cov_report(np.asarray(nonlinear_estimates), predicted, limits)
        nonlinear['optimizer_success_count'] = sum(s['success'] for s in statuses)
        nonlinear['optimizer_failure_count'] = sum(not s['success'] for s in statuses)
        nonlinear['max_nfev'] = max(s['nfev'] for s in statuses)
        nonlinear['max_optimality'] = max(s['optimality'] for s in statuses)
        nonlinear['shadow_visibility_flips'] = visibility_flips
        H_all = np.eye(nscene)
        H_albedo = np.eye(nscene)[::3]
        H_normal = np.eye(nscene)[np.array([j for j in range(nscene) if j % 3])]
        tasks = {}
        for name, H in [('all_intrinsic', H_all), ('log_albedo', H_albedo),
                        ('normal_tangent', H_normal)]:
            dense = float(np.trace(H @ np.linalg.solve(F, H.T)))
            lowrank = model.task_risk_lowrank(t, H)
            tasks[name] = {'risk_dense': dense, 'risk_lowrank': lowrank,
                           'relative_difference': abs(dense-lowrank)/dense}
        rows.append({'precision_multiplier': precision, 'linear': linear,
                     'nonlinear': nonlinear, 'tasks': tasks,
                     'smallest_information_eigenvalue': float(np.linalg.eigvalsh(F)[0])})
        print(f'precision={precision}: linear [{linear["min_ratio"]:.4f},'
              f'{linear["max_ratio"]:.4f}], nonlinear [{nonlinear["min_ratio"]:.4f},'
              f'{nonlinear["max_ratio"]:.4f}]', flush=True)
    F1, Fk = model.fisher(np.ones(model.L)), model.fisher(10*np.ones(model.L))
    values = {name: 1-rows[1]['tasks'][name]['risk_dense']/rows[0]['tasks'][name]['risk_dense']
              for name in rows[0]['tasks']}
    t = np.full(model.L, 1.4)
    certificate = model.certificate(t, np.eye(nscene), budget=4.0, kappa=3.0)
    import itertools
    feasible_values = []
    for S in itertools.combinations(range(model.L), 2):
        td = np.ones(model.L)
        td[list(S)] = 3
        feasible_values.append(model.risk_and_gradient(td, np.eye(nscene))[0])
    summary = {
        'gate': cfg['experiment'], 'claim': cfg['claim'],
        'status': 'independent_synthetic_validation', 'config': cfg,
        'dimension_and_gauge': constraints, 'rows': rows,
        'ceiling': {'kappa': 10.0, 'ceiling': .9, 'task_values': values,
                    'min_eigenvalue_kappa_F1_minus_Fk': float(np.linalg.eigvalsh(10*F1-Fk)[0]),
                    'all_task_values_below_ceiling': bool(all(0 <= v <= .9+1e-12 for v in values.values()))},
        'certificate': {'risk': certificate['risk'], 'gap': certificate['gap'],
                        'lower_bound': certificate['lower_bound'],
                        'enumerated_discrete_count': len(feasible_values),
                        'best_discrete_risk': min(feasible_values),
                        'minimum_discrete_slack': min(feasible_values)-certificate['lower_bound']},
        'acceptance': {
            'linear': all(row['linear']['within_preregistered_interval'] for row in rows),
            'nonlinear': all(row['nonlinear']['within_preregistered_interval']
                             and row['nonlinear']['optimizer_failure_count'] == 0 for row in rows)},
        'scope': cfg['scope'],
        'provenance': {'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                       'implementation_sha256': hashlib.sha256(
                           (REPO/'experiments/theory_joint.py').read_bytes()).hexdigest(),
                       'elapsed_s': time.monotonic()-started,
                       'no_historical_results_modified': True},
    }
    out = Path(out_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(REPO/'configs/joint_information_extension_20260918.json'))
    parser.add_argument('--out', default=str(REPO/'results/theory_extension_20260918/joint_information_extension.json'))
    args = parser.parse_args()
    run(args.config, args.out)
