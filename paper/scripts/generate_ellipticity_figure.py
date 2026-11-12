#!/usr/bin/env python3
"""Generate and verify the manuscript's Rayleigh ellipticity node sensitivities.

Uses the current solver, not a historical reference snapshot. Numerical outputs
are separate from frozen inputs; plotting is performed by the manuscript repo.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical(materials):
    rho, A, C, F, L, N = np.asarray(materials).T
    return np.array([rho, np.sqrt(A/rho), np.sqrt(C/rho),
                     np.sqrt(N/rho), np.sqrt(L/rho), F/(A-2*L)]).T


def material_row(p):
    rho, ph, pv, sh, sv, eta = p
    return [rho, rho*ph**2, rho*pv**2,
            eta*rho*(ph**2-2*sv**2), rho*sv**2, rho*sh**2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disper26-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.disper26_root.resolve()
    sys.path.insert(0, str(source))
    from paper.provenance import validate_output
    import disper26 as api
    import disper26py as dp
    sys.path.insert(0, str(source/'paper/scripts'))
    import generate_kernel_figures as g
    g.dp = dp
    assert Path(dp.__file__).resolve().parent == source
    out = validate_output(args.output_dir)
    if args.model.resolve() == out/'prem_ellipticity_signed_nodes.npz':
        parser.error('Output would overwrite the model')
    out.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.model.read_text())
    params = physical(model['materials'])
    arrays = {'depth_km': np.array(model['depth_km']), 'physical_parameters': params,
              'frequency_hz': np.array([.05, .1])}
    names = ['rho', 'alpha_H', 'alpha_V', 'beta_H', 'beta_V', 'eta']
    report = {'definition': 'epsilon = y3/y1 at the solid free surface; z upward, ux=i*y3*exp(i(kx-omega*t)); negative retrograde, positive prograde',
              'mode': 0, 'interpolation': 'density and stiffness, linear between nodes',
              'parameter_order': names, 'levels': list(range(9)),
              'relative_fd_steps': [1e-4, 5e-5], 'root_tolerance': 1e-12,
              'root_polishing': {'method': 'scipy.optimize.brentq',
                                 'xtol': 1e-14, 'rtol': 1e-14,
                                 'bracket_half_width': 1e-5,
                                 'rationale': 'Resolve the small root shifts needed for finite '
                                     'differences; certify the bracket with integer counts 0 and 1.'},
              'normalized_tolerance': 1e-5,
              'tolerance_rationale': 'Reuse the manuscript node-kernel criterion; '
                  'each nonzero physical-parameter column is scaled by its peak sensitivity.',
              'model_sha256': sha(args.model), 'generator_sha256': sha(__file__),
              'extension_sha256': sha(dp.__file__), 'cases': {}}
    sources = [p for folder in ('src', 'include', 'disper26')
               for p in (source/folder).rglob('*')
               if p.is_file() and p.suffix in ('.cpp', '.hpp', '.h', '.py')]
    sources += [source/'setup.py', source/'paper/scripts/generate_kernel_figures.py']
    report['source_files'] = {str(p.relative_to(source)): sha(p) for p in sorted(sources)}
    report['source_tree_sha256'] = hashlib.sha256(json.dumps(
        report['source_files'], sort_keys=True).encode()).hexdigest()
    mode_checks = 0
    maximum_root_correction = 0.0

    def values(m, frequency, guesses=None):
        nonlocal mode_checks, maximum_root_correction
        roots, hv = [], []
        for level in range(9):
            mat, thickness, _ = g.mesh(m, level)
            rows = np.vstack([mat, m['materials'][-1]])
            rho, A, C, F, L, N = rows.T
            pancake = api.create_model(np.r_[0., np.cumsum(thickness)], kind='pancake',
                                       rho=rho, A=A, C=C, F=F, L=L, N=N)
            guess = guesses[level] if guesses is not None else (roots[-1] if roots else 3.8)
            solution = api.compute_dispersion(pancake, period=1/frequency, mode=0,
                outputs=('phase_velocity', 'hv_ratio'),
                options=api.Options(method='continuation', verify_mode=True,
                                    c_guess=float(guess), root_tolerance=1e-12))
            assert solution.diagnostics.mode_verified, 'Unconfirmed mode'
            # A mode label does not ensure that the root is accurate enough
            # to resolve the tiny changes used in finite differences. Polish
            # the secular zero with an independent interval-based solver.
            layers = [g.parameters(row) for row in mat]
            halfspace = g.parameters(m['materials'][-1])
            h = thickness.tolist()
            omega = 2*np.pi*frequency
            lo, hi = solution.phase_velocity-1e-5, solution.phase_velocity+1e-5
            assert dp.compute_mode_count(layers, h, halfspace, omega, lo, 'Rayleigh') == 0
            assert dp.compute_mode_count(layers, h, halfspace, omega, hi, 'Rayleigh') == 1
            def secular(c):
                return dp.compute_dispersion_value(layers, h, halfspace, omega, c, 'Rayleigh')
            root = brentq(secular, lo, hi, xtol=1e-14, rtol=1e-14)
            maximum_root_correction = max(maximum_root_correction,
                                          abs(root-solution.phase_velocity))
            mode_checks += 1
            roots.append(root)
            hv.append(dp.compute_hv_ratio(layers, h, halfspace, omega, root, 'Rayleigh'))
        assert np.all(np.isfinite(hv))
        return np.array(roots), np.array(hv)

    for f, key in ((.05, 'Rayleigh_05'), (.1, 'Rayleigh_10')):
        roots, raw_e = values(model, f)
        gradients, ellipticities = [], []
        for level in (7, 8):
            r = dp.compute_kernels_ad_grid(model['depth_km'],
                [g.parameters(a) for a in model['materials']], model['is_discontinuity'],
                1/f, wave_type='Rayleigh', c_guess=float(g.extrap(roots[:level+1])),
                sensitivity_mode='VTI', max_steps=level, tolerance=1e-12,
                compute_c=False, want_kernel_u=False, want_kernel_hv=True,
                kernel_extrapolation=True, phase_velocity_roots=roots[:level+1].tolist(),
                phase_accepted_level=level)
            stiffness = np.zeros((len(params), 6))
            seen = set()
            for k in r[5]:
                if k.parameter_name in g.K_NAMES and 0 <= k.layer_index < len(params):
                    index = g.K_NAMES.index(k.parameter_name)
                    stiffness[k.layer_index, index] = k.d_HV
                    seen.add((k.layer_index, index))
            assert len(seen) == len(params)*6
            gradient = g.velocity_basis(stiffness, np.array(model['materials']))
            assert np.all(np.isfinite(gradient))
            assert np.all(gradient[:, 3] == 0), 'Rayleigh depends on beta_H'
            e = float(g.extrap(raw_e[:level+1]))
            assert abs(e-r[2])/abs(e) < 1e-12, 'H/V value mismatch between paths'
            arrays[key+f'_gradient_l{level}'] = gradient
            arrays[key+f'_epsilon_l{level}'] = np.array(e)
            gradients.append(gradient)
            ellipticities.append(e)
        grad = gradients[-1]
        scales = np.max(abs(grad), axis=0)
        active = scales > 0
        conv = np.zeros(6)
        conv[active] = np.max(abs(grad-gradients[0]), axis=0)[active]/scales[active]
        case = {'epsilon': ellipticities[-1], 'phase_velocity': float(g.extrap(roots)),
                'epsilon_l7_l8_relative': abs(ellipticities[1]-ellipticities[0])/abs(ellipticities[1]),
                'gradient_l7_l8_normalized': conv.tolist(), 'finite_differences': []}
        arrays[key+'_roots'] = roots
        arrays[key+'_epsilon_raw'] = raw_e
        arrays[key+'_normalized_gradient'] = params*grad/ellipticities[-1]
        print(key, 'AD and convergence', case, flush=True)
        # All surface parameters, and beta_V at both sides of every interface
        # and at the deepest node (which also controls the half-space).
        directions = [(0, j) for j in range(6)] + [(i, 4) for i in range(1, len(params))]
        for node, column in directions:
            fd, errors = [], []
            for eps in report['relative_fd_steps']:
                pair = []
                for sign in (-1, 1):
                    m = dict(model, materials=np.array(model['materials']).copy())
                    p = params[node].copy()
                    p[column] *= 1+sign*eps
                    m['materials'][node] = material_row(p)
                    rp, ep = values(m, f, roots)
                    tag = f'{key}_fd_n{node}_p{column}_h{eps}_s{sign}'
                    arrays[tag+'_roots'] = rp
                    arrays[tag+'_epsilon_raw'] = ep
                    pair.append(float(g.extrap(ep)))
                derivative = (pair[1]-pair[0])/(2*eps*params[node, column])
                fd.append(derivative)
                if column == 3:
                    assert derivative == 0, 'H/V changes with beta_H'
                    errors.append(0.)
                else:
                    errors.append(abs(derivative-grad[node, column])/scales[column])
            entry = {'node': node, 'parameter': names[column], 'ad': grad[node, column],
                     'fd': fd, 'normalized_errors': errors,
                     'signs_match': all(np.sign(v) == np.sign(grad[node, column]) for v in fd)}
            case['finite_differences'].append(entry)
            print(key, node, names[column], 'FD errors', errors, flush=True)
        case['maximum_fd_normalized_error'] = max(
            max(v['normalized_errors']) for v in case['finite_differences'])
        report['cases'][key] = case
    report['confirmed_mode_evaluations'] = mode_checks
    report['root_polishing']['maximum_phase_velocity_correction'] = maximum_root_correction
    report['passed'] = all(
        c['epsilon_l7_l8_relative'] < 1e-5 and max(c['gradient_l7_l8_normalized']) < 1e-5
        and c['maximum_fd_normalized_error'] < 1e-5
        and all(v['signs_match'] for v in c['finite_differences'])
        for c in report['cases'].values())
    data = out/'prem_ellipticity_signed_nodes.npz'
    np.savez_compressed(data, **arrays)
    report['data_sha256'] = sha(data)
    (out/'prem_ellipticity_signed_nodes_report.json').write_text(json.dumps(report, indent=2)+'\n')
    assert report['passed'], 'Validation failed; outputs retained for diagnosis'
    print('PASS', mode_checks, 'confirmed mode evaluations', flush=True)


if __name__ == '__main__':
    main()
