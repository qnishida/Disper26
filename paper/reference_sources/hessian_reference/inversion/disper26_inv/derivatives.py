"""Exact chain rule from the node stiffness basis to log-velocity coordinates.

The engine returns gradients/Hessians in the node stiffness basis
a = (A, C, F, L, N, Rho). The velocity part of the inversion works in
x = log-perturbations of (rho, vph, vpv, vsh, vsv, eta) at nodes (then pulled
back through the InversionBasis operator P). An optional interface-depth
coordinate is appended by the depth backend. In log coordinates every
stiffness component is a sum of at most two pure exponentials:

    A   = A0   * exp(xr + 2*xph)
    C   = C0   * exp(xr + 2*xpv)
    F   = eta*A - 2*eta*L
        = (eta0*A0) * exp(xr + 2*xph + xe)  -  (2*eta0*L0) * exp(xr + 2*xsv + xe)
    L   = L0   * exp(xr + 2*xsv)
    N   = N0   * exp(xr + 2*xsh)
    Rho = rho0 * exp(xr)

so first and second derivatives of a(x) are exact closed forms
(d a_k = sum_i s_i w_i, d^2 a_k = sum_i s_i w_i w_i^T) and

    g_x = T^T g_a,   H_x = T^T H_a T + sum_k g_a[k] * d^2 a_k/dx^2

with the curvature-of-the-map term block-diagonal in nodes.

Ordering conventions: FIELD_ORDER matches model.VELOCITY_FIELDS; node log-space
vectors are parameter-major (index = p * n_fwd + j), matching InversionBasis.
"""

from __future__ import annotations

import numpy as np

from .model import VELOCITY_FIELDS, ForwardGridModel, InversionBasis

FIELD_ORDER = VELOCITY_FIELDS  # ("rho", "vph", "vpv", "vsh", "vsv", "eta")
STIFF_ORDER = ("A", "C", "F", "L", "N", "Rho")
N_FIELDS = len(FIELD_ORDER)
N_STIFF = len(STIFF_ORDER)

# Exponent vectors w over (xr, xph, xpv, xsh, xsv, xe) for each exponential term.
_W = {
    "A": [("A", (1.0, 2.0, 0.0, 0.0, 0.0, 0.0))],
    "C": [("C", (1.0, 0.0, 2.0, 0.0, 0.0, 0.0))],
    "F": [("etaA", (1.0, 2.0, 0.0, 0.0, 0.0, 1.0)), ("m2etaL", (1.0, 0.0, 0.0, 0.0, 2.0, 1.0))],
    "L": [("L", (1.0, 0.0, 0.0, 0.0, 2.0, 0.0))],
    "N": [("N", (1.0, 0.0, 0.0, 2.0, 0.0, 0.0))],
    "Rho": [("rho", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))],
}


def _term_coefficients(model: ForwardGridModel) -> dict:
    stiff = model.stiffness()
    return {
        "A": stiff["A"],
        "C": stiff["C"],
        "L": stiff["L"],
        "N": stiff["N"],
        "rho": stiff["rho"],
        "etaA": model.eta * stiff["A"],
        "m2etaL": -2.0 * model.eta * stiff["L"],
    }


def stiffness_jacobian(model: ForwardGridModel) -> np.ndarray:
    """T[j, k, p] = d a_k / d x_p at node j (exact)."""
    coeff = _term_coefficients(model)
    T = np.zeros((model.n_nodes, N_STIFF, N_FIELDS))
    for k, name in enumerate(STIFF_ORDER):
        for coeff_name, w in _W[name]:
            T[:, k, :] += np.outer(coeff[coeff_name], np.asarray(w))
    return T

def stiffness_curvature(model: ForwardGridModel) -> np.ndarray:
    """K2[j, k, p, q] = d^2 a_k / (d x_p d x_q) at node j (exact)."""
    coeff = _term_coefficients(model)
    K2 = np.zeros((model.n_nodes, N_STIFF, N_FIELDS, N_FIELDS))
    for k, name in enumerate(STIFF_ORDER):
        for coeff_name, w in _W[name]:
            w = np.asarray(w)
            K2[:, k, :, :] += coeff[coeff_name][:, None, None] * np.outer(w, w)[None, :, :]
    return K2


def _stiffness_index_map(parameter_names, layer_indices, n_fwd: int) -> np.ndarray:
    """positions[j, k] = row index of (STIFF_ORDER[k], node j) in the flat arrays."""
    lookup = {}
    for pos, (name, idx) in enumerate(zip(parameter_names, layer_indices)):
        lookup[(name, int(idx))] = pos
    positions = np.empty((n_fwd, N_STIFF), dtype=int)
    for j in range(n_fwd):
        for k, name in enumerate(STIFF_ORDER):
            key = (name, j)
            if key not in lookup:
                raise KeyError(f"missing stiffness entry {key} in Hessian/kernel output")
            positions[j, k] = lookup[key]
    return positions


def kernels_to_stiffness_gradient(kernels, n_fwd: int) -> np.ndarray:
    """g_a[j, k] from a SensitivityKernel list (params family, mode VTI)."""
    g = np.zeros((n_fwd, N_STIFF))
    seen = np.zeros((n_fwd, N_STIFF), dtype=bool)
    k_index = {name: k for k, name in enumerate(STIFF_ORDER)}
    for kernel in kernels:
        k = k_index.get(kernel.parameter_name)
        if k is None:
            # ``Depth`` kernels are mapped separately by the depth backend;
            # this helper intentionally returns only the six stiffness fields.
            continue
        j = int(kernel.layer_index)
        g[j, k] = float(kernel.d_c)
        seen[j, k] = True
    if not seen.all():
        missing = np.argwhere(~seen)[:3]
        raise KeyError(f"kernel list is missing entries, e.g. {missing.tolist()}")
    return g


def kernels_to_depth_gradient(kernels, pair_start: int) -> float:
    """Return ``dc/dz`` for one duplicated interface from a kernel list.

    The GRID binding labels an interface kernel with the first node index of
    its duplicated pair.  Requiring an exact match prevents accidentally
    using the sediment or another discontinuity when a model contains more
    than one interface.
    """
    pair_start = int(pair_start)
    matches = [
        kernel for kernel in kernels
        if kernel.parameter_name == "Depth" and int(kernel.layer_index) == pair_start
    ]
    if len(matches) != 1:
        raise KeyError(
            f"expected one Depth kernel for discontinuity pair {pair_start}, "
            f"found {len(matches)}"
        )
    return float(matches[0].d_c)


def gradient_log(model: ForwardGridModel, g_a: np.ndarray) -> np.ndarray:
    """G[p, j] = dc/dx_p at node j from the stiffness gradient g_a[j, k]."""
    T = stiffness_jacobian(model)
    return np.einsum("jk,jkp->pj", g_a, T)


def hessian_log(model: ForwardGridModel, g_a: np.ndarray, H_a_nodes: np.ndarray) -> np.ndarray:
    """H[p, j, q, m] in log coordinates from H_a_nodes[j, k, m, l] and g_a[j, k]."""
    T = stiffness_jacobian(model)
    H_x = np.einsum("jkp,jkml,mlq->pjqm", T, H_a_nodes, T, optimize=True)
    K2 = stiffness_curvature(model)
    curvature = np.einsum("jk,jkpq->jpq", g_a, K2)
    n_fwd = model.n_nodes
    for j in range(n_fwd):
        H_x[:, j, :, j] += curvature[j]
    return H_x


def hessian_result_to_node_arrays(model: ForwardGridModel, hess: dict):
    """(g_a[j,k], H_a[j,k,m,l]) from an engine.datum_hessian result dict."""
    n_fwd = model.n_nodes
    positions = _stiffness_index_map(hess["parameter_names"], hess["layer_indices"], n_fwd)
    g_flat = np.asarray(hess["gradient"], dtype=float)
    H_flat = np.asarray(hess["hessian"], dtype=float)
    g_a = g_flat[positions]
    H_a = H_flat[positions.reshape(-1)][:, positions.reshape(-1)].reshape(
        n_fwd, N_STIFF, n_fwd, N_STIFF
    )
    return g_a, H_a


def _log_field_operator(basis: InversionBasis, model: ForwardGridModel = None) -> np.ndarray:
    """Return the local field map, supporting nonlinear constrained bases."""
    if hasattr(basis, "log_field_operator"):
        return basis.log_field_operator(model)
    return basis.full_log_operator()


def project_gradient(
    G: np.ndarray, basis: InversionBasis, model: ForwardGridModel = None
) -> np.ndarray:
    """Full-field G[p, j] -> inversion gradient via the basis pullback."""
    if hasattr(basis, "full_log_operator"):
        return _log_field_operator(basis, model).T @ np.asarray(G, dtype=float).reshape(-1)
    rows = [G[FIELD_ORDER.index(name)] for name in basis.param_names]
    return basis.pullback_gradient(np.concatenate(rows))


def project_hessian(
    H_x: np.ndarray,
    basis: InversionBasis,
    model: ForwardGridModel = None,
    g_log: np.ndarray = None,
) -> np.ndarray:
    """Full-field H[p, j, q, m] -> inversion Hessian via the basis pullback."""
    if hasattr(basis, "full_log_operator"):
        n_fwd = H_x.shape[1]
        T = _log_field_operator(basis, model)
        result = T.T @ np.asarray(H_x, dtype=float).reshape(6 * n_fwd, 6 * n_fwd) @ T
        if g_log is not None and hasattr(basis, "log_field_curvature_contraction"):
            result += basis.log_field_curvature_contraction(model, g_log)
        return result
    sel = [FIELD_ORDER.index(name) for name in basis.param_names]
    n_fwd = H_x.shape[1]
    n_p = len(sel)
    H_sel = H_x[np.ix_(sel, range(n_fwd), sel, range(n_fwd))]
    result = basis.pullback_hessian(H_sel.reshape(n_p * n_fwd, n_p * n_fwd))
    if g_log is not None and hasattr(basis, "log_field_curvature_contraction"):
        result += basis.log_field_curvature_contraction(model, g_log)
    return result


def jacobian_row(model: ForwardGridModel, basis: InversionBasis, kernels) -> np.ndarray:
    """One row of the Jacobian dc/dm from a params-family kernel list."""
    g_a = kernels_to_stiffness_gradient(kernels, model.n_nodes)
    return project_gradient(gradient_log(model, g_a), basis, model=model)


def datum_hessian_inv(model: ForwardGridModel, basis: InversionBasis, hess: dict):
    """(g_inv, H_inv) in inversion coordinates from an engine.datum_hessian result."""
    g_a, H_a = hessian_result_to_node_arrays(model, hess)
    G_log = gradient_log(model, g_a)
    g_inv = project_gradient(G_log, basis, model=model)
    H_inv = project_hessian(hessian_log(model, g_a, H_a), basis, model=model, g_log=G_log)
    return g_inv, H_inv


# --- Reduced-basis direct Hessian (C++ seeds basis directions, not node-VTI
# unit vectors) -----------------------------------------------------------
#
# The C++ grid weak-form Hessian's node-VTI row order (compute_phase_velocity_
# hessian_weakform_*_grid_reduced's `gradient_node`/the `J` it expects) is
# node-major with modulus order [Rho, A, C, F, L, N] per node (matching
# build_grid_hessian_result_from_vti's VTI-mode parameter_names in
# src/grid_hessian.hpp). That is a permutation of this module's STIFF_ORDER
# ("A","C","F","L","N","Rho"); these two constants translate between them.
CPP_MODULUS_ORDER = ("Rho", "A", "C", "F", "L", "N")
_CPP_MODULUS_TO_STIFF_K = np.array([STIFF_ORDER.index(name) for name in CPP_MODULUS_ORDER])
_STIFF_K_TO_CPP_MODULUS = np.argsort(_CPP_MODULUS_TO_STIFF_K)


def basis_node_jacobian(model: ForwardGridModel, basis: InversionBasis) -> np.ndarray:
    """J[node*6+m, r] = d(node-VTI modulus m)/d(reduced basis param r).

    m is C++ node-VTI row order (CPP_MODULUS_ORDER), r is the basis's own
    parameter index (``basis.n_params``, e.g. 19 for
    ``InterfaceShiftBasis(ScaledShearBasis(...))``). This is T @ B evaluated at
    the model's CURRENT parameters (T = stiffness_jacobian(model), cheap and
    state-dependent; B = basis.full_log_operator(), fixed/cached geometry) --
    see docs/Weakform_AD_Hessian.md and derivatives.py's module docstring for
    the underlying chain rule.

    If the basis has a depth coordinate (``InterfaceShiftBasis``),
    ``full_log_operator`` already carries an all-zero column for it, so the
    corresponding row/column of ``J^T H_node J`` comes out zero -- consistent
    with the existing convention that the depth Hessian is supplied separately
    by the FD depth backend (depth_hessian.py) and overwritten in by the
    caller (see pipeline.py's handling of ``depth_hessian_column``).
    """
    T = stiffness_jacobian(model)          # (n_fwd, N_STIFF, N_FIELDS)
    B = _log_field_operator(basis, model) # (N_FIELDS*n_fwd, n_params), field-major rows
    n_fwd = model.n_nodes
    n_params = B.shape[1]
    B3 = np.asarray(B, dtype=float).reshape(N_FIELDS, n_fwd, n_params)  # (p, j, r)
    J_stiff = np.einsum("jkp,pjr->jkr", T, B3, optimize=True)  # (n_fwd, N_STIFF, n_params)
    J_cpp = J_stiff[:, _CPP_MODULUS_TO_STIFF_K, :]              # reorder k -> CPP_MODULUS_ORDER
    return J_cpp.reshape(n_fwd * 6, n_params)


def reduced_result_to_node_gradient(model: ForwardGridModel, gradient_node: np.ndarray) -> np.ndarray:
    """``gradient_node`` (n_nodes*6, C++ VTI row order) -> g_a[j,k] (STIFF_ORDER).

    Same layout conversion as ``hessian_result_to_node_arrays``, for the
    reduced C++ entry points that return a flat, fixed-order node gradient
    instead of ``parameter_names``/``layer_indices``.
    """
    n_fwd = model.n_nodes
    g_cpp = np.asarray(gradient_node, dtype=float).reshape(n_fwd, 6)
    return g_cpp[:, _STIFF_K_TO_CPP_MODULUS]


def hessian_curvature_only(model: ForwardGridModel, g_a: np.ndarray, basis: InversionBasis) -> np.ndarray:
    """The K2-based node-diagonal curvature correction from ``hessian_log``,
    already projected through the basis -- WITHOUT the ``T^T H_a T`` sandwich.

    ``project_hessian`` is a fixed linear sandwich (``B^T (.) B``), so it is
    linear in its argument: ``project_hessian(hessian_log(model, g_a, H_a),
    basis) == basis_node_jacobian(model, basis).T @ H_a @ basis_node_jacobian(model, basis)
    + hessian_curvature_only(model, g_a, basis)``. The first term is computed
    directly in C++ by seeding ``basis_node_jacobian``'s columns as the AD
    directions (see ``compute_phase_velocity_hessian_weakform_*_grid_reduced``);
    this function supplies the remaining piece, which only needs the cheap
    528-dim gradient, never the 528x528 Hessian.
    """
    K2 = stiffness_curvature(model)  # (n_fwd, N_STIFF, N_FIELDS, N_FIELDS)
    curvature = np.einsum("jk,jkpq->jpq", g_a, K2)  # (n_fwd, N_FIELDS, N_FIELDS)
    n_fwd = model.n_nodes
    H_x = np.zeros((N_FIELDS, n_fwd, N_FIELDS, n_fwd))
    for j in range(n_fwd):
        H_x[:, j, :, j] = curvature[j]
    result = project_hessian(H_x, basis, model=model)
    G_log = gradient_log(model, g_a)
    if hasattr(basis, "log_field_curvature_contraction"):
        result += basis.log_field_curvature_contraction(model, G_log)
    return result


def datum_hessian_inv_reduced(model: ForwardGridModel, basis: InversionBasis, reduced: dict):
    """(g_inv, H_inv) in inversion coordinates from a *_grid_reduced C++ result.

    ``reduced`` is the dict returned by
    ``compute_phase_velocity_hessian_weakform_rayleigh_grid_reduced`` /
    ``..._love_grid_reduced`` (keys ``hessian_reduced`` (P x P) and
    ``gradient_node`` (n_nodes*6, unprojected)). Mathematically identical to
    ``datum_hessian_inv`` on the equivalent non-reduced result, to
    round-off -- see tests/python/verify_inv_derivatives_reduced.py.
    """
    g_a = reduced_result_to_node_gradient(model, reduced["gradient_node"])
    g_inv = project_gradient(gradient_log(model, g_a), basis, model=model)
    H_inv = np.asarray(reduced["hessian_reduced"], dtype=float) + hessian_curvature_only(model, g_a, basis)
    return g_inv, H_inv


def gradient_consistency(model, basis, kernels, hess) -> float:
    """Max relative difference between kernel-based and Hessian-based gradients.

    Both are exact for the same params-family model; a mismatch beyond solver
    tolerance indicates an ordering/basis bug and must be treated as fatal.
    """
    g_kernel = jacobian_row(model, basis, kernels)
    g_a, _H_a = hessian_result_to_node_arrays(model, hess)
    g_hess = project_gradient(gradient_log(model, g_a), basis, model=model)
    scale = float(np.max(np.abs(g_hess))) or 1.0
    return float(np.max(np.abs(g_kernel - g_hess)) / scale)
