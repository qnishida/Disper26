"""Phase 3: Levenberg-Marquardt landing on the true forward + mode identity.

Hand-rolled LM (not scipy.least_squares): every iteration gets residual AND
analytic Jacobian from one kernels call per datum, warm-started from the
previous iterate's phase velocities — scipy's residual/jac split would double
the forward cost and discard the warm starts.

Mode identity: every engine.phase_velocity[_with_kernels] call re-verifies the
branch label by mode-count bisection and raises BranchError on mismatch, so a
mode swap during LM shows up as an infeasible step (rejected / halved) and is
recorded per datum rather than silently contaminating the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .data import DatasetSpec, ObservationCovariance
from .derivatives import gradient_log, kernels_to_stiffness_gradient, project_gradient
from .depth_hessian import depth_backend_config, make_depth_backend
from .engine import BranchError, ForwardEngine
from .model import InversionBasis
from .physics import check_stiffness_posdef
from .priors import PriorCovariance


@dataclass
class LandingResult:
    dm: np.ndarray
    c_pred: np.ndarray
    J: np.ndarray
    misfit: float
    chi2: float
    n_iter: int
    converged: bool
    mode_failures: list          # (iteration, datum label) branch-identity events
    forward_calls: int
    history: list = field(default_factory=list)
    termination_reason: str = ""

    @property
    def mode_id_ok(self) -> bool:
        return not self.mode_failures


class LMOptimizer:
    def __init__(
        self,
        engine: ForwardEngine,
        basis: InversionBasis,
        dataset: DatasetSpec,
        c_obs: np.ndarray,
        sigma: np.ndarray,
        prior: PriorCovariance,
        m0_minus_mprior: np.ndarray = None,
        max_iter: int = 25,
        gtol_rel: float = 1e-6,
        xtol: float = 1e-6,
        # chi^2 stabilizes within 2-3 accepted steps; tighter ftol only polishes
        # the weak (eta/rho) directions at ~0.01% misfit changes per step while
        # doubling the forward cost. Phase 4's posdef repair is the safety net.
        ftol_rel: float = 1e-4,
        lambda0: float = 1e-4,
        processes: int = 1,
        engine_kwargs: dict = None,
    ):
        self.engine = engine
        self.basis = basis
        self.dataset = dataset
        self.c_obs = np.asarray(c_obs, dtype=float)
        self.covariance = ObservationCovariance(sigma, n_data=len(self.dataset))
        self.sigma = self.covariance.sigma
        # Historical public attribute retained for diagonal callers.
        self.w = 1.0 / self.covariance.sigma**2
        self.prior = prior
        self.m0_minus_mprior = (
            np.zeros(basis.n_params) if m0_minus_mprior is None else np.asarray(m0_minus_mprior)
        )
        self.max_iter = max_iter
        self.gtol_rel = gtol_rel
        self.xtol = xtol
        self.ftol_rel = ftol_rel
        self.lambda0 = lambda0
        self.processes = processes
        self.engine_kwargs = engine_kwargs or {}
        self._pool_calls = 0  # forward+kernel calls made inside pool workers
        self._depth_config = depth_backend_config(basis)
        self._depth_backend = make_depth_backend(
            self.engine, self._depth_config, basis=self.basis
        )

    # -- forward evaluations ------------------------------------------------

    def _predict(self, dm: np.ndarray, c_hints, with_jacobian: bool):
        """(c_pred, J or None, mode_fail_labels)."""
        # Explicit interface-depth coordinates have hard neighboring-grid
        # bounds.  Treat an LM trial outside those bounds like any other
        # infeasible (non-positive-definite / branch-failed) trial so the
        # damping line search can reject it and continue; do not discard the
        # entire landing with an uncaught ValueError.
        try:
            model = self.basis.m_to_forward_model(dm)
        except (IndexError, ValueError, FloatingPointError):
            return None, None, ["infeasible_model"]
        if not check_stiffness_posdef(model).ok:
            return None, None, ["posdef"]
        n_data = len(self.dataset)
        if self.processes > 1:
            from .parallel import map_over_data

            results = map_over_data(
                model,
                self.dataset.data,
                c_hints=c_hints,
                want_hessian=False,
                want_kernels=with_jacobian,
                processes=self.processes,
                engine_kwargs=self.engine_kwargs,
                depth_config=self._depth_config,
                basis=self.basis,
            )
            failures = [res["failed"] for res in results if "failed" in res]
            if failures:
                return None, None, failures
            self._pool_calls += n_data
            c_pred = np.asarray([res["c"] for res in results])
            if not np.all(np.isfinite(c_pred)):
                return None, None, ["nonfinite_forward"]
            J = None
            if with_jacobian:
                J = np.empty((n_data, self.basis.n_params))
                for i, res in enumerate(results):
                    if self._depth_backend is not None:
                        J[i] = res["g_inv_kernel"]
                    else:
                        J[i] = project_gradient(
                            gradient_log(model, res["g_a_ad"]), self.basis, model=model
                        )
                if not np.all(np.isfinite(J)):
                    return None, None, ["nonfinite_jacobian"]
            return c_pred, J, []

        c_pred = np.empty(n_data)
        J = np.empty((n_data, self.basis.n_params)) if with_jacobian else None
        failures = []
        for i, datum in enumerate(self.dataset.data):
            hint = None if c_hints is None else float(c_hints[i])
            try:
                if with_jacobian:
                    solution, kernels = self.engine.phase_velocity_with_kernels(
                        model, datum, c_hint=hint
                    )
                    g_a = kernels_to_stiffness_gradient(kernels, model.n_nodes)
                    if self._depth_backend is not None:
                        J[i] = self._depth_backend.gradient_row(model, kernels)
                    else:
                        J[i] = project_gradient(
                            gradient_log(model, g_a), self.basis, model=model
                        )
                else:
                    solution = self.engine.phase_velocity(model, datum, c_hint=hint)
                c_pred[i] = solution.c
            except BranchError:
                failures.append(datum.label())
                c_pred[i] = np.nan
        if failures:
            return None, None, failures
        if not np.all(np.isfinite(c_pred)):
            return None, None, ["nonfinite_forward"]
        if with_jacobian and not np.all(np.isfinite(J)):
            return None, None, ["nonfinite_jacobian"]
        return c_pred, J, []

    def _misfit(self, dm: np.ndarray, c_pred: np.ndarray) -> float:
        r = self.c_obs - c_pred
        d = dm + self.m0_minus_mprior
        return 0.5 * self.covariance.quadratic(r) + 0.5 * float(d @ self.prior.solve(d))

    def _curvature(self, dm: np.ndarray, J: np.ndarray, r: np.ndarray,
                   c_pred: np.ndarray) -> np.ndarray:
        """Matrix the LM step is built from. Gauss-Newton: first derivatives only.

        Kept as a hook so the true-Hessian variant can be measured against this
        one without duplicating the loop (see
        tests/python/diagnose_newton_vs_gauss_newton.py). Overriding it changes
        only the step direction; the damping, line search and acceptance test
        are unchanged.
        """
        return J.T @ self.covariance.solve(J) + self.prior.inv()

    # -- main loop ------------------------------------------------------------

    def run(self, dm0: np.ndarray, c_hints: np.ndarray = None) -> LandingResult:
        self._pool_calls = 0
        calls_before = self.engine.counters["forward"] + self.engine.counters["kernels"]
        dm = np.asarray(dm0, dtype=float).copy()
        mode_failures = []
        history = []
        termination_reason = "max_iter"

        c_pred, J, fails = self._predict(dm, c_hints, with_jacobian=True)
        if fails:
            raise BranchError(f"seed point infeasible: {fails}")
        value = self._misfit(dm, c_pred)
        lam = self.lambda0
        converged = False
        n_iter = 0
        g_scale = None

        for iteration in range(1, self.max_iter + 1):
            n_iter = iteration
            r = self.c_obs - c_pred
            H_GN = self._curvature(dm, J, r, c_pred)
            g = -J.T @ self.covariance.solve(r) + self.prior.solve(dm + self.m0_minus_mprior)
            if g_scale is None:
                g_scale = max(float(np.linalg.norm(g)), 1e-300)
            if np.linalg.norm(g) < self.gtol_rel * g_scale:
                converged = True
                termination_reason = "gradient_tol"
                break
            D = np.diag(np.diag(H_GN))

            accepted = False
            for _attempt in range(8):
                try:
                    factor = cho_factor(H_GN + lam * D)
                except np.linalg.LinAlgError:
                    lam *= 4.0
                    continue
                step = -cho_solve(factor, g)
                predicted_decrease = -float(g @ step) - 0.5 * float(step @ (H_GN @ step))
                trial = dm + step
                c_trial, _, fails = self._predict(trial, c_pred, with_jacobian=False)
                if fails:
                    mode_failures.append((iteration, tuple(fails)))
                    lam *= 4.0
                    continue
                trial_value = self._misfit(trial, c_trial)
                gain = (value - trial_value) / max(predicted_decrease, 1e-300)
                if trial_value < value:
                    rel_decrease = (value - trial_value) / max(abs(value), 1e-300)
                    dm = trial
                    value = trial_value
                    lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3), 1e-12)
                    accepted = True
                    step_norm = self.prior.scaled_norm(step)
                    c_pred, J, fails = self._predict(dm, c_trial, with_jacobian=True)
                    if fails:
                        raise BranchError(f"accepted point lost branch identity: {fails}")
                    history.append(
                        {"iter": iteration, "misfit": value, "lambda": lam, "step_scaled": step_norm}
                    )
                    if step_norm < self.xtol or rel_decrease < self.ftol_rel:
                        converged = True
                        termination_reason = "step_or_ftol"
                    break
                lam *= 4.0
            if not accepted:
                # A failed line search is not evidence of convergence.  Keep
                # the result for diagnostics, but require the explicit
                # gradient criterion above or a small accepted step.
                termination_reason = "line_search_stall"
                converged = False
                break
            if converged:
                break

        r = self.c_obs - c_pred
        chi2 = self.covariance.quadratic(r)
        calls_after = self.engine.counters["forward"] + self.engine.counters["kernels"]
        return LandingResult(
            dm=dm,
            c_pred=c_pred,
            J=J,
            misfit=value,
            chi2=chi2,
            n_iter=n_iter,
            converged=converged,
            mode_failures=mode_failures,
            forward_calls=calls_after - calls_before + self._pool_calls,
            history=history,
            termination_reason=termination_reason,
        )


def merge_minima(landings, prior: PriorCovariance, sigma: np.ndarray, threshold: float = 1.0):
    """Merge converged minima by Mahalanobis distance under the better one's H_GN."""
    unique = []
    metric = ObservationCovariance(sigma, n_data=landings[0].J.shape[0])
    for landing in sorted(landings, key=lambda l: l.misfit):
        H_GN = landing.J.T @ metric.solve(landing.J) + prior.inv()
        merged = False
        for group in unique:
            delta = landing.dm - group[0].dm
            d2 = float(delta @ H_GN @ delta)
            if d2 < threshold**2:
                group.append(landing)
                merged = True
                break
        if not merged:
            unique.append([landing])
    return unique
