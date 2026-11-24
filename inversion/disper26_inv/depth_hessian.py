"""Backend interface for interface-depth derivatives.

The inversion basis can include one interface shift in km. Both coincident
nodes move together, with their material values fixed in the model's input
coordinates. Flat GRID uses the variational boundary kernel and its AD
derivative. Spherical stiffness GRID uses AD through interpolation and EFT
to obtain the gradient and Hessian row in physical depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivatives import gradient_log, kernels_to_depth_gradient, kernels_to_stiffness_gradient
from .model import ForwardGridModel, shift_discontinuity


class DepthHessianError(RuntimeError):
    """A depth perturbation could not be evaluated on the requested branch."""


@dataclass
class DepthHessianColumn:
    """Result and provenance for one depth-Hessian column."""

    column: np.ndarray | None
    step_km: float
    plus_success: bool
    minus_success: bool
    convergence_checked: bool
    converged: bool | None
    convergence_error: float | None
    method: str = "fd"
    plus_error: str | None = None
    minus_error: str | None = None

    @property
    def valid(self) -> bool:
        return (
            self.column is not None
            and np.all(np.isfinite(self.column))
            and self.plus_success
            and self.minus_success
            and (not self.convergence_checked or self.converged is True)
        )

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "step_km": float(self.step_km),
            "plus_success": bool(self.plus_success),
            "minus_success": bool(self.minus_success),
            "convergence_checked": bool(self.convergence_checked),
            "converged": None if self.converged is None else bool(self.converged),
            "convergence_error": (
                None if self.convergence_error is None else float(self.convergence_error)
            ),
            "plus_error": self.plus_error,
            "minus_error": self.minus_error,
        }


class DepthHessianBackend:
    """Small backend contract used by the inversion orchestration.

    ``gradient_row`` returns the complete inversion-coordinate Jacobian row.
    ``interface_hessian_row`` returns the corresponding phase-velocity Hessian
    column.  The velocity-to-node operator is supplied by the basis, so this
    backend does not depend on a particular velocity parameterization.
    """

    def gradient_row(self, model, kernels, solution=None):  # pragma: no cover - interface only
        raise NotImplementedError

    def interface_hessian_row(
        self, model, datum, c, solution=None
    ):  # pragma: no cover - interface only
        raise NotImplementedError


class FdDepthHessianBackend(DepthHessianBackend):
    """Central difference of the exact AD gradient in one interface direction."""

    def __init__(
        self,
        engine,
        pair_start: int,
        velocity_operator: np.ndarray,
        depth_parameter_index: int,
        step_km: float = 0.05,
        validation_step_km: float | None = None,
        convergence_tolerance: float = 2.0e-3,
        basis=None,
        # Accepted and ignored so that one depth_backend_config() dict can be
        # splatted into either backend, mirroring how the analytic backend
        # accepts and ignores step_km / validation_step_km.
        discontinuity_index: int | None = None,
    ):
        self.engine = engine
        self.basis = basis
        self.pair_start = int(pair_start)
        self.velocity_operator = np.asarray(velocity_operator, dtype=float)
        self.depth_parameter_index = int(depth_parameter_index)
        self.step_km = float(step_km)
        self.validation_step_km = (
            None if validation_step_km is None else float(validation_step_km)
        )
        self.convergence_tolerance = float(convergence_tolerance)
        if self.step_km <= 0.0 or not np.isfinite(self.step_km):
            raise ValueError("step_km must be finite and positive")
        if self.validation_step_km is not None and (
            self.validation_step_km <= 0.0 or not np.isfinite(self.validation_step_km)
        ):
            raise ValueError("validation_step_km must be finite and positive")
        if self.velocity_operator.ndim != 2:
            raise ValueError("velocity_operator must be a two-dimensional array")
        if self.depth_parameter_index != self.velocity_operator.shape[1]:
            raise ValueError(
                "depth_parameter_index must equal the number of velocity columns"
            )

    def _gradient_from_kernels(self, model: ForwardGridModel, kernels) -> np.ndarray:
        g_a = kernels_to_stiffness_gradient(kernels, model.n_nodes)
        g_log = gradient_log(model, g_a)
        operator = self._operator(model)
        row = operator.T @ g_log.reshape(-1)
        depth = kernels_to_depth_gradient(kernels, self.pair_start)
        return np.concatenate([np.asarray(row, dtype=float), [depth]])

    def _operator(self, model):
        if self.basis is None:
            return self.velocity_operator
        return np.asarray(self.basis.log_field_operator(model)[:, : self.depth_parameter_index], dtype=float)

    def gradient_row(self, model: ForwardGridModel, kernels, solution=None) -> np.ndarray:
        """Map stiffness and ``Depth`` kernels to inversion coordinates.

        ``solution`` is accepted for interface compatibility with
        :class:`AdDepthHessianBackend` and ignored here.
        """
        return self._gradient_from_kernels(model, kernels)

    def _central_column(self, model, datum, c, step_km: float):
        plus_ok = minus_ok = False
        plus_error = minus_error = None
        plus_gradient = minus_gradient = None
        try:
            plus_model = shift_discontinuity(model, self.pair_start, step_km)
            plus_solution, plus_kernels = self.engine.phase_velocity_with_kernels(
                plus_model, datum, c_hint=float(c)
            )
            plus_gradient = self._gradient_from_kernels(plus_model, plus_kernels)
            plus_ok = bool(
                np.isfinite(plus_solution.c) and np.all(np.isfinite(plus_gradient))
            )
            if not plus_ok:
                plus_error = "nonfinite phase-velocity gradient"
        except Exception as exc:  # branch identity and physical-bound failures
            plus_error = f"{type(exc).__name__}: {exc}"
        try:
            minus_model = shift_discontinuity(model, self.pair_start, -step_km)
            minus_solution, minus_kernels = self.engine.phase_velocity_with_kernels(
                minus_model, datum, c_hint=float(c)
            )
            minus_gradient = self._gradient_from_kernels(minus_model, minus_kernels)
            minus_ok = bool(
                np.isfinite(minus_solution.c) and np.all(np.isfinite(minus_gradient))
            )
            if not minus_ok:
                minus_error = "nonfinite phase-velocity gradient"
        except Exception as exc:
            minus_error = f"{type(exc).__name__}: {exc}"
        if not plus_ok or not minus_ok:
            return None, plus_ok, minus_ok, plus_error, minus_error
        return (
            (plus_gradient - minus_gradient) / (2.0 * step_km),
            plus_ok,
            minus_ok,
            plus_error,
            minus_error,
        )

    def interface_hessian_row(
        self, model: ForwardGridModel, datum, c: float, solution=None
    ) -> DepthHessianColumn:
        """Return ``d/dz (dc/dm)`` with branch and convergence provenance."""
        result = self._central_column(model, datum, float(c), self.step_km)
        column, plus_ok, minus_ok, plus_error, minus_error = result
        convergence_checked = self.validation_step_km is not None
        converged = None
        convergence_error = None
        if column is not None and self.validation_step_km is not None:
            reference = self._central_column(
                model, datum, float(c), self.validation_step_km
            )
            ref_column = reference[0]
            if ref_column is not None:
                scale = max(
                    float(np.max(np.abs(column))),
                    float(np.max(np.abs(ref_column))),
                    1.0e-15,
                )
                convergence_error = float(np.max(np.abs(column - ref_column)) / scale)
                converged = bool(convergence_error <= self.convergence_tolerance)
            else:
                converged = False
                convergence_error = float("inf")
                plus_ok = plus_ok and reference[1]
                minus_ok = minus_ok and reference[2]
        return DepthHessianColumn(
            column=column,
            step_km=self.step_km,
            plus_success=plus_ok,
            minus_success=minus_ok,
            convergence_checked=convergence_checked,
            converged=converged,
            convergence_error=convergence_error,
            plus_error=plus_error,
            minus_error=minus_error,
        )


def _flat_interface_inputs(model):
    if model.refinement_policy.kind != "flat":
        raise ValueError(
            "Variational interface depth derivatives require a flat model. Use the default "
            "spherical AD backend for spherical depths, or load_prem_flat for flat depths."
        )
    return model.to_params()


class AdDepthHessianBackend(DepthHessianBackend):
    """Analytic depth column via nested forward AD (no finite difference).

    Differentiates interface depth in the Lagrangian parameterization. The C++ side
    (``compute_interface_hessian_row_ad_grid``) runs one nested-AD pass whose outer
    level carries the interface direction, returning both ``dc/dp`` and
    ``d(dc/dp)/dz`` in raw node-variable order; this class projects the latter
    onto the inversion basis exactly as ``gradient_row`` does for the former.

    Versus :class:`FdDepthHessianBackend` this removes the
    ``z +/- step_km`` evaluations, so it costs one pass instead of two and does
    not require both shifted models to be feasible and on the same branch --
    the failure mode that bites when a landing sits on an interface bound.
    Accuracy is equivalent, not better: the FD column already achieves the same
    Taylor convergence order in the historical wide-search study. Its detailed
    report is kept in the research archive.
    """

    def __init__(
        self,
        engine,
        pair_start: int,
        velocity_operator: np.ndarray,
        depth_parameter_index: int,
        discontinuity_index: int,
        sensitivity_mode: str = "VTI",
        # Accepted and ignored so a config built for the FD backend can be
        # handed to either implementation unchanged.
        step_km: float = 0.05,
        validation_step_km: float | None = None,
        convergence_tolerance: float = 2.0e-3,
        basis=None,
    ):
        self.engine = engine
        self.basis = basis
        self.pair_start = int(pair_start)
        self.velocity_operator = np.asarray(velocity_operator, dtype=float)
        self.depth_parameter_index = int(depth_parameter_index)
        self.discontinuity_index = int(discontinuity_index)
        self.sensitivity_mode = str(sensitivity_mode)
        if self.velocity_operator.ndim != 2:
            raise ValueError("velocity_operator must be a two-dimensional array")
        if self.depth_parameter_index != self.velocity_operator.shape[1]:
            raise ValueError(
                "depth_parameter_index must equal the number of velocity columns"
            )

    def gradient_row(self, model: ForwardGridModel, kernels, solution=None) -> np.ndarray:
        """Identical to the FD backend: the first-order path is already analytic.

        The caller's ``kernels`` are extrapolated, so their dc/dz is returned
        as the Jacobian entry but is not reused as the seed of a fixed-level
        Hessian calculation. The current ``ModeSolution`` is passed explicitly
        to :meth:`interface_hessian_row`; no cross-datum trace cache is used.
        """
        g_a = kernels_to_stiffness_gradient(kernels, model.n_nodes)
        g_log = gradient_log(model, g_a)
        row = self._operator(model).T @ g_log.reshape(-1)
        depth = kernels_to_depth_gradient(kernels, self.pair_start)
        return np.concatenate([np.asarray(row, dtype=float), [depth]])

    def interface_hessian_row(
        self, model: ForwardGridModel, datum, c: float, solution=None
    ) -> DepthHessianColumn:
        try:
            if solution is None:
                # Standalone use: obtain provenance for this exact datum. Do
                # not reuse mutable state left by a previous branch/frequency.
                solution = self.engine.phase_velocity(
                    model, datum, c_hint=float(c)
                )
            d2 = self._raw_column(model, datum, solution)
        except Exception as exc:
            return DepthHessianColumn(
                column=None, step_km=0.0, plus_success=False, minus_success=False,
                convergence_checked=False, converged=None, convergence_error=None,
                method="ad", plus_error=f"{type(exc).__name__}: {exc}",
            )

        column = self._project_column(model, np.asarray(d2, dtype=float))
        ok = bool(np.all(np.isfinite(column)))
        return DepthHessianColumn(
            column=column if ok else None,
            step_km=0.0,
            plus_success=ok,
            minus_success=ok,
            convergence_checked=False,
            converged=None,
            convergence_error=None,
            method="ad",
            plus_error=None if ok else "non-finite analytic depth column",
        )

    def _raw_column(self, model, datum, solution):
        import disper26py as dp

        depths, params, is_disc = model.to_params()
        level = int(solution.level)
        c = float(solution.c)
        romberg_roots = solution.romberg_roots

        # The nested-AD row is evaluated on one refinement mesh, so its
        # outer root seed must be dc_k/dz on that same mesh rather than
        # the extrapolated dc/dz returned by phase_velocity_with_kernels.
        level_result = dp.compute_kernels_ad_grid(
            depths,
            params,
            is_disc,
            datum.period,
            datum.wave,
            float(c),
            self.sensitivity_mode,
            max_steps=int(level),
            tolerance=self.engine.tolerance,
            compute_c=True,
            want_kernel_u=False,
            want_kernel_hv=False,
            u_kernel_mode="variational",
            kernel_extrapolation=False,
            phase_velocity_roots=romberg_roots,
            phase_accepted_level=int(level),
            **model.refinement_kwargs(),
        )
        dc_dz = float(kernels_to_depth_gradient(
            level_result[5], self.pair_start
        ))
        _, d2 = dp.compute_interface_hessian_row_ad_grid(
            depths, params, is_disc, datum.period, datum.wave,
            float(c), dc_dz, self.discontinuity_index,
            int(level), self.sensitivity_mode,
            phase_velocity_roots=romberg_roots,
            phase_accepted_level=int(level),
            **model.refinement_kwargs(),
        )
        return d2

    def _project_column(self, model: ForwardGridModel, d2: np.ndarray) -> np.ndarray:
        """Raw node-variable column -> inversion coordinates.

        The C++ entry points return NUM_PARAM_VARS moduli entries followed by
        one per discontinuity, in stiffness order. Same projection
        ``gradient_row`` applies to the first-order row, so the Hessian column
        lands in the same coordinates as the rest of the block.
        """
        from .derivatives import STIFF_ORDER

        n_nodes = model.n_nodes
        n_stiff = len(STIFF_ORDER)
        d2_node = d2[: n_nodes * n_stiff].reshape(n_nodes, n_stiff)
        d2_log = gradient_log(model, d2_node).reshape(-1)
        operator = self._operator(model)
        return np.concatenate([
            operator.T @ d2_log,
            [d2[n_nodes * n_stiff + self.discontinuity_index]],
        ])

    def _operator(self, model):
        if self.basis is None:
            return self.velocity_operator
        return np.asarray(self.basis.log_field_operator(model)[:, : self.depth_parameter_index], dtype=float)


class VariationalAdDepthHessianBackend(AdDepthHessianBackend):
    """Depth column from the variational boundary kernel K_d, Romberg-extrapolated.

    Same quantity as :class:`AdDepthHessianBackend` -- the two routes were
    shown in historical diagnostics to converge to the same continuum row
    (agreement 2.5e-08..2.1e-06 once both are extrapolated; the scripts are
    retained in the research archive) -- but reached
    differently and more cheaply. Nested AD differentiates the secular function
    with an outer AD level carrying the interface direction; K_d differentiates
    the spatial-Hamiltonian boundary kernel in ONE level, since
    dc/dz_j = sum_b [K_d]_b w_b has purely geometric weights.

    Two things changed relative to nested AD:

    * **Cost.** One AD level instead of two: historical benchmarks recorded
      2.48x faster calls at equal refinement.
    * **Truncation.** The C++ entry point evaluates every Romberg level and
      Neville-extrapolates, the treatment
      ``compute_kernels_ad_grid`` already gives the first-order
      kernels. The single-level column both older backends produce carries an
      O(h^2) error (~1e-4 relative, falling exactly 4x per level) and was the
      only part of H left un-extrapolated in the historical 2026-08-07 study.
      Levels 0..L cost about twice level L alone, so the net still sits below
      the nested-AD single level.

    That 1e-4 was measured NOT to reach the posterior on the case tried: Delta
    shifted 1.3e-04 and the weight ratio 0.99987, against the 1.38 between-minima
    spread that moves a weight 4x
    (historical depth-extrapolation impact study, 2026-08-07). Extrapolation is
    taken because it is cheaper anyway, and because that measurement covers one
    structure rather than all of them.
    """

    def depth_gradient_variational(
        self,
        model: ForwardGridModel,
        datum,
        c: float,
        level: int,
        phase_velocity_roots=None,
    ) -> float:
        """dc/dz alone, via K_d -- needs no AD kernel pass.

        ``gradient_row`` (inherited from :class:`AdDepthHessianBackend`)
        still reads the AD kernel list's ``Depth`` entry, because that is what
        a caller who already paid for the AD pass gets for free. This is the
        route for a caller who did not: ``level`` is a byproduct of the plain
        ``phase_velocity`` call, so the K_d evaluation here needs nothing the
        AD pass would have supplied. Agreement with the AD ``Depth`` kernel is
        7.7e-12 or better across the cases checked --
        tests/python/test_interface_gradient_variational.py.

        The caller passes the same ``ModeSolution`` explicitly to the Hessian
        row, so no mutable cross-datum trace cache is needed.
        """
        import disper26py as dp

        depths, params, is_disc = _flat_interface_inputs(model)
        dc_dz = dp.compute_interface_gradient_variational_grid(
            depths, params, is_disc, datum.period, datum.wave, float(c),
            self.discontinuity_index, int(level),
            phase_velocity_roots=phase_velocity_roots,
            phase_accepted_level=int(level),
        )
        return float(dc_dz)

    def interface_hessian_row(
        self, model: ForwardGridModel, datum, c: float, solution=None
    ) -> DepthHessianColumn:
        import disper26py as dp

        depths, params, is_disc = _flat_interface_inputs(model)
        try:
            if solution is None:
                # Standalone use: obtain provenance for this exact datum.
                solution = self.engine.phase_velocity(
                    model, datum, c_hint=float(c)
                )
            level = int(solution.level)
            c = float(solution.c)
            romberg_roots = solution.romberg_roots
            d2 = dp.compute_interface_hessian_row_variational_ad_grid_extrapolated(
                depths, params, is_disc, datum.period, datum.wave,
                float(c), self.discontinuity_index, int(level),
                self.sensitivity_mode,
                phase_velocity_roots=romberg_roots,
                phase_accepted_level=int(level),
            )
        except Exception as exc:
            return DepthHessianColumn(
                column=None, step_km=0.0, plus_success=False, minus_success=False,
                convergence_checked=False, converged=None, convergence_error=None,
                method="variational_ad",
                plus_error=f"{type(exc).__name__}: {exc}",
            )

        column = self._project_column(model, np.asarray(d2, dtype=float))
        ok = bool(np.all(np.isfinite(column)))
        return DepthHessianColumn(
            column=column if ok else None,
            step_km=0.0,
            plus_success=ok,
            minus_success=ok,
            convergence_checked=False,
            converged=None,
            convergence_error=None,
            method="variational_ad",
            plus_error=None if ok else "non-finite variational depth column",
        )


class SphericalAdDepthHessianBackend(AdDepthHessianBackend):
    """Extrapolate the secular AD derivatives in physical spherical coordinates."""

    def _evaluate(self, model, datum, c, level, roots, row):
        import disper26py as dp

        return dp._compute_spherical_interface_ad_grid(
            *model.to_params(), datum.period, datum.wave, float(c), int(level),
            roots, int(level), row, **model.refinement_kwargs()
        )

    def depth_gradient(self, model, datum, c, level, phase_velocity_roots):
        gradient = self._evaluate(model, datum, c, level, phase_velocity_roots, None)
        return float(gradient[self.discontinuity_index])

    def _raw_column(self, model, datum, solution):
        return self._evaluate(model, datum, solution.c, solution.level,
                              solution.romberg_roots, self.discontinuity_index)


def depth_backend_config(
    basis,
    step_km: float = 0.05,
    validation_step_km: float | None = None,
    convergence_tolerance: float = 2.0e-3,
) -> dict | None:
    """Serialize the depth coordinate needed by process-pool workers.

    A basis without ``depth_parameter_index`` follows the historical
    velocity-only path and returns ``None``.
    """
    if not hasattr(basis, "depth_parameter_index"):
        return None
    depth_index = int(basis.depth_parameter_index)
    operator = np.asarray(basis.full_log_operator(), dtype=float)
    if operator.shape[1] != depth_index + 1:
        raise ValueError("depth basis operator must end with one depth column")
    # The analytic backend addresses interfaces by their ordinal among the
    # reference model's discontinuities, which is what the C++ variable layout
    # uses; the FD backend addresses them by node index.  Carry both so one
    # config serves either implementation.
    flags = np.asarray(basis.reference.is_discontinuity, dtype=bool)
    disc_nodes = [int(i) for i in np.nonzero(flags)[0]]
    try:
        discontinuity_index = disc_nodes.index(int(basis.pair_start))
    except ValueError as exc:
        raise ValueError(
            f"pair_start {basis.pair_start} is not a discontinuity of the reference model"
        ) from exc
    return {
        **({"backend": "spherical_ad"} if basis.reference.refinement_policy.kind != "flat" else {}),
        "pair_start": int(basis.pair_start),
        "depth_parameter_index": depth_index,
        "discontinuity_index": discontinuity_index,
        "velocity_operator": operator[:, :depth_index],
        "step_km": float(step_km),
        "validation_step_km": (
            None if validation_step_km is None else float(validation_step_km)
        ),
        "convergence_tolerance": float(convergence_tolerance),
    }


# Selects the implementation used by make_depth_backend. The backends share the
# DepthHessianBackend contract, so LM / phase 4 / GMM are unaffected either way,
# and `make_depth_backend(..., backend="fd")` reverts.
#
# The analytic backend is the default because the finite-difference one cannot
# produce a column at all once a landing sits on an interface bound: it needs
# the model at z +/- step_km, and one side is then out of range.  That is not a
# corner case -- four of five landings in the wide-search study were pinned to a
# bound, and run_phase4 raises on an invalid depth column, so the FD path takes
# the whole inversion down there (measured 0/4 valid columns vs 4/4 for the
# analytic path in the historical bound diagnostic).  It was also 1.73x
# faster end-to-end at equivalent accuracy in the 2026-08-06 study. Those
# diagnostics and reports are retained in the research archive.
#
# Flat GRID defaults to `variational_ad`, which differentiates the boundary
# kernel once and extrapolates. The fixed-level `ad` backend differentiates
# the dispersion equation and provides an independent check. A spherical
# basis selects `spherical_ad`, which differentiates through EFT at each
# refinement level before extrapolation.
DEPTH_HESSIAN_BACKENDS = {
    "fd": FdDepthHessianBackend,
    "ad": AdDepthHessianBackend,
    "spherical_ad": SphericalAdDepthHessianBackend,
    "variational_ad": VariationalAdDepthHessianBackend,
}
DEFAULT_DEPTH_HESSIAN_BACKEND = "variational_ad"


def make_depth_backend(engine, config: dict | None, backend: str = None, basis=None):
    if config is None:
        return None
    name = backend or config.get("backend") or DEFAULT_DEPTH_HESSIAN_BACKEND
    try:
        cls = DEPTH_HESSIAN_BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown depth Hessian backend {name!r}; "
            f"choose from {sorted(DEPTH_HESSIAN_BACKENDS)}"
        ) from exc
    kwargs = {k: v for k, v in config.items() if k != "backend"}
    kwargs["basis"] = basis
    return cls(engine=engine, **kwargs)
