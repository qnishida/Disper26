"""Forward facade: branch-labeled phase velocities, kernels, Hessians.

Hides the omega/Hz/period inconsistencies of the raw bindings (see units.py)
and implements the overtone-branch root resolution by mode-count bisection:

1. compute_mode_count(c) on Mesher-refined layers of the grid model is a
   monotone step function counting roots below c; the branch-n root is bracketed
   by bisecting the n -> n+1 jump. This is immune to the missed-sign-change
   failures of a plain root scan (observed for PREM Love at 25 s).
2. High-precision Romberg refinement seeded by the bracket midpoint.
3. Label re-verification at the converged velocity (count just below/above must
   be n / n+1).  If the initial window contains several roots, shrink only that
   window until it isolates the requested jump; otherwise escalate the count
   pancake through ``confirmation_max_scan_level`` before raising BranchError.
   The per-level counts, window attempts, and scan-root offset are retained in
   ``ModeSolution.confirmation``.

Forward-call counters support the phase-2 "zero forward evaluations" assertion
and the Stage E survival statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import disper26py as dp

from .model import ForwardGridModel
from .units import period_to_freq, period_to_omega

WAVE_TYPES = ("Rayleigh", "Love")


class BranchError(RuntimeError):
    """Requested mode branch could not be resolved or verified."""


def canonical_wave(wave: str) -> str:
    w = wave.strip().capitalize()
    if w not in WAVE_TYPES:
        raise ValueError(f"wave must be one of {WAVE_TYPES}, got {wave!r}")
    return w


@dataclass(frozen=True)
class Datum:
    """One observable: (wave, overtone branch n>=0, period [s])."""

    wave: str
    branch: int
    period: float

    def __post_init__(self):
        object.__setattr__(self, "wave", canonical_wave(self.wave))
        if self.branch < 0:
            raise ValueError("branch must be >= 0")
        if self.period <= 0.0:
            raise ValueError("period must be positive")

    def label(self) -> str:
        return f"{self.wave[0]}{self.branch}_{self.period:g}s"


@dataclass
class ModeSolution:
    c: float
    U: float
    level: int
    rel_err: float
    branch: int
    c_scan: float  # labeled coarse root that seeded the refinement (nan for warm starts)
    confirmation: dict | None = None
    romberg_roots: tuple[float, ...] | None = None


class ForwardEngine:
    def __init__(
        self,
        max_steps: int = 6,
        tolerance: float = 1e-10,
        hessian_max_levels: int = 5,
        hessian_tolerance: float = 1e-8,
        scan_level: int = 1,
        bisection_tol_rel: float = 1e-8,
        confirm_delta_rel: float = 3e-4,
        initial_root_window_frac: float = 1e-3,
        confirmation_max_scan_level: int = 5,
        eigen_sub_layers: int = 8,
        hessian_backend: str = "vector",
        want_kernel_u: bool = False,
        want_kernel_hv: bool = False,
        root_tolerance: float | None = None,
    ):
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.root_tolerance = root_tolerance
        if root_tolerance is not None and (
            not math.isfinite(root_tolerance) or root_tolerance <= 0.0
        ):
            raise ValueError("root_tolerance must be finite and positive")
        self.hessian_max_levels = hessian_max_levels
        self.hessian_tolerance = hessian_tolerance
        # Weak-form Hessian AD backend. "vector" runs one blocked vectorized
        # forward pass per group of derivative directions instead of one dual
        # pass per node parameter (~3.4x on this 528-column grid). "dual" is the
        # reference N-pass implementation to fall back to.
        self.hessian_backend = hessian_backend
        # dU/dp and d(H/V)/dp are computed inside the same 528-direction AD
        # pass as dc/dp, once per Romberg level, and cost 84% of it -- measured
        # 6.833 -> 1.091 s/datum on the 88-node shallow grid. Nothing in this
        # package reads them: derivatives.py takes kernel.d_c only, and Datum
        # carries no U or ellipticity observable. So they are off by default
        # and switched on by whoever adds those observables. dc/dp is
        # bit-identical either way.
        self.want_kernel_u = bool(want_kernel_u)
        self.want_kernel_hv = bool(want_kernel_hv)
        self.scan_level = scan_level
        self.bisection_tol_rel = bisection_tol_rel
        # Initial verification window: it must exceed the scan-model versus
        # extrapolated-root offset.  When this window contains more than one
        # count jump, confirmation shrinks it locally instead of weakening
        # the requested branch label or changing the global default.
        if confirm_delta_rel <= 0.0 or not math.isfinite(confirm_delta_rel):
            raise ValueError("confirm_delta_rel must be finite and positive")
        if int(confirmation_max_scan_level) != confirmation_max_scan_level:
            raise ValueError("confirmation_max_scan_level must be an integer")
        if confirmation_max_scan_level < scan_level:
            raise ValueError("confirmation_max_scan_level must be >= scan_level")
        self.confirm_delta_rel = confirm_delta_rel
        self.confirmation_max_scan_level = int(confirmation_max_scan_level)
        if initial_root_window_frac < 0.0 or initial_root_window_frac > 0.2:
            raise ValueError("initial_root_window_frac must be in [0, 0.2]")
        # Keep the coarsest Romberg root search local to a predictor-quality
        # seed.  Branch confirmation below still falls back to mode counting.
        self.initial_root_window_frac = float(initial_root_window_frac)
        self.eigen_sub_layers = eigen_sub_layers
        self.counters = {"forward": 0, "kernels": 0, "hessian": 0, "count": 0, "eigen": 0}

    def reset_counters(self):
        for key in self.counters:
            self.counters[key] = 0

    # ------------------------------------------------------------------
    # scan-model helpers (Mesher-refined layered approximation)
    # ------------------------------------------------------------------

    @staticmethod
    def _wave_speed(param, wave: str) -> float:
        rho = float(param.rho)
        c_mod = float(param.c66) if wave == "Love" else float(param.c44)
        return math.sqrt(max(c_mod, 0.0) / rho)

    def _scan_bounds(self, layers, halfspace, wave: str):
        speeds = [self._wave_speed(p, wave) for p in layers]
        hs_speed = self._wave_speed(halfspace, wave)
        c_min = max(0.05, 0.7 * min(speeds + [hs_speed]))
        c_max = 0.9995 * hs_speed
        if c_max <= c_min:
            raise BranchError("empty scan window: halfspace slower than the stack")
        return c_min, c_max

    def _mode_count(self, layers, thicknesses, halfspace, omega: float, c: float, wave: str) -> int:
        self.counters["count"] += 1
        return int(dp.compute_mode_count(layers, thicknesses, halfspace, omega, c, wave))

    def _scan_context(
        self, model: ForwardGridModel, wave: str, period: float, scan_level: int = None
    ):
        level = self.scan_level if scan_level is None else int(scan_level)
        layers, thicknesses, halfspace = model.refined_layers(level)
        omega = period_to_omega(period)
        c_min, c_max = self._scan_bounds(layers, halfspace, wave)

        def count(c: float) -> int:
            return self._mode_count(layers, thicknesses, halfspace, omega, c, wave)

        return count, c_min, c_max

    def _bracket_branch(self, count, c_min: float, c_max: float, branch: int, datum_label: str):
        """Bisect the branch -> branch+1 jump of the monotone mode count.

        Immune to missed sign changes: no secular-function scan is involved.
        """
        n_lo = count(c_min)
        n_hi = count(c_max)
        if n_lo > branch:
            raise BranchError(
                f"scan window starts above branch {branch} for {datum_label} "
                f"(count at c_min={c_min:.3f} is {n_lo})"
            )
        if n_hi <= branch:
            raise BranchError(
                f"branch {branch} does not exist below the halfspace cutoff for {datum_label} "
                f"(only {n_hi} branches in window; deepen the model or drop this datum)"
            )
        lo, hi = c_min, c_max
        tol = self.bisection_tol_rel * c_max
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if count(mid) <= branch:
                lo = mid
            else:
                hi = mid
        return lo, hi

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def phase_velocity(
        self,
        model: ForwardGridModel,
        datum: Datum,
        c_hint: float = None,
        *,
        minimum_trace_level: int = 0,
    ) -> ModeSolution:
        """Compute one branch-confirmed mode and retain its Romberg roots.

        ``minimum_trace_level`` extends only the retained root sequence for a
        later derivative calculation.  The default performs no trace-only
        refinement, so ordinary forward and line-search calls do not pay the
        Hessian mesh depth.
        """
        confirmation = None
        if c_hint is not None:
            solution = self._romberg(
                model,
                datum,
                c_hint,
                c_scan=math.nan,
                minimum_trace_level=minimum_trace_level,
            )
            confirmation = self._confirm_branch(model, datum, solution.c)
            if confirmation["accepted"]:
                solution.confirmation = confirmation
                return solution
            # warm start landed on a different branch: fall through to full bisection

        bracket_level = (
            confirmation["last_level"] if confirmation is not None else self.scan_level
        )
        count, c_min, c_max = self._scan_context(
            model, datum.wave, datum.period, scan_level=bracket_level
        )
        lo, hi = self._bracket_branch(count, c_min, c_max, datum.branch, datum.label())
        c_seed = 0.5 * (lo + hi)
        solution = self._romberg(
            model,
            datum,
            c_seed,
            c_scan=c_seed,
            minimum_trace_level=minimum_trace_level,
        )
        confirmation = self._confirm_branch(model, datum, solution.c)
        if not confirmation["accepted"]:
            # Nearest-root continuation can switch mode order while the grid
            # is refined near an osculation.  Retry with the requested AD-z
            # mode-count jump selected independently on every Romberg level.
            try:
                solution = self._romberg(
                    model,
                    datum,
                    c_seed,
                    c_scan=c_seed,
                    enforce_branch=True,
                    minimum_trace_level=minimum_trace_level,
                )
            except Exception as exc:
                raise BranchError(
                    f"branch-indexed Romberg refinement failed for {datum.label()}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            confirmation = self._confirm_branch(model, datum, solution.c)
        if not confirmation["accepted"]:
            raise BranchError(
                f"branch label could not be verified after Romberg refinement for {datum.label()} "
                f"(seed c={c_seed:.6f}, converged c={solution.c:.6f}); adjacent branch closer than "
                f"{self.confirm_delta_rel:.0e} rel — {confirmation['status']} "
                f"(confirmation levels {confirmation['base_level']}--{confirmation['last_level']}; "
                f"records={confirmation['records']})"
            )
        solution.confirmation = confirmation
        return solution

    def _romberg(
        self,
        model: ForwardGridModel,
        datum: Datum,
        c_guess: float,
        c_scan: float,
        *,
        enforce_branch: bool = False,
        minimum_trace_level: int = 0,
    ) -> ModeSolution:
        # Params (stiffness) family throughout: the weak-form grid Hessian only
        # exists in this family, and the velocity-basis forward interpolates
        # velocities (not stiffnesses) between nodes — a different continuum
        # model whose curvature differs at O(1) along stiffness directions
        # (historical comparison, 2026-07-12).
        depths, params, is_disc = model.to_params()
        self.counters["forward"] += 1
        result = dp.compute_dispersion_romberg_levels(
            depths,
            params,
            is_disc,
            datum.period,
            datum.wave,
            float(c_guess),
            self.max_steps,
            self.tolerance,
            want_value_u=True,
            want_value_hv=True,
            adaptive_root_scan=True,
            initial_root_window_frac=self.initial_root_window_frac,
            branch=(
                datum.branch
                if enforce_branch or minimum_trace_level > 0
                else None
            ),
            minimum_trace_level=minimum_trace_level,
            root_tolerance=self.root_tolerance,
            **model.refinement_kwargs(),
        )
        return ModeSolution(
            c=float(result["phase_velocity"]),
            U=float(result["group_velocity"]),
            level=int(result["converged_level"]),
            rel_err=float(result["relative_error"]),
            branch=datum.branch,
            c_scan=c_scan,
            romberg_roots=tuple(
                float(value) for value in result["phase_velocity_roots"]
            ),
        )

    @staticmethod
    def _group_predict(frequency_hz: float, c: float, U: float, next_frequency_hz: float) -> float:
        """Predict phase velocity at the next frequency from group velocity."""
        if (
            frequency_hz <= 0.0
            or not (math.isfinite(c) and math.isfinite(U))
            or abs(U) <= 1e-30
        ):
            return float("nan")
        return c + c * (1.0 - c / U) * ((next_frequency_hz - frequency_hz) / frequency_hz)

    def trace_dispersion_curve(
        self,
        model: ForwardGridModel,
        wave: str,
        branch: int,
        frequencies_hz,
        *,
        minimum_trace_level: int = 0,
    ):
        """Trace one branch on an ascending frequency grid.

        The first point uses mode-count bisection.  Later points use the
        group-velocity predictor; ``phase_velocity`` verifies the branch and
        falls back to bisection when a predictor lands on a neighboring root.
        """
        frequencies = [float(f) for f in frequencies_hz]
        if not frequencies or any(f <= 0.0 for f in frequencies):
            raise ValueError("frequencies_hz must contain positive values")
        if any(b <= a for a, b in zip(frequencies, frequencies[1:])):
            raise ValueError("frequencies_hz must be strictly increasing")

        solutions = []
        previous = None
        previous_frequency = None
        for frequency in frequencies:
            hint = None
            if previous is not None:
                hint = self._group_predict(
                    previous_frequency, previous.c, previous.U, frequency
                )
                if not math.isfinite(hint) or hint <= 0.0:
                    hint = previous.c
            solution = self.phase_velocity(
                model,
                Datum(wave, branch, 1.0 / frequency),
                c_hint=hint,
                minimum_trace_level=minimum_trace_level,
            )
            solutions.append(solution)
            previous = solution
            previous_frequency = frequency
        return solutions

    def trace_dataset(self, model: ForwardGridModel, data) -> np.ndarray:
        """Evaluate each (wave, branch) curve by frequency continuation."""
        groups = {}
        for index, datum in enumerate(data):
            groups.setdefault((datum.wave, datum.branch), []).append(
                (1.0 / float(datum.period), index)
            )

        out = np.empty(len(data), dtype=float)
        for (wave, branch), entries in groups.items():
            entries.sort(key=lambda item: item[0])
            frequencies = [item[0] for item in entries]
            solutions = self.trace_dispersion_curve(model, wave, branch, frequencies)
            for (_, index), solution in zip(entries, solutions):
                out[index] = solution.c
        return out

    def _confirmation_window_attempts(self, count, branch: int, c: float):
        """Try the configured count window, shrinking only around close roots.

        A successful attempt must contain exactly the requested monotone count
        jump ``branch -> branch + 1``.  A window is narrowed only when it still
        contains that jump *and* at least one additional root.  Consequently a
        candidate centered on a neighboring root is not relabeled when the
        window shrinks.  The floor stays above the mode-count bisection scale.
        """

        minimum_window_rel = min(
            self.confirm_delta_rel,
            max(
                16.0 * max(float(self.bisection_tol_rel), 0.0),
                512.0 * np.finfo(float).eps,
            ),
        )
        window_rel = float(self.confirm_delta_rel)
        attempts = []
        for _ in range(32):
            delta = window_rel * c
            try:
                count_lo = int(count(c - delta))
                count_hi = int(count(c + delta))
            except Exception as exc:
                attempts.append(
                    {
                        "window_rel": float(window_rel),
                        "accepted": False,
                        "count_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
            accepted = count_lo == branch and count_hi == branch + 1
            attempts.append(
                {
                    "count_below": count_lo,
                    "count_above": count_hi,
                    "window_rel": float(window_rel),
                    "accepted": bool(accepted),
                }
            )
            if accepted:
                break
            contains_requested_jump = count_lo <= branch < count_hi
            contains_multiple_roots = count_hi - count_lo > 1
            if not (contains_requested_jump and contains_multiple_roots):
                break
            next_window_rel = max(minimum_window_rel, 0.5 * window_rel)
            if not next_window_rel < window_rel:
                break
            window_rel = next_window_rel
        return attempts

    def _confirm(self, count, branch: int, c: float) -> bool:
        """Return whether an adaptive count window isolates `branch`."""

        attempts = self._confirmation_window_attempts(count, branch, c)
        return bool(attempts and attempts[-1]["accepted"])

    def _confirm_branch(self, model: ForwardGridModel, datum: Datum, c: float) -> dict:
        """Confirm a Romberg root, escalating the independent count mesh on failure.

        Romberg extrapolates toward the continuous model, while the mode count
        is evaluated on a finite refined pancake.  A low count scan level can
        therefore put the count jump outside the confirmation window without
        any root jump.  Escalation is only used after a failed confirmation.
        """
        records = []
        base_level = self.scan_level
        max_level = max(base_level, self.confirmation_max_scan_level)
        for level in range(base_level, max_level + 1):
            count, c_min, c_max = self._scan_context(
                model, datum.wave, datum.period, scan_level=level
            )
            window_attempts = self._confirmation_window_attempts(
                count, datum.branch, c
            )
            last_attempt = window_attempts[-1]
            record = {
                "scan_level": int(level),
                **last_attempt,
                "window_attempts": window_attempts,
            }
            accepted = bool(last_attempt["accepted"])
            if "count_error" in last_attempt:
                records.append(record)
                continue
            if not accepted:
                try:
                    lo, hi = self._bracket_branch(
                        count, c_min, c_max, datum.branch, datum.label()
                    )
                    scan_root = 0.5 * (lo + hi)
                    record["scan_root"] = float(scan_root)
                    record["root_offset_rel"] = float((scan_root - c) / c)
                except BranchError as exc:
                    record["bracket_error"] = str(exc)
            records.append(record)
            if accepted:
                return {
                    "accepted": True,
                    "status": "accepted",
                    "base_level": int(base_level),
                    "accepted_level": int(level),
                    "last_level": int(level),
                    "records": records,
                }

        last = records[-1]
        if "count_error" in last:
            status = "mode_count_evaluation_failed"
        elif "bracket_error" in last:
            status = "branch_absent_or_wrong_branch"
        elif last["count_above"] - last["count_below"] > 1:
            status = "multiple_roots_in_confirmation_window"
        else:
            offsets = [
                abs(float(item["root_offset_rel"]))
                for item in records
                if "root_offset_rel" in item
            ]
            status = (
                "scan_discretization_not_converged"
                if len(offsets) >= 2 and offsets[-1] < offsets[0]
                else "branch_confirmation_failed"
            )
        return {
            "accepted": False,
            "status": status,
            "base_level": int(base_level),
            "last_level": int(last["scan_level"]),
            "records": records,
        }

    def _branch_confirmed(self, model: ForwardGridModel, datum: Datum, c: float) -> bool:
        return bool(self._confirm_branch(model, datum, c)["accepted"])

    def phase_velocity_with_kernels(
        self,
        model: ForwardGridModel,
        datum: Datum,
        c_hint: float = None,
        *,
        minimum_trace_level: int = 0,
    ):
        """(ModeSolution, kernels) with node kernels in {A,C,F,L,N,Rho,Depth}.

        Kernel values are dc/dp at the converged root (kernel_extrapolation on).
        The velocity projection consumes the six stiffness-field entries;
        ``Depth`` entries are consumed by ``depth_hessian`` for explicit
        interface-depth parameterizations.
        Branch resolution mirrors phase_velocity.

        ``kernel.d_U`` and ``kernel.d_HV`` are zero unless the engine was built
        with ``want_kernel_u`` / ``want_kernel_hv``; see __init__. ``want_value_u``
        / ``want_value_hv`` stay True regardless, because they govern the U and
        H/V *values* rather than their kernels, buy a further 0.1%, and would
        make ``solution.U`` NaN if turned off.

        Set ``minimum_trace_level=hessian_max_levels`` when the returned
        solution will subsequently feed a full or reduced Hessian. Plain
        kernel evaluation needs only the accepted phase levels.
        """
        # Keep the direct-dispersion level roots that own branch identity.
        # The AD binding reconstructs its value tuple and evaluates every
        # kernel level at this trace; it performs no second root search.
        direct_solution = self.phase_velocity(
            model,
            datum,
            c_hint=c_hint,
            minimum_trace_level=minimum_trace_level,
        )
        c_hint = direct_solution.c
        depths, params, is_disc = model.to_params()
        self.counters["kernels"] += 1
        result = dp.compute_kernels_ad_grid(
            depths,
            params,
            is_disc,
            datum.period,
            datum.wave,
            float(c_hint),
            "VTI",
            self.max_steps,
            self.tolerance,
            want_kernel_u=self.want_kernel_u,
            want_kernel_hv=self.want_kernel_hv,
            want_value_u=True,
            want_value_hv=True,
            adaptive_root_scan=True,
            initial_root_window_frac=self.initial_root_window_frac,
            phase_velocity_roots=direct_solution.romberg_roots,
            phase_accepted_level=direct_solution.level,
            **model.refinement_kwargs(),
        )
        # The trace-bearing direct solve above owns the branch label and value
        # tuple.  The binding has independently reconstructed the same values
        # from those roots while evaluating kernels; assert that contract but
        # do not perform another branch-selection pass.
        if abs(float(result[0]) - direct_solution.c) > 5.0e-10:
            raise BranchError(
                f"trace-driven kernel value disagrees with direct dispersion "
                f"for {datum.label()}"
            )
        return direct_solution, result[5]

    def datum_hessian(
        self,
        model: ForwardGridModel,
        datum: Datum,
        c: float,
        phase_velocity_roots=None,
        phase_accepted_level: int = -1,
    ) -> dict:
        """Weak-form AD Hessian in the node (A,C,F,L,N,Rho) basis (mode VTI)."""
        depths, params, is_disc = model.to_params()
        freq = period_to_freq(datum.period)
        self.counters["hessian"] += 1
        if datum.wave == "Love":
            result = dp.compute_hessian_variational_ad_love_grid(
                depths, params, is_disc, freq, float(c), "VTI",
                self.hessian_max_levels, self.hessian_tolerance,
                self.initial_root_window_frac, self.hessian_backend,
                phase_velocity_roots=phase_velocity_roots,
                phase_accepted_level=phase_accepted_level,
                **model.refinement_kwargs(),
            )
        else:
            result = dp.compute_hessian_variational_ad_rayleigh_grid(
                depths, params, is_disc, freq, float(c), "VTI",
                self.hessian_max_levels, self.hessian_tolerance,
                self.hessian_backend,
                phase_velocity_roots=phase_velocity_roots,
                phase_accepted_level=phase_accepted_level,
                **model.refinement_kwargs(),
            )
        hessian = np.asarray(result["hessian"], dtype=float)
        asym = float(np.linalg.norm(hessian - hessian.T) / max(np.linalg.norm(hessian), 1e-300))
        return {
            "gradient": np.asarray(result["gradient"], dtype=float),
            "hessian": 0.5 * (hessian + hessian.T),
            "parameter_names": list(result["parameter_names"]),
            "layer_indices": [int(i) for i in result["layer_indices"]],
            "asymmetry_rel": asym,
        }

    def datum_hessian_reduced(
        self,
        model: ForwardGridModel,
        datum: Datum,
        c: float,
        J: np.ndarray,
        phase_velocity_roots=None,
        phase_accepted_level: int = -1,
    ) -> dict:
        """Weak-form AD Hessian seeded directly in a caller-supplied basis.

        ``J`` (n_nodes*6 x P, node-VTI rows in derivatives.basis_node_jacobian's
        layout) replaces the node-VTI unit-vector seeding of ``datum_hessian``,
        so the AD pass count drops from n_nodes*6 to P (e.g. 528 -> ~19 for the
        production velocity+depth basis) and the C++ side never materializes
        the n_nodes*6 square Hessian. Returns 'hessian_reduced' (P x P, NOT yet
        symmetrized against numerical asymmetry the way datum_hessian's output
        is -- callers combine it with derivatives.hessian_curvature_only via
        derivatives.datum_hessian_inv_reduced, which is the point where
        symmetrization belongs) and 'gradient_node' (n_nodes*6, unprojected,
        needed for that curvature correction). See
        docs/sensitivity_implementation_map.md and derivatives.py.
        """
        depths, params, is_disc = model.to_params()
        freq = period_to_freq(datum.period)
        self.counters["hessian"] += 1
        if datum.wave == "Love":
            result = dp.compute_hessian_variational_ad_love_grid_reduced(
                depths, params, is_disc, freq, float(c), J,
                self.hessian_max_levels, self.hessian_tolerance,
                self.initial_root_window_frac, self.hessian_backend,
                phase_velocity_roots=phase_velocity_roots,
                phase_accepted_level=phase_accepted_level,
                **model.refinement_kwargs(),
            )
        else:
            result = dp.compute_hessian_variational_ad_rayleigh_grid_reduced(
                depths, params, is_disc, freq, float(c), J,
                self.hessian_max_levels, self.hessian_tolerance,
                self.hessian_backend,
                phase_velocity_roots=phase_velocity_roots,
                phase_accepted_level=phase_accepted_level,
                **model.refinement_kwargs(),
            )
        hessian_reduced = np.asarray(result["hessian_reduced"], dtype=float)
        asym = float(np.linalg.norm(hessian_reduced - hessian_reduced.T)
                     / max(np.linalg.norm(hessian_reduced), 1e-300))
        return {
            "gradient_node": np.asarray(result["gradient_node"], dtype=float),
            "hessian_reduced": hessian_reduced,
            "asymmetry_rel": asym,
        }

    def eigenfunctions(self, model: ForwardGridModel, datum: Datum, c: float) -> dict:
        layers, thicknesses, halfspace = model.refined_layers(self.scan_level)
        omega = period_to_omega(datum.period)
        self.counters["eigen"] += 1
        if datum.wave == "Love":
            return dp.compute_love_eigenfunctions(layers, thicknesses, halfspace, omega, float(c), self.eigen_sub_layers)
        return dp.compute_rayleigh_eigenfunctions(layers, thicknesses, halfspace, omega, float(c), self.eigen_sub_layers)

    def branch_velocities(self, model: ForwardGridModel, wave: str, period: float, max_branch: int = 5):
        """{branch: coarse c} via mode-count bisection — for osculation maps and vetting."""
        count, c_min, c_max = self._scan_context(model, canonical_wave(wave), period)
        n_total = count(c_max)
        out = {}
        for n in range(min(max_branch + 1, n_total)):
            lo, hi = self._bracket_branch(count, c_min, c_max, n, f"{wave}{n}_{period:g}s")
            out[n] = 0.5 * (lo + hi)
        return out
