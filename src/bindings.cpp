#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

// Include the main library implementation
#include "disper26.cpp"
#include "disper26_analytical.hpp"
#include "sensitivity_utils.hpp" // Now needed for sensitivity
#include "romberg_engine.hpp"    // GRID Mode Romberg Acceleration
#include "grid_hessian.hpp"      // Grid Mode WF Hessian with Romberg
#include "dispersion_interface.hpp"
#include "velocity_grid.hpp"     // Velocity-basis GRID/Romberg helpers
#include "api_impl.hpp"          // Shared Python-independent public API
#include "api_python.hpp"
// Forced Rebuild Check 109

namespace py = pybind11;
using namespace Analytical; // For analytical integration

// --- Helpers ---

py::dict py_debugModeCount(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    WaveType type = parseWaveType(wave_type_str);

    ModeCountDebugInfo<double> info;
    calculateTotalPhaseAccumulationWithDebug(
        layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type, &info);

    py::dict d;
    d["theta_init"] = info.theta_init;
    d["y_init"] = info.y_init;
    d["x_init"] = info.x_init;
    d["init_phase"] = info.init_phase;
    d["halfspace_virtual_phase"] = info.halfspace_virtual_phase;
    d["layer_phase_sum"] = info.layer_phase_sum;
    d["total_phase"] = info.total_phase;
    d["total_phase_over_pi"] = info.total_phase_over_pi;
    d["kappa_sq"] = info.kappa_sq;
    d["layer_mode_sum"] = info.layer_mode_sum;
    d["floor_count"] = info.floor_count;
    d["abs_floor_count"] = info.abs_floor_count;
    d["supra_rayleigh_pi_correction_applied"] = info.supra_rayleigh_pi_correction_applied;
    d["used_virtual_halfspace"] = info.used_virtual_halfspace;
    d["used_liquid_mode"] = info.used_liquid_mode;
    return d;
}

py::dict branModeCountResultToDict(
    const BranModeCountResult& info,
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    WaveType existing_type
) {
    py::dict d;
    d["n_integer"] = info.n_integer;
    d["signed_count"] = info.signed_count;
    d["initial_signed_count"] = info.initial_signed_count;
    d["calibrated_signed_count"] = info.calibrated_signed_count;
    d["crossing_count"] = static_cast<int>(info.crossings.size());
    d["unsupported_liquid"] = info.unsupported_liquid;
    d["unsupported_love"] = info.unsupported_love;
    d["nan_or_radiative"] = info.nan_or_radiative;
    // Only run the legacy counters on models the bran counter supports: the
    // legacy Love propagation has no guard for liquid layers below solids
    // (computeLayerMatrixSH divides by c44), so it would produce NaN there.
    if (!info.unsupported_liquid && !info.unsupported_love && !info.nan_or_radiative) {
        double legacy_continuous_count = calculateContinuousModeCount(
            layers, thicknesses, halfSpaceParams, omega, phaseVelocity, existing_type);
        d["legacy_continuous_count"] = legacy_continuous_count;
        d["existing_continuous_count"] = legacy_continuous_count;
    } else {
        d["legacy_continuous_count"] = py::none();
        d["existing_continuous_count"] = py::none();
    }

    py::list crossings;
    for (const auto& crossing : info.crossings) {
        py::dict c;
        c["layer_index"] = crossing.layer_index;
        c["local_depth_km"] = crossing.local_depth_km;
        c["contribution"] = crossing.contribution;
        c["secular"] = crossing.secular;
        c["secular_derivative"] = crossing.secular_derivative;
        c["conjugate"] = crossing.conjugate;
        c["test_value"] = crossing.test_value;
        crossings.append(c);
    }
    d["crossings"] = crossings;

    std::string status = "ok";
    if (info.unsupported_love) status = "unsupported_love";
    if (info.unsupported_liquid) status = "unsupported_liquid";
    if (info.nan_or_radiative) status = "nan_or_radiative";
    d["status"] = status;
    return d;
}

py::dict py_debugBranModeCount(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    BranModeCountResult info;
    if (wave_type_str == "Love" || wave_type_str == "love") {
        info = calculateBranModeCountLoveSolid(
            layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
        return branModeCountResultToDict(info, layers, thicknesses, halfSpaceParams, omega, phaseVelocity, WaveType::Love);
    }
    info = calculateBranModeCountRayleighSolid(
        layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
    return branModeCountResultToDict(info, layers, thicknesses, halfSpaceParams, omega, phaseVelocity, WaveType::Rayleigh);
}

py::dict py_computeFrequencyCurvatureAdGrid(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double period,
    const std::string& wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool adaptive_root_scan,
    double initial_root_window_frac,
    int extrapolate_start_level,
    const std::string& grid_policy,
    double planet_radius,
    double flattening_exponent
) {
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    model.require_valid();

    Mesher mesher(model);
    auto refiner = [&mesher](int level) {
        VelocityGridRefinement mesh;
        std::tie(mesh.layers, mesh.thicknesses) =
            mesher.get_refined_layers(level);
        mesh.halfspace = mesher.get_halfspace();
        return mesh;
    };

    const FrequencyCurvatureRombergResult result =
        compute_frequency_curvature_ad_grid_from_refiner(
            refiner, period, wave_type_str, c_guess, max_steps, tolerance,
            adaptive_root_scan, initial_root_window_frac,
            extrapolate_start_level);

    const char* status = "unsafe_derivative";
    if (result.status == FrequencyCurvatureStatus::Ok) {
        status = "ok";
    } else if (result.status == FrequencyCurvatureStatus::UnsafeRoot) {
        status = "unsafe_root";
    }

    py::dict out;
    out["phase_velocity"] = result.phase_velocity;
    out["group_velocity"] = result.group_velocity;
    out["group_velocity_omega_derivative"] =
        result.group_velocity_omega_derivative;
    out["wavenumber_omega_second_derivative"] =
        result.wavenumber_omega_second_derivative;
    out["dimensionless_phase_group_dispersion"] =
        result.dimensionless_phase_group_dispersion;
    out["dimensionless_frequency_curvature"] =
        result.dimensionless_frequency_curvature;
    out["accepted_level"] = result.accepted_level;
    out["phase_velocity_relative_error"] =
        result.phase_velocity_relative_error;
    out["group_velocity_omega_derivative_error"] =
        result.group_velocity_omega_derivative_error;
    out["maximum_root_residual"] = result.maximum_root_residual;
    out["status"] = status;
    return out;
}

py::dict py_computeDispersionRombergLevels(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool phase_only = false,
    bool want_value_u = true,
    bool want_value_hv = true,
    bool adaptive_root_scan = true,
    double initial_root_window_frac = 0.0,
    int extrapolate_start_level = 0,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    std::optional<int> branch = std::nullopt,
    int minimum_trace_level = 0,
    std::optional<double> root_tolerance = std::nullopt
) {
    if (want_value_u != want_value_hv) {
        throw std::invalid_argument(
            "want_value_u and want_value_hv must currently agree: the Romberg "
            "engine computes U and H/V values together, so independent "
            "per-value control is not yet implemented.");
    }
    phase_only = phase_only || !want_value_u;
    if (extrapolate_start_level != 0) {
        throw std::invalid_argument(
            "compute_dispersion_romberg_levels currently requires "
            "extrapolate_start_level=0 so every trace consumer differentiates "
            "the same Romberg table; use compute_dispersion_romberg for the "
            "legacy nonzero-start value-only path");
    }
    if (minimum_trace_level > 0 && !branch.has_value()) {
        throw std::invalid_argument(
            "minimum_trace_level > 0 requires branch so trace-only roots are "
            "selected by mode order");
    }

    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    model.require_valid();

    Mesher mesher(model);
    auto refiner = [&mesher](int k) {
        VelocityGridRefinement mesh;
        std::tie(mesh.layers, mesh.thicknesses) = mesher.get_refined_layers(k);
        mesh.halfspace = mesher.get_halfspace();
        return mesh;
    };
    RombergResult result = compute_dispersion_romberg_result_from_refiner(
        refiner, period, wave_type_str, c_guess, max_steps, tolerance,
        phase_only, adaptive_root_scan, initial_root_window_frac,
        extrapolate_start_level, branch, minimum_trace_level,
        root_tolerance);

    py::dict out;
    out["phase_velocity"] = result.phase_velocity;
    out["group_velocity"] = result.group_velocity;
    out["hv_ratio"] = result.hv_ratio;
    out["converged_level"] = result.converged_level;
    out["relative_error"] = result.relative_error;
    out["phase_velocity_roots"] = result.root_trace.phase_velocity_roots;
    out["phase_accepted_level"] = result.root_trace.phase_accepted_level;
    out["generated_level"] = result.root_trace.generated_level;
    out["extrapolate_start_level"] =
        result.root_trace.extrapolate_start_level;
    return out;
}

py::dict phase_velocity_curve_result_to_dict(
    const std::vector<double>& periods,
    const PhaseVelocityCurveResult& result
) {
    py::dict out;
    out["periods"] = periods;
    out["phase_velocities"] = result.phase_velocities;
    out["converged_levels"] = result.converged_levels;
    out["relative_errors"] = result.relative_errors;
    out["fallback_mask"] = result.fallback_mask;
    out["fallback_levels"] = result.fallback_levels;
    out["fallback_count"] = static_cast<int>(std::count(
        result.fallback_mask.begin(), result.fallback_mask.end(), true));
    out["direct_prediction_count"] = result.direct_prediction_count;
    return out;
}

py::dict py_computeDispersionCurveLayerCake(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    const std::vector<double>& periods,
    const std::string& wave_type_str,
    int branch,
    double c_min,
    double c_max,
    int scan_points,
    double root_tolerance,
    bool verify_branch
) {
    PhaseVelocityCurveResult result;
    {
        py::gil_scoped_release release;
        result = compute_dispersion_curve_layer_cake_impl(
            layers, thicknesses, halfspace, periods,
            parseWaveType(wave_type_str), branch, c_min, c_max,
            scan_points, root_tolerance, verify_branch);
    }
    return phase_velocity_curve_result_to_dict(periods, result);
}

py::dict py_computeDispersionCurveRombergGridVelocityVti(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vphs,
    const std::vector<double>& vpvs,
    const std::vector<double>& vshs,
    const std::vector<double>& vsvs,
    const std::vector<double>& etas,
    const std::vector<bool>& is_discontinuity,
    const std::vector<double>& periods,
    const std::string& wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool adaptive_root_scan,
    double initial_root_window_frac,
    int extrapolate_start_level,
    const std::string& grid_policy,
    double planet_radius,
    double flattening_exponent,
    std::optional<int> branch
) {
    const GridRefinementPolicy refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    PhaseVelocityCurveResult result;
    {
        py::gil_scoped_release release;
        result = compute_dispersion_curve_romberg_grid_velocity_vti_impl(
            depths, rhos, vphs, vpvs, vshs, vsvs, etas,
            is_discontinuity, periods, parseWaveType(wave_type_str),
            c_guess, max_steps, tolerance, adaptive_root_scan,
            initial_root_window_frac, extrapolate_start_level,
            refinement_policy, branch);
    }
    return phase_velocity_curve_result_to_dict(periods, result);
}

py::dict py_computeRayleighEigenfunctions(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    int numSubLayers
) {
    auto results = computeRayleighEigenfunctions(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, numSubLayers);
    size_t n = results.size();
    auto depth_arr = py::array_t<double>(n);
    auto ur_arr = py::array_t<std::complex<double>>(n);
    auto uz_arr = py::array_t<std::complex<double>>(n);
    auto szz_arr = py::array_t<std::complex<double>>(n);
    auto szr_arr = py::array_t<std::complex<double>>(n);
    auto is_fluid_arr = py::array_t<bool>(n);
    auto depth_ptr = depth_arr.mutable_unchecked<1>();
    auto ur_ptr = ur_arr.mutable_unchecked<1>();
    auto uz_ptr = uz_arr.mutable_unchecked<1>();
    auto szz_ptr = szz_arr.mutable_unchecked<1>();
    auto szr_ptr = szr_arr.mutable_unchecked<1>();
    auto is_fluid_ptr = is_fluid_arr.mutable_unchecked<1>();
    for(size_t i=0; i<n; ++i) {
        depth_ptr(i) = results[i].depth;
        ur_ptr(i) = results[i].ur;
        uz_ptr(i) = results[i].uz;
        szz_ptr(i) = results[i].szz;
        szr_ptr(i) = results[i].szr;
        is_fluid_ptr(i) = results[i].is_fluid;
    }
    py::dict d;
    d["depth"] = depth_arr; d["ur"] = ur_arr; d["uz"] = uz_arr; d["szz"] = szz_arr; d["szr"] = szr_arr;
    d["is_fluid"] = is_fluid_arr;
    return d;
}

py::dict py_computeLoveEigenfunctions(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    int numSubLayers
) {
    auto results = computeLoveEigenfunctions(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, numSubLayers);
    size_t n = results.size();
    auto depth_arr = py::array_t<double>(n);
    auto uy_arr = py::array_t<std::complex<double>>(n);
    auto tyz_arr = py::array_t<std::complex<double>>(n);
    auto depth_ptr = depth_arr.mutable_unchecked<1>();
    auto uy_ptr = uy_arr.mutable_unchecked<1>();
    auto tyz_ptr = tyz_arr.mutable_unchecked<1>();
    for(size_t i=0; i<n; ++i) {
        depth_ptr(i) = results[i].depth;
        uy_ptr(i) = results[i].uy;
        tyz_ptr(i) = results[i].tyz;
    }
    py::dict d; d["depth"] = depth_arr; d["uy"] = uy_arr; d["tyz"] = tyz_arr;
    return d;
}

py::dict py_debugRayleighWeakformTrace(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    int numSubLayers,
    bool useAnalytical
) {
    require_non_gravitating_derivatives(layers, &halfSpaceParams);
    py::dict out;
    const double zero = 0.0;
    const double k = omega / phaseVelocity;
    const int N = static_cast<int>(layers.size());

    std::vector<RayleighStatePoint<double>> states;
    RayleighEnergyResult<double> energy;
    if (useAnalytical) {
        states = propagateRayleighEigenstates<double>(
            layers, thicknesses, halfSpaceParams, omega, phaseVelocity, 1);
        energy = computeRayleighEnergyIntegrals_Analytical<double>(
            states, layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
    } else {
        states = propagateRayleighEigenstates<double>(
            layers, thicknesses, halfSpaceParams, omega, phaseVelocity, numSubLayers);
        energy = computeRayleighEnergyIntegrals_CT<double>(
            states, layers, thicknesses, halfSpaceParams, omega, phaseVelocity, numSubLayers);
    }

    auto state_norm = [](const RayleighStatePoint<double>& pt) {
        return std::sqrt(
            std::norm(pt.ur) + std::norm(pt.uz) + std::norm(pt.szz) + std::norm(pt.szr));
    };

    size_t n_states = states.size();
    auto depth_arr = py::array_t<double>(n_states);
    auto ur_arr = py::array_t<std::complex<double>>(n_states);
    auto uz_arr = py::array_t<std::complex<double>>(n_states);
    auto szz_arr = py::array_t<std::complex<double>>(n_states);
    auto szr_arr = py::array_t<std::complex<double>>(n_states);
    auto norm_arr = py::array_t<double>(n_states);
    auto depth_ptr = depth_arr.mutable_unchecked<1>();
    auto ur_ptr = ur_arr.mutable_unchecked<1>();
    auto uz_ptr = uz_arr.mutable_unchecked<1>();
    auto szz_ptr = szz_arr.mutable_unchecked<1>();
    auto szr_ptr = szr_arr.mutable_unchecked<1>();
    auto norm_ptr = norm_arr.mutable_unchecked<1>();
    for (size_t i = 0; i < n_states; ++i) {
        depth_ptr(i) = states[i].depth;
        ur_ptr(i) = states[i].ur;
        uz_ptr(i) = states[i].uz;
        szz_ptr(i) = states[i].szz;
        szr_ptr(i) = states[i].szr;
        norm_ptr(i) = state_norm(states[i]);
    }

    std::vector<double> layer_norm_top(N, zero);
    std::vector<double> layer_norm_bottom(N, zero);
    std::vector<double> layer_norm_max(N, zero);
    if (!states.empty()) {
        std::vector<double> layer_bottoms(N, zero);
        double cumulative_depth = zero;
        for (int layer_i = 0; layer_i < N; ++layer_i) {
            cumulative_depth += thicknesses[layer_i];
            layer_bottoms[layer_i] = cumulative_depth;
        }

        size_t idx = 0;
        for (int layer_i = 0; layer_i < N; ++layer_i) {
            if (idx >= states.size()) break;
            const double layer_bottom = layer_bottoms[layer_i];
            const double tol = 1e-9 * std::max(1.0, layer_bottom);
            layer_norm_top[layer_i] = state_norm(states[idx]);
            double max_norm = layer_norm_top[layer_i];
            while (idx + 1 < states.size() && states[idx + 1].depth <= layer_bottom + tol) {
                ++idx;
                max_norm = std::max(max_norm, state_norm(states[idx]));
            }
            layer_norm_bottom[layer_i] = state_norm(states[idx]);
            layer_norm_max[layer_i] = max_norm;
        }
    }

    py::dict layer_integrals;
    auto int_uz2_ur2 = py::array_t<double>(N + 1);
    auto int_szz2 = py::array_t<double>(N + 1);
    auto int_szr2 = py::array_t<double>(N + 1);
    auto int_ur2 = py::array_t<double>(N + 1);
    auto int_uz_szr = py::array_t<double>(N + 1);
    auto int_szz_ur = py::array_t<double>(N + 1);
    auto i1_contrib = py::array_t<double>(N + 1);
    auto i2_contrib = py::array_t<double>(N + 1);
    auto i3_contrib = py::array_t<double>(N + 1);
    auto layer_thickness = py::array_t<double>(N + 1);
    auto i_uz2_ur2_ptr = int_uz2_ur2.mutable_unchecked<1>();
    auto i_szz2_ptr = int_szz2.mutable_unchecked<1>();
    auto i_szr2_ptr = int_szr2.mutable_unchecked<1>();
    auto i_ur2_ptr = int_ur2.mutable_unchecked<1>();
    auto i_uz_szr_ptr = int_uz_szr.mutable_unchecked<1>();
    auto i_szz_ur_ptr = int_szz_ur.mutable_unchecked<1>();
    auto i1_ptr = i1_contrib.mutable_unchecked<1>();
    auto i2_ptr = i2_contrib.mutable_unchecked<1>();
    auto i3_ptr = i3_contrib.mutable_unchecked<1>();
    auto thick_ptr = layer_thickness.mutable_unchecked<1>();

    for (int j = 0; j <= N; ++j) {
        const auto& lint = energy.layer_integrals[j];
        i_uz2_ur2_ptr(j) = lint.int_uz2_ur2;
        i_szz2_ptr(j) = lint.int_szz2;
        i_szr2_ptr(j) = lint.int_szr2;
        i_ur2_ptr(j) = lint.int_ur2;
        i_uz_szr_ptr(j) = lint.int_uz_szr;
        i_szz_ur_ptr(j) = lint.int_szz_ur;

        if (j < N) {
            const auto& p = layers[j];
            i1_ptr(j) = p.rho * lint.int_uz2_ur2;
            if (isLiquidLayer(p)) {
                i2_ptr(j) = lint.int_szz2 / p.c33;
                i3_ptr(j) = -lint.int_szz_ur / k;
            } else {
                const double a_coeff = p.c11 - p.c13 * p.c13 / p.c33;
                i2_ptr(j) = lint.int_szz2 / p.c33 + lint.int_szr2 / p.c44 + k * k * a_coeff * lint.int_ur2;
                i3_ptr(j) = a_coeff * lint.int_ur2 + (lint.int_uz_szr - (p.c13 / p.c33) * lint.int_szz_ur) / k;
            }
            thick_ptr(j) = thicknesses[j];
        } else {
            const auto& p = halfSpaceParams;
            const double a_coeff = p.c11 - p.c13 * p.c13 / p.c33;
            i1_ptr(j) = p.rho * lint.int_uz2_ur2;
            i2_ptr(j) = lint.int_szz2 / p.c33 + lint.int_szr2 / p.c44 + k * k * a_coeff * lint.int_ur2;
            i3_ptr(j) = a_coeff * lint.int_ur2 + (lint.int_uz_szr - (p.c13 / p.c33) * lint.int_szz_ur) / k;
            thick_ptr(j) = zero;
        }
    }

    layer_integrals["int_uz2_ur2"] = int_uz2_ur2;
    layer_integrals["int_szz2"] = int_szz2;
    layer_integrals["int_szr2"] = int_szr2;
    layer_integrals["int_ur2"] = int_ur2;
    layer_integrals["int_uz_szr"] = int_uz_szr;
    layer_integrals["int_szz_ur"] = int_szz_ur;
    layer_integrals["I1_contrib"] = i1_contrib;
    layer_integrals["I2_contrib"] = i2_contrib;
    layer_integrals["I3_contrib"] = i3_contrib;
    layer_integrals["thickness"] = layer_thickness;

    out["depth"] = depth_arr;
    out["ur"] = ur_arr;
    out["uz"] = uz_arr;
    out["szz"] = szz_arr;
    out["szr"] = szr_arr;
    out["state_norm"] = norm_arr;
    out["layer_norm_top"] = layer_norm_top;
    out["layer_norm_bottom"] = layer_norm_bottom;
    out["layer_norm_max"] = layer_norm_max;
    out["layer_integrals"] = layer_integrals;
    out["I1"] = energy.I1;
    out["I2"] = energy.I2;
    out["I3"] = energy.I3;
    out["group_velocity_energy"] =
        (std::abs(energy.I1) > 1e-30) ? (energy.I3 / (phaseVelocity * energy.I1)) : std::numeric_limits<double>::quiet_NaN();
    return out;
}

py::tuple py_computeEnergyIntegrals(
    py::dict ef_dict,
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type
) {
    if (wave_type == "Rayleigh") {
        auto depth_arr = ef_dict["depth"].cast<py::array_t<double>>();
        auto ur_arr = ef_dict["ur"].cast<py::array_t<std::complex<double>>>();
        auto uz_arr = ef_dict["uz"].cast<py::array_t<std::complex<double>>>();
        auto szz_arr = ef_dict["szz"].cast<py::array_t<std::complex<double>>>();
        auto szr_arr = ef_dict["szr"].cast<py::array_t<std::complex<double>>>();
        size_t n = depth_arr.size();
        std::vector<RayleighEigenfunctionPoint<double>> efs(n);
        auto depth_ptr = depth_arr.data(); auto ur_ptr = ur_arr.data();
        auto uz_ptr = uz_arr.data(); auto szz_ptr = szz_arr.data(); auto szr_ptr = szr_arr.data();
        for(size_t i=0; i<n; ++i) {
            efs[i].depth = depth_ptr[i]; efs[i].ur = ur_ptr[i]; efs[i].uz = uz_ptr[i];
            efs[i].szz = szz_ptr[i]; efs[i].szr = szr_ptr[i];
        }
        auto res = computeRayleighEnergyIntegrals(efs, layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
        return py::make_tuple(res.I1, res.I2, res.I3, res.phaseVelocity, res.groupVelocity_Energy);
    } else {
        auto depth_arr = ef_dict["depth"].cast<py::array_t<double>>();
        auto uy_arr = ef_dict["uy"].cast<py::array_t<std::complex<double>>>();
        auto tyz_arr = ef_dict["tyz"].cast<py::array_t<std::complex<double>>>();
        size_t n = depth_arr.size();
        std::vector<LoveEigenfunctionPoint<double>> efs(n);
        auto depth_ptr = depth_arr.data(); auto uy_ptr = uy_arr.data(); auto tyz_ptr = tyz_arr.data();
        for(size_t i=0; i<n; ++i) {
            efs[i].depth = depth_ptr[i]; efs[i].uy = uy_ptr[i]; efs[i].tyz = tyz_ptr[i];
        }
        auto res = computeLoveEnergyIntegrals(efs, omega, phaseVelocity, layers, thicknesses, halfSpaceParams);
        return py::make_tuple(res.I1, res.I2, res.I3, res.phaseVelocity, res.groupVelocity_Energy);
    }
}

py::tuple py_computeRayleighEnergyIntegrals_Analytical(
    py::dict ef_dict,
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity
) {
    auto depth_arr = ef_dict["depth"].cast<py::array_t<double>>();
    auto ur_arr = ef_dict["ur"].cast<py::array_t<std::complex<double>>>();
    auto uz_arr = ef_dict["uz"].cast<py::array_t<std::complex<double>>>();
    auto szz_arr = ef_dict["szz"].cast<py::array_t<std::complex<double>>>();
    auto szr_arr = ef_dict["szr"].cast<py::array_t<std::complex<double>>>();
    size_t n = depth_arr.size();
    std::vector<RayleighEigenfunctionPoint<double>> efs(n);
    auto depth_ptr = depth_arr.data(); auto ur_ptr = ur_arr.data();
    auto uz_ptr = uz_arr.data(); auto szz_ptr = szz_arr.data(); auto szr_ptr = szr_arr.data();
    for(size_t i=0; i<n; ++i) {
        efs[i].depth = depth_ptr[i]; efs[i].ur = ur_ptr[i]; efs[i].uz = uz_ptr[i];
        efs[i].szz = szz_ptr[i]; efs[i].szr = szr_ptr[i];
    }
    auto res = computeRayleighEnergyIntegrals_Analytical(efs, layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
    return py::make_tuple(res.I1, res.I2, res.I3, res.phaseVelocity);
}

py::tuple py_computeLoveEnergyIntegrals_Analytical(
    py::dict ef_dict,
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity
) {
    auto depth_arr = ef_dict["depth"].cast<py::array_t<double>>();
    auto uy_arr = ef_dict["uy"].cast<py::array_t<std::complex<double>>>();
    auto tyz_arr = ef_dict["tyz"].cast<py::array_t<std::complex<double>>>();
    size_t n = depth_arr.size();
    std::vector<LoveEigenfunctionPoint<double>> efs(n);
    auto depth_ptr = depth_arr.data(); auto uy_ptr = uy_arr.data(); auto tyz_ptr = tyz_arr.data();
    for(size_t i=0; i<n; ++i) {
        efs[i].depth = depth_ptr[i]; efs[i].uy = uy_ptr[i]; efs[i].tyz = tyz_ptr[i];
    }
    auto res = computeLoveEnergyIntegrals_Analytical(efs, layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
    return py::make_tuple(res.I1, res.I2, res.I3, res.phaseVelocity);
}

py::tuple py_integrateGridLayer_Romberg(
    py::array_t<std::complex<double>> y_top_arr,
    const PhysicalParameters<double>& p_top,
    const PhysicalParameters<double>& p_bot,
    double thickness,
    double omega,
    double phaseVelocity,
    const std::string& wave_type,
    int max_depth = 5,
    double tolerance = 1e-8,
    bool use_romberg = true
) {
    auto buf = y_top_arr.request();
    Eigen::VectorXcd y_top = Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf.ptr, buf.size);
    WaveType type = (wave_type == "Love") ? WaveType::Love : WaveType::Rayleigh;
    // These generic depth utilities historically take [u,c44*du/ddepth].
    // Keep that public contract; the internal SH propagation uses upward T.
    if (type == WaveType::Love) y_top(1) = -y_top(1);
    auto res = integrateGridLayer_Romberg(y_top, p_top, p_bot, thickness, omega, phaseVelocity, type, max_depth, tolerance, use_romberg);
    return py::make_tuple(res.I1, res.I2, res.I3, res.phaseVelocity);
}

py::array_t<std::complex<double>> py_compute_layer_matrix_p_sv(
    const PhysicalParameters<double>& p,
    double thickness,
    double omega,
    double phaseVelocity
) {
    WaveVariables<double> wave;
    wave.omega = omega; wave.k = omega / phaseVelocity; wave.z = thickness;
    auto B = computeLayerMatrixP_SV_Explicit(p, wave);
    py::array_t<std::complex<double>> result({4, 4});
    auto buf = result.request();
    std::complex<double>* ptr = (std::complex<double>*)buf.ptr;
    Eigen::Map<Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor>> map(ptr);
    map = B;
    return result;
}

py::array_t<std::complex<double>> py_compute_layer_matrix_sh(
    const PhysicalParameters<double>& p,
    double thickness,
    double omega,
    double phaseVelocity
) {
    WaveVariables<double> wave;
    wave.omega = omega; wave.k = omega / phaseVelocity; wave.z = thickness;
    auto B = computeLayerMatrixSH(p, wave);
    py::array_t<std::complex<double>> result({2, 2});
    auto buf = result.request();
    std::complex<double>* ptr = (std::complex<double>*)buf.ptr;
    Eigen::Map<Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor>> map(ptr);
    map = B;
    return result;
}

py::array_t<std::complex<double>> py_propagate_romberg_vector(
    py::array_t<std::complex<double>> y_top_arr,
    const PhysicalParameters<double>& p_top_layer,
    const PhysicalParameters<double>& p_bot_layer,
    double full_layer_thickness,
    double target_depth,
    double omega,
    double phaseVelocity,
    const std::string& wave_type,
    int max_depth,
    double tolerance
) {
    auto buf = y_top_arr.request();
    Eigen::VectorXcd y_top = Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf.ptr, buf.size);
    WaveType type = (wave_type == "Love") ? WaveType::Love : WaveType::Rayleigh;
    // Legacy depth-state API -> upward SH state.
    if (type == WaveType::Love) y_top(1) = -y_top(1);
    Eigen::VectorXcd res = propagate_romberg_vector(
        y_top, p_top_layer, p_bot_layer, full_layer_thickness, target_depth, 
        omega, phaseVelocity, type, max_depth, tolerance
    );
    if (type == WaveType::Love) res(1) = -res(1);
    py::array_t<std::complex<double>> result(res.size());
    auto buf2 = result.request();
    Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf2.ptr, res.size()) = res;
    return result;
}

py::array_t<std::complex<double>> py_compute_state_at_depth(
    py::array_t<std::complex<double>> y_top_arr,
    const PhysicalParameters<double>& p_top_layer,
    const PhysicalParameters<double>& p_bot_layer,
    double full_layer_thickness,
    double target_depth,
    double omega,
    double phaseVelocity,
    const std::string& wave_type,
    const std::string& mode, // "GRID" or "LAYER_CAKE"
    int max_depth,
    double tolerance
) {
    auto buf = y_top_arr.request();
    Eigen::VectorXcd y_top = Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf.ptr, buf.size);
    WaveType type = (wave_type == "Love") ? WaveType::Love : WaveType::Rayleigh;
    if (type == WaveType::Love) y_top(1) = -y_top(1);
    bool use_romberg = (mode == "GRID");
    Eigen::VectorXcd res = computeStateAtDepth(
        y_top, p_top_layer, p_bot_layer, full_layer_thickness, target_depth,
        omega, phaseVelocity, type, use_romberg, max_depth, tolerance
    );
    if (type == WaveType::Love) res(1) = -res(1);
    py::array_t<std::complex<double>> result(res.size());
    auto buf2 = result.request();
    Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf2.ptr, res.size()) = res;
    return result;
}

py::array_t<std::complex<double>> py_evaluate_global_state(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    py::array_t<std::complex<double>> y_initial_arr,
    double target_depth,
    double omega,
    double phaseVelocity,
    const std::string& wave_type,
    const std::string& mode, // "GRID" or "LAYER_CAKE"
    int max_depth,
    double tolerance
) {
    auto buf = y_initial_arr.request();
    Eigen::VectorXcd y_curr = Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf.ptr, buf.size);
    WaveType type = (wave_type == "Love") ? WaveType::Love : WaveType::Rayleigh;
    if (type == WaveType::Love) y_curr(1) = -y_curr(1);
    bool use_romberg = (mode == "GRID");
    double current_z = 0.0;
    int N = layers.size();
    bool reached = false;
    for (int i = 0; i < N; ++i) {
        double H = thicknesses[i];
        double next_z = current_z + H;
        if (target_depth <= next_z) {
            double depth_in_layer = target_depth - current_z;
            y_curr = computeStateAtDepth(y_curr, layers[i], layers[i], H, depth_in_layer, omega, phaseVelocity, type, use_romberg, max_depth, tolerance);
            reached = true; break;
        } else {
            y_curr = computeStateAtDepth(y_curr, layers[i], layers[i], H, H, omega, phaseVelocity, type, use_romberg, max_depth, tolerance);
            current_z = next_z;
        }
    }
    if (!reached) {
        double depth_in_hs = target_depth - current_z;
        y_curr = computeStateAtDepth(y_curr, halfSpaceParams, halfSpaceParams, depth_in_hs + 1.0, depth_in_hs, omega, phaseVelocity, type, use_romberg, max_depth, tolerance);
    }
    if (type == WaveType::Love) y_curr(1) = -y_curr(1);
    py::array_t<std::complex<double>> result(y_curr.size());
    auto buf2 = result.request();
    Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf2.ptr, y_curr.size()) = y_curr;
    return result;
}

py::dict py_computePhaseVelocityHessian(
    std::vector<PhysicalParameters<double>> layers,
    std::vector<double> thicknesses,
    PhysicalParameters<double> hs,
    double omega, double cp,
    std::string wave_type_str, std::string mode_str,
    bool compute_thickness
) {
    require_non_gravitating_derivatives(layers, &hs);
    WaveType type = (wave_type_str == "Love" || wave_type_str == "love") ? WaveType::Love : WaveType::Rayleigh;
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    // Route through the wave-type dispatcher (not ...Stable directly): Stable
    // only consults `type` for the initial Newton polish, and always assembles
    // the Hessian with the Rayleigh propagator. A bare Love argument used to
    // silently mix a Love root with a Rayleigh derivative assembly.
    HessianResult res = computePhaseVelocityHessianFullAD(layers, thicknesses, hs, omega / (2.0 * M_PI), cp, type, mode, compute_thickness);
    py::dict out;
    out["hessian"] = res.hessian;
    out["gradient"] = res.gradient;
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"] = res.layerIndices;
    return out;
}

py::dict py_computePhaseVelocityHessianFullAD(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string wave_type_str, std::string mode_str,
    bool compute_thickness
) {
    require_non_gravitating_derivatives(layers, &hs);
    WaveType type = (wave_type_str == "Love" || wave_type_str == "love") ? WaveType::Love : WaveType::Rayleigh;
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    auto res = computePhaseVelocityHessianFullAD(layers, thicknesses, hs, omega / (2.0 * M_PI), cp, type, mode, compute_thickness);
    py::dict out;
    out["hessian"] = Eigen::MatrixXd(res.hessian);
    out["gradient"] = Eigen::VectorXd(res.gradient);
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"] = res.layerIndices;
    return out;
}

// Fixed-level full-AD Hessian for a small GRID model.  This is intentionally a
// flat-policy validation path: the full AD engine differentiates the refined
// layer stack, and the cached linear Mesher map pulls the result back to node
// VTI parameters.  Production GRID Hessians remain on the variational path.
py::dict py_computePhaseVelocityHessianGridFullAD(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq, double c_guess,
    const std::string& wave_type_str = "Rayleigh",
    const std::string& mode_str = "VTI",
    int refinement_level = 0,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN()
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    model.require_valid();
    if (!model.refinement_policy.is_flat()) {
        throw std::invalid_argument(
            "compute_hessian_ad_grid is a flat-policy validation path; "
            "use variational Grid Hessian for spherical policies");
    }
    if (mode_str != "VTI" && mode_str != "vti") {
        throw std::invalid_argument(
            "compute_hessian_ad_grid currently requires sensitivity_mode='VTI'");
    }

    const WaveType type = parseWaveType(wave_type_str);
    Mesher mesher(model);
    auto [layers, thicknesses] = mesher.get_refined_layers(refinement_level);
    const auto halfspace = mesher.get_halfspace();
    const double omega = 2.0 * M_PI * freq;
    const double c_root = find_dispersion_root_near_by_scan(
        layers, thicknesses, halfspace, omega, c_guess, type, 1e-6);
    const int npl = 6;
    const int n_nodes = static_cast<int>(params.size());
    const int n_sub = static_cast<int>(layers.size());
    const int n_ref = (n_sub + 1) * npl;
    if (n_ref + 1 > MAX_AD_PARAMS) {
        throw std::invalid_argument(
            "compute_hessian_ad_grid exceeds MAX_AD_PARAMS at this refinement level");
    }

    const auto refined = computePhaseVelocityHessianFullAD(
        layers, thicknesses, halfspace, freq, c_root, type,
        SensitivityMode::VTI, false);

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_ref, n_nodes * npl);
    const auto& geo = mesher.get_refined_geometry(refinement_level);
    if (static_cast<int>(geo.size()) != n_sub) {
        throw std::runtime_error("GRID geometry/layer count mismatch in full-AD path");
    }
    for (int s = 0; s < n_sub; ++s) {
        const auto& g = geo[static_cast<size_t>(s)];
        const int i = g.parent_node;
        for (int p = 0; p < npl; ++p) {
            const double scale = (p == 0) ? g.density_scale : g.elastic_scale;
            B(s * npl + p, i * npl + p) += scale * (1.0 - g.alpha);
            B(s * npl + p, (i + 1) * npl + p) += scale * g.alpha;
        }
    }
    for (int p = 0; p < npl; ++p) {
        B(n_sub * npl + p, (n_nodes - 1) * npl + p) = 1.0;
    }

    Eigen::VectorXd gradient_node = B.transpose() * refined.gradient;
    Eigen::MatrixXd hessian_node = B.transpose() * refined.hessian * B;

    py::dict out;
    out["hessian"] = hessian_node;
    out["gradient"] = gradient_node;
    std::vector<std::string> names;
    std::vector<int> indices;
    const char* vti_names[] = {"Rho", "A", "C", "F", "L", "N"};
    for (int i = 0; i < n_nodes; ++i) {
        for (const char* name : vti_names) {
            names.emplace_back(name);
            indices.push_back(i);
        }
    }
    out["parameter_names"] = names;
    out["layer_indices"] = indices;
    out["phase_velocity"] = c_root;
    out["refinement_level"] = refinement_level;
    return out;
}

py::dict py_debugGetRefinedGridLayers(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    int level,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN()
) {
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();

    Mesher mesher(model);
    auto [layers, thicknesses] = mesher.get_refined_layers(level);
    auto halfspace = mesher.get_halfspace();

    py::dict out;
    out["layers"] = layers;
    out["thicknesses"] = thicknesses;
    out["halfspace"] = halfspace;
    return out;
}

py::dict py_debugGridLoveLevelGradientPieces(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq,
    double c_guess,
    int level,
    bool use_analytical,
    int num_sub_layers,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN()
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();

    Mesher mesher(model);
    auto [layers, thicknesses] = mesher.get_refined_layers(level);
    auto halfspace = mesher.get_halfspace();
    double omega = 2.0 * M_PI * freq;
    double c_root = find_dispersion_root_near_by_scan(
        layers, thicknesses, halfspace, omega, c_guess, WaveType::Love, 1e-6);
    auto pieces = computeVariationalGradientLovePieces_T<double>(
        layers, thicknesses, halfspace, omega, c_root, SensitivityMode::VTI,
        use_analytical, num_sub_layers, LoveBranchPolicy::Auto);

    py::dict out;
    out["cp"] = c_root;
    out["raw_gradient"] = Eigen::VectorXd(pieces.raw_gradient);
    out["gradient"] = Eigen::VectorXd(pieces.gradient);
    out["coeff"] = pieces.coeff;
    return out;
}

py::dict py_computePhaseVelocityHessianVariationalFDLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string mode_str,
    bool compute_thickness,
    int numSubLayers,
    double fdEps
) {
    require_non_gravitating_derivatives(layers, &hs);
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    auto res = computePhaseVelocityHessianVariationalFDLove(layers, thicknesses, hs, omega / (2.0 * M_PI), cp, mode, compute_thickness, numSubLayers, fdEps);
    py::dict out;
    out["hessian"] = Eigen::MatrixXd(res.hessian);
    out["gradient"] = Eigen::VectorXd(res.gradient);
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"] = res.layerIndices;
    return out;
}

py::dict py_computePhaseVelocityHessianWeakFormLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string mode_str,
    bool compute_thickness,
    bool use_analytical,
    std::string branch_policy_str,
    std::string backend_str
) {
    require_non_gravitating_derivatives(layers, &hs);
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    LoveBranchPolicy branch_policy = parseLoveBranchPolicy(branch_policy_str);
    WFHessianBackend backend = parseWFHessianBackend(backend_str);
    auto res = computePhaseVelocityHessianWeakFormLove(
        layers, thicknesses, hs, omega / (2.0 * M_PI), cp, mode, compute_thickness, 200, use_analytical, branch_policy,
        1.0, backend);
    py::dict out;
    out["hessian"] = Eigen::MatrixXd(res.hessian);
    out["gradient"] = Eigen::VectorXd(res.gradient);
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"] = res.layerIndices;
    return out;
}

py::dict py_debugLoveWeakformHessianBreakdown(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string mode_str,
    bool use_analytical,
    std::string branch_policy_str,
    int num_sub_layers,
    const std::vector<double>& provided_scales,
    double cp_grad_multiplier
) {
    require_non_gravitating_derivatives(layers, &hs);
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    LoveBranchPolicy branch_policy = parseLoveBranchPolicy(branch_policy_str);
    const std::vector<double>* scale_ptr = provided_scales.empty() ? nullptr : &provided_scales;
    auto res = computePhaseVelocityHessianWeakFormLoveBreakdown(
        layers, thicknesses, hs, omega / (2.0 * M_PI), cp, mode, num_sub_layers, use_analytical, branch_policy,
        scale_ptr, cp_grad_multiplier);
    py::dict out;
    out["hessian"] = Eigen::MatrixXd(res.hessian);
    out["hessian_unsym"] = Eigen::MatrixXd(res.hessian_unsym);
    out["hessian_coeff"] = Eigen::MatrixXd(res.hessian_coeff);
    out["hessian_coeff_unsym"] = Eigen::MatrixXd(res.hessian_coeff_unsym);
    out["hessian_raw"] = Eigen::MatrixXd(res.hessian_raw);
    out["hessian_raw_unsym"] = Eigen::MatrixXd(res.hessian_raw_unsym);
    out["gradient"] = Eigen::VectorXd(res.gradient);
    out["raw_gradient"] = Eigen::VectorXd(res.raw_gradient);
    out["coeff"] = res.coeff;
    out["I3"] = res.I3;
    out["I1_layer"] = res.I1_layer;
    out["I2_layer"] = res.I2_layer;
    out["I3_layer"] = res.I3_layer;
    out["I1_hs"] = res.I1_hs;
    out["I2_hs"] = res.I2_hs;
    out["I3_hs"] = res.I3_hs;
    out["int_u2_layer"] = res.int_u2_layer;
    out["int_dudz2_layer"] = res.int_dudz2_layer;
    out["int_u2_hs"] = res.int_u2_hs;
    out["int_dudz2_hs"] = res.int_dudz2_hs;
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"] = res.layerIndices;
    return out;
}

py::dict py_debugLoveWeakformGradientPieces(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string mode_str,
    bool use_analytical,
    std::string branch_policy_str,
    int num_sub_layers,
    const std::vector<double>& provided_scales
) {
    require_non_gravitating_derivatives(layers, &hs);
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    LoveBranchPolicy branch_policy = parseLoveBranchPolicy(branch_policy_str);
    double c_root = findLoveRootNear(layers, thicknesses, hs, omega, cp);
    const std::vector<double>* scale_ptr = provided_scales.empty() ? nullptr : &provided_scales;
    auto pieces = computeVariationalGradientLovePieces_T<double>(
        layers, thicknesses, hs, omega, c_root, mode, use_analytical, num_sub_layers, branch_policy, scale_ptr);

    py::dict out;
    out["coeff"] = pieces.coeff;
    out["raw_gradient"] = Eigen::VectorXd(pieces.raw_gradient);
    out["gradient"] = Eigen::VectorXd(pieces.gradient);
    out["cp"] = c_root;
    if (use_analytical) {
        auto scaled = buildLoveScaledBoundaryDataUpward<double>(
            layers, thicknesses, hs, omega, c_root, branch_policy, scale_ptr);
        out["scale_factors"] = scaled.scale_factors;
    } else {
        out["scale_factors"] = std::vector<double>{};
    }
    return out;
}

py::dict py_debugLoveLayerIntegralProbe(
    const PhysicalParameters<double>& layer,
    double thickness,
    double top_u,
    double top_tau,
    double omega,
    double cp,
    std::string branch_policy_str
) {
    LoveBranchPolicy branch_policy = parseLoveBranchPolicy(branch_policy_str);
    auto li = computeLoveLayerIntegrals<double>(
        top_u, top_tau, layer, thickness, omega, omega / cp, branch_policy);
    py::dict out;
    out["int_u2"] = li.int_u2;
    out["int_dudz2"] = li.int_dudz2;
    out["I1"] = layer.rho * li.int_u2;
    out["I2"] = layer.c44 * li.int_dudz2 + (omega / cp) * (omega / cp) * layer.c66 * li.int_u2;
    out["I3"] = layer.c66 * li.int_u2;
    return out;
}

py::dict py_debugLoveLayerIntegralHessian(
    const PhysicalParameters<double>& layer,
    double thickness,
    double top_u,
    double top_tau,
    double omega,
    double cp,
    std::string branch_policy_str
) {
    LoveBranchPolicy branch_policy = parseLoveBranchPolicy(branch_policy_str);
    using AD = HessianAD<MAX_AD_PARAMS>;
    const int n = 6;
    AD::s_an = n;

    auto make_var = [](double v, int idx) {
        AD a = AD::make_var(v, idx);
        return a;
    };

    PhysicalParameters<AD> layer_ad;
    layer_ad.rho = make_var(layer.rho, 0);
    layer_ad.c11 = make_var(layer.c11, 1);
    layer_ad.c33 = make_var(layer.c33, 2);
    layer_ad.c13 = make_var(layer.c13, 3);
    layer_ad.c44 = make_var(layer.c44, 4);
    layer_ad.c66 = make_var(layer.c66, 5);

    auto li = computeLoveLayerIntegrals<AD>(
        AD(top_u), AD(top_tau), layer_ad, AD(thickness), AD(omega), AD(omega / cp), branch_policy);
    AD i1 = layer_ad.rho * li.int_u2;
    AD i2 = layer_ad.c44 * li.int_dudz2 + AD((omega / cp) * (omega / cp)) * layer_ad.c66 * li.int_u2;

    py::dict out;
    out["I1"] = get_val(i1);
    out["I2"] = get_val(i2);
    out["I1_hess"] = Eigen::MatrixXd(i1.hr().block(0, 0, n, n).eval());
    out["I2_hess"] = Eigen::MatrixXd(i2.hr().block(0, 0, n, n).eval());
    return out;
}

py::dict py_debugLoveUpwardIFTHessianBreakdown(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string mode_str
) {
    require_non_gravitating_derivatives(layers, &hs);
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    auto res = computePhaseVelocityHessianLoveUpwardIFTBreakdown(
        layers, thicknesses, hs, omega / (2.0 * M_PI), cp, mode, false);
    py::dict out;
    out["hessian"] = Eigen::MatrixXd(res.hessian);
    out["hessian_fpp"] = Eigen::MatrixXd(res.hessian_fpp);
    out["hessian_fpc"] = Eigen::MatrixXd(res.hessian_fpc);
    out["hessian_fcc"] = Eigen::MatrixXd(res.hessian_fcc);
    out["hessian_fixed_unsym"] = Eigen::MatrixXd(res.hessian_fixed_unsym);
    out["hessian_chain_unsym"] = Eigen::MatrixXd(res.hessian_chain_unsym);
    out["gradient"] = Eigen::VectorXd(res.gradient);
    out["Fp"] = Eigen::VectorXd(res.Fp);
    out["Fpc"] = Eigen::VectorXd(res.Fpc);
    out["Fpp"] = Eigen::MatrixXd(res.Fpp);
    out["I1pp"] = Eigen::MatrixXd(res.I1pp);
    out["I2pp"] = Eigen::MatrixXd(res.I2pp);
    out["I1pp_layer"] = res.I1pp_layer;
    out["I2pp_layer"] = res.I2pp_layer;
    out["I1pp_layer_fixed_state"] = res.I1pp_layer_fixed_state;
    out["I2pp_layer_fixed_state"] = res.I2pp_layer_fixed_state;
    out["I1pp_hs"] = Eigen::MatrixXd(res.I1pp_hs);
    out["I2pp_hs"] = Eigen::MatrixXd(res.I2pp_hs);
    out["top_state_u"] = res.top_state_u;
    out["top_state_tau"] = res.top_state_tau;
    out["top_state_u_grad"] = res.top_state_u_grad;
    out["top_state_tau_grad"] = res.top_state_tau_grad;
    out["top_state_u_hess"] = res.top_state_u_hess;
    out["top_state_tau_hess"] = res.top_state_tau_hess;
    out["bottom_state_u"] = res.bottom_state_u;
    out["bottom_state_tau"] = res.bottom_state_tau;
    out["bottom_state_u_grad"] = res.bottom_state_u_grad;
    out["bottom_state_tau_grad"] = res.bottom_state_tau_grad;
    out["bottom_state_u_hess"] = res.bottom_state_u_hess;
    out["bottom_state_tau_hess"] = res.bottom_state_tau_hess;
    out["Fc"] = res.Fc;
    out["Fcc"] = res.Fcc;
    out["F"] = res.F;
    out["root_c_used"] = res.root_c_used;
    out["I1"] = res.I1;
    out["I2"] = res.I2;
    out["I3"] = res.I3;
    out["I1_layer"] = res.I1_layer;
    out["I2_layer"] = res.I2_layer;
    out["I3_layer"] = res.I3_layer;
    out["I1_hs"] = res.I1_hs;
    out["I2_hs"] = res.I2_hs;
    out["I3_hs"] = res.I3_hs;
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"] = res.layerIndices;
    return out;
}

py::dict py_computePhaseVelocityHessianWeakFormRayleigh(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string mode_str,
    bool compute_thickness,
    int numSubLayers,
    bool useAnalytical,
    std::string backend_str
) {
    require_non_gravitating_derivatives(layers, &hs);
    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ? SensitivityMode::VTI : SensitivityMode::Isotropic;
    WFHessianBackend backend = parseWFHessianBackend(backend_str);
    auto res = computePhaseVelocityHessianWeakFormRayleigh(layers, thicknesses, hs, omega / (2.0 * M_PI), cp, mode, compute_thickness, numSubLayers, useAnalytical, backend);
    py::dict out;
    out["hessian"] = Eigen::MatrixXd(res.hessian);
    out["gradient"] = Eigen::VectorXd(res.gradient);
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"] = res.layerIndices;
    return out;
}

py::dict py_computePhaseVelocityHessianWeakFormRayleigh_Grid(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq, double c_guess,
    const std::string& mode_str = "Isotropic",
    int max_levels = 5,
    double tolerance = 1e-6,
    const std::string& backend_str = "dual",
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();

    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ?
        SensitivityMode::VTI : SensitivityMode::Isotropic;
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_guess);
    }

    auto res = computePhaseVelocityHessianWeakFormRayleigh_Grid(
        model, freq, c_guess, mode, max_levels, tolerance,
        parseWFHessianBackend(backend_str),
        phase_velocity_roots ? &*phase_velocity_roots : nullptr);

    py::dict out;
    out["hessian"]         = Eigen::MatrixXd(res.hessian);
    out["gradient"]        = res.gradient;
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"]   = res.layerIndices;
    return out;
}

py::dict py_computePhaseVelocityHessianWeakFormLove_Grid(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq, double c_guess,
    const std::string& mode_str = "Isotropic",
    int max_levels = 5,
    double tolerance = 1e-6,
    double initial_root_window_frac = 0.0,
    const std::string& backend_str = "dual",
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();

    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ?
        SensitivityMode::VTI : SensitivityMode::Isotropic;
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_guess);
    }

    auto res = computePhaseVelocityHessianWeakFormLove_Grid(
        model, freq, c_guess, mode, max_levels, tolerance, initial_root_window_frac,
        parseWFHessianBackend(backend_str),
        phase_velocity_roots ? &*phase_velocity_roots : nullptr);

    py::dict out;
    out["hessian"]         = Eigen::MatrixXd(res.hessian);
    out["gradient"]        = res.gradient;
    out["parameter_names"] = res.parameterNames;
    out["layer_indices"]   = res.layerIndices;
    return out;
}

// Reduced-basis grid Hessian wrappers: J (num_params_int x P, node-VTI rows in
// [node*6 + modulus] order, modulus order [rho,c11,c33,c13,c44,c66]) is
// computed by the Python caller (inversion/disper26_inv/derivatives.py
// basis_node_jacobian) and seeds the AD directly, so the C++ side never
// materializes the full node-space Hessian. See
// docs/sensitivity_implementation_map.md and computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced.
py::dict py_computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq, double c_guess,
    const Eigen::MatrixXd& J,
    int max_levels = 5,
    double tolerance = 1e-6,
    const std::string& backend_str = "dual",
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_guess);
    }

    auto res = computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced(
        model, freq, c_guess, J, max_levels, tolerance,
        parseWFHessianBackend(backend_str),
        phase_velocity_roots ? &*phase_velocity_roots : nullptr);

    py::dict out;
    out["hessian_reduced"] = Eigen::MatrixXd(res.hessian_reduced);
    out["gradient_node"]   = Eigen::VectorXd(res.gradient_node);
    return out;
}

py::dict py_computePhaseVelocityHessianWeakFormLove_Grid_Reduced(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq, double c_guess,
    const Eigen::MatrixXd& J,
    int max_levels = 5,
    double tolerance = 1e-6,
    double initial_root_window_frac = 0.0,
    const std::string& backend_str = "dual",
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_guess);
    }

    auto res = computePhaseVelocityHessianWeakFormLove_Grid_Reduced(
        model, freq, c_guess, J, max_levels, tolerance, true, 1,
        initial_root_window_frac, parseWFHessianBackend(backend_str),
        phase_velocity_roots ? &*phase_velocity_roots : nullptr);

    py::dict out;
    out["hessian_reduced"] = Eigen::MatrixXd(res.hessian_reduced);
    out["gradient_node"]   = Eigen::VectorXd(res.gradient_node);
    return out;
}

// Standalone GRID x variational x 1st-order entry point (docs/naming_convention.md
// SS7.3, "space GRID x variational x 1st order"). Until now this quantity was only
// reachable as the gradient_node byproduct of the *_grid_reduced Hessian calls.
// Reuses that machinery with an empty (num_params_int x 0) J: the per-level
// AD Hessian-block loops are keyed on P = J.cols(), so with P=0 they do not run
// at all -- only the double-precision variational gradient g0_node_vti (computed
// once per level regardless of J) and its Neville extrapolation across levels
// execute, which is exactly this entry's job and skips the AD block cost that
// makes the Hessian calls expensive.
py::dict py_computeKernelsVariationalGrid(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq, double c_guess,
    const std::string& wave_type_str = "Rayleigh",
    int max_levels = 5,
    double tolerance = 1e-6,
    double initial_root_window_frac = 0.0,
    const std::string& backend_str = "dual",
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_guess);
    }

    // Parse for validation/backward compatibility even though the dedicated
    // first-order loop does not need an AD Hessian backend.
    (void)parseWFHessianBackend(backend_str);
    const WaveType wave_type = parseWaveType(wave_type_str);
    const auto result = computeKernelsVariationalGrid(
        model, freq, c_guess, wave_type, max_levels, tolerance,
        initial_root_window_frac,
        phase_velocity_roots ? &*phase_velocity_roots : nullptr);

    py::dict out;
    out["gradient_node"] = Eigen::VectorXd(result.gradient_node);
    out["accepted_level"] = result.accepted_level;
    out["relative_step"] = result.relative_step;
    out["phase_velocity_roots_used"] = result.phase_velocity_roots_used;
    return out;
}

py::dict py_debugPhaseVelocityHessianWeakFormLove_Grid(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double freq, double c_guess,
    const std::string& mode_str = "Isotropic",
    int max_levels = 5,
    double tolerance = 1e-6,
    bool use_analytical = true,
    int num_sub_layers = 1,
    double initial_root_window_frac = 0.0,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN()
) {
    require_non_gravitating_derivatives(params);
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    model.require_valid();

    SensitivityMode mode = (mode_str == "VTI" || mode_str == "vti") ?
        SensitivityMode::VTI : SensitivityMode::Isotropic;

    auto res = computePhaseVelocityHessianWeakFormLove_Grid_Debug(
        model, freq, c_guess, mode, max_levels, tolerance,
        use_analytical, num_sub_layers, initial_root_window_frac);

    py::list gradients_raw;
    py::list sublayer_gradients_raw;
    py::list hessians_raw;
    for (const auto& g_sub : res.sublayer_gradients_raw) sublayer_gradients_raw.append(Eigen::VectorXd(g_sub));
    for (const auto& g : res.gradients_raw) gradients_raw.append(Eigen::VectorXd(g));
    for (const auto& H : res.hessians_raw) hessians_raw.append(Eigen::MatrixXd(H));

    py::dict out;
    out["hessian"] = Eigen::MatrixXd(res.final_result.hessian);
    out["gradient"] = Eigen::VectorXd(res.final_result.gradient);
    out["parameter_names"] = res.final_result.parameterNames;
    out["layer_indices"] = res.final_result.layerIndices;
    out["attempted_levels"] = res.attempted_levels;
    out["accepted_levels"] = res.accepted_levels;
    out["accepted_mask"] = res.accepted_mask;
    out["c_roots_raw"] = res.c_roots_raw;
    out["gradient_norms_raw"] = res.gradient_norms_raw;
    out["hessian_norms_raw"] = res.hessian_norms_raw;
    out["gradient_rel_step_raw"] = res.gradient_rel_step_raw;
    out["hessian_rel_step_raw"] = res.hessian_rel_step_raw;
    out["sublayer_gradients_raw"] = sublayer_gradients_raw;
    out["gradients_raw"] = gradients_raw;
    out["hessians_raw"] = hessians_raw;
    out["stopped_on_error_increase"] = res.stopped_on_error_increase;
    out["stopped_on_tolerance"] = res.stopped_on_tolerance;
    out["rejected_level"] = res.rejected_level;
    return out;
}

py::dict py_computePhaseVelocityHessianRef(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& hs,
    double omega, double cp,
    std::string wave_type_str, std::string mode_str,
    bool compute_thickness
) {
    require_non_gravitating_derivatives(layers, &hs);
    WaveType type = (wave_type_str == "Love" || wave_type_str == "love") ? WaveType::Love : WaveType::Rayleigh;
    auto res = computePhaseVelocityHessianReference(layers, thicknesses, hs, omega / (2.0 * M_PI), cp, type);
    py::dict d;
    d["hessian"] = res.hessian;
    d["gradient"] = res.gradient;
    d["parameter_names"] = res.parameterNames;
    d["layer_indices"] = res.layerIndices;
    return d;
}

PYBIND11_MODULE(disper26py, m) {
    disper26::api::python::register_api(m);
    m.doc() = "Python bindings for Disper26 VTI Dispersion Engine";
    py::enum_<SensitivityMode>(m, "SensitivityMode")
        .value("Isotropic", SensitivityMode::Isotropic)
        .value("VTI", SensitivityMode::VTI)
        .export_values();
    py::enum_<NormalizationMode>(m, "NormalizationMode")
        .value("None", NormalizationMode::None)
        .value("Vertical", NormalizationMode::Vertical)
        .value("VectorNorm", NormalizationMode::VectorNorm)
        .export_values();
    py::class_<PhysicalParameters<double>>(m, "PhysicalParameters")
        .def(py::init<>())
        .def_readwrite("rho", &PhysicalParameters<double>::rho)
        .def_readwrite("c11", &PhysicalParameters<double>::c11)
        .def_readwrite("c13", &PhysicalParameters<double>::c13)
        .def_readwrite("c33", &PhysicalParameters<double>::c33)
        .def_readwrite("c44", &PhysicalParameters<double>::c44)
        .def_readwrite("c66", &PhysicalParameters<double>::c66)
        .def_readwrite("gravity", &PhysicalParameters<double>::gravity)
        .def_readwrite("gravity_internal_scale", &PhysicalParameters<double>::gravity_internal_scale)
        .def_readwrite("gravity_surface_scale", &PhysicalParameters<double>::gravity_surface_scale)
        .def("__repr__", [](const PhysicalParameters<double> &p) {
            std::stringstream ss;
            double rho = p.rho;
            if (rho > 1e-6) {
                double vs = std::sqrt(p.c44 / rho);
                double vp = std::sqrt(p.c11 / rho);
                bool iso = (std::abs(p.c11 - p.c33) < 1e-5 && std::abs(p.c44 - p.c66) < 1e-5);
                if (iso) {
                    if (p.c44 < 1e-10) ss << "<PhysicalParameters (Fluid) Vp=" << vp << " Rho=" << rho << " g=" << p.gravity << ">";
                    else ss << "<PhysicalParameters (Iso) Vp=" << vp << " Vs=" << vs << " Rho=" << rho << ">";
                } else ss << "<PhysicalParameters (VTI) Rho=" << rho << " ...>";
            } else ss << "<PhysicalParameters (Empty) Rho=" << rho << ">";
            return ss.str();
        })
        .def("get_vp", [](const PhysicalParameters<double> &p) { return std::sqrt(p.c11 / p.rho); })
        .def("get_vs", [](const PhysicalParameters<double> &p) { return std::sqrt(p.c44 / p.rho); });
        
    py::class_<SensitivityKernel>(m, "SensitivityKernel",
        "Sensitivity of a dispersion observable to one model parameter at one node/layer.\n"
        "Fields: parameter_name (e.g. 'Vs','Vp','Rho' for Isotropic; 'A','C','F','L','N','Rho'\n"
        "for VTI; 'Depth' for a discontinuity depth), layer_index (node/layer index; -1 for the\n"
        "halfspace), d_c = dc/dp (phase velocity), d_U = dU/dp (group velocity),\n"
        "d_HV = d(H/V)/dp. Unrequested components (compute_c/u/hv=False) are 0; d_U/d_HV are\n"
        "NaN where not defined (e.g. Love H/V).")
        .def_readonly("parameter_name", &SensitivityKernel::parameterName)
        .def_readonly("layer_index", &SensitivityKernel::layerIndex)
        .def_readonly("d_c", &SensitivityKernel::d_PhaseVelocity)
        .def_readonly("d_U", &SensitivityKernel::d_GroupVelocity)
        .def_readonly("d_HV", &SensitivityKernel::d_HVRatio);

    m.def("create_isotropic_parameters", &createIsotropicParameters<double>, py::arg("vs"), py::arg("vp"), py::arg("rho"));
    m.def("create_fluid_parameters", [](double vp, double rho, double gravity, double internal_scale, double surface_scale) {
        PhysicalParameters<double> p; p.rho = rho; p.c44 = 0.0; p.c11 = rho * vp * vp; p.c33 = p.c11; p.c13 = p.c11; p.c66 = 0.0; p.gravity = gravity; p.gravity_internal_scale = internal_scale; p.gravity_surface_scale = surface_scale;
        return p;
    }, py::arg("vp"), py::arg("rho"), py::arg("gravity") = 0.0, py::arg("internal_scale") = 1.0, py::arg("surface_scale") = 0.0);
    m.def("create_vti_layer_from_depth", &py_createVTILayerFromDepth);
    m.def("compute_mode_count", &py_calculateTotalModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Integer mode label: the AD-z zero-crossing counter. This is the counter to use.");
    m.def("calculate_mode_count", &py_calculateTotalModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use compute_mode_count.");
    m.def("calculate_total_mode_count", &py_calculateTotalModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use compute_mode_count. "
          "'total' says nothing about the method.");
    m.def("compute_continuous_mode_count", &py_calculateContinuousModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Legacy phase-accumulation diagnostic, not the production integer mode label. Use compute_mode_count for integer labels. The phase is undefined when both phase-pair components vanish; it is not globally smooth.");
    m.def("calculate_continuous_mode_count", &py_calculateContinuousModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use compute_continuous_mode_count.");
    m.def("calculate_legacy_continuous_mode_count", &py_calculateContinuousModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use compute_continuous_mode_count.");
    m.def("debug_continuous_mode_count", &py_debugModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Layer-by-layer legacy phase diagnostics, not production crossing labels. "
          "Phase-derived integer fields are diagnostic only. The phase is undefined "
          "when both phase-pair components vanish; use debug_mode_count for crossings.");
    m.def("debug_legacy_mode_count", &py_debugModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use debug_continuous_mode_count.");
    m.def("calculate_bran_mode_count", &py_calculateBranModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use compute_mode_count. There is "
          "no separate 'bran' counter: both names reach calculateAdModeCount with the same "
          "arguments.");
    m.def("debug_mode_count", &py_debugBranModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Crossing-by-crossing detail behind compute_mode_count. The legacy_continuous_count comparison is a legacy phase diagnostic and does not determine n_integer.");
    m.def("debug_mode_count_crossings", &py_debugBranModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use debug_mode_count.");
    m.def("debug_bran_mode_count", &py_debugBranModeCount,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh",
          "Deprecated alias. Use debug_mode_count.");
    m.def("compute_dispersion_value", &py_calculateDispersionValue);
    m.def("calculate_dispersion_value", &py_calculateDispersionValue,
          "Deprecated alias. Use compute_dispersion_value.");
    m.def("search_roots", &py_searchRoots);
    m.def("compute_kernels_fd_layer_cake", &py_computeSensitivityFD, py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"), py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false);
    m.def("compute_kernels_fd_pancake", &py_computeSensitivityFD, py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"), py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false, "Deprecated alias. Use compute_kernels_fd_layer_cake.\n");
    m.def("compute_sensitivity_kernels_fd", &py_computeSensitivityFD, py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"), py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_kernels_fd_layer_cake.");
    m.def("get_rayleigh_minors", &py_getRayleighMinors);
    m.def("compute_group_velocity_ad", &py_calculateGroupVelocityFullAD);
    m.def("calculate_group_velocity", &py_calculateGroupVelocity,
          "Deprecated alias. Use compute_group_velocity_ad.");
    m.def("compute_hv_ratio", &py_calculateHVRatio,
          "Signed y3/y1 at the solid surface/seabed; negative retrograde for k>0, z upward, ux=i*y3*exp(i(kx-omega*t)).");
    m.def("calculate_hv_ratio", &py_calculateHVRatio,
          "Deprecated alias. Use compute_hv_ratio.");
    m.def("mineos_physical_dispersion_factor", &py_mineosPhysicalDispersionFactor,
          py::arg("tref_s"), py::arg("period_s"),
          "Return the Mineos physical-dispersion factor 2*log(tref_s/period_s)/pi; returns 0 when tref_s <= 0.");
    m.def("apply_mineos_physical_dispersion", &py_applyMineosPhysicalDispersion,
          py::arg("params"), py::arg("qkappa"), py::arg("qmu"), py::arg("tref_s"), py::arg("period_s"),
          "Apply Mineos-style Q physical dispersion to one PhysicalParameters object.");
    m.def("apply_mineos_physical_dispersion_model", &py_applyMineosPhysicalDispersionModel,
          py::arg("params"), py::arg("qkappas"), py::arg("qmus"), py::arg("tref_s"), py::arg("period_s"),
          "Apply Mineos-style Q physical dispersion to a list of PhysicalParameters.");
    m.def("earth_flattening_transformation_layers", &py_earthFlatteningLayers,
          py::arg("layers_spherical"), py::arg("spherical_thicknesses"),
          py::arg("planet_radius"), py::arg("flattening_exponent") = -2.0,
          "Transform a spherical layer/layer_cake model to a flat layer model using inverse Bhattacharya-Arora P-SV scaling: solid elastic moduli scale as (r/R)^(-p), solid density as (r/R)^(2-p); fluid density is fixed and fluid C scales as (r/R)^(-2).");
    m.def("earth_flattening_transformation_grid", &py_earthFlatteningGrid,
          py::arg("depths_spherical"), py::arg("params_spherical"), py::arg("is_discontinuity"),
          py::arg("planet_radius"), py::arg("flattening_exponent") = -2.0,
          "Transform a spherical node/grid model to a flat grid model using inverse Bhattacharya-Arora P-SV scaling: solid elastic moduli scale as (r/R)^(-p), solid density as (r/R)^(2-p); fluid density is fixed and fluid C scales as (r/R)^(-2).");
    m.def("earth_flattening_transformation_layers_experimental_density", &py_earthFlatteningLayersExperimentalDensity,
          py::arg("layers_spherical"), py::arg("spherical_thicknesses"),
          py::arg("planet_radius"), py::arg("density_exponent") = 2.275,
          "Experimental legacy EFT: velocity scales as R/r, density as (r/R)^p, and elastic moduli follow rho*v^2.");
    m.def("earth_flattening_transformation_grid_experimental_density", &py_earthFlatteningGridExperimentalDensity,
          py::arg("depths_spherical"), py::arg("params_spherical"), py::arg("is_discontinuity"),
          py::arg("planet_radius"), py::arg("density_exponent") = 2.275,
          "Experimental legacy EFT for grid models: velocity scales as R/r, density as (r/R)^p, and elastic moduli follow rho*v^2.");
    m.def("compute_kernels_ad_layer_cake", &py_computeSensitivityFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          "LAYER_CAKE-mode sensitivity kernels (dc/dp, dU/dp, dHV/dp) on the given fixed layers via full "
          "(nested) AD through the propagator. Returns list[SensitivityKernel]. No Romberg extrapolation "
          "(fixed mesh). For the GRID/node-based version see compute_kernels_ad_grid; "
          "for the faster variational layer_cake version see compute_kernels_variational_layer_cake.");
    m.def("compute_kernels_ad_pancake", &py_computeSensitivityFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_kernels_ad_layer_cake.\n"
          "LAYER_CAKE-mode sensitivity kernels (dc/dp, dU/dp, dHV/dp) on the given fixed layers via full "
          "(nested) AD through the propagator. Returns list[SensitivityKernel]. No Romberg extrapolation "
          "(fixed mesh). For the GRID/node-based version see compute_kernels_ad_grid; "
          "for the faster variational layer_cake version see compute_kernels_variational_layer_cake.");
    m.def("compute_sensitivity_kernels_full_ad", &py_computeSensitivityFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_kernels_ad_layer_cake.");
    m.def("compute_kernels_variational_layer_cake", &py_computeSensitivityWeakForm,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false,
          py::arg("u_kernel_mode") = "variational_frequency_ad",
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true,
          py::arg("want_kernel_hv") = true,
          "Variational (weak-form) sensitivity kernels: dc/dp from the frozen-displacement "
          "variational gradient. dU/dp uses a width-1 frequency derivative of the phase "
          "variational kernel by default; u_kernel_mode='variational' retains the per-parameter "
          "energy-integral AD path. The target switches zero unrequested kernel fields.");
    m.def("compute_kernels_variational_pancake", &py_computeSensitivityWeakForm,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false,
          py::arg("u_kernel_mode") = "variational_frequency_ad",
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true,
          py::arg("want_kernel_hv") = true,
          "Deprecated alias. Use compute_kernels_variational_layer_cake.\n"
          "Variational (weak-form) sensitivity kernels: dc/dp from the frozen-displacement "
          "variational gradient. dU/dp uses a width-1 frequency derivative of the phase "
          "variational kernel by default; u_kernel_mode='variational' retains the per-parameter "
          "energy-integral AD path. The target switches zero unrequested kernel fields.");
    m.def("compute_sensitivity_kernels_weakform", &py_computeSensitivityWeakForm,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false,
          py::arg("u_kernel_mode") = "variational_frequency_ad",
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true,
          py::arg("want_kernel_hv") = true,
          "Deprecated alias. Use compute_kernels_variational_layer_cake.");
    m.def("calculate_group_velocity_full_ad", &py_calculateGroupVelocityFullAD,
          "Deprecated alias. Use compute_group_velocity_ad.");
    m.def("get_dispersion_eval_counts", []() {
        py::dict d;
        d["total"] = disper26_stats::dispersion_evals_total;
        d["double"] = disper26_stats::dispersion_evals_double;
        d["ad"] = disper26_stats::dispersion_evals_total - disper26_stats::dispersion_evals_double;
        return d;
    }, "Per-thread counts of dispersion-function evaluations since last reset: total, double (plain scalar), ad (AD-typed).");
    m.def("reset_dispersion_eval_counts", []() { disper26_stats::reset_dispersion_evals(); },
          "Reset the per-thread dispersion-function evaluation counters to zero.");
    m.def("compute_frequency_curvature_ad_grid", &py_computeFrequencyCurvatureAdGrid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type") = "Rayleigh",
          py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 6,
          py::arg("tolerance") = 1e-10,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("extrapolate_start_level") = 0,
          py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") =
              std::numeric_limits<double>::quiet_NaN(),
          "Compute GRID phase velocity c, group velocity U, total dU/domega, "
          "and d2k/domega2 with a width-1 outer AD pass at each independent "
          "Romberg refinement level. omega is angular frequency [rad/s]. "
          "The dedicated curvature path always evaluates through max_steps "
          "unless phase-velocity round-off causes the last level to be "
          "reverted. Existing dispersion APIs and stopping rules are not "
          "modified. Returns a dict containing dimensional derivatives, "
          "dimensionless D1/D2 diagnostics, errors, root residual, and status.");
    m.def("compute_dispersion_romberg", &py_computeDispersionRomberg,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("extrapolate_start_level") = 0,
          py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("branch") = py::none(),
          "GRID-mode phase/group velocity and H/V via Romberg (Richardson) extrapolation over "
          "layer-refinement levels.\n"
          "Returns (c, U, HV, level, rel_err): phase velocity c [km/s], group velocity U, H/V ratio, "
          "converged Romberg level, relative error.\n"
          "phase_only=True skips U and H/V (returned NaN) for a faster phase-velocity-only sweep. "
          "adaptive_root_scan=True (default) shrinks the per-level root search. "
          "branch=n selects mode order n independently on every refinement level; "
          "the default None preserves nearest-root behavior. "
          "initial_root_window_frac>0 tries a narrow level-0 sweep bracket before falling back. "
          "extrapolate_start_level=1 excludes pre-asymptotic level 0 from the extrapolation table. "
          "See docs/Romberg.md.");
    m.def("compute_dispersion_romberg_levels", &py_computeDispersionRombergLevels,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type") = "Rayleigh",
          py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("extrapolate_start_level") = 0,
          py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") =
              std::numeric_limits<double>::quiet_NaN(),
          py::arg("branch") = py::none(),
          py::arg("minimum_trace_level") = 0,
          py::arg("root_tolerance") = py::none(),
          "Trace-bearing counterpart of compute_dispersion_romberg. The "
          "reported dispersion values use the same accepted phase table as "
          "the legacy five-value API, while phase_velocity_roots retains the "
          "raw root selected on each generated refinement level. "
          "minimum_trace_level may extend root generation without changing "
          "the reported dispersion values. root_tolerance controls only the "
          "per-level root solve; None preserves the legacy coupling to "
          "tolerance. Trace-only extension is capped at level 12 and fails "
          "closed if round-off or root failure prevents the requested depth. "
          "A positive minimum_trace_level requires branch so extra roots are "
          "selected by mode order. "
          "This trace API currently requires extrapolate_start_level=0; the "
          "legacy five-value API retains nonzero-start value calculations. "
          "Returns a dict.");
    m.def("compute_dispersion_romberg_velocity_iso", &py_computeDispersionRombergVelocityIso,
          py::arg("depths"), py::arg("rhos"), py::arg("vps"), py::arg("vss"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("extrapolate_start_level") = 0,
          py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Velocity-basis (rho,vp,vs) GRID dispersion (no kernels). "
          "Returns (c, U, HV, level, rel_err). See compute_dispersion_romberg.");
    m.def("compute_dispersion_romberg_velocity_vti", &py_computeDispersionRombergVelocityVti,
          py::arg("depths"), py::arg("rhos"), py::arg("vphs"), py::arg("vpvs"), py::arg("vshs"), py::arg("vsvs"), py::arg("etas"),
          py::arg("is_discontinuity"), py::arg("period"), py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("extrapolate_start_level") = 0,
          py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Velocity-basis (rho,Vph,Vpv,Vsh,Vsv,eta) GRID dispersion (no kernels). "
          "Returns (c, U, HV, level, rel_err). See compute_dispersion_romberg.");
    m.def("compute_dispersion_curve_layer_cake", &py_computeDispersionCurveLayerCake,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("periods"), py::arg("wave_type") = "Rayleigh",
          py::arg("branch") = 0, py::arg("c_min") = 0.1,
          py::arg("c_max") = 4.5, py::arg("scan_points") = 2400,
          py::arg("root_tolerance") = 1e-12,
          py::arg("verify_branch") = false,
          "Compute a strictly monotonic LAYER_CAKE phase-velocity period curve "
          "inside one C++ call. Roots are ordered by increasing phase "
          "velocity at the initial/fallback scan. Continuation uses a local "
          "predicted bracket and falls back to the full scan. "
          "verify_branch=True confirms predicted roots with the integer mode "
          "counter at additional cost. Returns a diagnostic dict.");
    m.def("compute_dispersion_curve_pancake", &py_computeDispersionCurveLayerCake,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("periods"), py::arg("wave_type") = "Rayleigh",
          py::arg("branch") = 0, py::arg("c_min") = 0.1,
          py::arg("c_max") = 4.5, py::arg("scan_points") = 2400,
          py::arg("root_tolerance") = 1e-12,
          py::arg("verify_branch") = false,
          "Deprecated alias. Use compute_dispersion_curve_layer_cake.\n"
          "Compute a strictly monotonic LAYER_CAKE phase-velocity period curve "
          "inside one C++ call. Roots are ordered by increasing phase "
          "velocity at the initial/fallback scan. Continuation uses a local "
          "predicted bracket and falls back to the full scan. "
          "verify_branch=True confirms predicted roots with the integer mode "
          "counter at additional cost. Returns a diagnostic dict.");
    m.def("compute_dispersion_curve_romberg_grid_velocity_vti",
          &py_computeDispersionCurveRombergGridVelocityVti,
          py::arg("depths"), py::arg("rhos"), py::arg("vphs"),
          py::arg("vpvs"), py::arg("vshs"), py::arg("vsvs"),
          py::arg("etas"), py::arg("is_discontinuity"),
          py::arg("periods"), py::arg("wave_type") = "Rayleigh",
          py::arg("c_guess") = 3.0, py::arg("max_steps") = 5,
          py::arg("tolerance") = 1e-10,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.01,
          py::arg("extrapolate_start_level") = 0,
          py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") =
              std::numeric_limits<double>::quiet_NaN(),
          py::arg("branch") = py::none(),
          "Compute a strictly monotonic velocity-VTI GRID/Romberg "
          "phase-velocity period curve inside one C++ call. Refined meshes "
          "are cached across periods; per-level root prediction fails closed "
          "to the scalar adaptive scan. branch=None preserves nearest-root "
          "continuation; an integer selects the established mode-order path. "
          "Returns a diagnostic dict.");
    m.def("compute_kernels_ad_grid_velocity_iso", &py_computeDispersionRombergVelocityIsoWithKernels,
          py::arg("depths"), py::arg("rhos"), py::arg("vps"), py::arg("vss"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true, py::arg("want_kernel_hv") = true,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("u_kernel_mode") = "variational_frequency_ad", py::arg("kernel_extrapolation") = true,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Velocity-basis (rho,vp,vs) GRID dispersion + kernels. Returns "
          "(c, U, HV, level, rel_err, kernels); kernel parameter_name in {'rho','vp','vs'}. "
          "Same u_kernel_mode / kernel_extrapolation semantics as compute_kernels_ad_grid.");
    m.def("compute_dispersion_romberg_velocity_iso_with_kernels", &py_computeDispersionRombergVelocityIsoWithKernels,
          py::arg("depths"), py::arg("rhos"), py::arg("vps"), py::arg("vss"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true, py::arg("want_kernel_hv") = true,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("u_kernel_mode") = "variational_frequency_ad", py::arg("kernel_extrapolation") = true,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_kernels_ad_grid_velocity_iso.");
    m.def("compute_kernels_ad_grid_velocity_vti", &py_computeDispersionRombergVelocityVtiWithKernels,
          py::arg("depths"), py::arg("rhos"), py::arg("vphs"), py::arg("vpvs"), py::arg("vshs"), py::arg("vsvs"), py::arg("etas"),
          py::arg("is_discontinuity"), py::arg("period"), py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true, py::arg("want_kernel_hv") = true,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("u_kernel_mode") = "variational_frequency_ad", py::arg("kernel_extrapolation") = true,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Velocity-basis (rho,vph,vpv,vsh,vsv,eta) GRID dispersion + kernels. Returns "
          "(c, U, HV, level, rel_err, kernels); kernel parameter_name in {'rho','Vph','Vpv','Vsh','Vsv','eta'}. "
          "Same u_kernel_mode / kernel_extrapolation semantics as compute_kernels_ad_grid.");
    m.def("compute_dispersion_romberg_velocity_vti_with_kernels", &py_computeDispersionRombergVelocityVtiWithKernels,
          py::arg("depths"), py::arg("rhos"), py::arg("vphs"), py::arg("vpvs"), py::arg("vshs"), py::arg("vsvs"), py::arg("etas"),
          py::arg("is_discontinuity"), py::arg("period"), py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0,
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true, py::arg("want_kernel_hv") = true,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("u_kernel_mode") = "variational_frequency_ad", py::arg("kernel_extrapolation") = true,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_kernels_ad_grid_velocity_vti.");
    m.def("compute_kernels_ad_grid", &py_computeDispersionRombergWithKernels,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"), py::arg("period"),
          py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0, py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true, py::arg("want_kernel_hv") = true,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("u_kernel_mode") = "variational_frequency_ad", py::arg("kernel_extrapolation") = true,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1, py::arg("include_interface_depths") = true,
          "GRID-mode dispersion plus node sensitivity kernels (dc/dp, dU/dp, dHV/dp) via Romberg extrapolation.\n"
          "Returns (c, U, HV, level, rel_err, kernels); kernels is a list[SensitivityKernel] over nodes x parameters.\n"
          "sensitivity_mode: 'Isotropic' (Vs,Vp,Rho per node) or 'VTI' (A,C,F,L,N,Rho). "
          "compute_c/u/hv select which kernel columns are filled (others 0).\n"
          "u_kernel_mode: 'variational_frequency_ad' (default; frequency derivative of the phase "
          "variational kernel), 'variational' (energy-integral parameter AD), or 'full_ad' (nested AD). "
          "Non-finite frequency-variational results auto-fall back to the established path.\n"
          "kernel_extrapolation=True (default) Neville-extrapolates the kernels across levels so they match the "
          "reported (extrapolated) values; False keeps the single converged-level kernel.\n"
          "When phase_velocity_roots is supplied, phase_accepted_level is required and its accepted prefix "
          "must reproduce c_guess; roots from another direct branch are rejected.\n"
          "See docs/sensitivity_implementation_map.md and docs/Romberg.md.");
    m.def("compute_dispersion_romberg_with_kernels", &py_computeDispersionRombergWithKernels,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"), py::arg("period"),
          py::arg("wave_type") = "Rayleigh", py::arg("c_guess") = 3.0, py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_steps") = 5, py::arg("tolerance") = 1e-10,
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true, py::arg("want_kernel_hv") = true,
          py::arg("phase_only") = false,
          py::arg("want_value_u") = true, py::arg("want_value_hv") = true,
          py::arg("adaptive_root_scan") = true,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("u_kernel_mode") = "variational_frequency_ad", py::arg("kernel_extrapolation") = true,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1, py::arg("include_interface_depths") = true,
          "Deprecated alias. Use compute_kernels_ad_grid.");
    m.def("_compute_spherical_interface_ad_grid", [](
            const std::vector<double>& depths, const std::vector<PhysicalParameters<double>>& params,
            const std::vector<bool>& flags, double period, const std::string& wave, double phase,
            int level, const std::vector<double>& roots, int accepted, std::optional<int> row,
            const std::string& scheme, double radius, double exponent) {
        GridModel model{depths, params, flags, make_grid_refinement_policy(scheme, radius, exponent)};
        return spherical_interface_ad_grid(model, period, wave, phase, level, roots, accepted, row);
    }, py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"), py::arg("period"),
       py::arg("wave"), py::arg("phase"), py::arg("max_level"), py::arg("phase_velocity_roots"),
       py::arg("phase_accepted_level"), py::arg("interface_index"), py::arg("grid_policy"),
       py::arg("planet_radius"), py::arg("flattening_exponent"));
    m.def("compute_interface_hessian_row_ad_grid", &py_computeDepthHessianColumnGrid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"), py::arg("period"),
          py::arg("wave_type"), py::arg("c_root"), py::arg("dc_dz"),
          py::arg("discontinuity_index"), py::arg("refinement_level"),
          py::arg("sensitivity_mode") = "VTI",
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1, py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Analytic interface-depth Hessian column via nested forward AD.\n"
          "Returns (dc_dp, d2c_dpdz), both length NUM_VARS in the same variable order as\n"
          "compute_kernels_ad_grid (node params, then one entry per discontinuity).\n"
          "d2c_dpdz[j] = d(dc/dp_j)/dz_k for the interface selected by discontinuity_index.\n"
          "phase_velocity_roots plus phase_accepted_level supplies the direct Romberg trace, verifies it against "
          "c_root, and disables rerooting.\n"
          "With a trace, `c_root` is the reported direct phase used for identity validation; without one it is "
          "polished on the level-`refinement_level` mesh. `dc_dz` must be that level's first-order interface "
          "sensitivity: the outer AD level seeds c with dc_dz so the\n"
          "chain rule advects the root (same device as the weak-form Hessian's cp_seed).\n"
          "The returned dc_dp reproduces the established first-order kernel and is intended as a\n"
          "free consistency check on the caller's dc_dz.\n"
          "See docs/lowlevel_api.md, Interface derivatives.");
    m.def("compute_depth_hessian_column_grid", &py_computeDepthHessianColumnGrid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"), py::arg("period"),
          py::arg("wave_type"), py::arg("c_root"), py::arg("dc_dz"),
          py::arg("discontinuity_index"), py::arg("refinement_level"),
          py::arg("sensitivity_mode") = "VTI",
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1, py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Deprecated alias. Use compute_interface_hessian_row_ad_grid.");
    m.def("compute_interface_kernels_variational_rayleigh_layer_cake", &py_computeBoundaryKernelsRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("use_analytical") = false,
          "Variational boundary kernel [K_d]_b for every stack boundary.\n"
          "Returns N values for N layers: entry b-1 is the jump at the boundary between\n"
          "layer b-1 and layer b (the last one being layer N-1 against the halfspace).\n"
          "These are per-boundary quantities; a geometric derivative is\n"
          "  dc/dq = sum_b [K_d]_b * dz_b/dq,\n"
          "and the weights dz_b/dq are what distinguish an interface-depth derivative\n"
          "from a layer-thickness one, so they are left to the caller.\n"
          "See docs/lowlevel_api.md, Interface derivatives.");
    m.def("compute_interface_kernels_variational_rayleigh_pancake", &py_computeBoundaryKernelsRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("use_analytical") = false,
          "Deprecated alias. Use compute_interface_kernels_variational_rayleigh_layer_cake.\n"
          "Variational boundary kernel [K_d]_b for every stack boundary.\n"
          "Returns N values for N layers: entry b-1 is the jump at the boundary between\n"
          "layer b-1 and layer b (the last one being layer N-1 against the halfspace).\n"
          "These are per-boundary quantities; a geometric derivative is\n"
          "  dc/dq = sum_b [K_d]_b * dz_b/dq,\n"
          "and the weights dz_b/dq are what distinguish an interface-depth derivative\n"
          "from a layer-thickness one, so they are left to the caller.\n"
          "See docs/lowlevel_api.md, Interface derivatives.");
    m.def("compute_boundary_kernels_rayleigh", &py_computeBoundaryKernelsRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("use_analytical") = false,
          "Deprecated alias. Use compute_interface_kernels_variational_rayleigh_layer_cake.");
    m.def("compute_interface_hessian_row_variational_ad_grid",
          &py_computeInterfaceHessianRowVariational,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type"), py::arg("c_root"),
          py::arg("dc_dp"), py::arg("boundary_weights"),
          py::arg("refinement_level"), py::arg("sensitivity_mode") = "VTI",
          "Interface Hessian row d(dc/dz_j)/dp, by forward-AD of the variational\n"
          "boundary kernel. Since dc/dz_j = sum_b [K_d]_b w_b with purely geometric\n"
          "weights, one AD pass over the kernel suffices -- no nesting.\n"
          "`dc_dp` is the caller's first-order kernel row and seeds c's derivative so\n"
          "the chain rule carries the implicit motion of the root; omitting it would\n"
          "give a partial derivative at fixed c, not a Hessian row.\n"
          "`boundary_weights` selects the parameterization (interface depth vs layer\n"
          "thickness) and must have one entry per stack boundary.\n"
          "Entry NUM_PARAM_VARS + j of the result is the diagonal d^2c/dz_j^2.");
    m.def("compute_interface_hessian_row_variational",
          &py_computeInterfaceHessianRowVariational,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type"), py::arg("c_root"),
          py::arg("dc_dp"), py::arg("boundary_weights"),
          py::arg("refinement_level"), py::arg("sensitivity_mode") = "VTI",
          "Deprecated alias. Use compute_interface_hessian_row_variational_ad_grid.");
    m.def("compute_interface_gradient_variational_grid",
          &py_computeInterfaceGradientVariationalExtrapolated,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type"), py::arg("c_root"),
          py::arg("discontinuity_index"), py::arg("max_level"),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "First-order interface-depth gradient dc/dz_j on a GRID mesh, "
          "Neville-extrapolated.\n"
          "The variational counterpart of the Depth entry that "
          "compute_kernels_ad_grid\n"
          "returns, and the entry point a weak-form Jacobian needs so that its Depth\n"
          "component does not have to come from the AD kernel pass (733 ms against\n"
          "about 30 ms for the weak-form gradient).\n"
          "Unlike the Hessian row below this needs no AD: dc/dz_j = sum_b [K_d]_b w_b\n"
          "with purely geometric weights, so the K_d values suffice.\n"
          "See docs/sensitivity_implementation_map.md.");
    m.def("compute_interface_gradient_variational_extrapolated",
          &py_computeInterfaceGradientVariationalExtrapolated,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type"), py::arg("c_root"),
          py::arg("discontinuity_index"), py::arg("max_level"),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_interface_gradient_variational_grid.");
    m.def("compute_interface_hessian_row_variational_ad_grid_extrapolated",
          &py_computeInterfaceHessianRowVariationalExtrapolated,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type"), py::arg("c_root"),
          py::arg("discontinuity_index"), py::arg("max_level"),
          py::arg("sensitivity_mode") = "VTI",
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Romberg-extrapolated interface Hessian row d(dc/dz_j)/dp via K_d.\n"
          "Evaluates every level 0..max_level and Neville-extrapolates, the same\n"
          "treatment compute_kernels_ad_grid gives the first-order\n"
          "kernels. The single-level row carries an O(h^2) truncation (measured to\n"
          "fall exactly 4x per level); extrapolation removes it.\n"
          "Unlike compute_interface_hessian_row_variational_ad_grid this needs no dc_dp and\n"
          "no boundary_weights: each level's first-order kernel comes from\n"
          "compute_grid_level_kernels at that mesh's own root, and the DEPTH\n"
          "parameterization weights are built internally from the mesh.\n"
          "Entry NUM_PARAM_VARS + j of the result is the diagonal d^2c/dz_j^2.");
    m.def("compute_interface_hessian_row_variational_extrapolated",
          &py_computeInterfaceHessianRowVariationalExtrapolated,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("period"), py::arg("wave_type"), py::arg("c_root"),
          py::arg("discontinuity_index"), py::arg("max_level"),
          py::arg("sensitivity_mode") = "VTI",
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_interface_hessian_row_variational_ad_grid_extrapolated.");
    m.def("compute_interface_kernels_variational_love_layer_cake", &py_computeBoundaryKernelsLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          "Love variational boundary kernel [K_d]_b, one jump per stack boundary.\n"
          "Same contract as compute_interface_kernels_variational_rayleigh_layer_cake: weights are the caller's.\n"
          "K_d = -Lambda (rho w^2 y1^2 + y2^2/L - k^2 N y1^2) with y1=u_y, y2=tau_yz.\n"
          "The PLUS on y2^2/L is the spatial-Hamiltonian form; the Lagrangian density\n"
          "T - V has a minus there and near-cancels, suppressing Love by ~50x.");
    m.def("compute_interface_kernels_variational_love_pancake", &py_computeBoundaryKernelsLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          "Deprecated alias. Use compute_interface_kernels_variational_love_layer_cake.\n"
          "Love variational boundary kernel [K_d]_b, one jump per stack boundary.\n"
          "Same contract as compute_interface_kernels_variational_rayleigh_layer_cake: weights are the caller's.\n"
          "K_d = -Lambda (rho w^2 y1^2 + y2^2/L - k^2 N y1^2) with y1=u_y, y2=tau_yz.\n"
          "The PLUS on y2^2/L is the spatial-Hamiltonian form; the Lagrangian density\n"
          "T - V has a minus there and near-cancels, suppressing Love by ~50x.");
    m.def("compute_boundary_kernels_love", &py_computeBoundaryKernelsLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          "Deprecated alias. Use compute_interface_kernels_variational_love_layer_cake.");
    m.def("compute_eigenfunctions_rayleigh", &py_computeRayleighEigenfunctions);
    m.def("compute_rayleigh_eigenfunctions", &py_computeRayleighEigenfunctions,
          "Deprecated alias. Use compute_eigenfunctions_rayleigh.");
    m.def("compute_eigenfunctions_love", &py_computeLoveEigenfunctions);
    m.def("compute_love_eigenfunctions", &py_computeLoveEigenfunctions,
          "Deprecated alias. Use compute_eigenfunctions_love.");
    m.def("compute_energy_integrals", &py_computeEnergyIntegrals);
    m.def("compute_energy_integrals_analytical_rayleigh", &py_computeRayleighEnergyIntegrals_Analytical);
    m.def("compute_rayleigh_energy_integrals_analytical", &py_computeRayleighEnergyIntegrals_Analytical,
          "Deprecated alias. Use compute_energy_integrals_analytical_rayleigh.");
    m.def("compute_energy_integrals_analytical_love", &py_computeLoveEnergyIntegrals_Analytical);
    m.def("compute_love_energy_integrals_analytical", &py_computeLoveEnergyIntegrals_Analytical,
          "Deprecated alias. Use compute_energy_integrals_analytical_love.");
    m.def("debug_kernels_variational_rayleigh_layer_cake", &py_debugRayleighWeakformTrace,
          py::arg("layers"), py::arg("thicknesses"), py::arg("halfspace"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false);
    m.def("debug_kernels_variational_rayleigh_pancake", &py_debugRayleighWeakformTrace,
          py::arg("layers"), py::arg("thicknesses"), py::arg("halfspace"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false, "Deprecated alias. Use debug_kernels_variational_rayleigh_layer_cake.\n");
    m.def("debug_rayleigh_weakform_trace", &py_debugRayleighWeakformTrace,
          py::arg("layers"), py::arg("thicknesses"), py::arg("halfspace"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false,
          "Deprecated alias. Use debug_kernels_variational_rayleigh_layer_cake.");
    m.def("integrate_grid_layer_quadrature", &py_integrateGridLayer_Romberg,
          py::arg("y_top"), py::arg("p_top"), py::arg("p_bot"), py::arg("thickness"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type"),
          py::arg("max_depth") = 5, py::arg("tolerance") = 1e-8, py::arg("use_romberg") = true);
    m.def("integrate_grid_layer_romberg", &py_integrateGridLayer_Romberg,
          py::arg("y_top"), py::arg("p_top"), py::arg("p_bot"), py::arg("thickness"),
          py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type"),
          py::arg("max_depth") = 5, py::arg("tolerance") = 1e-8, py::arg("use_romberg") = true,
          "Deprecated alias. Use integrate_grid_layer_quadrature.");
    m.def("compute_layer_matrix_p_sv", &py_compute_layer_matrix_p_sv);
    m.def("compute_layer_matrix_sh", &py_compute_layer_matrix_sh);
    m.def("propagate_layer_quadrature_vector", &py_propagate_romberg_vector);
    m.def("propagate_romberg_vector", &py_propagate_romberg_vector,
          "Deprecated alias. Use propagate_layer_quadrature_vector.");
    m.def("eval_layer_state", &py_compute_state_at_depth);
    m.def("eval_global_state", &py_evaluate_global_state);

    // Hessian API
    m.def("compute_hessian_ad_layer_cake", &py_computePhaseVelocityHessianFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "LAYER_CAKE-mode full-AD Hessian d2c/dp2 via nested second-order AD (HessianAD<N>, O(L*N^2)). "
          "Fixed mesh (no Romberg). For the lighter weak-form version see "
          "compute_hessian_variational_ad_rayleigh_layer_cake/_love; for GRID see the *_grid variants. "
          "See docs/sensitivity_implementation_map.md.");
    m.def("compute_hessian_ad_pancake", &py_computePhaseVelocityHessianFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_hessian_ad_layer_cake.\n"
          "LAYER_CAKE-mode full-AD Hessian d2c/dp2 via nested second-order AD (HessianAD<N>, O(L*N^2)). "
          "Fixed mesh (no Romberg). For the lighter weak-form version see "
          "compute_hessian_variational_ad_rayleigh_layer_cake/_love; for GRID see the *_grid variants. "
          "See docs/sensitivity_implementation_map.md.");
    m.def("compute_hessian_ad_grid", &py_computePhaseVelocityHessianGridFullAD,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("wave_type") = "Rayleigh",
          py::arg("sensitivity_mode") = "VTI",
          py::arg("refinement_level") = 0,
          py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Small fixed-level GRID full-AD Hessian for flat-policy validation. "
          "Returns node-VTI gradient/Hessian; production GRID calculations use "
          "the variational Hessian entry points.");
    m.def("compute_phase_velocity_hessian", &py_computePhaseVelocityHessian,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_hessian_ad_layer_cake.");
    m.def("compute_phase_velocity_hessian_full_ad", &py_computePhaseVelocityHessianFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_hessian_ad_layer_cake.");
    m.def("compute_phase_velocity_hessian_reference", &py_computePhaseVelocityHessianRef,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_hessian_ad_layer_cake.");
    m.def("compute_hessian_variational_fd_love_layer_cake", &py_computePhaseVelocityHessianVariationalFDLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 200, py::arg("fd_eps") = 1e-5);
    m.def("compute_hessian_variational_fd_love_pancake", &py_computePhaseVelocityHessianVariationalFDLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 200, py::arg("fd_eps") = 1e-5, "Deprecated alias. Use compute_hessian_variational_fd_love_layer_cake.\n");
    m.def("compute_phase_velocity_hessian_variational_fd_love", &py_computePhaseVelocityHessianVariationalFDLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 200, py::arg("fd_eps") = 1e-5,
          "Deprecated alias. Use compute_hessian_variational_fd_love_layer_cake.");
    m.def("compute_hessian_variational_ad_love_layer_cake", &py_computePhaseVelocityHessianWeakFormLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("use_analytical") = true, py::arg("branch_policy") = "auto",
          py::arg("backend") = "dual");
    m.def("compute_hessian_variational_ad_love_pancake", &py_computePhaseVelocityHessianWeakFormLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("use_analytical") = true, py::arg("branch_policy") = "auto",
          py::arg("backend") = "dual", "Deprecated alias. Use compute_hessian_variational_ad_love_layer_cake.\n");
    m.def("compute_phase_velocity_hessian_weakform_love", &py_computePhaseVelocityHessianWeakFormLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("use_analytical") = true, py::arg("branch_policy") = "auto",
          py::arg("backend") = "dual",
          "Deprecated alias. Use compute_hessian_variational_ad_love_layer_cake.");
    m.def("debug_hessian_variational_ad_love_layer_cake", &py_debugLoveWeakformHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{},
          py::arg("cp_grad_multiplier") = 1.0);
    m.def("debug_hessian_variational_ad_love_pancake", &py_debugLoveWeakformHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{},
          py::arg("cp_grad_multiplier") = 1.0, "Deprecated alias. Use debug_hessian_variational_ad_love_layer_cake.\n");
    m.def("debug_love_weakform_hessian_breakdown", &py_debugLoveWeakformHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{},
          py::arg("cp_grad_multiplier") = 1.0,
          "Deprecated alias. Use debug_hessian_variational_ad_love_layer_cake.");
    m.def("debug_gradient_variational_love_layer_cake", &py_debugLoveWeakformGradientPieces,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{});
    m.def("debug_gradient_variational_love_pancake", &py_debugLoveWeakformGradientPieces,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{}, "Deprecated alias. Use debug_gradient_variational_love_layer_cake.\n");
    m.def("debug_love_weakform_gradient_pieces", &py_debugLoveWeakformGradientPieces,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{},
          "Deprecated alias. Use debug_gradient_variational_love_layer_cake.");
    m.def("debug_layer_integrals_love_layer_cake", &py_debugLoveLayerIntegralProbe,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto");
    m.def("debug_layer_integrals_love_pancake", &py_debugLoveLayerIntegralProbe,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto", "Deprecated alias. Use debug_layer_integrals_love_layer_cake.\n");
    m.def("debug_love_layer_integral_probe", &py_debugLoveLayerIntegralProbe,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto",
          "Deprecated alias. Use debug_layer_integrals_love_layer_cake.");
    m.def("debug_layer_integrals_hessian_love_layer_cake", &py_debugLoveLayerIntegralHessian,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto");
    m.def("debug_layer_integrals_hessian_love_pancake", &py_debugLoveLayerIntegralHessian,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto", "Deprecated alias. Use debug_layer_integrals_hessian_love_layer_cake.\n");
    m.def("debug_love_layer_integral_hessian", &py_debugLoveLayerIntegralHessian,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto",
          "Deprecated alias. Use debug_layer_integrals_hessian_love_layer_cake.");
    m.def("debug_hessian_ad_love_upward_ift_layer_cake", &py_debugLoveUpwardIFTHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic");
    m.def("debug_hessian_ad_love_upward_ift_pancake", &py_debugLoveUpwardIFTHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", "Deprecated alias. Use debug_hessian_ad_love_upward_ift_layer_cake.\n");
    m.def("debug_love_upward_ift_hessian_breakdown", &py_debugLoveUpwardIFTHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic",
          "Deprecated alias. Use debug_hessian_ad_love_upward_ift_layer_cake.");
    m.def("compute_hessian_variational_ad_rayleigh_layer_cake", &py_computePhaseVelocityHessianWeakFormRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false, py::arg("num_sub_layers") = 10,
          py::arg("use_analytical") = false, py::arg("backend") = "dual");
    m.def("compute_hessian_variational_ad_rayleigh_pancake", &py_computePhaseVelocityHessianWeakFormRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false, py::arg("num_sub_layers") = 10,
          py::arg("use_analytical") = false, py::arg("backend") = "dual", "Deprecated alias. Use compute_hessian_variational_ad_rayleigh_layer_cake.\n");
    m.def("compute_phase_velocity_hessian_weakform_rayleigh", &py_computePhaseVelocityHessianWeakFormRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false, py::arg("num_sub_layers") = 10,
          py::arg("use_analytical") = false, py::arg("backend") = "dual",
          "Deprecated alias. Use compute_hessian_variational_ad_rayleigh_layer_cake.");

    m.def("compute_hessian_variational_ad_rayleigh_grid",
          &py_computePhaseVelocityHessianWeakFormRayleigh_Grid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "GRID-mode Rayleigh phase-velocity gradient dc/dp and Hessian d2c/dp2 via the weak-form "
          "(variational) method, Neville-extrapolated across Romberg levels. "
          "For fixed-level flat VTI validation see compute_hessian_ad_grid. freq is in Hz.\n"
          "Returns a dict with keys 'gradient', 'hessian', 'parameter_names', 'layer_indices'. "
          "See docs/sensitivity_implementation_map.md and docs/grid_wf_hessian.md.");
    m.def("compute_phase_velocity_hessian_weakform_rayleigh_grid",
          &py_computePhaseVelocityHessianWeakFormRayleigh_Grid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_hessian_variational_ad_rayleigh_grid.");

    m.def("compute_hessian_variational_ad_love_grid",
          &py_computePhaseVelocityHessianWeakFormLove_Grid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "GRID-mode Love phase-velocity gradient and Hessian via the weak-form (variational) method, "
          "Neville-extrapolated across Romberg levels. freq [Hz]. Returns a dict with keys 'gradient', "
          "'hessian', 'parameter_names', 'layer_indices'. See docs/sensitivity_implementation_map.md.");
    m.def("compute_phase_velocity_hessian_weakform_love_grid",
          &py_computePhaseVelocityHessianWeakFormLove_Grid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_hessian_variational_ad_love_grid.");

    m.def("compute_hessian_variational_ad_rayleigh_grid_reduced",
          &py_computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"), py::arg("J"),
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "GRID-mode Rayleigh weak-form Hessian projected directly into a caller-supplied reduced "
          "basis: J has shape (n_nodes*6, P), node-VTI rows ordered [node*6 + modulus] with modulus "
          "order [rho,c11,c33,c13,c44,c66] (matches inversion/disper26_inv/derivatives.py STIFF_ORDER "
          "after permutation), P columns are the reduced basis directions. AD seeds J's columns "
          "directly instead of node-VTI unit vectors, so the full (n_nodes*6)^2 Hessian is never "
          "materialized. Returns a dict with keys 'hessian_reduced' (P x P) and 'gradient_node' "
          "(n_nodes*6, unprojected, needed by the caller for any nonlinear-basis curvature "
          "correction). See docs/sensitivity_implementation_map.md.");
    m.def("compute_phase_velocity_hessian_weakform_rayleigh_grid_reduced",
          &py_computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"), py::arg("J"),
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_hessian_variational_ad_rayleigh_grid_reduced.");

    m.def("compute_hessian_variational_ad_love_grid_reduced",
          &py_computePhaseVelocityHessianWeakFormLove_Grid_Reduced,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"), py::arg("J"),
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Love counterpart of compute_hessian_variational_ad_rayleigh_grid_reduced.");
    m.def("compute_phase_velocity_hessian_weakform_love_grid_reduced",
          &py_computePhaseVelocityHessianWeakFormLove_Grid_Reduced,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"), py::arg("J"),
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_hessian_variational_ad_love_grid_reduced.");

    m.def("compute_kernels_variational_grid", &py_computeKernelsVariationalGrid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("wave_type") = "Rayleigh",
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("backend") = "dual",
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "GRID x variational x 1st order, standalone (docs/naming_convention.md SS7.3): "
          "the node-VTI gradient dc/d(node modulus), Neville-extrapolated across Romberg "
          "levels, without building any Hessian. Node-VTI order matches "
          "compute_hessian_variational_ad_rayleigh_grid_reduced's gradient_node "
          "(rows [node*6 + modulus], modulus order [rho,c11,c33,c13,c44,c66]). "
          "Returns a dict with key 'gradient_node' (n_nodes*6,). "
          "wave_type: 'Rayleigh' or 'Love' (Love-only initial_root_window_frac ignored for Rayleigh). "
          "Uses a dedicated first-order convergence loop; no AD Hessian block is built. "
          "phase_velocity_roots plus phase_accepted_level reuses the branch-confirmed direct roots, "
          "verifies the accepted prefix against c_guess, and disables "
          "independent per-level root searches.");

    m.def("debug_refined_grid_layers",
          &py_debugGetRefinedGridLayers,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("level"), py::arg("grid_policy") = "flat",
          py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN());
    m.def("debug_get_refined_grid_layers",
          &py_debugGetRefinedGridLayers,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("level"),
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Deprecated alias. Use debug_refined_grid_layers.");

    m.def("debug_gradient_variational_love_grid",
          &py_debugGridLoveLevelGradientPieces,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"), py::arg("level"),
          py::arg("use_analytical") = false,
          py::arg("num_sub_layers") = 1,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN());
    m.def("debug_grid_love_level_gradient_pieces",
          &py_debugGridLoveLevelGradientPieces,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"), py::arg("level"),
          py::arg("use_analytical") = false,
          py::arg("num_sub_layers") = 1,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Deprecated alias. Use debug_gradient_variational_love_grid.");

    m.def("debug_hessian_variational_ad_love_grid",
          &py_debugPhaseVelocityHessianWeakFormLove_Grid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("use_analytical") = true,
          py::arg("num_sub_layers") = 1,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN());
    m.def("debug_phase_velocity_hessian_weakform_love_grid",
          &py_debugPhaseVelocityHessianWeakFormLove_Grid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"),
          py::arg("freq"), py::arg("c_guess"),
          py::arg("sensitivity_mode") = "Isotropic",
          py::arg("max_levels") = 5,
          py::arg("tolerance") = 1e-6,
          py::arg("use_analytical") = true,
          py::arg("num_sub_layers") = 1,
          py::arg("initial_root_window_frac") = 0.0,
          py::arg("grid_policy") = "flat", py::arg("planet_radius") = 6371.0,
          py::arg("flattening_exponent") = std::numeric_limits<double>::quiet_NaN(),
          "Deprecated alias. Use debug_hessian_variational_ad_love_grid.");
}
