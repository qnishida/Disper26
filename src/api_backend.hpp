#pragma once

// Internal Python-independent adapters shared by the legacy binding and the
// public C++ API. The historical py_ names remain internal so extraction does
// not change the legacy numerical paths or their argument/default contracts.
// Like dispersion_interface.hpp, include this only in an implementation unit:
// the existing numerical core is still compiled by including disper26.cpp.
#include "dispersion_interface.hpp"
#include "disper26_analytical.hpp"
#include "sensitivity_utils.hpp"
#include "grid_hessian.hpp"
#include "dispersion_context.hpp"
#include <optional>
#include <sstream>
#include <utility>

using namespace Analytical;

// VTI Layer Creation Helper
// Returns (PhysicalParameters, thickness)
inline std::tuple<PhysicalParameters<double>, double> py_createVTILayerFromDepth(
    double A, double C, double F, double L, double N, double Rho,
    double top_depth, double bottom_depth
) {
    PhysicalParameters<double> p;
    p.c11 = A;
    p.c33 = C;
    p.c13 = F;
    p.c44 = L;
    p.c66 = N;
    p.rho = Rho;
    p.gravity = 0.0;
    p.gravity_internal_scale = 1.0;
    p.gravity_surface_scale = 0.0;

    double thickness = std::abs(bottom_depth - top_depth);
    return std::make_tuple(p, thickness);
}

inline double py_mineosPhysicalDispersionFactor(double tref_s, double period_s) {
    if (tref_s <= 0.0) return 0.0;
    if (period_s <= 0.0) {
        throw std::runtime_error("period_s must be positive.");
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    return 2.0 * std::log(tref_s / period_s) / pi;
}

inline double py_qInverseMineos(double q_value) {
    return (q_value > 0.0) ? 1.0 / q_value : q_value;
}

inline PhysicalParameters<double> py_applyMineosPhysicalDispersion(
    const PhysicalParameters<double>& params,
    double qkappa,
    double qmu,
    double tref_s,
    double period_s
) {
    const double fct = py_mineosPhysicalDispersionFactor(tref_s, period_s);
    if (fct == 0.0) return params;

    const double A = params.c11;
    const double C = params.c33;
    const double F = params.c13;
    const double L = params.c44;
    const double N = params.c66;

    const double fmu = (A + C - 2.0 * F + 5.0 * N + 6.0 * L) / 15.0;
    const double flam = (4.0 * (A + F - N) + C) / 9.0 - 2.0 * fmu / 3.0;
    const double denom = flam + 2.0 * fmu;
    if (std::abs(denom) < 1.0e-30) return params;

    const double qshear = py_qInverseMineos(qmu);
    const double qbulk = py_qInverseMineos(qkappa);
    const double rat = 4.0 * fmu / (3.0 * denom);
    const double xlam_denom = 1.0 - 1.5 * rat;
    const double xlam = (std::abs(xlam_denom) < 1.0e-30)
        ? 0.0
        : ((1.0 - rat) * qbulk - 0.5 * rat * qshear) / xlam_denom;
    const double xa2 = (1.0 - rat) * qbulk + rat * qshear;

    PhysicalParameters<double> corrected = params;
    corrected.c11 = A * (1.0 + xa2 * fct);
    corrected.c33 = C * (1.0 + xa2 * fct);
    corrected.c13 = F * (1.0 + xlam * fct);
    corrected.c44 = L * (1.0 + qshear * fct);
    corrected.c66 = N * (1.0 + qshear * fct);
    return corrected;
}

inline std::vector<PhysicalParameters<double>> py_applyMineosPhysicalDispersionModel(
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<double>& qkappas,
    const std::vector<double>& qmus,
    double tref_s,
    double period_s
) {
    if (params.size() != qkappas.size() || params.size() != qmus.size()) {
        throw std::runtime_error("params, qkappas, and qmus must have the same size.");
    }
    std::vector<PhysicalParameters<double>> corrected;
    corrected.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        corrected.push_back(py_applyMineosPhysicalDispersion(params[i], qkappas[i], qmus[i], tref_s, period_s));
    }
    return corrected;
}

inline WaveType parseWaveType(const std::string& wave_type_str) {
    if (wave_type_str == "Love" || wave_type_str == "love") {
        return WaveType::Love;
    }
    return WaveType::Rayleigh;
}

// Primary integer mode count: AD-z bran-style crossing counter.
inline int py_calculateTotalModeCount(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    return calculateAdModeCount(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, parseWaveType(wave_type_str));
}

// Wrapper for Continuous Mode Count
inline double py_calculateContinuousModeCount(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    WaveType type = parseWaveType(wave_type_str);
    return calculateContinuousModeCount(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type);
}


inline int py_calculateBranModeCount(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    if (wave_type_str == "Love" || wave_type_str == "love") {
        return calculateAdModeCount(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, WaveType::Love);
    }
    return calculateAdModeCount(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, WaveType::Rayleigh);
}


// Wrapper for Dispersion Function Value (Returning characteristic function)
// For Rayleigh: Returns Y[4] (Stress minor)
// For Love: returns the historical secular residual -T_surface, with
// T=c44*du_y/dz and z positive upward (not the public eigenfunction tyz sign).
inline double py_calculateDispersionValue(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") {
        type = WaveType::Love;
    }

    return calculateDispersionPhaseValueDouble(
        layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type);
}

// Wrapper for Root Searching
inline std::vector<double> py_searchRoots(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double c_min,
    double c_max,
    int num_points, // Used to define step
    std::string wave_type_str
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    double step = (c_max - c_min) / std::max(1, num_points);

    // Define the function to zero
    auto func = [&](double c) {
        return calculateDispersionPhaseValueDouble(
            layers, thicknesses, halfSpaceParams, omega, c, type);
    };

    return searchRoots(func, c_min, c_max, step);
}

// Wrapper for debugging: Return full 5-component vector
inline std::vector<double> py_getRayleighMinors(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    auto res = calculateDispersionFunction(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type);

    std::vector<double> vec(5);
    for(int i=0; i<5; ++i) {
        vec[i] = res(i);
    }
    return vec;
}

// Wrapper for Finite Difference Sensitivity Kernels
inline std::vector<SensitivityKernel> py_computeSensitivityFD(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str,
    std::string mode_str,
    bool compute_thickness
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    SensitivityMode mode = SensitivityMode::Isotropic;
    if (mode_str == "VTI" || mode_str == "vti") mode = SensitivityMode::VTI;

    // Convert freq from omega
    double freq = omega / (2.0 * M_PI);

    return computeSensitivityKernelsFD(layers, thicknesses, halfSpaceParams, freq, phaseVelocity, type, mode, compute_thickness);
}
// Standalone Group Velocity Calculator
// Uses simplified AutoDiff for (c, omega) only
inline double py_calculateGroupVelocity(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;
    return calculateGroupVelocityFullAD(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type);
}

// Standalone H/V Ratio Calculator
// Returns signed y3/y1 at the solid surface or solid side of the seabed
// For Rayleigh waves only. Returns NaN for Love.
inline double py_calculateHVRatio(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") return std::numeric_limits<double>::quiet_NaN();

    // Capture seabed state if needed
    Eigen::Matrix<double, 5, 1> seabedState;
    seabedState.setZero();
    auto res = calculateDispersionFunction(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type, NormalizationMode::Vertical, &seabedState);

    bool topIsLiquid = false;
    if (!layers.empty() && isLiquidLayer(layers[0])) topIsLiquid = true;

    if (topIsLiquid) {
        return calculateHVRatio(seabedState);
    } else {
        return calculateHVRatio(res);
    }
}

// Full AD Sensitivity Kernels Wrapper
inline std::vector<SensitivityKernel> py_computeSensitivityFullAD(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str,
    std::string mode_str,
    bool compute_thickness
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    SensitivityMode mode = SensitivityMode::Isotropic;
    if (mode_str == "VTI" || mode_str == "vti") mode = SensitivityMode::VTI;

    double freq = omega / (2.0 * M_PI);

    return computeSensitivityKernelsFullAD(layers, thicknesses, halfSpaceParams, freq, phaseVelocity, type, mode, compute_thickness);
}

// Variational (weak-form) Sensitivity Kernels Wrapper.
// Computes dc/dp via the variational (frozen-displacement) gradient and
// dU/dp via one first-order AD pass over U = I3/(c*I1) with IFT-seeded c.
inline std::vector<SensitivityKernel> py_computeSensitivityWeakForm(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str,
    std::string mode_str,
    bool compute_thickness,
    int num_sub_layers,
    bool use_analytical,
    std::string u_kernel_mode,
    bool compute_c,
    bool compute_u,
    bool compute_hv
) {
    SensitivityMode mode = SensitivityMode::Isotropic;
    if (mode_str == "VTI" || mode_str == "vti") mode = SensitivityMode::VTI;

    double freq = omega / (2.0 * M_PI);

    if (wave_type_str == "Love" || wave_type_str == "love") {
        return computeSensitivityKernelsWeakFormLove(
            layers, thicknesses, halfSpaceParams, freq, phaseVelocity, mode,
            compute_thickness, u_kernel_mode, compute_c, compute_u, compute_hv);
    }
    return computeSensitivityKernelsWeakFormRayleigh(
        layers, thicknesses, halfSpaceParams, freq, phaseVelocity, mode, compute_thickness,
        num_sub_layers, use_analytical, u_kernel_mode,
        compute_c, compute_u, compute_hv);
}

// Full AD Group Velocity Wrapper
inline double py_calculateGroupVelocityFullAD(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    std::string wave_type_str
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    return calculateGroupVelocityFullAD(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type);
}


inline PhysicalParameters<double> applyEarthFlatteningScaleBA97(
    const PhysicalParameters<double>& p_s,
    double radius,
    double planet_radius,
    double flattening_exponent
) {
    double radius_ratio = radius / planet_radius;
    bool is_fluid = std::abs(p_s.c44) <= 1.0e-12 && std::abs(p_s.c66) <= 1.0e-12;
    double scale_elastic = is_fluid
        ? std::pow(radius_ratio, -2.0)
        : std::pow(radius_ratio, -flattening_exponent);
    double scale_rho = is_fluid
        ? 1.0
        : std::pow(radius_ratio, 2.0 - flattening_exponent);

    PhysicalParameters<double> p_f;
    p_f.rho = p_s.rho * scale_rho;
    p_f.c11 = p_s.c11 * scale_elastic;
    p_f.c33 = p_s.c33 * scale_elastic;
    p_f.c13 = p_s.c13 * scale_elastic;
    p_f.c44 = p_s.c44 * scale_elastic;
    p_f.c66 = p_s.c66 * scale_elastic;
    p_f.gravity = p_s.gravity;
    p_f.gravity_internal_scale = p_s.gravity_internal_scale;
    p_f.gravity_surface_scale = p_s.gravity_surface_scale;
    return p_f;
}

inline PhysicalParameters<double> applyEarthFlatteningScaleExperimentalDensity(
    const PhysicalParameters<double>& p_s,
    double radius,
    double planet_radius,
    double density_exponent
) {
    double scale_vel = planet_radius / radius;
    double scale_rho = std::pow(radius / planet_radius, density_exponent);
    double scale_elastic = scale_rho * scale_vel * scale_vel;

    PhysicalParameters<double> p_f;
    p_f.rho = p_s.rho * scale_rho;
    p_f.c11 = p_s.c11 * scale_elastic;
    p_f.c33 = p_s.c33 * scale_elastic;
    p_f.c13 = p_s.c13 * scale_elastic;
    p_f.c44 = p_s.c44 * scale_elastic;
    p_f.c66 = p_s.c66 * scale_elastic;
    p_f.gravity = p_s.gravity;
    p_f.gravity_internal_scale = p_s.gravity_internal_scale;
    p_f.gravity_surface_scale = p_s.gravity_surface_scale;
    return p_f;
}

// Earth Flattening Transformation for layer/layer_cake models (Bhattacharya & Arora 1997).
inline std::tuple<std::vector<PhysicalParameters<double>>, std::vector<double>> py_earthFlatteningLayers(
    const std::vector<PhysicalParameters<double>>& layers_spherical,
    const std::vector<double>& spherical_thicknesses,
    double planet_radius,
    double flattening_exponent
) {
    if (layers_spherical.size() != spherical_thicknesses.size()) {
        throw std::runtime_error("Size mismatch: layers and thicknesses must explain the same number of layers.");
    }

    std::vector<PhysicalParameters<double>> l_flat;
    std::vector<double> h_flat;

    double current_depth = 0.0;

    for(size_t i=0; i<layers_spherical.size(); ++i) {
        double d_top = current_depth;
        double d_bot = current_depth + spherical_thicknesses[i];
        current_depth = d_bot;

        double r_top = planet_radius - d_top;
        double r_bot = planet_radius - d_bot;

        if (r_bot <= 0) throw std::runtime_error("Layer extends to center of the Earth or beyond (r<=0).");

        double h_f = planet_radius * std::log(r_top / r_bot);
        h_flat.push_back(h_f);

        double r_mid = (r_top + r_bot) / 2.0;
        l_flat.push_back(applyEarthFlatteningScaleBA97(layers_spherical[i], r_mid, planet_radius, flattening_exponent));
    }

    return std::make_tuple(l_flat, h_flat);
}

// Experimental density-exponent EFT retained for old Mineos diagnostics.
inline std::tuple<std::vector<PhysicalParameters<double>>, std::vector<double>> py_earthFlatteningLayersExperimentalDensity(
    const std::vector<PhysicalParameters<double>>& layers_spherical,
    const std::vector<double>& spherical_thicknesses,
    double planet_radius,
    double density_exponent
) {
    if (layers_spherical.size() != spherical_thicknesses.size()) {
        throw std::runtime_error("Size mismatch: layers and thicknesses must explain the same number of layers.");
    }

    std::vector<PhysicalParameters<double>> l_flat;
    std::vector<double> h_flat;

    double current_depth = 0.0;

    for(size_t i=0; i<layers_spherical.size(); ++i) {
        double d_top = current_depth;
        double d_bot = current_depth + spherical_thicknesses[i];
        current_depth = d_bot;

        double r_top = planet_radius - d_top;
        double r_bot = planet_radius - d_bot;

        if (r_bot <= 0) throw std::runtime_error("Layer extends to center of the Earth or beyond (r<=0).");

        double h_f = planet_radius * std::log(r_top / r_bot);
        h_flat.push_back(h_f);

        double r_mid = (r_top + r_bot) / 2.0;
        l_flat.push_back(applyEarthFlatteningScaleExperimentalDensity(layers_spherical[i], r_mid, planet_radius, density_exponent));
    }

    return std::make_tuple(l_flat, h_flat);
}

// Earth Flattening Transformation for node/grid Romberg models (Bhattacharya & Arora 1997).
inline std::tuple<std::vector<double>, std::vector<PhysicalParameters<double>>, std::vector<bool>> py_earthFlatteningGrid(
    const std::vector<double>& depths_spherical,
    const std::vector<PhysicalParameters<double>>& params_spherical,
    const std::vector<bool>& is_discontinuity,
    double planet_radius,
    double flattening_exponent
) {
    if (depths_spherical.size() != params_spherical.size() || depths_spherical.size() != is_discontinuity.size()) {
        throw std::runtime_error("Size mismatch: depths, params, and is_discontinuity must have the same length.");
    }
    if (depths_spherical.empty()) {
        return std::make_tuple(std::vector<double>{}, std::vector<PhysicalParameters<double>>{}, std::vector<bool>{});
    }

    std::vector<double> depths_flat;
    std::vector<PhysicalParameters<double>> params_flat;
    depths_flat.reserve(depths_spherical.size());
    params_flat.reserve(params_spherical.size());

    double flat_origin = 0.0;
    for (size_t i = 0; i < depths_spherical.size(); ++i) {
        double radius = planet_radius - depths_spherical[i];
        if (radius <= 0.0) {
            throw std::runtime_error("Grid node is at or below the center of the Earth (r<=0).");
        }
        double z_flat = planet_radius * std::log(planet_radius / radius);
        if (i == 0) {
            flat_origin = z_flat;
        }
        depths_flat.push_back(z_flat - flat_origin);
        params_flat.push_back(applyEarthFlatteningScaleBA97(params_spherical[i], radius, planet_radius, flattening_exponent));
    }

    return std::make_tuple(depths_flat, params_flat, is_discontinuity);
}

// Experimental density-exponent EFT for node/grid Romberg models.
inline std::tuple<std::vector<double>, std::vector<PhysicalParameters<double>>, std::vector<bool>> py_earthFlatteningGridExperimentalDensity(
    const std::vector<double>& depths_spherical,
    const std::vector<PhysicalParameters<double>>& params_spherical,
    const std::vector<bool>& is_discontinuity,
    double planet_radius,
    double density_exponent
) {
    if (depths_spherical.size() != params_spherical.size() || depths_spherical.size() != is_discontinuity.size()) {
        throw std::runtime_error("Size mismatch: depths, params, and is_discontinuity must have the same length.");
    }
    if (depths_spherical.empty()) {
        return std::make_tuple(std::vector<double>{}, std::vector<PhysicalParameters<double>>{}, std::vector<bool>{});
    }

    std::vector<double> depths_flat;
    std::vector<PhysicalParameters<double>> params_flat;
    depths_flat.reserve(depths_spherical.size());
    params_flat.reserve(params_spherical.size());

    double flat_origin = 0.0;
    for (size_t i = 0; i < depths_spherical.size(); ++i) {
        double radius = planet_radius - depths_spherical[i];
        if (radius <= 0.0) {
            throw std::runtime_error("Grid node is at or below the center of the Earth (r<=0).");
        }
        double z_flat = planet_radius * std::log(planet_radius / radius);
        if (i == 0) {
            flat_origin = z_flat;
        }
        depths_flat.push_back(z_flat - flat_origin);
        params_flat.push_back(applyEarthFlatteningScaleExperimentalDensity(params_spherical[i], radius, planet_radius, density_exponent));
    }

    return std::make_tuple(depths_flat, params_flat, is_discontinuity);
}

inline double validate_trace_phase_identity(
    const std::vector<double>& phase_velocity_roots,
    int accepted_level,
    double expected_phase_velocity)
{
    if (accepted_level < 0 ||
        static_cast<size_t>(accepted_level) >= phase_velocity_roots.size()) {
        throw std::invalid_argument(
            "phase_accepted_level is outside phase_velocity_roots");
    }
    if (!std::isfinite(expected_phase_velocity) ||
        expected_phase_velocity <= 0.0) {
        throw std::invalid_argument(
            "the expected phase velocity is non-finite or non-positive");
    }
    std::vector<double> accepted_roots(
        phase_velocity_roots.begin(),
        phase_velocity_roots.begin() + accepted_level + 1);
    for (double root : accepted_roots) {
        if (!std::isfinite(root) || root <= 0.0) {
            throw std::invalid_argument(
                "phase_velocity_roots contains an invalid accepted root");
        }
    }
    const double traced_phase_velocity =
        neville_extrapolation(accepted_roots).first;
    const double scale = std::max(
        {1.0, std::abs(expected_phase_velocity),
         std::abs(traced_phase_velocity)});
    if (!std::isfinite(traced_phase_velocity) ||
        std::abs(traced_phase_velocity - expected_phase_velocity) >
            5.0e-10 * scale) {
        throw std::invalid_argument(
            "phase_velocity_roots belongs to a different direct phase "
            "solution or branch");
    }
    return traced_phase_velocity;
}

template <typename Refiner>
inline RombergResult evaluate_dispersion_at_romberg_roots(
    Refiner&& refiner,
    double period,
    WaveType wave_type,
    const std::vector<double>& phase_velocity_roots,
    int accepted_level,
    bool phase_only,
    int max_steps,
    double expected_phase_velocity)
{
    const double traced_phase_velocity = validate_trace_phase_identity(
        phase_velocity_roots, accepted_level, expected_phase_velocity);
    if (accepted_level > max_steps) {
        throw std::invalid_argument(
            "phase_accepted_level exceeds the kernel max_steps contract");
    }
    const double omega = 2.0 * M_PI / period;
    std::vector<double> c_table;
    std::vector<double> u_table;
    std::vector<double> hv_table;
    c_table.reserve(static_cast<size_t>(accepted_level) + 1);
    u_table.reserve(c_table.capacity());
    hv_table.reserve(c_table.capacity());

    for (int level = 0; level <= accepted_level; ++level) {
        const auto grid = refiner(level);
        const double c_root = phase_velocity_roots[static_cast<size_t>(level)];
        if (!std::isfinite(c_root) || c_root <= 0.0) {
            throw std::invalid_argument(
                "phase_velocity_roots contains an invalid value");
        }
        const double residual = std::abs(calculateDispersionPhaseValueDouble(
            grid.layers, grid.thicknesses, grid.halfspace,
            omega, c_root, wave_type));
        if (!std::isfinite(residual) ||
            residual > kAdaptiveResidualLimit) {
            throw std::invalid_argument(
                "phase_velocity_roots is not residual-compatible with the "
                "GRID model, frequency, wave type, and refinement level");
        }
        c_table.push_back(c_root);
        if (phase_only) continue;
        u_table.push_back(calculateGroupVelocityFullAD(
            grid.layers, grid.thicknesses, grid.halfspace,
            omega, c_root, wave_type));
        if (wave_type == WaveType::Love) {
            hv_table.push_back(std::numeric_limits<double>::quiet_NaN());
        } else {
            Eigen::Matrix<double, 5, 1> seabed_state;
            seabed_state.setZero();
            const auto surface_state = calculateDispersionFunction(
                grid.layers, grid.thicknesses, grid.halfspace,
                omega, c_root, wave_type, NormalizationMode::Vertical,
                &seabed_state);
            const bool top_is_liquid =
                !grid.layers.empty() && isLiquidLayer(grid.layers.front());
            hv_table.push_back(
                top_is_liquid ? calculateHVRatio(seabed_state)
                              : calculateHVRatio(surface_state));
        }
    }

    const auto [c, c_error] = neville_extrapolation(c_table);
    if (std::abs(c - traced_phase_velocity) >
        32.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, std::abs(c))) {
        throw std::runtime_error(
            "Romberg trace replay changed the validated phase value");
    }
    RombergResult result;
    result.phase_velocity = c;
    result.group_velocity = phase_only
        ? std::numeric_limits<double>::quiet_NaN()
        : neville_extrapolation(u_table).first;
    result.hv_ratio = phase_only
        ? std::numeric_limits<double>::quiet_NaN()
        : neville_extrapolation(hv_table).first;
    result.converged_level = accepted_level;
    result.relative_error = c_error / std::max(std::abs(c), 1.0e-300);
    result.root_trace.phase_velocity_roots = phase_velocity_roots;
    result.root_trace.phase_accepted_level = accepted_level;
    result.root_trace.generated_level =
        static_cast<int>(phase_velocity_roots.size()) - 1;
    result.root_trace.extrapolate_start_level = 0;
    return result;
}

template <typename Refiner>
inline FrequencyCurvatureRombergResult compute_frequency_curvature_ad_grid_from_refiner(
    Refiner&& refiner,
    double period,
    const std::string& wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool adaptive_root_scan,
    double initial_root_window_frac,
    int extrapolate_start_level
) {
    const WaveType type = parseWaveType(wave_type_str);

    auto find_root = [type, tolerance, adaptive_root_scan,
                      initial_root_window_frac,
                      initial_branch_anchor = c_guess](
        const VelocityGridRefinement& grid,
        double omega,
        double c_guess_level,
        const std::vector<double>& c_history
    ) -> double {
        if (adaptive_root_scan) {
            return find_dispersion_root_adaptive(
                grid.layers, grid.thicknesses, grid.halfspace,
                omega, c_guess_level, type, tolerance, c_history,
                initial_root_window_frac, initial_branch_anchor);
        }
        return find_dispersion_root_near_by_scan(
            grid.layers, grid.thicknesses, grid.halfspace,
            omega, c_guess_level, type, tolerance);
    };

    auto root_residual = [type](
        const VelocityGridRefinement& grid,
        double omega,
        double phase_velocity
    ) -> double {
        return std::abs(calculateDispersionPhaseValueDouble(
            grid.layers, grid.thicknesses, grid.halfspace,
            omega, phase_velocity, type));
    };

    auto frequency_curvature = [type](
        const VelocityGridRefinement& grid,
        double omega,
        double phase_velocity
    ) -> FrequencyCurvatureLevelResult {
        return compute_frequency_curvature_ad_level(
            grid.layers, grid.thicknesses, grid.halfspace,
            omega, phase_velocity, type);
    };

    return compute_frequency_curvature_with_romberg_core(
        std::forward<Refiner>(refiner), period, c_guess,
        find_root, root_residual, frequency_curvature,
        max_steps, extrapolate_start_level);
}


inline std::tuple<double, double, double, int, double> py_computeDispersionRomberg(
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
    std::optional<int> branch = std::nullopt
) {
    if (want_value_u != want_value_hv) {
        throw std::invalid_argument(
            "want_value_u and want_value_hv must currently agree: the Romberg "
            "engine computes U and H/V values together, so independent "
            "per-value control is not yet implemented.");
    }
    phase_only = phase_only || !want_value_u;

    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);

    if (!model.validate()) {
        throw std::runtime_error("Invalid GridModel: validation failed.");
    }

    Mesher mesher(model);
    auto refiner = [&mesher](int k) {
        VelocityGridRefinement mesh;
        std::tie(mesh.layers, mesh.thicknesses) = mesher.get_refined_layers(k);
        mesh.halfspace = mesher.get_halfspace();
        return mesh;
    };

    return compute_dispersion_romberg_from_refiner(
        refiner, period, wave_type_str, c_guess, max_steps, tolerance, phase_only,
        adaptive_root_scan, initial_root_window_frac,
        extrapolate_start_level, branch);
}


inline std::tuple<double, double, double, int, double> py_computeDispersionRombergVelocityIso(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vps,
    const std::vector<double>& vss,
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
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN()
) {
    if (want_value_u != want_value_hv) {
        throw std::invalid_argument(
            "want_value_u and want_value_hv must currently agree: the Romberg "
            "engine computes U and H/V values together, so independent "
            "per-value control is not yet implemented.");
    }
    phase_only = phase_only || !want_value_u;

    const GridRefinementPolicy refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    auto refiner = [&](int k) {
        return refine_velocity_grid_iso(
            depths, rhos, vps, vss, is_discontinuity, k,
            refinement_policy);
    };
    return compute_dispersion_romberg_from_refiner(
        refiner, period, wave_type_str, c_guess, max_steps, tolerance, phase_only,
        adaptive_root_scan, initial_root_window_frac,
        extrapolate_start_level);
}

inline std::tuple<double, double, double, int, double> py_computeDispersionRombergVelocityVti(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vphs,
    const std::vector<double>& vpvs,
    const std::vector<double>& vshs,
    const std::vector<double>& vsvs,
    const std::vector<double>& etas,
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
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN()
) {
    if (want_value_u != want_value_hv) {
        throw std::invalid_argument(
            "want_value_u and want_value_hv must currently agree: the Romberg "
            "engine computes U and H/V values together, so independent "
            "per-value control is not yet implemented.");
    }
    phase_only = phase_only || !want_value_u;

    const GridRefinementPolicy refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    auto refiner = [&](int k) {
        return refine_velocity_grid_vti(
            depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_discontinuity, k,
            refinement_policy);
    };
    return compute_dispersion_romberg_from_refiner(
        refiner, period, wave_type_str, c_guess, max_steps, tolerance, phase_only,
        adaptive_root_scan, initial_root_window_frac,
        extrapolate_start_level);
}

struct PhaseVelocityCurveResult {
    std::vector<double> phase_velocities;
    std::vector<int> converged_levels;
    std::vector<double> relative_errors;
    std::vector<bool> fallback_mask;
    std::vector<int> fallback_levels;
    int direct_prediction_count = 0;
};

struct PhaseVelocityRootHistory {
    std::vector<double> periods;
    std::vector<double> roots;
    double previous_prediction_error =
        std::numeric_limits<double>::quiet_NaN();
    bool previous_prediction_failed = false;
};

// Within-period adaptive Brent is already efficient from level 2 onward.
// Cross-period prediction is reserved for the two coarse levels where the
// scalar path otherwise spends most of its time scanning.
inline constexpr size_t kPhaseVelocityCurvePredictedGridLevels = 2;

inline void validate_phase_velocity_curve_periods(const std::vector<double>& periods) {
    if (periods.empty()) {
        throw std::invalid_argument("periods must not be empty");
    }
    for (double period : periods) {
        if (!(period > 0.0) || !std::isfinite(period)) {
            throw std::invalid_argument(
                "periods must contain only positive finite values");
        }
    }
    if (periods.size() < 2) return;
    const bool increasing = periods[1] > periods[0];
    if (periods[1] == periods[0]) {
        throw std::invalid_argument("periods must be strictly monotonic");
    }
    for (size_t i = 2; i < periods.size(); ++i) {
        if ((increasing && !(periods[i] > periods[i - 1])) ||
            (!increasing && !(periods[i] < periods[i - 1]))) {
            throw std::invalid_argument("periods must be strictly monotonic");
        }
    }
}

inline double predict_phase_velocity_curve_root(
    const PhaseVelocityRootHistory& history,
    double next_period
) {
    const size_t available = history.roots.size();
    if (available == 0 || history.periods.size() != available) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (available >= 2) {
        const double previous_step = std::abs(
            history.periods[available - 1] -
            history.periods[available - 2]);
        const double extrapolation_step = std::abs(
            next_period - history.periods.back());
        if (!(previous_step > 0.0) ||
            extrapolation_step > 1.5 * previous_step) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    const size_t count = std::min<size_t>(3, available);
    const size_t start = available - count;
    double predicted = 0.0;
    for (size_t j = start; j < available; ++j) {
        double basis = 1.0;
        for (size_t m = start; m < available; ++m) {
            if (m == j) continue;
            const double denominator = history.periods[j] - history.periods[m];
            if (denominator == 0.0 || !std::isfinite(denominator)) {
                return history.roots.back();
            }
            basis *= (next_period - history.periods[m]) / denominator;
        }
        predicted += history.roots[j] * basis;
    }
    return std::isfinite(predicted) && predicted > 0.0
        ? predicted
        : history.roots.back();
}

template <typename Func>
inline std::optional<double> try_phase_velocity_root_bracket(
    Func&& func,
    double centre,
    double minimum_velocity,
    double maximum_velocity,
    double initial_half_width,
    double maximum_half_width,
    double root_tolerance,
    double unique_scan_step = 0.0,
    double* root_residual = nullptr
) {
    if (root_residual) *root_residual=std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(centre) || !(centre > minimum_velocity) ||
        !(maximum_velocity > centre) || !(initial_half_width > 0.0) ||
        !(maximum_half_width >= initial_half_width)) {
        return std::nullopt;
    }

    double half_width = initial_half_width;
    for (int attempt = 0; attempt < 6; ++attempt) {
        half_width = std::min(half_width, maximum_half_width);
        const double a = std::max(minimum_velocity, centre - half_width);
        const double b = std::min(maximum_velocity, centre + half_width);
        if (!(b > a)) return std::nullopt;

        const bool require_unique_scan =
            unique_scan_step > 0.0 && (b - a) > unique_scan_step;
        if (require_unique_scan) {
            const int points = std::max(
                kAdaptiveMinScanPoints,
                static_cast<int>(std::ceil((b - a) / unique_scan_step)));
            std::vector<double> residuals;
            auto roots = searchRoots(func, a, b, (b-a)/points, root_tolerance,
                                     root_residual ? &residuals : nullptr);
            if (roots.size() == 1) {
                if (root_residual) *root_residual=residuals.front();
                return roots.front();
            }
            if (roots.size() > 1) return std::nullopt;
        } else {
            const double fa = func(a);
            const double fb = func(b);
            if (!std::isfinite(fa) || !std::isfinite(fb)) {
                return std::nullopt;
            }
            if (fa * fb <= 0.0) {
                const auto refined = brents_method_with_values(
                    func, a, b, fa, fb, root_tolerance);
                if (refined.root > 0.0 &&
                    std::isfinite(refined.function_value) &&
                    std::abs(refined.function_value) < kAdaptiveResidualLimit) {
                    if (root_residual) *root_residual=refined.function_value;
                    return refined.root;
                }
                return std::nullopt;
            }
        }

        if (half_width >= maximum_half_width) break;
        half_width = std::min(maximum_half_width, 4.0 * half_width);
    }
    return std::nullopt;
}

// Shared by the typed API and the legacy GRID curve API. A narrow
// prediction bracket uses Brent directly; wider brackets retain the legacy
// scan resolution rather than silently hiding resolvable neighbouring roots.
template <typename Func>
inline std::optional<double> try_predicted_phase_velocity_root(
    Func&& func, double predicted, double previous_error,
    double minimum_velocity, double maximum_velocity,
    double maximum_half_width, double root_tolerance,
    double* root_residual = nullptr
) {
    double half_width = 1.0e-4 * predicted;
    if (std::isfinite(previous_error))
        half_width = std::max(half_width, 8.0 * previous_error);
    half_width = std::min(half_width, maximum_half_width);
    const double legacy_step =
        2.0 * kLegacyScanHalfWidthFrac * predicted / kLegacyScanPoints;
    return try_phase_velocity_root_bracket(func, predicted,
        minimum_velocity, maximum_velocity, half_width,
        maximum_half_width, root_tolerance, legacy_step, root_residual);
}

template <typename Func>
inline double scan_phase_velocity_curve_branch(
    Func&& func,
    double c_min,
    double c_max,
    int scan_points,
    int branch,
    double root_tolerance
) {
    auto roots = searchRoots(
        func, c_min, c_max, (c_max - c_min) / scan_points,
        root_tolerance);
    roots.erase(
        std::remove_if(
            roots.begin(), roots.end(),
            [c_min, c_max](double root) {
                return !std::isfinite(root) ||
                    !(root > c_min) || !(root < c_max);
            }),
        roots.end());
    std::sort(roots.begin(), roots.end());
    if (static_cast<size_t>(branch) >= roots.size()) {
        std::ostringstream message;
        message << "layer_cake branch " << branch
                << " was not found in the velocity interval ["
                << c_min << ", " << c_max << "]";
        throw std::runtime_error(message.str());
    }
    return roots[static_cast<size_t>(branch)];
}

inline PhaseVelocityCurveResult compute_dispersion_curve_layer_cake_impl(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    const std::vector<double>& periods,
    WaveType type,
    int branch,
    double c_min,
    double c_max,
    int scan_points,
    double root_tolerance,
    bool verify_branch
) {
    validate_phase_velocity_curve_periods(periods);
    if (layers.size() != thicknesses.size()) {
        throw std::invalid_argument(
            "layers and thicknesses must have the same size");
    }
    if (branch < 0) {
        throw std::invalid_argument("branch must be nonnegative");
    }
    if (!(c_min > 0.0) || !std::isfinite(c_min) ||
        !(c_max > c_min) || !std::isfinite(c_max)) {
        throw std::invalid_argument(
            "c_min and c_max must define a positive finite interval");
    }
    if (scan_points < 2) {
        throw std::invalid_argument("scan_points must be at least 2");
    }
    if (!(root_tolerance > 0.0) || !std::isfinite(root_tolerance)) {
        throw std::invalid_argument(
            "root_tolerance must be positive and finite");
    }

    PhaseVelocityCurveResult result;
    result.phase_velocities.reserve(periods.size());
    result.converged_levels.assign(periods.size(), -1);
    result.relative_errors.assign(
        periods.size(), std::numeric_limits<double>::quiet_NaN());
    result.fallback_mask.reserve(periods.size());
    result.fallback_levels.reserve(periods.size());
    PhaseVelocityRootHistory history;
    history.periods.reserve(periods.size());
    history.roots.reserve(periods.size());

    for (size_t i = 0; i < periods.size(); ++i) {
        const double omega = 2.0 * M_PI / periods[i];
        auto dispersion = [&](double phase_velocity) {
            return calculateDispersionPhaseValueDouble(
                layers, thicknesses, halfspace,
                omega, phase_velocity, type);
        };

        bool used_fallback = false;
        double root = std::numeric_limits<double>::quiet_NaN();
        double predicted = std::numeric_limits<double>::quiet_NaN();
        if (!history.roots.empty()) {
            predicted = predict_phase_velocity_curve_root(history, periods[i]);
            const double maximum_half_width =
                std::min(0.32, 0.2 * predicted);
            const auto local = try_phase_velocity_root_bracket(
                dispersion, predicted, c_min, c_max,
                std::min(0.04, maximum_half_width), maximum_half_width,
                root_tolerance);
            if (local.has_value()) {
                root = *local;
                if (verify_branch &&
                    calculateAdModeCount(
                        layers, thicknesses, halfspace,
                        omega, root, type) != branch) {
                    root = std::numeric_limits<double>::quiet_NaN();
                } else {
                    ++result.direct_prediction_count;
                }
            }
        }

        if (!std::isfinite(root) || !(root > 0.0)) {
            used_fallback = !history.roots.empty();
            root = scan_phase_velocity_curve_branch(
                dispersion, c_min, c_max, scan_points,
                branch, root_tolerance);
        }
        if (std::isfinite(predicted)) {
            history.previous_prediction_error = std::abs(root - predicted);
        }
        history.periods.push_back(periods[i]);
        history.roots.push_back(root);
        result.phase_velocities.push_back(root);
        result.fallback_mask.push_back(used_fallback);
        result.fallback_levels.push_back(used_fallback ? 1 : 0);
    }
    return result;
}

inline PhaseVelocityCurveResult compute_dispersion_curve_romberg_grid_velocity_vti_impl(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vphs,
    const std::vector<double>& vpvs,
    const std::vector<double>& vshs,
    const std::vector<double>& vsvs,
    const std::vector<double>& etas,
    const std::vector<bool>& is_discontinuity,
    const std::vector<double>& periods,
    WaveType type,
    double initial_c_guess,
    int max_steps,
    double tolerance,
    bool adaptive_root_scan,
    double initial_root_window_frac,
    int extrapolate_start_level,
    const GridRefinementPolicy& refinement_policy,
    std::optional<int> branch
) {
    validate_phase_velocity_curve_periods(periods);
    if (max_steps < 0) {
        throw std::invalid_argument("max_steps must be non-negative");
    }
    if (!(initial_c_guess > 0.0) || !std::isfinite(initial_c_guess)) {
        throw std::invalid_argument(
            "c_guess must be positive and finite");
    }
    if (!(tolerance > 0.0) || !std::isfinite(tolerance)) {
        throw std::invalid_argument(
            "tolerance must be positive and finite");
    }
    if (branch.has_value() && *branch < 0) {
        throw std::invalid_argument("branch must be nonnegative");
    }
    validate_velocity_grid_sizes(
        depths, is_discontinuity,
        {rhos.size(), vphs.size(), vpvs.size(), vshs.size(),
         vsvs.size(), etas.size()});

    auto refiner = disper26::detail::make_dispersion_mesh_cache<VelocityGridRefinement>(
        max_steps, [&](int level) {
            return refine_velocity_grid_vti(
                depths, rhos, vphs, vpvs, vshs, vsvs, etas,
                is_discontinuity, level, refinement_policy);
        });

    std::vector<PhaseVelocityRootHistory> level_histories(
        static_cast<size_t>(max_steps) + 1);
    PhaseVelocityCurveResult output;
    output.phase_velocities.reserve(periods.size());
    output.converged_levels.reserve(periods.size());
    output.relative_errors.reserve(periods.size());
    output.fallback_mask.reserve(periods.size());
    output.fallback_levels.reserve(periods.size());
    double c_guess = initial_c_guess;
    const double root_tolerance = std::min(tolerance, 1.0e-8);

    for (size_t period_index = 0;
         period_index < periods.size(); ++period_index) {
        std::vector<double> predictions(level_histories.size(),
            std::numeric_limits<double>::quiet_NaN());
        int guarded_fallback_levels = 0;
        if (period_index >= 2 && !branch.has_value()) {
            const size_t predicted_levels = std::min(
                level_histories.size(),
                kPhaseVelocityCurvePredictedGridLevels);
            for (size_t level = 0; level < predicted_levels; ++level) {
                if (level_histories[level].roots.size() >= 2) {
                    if (!level_histories[level].previous_prediction_failed) {
                        predictions[level] = predict_phase_velocity_curve_root(
                            level_histories[level], periods[period_index]);
                    }
                    if (!std::isfinite(predictions[level])) {
                        ++guarded_fallback_levels;
                    }
                }
            }
        }

        int fallback_levels = guarded_fallback_levels;
        std::vector<bool> prediction_failed(
            level_histories.size(), false);
        const double branch_anchor = c_guess;
        auto find_root = [&](
            const VelocityGridRefinement& grid,
            double omega,
            double level_c_guess,
            const std::vector<double>& within_period_history
        ) {
            const size_t level = within_period_history.size();
            if (level < predictions.size() &&
                std::isfinite(predictions[level])) {
                const double predicted = predictions[level];
                const double previous_error =
                    level_histories[level].previous_prediction_error;
                const double maximum_half_width = 1.0e-2 * predicted;
                auto dispersion = [&](double phase_velocity) {
                    return calculateDispersionPhaseValueDouble(
                        grid.layers, grid.thicknesses, grid.halfspace,
                        omega, phase_velocity, type);
                };
                // A direct bracket no wider than one legacy scan interval
                // cannot newly hide a pair that the scalar scan resolved.
                // Wider expansions use an explicit >=8-point uniqueness scan.
                const auto local = try_predicted_phase_velocity_root(
                    dispersion, predicted, previous_error, 0.1,
                    predicted + maximum_half_width,
                    maximum_half_width, root_tolerance);
                if (local.has_value()) {
                    ++output.direct_prediction_count;
                    return *local;
                }
                ++fallback_levels;
                prediction_failed[level] = true;
            }

            if (branch.has_value()) {
                return find_dispersion_root_by_mode_order(
                    grid, omega, type, *branch, tolerance,
                    level_c_guess, within_period_history);
            }
            if (adaptive_root_scan) {
                return find_dispersion_root_adaptive(
                    grid.layers, grid.thicknesses, grid.halfspace,
                    omega, level_c_guess, type, tolerance,
                    within_period_history, initial_root_window_frac,
                    branch_anchor);
            }
            return find_dispersion_root_near_by_scan(
                grid.layers, grid.thicknesses, grid.halfspace,
                omega, level_c_guess, type, tolerance);
        };
        auto unused_group_velocity = [](
            const VelocityGridRefinement&, double, double) {
            return std::numeric_limits<double>::quiet_NaN();
        };
        auto unused_hv = unused_group_velocity;

        const RombergResult result = compute_with_romberg_core(
            refiner, periods[period_index], c_guess,
            find_root, unused_group_velocity, unused_hv,
            max_steps, tolerance, true, extrapolate_start_level);
        if (!std::isfinite(result.phase_velocity) ||
            !(result.phase_velocity > 0.0)) {
            throw std::runtime_error(
                "Romberg phase-velocity curve produced an invalid root");
        }

        const auto& raw_roots = result.root_trace.phase_velocity_roots;
        for (size_t level = 0;
             level < raw_roots.size() && level < level_histories.size();
             ++level) {
            if (std::isfinite(predictions[level])) {
                level_histories[level].previous_prediction_error =
                    std::abs(raw_roots[level] - predictions[level]);
            }
            level_histories[level].previous_prediction_failed =
                prediction_failed[level];
            level_histories[level].periods.push_back(periods[period_index]);
            level_histories[level].roots.push_back(raw_roots[level]);
        }
        c_guess = result.phase_velocity;
        output.phase_velocities.push_back(result.phase_velocity);
        output.converged_levels.push_back(result.converged_level);
        output.relative_errors.push_back(result.relative_error);
        output.fallback_mask.push_back(fallback_levels > 0);
        output.fallback_levels.push_back(fallback_levels);
    }
    return output;
}


using VectorADScalar = Eigen::AutoDiffScalar<Eigen::VectorXd>;

inline VectorADScalar make_vector_ad(double value, int num_vars, int active_idx = -1) {
    VectorADScalar out;
    out.value() = value;
    out.derivatives() = Eigen::VectorXd::Zero(num_vars);
    if (active_idx >= 0) out.derivatives()(active_idx) = 1.0;
    return out;
}

inline bool is_valid_u_kernel_mode(const std::string& mode) {
    return mode == "variational_frequency_ad" ||
           mode == "variational" || mode == "full_ad";
}

inline bool is_frequency_variational_u_kernel_mode(const std::string& mode) {
    return mode == "variational_frequency_ad";
}

// Apply the transpose of the velocity-grid refinement Jacobian without ever
// materializing its dense (refined-parameter x node-parameter) form.  Each
// refined layer depends on only its two parent nodes, so this is O(L + P).
inline Eigen::VectorXd map_vti_gradient_to_velocity_iso_nodes(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vps,
    const std::vector<double>& vss,
    const std::vector<bool>& is_discontinuity,
    int refinement_level,
    const GridRefinementPolicy& policy,
    const Eigen::VectorXd& g_vti
) {
    const int n = static_cast<int>(depths.size());
    std::vector<bool> fluid_nodes(depths.size(), false);
    const auto geometry = build_grid_sublayer_geometry(
        depths, is_discontinuity, fluid_nodes, refinement_level, policy);
    if (g_vti.size() != (static_cast<int>(geometry.size()) + 1) * 6) {
        throw std::runtime_error("velocity-iso sublayer gradient size mismatch");
    }
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n * 3);

    auto accumulate = [&](int sublayer, int node, double weight,
                          double density_scale, double velocity_scale,
                          double rho, double vp, double vs) {
        const double grho = g_vti(sublayer * 6 + 0);
        const double gA = g_vti(sublayer * 6 + 1);
        const double gC = g_vti(sublayer * 6 + 2);
        const double gF = g_vti(sublayer * 6 + 3);
        const double gL = g_vti(sublayer * 6 + 4);
        const double gN = g_vti(sublayer * 6 + 5);
        const double q_rho = grho + (gA + gC) * vp * vp
                           + gF * (vp * vp - 2.0 * vs * vs)
                           + (gL + gN) * vs * vs;
        const double q_vp = 2.0 * rho * vp * (gA + gC + gF);
        const double q_vs = 2.0 * rho * vs * (gL + gN - 2.0 * gF);
        out(node * 3 + 0) += weight * density_scale * q_rho;
        out(node * 3 + 1) += weight * velocity_scale * q_vp;
        out(node * 3 + 2) += weight * velocity_scale * q_vs;
    };

    int s = 0;
    for (const auto& geo : geometry) {
        const int i = geo.parent_node;
        const double a = geo.alpha;
        const double rho = ((1.0 - a) * rhos[i] + a * rhos[i + 1]) * geo.density_scale;
        const double vp = ((1.0 - a) * vps[i] + a * vps[i + 1]) * geo.velocity_scale;
        const double vs = ((1.0 - a) * vss[i] + a * vss[i + 1]) * geo.velocity_scale;
        accumulate(s, i, 1.0 - a, geo.density_scale, geo.velocity_scale,
                   rho, vp, vs);
        accumulate(s, i + 1, a, geo.density_scale, geo.velocity_scale,
                   rho, vp, vs);
        ++s;
    }
    GridSublayerGeometry hs;
    hs.radius_ratio = policy.is_flat() ? 1.0
        : (policy.planet_radius - depths.back()) / policy.planet_radius;
    grid_fill_flattening_scales(hs, policy);
    accumulate(s, n - 1, 1.0, hs.density_scale, hs.velocity_scale,
               rhos.back() * hs.density_scale,
               vps.back() * hs.velocity_scale,
               vss.back() * hs.velocity_scale);
    for (int i = 0; i < n; ++i)
        if (std::abs(vss[i]) < 1e-12) out.segment(i * 3, 3).setZero();
    return out;
}

inline Eigen::VectorXd map_vti_gradient_to_velocity_vti_nodes(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vphs,
    const std::vector<double>& vpvs,
    const std::vector<double>& vshs,
    const std::vector<double>& vsvs,
    const std::vector<double>& etas,
    const std::vector<bool>& is_discontinuity,
    int refinement_level,
    const GridRefinementPolicy& policy,
    const Eigen::VectorXd& g_vti
) {
    const int n = static_cast<int>(depths.size());
    std::vector<bool> fluid_nodes(depths.size(), false);
    const auto geometry = build_grid_sublayer_geometry(
        depths, is_discontinuity, fluid_nodes, refinement_level, policy);
    if (g_vti.size() != (static_cast<int>(geometry.size()) + 1) * 6) {
        throw std::runtime_error("velocity-vti sublayer gradient size mismatch");
    }
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n * 6);

    auto accumulate = [&](int sublayer, int node, double weight,
                          double density_scale, double velocity_scale,
                          double rho, double vph, double vpv, double vsh,
                          double vsv, double eta) {
        const double grho = g_vti(sublayer * 6 + 0);
        const double gA = g_vti(sublayer * 6 + 1);
        const double gC = g_vti(sublayer * 6 + 2);
        const double gF = g_vti(sublayer * 6 + 3);
        const double gL = g_vti(sublayer * 6 + 4);
        const double gN = g_vti(sublayer * 6 + 5);
        const double q_rho = grho + gA * vph * vph + gC * vpv * vpv
                           + gL * vsv * vsv + gN * vsh * vsh
                           + gF * eta * (vph * vph - 2.0 * vsv * vsv);
        const double q_vph = 2.0 * rho * vph * (gA + eta * gF);
        const double q_vpv = 2.0 * rho * vpv * gC;
        const double q_vsh = 2.0 * rho * vsh * gN;
        const double q_vsv = 2.0 * rho * vsv * (gL - 2.0 * eta * gF);
        const double q_eta = gF * rho * (vph * vph - 2.0 * vsv * vsv);
        out(node * 6 + 0) += weight * density_scale * q_rho;
        out(node * 6 + 1) += weight * velocity_scale * q_vph;
        out(node * 6 + 2) += weight * velocity_scale * q_vpv;
        out(node * 6 + 3) += weight * velocity_scale * q_vsh;
        out(node * 6 + 4) += weight * velocity_scale * q_vsv;
        out(node * 6 + 5) += weight * q_eta;
    };

    int s = 0;
    for (const auto& geo : geometry) {
        const int i = geo.parent_node;
        const double a = geo.alpha;
        auto interp = [&](const std::vector<double>& v) {
            return (1.0 - a) * v[i] + a * v[i + 1];
        };
        const double rho = interp(rhos) * geo.density_scale;
        const double vph = interp(vphs) * geo.velocity_scale;
        const double vpv = interp(vpvs) * geo.velocity_scale;
        const double vsh = interp(vshs) * geo.velocity_scale;
        const double vsv = interp(vsvs) * geo.velocity_scale;
        const double eta = interp(etas);
        accumulate(s, i, 1.0 - a, geo.density_scale, geo.velocity_scale,
                   rho, vph, vpv, vsh, vsv, eta);
        accumulate(s, i + 1, a, geo.density_scale, geo.velocity_scale,
                   rho, vph, vpv, vsh, vsv, eta);
        ++s;
    }
    GridSublayerGeometry hs;
    hs.radius_ratio = policy.is_flat() ? 1.0
        : (policy.planet_radius - depths.back()) / policy.planet_radius;
    grid_fill_flattening_scales(hs, policy);
    accumulate(s, n - 1, 1.0, hs.density_scale, hs.velocity_scale,
               rhos.back() * hs.density_scale,
               vphs.back() * hs.velocity_scale,
               vpvs.back() * hs.velocity_scale,
               vshs.back() * hs.velocity_scale,
               vsvs.back() * hs.velocity_scale, etas.back());
    for (int i = 0; i < n; ++i)
        if (std::abs(vshs[i]) < 1e-12 && std::abs(vsvs[i]) < 1e-12)
            out.segment(i * 6, 6).setZero();
    return out;
}

inline std::vector<double> interfaceBoundaryWeights(
    const std::vector<bool>& is_discontinuity, int n_nodes, int pair_start,
    int refinement_level);

template <typename RefinerDouble, typename RefinerAD, typename FrequencyMapper>
inline std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
compute_dispersion_romberg_velocity_kernels_from_refiners(
    RefinerDouble&& refiner_double,
    RefinerAD&& refiner_ad,
    FrequencyMapper&& frequency_mapper,
    int node_count,
    const std::vector<std::string>& param_names,
    double period,
    std::string wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool compute_c,
    bool compute_u,
    bool compute_hv,
    bool phase_only = false,
    bool adaptive_root_scan = true,
    double initial_root_window_frac = 0.0,
    std::string u_kernel_mode = "variational_frequency_ad",
    bool kernel_extrapolation = true,
    const std::vector<double>* provided_phase_velocity_roots = nullptr,
    int provided_phase_accepted_level = -1
) {
    if (phase_only && (compute_u || compute_hv)) {
        throw std::invalid_argument(
            "phase_only (or want_value_u=False/want_value_hv=False) requires "
            "want_kernel_u=False and want_kernel_hv=False");
    }
    if (!is_valid_u_kernel_mode(u_kernel_mode)) {
        throw std::invalid_argument(
            "u_kernel_mode must be \"variational_frequency_ad\", "
            "\"variational\", or \"full_ad\"");
    }

    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    RombergResult dispersion = provided_phase_velocity_roots
        ? evaluate_dispersion_at_romberg_roots(
              refiner_double, period, type, *provided_phase_velocity_roots,
              provided_phase_accepted_level, phase_only, max_steps, c_guess)
        : compute_dispersion_romberg_result_from_refiner(
              refiner_double, period, wave_type_str, c_guess, max_steps,
              tolerance, phase_only, adaptive_root_scan,
              initial_root_window_frac);
    const double c = dispersion.phase_velocity;
    const double U = dispersion.group_velocity;
    const double HV = dispersion.hv_ratio;
    const int level = dispersion.converged_level;
    const double rel_err = dispersion.relative_error;
    const auto& phase_velocity_roots =
        dispersion.root_trace.phase_velocity_roots;
    if (phase_velocity_roots.size() <= static_cast<size_t>(level)) {
        throw std::runtime_error(
            "direct Romberg result did not retain every accepted level root");
    }

    const int num_params_per_node = static_cast<int>(param_names.size());
    const int num_vars = node_count * num_params_per_node;
    double omega = 2.0 * M_PI / period;

    auto eval_level = [&](int k, double c_guess_k,
                          Eigen::VectorXd& dc_out, Eigen::VectorXd& dU_out,
                          Eigen::VectorXd& dHV_out) {
        auto grid_dbl = refiner_double(k);
        if (is_frequency_variational_u_kernel_mode(u_kernel_mode) && compute_u) {
            const double c_root_k = c_guess_k;
            try {
                const auto frequency =
                    computeGroupVelocityKernelFrequencyVariationalLevel(
                        grid_dbl.layers, grid_dbl.thicknesses,
                        grid_dbl.halfspace, omega, c_root_k, type, false);
                dU_out = frequency_mapper(k, frequency.group_material_vti);
            } catch (const std::exception&) {
                auto grid_ad = refiner_ad(k);
                compute_grid_level_kernels<VectorADScalar>(
                    grid_dbl.layers, grid_dbl.thicknesses, grid_dbl.halfspace,
                    grid_ad.layers, grid_ad.thicknesses, grid_ad.halfspace,
                    omega, c_guess_k, type, false, true, true, num_vars,
                    dc_out, dU_out, dHV_out, false);
            }
            if (compute_c || compute_hv) {
                auto grid_ad = refiner_ad(k);
                Eigen::VectorXd unused_u;
                compute_grid_level_kernels<VectorADScalar>(
                    grid_dbl.layers, grid_dbl.thicknesses, grid_dbl.halfspace,
                    grid_ad.layers, grid_ad.thicknesses, grid_ad.halfspace,
                    omega, c_guess_k, type, compute_hv, false, true, num_vars,
                    dc_out, unused_u, dHV_out, false);
            } else {
                dc_out = Eigen::VectorXd::Zero(num_vars);
                dHV_out = Eigen::VectorXd::Zero(num_vars);
            }
        } else {
            auto grid_ad = refiner_ad(k);
            compute_grid_level_kernels<VectorADScalar>(
                grid_dbl.layers, grid_dbl.thicknesses, grid_dbl.halfspace,
                grid_ad.layers, grid_ad.thicknesses, grid_ad.halfspace,
                omega, c_guess_k, type, compute_hv, compute_u,
                u_kernel_mode == "variational", num_vars,
                dc_out, dU_out, dHV_out, false);
        }
    };

    Eigen::VectorXd dc_dp = Eigen::VectorXd::Zero(num_vars);
    Eigen::VectorXd dU_dp = Eigen::VectorXd::Zero(num_vars);
    Eigen::VectorXd dHV_dp = Eigen::VectorXd::Zero(num_vars);

    if (compute_c || compute_u || compute_hv) {
        if (kernel_extrapolation) {
            std::vector<Eigen::VectorXd> dc_tab, dU_tab, dHV_tab;
            for (int k = 0; k <= level; ++k) {
                Eigen::VectorXd dck, dUk, dHVk;
                eval_level(k, phase_velocity_roots[static_cast<size_t>(k)],
                           dck, dUk, dHVk);
                dc_tab.push_back(dck); dU_tab.push_back(dUk); dHV_tab.push_back(dHVk);
            }
            auto neville_vec = [&](const std::vector<Eigen::VectorXd>& tab) {
                Eigen::VectorXd out = Eigen::VectorXd::Zero(num_vars);
                for (int idx = 0; idx < num_vars; ++idx) {
                    std::vector<double> col;
                    col.reserve(tab.size());
                    for (const auto& v : tab) col.push_back(v(idx));
                    out(idx) = neville_extrapolation(col).first;
                }
                return out;
            };
            dc_dp = neville_vec(dc_tab);
            dU_dp = neville_vec(dU_tab);
            dHV_dp = neville_vec(dHV_tab);
        } else {
            eval_level(level, phase_velocity_roots[static_cast<size_t>(level)],
                       dc_dp, dU_dp, dHV_dp);
        }
    }

    std::vector<SensitivityKernel> kernels;
    int var_idx = 0;
    for (int i = 0; i < node_count; ++i) {
        for (const auto& param_name : param_names) {
            SensitivityKernel k;
            k.parameterName = param_name;
            k.layerIndex = i;
            k.d_PhaseVelocity = compute_c ? dc_dp(var_idx) : 0.0;
            k.d_GroupVelocity = compute_u ? dU_dp(var_idx) : 0.0;
            k.d_HVRatio = compute_hv ? dHV_dp(var_idx) : 0.0;
            kernels.push_back(k);
            ++var_idx;
        }
    }

    return std::make_tuple(c, U, HV, level, rel_err, kernels);
}

inline std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
py_computeDispersionRombergVelocityIsoWithKernels(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vps,
    const std::vector<double>& vss,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool compute_c = true,
    bool compute_u = true,
    bool compute_hv = true,
    bool phase_only = false,
    bool want_value_u = true,
    bool want_value_hv = true,
    bool adaptive_root_scan = true,
    double initial_root_window_frac = 0.0,
    std::string u_kernel_mode = "variational_frequency_ad",
    bool kernel_extrapolation = true,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    if (want_value_u != want_value_hv) {
        throw std::invalid_argument(
            "want_value_u and want_value_hv must currently agree: the Romberg "
            "engine computes U and H/V values together, so independent "
            "per-value control is not yet implemented.");
    }
    phase_only = phase_only || !want_value_u;

    const GridRefinementPolicy refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    validate_velocity_grid_sizes(depths, is_discontinuity, {rhos.size(), vps.size(), vss.size()});
    const int node_count = static_cast<int>(depths.size());
    const int num_vars = node_count * 3;

    std::vector<VectorADScalar> rhos_ad, vps_ad, vss_ad;
    bool ad_nodes_initialized = false;
    auto ensure_ad_nodes = [&]() {
        if (ad_nodes_initialized) return;
        rhos_ad.resize(node_count);
        vps_ad.resize(node_count);
        vss_ad.resize(node_count);
        int var_idx = 0;
        for (int i = 0; i < node_count; ++i) {
            const bool is_fluid = std::abs(vss[i]) < 1e-12;
            rhos_ad[i] = make_vector_ad(
                rhos[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
            vps_ad[i] = make_vector_ad(
                vps[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
            vss_ad[i] = make_vector_ad(
                vss[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
        }
        ad_nodes_initialized = true;
    };

    auto refiner_double = [&](int k) {
        return refine_velocity_grid_iso(
            depths, rhos, vps, vss, is_discontinuity, k,
            refinement_policy);
    };
    auto refiner_ad = [&](int k) {
        ensure_ad_nodes();
        return refine_velocity_grid_iso_values(
            depths, rhos_ad, vps_ad, vss_ad, is_discontinuity, k,
            refinement_policy);
    };
    auto frequency_mapper = [&](int k, const Eigen::VectorXd& g_vti) {
        return map_vti_gradient_to_velocity_iso_nodes(
            depths, rhos, vps, vss, is_discontinuity, k,
            refinement_policy, g_vti);
    };

    return compute_dispersion_romberg_velocity_kernels_from_refiners(
        refiner_double, refiner_ad, frequency_mapper,
        node_count, {"rho", "vp", "vs"},
        period, wave_type_str, c_guess, max_steps, tolerance, compute_c, compute_u, compute_hv,
        phase_only, adaptive_root_scan, initial_root_window_frac,
        u_kernel_mode, kernel_extrapolation,
        phase_velocity_roots ? &*phase_velocity_roots : nullptr,
        phase_accepted_level
    );
}

inline std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
py_computeDispersionRombergVelocityVtiWithKernels(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vphs,
    const std::vector<double>& vpvs,
    const std::vector<double>& vshs,
    const std::vector<double>& vsvs,
    const std::vector<double>& etas,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool compute_c = true,
    bool compute_u = true,
    bool compute_hv = true,
    bool phase_only = false,
    bool want_value_u = true,
    bool want_value_hv = true,
    bool adaptive_root_scan = true,
    double initial_root_window_frac = 0.0,
    std::string u_kernel_mode = "variational_frequency_ad",
    bool kernel_extrapolation = true,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    if (want_value_u != want_value_hv) {
        throw std::invalid_argument(
            "want_value_u and want_value_hv must currently agree: the Romberg "
            "engine computes U and H/V values together, so independent "
            "per-value control is not yet implemented.");
    }
    phase_only = phase_only || !want_value_u;

    const GridRefinementPolicy refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    validate_velocity_grid_sizes(depths, is_discontinuity, {rhos.size(), vphs.size(), vpvs.size(), vshs.size(), vsvs.size(), etas.size()});
    const int node_count = static_cast<int>(depths.size());
    const int num_vars = node_count * 6;

    std::vector<VectorADScalar> rhos_ad, vphs_ad, vpvs_ad;
    std::vector<VectorADScalar> vshs_ad, vsvs_ad, etas_ad;
    bool ad_nodes_initialized = false;
    auto ensure_ad_nodes = [&]() {
        if (ad_nodes_initialized) return;
        rhos_ad.resize(node_count);
        vphs_ad.resize(node_count);
        vpvs_ad.resize(node_count);
        vshs_ad.resize(node_count);
        vsvs_ad.resize(node_count);
        etas_ad.resize(node_count);
        int var_idx = 0;
        for (int i = 0; i < node_count; ++i) {
            const bool is_fluid =
                std::abs(vsvs[i]) < 1e-12 && std::abs(vshs[i]) < 1e-12;
            rhos_ad[i] = make_vector_ad(
                rhos[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
            vphs_ad[i] = make_vector_ad(
                vphs[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
            vpvs_ad[i] = make_vector_ad(
                vpvs[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
            vshs_ad[i] = make_vector_ad(
                vshs[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
            vsvs_ad[i] = make_vector_ad(
                vsvs[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
            etas_ad[i] = make_vector_ad(
                etas[i], num_vars, is_fluid ? -1 : var_idx);
            ++var_idx;
        }
        ad_nodes_initialized = true;
    };

    auto refiner_double = [&](int k) {
        return refine_velocity_grid_vti(
            depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_discontinuity, k,
            refinement_policy);
    };
    auto refiner_ad = [&](int k) {
        ensure_ad_nodes();
        return refine_velocity_grid_vti_values(
            depths, rhos_ad, vphs_ad, vpvs_ad, vshs_ad, vsvs_ad, etas_ad,
            is_discontinuity, k,
            refinement_policy);
    };
    auto frequency_mapper = [&](int k, const Eigen::VectorXd& g_vti) {
        return map_vti_gradient_to_velocity_vti_nodes(
            depths, rhos, vphs, vpvs, vshs, vsvs, etas,
            is_discontinuity, k, refinement_policy, g_vti);
    };

    return compute_dispersion_romberg_velocity_kernels_from_refiners(
        refiner_double, refiner_ad, frequency_mapper,
        node_count, {"rho", "Vph", "Vpv", "Vsh", "Vsv", "eta"},
        period, wave_type_str, c_guess, max_steps, tolerance, compute_c, compute_u, compute_hv,
        phase_only, adaptive_root_scan, initial_root_window_frac,
        u_kernel_mode, kernel_extrapolation,
        phase_velocity_roots ? &*phase_velocity_roots : nullptr,
        phase_accepted_level
    );
}

inline Eigen::VectorXd map_node_vti_gradient_to_kernel_order(
    const std::vector<PhysicalParameters<double>>& params,
    const Eigen::VectorXd& g_vti,
    SensitivityMode mode
) {
    const int n = static_cast<int>(params.size());
    if (g_vti.size() != n * 6) {
        throw std::runtime_error("node VTI gradient size mismatch");
    }
    if (mode == SensitivityMode::VTI) {
        // Internal order is [rho,A,C,F,L,N]; the established GRID kernel API
        // exposes [A,C,F,L,N,Rho].
        Eigen::VectorXd out = Eigen::VectorXd::Zero(n * 6);
        constexpr int order[6] = {1, 2, 3, 4, 5, 0};
        for (int i = 0; i < n; ++i) {
            if (isLiquidLayer(params[i])) continue;
            for (int p = 0; p < 6; ++p)
                out(i * 6 + p) = g_vti(i * 6 + order[p]);
        }
        return out;
    }

    Eigen::VectorXd out = Eigen::VectorXd::Zero(n * 3);
    for (int i = 0; i < n; ++i) {
        if (isLiquidLayer(params[i])) continue;
        const double rho = params[i].rho;
        const double vs = params[i].vs();
        const double vp = params[i].vp();
        const double grho = g_vti(i * 6 + 0);
        const double gA = g_vti(i * 6 + 1);
        const double gC = g_vti(i * 6 + 2);
        const double gF = g_vti(i * 6 + 3);
        const double gL = g_vti(i * 6 + 4);
        const double gN = g_vti(i * 6 + 5);
        out(i * 3 + 0) = 2.0 * rho * vs * (gL + gN - 2.0 * gF);
        out(i * 3 + 1) = 2.0 * rho * vp * (gA + gC + gF);
        out(i * 3 + 2) = grho + (gA + gC) * vp * vp
                       + gF * (vp * vp - 2.0 * vs * vs)
                       + (gL + gN) * vs * vs;
    }
    return out;
}

// --- GRID Mode Romberg with Node-Based Sensitivity Kernels ---
inline std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
py_computeDispersionRombergWithKernels(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_guess,
    std::string sensitivity_mode_str,
    int max_steps,
    double tolerance,
    bool compute_c = true,
    bool compute_u = true,
    bool compute_hv = true,
    bool phase_only = false,
    bool want_value_u = true,
    bool want_value_hv = true,
    bool adaptive_root_scan = true,
    double initial_root_window_frac = 0.0,
    std::string u_kernel_mode = "variational_frequency_ad",
    bool kernel_extrapolation = true,
    const std::string& grid_policy = "flat",
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN(),
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    if (want_value_u != want_value_hv) {
        throw std::invalid_argument(
            "want_value_u and want_value_hv must currently agree: the Romberg "
            "engine computes U and H/V values together, so independent "
            "per-value control is not yet implemented.");
    }
    phase_only = phase_only || !want_value_u;
    if (phase_only && (compute_u || compute_hv)) {
        throw std::invalid_argument(
            "phase_only (or want_value_u=False/want_value_hv=False) requires "
            "want_kernel_u=False and want_kernel_hv=False");
    }
    if (!is_valid_u_kernel_mode(u_kernel_mode)) {
        throw std::invalid_argument(
            "u_kernel_mode must be \"variational_frequency_ad\", "
            "\"variational\", or \"full_ad\"");
    }

    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    SensitivityMode sens_mode = SensitivityMode::Isotropic;
    if (sensitivity_mode_str == "VTI" || sensitivity_mode_str == "vti") sens_mode = SensitivityMode::VTI;

    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);

    if (!model.validate()) {
        throw std::runtime_error("Invalid GridModel: validation failed.");
    }

    Mesher mesher(model);
    auto refiner_double = [&mesher](int k) {
        VelocityGridRefinement grid;
        std::tie(grid.layers, grid.thicknesses) = mesher.get_refined_layers(k);
        grid.halfspace = mesher.get_halfspace();
        return grid;
    };
    RombergResult dispersion;
    if (phase_velocity_roots) {
        dispersion = evaluate_dispersion_at_romberg_roots(
            refiner_double, period, type, *phase_velocity_roots,
            phase_accepted_level, phase_only, max_steps, c_guess);
    } else {
        dispersion = compute_dispersion_romberg_result_from_refiner(
            refiner_double, period, wave_type_str, c_guess, max_steps,
            tolerance, phase_only, adaptive_root_scan,
            initial_root_window_frac);
    }
    const double c = dispersion.phase_velocity;
    const double U = dispersion.group_velocity;
    const double HV = dispersion.hv_ratio;
    const int level = dispersion.converged_level;
    const double rel_err = dispersion.relative_error;
    const auto& roots = dispersion.root_trace.phase_velocity_roots;
    if (roots.size() <= static_cast<size_t>(level)) {
        throw std::runtime_error(
            "direct Romberg result did not retain every accepted level root");
    }

    double omega = 2.0 * M_PI / period;

    // Per-level meshes are fetched inside eval_level below (kernels are
    // extrapolated across levels); only the level-independent halfspace is
    // needed here.
    auto halfspace_dbl = mesher.get_halfspace();

    int num_discontinuities = 0;
    std::vector<int> node_to_discontinuity_idx(mesher.get_node_count(), -1);
    std::vector<int> discontinuity_pair_starts;
    const auto& discon_flags = mesher.get_discontinuities();
    for (size_t i = 0; i < discon_flags.size(); ++i) {
        if (discon_flags[i]) {
            discontinuity_pair_starts.push_back(static_cast<int>(i));
            node_to_discontinuity_idx[i] = num_discontinuities;
            if (i + 1 < node_to_discontinuity_idx.size()) {
                node_to_discontinuity_idx[i+1] = num_discontinuities;
            }
            num_discontinuities++;
        }
    }

    const int N = mesher.get_node_count();
    const int NUM_PARAMS_PER_NODE = (sens_mode == SensitivityMode::VTI) ? 6 : 3;
    const int NUM_PARAM_VARS = NUM_PARAMS_PER_NODE * N;
    const int NUM_VARS = NUM_PARAM_VARS + num_discontinuities;
    std::vector<bool> fluid_discontinuities(num_discontinuities, false);
    for (int d = 0; d < num_discontinuities; ++d) {
        for (int i = 0; i < N; ++i) {
            if (node_to_discontinuity_idx[i] == d &&
                std::abs(params[i].c44) < 1e-12) {
                fluid_discontinuities[d] = true;
                break;
            }
        }
    }

    using ADScalar = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    std::vector<PhysicalParameters<ADScalar>> node_params_ad;
    std::vector<ADScalar> node_depths_ad;
    PhysicalParameters<ADScalar> halfspace_ad;
    bool ad_model_initialized = false;
    auto ensure_ad_model = [&]() {
        if (ad_model_initialized) return;
        node_params_ad.resize(N);
        node_depths_ad.resize(N);
        auto make_ad_constant = [&](double value) {
            ADScalar out;
            out.value() = value;
            out.derivatives() = Eigen::VectorXd::Zero(NUM_VARS);
            return out;
        };
        auto copy_ad_constants = [&](PhysicalParameters<ADScalar>& dst,
                                     const PhysicalParameters<double>& src) {
            dst.rho = make_ad_constant(src.rho);
            dst.c11 = make_ad_constant(src.c11);
            dst.c13 = make_ad_constant(src.c13);
            dst.c33 = make_ad_constant(src.c33);
            dst.c44 = make_ad_constant(src.c44);
            dst.c66 = make_ad_constant(src.c66);
        };
        int var_idx = 0;
        for (int i = 0; i < N; ++i) {
            const auto& p = params[i];
            const bool is_fluid_node = std::abs(p.c44) < 1e-12;
            if (is_fluid_node) {
                copy_ad_constants(node_params_ad[i], p);
                var_idx += NUM_PARAMS_PER_NODE;
            } else if (sens_mode == SensitivityMode::Isotropic) {
                const double vs = std::sqrt(p.c44 / p.rho);
                const double vp = std::sqrt(p.c11 / p.rho);
                ADScalar vs_ad, vp_ad, rho_ad;
                vs_ad.value() = vs;
                vs_ad.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                vp_ad.value() = vp;
                vp_ad.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                rho_ad.value() = p.rho;
                rho_ad.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                node_params_ad[i].rho = rho_ad;
                node_params_ad[i].c44 = rho_ad * vs_ad * vs_ad;
                node_params_ad[i].c66 = node_params_ad[i].c44;
                node_params_ad[i].c11 = rho_ad * vp_ad * vp_ad;
                node_params_ad[i].c33 = node_params_ad[i].c11;
                node_params_ad[i].c13 =
                    node_params_ad[i].c11 - ADScalar(2.0) * node_params_ad[i].c44;
            } else {
                node_params_ad[i].c11.value() = p.c11;
                node_params_ad[i].c11.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                node_params_ad[i].c33.value() = p.c33;
                node_params_ad[i].c33.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                node_params_ad[i].c13.value() = p.c13;
                node_params_ad[i].c13.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                node_params_ad[i].c44.value() = p.c44;
                node_params_ad[i].c44.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                node_params_ad[i].c66.value() = p.c66;
                node_params_ad[i].c66.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
                node_params_ad[i].rho.value() = p.rho;
                node_params_ad[i].rho.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, var_idx++);
            }
            node_params_ad[i].gravity = make_ad_constant(p.gravity);
            node_params_ad[i].gravity_internal_scale =
                make_ad_constant(p.gravity_internal_scale);
            node_params_ad[i].gravity_surface_scale =
                make_ad_constant(p.gravity_surface_scale);
        }

        const auto& raw_depths = mesher.get_depths();
        for (int i = 0; i < N; ++i) {
            node_depths_ad[i].value() = raw_depths[i];
            const int disc_idx = node_to_discontinuity_idx[i];
            if (disc_idx >= 0 && !fluid_discontinuities[disc_idx]) {
                const int depth_var_idx = NUM_PARAM_VARS + disc_idx;
                node_depths_ad[i].derivatives() =
                    Eigen::VectorXd::Unit(NUM_VARS, depth_var_idx);
            } else {
                node_depths_ad[i].derivatives() = Eigen::VectorXd::Zero(NUM_VARS);
            }
        }
        halfspace_ad = mesher.get_halfspace_ad(node_params_ad);
        ad_model_initialized = true;
    };

    Eigen::VectorXd dc_dp = Eigen::VectorXd::Zero(NUM_VARS);
    Eigen::VectorXd dU_dp = Eigen::VectorXd::Zero(NUM_VARS);
    Eigen::VectorXd dHV_dp = Eigen::VectorXd::Zero(NUM_VARS);
    std::vector<SensitivityKernel> kernels;

    // Per-level kernel evaluator: dc/dp, dU/dp, dHV/dp on the level-k mesh at
    // that mesh's Newton-polished root. Called once per Romberg level so the
    // kernels get Richardson/Neville-extrapolated the same way the reported
    // c/U/HV values are, instead of being frozen at the single converged-level
    // mesh (which leaves an O(h^2) truncation in the derivative).
    auto eval_level = [&](int k, double c_guess_k,
                          Eigen::VectorXd& dc_out, Eigen::VectorXd& dU_out,
                          Eigen::VectorXd& dHV_out) {
        auto [layers_k_dbl, thick_k_dbl] = mesher.get_refined_layers(k);
        auto evaluate_legacy = [&](bool legacy_compute_u,
                                   Eigen::VectorXd& legacy_dc,
                                   Eigen::VectorXd& legacy_du,
                                   Eigen::VectorXd& legacy_dhv) {
            ensure_ad_model();
            std::vector<PhysicalParameters<ADScalar>> layers_k_ad;
            std::vector<ADScalar> thick_k_ad;
            if (model.refinement_policy.is_flat()) {
                std::tie(layers_k_ad, thick_k_ad) =
                    mesher.get_refined_layers_ad_with_depths(
                        k, node_params_ad, node_depths_ad);
            } else {
                std::tie(layers_k_ad, thick_k_ad) =
                    mesher.get_refined_layers_ad(k, node_params_ad);
            }
            compute_grid_level_kernels<ADScalar>(
                layers_k_dbl, thick_k_dbl, halfspace_dbl,
                layers_k_ad, thick_k_ad, halfspace_ad,
                omega, c_guess_k, type, compute_hv, legacy_compute_u,
                u_kernel_mode != "full_ad", NUM_VARS,
                legacy_dc, legacy_du, legacy_dhv, false);
        };

        if (is_frequency_variational_u_kernel_mode(u_kernel_mode) && compute_u) {
            bool frequency_ok = false;
            try {
                const double c_root_k = c_guess_k;
                const bool need_boundary = num_discontinuities > 0 &&
                    model.refinement_policy.is_flat();
                const auto frequency =
                    computeGroupVelocityKernelFrequencyVariationalLevel(
                        layers_k_dbl, thick_k_dbl, halfspace_dbl,
                        omega, c_root_k, type, need_boundary);
                const Eigen::VectorXd node_vti =
                    mesher.apply_sublayer_grad_to_node(
                        k, frequency.group_material_vti, 6);
                const Eigen::VectorXd material =
                    map_node_vti_gradient_to_kernel_order(
                        params, node_vti, sens_mode);
                dU_out = Eigen::VectorXd::Zero(NUM_VARS);
                dU_out.head(NUM_PARAM_VARS) = material;
                if (need_boundary) {
                    for (int d = 0; d < num_discontinuities; ++d) {
                        if (fluid_discontinuities[d]) continue;
                        const auto weights = interfaceBoundaryWeights(
                            is_discontinuity, N,
                            discontinuity_pair_starts[d], k);
                        if (static_cast<int>(weights.size()) <
                            frequency.group_boundary.size() + 1) {
                            throw std::runtime_error(
                                "boundary weight count does not match frequency kernel");
                        }
                        double value = 0.0;
                        for (int b = 0; b < frequency.group_boundary.size(); ++b)
                            value += weights[b + 1] * frequency.group_boundary(b);
                        dU_out(NUM_PARAM_VARS + d) = value;
                    }
                }
                frequency_ok = dU_out.allFinite();
            } catch (const std::exception&) {
                frequency_ok = false;
            }

            if (!frequency_ok) {
                evaluate_legacy(true, dc_out, dU_out, dHV_out);
            } else if (compute_c || compute_hv) {
                Eigen::VectorXd unused_u;
                evaluate_legacy(false, dc_out, unused_u, dHV_out);
            } else {
                dc_out = Eigen::VectorXd::Zero(NUM_VARS);
                dHV_out = Eigen::VectorXd::Zero(NUM_VARS);
            }
        } else {
            evaluate_legacy(compute_u, dc_out, dU_out, dHV_out);
        }
    };

    if (compute_c || compute_hv || compute_u) {
        if (kernel_extrapolation) {
            std::vector<Eigen::VectorXd> dc_tab, dU_tab, dHV_tab;
            for (int k = 0; k <= level; ++k) {
                Eigen::VectorXd dck, dUk, dHVk;
                eval_level(k, roots[static_cast<size_t>(k)],
                           dck, dUk, dHVk);
                dc_tab.push_back(dck); dU_tab.push_back(dUk); dHV_tab.push_back(dHVk);
            }
            auto neville_vec = [&](const std::vector<Eigen::VectorXd>& tab) {
                Eigen::VectorXd out = Eigen::VectorXd::Zero(NUM_VARS);
                for (int idx = 0; idx < NUM_VARS; ++idx) {
                    std::vector<double> col;
                    col.reserve(tab.size());
                    for (const auto& v : tab) col.push_back(v(idx));
                    out(idx) = neville_extrapolation(col).first;
                }
                return out;
            };
            dc_dp = neville_vec(dc_tab);
            dU_dp = neville_vec(dU_tab);
            dHV_dp = neville_vec(dHV_tab);
        } else {
            eval_level(level, roots[static_cast<size_t>(level)],
                       dc_dp, dU_dp, dHV_dp);
        }
        int kernel_var_idx = 0;
        for (int i = 0; i < N; ++i) {
            if (sens_mode == SensitivityMode::Isotropic) {
                const char* param_names[] = {"Vs", "Vp", "Rho"};
                for (int j = 0; j < 3; ++j) {
                    SensitivityKernel k; k.parameterName = param_names[j]; k.layerIndex = i;
                    k.d_PhaseVelocity = compute_c ? dc_dp(kernel_var_idx) : 0.0;
                    k.d_GroupVelocity = compute_u ? dU_dp(kernel_var_idx) : 0.0;
                    k.d_HVRatio       = compute_hv ? dHV_dp(kernel_var_idx) : 0.0;
                    ++kernel_var_idx; kernels.push_back(k);
                }
            } else {
                const char* names[] = {"A", "C", "F", "L", "N", "Rho"};
                for (int j = 0; j < 6; ++j) {
                    SensitivityKernel k; k.parameterName = names[j]; k.layerIndex = i;
                    k.d_PhaseVelocity = compute_c ? dc_dp(kernel_var_idx) : 0.0;
                    k.d_GroupVelocity = compute_u ? dU_dp(kernel_var_idx) : 0.0;
                    k.d_HVRatio       = compute_hv ? dHV_dp(kernel_var_idx) : 0.0;
                    ++kernel_var_idx; kernels.push_back(k);
                }
            }
        }
        for (int k = 0; k < num_discontinuities; ++k) {
            int depth_var_idx = NUM_PARAM_VARS + k;
            int node_idx = -1;
            for (int i = 0; i < N; ++i) if (node_to_discontinuity_idx[i] == k) { node_idx = i; break; }
            SensitivityKernel sk; sk.parameterName = "Depth"; sk.layerIndex = node_idx;
            sk.d_PhaseVelocity = compute_c ? dc_dp(depth_var_idx) : 0.0;
            sk.d_GroupVelocity = compute_u ? dU_dp(depth_var_idx) : 0.0;
            sk.d_HVRatio       = compute_hv ? dHV_dp(depth_var_idx) : 0.0;
            kernels.push_back(sk);
        }
    }
    return std::make_tuple(c, U, HV, level, rel_err, kernels);
}

// --- GRID Mode: analytic interface-depth Hessian column ---
//
// Replaces the central-difference depth column
// (inversion/disper26_inv/depth_hessian.py) with a nested forward-AD pass.
//
// Formulation: docs/depth_sensitivity_kernels.tex section "Interface Depth
// Derivatives (Lagrangian)". The outer AD level carries the single interface
// direction z_k; the inner level carries the node parameters plus one slot for
// c. Because F is defined only up to a normalization factor, only the implicit
// function theorem ratio is meaningful, so we return
//
//     dc/dp        = -F_p / F_c
//     d(dc/dp)/dz  = -(1/F_c) [ dF_p/dz - (F_p/F_c) dF_c/dz ]
//
// where d/dz are TOTAL derivatives along the root: c's outer derivative is
// seeded with dc/dz so the chain rule advects the root. This mirrors the
// cp_seed trick already used by the weak-form Hessian in grid_hessian.hpp.
//
// Returns (dc_dp, d2c_dpdz), both length NUM_VARS in the same variable order
// as compute_dispersion_romberg_with_kernels (node params, then one entry per
// discontinuity). The dc_dp return exists so callers can cross-check against
// the established first-order kernel path for free.
inline std::tuple<std::vector<double>, std::vector<double>>
py_computeDepthHessianColumnGrid(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_root,
    double dc_dz,
    int discontinuity_index,
    int refinement_level,
    std::string sensitivity_mode_str,
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_root);
    }
    WaveType type = (wave_type_str == "Love" || wave_type_str == "love")
                        ? WaveType::Love : WaveType::Rayleigh;
    SensitivityMode sens_mode =
        (sensitivity_mode_str == "VTI" || sensitivity_mode_str == "vti")
            ? SensitivityMode::VTI : SensitivityMode::Isotropic;

    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    if (!model.validate()) {
        throw std::runtime_error("Invalid GridModel: validation failed.");
    }

    Mesher mesher(model);
    const double omega = 2.0 * M_PI / period;

    int num_discontinuities = 0;
    std::vector<int> node_to_discontinuity_idx(mesher.get_node_count(), -1);
    const auto& discon_flags = mesher.get_discontinuities();
    for (size_t i = 0; i < discon_flags.size(); ++i) {
        if (discon_flags[i]) {
            node_to_discontinuity_idx[i] = num_discontinuities;
            if (i + 1 < node_to_discontinuity_idx.size()) {
                node_to_discontinuity_idx[i + 1] = num_discontinuities;
            }
            num_discontinuities++;
        }
    }
    if (discontinuity_index < 0 || discontinuity_index >= num_discontinuities) {
        throw std::invalid_argument("discontinuity_index out of range");
    }

    const int N = mesher.get_node_count();
    const int NUM_PARAMS_PER_NODE = (sens_mode == SensitivityMode::VTI) ? 6 : 3;
    const int NUM_PARAM_VARS = NUM_PARAMS_PER_NODE * N;
    const int NUM_VARS = NUM_PARAM_VARS + num_discontinuities;
    const int C_SLOT = NUM_VARS;          // extra inner slot carrying c
    const int INNER = NUM_VARS + 1;

    // Use the direct loop's branch-confirmed root when supplied.  The legacy
    // standalone path still Newton-polishes an extrapolated c to this mesh.
    {
        auto [layers_p, thick_p] = mesher.get_refined_layers(refinement_level);
        PhysicalParameters<double> hs_p = mesher.get_halfspace();
        if (phase_velocity_roots) {
            c_root = validated_grid_trace_root(
                layers_p, thick_p, hs_p, omega, type, refinement_level,
                *phase_velocity_roots);
        } else {
            auto f_double = [&](double cv) {
                return calculateDispersionFunction<double>(
                    layers_p, thick_p, hs_p, omega, cv, type,
                    NormalizationMode::VectorNorm)(I24);
            };
            double cr = c_root;
            for (int it = 0; it < 15; ++it) {
                double f0 = f_double(cr), epsn = cr * 1e-7;
                double df = (f_double(cr + epsn) - f0) / epsn;
                if (!std::isfinite(f0) || !std::isfinite(df) ||
                    std::abs(df) < 1e-30) break;
                double stp = f0 / df, cand = cr - stp;
                if (!std::isfinite(cand) || cand <= 0.0) break;
                cr = cand;
                if (std::abs(stp) < 1e-11) break;
            }
            if (std::isfinite(cr) && cr > 0.0) c_root = cr;
        }
    }

    using Inner = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    using Outer = Eigen::AutoDiffScalar<Eigen::Matrix<Inner, 1, 1>>;

    auto inner_const = [&](double v) {
        Inner o; o.value() = v; o.derivatives() = Eigen::VectorXd::Zero(INNER); return o;
    };
    auto inner_unit = [&](double v, int slot) {
        Inner o; o.value() = v; o.derivatives() = Eigen::VectorXd::Unit(INNER, slot); return o;
    };
    // Outer scalar with inner value `iv` and outer derivative `dv`.
    auto outer_of = [&](const Inner& iv, const Inner& dv) {
        Outer o; o.value() = iv; o.derivatives()(0) = dv; return o;
    };
    auto outer_const = [&](double v) {
        return outer_of(inner_const(v), inner_const(0.0));
    };

    // --- node parameters: inner slots, no outer (depth) dependence ---
    std::vector<PhysicalParameters<Outer>> node_ad(N);
    int var_idx = 0;
    for (int i = 0; i < N; ++i) {
        const auto& p = params[i];
        const bool is_fluid_node = std::abs(p.c44) < 1e-12;
        auto set_const_block = [&]() {
            node_ad[i].rho = outer_const(p.rho);
            node_ad[i].c11 = outer_const(p.c11);
            node_ad[i].c13 = outer_const(p.c13);
            node_ad[i].c33 = outer_const(p.c33);
            node_ad[i].c44 = outer_const(p.c44);
            node_ad[i].c66 = outer_const(p.c66);
        };
        if (is_fluid_node) {
            set_const_block();
            var_idx += NUM_PARAMS_PER_NODE;
        } else if (sens_mode == SensitivityMode::Isotropic) {
            const double vs = std::sqrt(p.c44 / p.rho);
            const double vp = std::sqrt(p.c11 / p.rho);
            Outer vs_ad = outer_of(inner_unit(vs, var_idx++), inner_const(0.0));
            Outer vp_ad = outer_of(inner_unit(vp, var_idx++), inner_const(0.0));
            Outer rho_ad = outer_of(inner_unit(p.rho, var_idx++), inner_const(0.0));
            node_ad[i].rho = rho_ad;
            node_ad[i].c44 = rho_ad * vs_ad * vs_ad;
            node_ad[i].c66 = node_ad[i].c44;
            node_ad[i].c11 = rho_ad * vp_ad * vp_ad;
            node_ad[i].c33 = node_ad[i].c11;
            node_ad[i].c13 = node_ad[i].c11 - outer_const(2.0) * node_ad[i].c44;
        } else {
            node_ad[i].c11 = outer_of(inner_unit(p.c11, var_idx++), inner_const(0.0));
            node_ad[i].c33 = outer_of(inner_unit(p.c33, var_idx++), inner_const(0.0));
            node_ad[i].c13 = outer_of(inner_unit(p.c13, var_idx++), inner_const(0.0));
            node_ad[i].c44 = outer_of(inner_unit(p.c44, var_idx++), inner_const(0.0));
            node_ad[i].c66 = outer_of(inner_unit(p.c66, var_idx++), inner_const(0.0));
            node_ad[i].rho = outer_of(inner_unit(p.rho, var_idx++), inner_const(0.0));
        }
        node_ad[i].gravity = outer_const(p.gravity);
        node_ad[i].gravity_internal_scale = outer_const(p.gravity_internal_scale);
        node_ad[i].gravity_surface_scale = outer_const(p.gravity_surface_scale);
    }

    // --- node depths: inner slot for their own discontinuity variable, and an
    // outer seed of 1 for the interface actually being differentiated. Both
    // nodes of a duplicated pair share one variable so the interface
    // translates rigidly (see docs/depth_sensitivity_kernels.tex).
    std::vector<Outer> node_depths_ad(N);
    const auto& raw_depths = mesher.get_depths();
    for (int i = 0; i < N; ++i) {
        const int disc_idx = node_to_discontinuity_idx[i];
        bool fluid_discontinuity = false;
        if (disc_idx >= 0) {
            for (int j = 0; j < N; ++j) {
                if (node_to_discontinuity_idx[j] == disc_idx &&
                    std::abs(params[j].c44) < 1e-12) {
                    fluid_discontinuity = true;
                    break;
                }
            }
        }
        const bool active = (disc_idx >= 0 && !fluid_discontinuity);
        Inner iv = active ? inner_unit(raw_depths[i], NUM_PARAM_VARS + disc_idx)
                          : inner_const(raw_depths[i]);
        Inner dv = inner_const((active && disc_idx == discontinuity_index) ? 1.0 : 0.0);
        node_depths_ad[i] = outer_of(iv, dv);
    }

    // c: inner unit in its own slot (so F_c is AD, not a difference), outer
    // derivative dc/dz so the root is advected by the chain rule.
    Outer c_ad = outer_of(inner_unit(c_root, C_SLOT), inner_const(dc_dz));
    Outer omega_ad = outer_const(omega);

    auto [layers_ad, thick_ad] =
        mesher.get_refined_layers_ad_with_depths(refinement_level, node_ad, node_depths_ad);
    PhysicalParameters<Outer> hs_ad = mesher.get_halfspace_ad(node_ad);

    Eigen::Matrix<Outer, 5, 1> seabed_ad; seabed_ad.setZero();
    auto Y_all = calculateDispersionFunction(
        layers_ad, thick_ad, hs_ad, omega_ad, c_ad, type,
        NormalizationMode::Vertical, &seabed_ad);
    Outer F = Y_all(I24);

    const Eigen::VectorXd dF   = F.value().derivatives();          // [F_p..., F_c]
    const Eigen::VectorXd dFdz = F.derivatives()(0).derivatives(); // total d/dz of the above
    const double F_c = dF(C_SLOT);
    const double dF_c_dz = dFdz(C_SLOT);
    if (!std::isfinite(F_c) || std::abs(F_c) < 1e-20) {
        throw std::runtime_error("depth Hessian: dF/dc is non-finite or near zero");
    }

    std::vector<double> dc_dp(NUM_VARS), d2c(NUM_VARS);
    for (int j = 0; j < NUM_VARS; ++j) {
        const double F_p = dF(j);
        dc_dp[j] = -F_p / F_c;
        d2c[j] = -(dFdz(j) - (F_p / F_c) * dF_c_dz) / F_c;
    }
    return std::make_tuple(dc_dp, d2c);
}

// --- Interface Hessian row via AD of the variational boundary kernel ---
//
// Option C. dc/dz_j = sum_b [K_d]_b w_b with the weights purely geometric, so
//
//     d/dp ( dc/dz_j ) = sum_b ( d[K_d]_b/dp ) w_b ,
//
// i.e. one forward-AD pass over computeBoundaryKernels{Rayleigh,Love}_T is
// enough -- no nesting, and no change to computeVariationalGradient*_T's
// output length (which 24 call sites depend on).
//
// As in grid_hessian.hpp, c is seeded with dc/dp so the chain rule carries the
// implicit motion of the root; without it this would be a partial derivative at
// fixed c rather than the Hessian row. The caller supplies dc_dp (the
// first-order kernel it already has) and the boundary weights, keeping the
// parameterization explicit.
//
// Returns d(dc/dz_j)/dp for every node variable, in the variable order of
// compute_dispersion_romberg_with_kernels. Entry NUM_PARAM_VARS + j is the
// diagonal d^2c/dz_j^2.
inline std::vector<double> py_computeInterfaceHessianRowVariational(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_root,
    const std::vector<double>& dc_dp,
    const std::vector<double>& boundary_weights,
    int refinement_level,
    std::string sensitivity_mode_str
) {
    WaveType type = (wave_type_str == "Love" || wave_type_str == "love")
                        ? WaveType::Love : WaveType::Rayleigh;
    SensitivityMode sens_mode =
        (sensitivity_mode_str == "VTI" || sensitivity_mode_str == "vti")
            ? SensitivityMode::VTI : SensitivityMode::Isotropic;

    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");

    Mesher mesher(model);
    const double omega = 2.0 * M_PI / period;

    int num_discontinuities = 0;
    std::vector<int> node_to_disc(mesher.get_node_count(), -1);
    const auto& flags = mesher.get_discontinuities();
    for (size_t i = 0; i < flags.size(); ++i) {
        if (flags[i]) {
            node_to_disc[i] = num_discontinuities;
            if (i + 1 < node_to_disc.size()) node_to_disc[i + 1] = num_discontinuities;
            num_discontinuities++;
        }
    }
    const int N = mesher.get_node_count();
    const int NPN = (sens_mode == SensitivityMode::VTI) ? 6 : 3;
    const int NUM_PARAM_VARS = NPN * N;
    const int NUM_VARS = NUM_PARAM_VARS + num_discontinuities;
    if ((int)dc_dp.size() != NUM_VARS) {
        throw std::invalid_argument("dc_dp must have NUM_VARS entries");
    }

    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto cst = [&](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(NUM_VARS); return a;
    };
    auto unit = [&](double v, int slot) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, slot); return a;
    };

    std::vector<PhysicalParameters<ADS>> node_ad(N);
    int vi = 0;
    for (int i = 0; i < N; ++i) {
        const auto& p = params[i];
        if (std::abs(p.c44) < 1e-12) {
            node_ad[i].rho = cst(p.rho); node_ad[i].c11 = cst(p.c11);
            node_ad[i].c13 = cst(p.c13); node_ad[i].c33 = cst(p.c33);
            node_ad[i].c44 = cst(p.c44); node_ad[i].c66 = cst(p.c66);
            vi += NPN;
        } else if (sens_mode == SensitivityMode::Isotropic) {
            ADS vs = unit(std::sqrt(p.c44 / p.rho), vi++);
            ADS vp = unit(std::sqrt(p.c11 / p.rho), vi++);
            ADS rho = unit(p.rho, vi++);
            node_ad[i].rho = rho;
            node_ad[i].c44 = rho * vs * vs; node_ad[i].c66 = node_ad[i].c44;
            node_ad[i].c11 = rho * vp * vp; node_ad[i].c33 = node_ad[i].c11;
            node_ad[i].c13 = node_ad[i].c11 - cst(2.0) * node_ad[i].c44;
        } else {
            node_ad[i].c11 = unit(p.c11, vi++); node_ad[i].c33 = unit(p.c33, vi++);
            node_ad[i].c13 = unit(p.c13, vi++); node_ad[i].c44 = unit(p.c44, vi++);
            node_ad[i].c66 = unit(p.c66, vi++); node_ad[i].rho = unit(p.rho, vi++);
        }
        node_ad[i].gravity = cst(p.gravity);
        node_ad[i].gravity_internal_scale = cst(p.gravity_internal_scale);
        node_ad[i].gravity_surface_scale = cst(p.gravity_surface_scale);
    }

    std::vector<ADS> node_depths_ad(N);
    const auto& raw_depths = mesher.get_depths();
    for (int i = 0; i < N; ++i) {
        const int d = node_to_disc[i];
        node_depths_ad[i] = (d >= 0) ? unit(raw_depths[i], NUM_PARAM_VARS + d)
                                     : cst(raw_depths[i]);
    }

    ADS c_ad; c_ad.value() = c_root;
    c_ad.derivatives() = Eigen::Map<const Eigen::VectorXd>(dc_dp.data(), NUM_VARS);
    ADS omega_ad = cst(omega);

    auto [layers_ad, thick_ad] =
        mesher.get_refined_layers_ad_with_depths(refinement_level, node_ad, node_depths_ad);
    PhysicalParameters<ADS> hs_ad = mesher.get_halfspace_ad(node_ad);

    Eigen::Matrix<ADS, Eigen::Dynamic, 1> kd =
        (type == WaveType::Love)
            ? computeBoundaryKernelsLove_T<ADS>(layers_ad, thick_ad, hs_ad,
                                                omega_ad, c_ad)
            : computeBoundaryKernelsRayleigh_T<ADS>(layers_ad, thick_ad, hs_ad,
                                                    omega_ad, c_ad, false);
    if ((int)boundary_weights.size() != (int)kd.size()) {
        throw std::invalid_argument(
            "boundary_weights length must equal the number of stack boundaries");
    }

    Eigen::VectorXd row = Eigen::VectorXd::Zero(NUM_VARS);
    for (int b = 0; b < (int)kd.size(); ++b) {
        if (boundary_weights[b] == 0.0) continue;
        row += boundary_weights[b] * kd(b).derivatives();
    }
    return std::vector<double>(row.data(), row.data() + row.size());
}

// dz_b/d(delta) for every boundary of the level-k refined stack, DEPTH
// parameterization. Shifting the duplicated node pair at `pair_start` by delta
// grows each sublayer of the segment ending at the interface by +h/n and
// shrinks each sublayer of the segment starting there by -h/n, so the running
// sum ramps 0 -> 1 through the segment above, is exactly 1 at the interface,
// ramps 1 -> 0 through the segment below, and is 0 deeper. (A THICKNESS
// derivative would instead leave every deeper boundary at 1.)
//
// Boundary b lies between sublayer b-1 and sublayer b; entry 0 is the free
// surface. The returned vector is indexed to match computeBoundaryKernels*_T,
// which emits one jump per stack boundary starting at boundary 1.
inline std::vector<double> interfaceBoundaryWeights(
    const std::vector<bool>& is_discontinuity, int n_nodes, int pair_start,
    int refinement_level
) {
    const int n_sub = 1 << refinement_level;
    std::vector<int> tags;               // +1 above segment, -1 below, 0 neither
    for (int i = 0; i < n_nodes - 1; ++i) {
        if (is_discontinuity[i]) continue;          // duplicated pair: no layer
        const int tag = (i == pair_start - 1) ? 1 : ((i == pair_start + 1) ? -1 : 0);
        tags.insert(tags.end(), n_sub, tag);
    }
    std::vector<double> weights(tags.size() + 1, 0.0);
    int n_above = 0, n_below = 0;
    for (size_t b = 0; b < weights.size(); ++b) {
        weights[b] = double(n_above - n_below) / double(n_sub);
        if (b < tags.size()) {
            if (tags[b] > 0) ++n_above;
            else if (tags[b] < 0) ++n_below;
        }
    }
    return weights;
}

// Romberg-extrapolated interface Hessian row via K_d.
//
// py_computeInterfaceHessianRowVariational evaluates on ONE refined mesh, which
// leaves the O(h^2) truncation that compute_dispersion_romberg_with_kernels
// already removes from the first-order kernels by evaluating every level and
// Neville-extrapolating (see the eval_level loop above). The depth Hessian
// column was the only part of H still frozen at a single level; this closes
// that. Measured: the single-level gap against nested AD falls 4.00x per level
// (pure O(h^2)) and Neville collapses it to ~1e-7 -- see
// tests/REPORT_KD_ACCURACY_20260807.md.
//
// The per-level first-order dc/dp the IFT c-seed needs is recomputed here with
// compute_grid_level_kernels, the same routine the Romberg kernel path calls,
// so each level is seeded from its own root rather than the extrapolated one.
//
// Returns d(dc/dz_j)/dp for every node variable in the variable order of
// compute_dispersion_romberg_with_kernels; entry NUM_PARAM_VARS + j is the
// diagonal d^2c/dz_j^2.
inline std::vector<double> py_computeInterfaceHessianRowVariationalExtrapolated(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_root,
    int discontinuity_index,
    int max_level,
    std::string sensitivity_mode_str,
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_root);
    }
    WaveType type = (wave_type_str == "Love" || wave_type_str == "love")
                        ? WaveType::Love : WaveType::Rayleigh;
    SensitivityMode sens_mode =
        (sensitivity_mode_str == "VTI" || sensitivity_mode_str == "vti")
            ? SensitivityMode::VTI : SensitivityMode::Isotropic;

    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");

    Mesher mesher(model);
    const double omega = 2.0 * M_PI / period;
    const int N = mesher.get_node_count();

    int num_discontinuities = 0, pair_start = -1;
    std::vector<int> node_to_disc(N, -1);
    const auto& flags = mesher.get_discontinuities();
    for (size_t i = 0; i < flags.size(); ++i) {
        if (flags[i]) {
            if (num_discontinuities == discontinuity_index) pair_start = (int)i;
            node_to_disc[i] = num_discontinuities;
            if (i + 1 < node_to_disc.size()) node_to_disc[i + 1] = num_discontinuities;
            num_discontinuities++;
        }
    }
    if (discontinuity_index < 0 || discontinuity_index >= num_discontinuities) {
        throw std::invalid_argument("discontinuity_index out of range");
    }
    if (max_level < 0) throw std::invalid_argument("max_level must be >= 0");

    const int NPN = (sens_mode == SensitivityMode::VTI) ? 6 : 3;
    const int NUM_PARAM_VARS = NPN * N;
    const int NUM_VARS = NUM_PARAM_VARS + num_discontinuities;

    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto cst = [&](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(NUM_VARS); return a;
    };
    auto unit = [&](double v, int slot) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Unit(NUM_VARS, slot); return a;
    };

    // Same seeding as py_computeInterfaceHessianRowVariational and as the
    // Romberg kernel path: node moduli first, then one slot per discontinuity
    // depth. Fluid nodes stay constant.
    std::vector<PhysicalParameters<ADS>> node_ad(N);
    int vi = 0;
    for (int i = 0; i < N; ++i) {
        const auto& p = params[i];
        if (std::abs(p.c44) < 1e-12) {
            node_ad[i].rho = cst(p.rho); node_ad[i].c11 = cst(p.c11);
            node_ad[i].c13 = cst(p.c13); node_ad[i].c33 = cst(p.c33);
            node_ad[i].c44 = cst(p.c44); node_ad[i].c66 = cst(p.c66);
            vi += NPN;
        } else if (sens_mode == SensitivityMode::Isotropic) {
            ADS vs = unit(std::sqrt(p.c44 / p.rho), vi++);
            ADS vp = unit(std::sqrt(p.c11 / p.rho), vi++);
            ADS rho = unit(p.rho, vi++);
            node_ad[i].rho = rho;
            node_ad[i].c44 = rho * vs * vs; node_ad[i].c66 = node_ad[i].c44;
            node_ad[i].c11 = rho * vp * vp; node_ad[i].c33 = node_ad[i].c11;
            node_ad[i].c13 = node_ad[i].c11 - cst(2.0) * node_ad[i].c44;
        } else {
            node_ad[i].c11 = unit(p.c11, vi++); node_ad[i].c33 = unit(p.c33, vi++);
            node_ad[i].c13 = unit(p.c13, vi++); node_ad[i].c44 = unit(p.c44, vi++);
            node_ad[i].c66 = unit(p.c66, vi++); node_ad[i].rho = unit(p.rho, vi++);
        }
        node_ad[i].gravity = cst(p.gravity);
        node_ad[i].gravity_internal_scale = cst(p.gravity_internal_scale);
        node_ad[i].gravity_surface_scale = cst(p.gravity_surface_scale);
    }

    std::vector<ADS> node_depths_ad(N);
    const auto& raw_depths = mesher.get_depths();
    for (int i = 0; i < N; ++i) {
        const int d = node_to_disc[i];
        node_depths_ad[i] = (d >= 0) ? unit(raw_depths[i], NUM_PARAM_VARS + d)
                                     : cst(raw_depths[i]);
    }

    auto halfspace_dbl = mesher.get_halfspace();
    auto halfspace_ad = mesher.get_halfspace_ad(node_ad);

    std::vector<Eigen::VectorXd> rows;
    rows.reserve(max_level + 1);
    for (int k = 0; k <= max_level; ++k) {
        auto [layers_k_dbl, thick_k_dbl] = mesher.get_refined_layers(k);
        auto [layers_k_ad, thick_k_ad] =
            mesher.get_refined_layers_ad_with_depths(k, node_ad, node_depths_ad);

        const double c_k = phase_velocity_roots
            ? validated_grid_trace_root(
                  layers_k_dbl, thick_k_dbl, halfspace_dbl, omega,
                  type, k, *phase_velocity_roots)
            : polish_grid_kernel_root(
                  layers_k_dbl, thick_k_dbl, halfspace_dbl,
                  omega, c_root, type);

        // Level-k first-order kernel. dc_out is the IFT c-seed the K_d AD
        // pass needs. A supplied direct trace disables independent polishing.
        Eigen::VectorXd dc_k, dU_k, dHV_k;
        compute_grid_level_kernels<ADS>(
            layers_k_dbl, thick_k_dbl, halfspace_dbl,
            layers_k_ad, thick_k_ad, halfspace_ad,
            omega, c_k, type, /*compute_hv=*/false, /*compute_u=*/false,
            /*u_variational=*/true, NUM_VARS, dc_k, dU_k, dHV_k,
            /*polish_root=*/false);

        ADS c_ad; c_ad.value() = c_k; c_ad.derivatives() = dc_k;
        ADS omega_ad = cst(omega);
        Eigen::Matrix<ADS, Eigen::Dynamic, 1> kd =
            (type == WaveType::Love)
                ? computeBoundaryKernelsLove_T<ADS>(layers_k_ad, thick_k_ad,
                                                    halfspace_ad, omega_ad, c_ad)
                : computeBoundaryKernelsRayleigh_T<ADS>(layers_k_ad, thick_k_ad,
                                                        halfspace_ad, omega_ad,
                                                        c_ad, false);

        const std::vector<double> w =
            interfaceBoundaryWeights(is_discontinuity, N, pair_start, k);
        if ((int)w.size() < (int)kd.size() + 1) {
            throw std::runtime_error(
                "boundary weight count does not match the level-k stack");
        }
        Eigen::VectorXd row_k = Eigen::VectorXd::Zero(NUM_VARS);
        for (int b = 0; b < (int)kd.size(); ++b) {
            // kd(b) is the jump at stack boundary b+1; weights are indexed by
            // boundary, entry 0 being the free surface.
            const double wb = w[b + 1];
            if (wb == 0.0) continue;
            row_k += wb * kd(b).derivatives();
        }
        rows.push_back(row_k);
    }

    Eigen::VectorXd out = Eigen::VectorXd::Zero(NUM_VARS);
    for (int idx = 0; idx < NUM_VARS; ++idx) {
        std::vector<double> col;
        col.reserve(rows.size());
        for (const auto& v : rows) col.push_back(v(idx));
        out(idx) = neville_extrapolation(col).first;
    }
    return std::vector<double>(out.data(), out.data() + out.size());
}

// First-order interface-depth gradient dc/dz on a GRID mesh, Neville-
// extrapolated over refinement levels.
//
// The same quantity the Jacobian's Depth component carries, reached the way
// py_computeInterfaceHessianRowVariationalExtrapolated reaches its row: K_d on
// each refined mesh, summed against the geometric boundary weights,
// extrapolated across levels. The difference is that first order needs no AD
// at all -- the Hessian row differentiates K_d once more and runs the whole
// stack through AutoDiffScalar, while dc/dz_j = sum_b [K_d]_b w_b only wants
// the values. Pure double, and correspondingly cheap.
//
// This is the entry point a weak-form Jacobian needs. Without it the Depth
// component has to come from the AD Romberg kernel pass at 733 ms against
// about 30 ms for the weak-form gradient, and keeping that pass for one number
// would cancel the reason to switch. See docs/sensitivity_implementation_map.md.
inline double py_computeInterfaceGradientVariationalExtrapolated(
    const std::vector<double>& depths,
    const std::vector<PhysicalParameters<double>>& params,
    const std::vector<bool>& is_discontinuity,
    double period,
    std::string wave_type_str,
    double c_root,
    int discontinuity_index,
    int max_level,
    const std::optional<std::vector<double>>& phase_velocity_roots = std::nullopt,
    int phase_accepted_level = -1
) {
    if (phase_velocity_roots) {
        (void)validate_trace_phase_identity(
            *phase_velocity_roots, phase_accepted_level, c_root);
    }
    WaveType type = (wave_type_str == "Love" || wave_type_str == "love")
                        ? WaveType::Love : WaveType::Rayleigh;

    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");

    Mesher mesher(model);
    const double omega = 2.0 * M_PI / period;
    const int N = mesher.get_node_count();

    int num_discontinuities = 0, pair_start = -1;
    const auto& flags = mesher.get_discontinuities();
    for (size_t i = 0; i < flags.size(); ++i) {
        if (flags[i]) {
            if (num_discontinuities == discontinuity_index) pair_start = (int)i;
            num_discontinuities++;
        }
    }
    if (pair_start < 0) {
        throw std::runtime_error(
            "discontinuity_index is out of range for this model");
    }

    auto halfspace_dbl = mesher.get_halfspace();

    std::vector<double> per_level;
    per_level.reserve(max_level + 1);
    for (int k = 0; k <= max_level; ++k) {
        auto [layers_k_dbl, thick_k_dbl] = mesher.get_refined_layers(k);

        const double c_k = phase_velocity_roots
            ? validated_grid_trace_root(
                  layers_k_dbl, thick_k_dbl, halfspace_dbl, omega,
                  type, k, *phase_velocity_roots)
            : polish_grid_kernel_root(
                  layers_k_dbl, thick_k_dbl, halfspace_dbl,
                  omega, c_root, type);

        Eigen::Matrix<double, Eigen::Dynamic, 1> kd =
            (type == WaveType::Love)
                ? computeBoundaryKernelsLove_T<double>(layers_k_dbl, thick_k_dbl,
                                                       halfspace_dbl, omega, c_k)
                : computeBoundaryKernelsRayleigh_T<double>(layers_k_dbl, thick_k_dbl,
                                                           halfspace_dbl, omega,
                                                           c_k, false);

        const std::vector<double> w =
            interfaceBoundaryWeights(is_discontinuity, N, pair_start, k);
        if ((int)w.size() < (int)kd.size() + 1) {
            throw std::runtime_error(
                "boundary weight count does not match the level-k stack");
        }
        double dc_dz_k = 0.0;
        for (int b = 0; b < (int)kd.size(); ++b) {
            // kd(b) is the jump at stack boundary b+1; weights are indexed by
            // boundary, entry 0 being the free surface.
            const double wb = w[b + 1];
            if (wb == 0.0) continue;
            dc_dz_k += wb * kd(b);
        }
        per_level.push_back(dc_dz_k);
    }

    return neville_extrapolation(per_level).first;
}

// --- Variational boundary kernel K_d, one jump per stack boundary ---
inline std::vector<double> py_computeBoundaryKernelsRayleigh(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double phase_velocity,
    bool use_analytical
) {
    auto v = computeBoundaryKernelsRayleigh_T<double>(
        layers, thicknesses, halfSpace, omega, phase_velocity, use_analytical);
    return std::vector<double>(v.data(), v.data() + v.size());
}

inline std::vector<double> py_computeBoundaryKernelsLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double phase_velocity
) {
    auto v = computeBoundaryKernelsLove_T<double>(
        layers, thicknesses, halfSpace, omega, phase_velocity);
    return std::vector<double>(v.data(), v.data() + v.size());
}


// --- Hessian Wrappers ---
inline LoveBranchPolicy parseLoveBranchPolicy(const std::string& branch_policy_str) {
    if (branch_policy_str == "auto" || branch_policy_str == "Auto") return LoveBranchPolicy::Auto;
    if (branch_policy_str == "sign_only" || branch_policy_str == "sign-only" || branch_policy_str == "sign") return LoveBranchPolicy::SignOnly;
    if (branch_policy_str == "force_evanescent" || branch_policy_str == "evanescent") return LoveBranchPolicy::ForceEvanescent;
    if (branch_policy_str == "force_propagating" || branch_policy_str == "propagating") return LoveBranchPolicy::ForcePropagating;
    if (branch_policy_str == "force_linear" || branch_policy_str == "linear") return LoveBranchPolicy::ForceLinear;
    throw std::runtime_error("invalid Love branch policy: " + branch_policy_str);
}

// Selects the AD backend of the weak-form Hessian. "dual" is the N-pass
// reference; "vector" runs the same math in one vectorized forward pass.
inline WFHessianBackend parseWFHessianBackend(const std::string& backend_str) {
    if (backend_str == "dual" || backend_str == "scalar") return WFHessianBackend::Dual;
    if (backend_str == "vector") return WFHessianBackend::Vector;
    if (backend_str == "vector_dyn") return WFHessianBackend::VectorDyn;
    throw std::runtime_error("invalid weak-form Hessian backend: " + backend_str +
                             " (expected \"dual\", \"vector\" or \"vector_dyn\")");
}

// Deprecated alias. Use compute_dispersion_curve_layer_cake_impl.
inline PhaseVelocityCurveResult compute_dispersion_curve_pancake_impl(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses, const PhysicalParameters<double>& halfspace,
    const std::vector<double>& periods, WaveType type, int branch,
    double c_min, double c_max, int scan_points, double root_tolerance, bool verify_branch) {
    return compute_dispersion_curve_layer_cake_impl(layers,thicknesses,halfspace,periods,
        type,branch,c_min,c_max,scan_points,root_tolerance,verify_branch);
}
