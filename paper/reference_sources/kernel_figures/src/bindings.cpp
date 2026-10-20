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
#include "velocity_grid.hpp"     // Velocity-basis GRID/Romberg helpers
// Forced Rebuild Check 109

namespace py = pybind11;
using namespace Analytical; // For analytical integration

// --- Helpers ---

// VTI Layer Creation Helper
// Returns (PhysicalParameters, thickness)
std::tuple<PhysicalParameters<double>, double> py_createVTILayerFromDepth(
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
    p.gravity_seabed_scale = 0.0;
    
    double thickness = std::abs(bottom_depth - top_depth);
    return std::make_tuple(p, thickness);
}

double py_mineosPhysicalDispersionFactor(double tref_s, double period_s) {
    if (tref_s <= 0.0) return 0.0;
    if (period_s <= 0.0) {
        throw std::runtime_error("period_s must be positive.");
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    return 2.0 * std::log(tref_s / period_s) / pi;
}

double py_qInverseMineos(double q_value) {
    return (q_value > 0.0) ? 1.0 / q_value : q_value;
}

PhysicalParameters<double> py_applyMineosPhysicalDispersion(
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

std::vector<PhysicalParameters<double>> py_applyMineosPhysicalDispersionModel(
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

WaveType parseWaveType(const std::string& wave_type_str) {
    if (wave_type_str == "Love" || wave_type_str == "love") {
        return WaveType::Love;
    }
    return WaveType::Rayleigh;
}

int calculateAdModeCount(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    WaveType type
) {
    BranModeCountResult info;
    if (type == WaveType::Love) {
        info = calculateBranModeCountLoveSolid(
            layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
    } else {
        info = calculateBranModeCountRayleighSolid(
            layers, thicknesses, halfSpaceParams, omega, phaseVelocity);
    }
    if (info.unsupported_liquid) {
        throw std::runtime_error("calculate_total_mode_count currently does not support liquid Love models or unsupported liquid transitions.");
    }
    if (info.nan_or_radiative) {
        throw std::runtime_error("calculate_total_mode_count encountered a radiative or invalid halfspace state.");
    }
    return info.n_integer;
}

// Primary integer mode count: AD-z bran-style crossing counter.
int py_calculateTotalModeCount(
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
double py_calculateContinuousModeCount(
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

int py_calculateBranModeCount(
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

// Wrapper for Dispersion Function Value (Returning characteristic function)
// For Rayleigh: Returns Y[4] (Stress minor)
// For Love: Returns Y[1] (Stress)
double py_calculateDispersionValue(
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
    
    // Call the function that returns the full vector
    auto result_vec = calculateDispersionFunction(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type);
    
    // Return the characteristic value (secular function)
    if (type == WaveType::Rayleigh) {
        return result_vec(I24);
    } else {
        // Love: Y = [v, tau]
        // Mapped to 5x1 as: Y_full(I12)=v, Y_full(I24)=tau
        return result_vec(I24);
    }
}

// Wrapper for Root Searching
std::vector<double> py_searchRoots(
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
        auto res = calculateDispersionFunction(layers, thicknesses, halfSpaceParams, omega, c, type);
        return res(I24); // Characteristic function
    };
    
    return searchRoots(func, c_min, c_max, step);
}

// Wrapper for debugging: Return full 5-component vector
std::vector<double> py_getRayleighMinors(
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
std::vector<SensitivityKernel> py_computeSensitivityFD(
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
double py_calculateGroupVelocity(
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
// Returns U/V (Horizontal/Vertical Displacement Ratio) at surface
// For Rayleigh waves only. Returns NaN for Love.
double py_calculateHVRatio(
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
        return std::abs(calculateHVRatio(seabedState));
    } else {
        return std::abs(calculateHVRatio(res));
    }
}

// Full AD Sensitivity Kernels Wrapper
std::vector<SensitivityKernel> py_computeSensitivityFullAD(
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
std::vector<SensitivityKernel> py_computeSensitivityWeakForm(
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
double py_calculateGroupVelocityFullAD(
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


PhysicalParameters<double> applyEarthFlatteningScaleBA97(
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
    p_f.gravity_seabed_scale = p_s.gravity_seabed_scale;
    return p_f;
}

PhysicalParameters<double> applyEarthFlatteningScaleExperimentalDensity(
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
    p_f.gravity_seabed_scale = p_s.gravity_seabed_scale;
    return p_f;
}

// Earth Flattening Transformation for layer/pancake models (Bhattacharya & Arora 1997).
std::tuple<std::vector<PhysicalParameters<double>>, std::vector<double>> py_earthFlatteningLayers(
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
std::tuple<std::vector<PhysicalParameters<double>>, std::vector<double>> py_earthFlatteningLayersExperimentalDensity(
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
std::tuple<std::vector<double>, std::vector<PhysicalParameters<double>>, std::vector<bool>> py_earthFlatteningGrid(
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
std::tuple<std::vector<double>, std::vector<PhysicalParameters<double>>, std::vector<bool>> py_earthFlatteningGridExperimentalDensity(
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

// --- GRID Mode Romberg Acceleration Wrapper ---
// Thin adapter: builds the shared root/U/HV callbacks for a refiner producing
// VelocityGridRefinement meshes and runs the single Romberg core loop.
double find_dispersion_root_by_mode_order(
    const VelocityGridRefinement& grid,
    double omega,
    WaveType type,
    int branch,
    double tolerance,
    double c_guess,
    const std::vector<double>& root_history
) {
    if (branch < 0) {
        throw std::invalid_argument("branch must be nonnegative");
    }

    auto shear_speed = [type](const PhysicalParameters<double>& parameter) {
        const double modulus =
            type == WaveType::Love ? parameter.c66 : parameter.c44;
        if (!(parameter.rho > 0.0) || modulus < 0.0
            || (type == WaveType::Love && !(modulus > 0.0))) {
            throw std::runtime_error(
                "branch-indexed Romberg root search encountered an unsupported "
                "density or shear modulus");
        }
        return std::sqrt(modulus / parameter.rho);
    };
    double minimum_speed = shear_speed(grid.halfspace);
    for (const auto& layer : grid.layers) {
        minimum_speed = std::min(minimum_speed, shear_speed(layer));
    }
    const double c_min = std::max(0.05, 0.7 * minimum_speed);
    const double c_max = 0.9995 * shear_speed(grid.halfspace);
    if (!(c_max > c_min)) {
        throw std::runtime_error(
            "branch-indexed Romberg root search has an empty velocity window");
    }

    auto count = [&](double phase_velocity) {
        return calculateAdModeCount(
            grid.layers, grid.thicknesses, grid.halfspace,
            omega, phase_velocity, type);
    };
    const double anchor = std::clamp(c_guess, c_min, c_max);
    double lo = anchor;
    double hi = anchor;
    int count_lo = count(anchor);
    int count_hi = count_lo;
    double half_width = std::max(1.0e-4 * anchor, 1.0e-8);
    if (root_history.size() >= 2) {
        half_width = std::max(
            half_width,
            8.0 * std::abs(
                root_history[root_history.size() - 1]
                - root_history[root_history.size() - 2]));
    }

    for (int attempt = 0;
         attempt < 16 && !(count_lo <= branch && count_hi > branch);
         ++attempt, half_width *= 2.0) {
        if (count_lo > branch) {
            lo = std::max(c_min, anchor - half_width);
            count_lo = count(lo);
        }
        if (count_hi <= branch) {
            hi = std::min(c_max, anchor + half_width);
            count_hi = count(hi);
        }
        if ((lo <= c_min && count_lo > branch)
            || (hi >= c_max && count_hi <= branch)) {
            break;
        }
    }
    if (!(count_lo <= branch && count_hi > branch)) {
        std::ostringstream message;
        message << "mode-order branch " << branch
                << " is outside the Romberg refinement velocity window "
                << "(local bracket counts " << count_lo << "--" << count_hi << ")";
        throw std::runtime_error(message.str());
    }

    const double relative_tolerance = std::max(
        std::min(tolerance, 1.0e-8),
        64.0 * std::numeric_limits<double>::epsilon());
    const double count_bracket_relative_tolerance = std::max(
        1.0e-5, std::sqrt(relative_tolerance));
    const double count_bracket_tolerance =
        count_bracket_relative_tolerance * c_max;
    while (hi - lo > count_bracket_tolerance) {
        const double mid = 0.5 * (lo + hi);
        if (count(mid) <= branch) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    count_lo = count(lo);
    count_hi = count(hi);
    if (count_lo != branch || count_hi != branch + 1) {
        std::ostringstream message;
        message << "mode-order branch " << branch
                << " is ambiguous on the Romberg refinement mesh "
                << "(bracket counts " << count_lo << "--" << count_hi << ")";
        throw std::runtime_error(message.str());
    }

    auto dispersion = [&](double phase_velocity) {
        return calculateDispersionPhaseValueDouble(
            grid.layers, grid.thicknesses, grid.halfspace,
            omega, phase_velocity, type);
    };
    const double f_lo = dispersion(lo);
    const double f_hi = dispersion(hi);
    double best_brent_residual = std::numeric_limits<double>::infinity();
    if (std::isfinite(f_lo) && std::isfinite(f_hi) && f_lo * f_hi <= 0.0) {
        // Brent's velocity-interval tolerance does not by itself guarantee
        // the absolute secular-residual contract used by trace consumers.
        // A steep, otherwise valid Love root can have an accurate phase
        // velocity but |f(c)| > 0.1 when the interval stops at 1e-9 km/s.
        // Preserve the ordinary result when it already satisfies the shared
        // contract; otherwise polish inside the same one-branch count bracket
        // by one decimal decade at a time.  Never move to another branch.
        double brent_tolerance = std::min(tolerance, 1.0e-8);
        const double tolerance_floor =
            64.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(lo), std::abs(hi)});
        for (int polish_attempt = 0; polish_attempt < 8; ++polish_attempt) {
            const auto refined = brents_method_with_values(
                dispersion, lo, hi, f_lo, f_hi, brent_tolerance, 100);
            if (std::isfinite(refined.root) &&
                refined.root >= lo && refined.root <= hi) {
                const double residual = std::abs(refined.function_value);
                if (std::isfinite(residual)) {
                    best_brent_residual = std::min(
                        best_brent_residual, residual);
                    if (residual <= kAdaptiveResidualLimit) {
                        return refined.root;
                    }
                }
            }
            if (brent_tolerance <= tolerance_floor) break;
            const double next_tolerance = std::max(
                tolerance_floor, 0.1 * brent_tolerance);
            if (!(next_tolerance < brent_tolerance)) break;
            brent_tolerance = next_tolerance;
        }
    }
    // Near a tangential Love root the secular function can touch zero without
    // changing sign.  The AD-z count still supplies the one-branch bracket;
    // accept only an endpoint whose residual is far below the established
    // adaptive-scan residual limit (0.1), rather than switching roots.
    constexpr double endpoint_residual_limit = 1.0e-6;
    if (std::isfinite(f_lo) && std::abs(f_lo) <= endpoint_residual_limit
        && (!std::isfinite(f_hi) || std::abs(f_lo) <= std::abs(f_hi))) {
        return lo;
    }
    if (std::isfinite(f_hi) && std::abs(f_hi) <= endpoint_residual_limit) {
        return hi;
    }

    // The integer-count jump and the secular zero can be slightly offset on
    // a finite refinement mesh.  In that case retain the nearest-root
    // candidate only when it lies inside the same narrow branch-confirmation
    // band used by ForwardEngine (3e-4 relative) around this one-branch count
    // bracket.  This handles a tangential/offset Love root without accepting
    // an arbitrary neighboring mode.
    const double candidate = find_dispersion_root_adaptive(
        grid.layers, grid.thicknesses, grid.halfspace, omega, c_guess,
        type, tolerance, root_history, 0.0, c_guess);
    const double candidate_residual = std::abs(dispersion(candidate));
    const double bracket_offset = candidate < lo
        ? lo - candidate
        : (candidate > hi ? candidate - hi : 0.0);
    const double confirmation_tolerance =
        3.0e-4 * std::max(1.0, std::abs(candidate));
    if (std::isfinite(candidate) && candidate > 0.0
        && std::isfinite(candidate_residual)
        && candidate_residual < kAdaptiveResidualLimit
        && bracket_offset <= confirmation_tolerance) {
        return candidate;
    }
    std::ostringstream message;
    message << "mode-order branch " << branch
            << " count bracket does not bracket a finite secular root "
            << "(c=" << lo << "--" << hi
            << ", f=" << f_lo << "--" << f_hi;
    if (std::isfinite(best_brent_residual)) {
        message << ", best Brent residual=" << best_brent_residual;
    }
    message << ")";
    throw std::runtime_error(message.str());
}

template <typename Refiner>
RombergResult compute_dispersion_romberg_result_from_refiner(
    Refiner&& refiner,
    double period,
    std::string wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool phase_only = false,
    bool adaptive_root_scan = true,
    double initial_root_window_frac = 0.0,
    int extrapolate_start_level = 0,
    std::optional<int> branch = std::nullopt,
    int minimum_trace_level = 0,
    std::optional<double> root_tolerance = std::nullopt
) {
    WaveType type = WaveType::Rayleigh;
    if (wave_type_str == "Love" || wave_type_str == "love") type = WaveType::Love;

    const double effective_root_tolerance = root_tolerance.value_or(tolerance);
    if (root_tolerance.has_value() &&
        (!(effective_root_tolerance > 0.0) ||
         !std::isfinite(effective_root_tolerance))) {
        throw std::invalid_argument(
            "root_tolerance must be finite and positive");
    }

    auto find_root = [type, effective_root_tolerance, adaptive_root_scan,
                      initial_root_window_frac,
                      initial_branch_anchor = c_guess, branch](
        const VelocityGridRefinement& grid,
        double omega_val,
        double c_guess_val,
        const std::vector<double>& c_history
    ) -> double {
        if (branch.has_value()) {
            return find_dispersion_root_by_mode_order(
                grid, omega_val, type, *branch, effective_root_tolerance,
                c_guess_val, c_history);
        }
        if (adaptive_root_scan) {
            return find_dispersion_root_adaptive(
                grid.layers, grid.thicknesses, grid.halfspace, omega_val, c_guess_val,
                type, effective_root_tolerance, c_history,
                initial_root_window_frac,
                initial_branch_anchor);
        }
        return find_dispersion_root_near_by_scan(
            grid.layers, grid.thicknesses, grid.halfspace, omega_val, c_guess_val, type,
            effective_root_tolerance);
    };

    auto calc_group_vel = [type](
        const VelocityGridRefinement& grid,
        double omega_val,
        double c_val
    ) -> double {
        return calculateGroupVelocityFullAD(grid.layers, grid.thicknesses, grid.halfspace, omega_val, c_val, type);
    };

    auto calc_hv = [type](
        const VelocityGridRefinement& grid,
        double omega_val,
        double c_val
    ) -> double {
        if (type == WaveType::Love) return std::numeric_limits<double>::quiet_NaN();
        Eigen::Matrix<double, 5, 1> seabedState;
        seabedState.setZero();
        auto res = calculateDispersionFunction(
            grid.layers, grid.thicknesses, grid.halfspace, omega_val, c_val, type,
            NormalizationMode::Vertical, &seabedState
        );
        bool topIsLiquid = (!grid.layers.empty() && isLiquidLayer(grid.layers[0]));
        return topIsLiquid ? std::abs(calculateHVRatio(seabedState)) : std::abs(calculateHVRatio(res));
    };

    RombergResult result = compute_with_romberg_core(
        refiner, period, c_guess,
        find_root, calc_group_vel, calc_hv,
        max_steps, tolerance, phase_only, extrapolate_start_level,
        minimum_trace_level
    );

    return result;
}

template <typename Refiner>
std::tuple<double, double, double, int, double> compute_dispersion_romberg_from_refiner(
    Refiner&& refiner,
    double period,
    std::string wave_type_str,
    double c_guess,
    int max_steps,
    double tolerance,
    bool phase_only = false,
    bool adaptive_root_scan = true,
    double initial_root_window_frac = 0.0,
    int extrapolate_start_level = 0,
    std::optional<int> branch = std::nullopt
) {
    RombergResult result = compute_dispersion_romberg_result_from_refiner(
        std::forward<Refiner>(refiner), period, std::move(wave_type_str),
        c_guess, max_steps, tolerance, phase_only, adaptive_root_scan,
        initial_root_window_frac, extrapolate_start_level, branch, 0);

    return std::make_tuple(
        result.phase_velocity,
        result.group_velocity,
        result.hv_ratio,
        result.converged_level,
        result.relative_error
    );
}

double validate_trace_phase_identity(
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
RombergResult evaluate_dispersion_at_romberg_roots(
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
            hv_table.push_back(std::abs(
                top_is_liquid ? calculateHVRatio(seabed_state)
                              : calculateHVRatio(surface_state)));
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
FrequencyCurvatureRombergResult compute_frequency_curvature_ad_grid_from_refiner(
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
    if (!model.validate()) {
        throw std::runtime_error("Invalid GridModel: validation failed.");
    }

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

std::tuple<double, double, double, int, double> py_computeDispersionRomberg(
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

std::tuple<double, double, double, int, double> py_computeDispersionRombergVelocityIso(
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

std::tuple<double, double, double, int, double> py_computeDispersionRombergVelocityVti(
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

void validate_phase_velocity_curve_periods(const std::vector<double>& periods) {
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

double predict_phase_velocity_curve_root(
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
std::optional<double> try_phase_velocity_root_bracket(
    Func&& func,
    double centre,
    double minimum_velocity,
    double maximum_velocity,
    double initial_half_width,
    double maximum_half_width,
    double root_tolerance,
    double unique_scan_step = 0.0
) {
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
            auto roots = searchRoots(
                func, a, b, (b - a) / points, root_tolerance);
            if (roots.size() == 1) return roots.front();
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

template <typename Func>
double scan_phase_velocity_curve_branch(
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
        message << "pancake branch " << branch
                << " was not found in the velocity interval ["
                << c_min << ", " << c_max << "]";
        throw std::runtime_error(message.str());
    }
    return roots[static_cast<size_t>(branch)];
}

PhaseVelocityCurveResult compute_dispersion_curve_pancake_impl(
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

PhaseVelocityCurveResult compute_dispersion_curve_romberg_grid_velocity_vti_impl(
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

    std::vector<std::optional<VelocityGridRefinement>> mesh_cache(
        static_cast<size_t>(max_steps) + 1);
    auto refiner = [&](int level) -> const VelocityGridRefinement& {
        auto& cached = mesh_cache.at(static_cast<size_t>(level));
        if (!cached.has_value()) {
            cached = refine_velocity_grid_vti(
                depths, rhos, vphs, vpvs, vshs, vsvs, etas,
                is_discontinuity, level, refinement_policy);
        }
        return *cached;
    };

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
                const double minimum_half_width =
                    1.0e-4 * predicted;
                double half_width = minimum_half_width;
                const double previous_error =
                    level_histories[level].previous_prediction_error;
                if (std::isfinite(previous_error)) {
                    half_width = std::max(
                        half_width, 8.0 * previous_error);
                }
                const double maximum_half_width = 1.0e-2 * predicted;
                half_width = std::min(half_width, maximum_half_width);
                auto dispersion = [&](double phase_velocity) {
                    return calculateDispersionPhaseValueDouble(
                        grid.layers, grid.thicknesses, grid.halfspace,
                        omega, phase_velocity, type);
                };
                // A direct bracket no wider than one legacy scan interval
                // cannot newly hide a pair that the scalar scan resolved.
                // Wider expansions use an explicit >=8-point uniqueness scan.
                const double legacy_step =
                    2.0 * kLegacyScanHalfWidthFrac * predicted /
                    kLegacyScanPoints;
                const auto local = try_phase_velocity_root_bracket(
                    dispersion, predicted, 0.1,
                    predicted + maximum_half_width,
                    half_width, maximum_half_width,
                    root_tolerance, legacy_step);
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

py::dict py_computeDispersionCurvePancake(
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
        result = compute_dispersion_curve_pancake_impl(
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

using VectorADScalar = Eigen::AutoDiffScalar<Eigen::VectorXd>;

VectorADScalar make_vector_ad(double value, int num_vars, int active_idx = -1) {
    VectorADScalar out;
    out.value() = value;
    out.derivatives() = Eigen::VectorXd::Zero(num_vars);
    if (active_idx >= 0) out.derivatives()(active_idx) = 1.0;
    return out;
}

bool is_valid_u_kernel_mode(const std::string& mode) {
    return mode == "variational_frequency_ad" ||
           mode == "variational" || mode == "full_ad";
}

bool is_frequency_variational_u_kernel_mode(const std::string& mode) {
    return mode == "variational_frequency_ad";
}

// Apply the transpose of the velocity-grid refinement Jacobian without ever
// materializing its dense (refined-parameter x node-parameter) form.  Each
// refined layer depends on only its two parent nodes, so this is O(L + P).
Eigen::VectorXd map_vti_gradient_to_velocity_iso_nodes(
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

Eigen::VectorXd map_vti_gradient_to_velocity_vti_nodes(
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

static std::vector<double> interfaceBoundaryWeights(
    const std::vector<bool>& is_discontinuity, int n_nodes, int pair_start,
    int refinement_level);

template <typename RefinerDouble, typename RefinerAD, typename FrequencyMapper>
std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
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

std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
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

std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
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

Eigen::VectorXd map_node_vti_gradient_to_kernel_order(
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
std::tuple<double, double, double, int, double, std::vector<SensitivityKernel>>
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
            node_params_ad[i].gravity_seabed_scale =
                make_ad_constant(p.gravity_seabed_scale);
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
std::tuple<std::vector<double>, std::vector<double>>
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
        node_ad[i].gravity_seabed_scale = outer_const(p.gravity_seabed_scale);
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
std::vector<double> py_computeInterfaceHessianRowVariational(
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
        node_ad[i].gravity_seabed_scale = cst(p.gravity_seabed_scale);
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
static std::vector<double> interfaceBoundaryWeights(
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
std::vector<double> py_computeInterfaceHessianRowVariationalExtrapolated(
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
        node_ad[i].gravity_seabed_scale = cst(p.gravity_seabed_scale);
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
double py_computeInterfaceGradientVariationalExtrapolated(
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
std::vector<double> py_computeBoundaryKernelsRayleigh(
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

std::vector<double> py_computeBoundaryKernelsLove(
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
    py::dict out;
    const double zero = 0.0;
    const double k = omega / phaseVelocity;
    const int N = static_cast<int>(layers.size());

    std::vector<RayleighStatePoint<double>> states;
    RayleighEnergyResult<double> energy;
    if ((hasLiquidLayer(layers) || isLiquidLayer(halfSpaceParams)) && useAnalytical) {
        throw std::runtime_error("Analytical Rayleigh weak-form trace is not implemented for fluid layers; use trapezoidal trace.");
    }
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
    Eigen::VectorXcd res = propagate_romberg_vector(
        y_top, p_top_layer, p_bot_layer, full_layer_thickness, target_depth, 
        omega, phaseVelocity, type, max_depth, tolerance
    );
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
    const std::string& mode, // "GRID" or "PANCAKE"
    int max_depth,
    double tolerance
) {
    auto buf = y_top_arr.request();
    Eigen::VectorXcd y_top = Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf.ptr, buf.size);
    WaveType type = (wave_type == "Love") ? WaveType::Love : WaveType::Rayleigh;
    bool use_romberg = (mode == "GRID");
    Eigen::VectorXcd res = computeStateAtDepth(
        y_top, p_top_layer, p_bot_layer, full_layer_thickness, target_depth,
        omega, phaseVelocity, type, use_romberg, max_depth, tolerance
    );
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
    const std::string& mode, // "GRID" or "PANCAKE"
    int max_depth,
    double tolerance
) {
    auto buf = y_initial_arr.request();
    Eigen::VectorXcd y_curr = Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf.ptr, buf.size);
    WaveType type = (wave_type == "Love") ? WaveType::Love : WaveType::Rayleigh;
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
    py::array_t<std::complex<double>> result(y_curr.size());
    auto buf2 = result.request();
    Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)buf2.ptr, y_curr.size()) = y_curr;
    return result;
}

// --- Hessian Wrappers ---
static LoveBranchPolicy parseLoveBranchPolicy(const std::string& branch_policy_str) {
    if (branch_policy_str == "auto" || branch_policy_str == "Auto") return LoveBranchPolicy::Auto;
    if (branch_policy_str == "sign_only" || branch_policy_str == "sign-only" || branch_policy_str == "sign") return LoveBranchPolicy::SignOnly;
    if (branch_policy_str == "force_evanescent" || branch_policy_str == "evanescent") return LoveBranchPolicy::ForceEvanescent;
    if (branch_policy_str == "force_propagating" || branch_policy_str == "propagating") return LoveBranchPolicy::ForcePropagating;
    if (branch_policy_str == "force_linear" || branch_policy_str == "linear") return LoveBranchPolicy::ForceLinear;
    throw std::runtime_error("invalid Love branch policy: " + branch_policy_str);
}

// Selects the AD backend of the weak-form Hessian. "dual" is the N-pass
// reference; "vector" runs the same math in one vectorized forward pass.
static WFHessianBackend parseWFHessianBackend(const std::string& backend_str) {
    if (backend_str == "dual" || backend_str == "scalar") return WFHessianBackend::Dual;
    if (backend_str == "vector") return WFHessianBackend::Vector;
    if (backend_str == "vector_dyn") return WFHessianBackend::VectorDyn;
    throw std::runtime_error("invalid weak-form Hessian backend: " + backend_str +
                             " (expected \"dual\", \"vector\" or \"vector_dyn\")");
}

py::dict py_computePhaseVelocityHessian(
    std::vector<PhysicalParameters<double>> layers,
    std::vector<double> thicknesses,
    PhysicalParameters<double> hs,
    double omega, double cp,
    std::string wave_type_str, std::string mode_str,
    bool compute_thickness
) {
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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(
        grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");
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
    if (!model.validate()) {
        throw std::runtime_error("Invalid GridModel");
    }

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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) {
        throw std::runtime_error("Invalid GridModel");
    }

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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");

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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");

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
// docs/Weakform_AD_Hessian.md and computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced.
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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");
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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");
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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");
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
    GridModel model;
    model.depths = depths;
    model.params = params;
    model.is_discontinuity = is_discontinuity;
    model.refinement_policy = make_grid_refinement_policy(grid_policy, planet_radius, flattening_exponent);
    if (!model.validate()) throw std::runtime_error("Invalid GridModel");

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
        .def_readwrite("gravity_seabed_scale", &PhysicalParameters<double>::gravity_seabed_scale)
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
    m.def("create_fluid_parameters", [](double vp, double rho, double gravity, double internal_scale, double surface_scale, double seabed_scale) {
        PhysicalParameters<double> p; p.rho = rho; p.c44 = 0.0; p.c11 = rho * vp * vp; p.c33 = p.c11; p.c13 = p.c11; p.c66 = 0.0; p.gravity = gravity; p.gravity_internal_scale = internal_scale; p.gravity_surface_scale = surface_scale; p.gravity_seabed_scale = seabed_scale;
        return p;
    }, py::arg("vp"), py::arg("rho"), py::arg("gravity") = 0.0, py::arg("internal_scale") = 1.0, py::arg("surface_scale") = 0.0, py::arg("seabed_scale") = 0.0);
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
          "Legacy continuous phase-accumulation mode count for debugging.");
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
          "Crossing-by-crossing detail for the legacy phase-accumulation counter. Its integer "
          "is gone; what remains is the continuous phase, which is a useful smooth diagnostic.");
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
          "Crossing-by-crossing detail behind compute_mode_count.");
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
    m.def("compute_kernels_fd_pancake", &py_computeSensitivityFD, py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"), py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false);
    m.def("compute_sensitivity_kernels_fd", &py_computeSensitivityFD, py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"), py::arg("omega"), py::arg("phase_velocity"), py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_kernels_fd_pancake.");
    m.def("get_rayleigh_minors", &py_getRayleighMinors);
    m.def("compute_group_velocity_ad", &py_calculateGroupVelocityFullAD);
    m.def("calculate_group_velocity", &py_calculateGroupVelocity,
          "Deprecated alias. Use compute_group_velocity_ad.");
    m.def("compute_hv_ratio", &py_calculateHVRatio);
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
          "Transform a spherical layer/pancake model to a flat layer model using inverse Bhattacharya-Arora P-SV scaling: solid elastic moduli scale as (r/R)^(-p), solid density as (r/R)^(2-p); fluid density is fixed and fluid C scales as (r/R)^(-2).");
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
    m.def("compute_kernels_ad_pancake", &py_computeSensitivityFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          "PANCAKE-mode sensitivity kernels (dc/dp, dU/dp, dHV/dp) on the given fixed layers via full "
          "(nested) AD through the propagator. Returns list[SensitivityKernel]. No Romberg extrapolation "
          "(fixed mesh). For the GRID/node-based version see compute_kernels_ad_grid; "
          "for the faster variational pancake version see compute_kernels_variational_pancake.");
    m.def("compute_sensitivity_kernels_full_ad", &py_computeSensitivityFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_kernels_ad_pancake.");
    m.def("compute_kernels_variational_pancake", &py_computeSensitivityWeakForm,
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
    m.def("compute_sensitivity_kernels_weakform", &py_computeSensitivityWeakForm,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("wave_type") = "Rayleigh", py::arg("sensitivity_mode") = "Isotropic",
          py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false,
          py::arg("u_kernel_mode") = "variational_frequency_ad",
          py::arg("compute_c") = true, py::arg("want_kernel_u") = true,
          py::arg("want_kernel_hv") = true,
          "Deprecated alias. Use compute_kernels_variational_pancake.");
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
    m.def("compute_dispersion_curve_pancake", &py_computeDispersionCurvePancake,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("periods"), py::arg("wave_type") = "Rayleigh",
          py::arg("branch") = 0, py::arg("c_min") = 0.1,
          py::arg("c_max") = 4.5, py::arg("scan_points") = 2400,
          py::arg("root_tolerance") = 1e-12,
          py::arg("verify_branch") = false,
          "Compute a strictly monotonic PANCAKE phase-velocity period curve "
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
          py::arg("phase_accepted_level") = -1,
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
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_kernels_ad_grid.");
    m.def("compute_interface_hessian_row_ad_grid", &py_computeDepthHessianColumnGrid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"), py::arg("period"),
          py::arg("wave_type"), py::arg("c_root"), py::arg("dc_dz"),
          py::arg("discontinuity_index"), py::arg("refinement_level"),
          py::arg("sensitivity_mode") = "VTI",
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
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
          "Formulation: docs/depth_sensitivity_kernels.tex, section 'Interface Depth Derivatives'.");
    m.def("compute_depth_hessian_column_grid", &py_computeDepthHessianColumnGrid,
          py::arg("depths"), py::arg("params"), py::arg("is_discontinuity"), py::arg("period"),
          py::arg("wave_type"), py::arg("c_root"), py::arg("dc_dz"),
          py::arg("discontinuity_index"), py::arg("refinement_level"),
          py::arg("sensitivity_mode") = "VTI",
          py::arg("phase_velocity_roots") = py::none(),
          py::arg("phase_accepted_level") = -1,
          "Deprecated alias. Use compute_interface_hessian_row_ad_grid.");
    m.def("compute_interface_kernels_variational_rayleigh_pancake", &py_computeBoundaryKernelsRayleigh,
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
          "Formulation: docs/depth_sensitivity_kernels.tex, sections 'The variational\n"
          "boundary kernel K_d' and 'K_d must be summed over every boundary that moves'.");
    m.def("compute_boundary_kernels_rayleigh", &py_computeBoundaryKernelsRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("use_analytical") = false,
          "Deprecated alias. Use compute_interface_kernels_variational_rayleigh_pancake.");
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
    m.def("compute_interface_kernels_variational_love_pancake", &py_computeBoundaryKernelsLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          "Love variational boundary kernel [K_d]_b, one jump per stack boundary.\n"
          "Same contract as compute_interface_kernels_variational_rayleigh_pancake: weights are the caller's.\n"
          "K_d = -Lambda (rho w^2 y1^2 + y2^2/L - k^2 N y1^2) with y1=u_y, y2=tau_yz.\n"
          "The PLUS on y2^2/L is the spatial-Hamiltonian form; the Lagrangian density\n"
          "T - V has a minus there and near-cancels, suppressing Love by ~50x.");
    m.def("compute_boundary_kernels_love", &py_computeBoundaryKernelsLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("half_space"),
          py::arg("omega"), py::arg("phase_velocity"),
          "Deprecated alias. Use compute_interface_kernels_variational_love_pancake.");
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
    m.def("debug_kernels_variational_rayleigh_pancake", &py_debugRayleighWeakformTrace,
          py::arg("layers"), py::arg("thicknesses"), py::arg("halfspace"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false);
    m.def("debug_rayleigh_weakform_trace", &py_debugRayleighWeakformTrace,
          py::arg("layers"), py::arg("thicknesses"), py::arg("halfspace"),
          py::arg("omega"), py::arg("phase_velocity"),
          py::arg("num_sub_layers") = 10, py::arg("use_analytical") = false,
          "Deprecated alias. Use debug_kernels_variational_rayleigh_pancake.");
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
    m.def("compute_hessian_ad_pancake", &py_computePhaseVelocityHessianFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "PANCAKE-mode full-AD Hessian d2c/dp2 via nested second-order AD (HessianAD<N>, O(L*N^2)). "
          "Fixed mesh (no Romberg). For the lighter weak-form version see "
          "compute_hessian_variational_ad_rayleigh_pancake/_love; for GRID see the *_grid variants. "
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
          "Deprecated alias. Use compute_hessian_ad_pancake.");
    m.def("compute_phase_velocity_hessian_full_ad", &py_computePhaseVelocityHessianFullAD,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_hessian_ad_pancake.");
    m.def("compute_phase_velocity_hessian_reference", &py_computePhaseVelocityHessianRef,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("wave_type"), py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          "Deprecated alias. Use compute_hessian_ad_pancake.");
    m.def("compute_hessian_variational_fd_love_pancake", &py_computePhaseVelocityHessianVariationalFDLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 200, py::arg("fd_eps") = 1e-5);
    m.def("compute_phase_velocity_hessian_variational_fd_love", &py_computePhaseVelocityHessianVariationalFDLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("num_sub_layers") = 200, py::arg("fd_eps") = 1e-5,
          "Deprecated alias. Use compute_hessian_variational_fd_love_pancake.");
    m.def("compute_hessian_variational_ad_love_pancake", &py_computePhaseVelocityHessianWeakFormLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("use_analytical") = true, py::arg("branch_policy") = "auto",
          py::arg("backend") = "dual");
    m.def("compute_phase_velocity_hessian_weakform_love", &py_computePhaseVelocityHessianWeakFormLove,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false,
          py::arg("use_analytical") = true, py::arg("branch_policy") = "auto",
          py::arg("backend") = "dual",
          "Deprecated alias. Use compute_hessian_variational_ad_love_pancake.");
    m.def("debug_hessian_variational_ad_love_pancake", &py_debugLoveWeakformHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{},
          py::arg("cp_grad_multiplier") = 1.0);
    m.def("debug_love_weakform_hessian_breakdown", &py_debugLoveWeakformHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{},
          py::arg("cp_grad_multiplier") = 1.0,
          "Deprecated alias. Use debug_hessian_variational_ad_love_pancake.");
    m.def("debug_gradient_variational_love_pancake", &py_debugLoveWeakformGradientPieces,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{});
    m.def("debug_love_weakform_gradient_pieces", &py_debugLoveWeakformGradientPieces,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("use_analytical") = true,
          py::arg("branch_policy") = "auto", py::arg("num_sub_layers") = 200,
          py::arg("provided_scales") = std::vector<double>{},
          "Deprecated alias. Use debug_gradient_variational_love_pancake.");
    m.def("debug_layer_integrals_love_pancake", &py_debugLoveLayerIntegralProbe,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto");
    m.def("debug_love_layer_integral_probe", &py_debugLoveLayerIntegralProbe,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto",
          "Deprecated alias. Use debug_layer_integrals_love_pancake.");
    m.def("debug_layer_integrals_hessian_love_pancake", &py_debugLoveLayerIntegralHessian,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto");
    m.def("debug_love_layer_integral_hessian", &py_debugLoveLayerIntegralHessian,
          py::arg("layer"), py::arg("thickness"), py::arg("top_u"), py::arg("top_tau"),
          py::arg("omega"), py::arg("cp"), py::arg("branch_policy") = "auto",
          "Deprecated alias. Use debug_layer_integrals_hessian_love_pancake.");
    m.def("debug_hessian_ad_love_upward_ift_pancake", &py_debugLoveUpwardIFTHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic");
    m.def("debug_love_upward_ift_hessian_breakdown", &py_debugLoveUpwardIFTHessianBreakdown,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic",
          "Deprecated alias. Use debug_hessian_ad_love_upward_ift_pancake.");
    m.def("compute_hessian_variational_ad_rayleigh_pancake", &py_computePhaseVelocityHessianWeakFormRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false, py::arg("num_sub_layers") = 10,
          py::arg("use_analytical") = false, py::arg("backend") = "dual");
    m.def("compute_phase_velocity_hessian_weakform_rayleigh", &py_computePhaseVelocityHessianWeakFormRayleigh,
          py::arg("layers"), py::arg("thicknesses"), py::arg("hs"), py::arg("omega"), py::arg("cp"),
          py::arg("sensitivity_mode") = "Isotropic", py::arg("compute_thickness") = false, py::arg("num_sub_layers") = 10,
          py::arg("use_analytical") = false, py::arg("backend") = "dual",
          "Deprecated alias. Use compute_hessian_variational_ad_rayleigh_pancake.");

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
          "(variational) method, Neville-extrapolated across Romberg levels (this is NOT full-AD; "
          "full-AD HessianAD is pancake-only). Note freq [Hz], not period.\n"
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
          "correction). See docs/Weakform_AD_Hessian.md.");
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
