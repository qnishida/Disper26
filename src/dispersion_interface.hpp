#pragma once

// Python-independent adapters shared by language bindings.
#include "disper26.cpp"
#include "romberg_engine.hpp"
#include "velocity_grid.hpp"
#include <optional>
#include <sstream>
#include <utility>

inline int calculateAdModeCount(
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

// --- GRID Mode Romberg Acceleration Wrapper ---
// Thin adapter: builds the shared root/U/HV callbacks for a refiner producing
// VelocityGridRefinement meshes and runs the single Romberg core loop.
inline double find_dispersion_root_by_mode_order(
    const VelocityGridRefinement& grid,
    double omega,
    WaveType type,
    int branch,
    double tolerance,
    double c_guess,
    const std::vector<double>& root_history,
    int* mode_count_evaluations = nullptr
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
        if (mode_count_evaluations) ++*mode_count_evaluations;
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
        const int count_mid = count(mid);
        if (count_mid <= branch) {
            lo = mid;
            count_lo = count_mid;
        } else {
            hi = mid;
            count_hi = count_mid;
        }
    }

    // Expansion and bisection retain the count at each current endpoint.
    // Reuse those exact evaluations for the one-branch certificate.
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
