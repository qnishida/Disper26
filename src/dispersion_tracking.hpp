#pragma once
#include <array>
// Included after api_backend.hpp: reuses its predictor, bracket and count solver.
namespace disper26 { namespace api { namespace detail {

inline bool modern_tracking(const Options& o) {
    return o.verify.has_value() || o.c_range.has_value() || o.on_violation.has_value();
}
inline std::string verification_policy(const Options& o) {
    if (o.verify) return *o.verify;
    if (o.verify_mode.value_or(false) ||
        (o.method != "continuation" && o.method != "continuation_recover")) return "every";
    return o.method == "continuation_recover" ? "initial" : "none";
}
inline double tracking_speed_cap(const VelocityGridRefinement& mesh, WaveType wave) {
    return 0.9995 * std::sqrt((wave == WaveType::Love ? mesh.halfspace.c66 :
        mesh.halfspace.c44) / mesh.halfspace.rho);
}

// Certify an unextrapolated root on its own refinement mesh.
inline bool certify_tracking_root(const VelocityGridRefinement& mesh, double omega,
        WaveType wave, int mode, double root, double tolerance, int& evaluations) {
    const double upper = tracking_speed_cap(mesh, wave);
    const double relative = std::max(std::min(tolerance,1.0e-8),
        64.0*std::numeric_limits<double>::epsilon());
    const double half_width = 0.5*upper*std::max(1.0e-5,std::sqrt(relative));
    const double lo = std::max(0.05,root-half_width);
    const double hi = std::min(upper,root+half_width);
    auto secular = [&](double c) {return calculateDispersionPhaseValueDouble(
        mesh.layers,mesh.thicknesses,mesh.halfspace,omega,c,wave);};
    if (!(lo < root && root < hi)) return false;
    const double f_lo = secular(lo), f_hi = secular(hi);
    if (!std::isfinite(f_lo) || !std::isfinite(f_hi) ||
        !(f_lo == 0.0 || f_hi == 0.0 || std::signbit(f_lo) != std::signbit(f_hi))) return false;
    ++evaluations;
    const int count_lo = calculateAdModeCount(mesh.layers,mesh.thicknesses,
        mesh.halfspace,omega,lo,wave);
    ++evaluations;
    const int count_hi = calculateAdModeCount(mesh.layers,mesh.thicknesses,
        mesh.halfspace,omega,hi,wave);
    return count_lo == mode && count_hi == mode+1;
}

struct DispersionTracking {
    std::array<PhaseVelocityRootHistory, 13> histories;
    std::array<double, 13> slopes;
    std::array<double, 13> coarse_offsets;
    double next_period = missing_value; // Set only by curve calls, after deduplication.
    bool initial_period = true;
    bool check_endpoint = true;

    DispersionTracking() {
        slopes.fill(missing_value);
        coarse_offsets.fill(missing_value);
    }

    double root(const VelocityGridRefinement& mesh, double period, WaveType wave,
                int mode, const Options& options, double guess,
                const std::vector<double>& within, Diagnostics& diagnostics) {
        const size_t level = within.size();
        auto& history = histories.at(level);
        const bool has_history = !history.roots.empty();
        const double omega = 2*M_PI/period;
        const auto policy = verification_policy(options);
        const bool recover = policy == "initial";
        const bool verify = policy == "every" || (policy == "endpoints" && check_endpoint);
        const bool modern = modern_tracking(options);
        const double cap = tracking_speed_cap(mesh,wave);
        double predicted = guess;
        std::string predictor = "coarse_level";
        // Checked auto solves always use the current coarse root at fine levels.
        // Select it before computing cross-period candidates that would be discarded.
        if (!(verify && options.predictor == "auto" && !within.empty())) {
            predicted = predict_phase_velocity_curve_root(history, period);
            predictor = has_history ? "history" : "initial";
            if (has_history && options.predictor != "history" && std::isfinite(slopes[level])) {
                const double previous_omega = 2*M_PI/history.periods.back();
                // Do not extrapolate an AD tangent over a large frequency jump.
                if (std::abs(omega-previous_omega) <= 0.25*previous_omega) {
                    const double tangent = history.roots.back()+slopes[level]*(omega-previous_omega);
                    if (std::isfinite(tangent) && tangent > 0) {
                        predicted = tangent;
                        predictor = "group_velocity";
                    }
                }
            }
            if (!within.empty() && std::isfinite(coarse_offsets[level])) {
                const double candidate = within.back()+coarse_offsets[level];
                if (!std::isfinite(predicted) ||
                    (std::isfinite(history.previous_prediction_error) &&
                     std::abs(coarse_offsets[level]) < history.previous_prediction_error)) {
                    predicted = candidate;
                    predictor = "coarse_level";
                }
            }
            if (!std::isfinite(predicted) || predicted <= 0) {
                predicted = guess;
                predictor = "initial";
            }
        }
        bool checked = verify || (recover && !has_history);
        double result = missing_value;
        std::string failure;
        auto mode_order = [&]() {
            checked = true;
            const double root = find_dispersion_root_by_mode_order(mesh,omega,wave,mode,
                options.root_tolerance,predicted,within,&diagnostics.mode_count_evaluations);
            if (!(root > 0.0 && root < cap))
                throw Error("root_not_found","No secular root below the trapped-wave speed cap");
            return root;
        };
        auto violation = [&](const std::string& reason) {
            ++diagnostics.fallback_count;
            diagnostics.fallback_reasons.push_back(reason);
            if (options.on_violation.value_or("fallback") != "fallback")
                throw Error("tracking_violation",reason);
            return mode_order();
        };
        const bool supplied_range = initial_period && options.c_range.has_value();
        if (supplied_range) {
            const double lo = options.c_range->first;
            const double hi = std::min(cap,options.c_range->second);
            if (!(lo < hi)) throw Error("invalid_argument","c_range is empty below the trapped-wave speed cap");
            predicted = lo+0.5*(hi-lo);
            predictor = "initial_range";
            auto secular = [&](double c) {return calculateDispersionPhaseValueDouble(
                mesh.layers,mesh.thicknesses,mesh.halfspace,omega,c,wave);};
            if (checked) {
                ++diagnostics.mode_count_evaluations;
                const int count_lo = calculateAdModeCount(mesh.layers,mesh.thicknesses,
                    mesh.halfspace,omega,lo,wave);
                ++diagnostics.mode_count_evaluations;
                const int count_hi = calculateAdModeCount(mesh.layers,mesh.thicknesses,
                    mesh.halfspace,omega,hi,wave);
                if (count_lo == mode && count_hi == mode+1) {
                    double residual = std::numeric_limits<double>::infinity();
                    const auto direct = polish_count_bracket_root(secular,lo,hi,
                        secular(lo),secular(hi),options.root_tolerance,residual);
                    if (direct) result = *direct;
                    else {
                        result = mode_order();
                        ++diagnostics.fallback_count;
                        diagnostics.fallback_reasons.push_back("initial_range_secular");
                    }
                } else {
                    int count_cap = count_hi;
                    if (hi != cap) {
                        ++diagnostics.mode_count_evaluations;
                        count_cap = calculateAdModeCount(mesh.layers,mesh.thicknesses,
                            mesh.halfspace,omega,cap,wave);
                    }
                    if (count_cap <= mode) throw ModeOrderRootNotFound("Requested mode is absent below the trapped-wave speed cap");
                    result = violation("initial_range_mode_count");
                }
            } else {
                const auto roots = searchRoots(secular,lo,hi,(hi-lo)/kLegacyScanPoints,
                    std::min(options.root_tolerance,1.0e-8));
                if (roots.size() == 1 && std::isfinite(roots.front()) &&
                    std::abs(secular(roots.front())) <= kAdaptiveResidualLimit) result = roots.front();
                else result = violation("initial_range_root_count");
            }
        }
        // Prefer within-period refinement once it can use direct Brent.
        // Otherwise retain the width-based choice: broad within-period
        // scans can be expensive on coarse grids with excellent history.
        // Explicit predictors retain their policy.
        double prediction_width = 1.0e-4*predicted;
        if (std::isfinite(history.previous_prediction_error))
            prediction_width = std::max(prediction_width,
                8.0*history.previous_prediction_error);
        prediction_width = std::min(prediction_width,std::min(0.04,0.01*predicted));
        const bool refine_within = !supplied_range && !checked && options.predictor == "auto" &&
            level >= kPhaseVelocityCurvePredictedGridLevels &&
            (history.roots.size() < 2 ||
             std::abs(within[level-1]-within[level-2]) <= kAdaptiveDirectBracketFrac*guess ||
             kAdaptiveWindowGrowth*std::abs(within[level-1]-within[level-2]) <=
                prediction_width);
        if (refine_within) {
            predicted = guess;
            predictor = "coarse_level";
            failure = "local_bracket_or_residual";
            try {
                double residual=missing_value;
                const double candidate = find_dispersion_root_adaptive(
                    mesh.layers,mesh.thicknesses,mesh.halfspace,omega,guess,
                    wave,options.root_tolerance,within,0.0,missing_value,&residual);
                if (std::isfinite(candidate) && candidate > 0.0 &&
                    std::abs(candidate-guess) <= std::min(0.04,0.01*guess) &&
                    std::abs(residual) < kAdaptiveResidualLimit) {
                    result = candidate;
                    failure.clear();
                }
            } catch (const std::runtime_error&) {
                // Recover through the search policy selected below.
            }
        }
        // Checked Love keeps its cheap count bracket. Unchecked Love, like
        // unchecked Rayleigh, must use secular-only initial and local searches.
        if (!supplied_range && !refine_within && (!checked || (verify && wave == WaveType::Rayleigh)) && (!verify || predictor != "initial")) {
            auto shear = [wave](const PhysicalParameters<double>& p) {
                return std::sqrt((wave==WaveType::Love?p.c66:p.c44)/p.rho);
            };
            double slowest = shear(mesh.halfspace);
            for (const auto& p:mesh.layers) slowest = std::min(slowest,shear(p));
            const double lower = std::max(0.05,0.7*slowest);
            const double upper = 0.9995*shear(mesh.halfspace);
            const double width = std::min(0.04,0.01*predicted);
            auto secular = [&](double c) {
                return calculateDispersionPhaseValueDouble(mesh.layers,mesh.thicknesses,
                    mesh.halfspace,omega,c,wave);
            };
            // Unchecked continuation samples for observable multiple roots.
            // Checked tracking instead certifies the candidate's narrow bracket
            // with the same integer counter as the fallback mode-order search.
            double residual=missing_value;
            const auto candidate = verify
                ? try_phase_velocity_root_bracket(secular,predicted,
                    lower,upper,width/4,width,options.root_tolerance)
                : try_predicted_phase_velocity_root(secular,predicted,
                    history.previous_prediction_error,lower,upper,width,
                    options.root_tolerance,&residual);
            if (candidate && std::isfinite(*candidate) &&
                std::abs(*candidate-predicted)<=width &&
                std::abs(verify ? secular(*candidate) : residual)<kAdaptiveResidualLimit) {
                if (!verify) {
                    result = *candidate;
                } else {
                    // Match the maximum certificate width of the existing
                    // count bisection; do not relax the mode-separation scale.
                    const double relative_tolerance = std::max(
                        std::min(options.root_tolerance,1.0e-8),
                        64.0*std::numeric_limits<double>::epsilon());
                    const double half_width = 0.5*upper*std::max(
                        1.0e-5,std::sqrt(relative_tolerance));
                    const double lo = std::max(lower,*candidate-half_width);
                    const double hi = std::min(upper,*candidate+half_width);
                    const double f_lo = secular(lo), f_hi = secular(hi);
                    failure = "local_mode_certificate";
                    if (lo < *candidate && *candidate < hi &&
                        std::isfinite(f_lo) && std::isfinite(f_hi) &&
                        (f_lo == 0.0 || f_hi == 0.0 ||
                         std::signbit(f_lo) != std::signbit(f_hi))) {
                        try {
                            ++diagnostics.mode_count_evaluations;
                            const int count_lo = calculateAdModeCount(mesh.layers,
                                mesh.thicknesses,mesh.halfspace,omega,lo,wave);
                            ++diagnostics.mode_count_evaluations;
                            const int count_hi = calculateAdModeCount(mesh.layers,
                                mesh.thicknesses,mesh.halfspace,omega,hi,wave);
                            if (count_lo == mode && count_hi == mode+1) {
                                result = *candidate;
                                failure.clear();
                            }
                        } catch (const std::runtime_error&) {
                            // Invalid/radiative local evaluation: the unchanged
                            // mode-order search must recover or report failure.
                        }
                    }
                }
            }
            else failure = "local_bracket_or_residual";
        } else if (!checked && !refine_within) failure = "prediction_unavailable";
        if (!std::isfinite(result)) {
            if (verify || recover) {
                if (modern && has_history && !(verify && wave==WaveType::Love)) {
                    result = violation(failure.empty() ? "local_bracket_or_residual" : failure);
                    failure.clear();
                } else result = mode_order();
            } else if (modern && has_history) {
                result = violation(failure.empty() ? "local_bracket_or_residual" : failure);
                failure.clear();
            } else {
                // Explicitly unchecked: initial selection and recovery use only
                // secular roots near the prediction, never an integer counter.
                const double lower = std::max(0.1,0.8*predicted);
                const double upper = std::min(cap,1.2*predicted);
                if ((!(lower < upper) || !std::isfinite(upper)) && !modern)
                    throw Error("root_not_found","No finite search interval near the unchecked prediction");
                auto secular = [&](double c) {
                    return calculateDispersionPhaseValueDouble(mesh.layers,
                        mesh.thicknesses,mesh.halfspace,omega,c,wave);
                };
                std::vector<double> roots;
                if (lower < upper && std::isfinite(upper))
                    roots = searchRoots(secular,lower,upper,(upper-lower)/kLegacyScanPoints,
                        std::min(options.root_tolerance,1.e-8));
                if (modern && roots.size() != 1) {
                    result = violation("initial_root_count");
                    failure.clear();
                } else if (roots.empty())
                    throw Error("root_not_found","No secular root near the unchecked prediction; supply c_guess or use mode_order");
                else result = *std::min_element(roots.begin(),roots.end(),
                    [predicted](double a,double b) {
                        return std::abs(a-predicted)<std::abs(b-predicted);
                    });
            }
            if (!failure.empty()) {
                ++diagnostics.fallback_count;
                diagnostics.fallback_reasons.push_back(failure);
            }
        }
        if (!(result > 0.0 && result < cap)) {
            if (modern) result = violation("trapped_wave_speed_cap");
            else throw Error("root_not_found","No secular root below the trapped-wave speed cap");
        }
        diagnostics.level_mode_verified.push_back(checked);
        diagnostics.predictors.push_back(predictor);
        history.previous_prediction_error = std::abs(result-predicted);
        if (!within.empty()) coarse_offsets[level] = result-within.back();
        // Only three samples are used by the predictor. Reserve on first use
        // so unused refinement levels do not allocate, and evict before append.
        if (history.roots.empty()) {
            history.periods.reserve(3);
            history.roots.reserve(3);
        }
        if (history.roots.size()==3) {
            history.periods.erase(history.periods.begin());
            history.roots.erase(history.roots.begin());
        }
        history.periods.push_back(period);
        history.roots.push_back(result);
        slopes[level] = missing_value;
        return result;
    }

    void group_velocity(size_t level, double omega, double c, double u) {
        if (std::isfinite(u) && u != 0)
            slopes.at(level) = c/omega*(1-c/u);
    }
};

}}} // namespace disper26::api::detail
