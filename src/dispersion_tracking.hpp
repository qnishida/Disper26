#pragma once
#include <array>
// Included after api_backend.hpp: reuses its predictor, bracket and count solver.
namespace disper26 { namespace api { namespace detail {

struct DispersionTracking {
    std::array<PhaseVelocityRootHistory, 13> histories;
    std::array<double, 13> slopes;
    std::array<double, 13> coarse_offsets;
    double next_period = missing_value; // Set only by curve calls, after deduplication.

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
        const bool recover = options.method == "continuation_recover";
        const bool verify = (options.method != "continuation" && !recover) || options.verify_mode.value_or(false);
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
        // Prefer within-period refinement once it can use direct Brent.
        // Otherwise retain the width-based choice: broad within-period
        // scans can be expensive on coarse grids with excellent history.
        // Explicit predictors retain their policy.
        double prediction_width = 1.0e-4*predicted;
        if (std::isfinite(history.previous_prediction_error))
            prediction_width = std::max(prediction_width,
                8.0*history.previous_prediction_error);
        prediction_width = std::min(prediction_width,std::min(0.04,0.01*predicted));
        const bool refine_within = !checked && options.predictor == "auto" &&
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
        if (!refine_within && (!checked || (verify && wave == WaveType::Rayleigh)) && (!verify || predictor != "initial")) {
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
                checked = true;
                result = find_dispersion_root_by_mode_order(mesh,omega,wave,mode,
                    options.root_tolerance,predicted,within,&diagnostics.mode_count_evaluations);
            } else {
                // Explicitly unchecked: initial selection and recovery use only
                // secular roots near the prediction, never an integer counter.
                const double lower = std::max(0.1,0.8*predicted);
                const double upper = 1.2*predicted;
                if (!(lower < upper) || !std::isfinite(upper))
                    throw Error("root_not_found","No finite search interval near the unchecked prediction");
                auto secular = [&](double c) {
                    return calculateDispersionPhaseValueDouble(mesh.layers,
                        mesh.thicknesses,mesh.halfspace,omega,c,wave);
                };
                const auto roots = searchRoots(secular,lower,upper,
                    (upper-lower)/kLegacyScanPoints,
                    std::min(options.root_tolerance,1.e-8));
                if (roots.empty())
                    throw Error("root_not_found","No secular root near the unchecked prediction; supply c_guess or use mode_order");
                result = *std::min_element(roots.begin(),roots.end(),
                    [predicted](double a,double b) {
                        return std::abs(a-predicted)<std::abs(b-predicted);
                    });
            }
            if (!failure.empty()) {
                ++diagnostics.fallback_count;
                diagnostics.fallback_reasons.push_back(failure);
            }
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
