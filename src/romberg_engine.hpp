#pragma once
// Romberg Acceleration Engine for GRID Mode
// Implements high-precision phase velocity, group velocity, and H/V calculation
// using Richardson extrapolation on layer-refined meshes.

#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <Eigen/Core>

#include "grid_refinement_geometry.hpp"

// Forward declarations - these are defined in disper26.cpp
template <typename T> struct PhysicalParameters;
enum class WaveType;

// --- Enum Definitions ---
enum class ModelInterpolation {
    LAYER_CAKE = 0, // Staircase approximation (use input layers as-is)
    PANCAKE = LAYER_CAKE, // Legacy alias.
    GRID = 1 // Grid mode (linear interpolation + discontinuity handling + Romberg)
};

// --- Grid Model Input Structure ---
// Discontinuities are represented by duplicating depth points.
// is_discontinuity[i] = true means params[i] is directly above, params[i+1] is directly below.
struct GridModel {
    std::vector<double> depths;                         // Depth points (0, z1, ..., z_disc, z_disc, ...)
    std::vector<PhysicalParameters<double>> params;     // Parameters at each depth point
    std::vector<bool> is_discontinuity;                 // is_discontinuity[i] = true → discontinuity between i and i+1
    GridRefinementPolicy refinement_policy{};            // flat by default; spherical is explicit
    
    void require_valid() const {
        if (depths.empty() || depths.size() != params.size())
            throw std::invalid_argument("Invalid GridModel depth/material counts");
        if (!std::is_sorted(depths.begin(), depths.end()))
            throw std::invalid_argument("Grid depths must be non-decreasing");
        validate_grid_interfaces(depths, is_discontinuity);
    }

    bool validate() const {
        try { require_valid(); }
        catch (const std::invalid_argument&) { return false; }
        return true;
    }
};

// Raw roots selected on the independently refined meshes.  The direct
// dispersion loop owns branch selection; derivative consumers must reuse
// these values instead of starting a second root search from the extrapolated
// phase velocity.
struct RombergLevelRootTrace {
    std::vector<double> phase_velocity_roots;
    int phase_accepted_level = -1;
    int generated_level = -1;
    int extrapolate_start_level = 0;
};

// --- Romberg Result Structure ---
struct RombergResult {
    double phase_velocity;
    double group_velocity;
    double hv_ratio;
    int converged_level;
    double relative_error;
    RombergLevelRootTrace root_trace;
};

// Independent result surface for the optional frequency-curvature path.
// Keeping this separate from RombergResult preserves the existing five-value
// GRID dispersion API and its convergence behavior.
enum class FrequencyCurvatureStatus {
    Ok,
    UnsafeRoot,
    UnsafeDerivative
};

struct FrequencyCurvatureRombergResult {
    double phase_velocity;
    double group_velocity;
    double group_velocity_omega_derivative;
    double wavenumber_omega_second_derivative;
    double dimensionless_phase_group_dispersion;
    double dimensionless_frequency_curvature;
    int accepted_level;
    double phase_velocity_relative_error;
    double group_velocity_omega_derivative_error;
    double maximum_root_residual;
    FrequencyCurvatureStatus status;
};

// --- Mesher Class ---
// Handles mesh refinement while preserving discontinuities
class Mesher {
private:
    GridModel model_;
    mutable std::vector<std::vector<GridSublayerGeometry>> geometry_cache_;

    std::vector<bool> fluid_nodes() const {
        std::vector<bool> out(model_.params.size(), false);
        for (size_t i = 0; i < model_.params.size(); ++i) {
            out[i] = isLiquidLayer(model_.params[i]);
        }
        return out;
    }

    const std::vector<GridSublayerGeometry>& geometry(int level) const {
        if (level < 0) {
            throw std::invalid_argument("refinement_level must be non-negative");
        }
        if (geometry_cache_.size() <= static_cast<size_t>(level)) {
            geometry_cache_.resize(static_cast<size_t>(level) + 1);
        }
        auto& cached = geometry_cache_[static_cast<size_t>(level)];
        if (cached.empty()) {
            cached = build_grid_sublayer_geometry(
                model_.depths, model_.is_discontinuity, fluid_nodes(), level,
                model_.refinement_policy);
        }
        return cached;
    }

    template <typename T, typename G>
    static PhysicalParameters<T> scale_parameters(
        PhysicalParameters<T> p, const GridSublayerGeometryT<G>& g)
    {
        p.rho *= g.density_scale;
        p.c11 *= g.elastic_scale;
        p.c13 *= g.elastic_scale;
        p.c33 *= g.elastic_scale;
        p.c44 *= g.elastic_scale;
        p.c66 *= g.elastic_scale;
        return p;
    }

    template <typename T, typename G>
    static PhysicalParameters<T> interpolate_scaled(
        const PhysicalParameters<T>& p1,
        const PhysicalParameters<T>& p2,
        const GridSublayerGeometryT<G>& g)
    {
        const G& t = g.alpha;
        PhysicalParameters<T> result;
        result.rho = p1.rho * (1.0 - t) + p2.rho * t;
        result.c11 = p1.c11 * (1.0 - t) + p2.c11 * t;
        result.c13 = p1.c13 * (1.0 - t) + p2.c13 * t;
        result.c33 = p1.c33 * (1.0 - t) + p2.c33 * t;
        result.c44 = p1.c44 * (1.0 - t) + p2.c44 * t;
        result.c66 = p1.c66 * (1.0 - t) + p2.c66 * t;
        result.gravity = p1.gravity * (1.0 - t) + p2.gravity * t;
        result.gravity_internal_scale = p1.gravity_internal_scale * (1.0 - t) + p2.gravity_internal_scale * t;
        result.gravity_surface_scale = p1.gravity_surface_scale * (1.0 - t) + p2.gravity_surface_scale * t;
        return scale_parameters(result, g);
    }

    GridSublayerGeometry halfspace_geometry() const {
        GridSublayerGeometry g;
        g.radius_ratio = model_.refinement_policy.is_flat()
            ? 1.0
            : (model_.refinement_policy.planet_radius - model_.depths.back()) /
              model_.refinement_policy.planet_radius;
        g.fluid = isLiquidLayer(model_.params.back());
        grid_fill_flattening_scales(g, model_.refinement_policy);
        return g;
    }
    
    // Linear interpolation helper for PhysicalParameters
    static PhysicalParameters<double> interpolate(
        const PhysicalParameters<double>& p1,
        const PhysicalParameters<double>& p2,
        double t  // t in [0, 1]
    ) {
        PhysicalParameters<double> result;
        result.rho = p1.rho * (1.0 - t) + p2.rho * t;
        result.c11 = p1.c11 * (1.0 - t) + p2.c11 * t;
        result.c13 = p1.c13 * (1.0 - t) + p2.c13 * t;
        result.c33 = p1.c33 * (1.0 - t) + p2.c33 * t;
        result.c44 = p1.c44 * (1.0 - t) + p2.c44 * t;
        result.c66 = p1.c66 * (1.0 - t) + p2.c66 * t;
        result.gravity = p1.gravity * (1.0 - t) + p2.gravity * t;
        result.gravity_internal_scale = p1.gravity_internal_scale * (1.0 - t) + p2.gravity_internal_scale * t;
        result.gravity_surface_scale = p1.gravity_surface_scale * (1.0 - t) + p2.gravity_surface_scale * t;
        return result;
    }
    
public:
    explicit Mesher(const GridModel& model) : model_(model) {
        model_.require_valid();
    }
    
    // Get number of nodes (for AD variable sizing)
    size_t get_node_count() const {
        return model_.params.size();
    }
    
    // Get depths for reference
    const std::vector<double>& get_depths() const {
        return model_.depths;
    }
    
    // Get discontinuity flags
    const std::vector<bool>& get_discontinuities() const {
        return model_.is_discontinuity;
    }

    const GridRefinementPolicy& get_refinement_policy() const {
        return model_.refinement_policy;
    }

    const std::vector<GridSublayerGeometry>& get_refined_geometry(int level) const {
        return geometry(level);
    }
    
    // Get refined layers for a given refinement level k
    // Each continuous segment is divided into 2^k sub-layers
    // Returns: (layers, thicknesses) for use with calculateDispersionFunction
    std::tuple<std::vector<PhysicalParameters<double>>, std::vector<double>>
    get_refined_layers(int refinement_level) const {
        std::vector<PhysicalParameters<double>> layers;
        std::vector<double> thicknesses;
        const auto& geo = geometry(refinement_level);
        layers.reserve(geo.size());
        thicknesses.reserve(geo.size());
        for (const auto& g : geo) {
            const int i = g.parent_node;
            layers.push_back(interpolate_scaled(model_.params[i], model_.params[i + 1], g));
            thicknesses.push_back(g.thickness);
        }
        return {layers, thicknesses};
    }
    
    // Caller-owned storage; all geometry and material values are recomputed.
    void fill_refined_layers(int level, std::vector<PhysicalParameters<double>>& layers,
            std::vector<double>& thicknesses, std::vector<GridSublayerGeometry>& geo,
            std::vector<bool>& fluids) const {
        fluids.resize(model_.params.size());
        for(size_t i=0;i<fluids.size();++i) fluids[i]=isLiquidLayer(model_.params[i]);
        fill_grid_sublayer_geometry(model_.depths,model_.is_discontinuity,fluids,
            level,model_.refinement_policy,geo);
        layers.clear(); thicknesses.clear();
        layers.reserve(geo.size()); thicknesses.reserve(geo.size());
        for(const auto& g:geo) {
            layers.push_back(interpolate_scaled(model_.params[g.parent_node],model_.params[g.parent_node+1],g));
            thicknesses.push_back(g.thickness);
        }
    }

    // AD-aware version: node_params are AD types, sublayers computed via interpolation
    // T should be an AD scalar type (e.g., Eigen::AutoDiffScalar)
    // Returns: (layers, thicknesses) with AD parameters, and mapping to parent nodes
    template <typename T>
    std::tuple<std::vector<PhysicalParameters<T>>, std::vector<T>>
    get_refined_layers_ad(
        int refinement_level,
        const std::vector<PhysicalParameters<T>>& node_params
    ) const {
        std::vector<PhysicalParameters<T>> layers;
        std::vector<T> thicknesses;
        const auto& geo = geometry(refinement_level);
        layers.reserve(geo.size());
        thicknesses.reserve(geo.size());
        for (const auto& g : geo) {
            const int i = g.parent_node;
            layers.push_back(interpolate_scaled(node_params[i], node_params[i + 1], g));
            thicknesses.push_back(T(g.thickness));
        }
        return {layers, thicknesses};
    }
    
    // AD-aware version for depths: node_depths are AD types
    // Used for computing sensitivity to discontinuity depth
    template <typename T>
    std::tuple<std::vector<PhysicalParameters<T>>, std::vector<T>>
    get_refined_layers_ad_with_depths(
        int refinement_level,
        const std::vector<PhysicalParameters<T>>& node_params,
        const std::vector<T>& node_depths
    ) const {
        if (!model_.refinement_policy.is_flat()) {
            std::vector<GridSublayerGeometryT<T>> geometry_ad;
            refine_grid_geometry(node_depths, model_.is_discontinuity, fluid_nodes(),
                                 refinement_level, model_.refinement_policy, geometry_ad);
            std::vector<PhysicalParameters<T>> layers;
            std::vector<T> thicknesses;
            layers.reserve(geometry_ad.size());
            thicknesses.reserve(geometry_ad.size());
            for (const auto& g : geometry_ad) {
                const int i = g.parent_node;
                layers.push_back(interpolate_scaled(node_params[i], node_params[i + 1], g));
                thicknesses.push_back(g.thickness);
            }
            return {std::move(layers), std::move(thicknesses)};
        }
        std::vector<PhysicalParameters<T>> layers;
        std::vector<T> thicknesses;
        
        int num_subdivisions = 1 << refinement_level;  // 2^k
        
        size_t i = 0;
        while (i < model_.depths.size() - 1) {
            // Check if this is a discontinuity
            if (model_.is_discontinuity[i]) {
                // Skip the discontinuity - the next segment will use params[i+1]
                ++i;
                continue;
            }
            
            // Check for zero tickness segment (should only happen at discontinuities, already handled)
            // But with AD depths, we rely on the original 'is_discontinuity' flag topology.
            // We assume the TOPOLOGY (which indices are discontinuities) doesn't change with perturbation.
            
            const T& z_start = node_depths[i];
            const T& z_end = node_depths[i + 1];
            T segment_thickness = z_end - z_start;
            
            const auto& p_start = node_params[i];
            const auto& p_end = node_params[i + 1];
            
            // Subdivide this segment
            T sub_thickness = segment_thickness / double(num_subdivisions);
            
            for (int j = 0; j < num_subdivisions; ++j) {
                // Interpolation parameter for midpoint of sub-layer
                double t_mid = (j + 0.5) / num_subdivisions;
                
                // Linear interpolation in AD graph
                PhysicalParameters<T> p_mid;
                p_mid.rho = p_start.rho * (1.0 - t_mid) + p_end.rho * t_mid;
                p_mid.c11 = p_start.c11 * (1.0 - t_mid) + p_end.c11 * t_mid;
                p_mid.c13 = p_start.c13 * (1.0 - t_mid) + p_end.c13 * t_mid;
                p_mid.c33 = p_start.c33 * (1.0 - t_mid) + p_end.c33 * t_mid;
                p_mid.c44 = p_start.c44 * (1.0 - t_mid) + p_end.c44 * t_mid;
                p_mid.c66 = p_start.c66 * (1.0 - t_mid) + p_end.c66 * t_mid;
                p_mid.gravity = p_start.gravity * (1.0 - t_mid) + p_end.gravity * t_mid;
                p_mid.gravity_internal_scale = p_start.gravity_internal_scale * (1.0 - t_mid) + p_end.gravity_internal_scale * t_mid;
                p_mid.gravity_surface_scale = p_start.gravity_surface_scale * (1.0 - t_mid) + p_end.gravity_surface_scale * t_mid;
                
                layers.push_back(p_mid);
                thicknesses.push_back(sub_thickness);
            }
            
            ++i;
        }
        
        return {layers, thicknesses};
    }
    
    // Get halfspace parameters (last point that is not a discontinuity upper boundary)
    PhysicalParameters<double> get_halfspace() const {
        return scale_parameters(model_.params.back(), halfspace_geometry());
    }
    
    // AD version of halfspace
    template <typename T>
    PhysicalParameters<T> get_halfspace_ad(const std::vector<PhysicalParameters<T>>& node_params) const {
        return scale_parameters(node_params.back(), halfspace_geometry());
    }

    template <typename T>
    PhysicalParameters<T> get_halfspace_ad(const std::vector<PhysicalParameters<T>>& nodes,
                                           const T& depth) const {
        if (model_.refinement_policy.is_flat()) return get_halfspace_ad(nodes);
        GridSublayerGeometryT<T> g;
        const double radius = model_.refinement_policy.planet_radius;
        g.radius_ratio = (radius - depth) / radius;
        g.fluid = isLiquidLayer(model_.params.back());
        grid_fill_flattening_scales(g, model_.refinement_policy);
        return scale_parameters(nodes.back(), g);
    }

    // Apply J^T: convert sublayer gradient (size (N_k+1)*npl) to node gradient (size M*npl).
    // g_sublayer[l*npl+p] = gradient w.r.t. sublayer l parameter p (l=0..N_k-1).
    // g_sublayer[N_k*npl+p] = gradient w.r.t. halfspace parameter p.
    // Returns g_node[i*npl+p] = gradient w.r.t. node i parameter p.
    Eigen::VectorXd apply_sublayer_grad_to_node(
        int level,
        const Eigen::VectorXd& g_sublayer,
        int npl
    ) const {
        int M = (int)model_.depths.size();
        Eigen::VectorXd g_node = Eigen::VectorXd::Zero(M * npl);
        int sublayer_idx = 0;
        for (const auto& g : geometry(level)) {
            const int i = g.parent_node;
            const double w_start = 1.0 - g.alpha;
            const double w_end = g.alpha;
            for (int p = 0; p < npl; ++p) {
                const double scale = (p == 0) ? g.density_scale : g.elastic_scale;
                const double v = g_sublayer[sublayer_idx * npl + p] * scale;
                g_node[i * npl + p] += w_start * v;
                g_node[(i + 1) * npl + p] += w_end * v;
            }
            ++sublayer_idx;
        }

        // Halfspace gradient → last node (M-1)
        const auto hs_geo = halfspace_geometry();
        for (int p = 0; p < npl; ++p) {
            const double scale = (p == 0) ? hs_geo.density_scale : hs_geo.elastic_scale;
            g_node[(M-1) * npl + p] += g_sublayer[sublayer_idx * npl + p] * scale;
        }

        return g_node;
    }

    // Same J^T mapping applied to a whole block of gradient columns at once.
    // The vectorized Hessian backend produces (N_k+1)*npl sublayer rows by W
    // derivative directions, and this walks the mesh once for all W instead of
    // once per direction.
    Eigen::MatrixXd apply_sublayer_grad_to_node(
        int level,
        const Eigen::MatrixXd& g_sublayer,
        int npl
    ) const {
        int M = (int)model_.depths.size();
        int W = (int)g_sublayer.cols();
        Eigen::MatrixXd g_node = Eigen::MatrixXd::Zero(M * npl, W);
        int sublayer_idx = 0;
        for (const auto& g : geometry(level)) {
            const int i = g.parent_node;
            const double w_start = 1.0 - g.alpha;
            const double w_end = g.alpha;
            for (int p = 0; p < npl; ++p) {
                const double scale = (p == 0) ? g.density_scale : g.elastic_scale;
                const auto v = g_sublayer.row(sublayer_idx * npl + p) * scale;
                g_node.row(i * npl + p) += w_start * v;
                g_node.row((i + 1) * npl + p) += w_end * v;
            }
            ++sublayer_idx;
        }

        const auto hs_geo = halfspace_geometry();
        for (int p = 0; p < npl; ++p) {
            const double scale = (p == 0) ? hs_geo.density_scale : hs_geo.elastic_scale;
            g_node.row((M-1) * npl + p) +=
                g_sublayer.row(sublayer_idx * npl + p) * scale;
        }

        return g_node;
    }
};

// --- Neville's Algorithm for Richardson Extrapolation ---
// Performs polynomial extrapolation to h=0
// table[k] contains the value computed with step size h/2^k
// Returns extrapolated value and estimated error
inline std::pair<double, double> neville_extrapolation(const std::vector<double>& table) {
    int n = table.size();
    if (n == 0) return {0.0, 1.0};
    if (n == 1) return {table[0], 1.0};
    
    // Neville's tableau
    std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));
    
    // Initialize with input values
    for (int i = 0; i < n; ++i) {
        P[i][0] = table[i];
    }
    
    // Fill tableau
    // Richardson extrapolation: error is O(h^2), so ratio is 4
    for (int j = 1; j < n; ++j) {
        double ratio = std::pow(4.0, j);  // 4^j for O(h^2) convergence
        for (int i = j; i < n; ++i) {
            P[i][j] = P[i][j-1] + (P[i][j-1] - P[i-1][j-1]) / (ratio - 1.0);
        }
    }
    
    double extrapolated = P[n-1][n-1];
    double error_estimate = (n >= 2) ? std::abs(P[n-1][n-1] - P[n-2][n-2]) : 1.0;
    
    return {extrapolated, error_estimate};
}

// Shared grid/root-selection policy used by GRID Romberg dispersion paths.
// Search for all roots in a local window around c_guess, then pick the closest.
inline double find_dispersion_root_near_by_scan(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    double omega,
    double c_guess,
    WaveType type,
    double tolerance,
    double* root_residual = nullptr
) {
    if (root_residual) *root_residual = std::numeric_limits<double>::quiet_NaN();
    auto func = [&](double c) {
        return calculateDispersionPhaseValueDouble(
            layers, thicknesses, halfspace, omega, c, type);
    };

    double search_min = std::max(0.1, c_guess * 0.8);
    double search_max = c_guess * 1.2;
    double step = (search_max - search_min) / 100.0;
    double root_tol = std::min(tolerance, 1e-8);

    std::vector<double> residuals;
    auto roots = searchRoots(func, search_min, search_max, step, root_tol,
                             root_residual ? &residuals : nullptr);
    if (!roots.empty()) {
        size_t best = 0;
        for (size_t i=1; i<roots.size(); ++i) {
            if (std::abs(roots[i]-c_guess) < std::abs(roots[best]-c_guess)) best=i;
        }
        if (root_residual) *root_residual=residuals[best];
        return roots[best];
    }
    if (root_residual) *root_residual=func(c_guess);
    return c_guess;
}

// --- Adaptive root finding for the Romberg level loop ---
// Once refinement levels converge, the root barely moves between levels, so
// the fixed +/-20% / 100-point scan above wastes evaluations. This variant
// derives a window from the inter-level root displacement in root_history
// (the c_table accumulated by compute_with_romberg_core) and, once the
// displacement is tiny, tries a direct Brent bracket around c_guess.
//
// Branch safety: every window is symmetric about c_guess, so a root found
// inside it is the one nearest to c_guess - the same selection the legacy
// scan would make whenever its pick lies inside the window. Scan density is
// never coarser than the legacy step, so close root pairs are not newly
// missed, and every failure falls back to the full legacy scan (which itself
// returns c_guess when nothing is found). The |f| < 0.1 residual filter of
// searchRoots applies on all scan paths and is re-checked on the direct
// Brent path.
inline constexpr double kAdaptiveFirstRefineHalfWidthFrac = 0.05;
inline constexpr double kAdaptiveWindowGrowth = 8.0;
inline constexpr double kAdaptiveMinHalfWidthFrac = 1e-5;
inline constexpr double kAdaptiveDirectBracketFrac = 1e-4;
inline constexpr double kAdaptiveResidualLimit = 0.1;   // same as searchRoots
inline constexpr int    kAdaptiveMinScanPoints = 8;
inline constexpr double kLegacyScanHalfWidthFrac = 0.2; // +/-20%
inline constexpr int    kLegacyScanPoints = 100;
// Trace-only refinement may go deeper than the phase table, but each level
// doubles the mesh density.  Keep the public extension bounded by the
// deepest level already exercised by the established adaptive-stop tests.
inline constexpr int    kMaximumTraceOnlyRefinementLevel = 12;

inline double find_dispersion_root_adaptive(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    double omega,
    double c_guess,
    WaveType type,
    double tolerance,
    const std::vector<double>& root_history,
    double initial_root_window_frac = 0.0,
    double initial_branch_anchor = std::numeric_limits<double>::quiet_NaN(),
    double* root_residual = nullptr
) {
    if (root_residual) *root_residual = std::numeric_limits<double>::quiet_NaN();
    const size_t n = root_history.size();

    auto func = [&](double c) {
        return calculateDispersionPhaseValueDouble(
            layers, thicknesses, halfspace, omega, c, type);
    };
    auto legacy_fallback = [&]() {
        return find_dispersion_root_near_by_scan(
            layers, thicknesses, halfspace, omega, c_guess, type, tolerance, root_residual);
    };

    const double root_tol = std::min(tolerance, 1e-8);
    const double legacy_step =
        2.0 * kLegacyScanHalfWidthFrac * c_guess / kLegacyScanPoints;

    if (n == 0) {
        // Level 0 normally preserves legacy behavior. For period sweeps with
        // a predictor-quality c_guess, try a narrow local window first and
        // fall back to the legacy scan if it is not safely bracketed/found.
        if (initial_root_window_frac > 0.0 && std::isfinite(initial_root_window_frac)) {
            const double half_width = std::clamp(
                initial_root_window_frac * c_guess,
                kAdaptiveMinHalfWidthFrac * c_guess,
                kLegacyScanHalfWidthFrac * c_guess);
            const double search_min = std::max(0.1, c_guess - half_width);
            const double search_max = c_guess + half_width;

            if (search_min > 0.0 && search_max > search_min) {
                double fa = func(search_min);
                double fb = func(search_max);
                if (std::isfinite(fa) && std::isfinite(fb) && fa * fb < 0.0) {
                    const auto refined = brents_method_with_values(
                        func, search_min, search_max, fa, fb, root_tol);
                    if (refined.root > 0.0 &&
                        std::abs(refined.function_value) < kAdaptiveResidualLimit) {
                        if (root_residual) *root_residual=refined.function_value;
                        return refined.root;
                    }
                }

                const double width = search_max - search_min;
                const int n_points = std::max(
                    kAdaptiveMinScanPoints,
                    static_cast<int>(std::ceil(width / legacy_step)));
                std::vector<double> residuals;
                auto roots = searchRoots(func, search_min, search_max, width / n_points, root_tol,
                                         root_residual ? &residuals : nullptr);
                if (!roots.empty()) {
                    size_t best = 0;
                    for (size_t i=1; i<roots.size(); ++i) {
                        if (std::abs(roots[i]-c_guess) < std::abs(roots[best]-c_guess)) best=i;
                    }
                    if (root_residual) *root_residual=residuals[best];
                    return roots[best];
                }
            }
        }
        return legacy_fallback();
    }

    double half_width;
    if (n == 1) {
        // First refinement: level-0 -> level-1 movement is the largest, stay generous.
        half_width = kAdaptiveFirstRefineHalfWidthFrac * c_guess;
    } else {
        const double d = std::abs(root_history[n - 1] - root_history[n - 2]);

        if (d <= kAdaptiveDirectBracketFrac * c_guess) {
            // Converged regime: try a direct symmetric bracket + Brent.
            double delta = std::max(kAdaptiveWindowGrowth * d,
                                    kAdaptiveMinHalfWidthFrac * c_guess);
            for (int attempt = 0; attempt < 2; ++attempt, delta *= 8.0) {
                double a = c_guess - delta;
                double b = c_guess + delta;
                if (a <= 0.0) break;
                double fa = func(a);
                double fb = func(b);
                if (!std::isfinite(fa) || !std::isfinite(fb)) break;
                if (fa * fb < 0.0) {
                    const auto refined = brents_method_with_values(
                        func, a, b, fa, fb, root_tol);
                    if (refined.root > 0.0 &&
                        std::abs(refined.function_value) < kAdaptiveResidualLimit) {
                        if (root_residual) *root_residual=refined.function_value;
                        return refined.root;
                    }
                    break; // bracketed a pole or Brent failed: be conservative
                }
            }
            // Conservative fallback: the full legacy scan, not a shrunk one.
            return legacy_fallback();
        }

        half_width = std::clamp(kAdaptiveWindowGrowth * d,
                                kAdaptiveMinHalfWidthFrac * c_guess,
                                kLegacyScanHalfWidthFrac * c_guess);
    }

    // Reduced scan, never coarser than the legacy scan density.
    const double search_min = std::max(0.1, c_guess - half_width);
    const double search_max = c_guess + half_width;
    const double width = search_max - search_min;
    const int n_points = std::max(
        kAdaptiveMinScanPoints, static_cast<int>(std::ceil(width / legacy_step)));
    const double step = width / n_points;

    std::vector<double> residuals;
    auto roots = searchRoots(func, search_min, search_max, step, root_tol,
                             root_residual ? &residuals : nullptr);
    if (!roots.empty()) {
        // At the first refinement the discretization shift of the requested
        // root can be slightly larger than its distance to an adjacent root.
        // The original continuation hint is then a safer identity anchor than
        // the level-0 root. Once two levels exist, retain historical nearest-
        // previous-root tracking.
        const double selection_target =
            n == 1 && std::isfinite(initial_branch_anchor)
                ? initial_branch_anchor
                : c_guess;
        size_t best = 0;
        for (size_t i=1; i<roots.size(); ++i) {
            if (std::abs(roots[i]-selection_target) <
                std::abs(roots[best]-selection_target)) best=i;
        }
        if (root_residual) *root_residual=residuals[best];
        return roots[best];
    }
    return legacy_fallback();
}

// --- Main Romberg Computation Function ---
// Single shared Romberg refinement loop for all GRID dispersion paths.
// Generic over the mesh representation: get_mesh(k) returns the refined model
// at level k (e.g. VelocityGridRefinement), and the three calculators receive
// that mesh object. find_root additionally receives the roots accepted at
// previous levels (c_history), which adaptive root finders may use.
// phase_only skips the per-level group velocity (full-AD) and H/V
// evaluations entirely; the root iteration and convergence control depend
// only on the phase velocity table, so c/converged_level/relative_error are
// unchanged, and U/HV are returned as NaN.
struct RombergWorkspace {
    std::vector<double> c, u, hv, roots;
};

template <typename MeshProvider, typename RootFinder, typename GroupVelCalc, typename HVCalc>
RombergResult compute_with_romberg_core(
    MeshProvider&& get_mesh,
    double period,
    double c_initial_guess,
    RootFinder&& find_root,
    GroupVelCalc&& calc_group_vel,
    HVCalc&& calc_hv,
    int max_steps = 5,
    double tolerance = 1e-10,
    bool phase_only = false,
    int extrapolate_start_level = 0,
    int minimum_trace_level = 0,
    RombergWorkspace* workspace = nullptr
) {
    if (max_steps < 0) {
        throw std::invalid_argument("max_steps must be non-negative");
    }
    if (minimum_trace_level < 0) {
        throw std::invalid_argument(
            "minimum_trace_level must be non-negative");
    }
    if (minimum_trace_level >
        std::max(max_steps, kMaximumTraceOnlyRefinementLevel)) {
        throw std::invalid_argument(
            "minimum_trace_level exceeds the safe trace-only refinement limit");
    }
    RombergWorkspace local;
    auto& scratch = workspace ? *workspace : local;
    auto& c_table = scratch.c; auto& U_table = scratch.u;
    auto& HV_table = scratch.hv; auto& root_history = scratch.roots;
    c_table.clear(); U_table.clear(); HV_table.clear(); root_history.clear();

    double omega = 2.0 * M_PI / period;
    double c_guess = c_initial_guess;

    int converged_level = max_steps;
    double rel_error = 1.0;
    double prev_rel_error = std::numeric_limits<double>::infinity();
    const int min_stop_level = std::min(4, max_steps);
    // The loop only reaches k = max_steps, so the round-off revert below must
    // be armed no later than the final level or it can never fire.
    const int min_revert_level = std::min(min_stop_level, max_steps - 1);
    const int extrapolate_start = std::max(0, extrapolate_start_level);
    bool phase_table_complete = false;

    auto extrapolate_from_start = [extrapolate_start](const std::vector<double>& table) {
        if (extrapolate_start <= 0 ||
            static_cast<size_t>(extrapolate_start) >= table.size()) {
            return neville_extrapolation(table);
        }
        std::vector<double> active(table.begin() + extrapolate_start, table.end());
        return neville_extrapolation(active);
    };

    const int final_root_level = std::max(max_steps, minimum_trace_level);
    for (int k = 0; k <= final_root_level; ++k) {
        // Preserve a reference when a batch provider caches refinement
        // levels, while still extending the lifetime of scalar-provider
        // temporaries through this loop iteration.
        const auto& mesh = get_mesh(k);

        // Find phase velocity root
        double c = find_root(mesh, omega, c_guess, root_history);
        if (std::isnan(c)) {
            // Preserve the legacy phase-table fallback, but never label a
            // copied previous-level value as a newly generated mesh root.
            if (phase_table_complete) {
                throw std::runtime_error(
                    "Romberg trace-only root solve failed before "
                    "minimum_trace_level");
            }
            c = c_guess;
        }
        c_guess = c;  // Update guess for next iteration
        root_history.push_back(c);

        // Once the legacy phase table has converged, optionally continue only
        // the root trace.  The reported c/U/HV and their convergence metadata
        // remain based on exactly the same prefix as the five-value API.
        if (phase_table_complete) {
            if (k >= minimum_trace_level) break;
            continue;
        }

        c_table.push_back(c);
        if (!phase_only) {
            U_table.push_back(calc_group_vel(mesh, omega, c));
            HV_table.push_back(calc_hv(mesh, omega, c));
        }

        const int table_start = std::min(extrapolate_start, static_cast<int>(c_table.size()) - 1);
        const int effective_points = static_cast<int>(c_table.size()) - table_start;

        // Check convergence (need at least 2 extrapolation points)
        if (effective_points >= 2) {
            auto [c_ext, c_err] = extrapolate_from_start(c_table);
            if (std::abs(c_ext) > 1e-20) {
                rel_error = c_err / std::abs(c_ext);

                // Adaptive stopping is only reliable after the extrapolation
                // sequence reaches its asymptotic regime.
                const bool can_revert =
                    extrapolate_start == 0 ? (k > min_revert_level) : (effective_points > 3);
                if (can_revert && rel_error > prev_rel_error) {
                    // The worsened extrapolation rejects this level only from
                    // the reported c/U/HV Romberg tables.  The raw mesh root
                    // has already been found and remains useful to derivative
                    // consumers when their requested trace reaches this
                    // level.  Preserve the legacy trace prefix otherwise.
                    const bool retain_root_for_trace =
                        k <= minimum_trace_level;
                    c_table.pop_back();
                    if (!retain_root_for_trace) {
                        root_history.pop_back();
                    }
                    if (!phase_only) {
                        U_table.pop_back();
                        HV_table.pop_back();
                    }
                    converged_level = k - 1;
                    rel_error = prev_rel_error;
                    if (k < minimum_trace_level) {
                        // Freeze the accepted phase table and continue with
                        // root-only refinement until the trace contract is
                        // satisfied.  Any subsequent root failure still
                        // fails closed in the phase_table_complete path.
                        phase_table_complete = true;
                        continue;
                    }
                    break;
                }

                const bool can_stop =
                    extrapolate_start == 0 ? (k >= min_stop_level) : (effective_points >= 3);
                if (can_stop && rel_error < tolerance) {
                    converged_level = k;
                    if (k >= minimum_trace_level) break;
                    phase_table_complete = true;
                }

                prev_rel_error = rel_error;
            }
        }

        // max_steps remains the last level allowed to affect the reported
        // dispersion tuple.  A deeper minimum_trace_level extends only the
        // raw-root trace for derivative consumers.
        if (!phase_table_complete && k == max_steps &&
            k < minimum_trace_level) {
            phase_table_complete = true;
        }
    }

    // Final extrapolation
    auto [c_final, c_err] = extrapolate_from_start(c_table);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double U_final = nan;
    double HV_final = nan;
    if (!phase_only) {
        U_final = extrapolate_from_start(U_table).first;
        HV_final = extrapolate_from_start(HV_table).first;
    }

    RombergResult result;
    result.phase_velocity = c_final;
    result.group_velocity = U_final;
    result.hv_ratio = HV_final;
    result.converged_level = converged_level;
    result.relative_error = rel_error;
    // Returned roots own their memory; reusable capacity stays in the workspace.
    if(workspace) result.root_trace.phase_velocity_roots = root_history;
    else result.root_trace.phase_velocity_roots = std::move(root_history);
    result.root_trace.phase_accepted_level = converged_level;
    result.root_trace.generated_level =
        static_cast<int>(result.root_trace.phase_velocity_roots.size()) - 1;
    result.root_trace.extrapolate_start_level = extrapolate_start;

    return result;
}

// Dedicated GRID/Romberg loop for frequency curvature.  This intentionally
// does not extend compute_with_romberg_core: the established dispersion path
// retains its result type, phase-velocity stopping rule, and bit-level call
// sequence.  Curvature is evaluated through max_steps because convergence of
// c alone does not imply convergence of dU/domega.
template <typename MeshProvider, typename RootFinder, typename RootResidualCalc,
          typename FrequencyCurvatureCalc>
FrequencyCurvatureRombergResult compute_frequency_curvature_with_romberg_core(
    MeshProvider&& get_mesh,
    double period,
    double c_initial_guess,
    RootFinder&& find_root,
    RootResidualCalc&& calc_root_residual,
    FrequencyCurvatureCalc&& calc_frequency_curvature,
    int max_steps = 6,
    int extrapolate_start_level = 0
) {
    if (!(period > 0.0) || !std::isfinite(period)) {
        throw std::invalid_argument("period must be positive and finite");
    }
    if (max_steps < 0) {
        throw std::invalid_argument("max_steps must be non-negative");
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto unsafe_result = [nan](FrequencyCurvatureStatus status,
                               int accepted_level,
                               double maximum_root_residual) {
        return FrequencyCurvatureRombergResult{
            nan, nan, nan, nan, nan, nan,
            accepted_level, nan, nan, maximum_root_residual, status
        };
    };

    std::vector<double> c_table;
    std::vector<double> U_table;
    std::vector<double> U_omega_table;
    std::vector<double> root_residual_table;

    const double omega = 2.0 * M_PI / period;
    double c_guess = c_initial_guess;
    double phase_relative_error = 1.0;
    double previous_phase_relative_error =
        std::numeric_limits<double>::infinity();
    int accepted_level = max_steps;

    const int min_revert_level = std::min(4, max_steps - 1);
    const int extrapolate_start = std::max(0, extrapolate_start_level);

    auto extrapolate_from_start = [extrapolate_start](
        const std::vector<double>& table) {
        if (extrapolate_start <= 0 ||
            static_cast<size_t>(extrapolate_start) >= table.size()) {
            return neville_extrapolation(table);
        }
        std::vector<double> active(
            table.begin() + extrapolate_start, table.end());
        return neville_extrapolation(active);
    };

    auto accepted_maximum_root_residual = [&root_residual_table]() {
        if (root_residual_table.empty()) return 0.0;
        return *std::max_element(
            root_residual_table.begin(), root_residual_table.end());
    };

    for (int k = 0; k <= max_steps; ++k) {
        auto mesh = get_mesh(k);
        const double c = find_root(mesh, omega, c_guess, c_table);
        const double root_residual = calc_root_residual(mesh, omega, c);
        if (!std::isfinite(c) || !(c > 0.0) ||
            !std::isfinite(root_residual) ||
            root_residual > kAdaptiveResidualLimit) {
            const double diagnostic_residual = std::isfinite(root_residual)
                ? std::max(accepted_maximum_root_residual(), root_residual)
                : root_residual;
            return unsafe_result(
                FrequencyCurvatureStatus::UnsafeRoot, k - 1,
                diagnostic_residual);
        }

        const auto level = calc_frequency_curvature(mesh, omega, c);
        if (!std::isfinite(level.group_velocity) ||
            !std::isfinite(level.group_velocity_omega_derivative) ||
            std::abs(level.group_velocity) < 1e-20) {
            return unsafe_result(
                FrequencyCurvatureStatus::UnsafeDerivative, k - 1,
                accepted_maximum_root_residual());
        }

        c_guess = c;
        c_table.push_back(c);
        U_table.push_back(level.group_velocity);
        U_omega_table.push_back(level.group_velocity_omega_derivative);
        root_residual_table.push_back(root_residual);

        const int table_start = std::min(
            extrapolate_start, static_cast<int>(c_table.size()) - 1);
        const int effective_points =
            static_cast<int>(c_table.size()) - table_start;
        if (effective_points >= 2) {
            const auto [c_ext, c_err] = extrapolate_from_start(c_table);
            if (std::isfinite(c_ext) && std::abs(c_ext) > 1e-20) {
                phase_relative_error = c_err / std::abs(c_ext);
                const bool can_revert = extrapolate_start == 0
                    ? (k > min_revert_level)
                    : (effective_points > 3);
                if (can_revert &&
                    phase_relative_error > previous_phase_relative_error) {
                    c_table.pop_back();
                    U_table.pop_back();
                    U_omega_table.pop_back();
                    root_residual_table.pop_back();
                    accepted_level = k - 1;
                    phase_relative_error = previous_phase_relative_error;
                    break;
                }
                previous_phase_relative_error = phase_relative_error;
            }
        }
    }

    const auto [phase_velocity, phase_error] =
        extrapolate_from_start(c_table);
    const auto [group_velocity, group_error] =
        extrapolate_from_start(U_table);
    const auto [group_velocity_omega_derivative, derivative_error] =
        extrapolate_from_start(U_omega_table);
    (void)phase_error;
    (void)group_error;

    if (!std::isfinite(phase_velocity) ||
        !std::isfinite(group_velocity) ||
        !std::isfinite(group_velocity_omega_derivative) ||
        std::abs(group_velocity) < 1e-20) {
        return unsafe_result(
            FrequencyCurvatureStatus::UnsafeDerivative, accepted_level,
            accepted_maximum_root_residual());
    }

    const double wavenumber_omega_second_derivative =
        -group_velocity_omega_derivative /
        (group_velocity * group_velocity);
    const double dimensionless_phase_group_dispersion =
        std::abs(1.0 - phase_velocity / group_velocity);
    const double dimensionless_frequency_curvature =
        std::abs((omega / group_velocity) *
                 group_velocity_omega_derivative);

    if (!std::isfinite(wavenumber_omega_second_derivative) ||
        !std::isfinite(dimensionless_phase_group_dispersion) ||
        !std::isfinite(dimensionless_frequency_curvature)) {
        return unsafe_result(
            FrequencyCurvatureStatus::UnsafeDerivative, accepted_level,
            accepted_maximum_root_residual());
    }

    return FrequencyCurvatureRombergResult{
        phase_velocity,
        group_velocity,
        group_velocity_omega_derivative,
        wavenumber_omega_second_derivative,
        dimensionless_phase_group_dispersion,
        dimensionless_frequency_curvature,
        accepted_level,
        phase_relative_error,
        derivative_error,
        accepted_maximum_root_residual(),
        FrequencyCurvatureStatus::Ok
    };
}
