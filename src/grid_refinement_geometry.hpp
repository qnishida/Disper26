#pragma once

// Shared geometry for flat and spherical-aware GRID refinement.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

enum class GridRefinementPolicyKind {
    Flat,
    SphericalBA97,
    SphericalExperimentalDensity,
};

struct GridRefinementPolicy {
    GridRefinementPolicyKind kind = GridRefinementPolicyKind::Flat;
    double planet_radius = 6371.0;
    double flattening_exponent = -2.275;

    bool is_flat() const { return kind == GridRefinementPolicyKind::Flat; }
};

inline GridRefinementPolicy make_grid_refinement_policy(
    const std::string& name,
    double planet_radius = 6371.0,
    double flattening_exponent = std::numeric_limits<double>::quiet_NaN())
{
    GridRefinementPolicy policy;
    policy.planet_radius = planet_radius;
    if (name == "flat" || name == "Flat") {
        policy.kind = GridRefinementPolicyKind::Flat;
        policy.flattening_exponent = -2.275;
        return policy;
    }
    if (name == "spherical_ba97" || name == "spherical-ba97" || name == "BA97") {
        policy.kind = GridRefinementPolicyKind::SphericalBA97;
        policy.flattening_exponent = std::isfinite(flattening_exponent)
            ? flattening_exponent : -2.0;
    } else if (name == "spherical_experimental_density" ||
               name == "spherical-experimental-density" ||
               name == "experimental_density") {
        policy.kind = GridRefinementPolicyKind::SphericalExperimentalDensity;
        policy.flattening_exponent = std::isfinite(flattening_exponent)
            ? flattening_exponent : -2.275;
    } else {
        throw std::invalid_argument("unknown grid refinement policy: " + name);
    }
    if (!std::isfinite(policy.planet_radius) || policy.planet_radius <= 0.0) {
        throw std::invalid_argument("planet_radius must be finite and positive");
    }
    if (!std::isfinite(policy.flattening_exponent)) {
        throw std::invalid_argument("flattening_exponent must be finite");
    }
    return policy;
}

template <typename T>
struct GridSublayerGeometryT {
    int parent_node = -1;
    T alpha{0.0};
    T thickness{0.0};
    T spherical_midpoint{0.0};
    T radius_ratio{1.0};
    T density_scale{1.0};
    T elastic_scale{1.0};
    T velocity_scale{1.0};
    bool fluid = false;
};

using GridSublayerGeometry = GridSublayerGeometryT<double>;

// The exponent is a policy constant at every AD level.
inline double grid_power(double q, double exponent) { return std::pow(q, exponent); }

template <typename T>
T grid_power(const T& q, double exponent) {
    T result;
    result.value() = grid_power(q.value(), exponent);
    result.derivatives() = q.derivatives() * (exponent * grid_power(q.value(), exponent - 1.0));
    return result;
}

template <typename T>
T grid_flat_depth_from_spherical_depth(const T& depth, double radius) {
    if constexpr (std::is_same_v<T, double>) {
        if (!std::isfinite(depth) || depth < 0.0 || depth >= radius)
            throw std::invalid_argument("spherical depth must satisfy 0 <= depth < planet_radius");
    }
    using std::log;
    return radius * log(T(radius / (radius - depth)));
}

template <typename T>
T grid_spherical_depth_from_flat_depth(const T& flat_depth, double radius) {
    if constexpr (std::is_same_v<T, double>) {
        if (!std::isfinite(flat_depth) || flat_depth < 0.0)
            throw std::invalid_argument("flat depth must be finite and non-negative");
    }
    using std::exp;
    return radius * (1.0 - exp(T(-flat_depth / radius)));
}

template <typename T>
inline void grid_fill_flattening_scales(
    GridSublayerGeometryT<T>& g,
    const GridRefinementPolicy& policy)
{
    if (policy.is_flat()) return;
    const T& q = g.radius_ratio;
    g.velocity_scale = 1.0 / q;
    if (policy.kind == GridRefinementPolicyKind::SphericalBA97) {
        g.density_scale = g.fluid ? T(1.0)
                                  : grid_power(q, 2.0 - policy.flattening_exponent);
        g.elastic_scale = g.fluid
            ? grid_power(q, -2.0)
            : grid_power(q, -policy.flattening_exponent);
    } else {
        g.density_scale = grid_power(q, policy.flattening_exponent);
        g.elastic_scale = g.density_scale * g.velocity_scale * g.velocity_scale;
    }
}

constexpr double minimum_grid_spacing_km = 1e-12;

inline void validate_grid_interfaces(const std::vector<double>& depths,
                                     const std::vector<bool>& flags) {
    if (depths.size() != flags.size())
        throw std::invalid_argument("Grid depth and discontinuity counts differ");
    for (size_t i = 0; i < depths.size(); ++i) {
        if (i && depths[i] > depths[i-1] && depths[i] - depths[i-1] < minimum_grid_spacing_km)
            throw std::invalid_argument("Positive GRID depth intervals must be at least 1e-12 km");
        const bool paired = i+1 < depths.size() && depths[i] == depths[i+1];
        if (flags[i] != paired || (paired && i && depths[i-1] == depths[i]))
            throw std::invalid_argument("Expected a flagged two-node discontinuity at node "
                + std::to_string(i) + ", depth " + std::to_string(depths[i]) + " km");
    }
}

// Topology and primal depths have already been validated by the caller.
template <typename T>
void refine_grid_geometry(const std::vector<T>& depths, const std::vector<bool>& is_discontinuity,
    const std::vector<bool>& fluid_nodes, int refinement_level, const GridRefinementPolicy& policy,
    std::vector<GridSublayerGeometryT<T>>& out) {
    const int nsub = 1 << refinement_level;
    out.clear();
    for (size_t i = 0; i + 1 < depths.size(); ++i) {
        if (is_discontinuity[i]) {
            continue;
        }
        if (fluid_nodes[i] != fluid_nodes[i + 1]) {
            throw std::invalid_argument(
                "fluid/solid transitions require a duplicate-depth discontinuity");
        }
        const T d0 = depths[i];
        const T d1 = depths[i + 1];

        const T z0 = policy.is_flat()
            ? d0 : grid_flat_depth_from_spherical_depth(d0, policy.planet_radius);
        const T z1 = policy.is_flat()
            ? d1 : grid_flat_depth_from_spherical_depth(d1, policy.planet_radius);
        const T dz = (z1 - z0) / nsub;
        for (int j = 0; j < nsub; ++j) {
            const T zmid = z0 + (j + 0.5) * dz;
            const T dmid = policy.is_flat()
                ? T(d0 + (j + 0.5) * (d1 - d0) / nsub)
                : grid_spherical_depth_from_flat_depth(zmid, policy.planet_radius);
            GridSublayerGeometryT<T> g;
            g.parent_node = static_cast<int>(i);
            g.alpha = policy.is_flat()
                ? T((j + 0.5) / nsub)
                : (dmid - d0) / (d1 - d0);
            g.thickness = dz;
            g.spherical_midpoint = dmid;
            g.radius_ratio = policy.is_flat()
                ? T(1.0) : (policy.planet_radius - dmid) / policy.planet_radius;
            g.fluid = fluid_nodes[i];
            grid_fill_flattening_scales(g, policy);
            out.push_back(g);
        }
    }
}

inline void fill_grid_sublayer_geometry(
    const std::vector<double>& depths,
    const std::vector<bool>& is_discontinuity,
    const std::vector<bool>& fluid_nodes,
    int refinement_level,
    const GridRefinementPolicy& policy,
    std::vector<GridSublayerGeometry>& out)
{
    if (refinement_level < 0 || refinement_level > 30) {
        throw std::invalid_argument("refinement_level must be in [0, 30]");
    }
    if (depths.size() < 2 || is_discontinuity.size() != depths.size() ||
        fluid_nodes.size() != depths.size()) {
        throw std::invalid_argument("invalid grid geometry input sizes");
    }
    for (size_t i = 1; i < depths.size(); ++i) {
        if (!std::isfinite(depths[i]) || depths[i] < depths[i - 1]) {
            throw std::invalid_argument("grid depths must be finite and non-decreasing");
        }
    }
    if (!std::isfinite(depths.front())) {
        throw std::invalid_argument("grid depths must be finite");
    }
    if (!policy.is_flat()) {
        if (depths.back() >= policy.planet_radius) {
            throw std::invalid_argument("spherical grid depth must be shallower than planet_radius");
        }
        for (size_t i = 0; i < depths.size(); ++i) {
            (void)grid_flat_depth_from_spherical_depth(depths[i], policy.planet_radius);
        }
    }

    validate_grid_interfaces(depths, is_discontinuity);
    refine_grid_geometry(depths, is_discontinuity, fluid_nodes, refinement_level, policy, out);
}

inline std::vector<GridSublayerGeometry> build_grid_sublayer_geometry(
    const std::vector<double>& depths, const std::vector<bool>& discontinuities,
    const std::vector<bool>& fluids, int level, const GridRefinementPolicy& policy) {
    std::vector<GridSublayerGeometry> out;
    fill_grid_sublayer_geometry(depths, discontinuities, fluids, level, policy, out);
    return out;
}
