#pragma once

// Shared geometry for flat and spherical-aware GRID refinement.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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

struct GridSublayerGeometry {
    int parent_node = -1;
    double alpha = 0.0;
    double thickness = 0.0;
    double spherical_midpoint = 0.0;
    double radius_ratio = 1.0;
    double density_scale = 1.0;
    double elastic_scale = 1.0;
    double velocity_scale = 1.0;
    bool fluid = false;
};

inline double grid_flat_depth_from_spherical_depth(double depth, double radius) {
    if (!std::isfinite(depth) || depth < 0.0 || depth >= radius) {
        throw std::invalid_argument("spherical depth must satisfy 0 <= depth < planet_radius");
    }
    return radius * std::log(radius / (radius - depth));
}

inline double grid_spherical_depth_from_flat_depth(double flat_depth, double radius) {
    if (!std::isfinite(flat_depth) || flat_depth < 0.0) {
        throw std::invalid_argument("flat depth must be finite and non-negative");
    }
    return radius * (1.0 - std::exp(-flat_depth / radius));
}

inline void grid_fill_flattening_scales(
    GridSublayerGeometry& g,
    const GridRefinementPolicy& policy)
{
    if (policy.is_flat()) return;
    const double q = g.radius_ratio;
    g.velocity_scale = 1.0 / q;
    if (policy.kind == GridRefinementPolicyKind::SphericalBA97) {
        g.density_scale = g.fluid ? 1.0
                                  : std::pow(q, 2.0 - policy.flattening_exponent);
        g.elastic_scale = g.fluid
            ? std::pow(q, -2.0)
            : std::pow(q, -policy.flattening_exponent);
    } else {
        g.density_scale = std::pow(q, policy.flattening_exponent);
        g.elastic_scale = g.density_scale * g.velocity_scale * g.velocity_scale;
    }
}

inline void validate_grid_interfaces(const std::vector<double>& depths,
                                     const std::vector<bool>& flags) {
    if (depths.size() != flags.size())
        throw std::invalid_argument("Grid depth and discontinuity counts differ");
    for (size_t i = 0; i < depths.size(); ++i) {
        const bool paired = i+1 < depths.size() && depths[i] == depths[i+1];
        if (flags[i] != paired || (paired && i && depths[i-1] == depths[i]))
            throw std::invalid_argument("Expected a flagged two-node discontinuity at node "
                + std::to_string(i) + ", depth " + std::to_string(depths[i]) + " km");
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
        const double d0 = depths[i];
        const double d1 = depths[i + 1];
        if (d1 - d0 < 1.0e-12) continue;

        const double z0 = policy.is_flat()
            ? d0 : grid_flat_depth_from_spherical_depth(d0, policy.planet_radius);
        const double z1 = policy.is_flat()
            ? d1 : grid_flat_depth_from_spherical_depth(d1, policy.planet_radius);
        const double dz = (z1 - z0) / nsub;
        for (int j = 0; j < nsub; ++j) {
            const double zmid = z0 + (j + 0.5) * dz;
            const double dmid = policy.is_flat()
                ? d0 + (j + 0.5) * (d1 - d0) / nsub
                : grid_spherical_depth_from_flat_depth(zmid, policy.planet_radius);
            GridSublayerGeometry g;
            g.parent_node = static_cast<int>(i);
            g.alpha = policy.is_flat()
                ? (j + 0.5) / nsub
                : (dmid - d0) / (d1 - d0);
            g.thickness = dz;
            g.spherical_midpoint = dmid;
            g.radius_ratio = policy.is_flat()
                ? 1.0 : (policy.planet_radius - dmid) / policy.planet_radius;
            g.fluid = fluid_nodes[i];
            grid_fill_flattening_scales(g, policy);
            out.push_back(g);
        }
    }
}

inline std::vector<GridSublayerGeometry> build_grid_sublayer_geometry(
    const std::vector<double>& depths, const std::vector<bool>& discontinuities,
    const std::vector<bool>& fluids, int level, const GridRefinementPolicy& policy) {
    std::vector<GridSublayerGeometry> out;
    fill_grid_sublayer_geometry(depths, discontinuities, fluids, level, policy, out);
    return out;
}
