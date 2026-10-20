#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "grid_refinement_geometry.hpp"

template <typename T> struct PhysicalParameters;

template <typename T>
struct VelocityGridRefinementT {
    std::vector<PhysicalParameters<T>> layers;
    std::vector<T> thicknesses;
    PhysicalParameters<T> halfspace;
};

using VelocityGridRefinement = VelocityGridRefinementT<double>;

inline void validate_velocity_grid_sizes(
    const std::vector<double>& depths,
    const std::vector<bool>& is_discontinuity,
    const std::vector<size_t>& field_sizes
) {
    if (depths.empty()) {
        throw std::runtime_error("Velocity grid must contain at least one node.");
    }
    if (is_discontinuity.size() != depths.size()) {
        throw std::runtime_error("Velocity grid size mismatch: is_discontinuity must match depths.");
    }
    for (size_t field_size : field_sizes) {
        if (field_size != depths.size()) {
            throw std::runtime_error("Velocity grid size mismatch: all parameter arrays must match depths.");
        }
    }
    for (size_t i = 1; i < depths.size(); ++i) {
        if (depths[i] < depths[i - 1]) {
            throw std::runtime_error("Velocity grid depths must be monotonically non-decreasing.");
        }
    }
    for (size_t i = 0; i < is_discontinuity.size(); ++i) {
        if (is_discontinuity[i] && (i + 1 >= depths.size() || std::abs(depths[i] - depths[i + 1]) > 1e-10)) {
            throw std::runtime_error("Velocity grid discontinuity flags require duplicated depths.");
        }
    }
}

template <typename T>
T velocity_constant(double value, const T& reference) {
    if constexpr (std::is_same_v<T, double>) {
        return value;
    } else {
        T out;
        out.value() = value;
        out.derivatives() = Eigen::VectorXd::Zero(reference.derivatives().size());
        return out;
    }
}

template <typename T>
PhysicalParameters<T> make_iso_from_velocities(
    const T& rho,
    const T& vp,
    const T& vs
) {
    PhysicalParameters<T> p;
    p.rho = rho;
    p.c44 = rho * vs * vs;
    p.c66 = p.c44;
    p.c11 = rho * vp * vp;
    p.c33 = p.c11;
    p.c13 = p.c11 - velocity_constant<T>(2.0, rho) * p.c44;
    p.gravity = velocity_constant<T>(0.0, rho);
    p.gravity_internal_scale = velocity_constant<T>(1.0, rho);
    p.gravity_surface_scale = velocity_constant<T>(0.0, rho);
    p.gravity_seabed_scale = velocity_constant<T>(0.0, rho);
    return p;
}

template <typename T>
PhysicalParameters<T> make_vti_from_velocities(
    const T& rho,
    const T& vph,
    const T& vpv,
    const T& vsh,
    const T& vsv,
    const T& eta
) {
    PhysicalParameters<T> p;
    p.rho = rho;
    p.c11 = rho * vph * vph;
    p.c33 = rho * vpv * vpv;
    p.c44 = rho * vsv * vsv;
    p.c66 = rho * vsh * vsh;
    p.c13 = eta * (p.c11 - velocity_constant<T>(2.0, rho) * p.c44);
    p.gravity = velocity_constant<T>(0.0, rho);
    p.gravity_internal_scale = velocity_constant<T>(1.0, rho);
    p.gravity_surface_scale = velocity_constant<T>(0.0, rho);
    p.gravity_seabed_scale = velocity_constant<T>(0.0, rho);
    return p;
}

template <typename T>
VelocityGridRefinementT<T> refine_velocity_grid_iso_values(
    const std::vector<double>& depths,
    const std::vector<T>& rhos,
    const std::vector<T>& vps,
    const std::vector<T>& vss,
    const std::vector<bool>& is_discontinuity,
    int refinement_level,
    const GridRefinementPolicy& policy = {}
) {
    validate_velocity_grid_sizes(depths, is_discontinuity, {rhos.size(), vps.size(), vss.size()});
    VelocityGridRefinementT<T> refined;
    std::vector<bool> fluid_nodes(depths.size(), false);
    const auto geometry = build_grid_sublayer_geometry(
        depths, is_discontinuity, fluid_nodes, refinement_level, policy);
    for (const auto& g : geometry) {
            const size_t i = static_cast<size_t>(g.parent_node);
            T t = velocity_constant<T>(g.alpha, rhos[i]);
            T one = velocity_constant<T>(1.0, rhos[i]);
            T rho = rhos[i] * (one - t) + rhos[i + 1] * t;
            T vp = vps[i] * (one - t) + vps[i + 1] * t;
            T vs = vss[i] * (one - t) + vss[i + 1] * t;
            rho *= g.density_scale;
            vp *= g.velocity_scale;
            vs *= g.velocity_scale;
            refined.layers.push_back(make_iso_from_velocities(rho, vp, vs));
            refined.thicknesses.push_back(velocity_constant<T>(g.thickness, rhos[i]));
    }

    GridSublayerGeometry hs_geo;
    hs_geo.radius_ratio = policy.is_flat() ? 1.0
        : (policy.planet_radius - depths.back()) / policy.planet_radius;
    grid_fill_flattening_scales(hs_geo, policy);
    T rho_h = rhos.back() * hs_geo.density_scale;
    T vp_h = vps.back() * hs_geo.velocity_scale;
    T vs_h = vss.back() * hs_geo.velocity_scale;
    refined.halfspace = make_iso_from_velocities(rho_h, vp_h, vs_h);
    return refined;
}

template <typename T>
VelocityGridRefinementT<T> refine_velocity_grid_vti_values(
    const std::vector<double>& depths,
    const std::vector<T>& rhos,
    const std::vector<T>& vphs,
    const std::vector<T>& vpvs,
    const std::vector<T>& vshs,
    const std::vector<T>& vsvs,
    const std::vector<T>& etas,
    const std::vector<bool>& is_discontinuity,
    int refinement_level,
    const GridRefinementPolicy& policy = {}
) {
    validate_velocity_grid_sizes(depths, is_discontinuity, {rhos.size(), vphs.size(), vpvs.size(), vshs.size(), vsvs.size(), etas.size()});
    VelocityGridRefinementT<T> refined;
    std::vector<bool> fluid_nodes(depths.size(), false);
    const auto geometry = build_grid_sublayer_geometry(
        depths, is_discontinuity, fluid_nodes, refinement_level, policy);
    for (const auto& g : geometry) {
            const size_t i = static_cast<size_t>(g.parent_node);
            T t = velocity_constant<T>(g.alpha, rhos[i]);
            T one = velocity_constant<T>(1.0, rhos[i]);
            T rho = rhos[i] * (one - t) + rhos[i + 1] * t;
            T vph = vphs[i] * (one - t) + vphs[i + 1] * t;
            T vpv = vpvs[i] * (one - t) + vpvs[i + 1] * t;
            T vsh = vshs[i] * (one - t) + vshs[i + 1] * t;
            T vsv = vsvs[i] * (one - t) + vsvs[i + 1] * t;
            T eta = etas[i] * (one - t) + etas[i + 1] * t;
            rho *= g.density_scale;
            vph *= g.velocity_scale;
            vpv *= g.velocity_scale;
            vsh *= g.velocity_scale;
            vsv *= g.velocity_scale;
            refined.layers.push_back(make_vti_from_velocities(rho, vph, vpv, vsh, vsv, eta));
            refined.thicknesses.push_back(velocity_constant<T>(g.thickness, rhos[i]));
    }

    GridSublayerGeometry hs_geo;
    hs_geo.radius_ratio = policy.is_flat() ? 1.0
        : (policy.planet_radius - depths.back()) / policy.planet_radius;
    grid_fill_flattening_scales(hs_geo, policy);
    T rho_h = rhos.back() * hs_geo.density_scale;
    T vph_h = vphs.back() * hs_geo.velocity_scale;
    T vpv_h = vpvs.back() * hs_geo.velocity_scale;
    T vsh_h = vshs.back() * hs_geo.velocity_scale;
    T vsv_h = vsvs.back() * hs_geo.velocity_scale;
    refined.halfspace = make_vti_from_velocities(rho_h, vph_h, vpv_h, vsh_h, vsv_h, etas.back());
    return refined;
}

inline VelocityGridRefinement refine_velocity_grid_iso(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vps,
    const std::vector<double>& vss,
    const std::vector<bool>& is_discontinuity,
    int refinement_level,
    const GridRefinementPolicy& policy = {}
) {
    return refine_velocity_grid_iso_values(depths, rhos, vps, vss, is_discontinuity, refinement_level, policy);
}

inline VelocityGridRefinement refine_velocity_grid_vti(
    const std::vector<double>& depths,
    const std::vector<double>& rhos,
    const std::vector<double>& vphs,
    const std::vector<double>& vpvs,
    const std::vector<double>& vshs,
    const std::vector<double>& vsvs,
    const std::vector<double>& etas,
    const std::vector<bool>& is_discontinuity,
    int refinement_level,
    const GridRefinementPolicy& policy = {}
) {
    return refine_velocity_grid_vti_values(depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_discontinuity, refinement_level, policy);
}
