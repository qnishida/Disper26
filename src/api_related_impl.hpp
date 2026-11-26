#pragma once

// Included by api_impl.hpp after its shared model, solution and trace helpers.
namespace disper26 { namespace api {
namespace detail {
inline int related_level(const ModeSolution& solution) {
    return solution.model().data().kind == "grid"
        ? std::max(0, solution.diagnostics().accepted_level) : 0;
}

inline Diagnostics related_diagnostic(const ModeSolution& solution,
    const std::string& method, const Options& options, int level) {
    auto out = diagnostic(method, options);
    if (solution.model().data().kind == "grid") {
        out.accepted_level = level;
        out.generated_level = level;
        out.messages.push_back("Evaluated on the refined GRID mesh at level " +
            std::to_string(level) + " using its own root; wavefields and energy "
            "integrals are not extrapolated.");
    }
    return out;
}

inline void related_require_method(const Options& options,
    const std::string& allowed) {
    check_options(options);
    if (options.method != "standard" && options.method != allowed)
        unsupported("This operation supports method='standard' or '" + allowed + "'.");
}

inline bool related_has_fluid(const VelocityGridRefinement& stack) {
    return hasLiquidLayer(stack.layers) || isLiquidLayer(stack.halfspace);
}
} // namespace detail

Eigenfunctions compute_eigenfunctions(const ModeSolution& solution, int num_sub_layers) {
    detail::require(num_sub_layers > 0, "num_sub_layers must be positive");
    const auto& data = Access::solution(solution);
    const int level = detail::related_level(solution);
    const auto stack = detail::mesh(data.model, level);
    const double root = detail::trace_roots(solution, level).at(level);
    const double omega = 2.0 * M_PI / data.period;
    Eigenfunctions out;
    out.diagnostics = detail::related_diagnostic(
        solution, "eigenfunctions", data.options, level);
    if (data.wave == "Love") {
        if (detail::related_has_fluid(stack))
            detail::unsupported("Love eigenfunctions require solid materials.");
        const auto points = computeLoveEigenfunctions(stack.layers,
            stack.thicknesses, stack.halfspace, omega, root, num_sub_layers);
        for (const auto& point : points) {
            out.depth.push_back(point.depth);
            out.components["uy"].push_back(point.uy);
            out.components["tyz"].push_back(point.tyz);
            out.is_fluid.push_back(false);
        }
    } else {
        const auto points = computeRayleighEigenfunctions(stack.layers,
            stack.thicknesses, stack.halfspace, omega, root, num_sub_layers);
        for (const auto& point : points) {
            out.depth.push_back(point.depth);
            out.components["ur"].push_back(point.ur);
            out.components["uz"].push_back(point.uz);
            out.components["szz"].push_back(point.szz);
            out.components["szr"].push_back(point.szr);
            out.is_fluid.push_back(point.is_fluid);
        }
    }
    for (const auto& entry : out.components) {
        for (const auto& value : entry.second)
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                throw Error("numerical_failure", "Eigenfunctions contain nonfinite values.");
        out.diagnostics.availability[entry.first] = "computed";
    }
    detail::finish(out.diagnostics, false);
    return out;
}

EnergyIntegrals compute_energy_integrals(const ModeSolution& solution,
    const Options& options) {
    detail::related_require_method(options, "energy");
    const auto& data = Access::solution(solution);
    const int level = detail::related_level(solution);
    const auto stack = detail::mesh(data.model, level);
    const double root = detail::trace_roots(solution, level).at(level);
    const double omega = 2.0 * M_PI / data.period;
    const bool fluid = detail::related_has_fluid(stack);
    std::size_t top_fluid = 0;
    while (top_fluid < stack.layers.size() && isLiquidLayer(stack.layers[top_fluid])) ++top_fluid;
    const bool fluid_top_only = !isLiquidLayer(stack.halfspace) &&
        std::none_of(stack.layers.begin() + top_fluid, stack.layers.end(),
            [](const PhysicalParameters<double>& p) { return isLiquidLayer(p); });
    const bool analytical = options.integration == "analytical" ||
        (options.integration == "auto" && fluid_top_only);
    if (options.integration != "auto" && options.integration != "analytical" &&
        options.integration != "trapezoidal")
        throw Error("invalid_argument", "integration must be auto, analytical or trapezoidal");
    EnergyIntegralResult<double> integral{};
    auto copy_integrals = [&](const EnergyIntegralResult<double>& result) {
        // Analytical results leave groupVelocity_Energy unset; copy only the
        // defined integrals and form the group velocity below.
        integral.I1 = result.I1; integral.I2 = result.I2; integral.I3 = result.I3;
    };
    if (data.wave == "Love") {
        if (fluid) detail::unsupported("Love energy integrals require solid materials.");
        const auto points = computeLoveEigenfunctions(stack.layers,
            stack.thicknesses, stack.halfspace, omega, root, options.num_sub_layers);
        if (analytical)
            copy_integrals(computeLoveEnergyIntegrals_Analytical(points, stack.layers,
                stack.thicknesses, stack.halfspace, omega, root));
        else
            copy_integrals(computeLoveEnergyIntegrals(points, omega, root, stack.layers,
                stack.thicknesses, stack.halfspace));
    } else {
        const auto points = computeRayleighEigenfunctions(stack.layers,
            stack.thicknesses, stack.halfspace, omega, root, options.num_sub_layers);
        if (analytical && fluid) {
            // Fluid layers use the boundary-state moments shared with the
            // variational kernels. The point-based analytical routine requires solids.
            const auto states = propagateRayleighEigenstates<double>(stack.layers,
                stack.thicknesses, stack.halfspace, omega, root, 1, false);
            const auto energy = computeRayleighEnergyIntegrals_Analytical<double>(states,
                stack.layers, stack.thicknesses, stack.halfspace, omega, root, false);
            integral.I1 = energy.I1; integral.I2 = energy.I2; integral.I3 = energy.I3;
        } else if (analytical)
            copy_integrals(computeRayleighEnergyIntegrals_Analytical(points, stack.layers,
                stack.thicknesses, stack.halfspace, omega, root));
        else
            copy_integrals(computeRayleighEnergyIntegrals(points, stack.layers,
                stack.thicknesses, stack.halfspace, omega, root));
    }
    EnergyIntegrals out;
    out.I1 = integral.I1; out.I2 = integral.I2; out.I3 = integral.I3;
    out.phase_velocity = root;
    out.group_velocity = integral.I3 / (root * integral.I1);
    out.diagnostics = detail::related_diagnostic(solution,
        analytical ? "analytical" : "trapezoidal", options, level);
    if (!std::isfinite(out.I1) || !std::isfinite(out.I2) ||
        !std::isfinite(out.I3) || !std::isfinite(out.group_velocity))
        throw Error("numerical_failure", "Energy integrals contain nonfinite values.");
    for (const char* quantity : {"I1", "I2", "I3", "phase_velocity", "group_velocity"})
        out.diagnostics.availability[quantity] = "computed";
    detail::finish(out.diagnostics, options.strict);
    return out;
}

FrequencyCurvature compute_frequency_curvature(const ModeSolution& solution,
    const Options& options) {
    detail::related_require_method(options, "ad");
    detail::ensure_mode_verified(solution);
    const auto& data = Access::solution(solution);
    const auto type = parseWaveType(data.wave);
    FrequencyCurvature out;
    out.diagnostics = detail::diagnostic("ad", options);
    out.diagnostics.mode_verified = true;
    const double omega = 2.0 * M_PI / data.period;
    if (detail::is_layer_cake_kind(data.model.data().kind)) {
        const auto stack = detail::mesh(data.model, 0);
        const auto result = compute_frequency_curvature_ad_level(stack.layers,
            stack.thicknesses, stack.halfspace, omega, data.c, type);
        out.phase_velocity = data.c;
        out.group_velocity = result.group_velocity;
        out.group_velocity_omega_derivative = result.group_velocity_omega_derivative;
        out.wavenumber_omega_second_derivative =
            -result.group_velocity_omega_derivative /
            (result.group_velocity * result.group_velocity);
    } else {
        const auto roots = detail::trace_roots(solution, options.max_steps);
        auto refiner = [&](int level) { return detail::mesh(data.model, level); };
        auto find_root = [&](const VelocityGridRefinement&, double, double,
                             const std::vector<double>& history) {
            return roots.at(history.size());
        };
        auto residual = [type](const VelocityGridRefinement& stack, double w, double c) {
            return std::abs(calculateDispersionPhaseValueDouble(stack.layers,
                stack.thicknesses, stack.halfspace, w, c, type));
        };
        auto curvature = [type](const VelocityGridRefinement& stack, double w, double c) {
            return compute_frequency_curvature_ad_level(stack.layers,
                stack.thicknesses, stack.halfspace, w, c, type);
        };
        const auto result = compute_frequency_curvature_with_romberg_core(refiner,
            data.period, data.c, find_root, residual, curvature, options.max_steps);
        if (result.status != FrequencyCurvatureStatus::Ok)
            throw Error("numerical_failure", result.status == FrequencyCurvatureStatus::UnsafeRoot
                ? "Frequency curvature encountered an unsafe root."
                : "Frequency curvature encountered an unsafe derivative.");
        out.phase_velocity = result.phase_velocity;
        out.group_velocity = result.group_velocity;
        out.group_velocity_omega_derivative = result.group_velocity_omega_derivative;
        out.wavenumber_omega_second_derivative = result.wavenumber_omega_second_derivative;
        out.diagnostics.accepted_level = result.accepted_level;
        out.diagnostics.generated_level = static_cast<int>(roots.size()) - 1;
        out.diagnostics.messages.push_back(
            "Curvature uses its own Romberg stopping rule with the solution's shared roots; "
            "its returned phase velocity can differ from the original solution.");
        // The existing curvature engine reports a step estimate, not a derivative
        // convergence test. A small phase step is not derivative convergence.
        out.diagnostics.messages.push_back("Derivative extrapolation absolute step: " +
            std::to_string(result.group_velocity_omega_derivative_error));
    }
    if (!std::isfinite(out.phase_velocity) || !std::isfinite(out.group_velocity) ||
        !std::isfinite(out.group_velocity_omega_derivative) ||
        !std::isfinite(out.wavenumber_omega_second_derivative))
        throw Error("numerical_failure", "Frequency curvature contains nonfinite values.");
    for (const char* quantity : {"phase_velocity", "group_velocity",
            "group_velocity_omega_derivative", "wavenumber_omega_second_derivative"})
        out.diagnostics.availability[quantity] = "computed";
    detail::finish(out.diagnostics, options.strict);
    return out;
}

Model earth_flattening_transformation(const Model& model) {
    ModelData out = model.data();
    detail::require(!out.flattened, "The model has already been earth-flattened.");
    detail::require(out.geometry.kind != "flat",
        "EFT requires a spherical model with an explicit flattening policy.");
    const auto policy = make_grid_refinement_policy(out.geometry.kind,
        out.geometry.planet_radius, out.geometry.flattening_exponent);
    const bool experimental = policy.kind == GridRefinementPolicyKind::SphericalExperimentalDensity;
    std::vector<PhysicalParameters<double>> materials;
    for (const auto& material : out.materials) materials.push_back(detail::unpack(material));
    std::vector<PhysicalParameters<double>> transformed;
    if (out.kind == "grid") {
        auto result = experimental
            ? py_earthFlatteningGridExperimentalDensity(out.depths, materials,
                out.discontinuities, policy.planet_radius, policy.flattening_exponent)
            : py_earthFlatteningGrid(out.depths, materials, out.discontinuities,
                policy.planet_radius, policy.flattening_exponent);
        out.depths = std::get<0>(result);
        transformed = std::get<1>(result);
        out.halfspace = detail::pack(transformed.back());
    } else {
        double bottom_depth = 0.0;
        for (double thickness : out.thicknesses) bottom_depth += thickness;
        detail::require(bottom_depth < policy.planet_radius,
            "The halfspace top must lie above the planet center.");
        auto result = experimental
            ? py_earthFlatteningLayersExperimentalDensity(materials, out.thicknesses,
                policy.planet_radius, policy.flattening_exponent)
            : py_earthFlatteningLayers(materials, out.thicknesses,
                policy.planet_radius, policy.flattening_exponent);
        transformed = std::get<0>(result);
        out.thicknesses = std::get<1>(result);
        // A homogeneous halfspace is scaled at its top, rather than extending
        // a spherical halfspace through the planet center.
        const auto halfspace = detail::unpack(out.halfspace);
        out.halfspace = detail::pack(experimental
            ? applyEarthFlatteningScaleExperimentalDensity(halfspace,
                policy.planet_radius - bottom_depth, policy.planet_radius,
                policy.flattening_exponent)
            : applyEarthFlatteningScaleBA97(halfspace,
                policy.planet_radius - bottom_depth, policy.planet_radius,
                policy.flattening_exponent));
    }
    out.materials.clear();
    for (const auto& material : transformed) out.materials.push_back(detail::pack(material));
    out.flattening_policy = out.geometry.kind;
    out.geometry.flattening_exponent = policy.flattening_exponent;
    out.geometry.kind = "flat";
    out.flattened = true;
    for (const auto& material : out.materials) detail::check_material(material);
    detail::check_material(out.halfspace);
    return Access::model(std::move(out));
}

Model apply_mineos_physical_dispersion(const Model& reference_model,
    const std::vector<double>& qkappa, const std::vector<double>& qmu,
    double reference_period, double period) {
    ModelData out = reference_model.data();
    detail::require(!out.q_period.has_value(),
        "Q dispersion requires the uncorrected reference model, not a previously corrected model.");
    detail::require(std::isfinite(reference_period) && reference_period > 0.0 &&
        std::isfinite(period) && period > 0.0, "Reference and target periods must be positive and finite.");
    const size_t expected = out.materials.size() + (detail::is_layer_cake_kind(out.kind) ? 1 : 0);
    detail::require(qkappa.size() == expected && qmu.size() == expected,
        "Q arrays must contain one value per material, including the halfspace "
        "(the final node for GRID, one extra entry for Layer-cake).");
    for (size_t index = 0; index < expected; ++index)
        detail::require(std::isfinite(qkappa[index]) && std::isfinite(qmu[index]),
            "Q values must be finite; nonpositive values retain the Mineos inverse-Q convention.");
    for (size_t index = 0; index < out.materials.size(); ++index)
        out.materials[index] = detail::pack(py_applyMineosPhysicalDispersion(
            detail::unpack(out.materials[index]), qkappa[index], qmu[index],
            reference_period, period));
    if (detail::is_layer_cake_kind(out.kind))
        out.halfspace = detail::pack(py_applyMineosPhysicalDispersion(
            detail::unpack(out.halfspace), qkappa.back(), qmu.back(), reference_period, period));
    else
        out.halfspace = out.materials.back();
    // Recompute eta where Q changes the elastic ratios. At a persistent
    // 0/0 node the supplied eta still represents the corrected stiffness.
    for(size_t i=0;i<out.velocity_eta.size();++i) {
        const auto& p=i<out.materials.size()?out.materials[i]:out.halfspace;
        const auto& old=i<out.materials.size()?reference_model.data().materials[i]:reference_model.data().halfspace;
        if(p.A==old.A && p.L==old.L && p.F==old.F) continue;
        if(p.A!=2*p.L) out.velocity_eta[i]=p.F/(p.A-2*p.L);
        else if(p.F!=0 || old.A!=2*old.L || old.F!=0)
            out.velocity_eta[i]=missing_value;
    }
    out.q_period = period;
    out.q_reference_period = reference_period;
    for (const auto& material : out.materials) detail::check_material(material);
    detail::check_material(out.halfspace);
    auto corrected = Access::model(std::move(out));
    if (corrected.data().interpolation == "velocity")
        (void)detail::velocities(corrected);
    return corrected;
}

namespace detail {
// Direct port of the adopted inversion/disper26_inv/gravity.py calculation.
// Preserve its integration, neighboring-frequency tracking and polishing;
// this is a post-correction and does not modify the underlying solution.
inline double gravity_polish_root(const VelocityGridRefinement& stack,
    double omega, double root) {
    auto f = [&](double c) { return py_calculateDispersionValue(stack.layers,
        stack.thicknesses, stack.halfspace, omega, c, "Rayleigh"); };
    double lo = root * (1.0 - 1e-7), hi = root * (1.0 + 1e-7);
    double f_lo = f(lo), f_hi = f(hi);
    if (!std::isfinite(f_lo) || !std::isfinite(f_hi) || f_lo * f_hi > 0.0) return root;
    for (int iteration = 0; iteration < 200; ++iteration) {
        const double mid = 0.5 * (lo + hi);
        if (hi - lo <= 1e-15 * mid) break;
        const double f_mid = f(mid);
        if (!std::isfinite(f_mid)) return root;
        if (f_lo * f_mid <= 0.0) { hi = mid; f_hi = f_mid; }
        else { lo = mid; f_lo = f_mid; }
    }
    return 0.5 * (lo + hi);
}

inline double gravity_delta_c(const VelocityGridRefinement& stack, double omega,
    double c, double u, double gravity, const std::string& region, int subdivisions) {
    const auto points = computeRayleighEigenfunctions(stack.layers,
        stack.thicknesses, stack.halfspace, omega, c, subdivisions);
    const double i1 = computeRayleighEnergyIntegrals(points, stack.layers,
        stack.thicknesses, stack.halfspace, omega, c).I1;
    if (!std::isfinite(i1) || std::abs(i1) <= 1e-30 ||
        !std::isfinite(u) || std::abs(u) <= 1e-30)
        throw Error("numerical_failure", "Gravity correction requires finite nonzero I1 and group velocity.");
    const double k = omega / c;
    const double prefactor = c * c / (omega * omega * u * i1);
    double total = 0.0;
    for (size_t index = 0; index + 1 < points.size(); ++index) {
        const auto& left = points[index]; const auto& right = points[index + 1];
        const double dz = right.depth - left.depth;
        if (dz <= 0.0) continue;
        const double midpoint = 0.5 * (left.depth + right.depth);
        const PhysicalParameters<double>* material = &stack.halfspace;
        double top = 0.0;
        for (size_t layer = 0; layer < stack.layers.size(); ++layer) {
            if (midpoint <= top + stack.thicknesses[layer] + 1e-10) {
                material = &stack.layers[layer]; break;
            }
            top += stack.thicknesses[layer];
        }
        const bool fluid = material->c44 <= 0.0;
        if ((region == "solid" && fluid) || (region == "fluid" && !fluid)) continue;
        const double y1 = 0.5 * (left.uz.real() + right.uz.real());
        const double y3 = 0.5 * (left.ur.real() + right.ur.real());
        total += prefactor * material->rho * k * y1 * y3 * gravity * dz;
    }
    return total;
}
} // namespace detail

GravityCorrection compute_gravity_correction(const ModeSolution& solution,
    double gravity, const std::string& region, int num_sub_layers,
    double domega_relative) {
    const auto& data = Access::solution(solution);
    if (data.wave != "Rayleigh")
        detail::unsupported("The gravity correction is applicable to Rayleigh waves only.");
    if (!detail::is_layer_cake_kind(data.model.data().kind) || data.model.data().geometry.kind != "flat")
        detail::unsupported("The adopted gravity correction requires a flat Layer-cake model.");
    detail::require(std::isfinite(gravity) && gravity >= 0.0,
        "gravity must be a finite nonnegative magnitude in km/s^2");
    detail::require(region == "all" || region == "solid" || region == "fluid",
        "region must be all, solid or fluid");
    detail::require(num_sub_layers > 0, "num_sub_layers must be positive");
    detail::require(std::isfinite(domega_relative) && domega_relative > 0.0 &&
        domega_relative < 1.0, "domega_relative must lie strictly between zero and one");
    const auto stack = detail::mesh(data.model, 0);
    detail::require(stack.halfspace.gravity == 0.0,
        "Gravity correction requires a non-gravitating reference model.");
    for (const auto& material : stack.layers)
        detail::require(material.gravity == 0.0,
            "Gravity correction requires a non-gravitating reference model.");
    detail::ensure_mode_verified(solution);
    auto at = [&](double omega, double near) {
        const auto roots = py_searchRoots(stack.layers, stack.thicknesses,
            stack.halfspace, omega, near * 0.99, near * 1.01, 6000, "Rayleigh");
        if (roots.empty()) throw Error("numerical_failure",
            "Gravity correction could not track the root within one percent of the reference velocity.");
        const auto found = std::min_element(roots.begin(), roots.end(),
            [near](double a, double b) { return std::abs(a - near) < std::abs(b - near); });
        const double c = detail::gravity_polish_root(stack, omega, *found);
        const double u = calculateGroupVelocityFullAD(stack.layers, stack.thicknesses,
            stack.halfspace, omega, c, WaveType::Rayleigh);
        const double delta = detail::gravity_delta_c(stack, omega, c, u,
            gravity, region, num_sub_layers);
        return std::make_tuple(c, u, delta);
    };
    const double omega = 2.0 * M_PI / data.period;
    const auto centre = at(omega, data.c);
    GravityCorrection out;
    out.phase_velocity = std::get<0>(centre);
    out.group_velocity = std::get<1>(centre);
    out.delta_c = std::get<2>(centre);
    const double high = std::get<2>(at(omega * (1.0 + domega_relative), out.phase_velocity));
    const double low = std::get<2>(at(omega * (1.0 - domega_relative), out.phase_velocity));
    out.omega_delta_c_omega_derivative = (high - low) / (2.0 * domega_relative);
    const double c = out.phase_velocity, u = out.group_velocity;
    out.delta_u = (u * u / (c * c)) *
        ((2.0 * c / u - 1.0) * out.delta_c + out.omega_delta_c_omega_derivative);
    if (!std::isfinite(out.delta_c) || !std::isfinite(out.delta_u))
        throw Error("numerical_failure", "Gravity correction contains nonfinite values.");
    out.region = region;
    out.diagnostics = detail::diagnostic("gravity_kg", data.options);
    out.diagnostics.mode_verified = true;
    out.diagnostics.messages.push_back(
        "First-order correction to subtract from observations, in km/s; "
        "the reference solution and its derivatives remain unchanged.");
    out.diagnostics.messages.push_back(
        "Group correction uses a central frequency difference with relative step " +
        std::to_string(domega_relative) + "; neighboring roots are tracked within one percent in velocity.");
    for (const char* quantity : {"phase_velocity", "group_velocity", "delta_c", "delta_u",
            "omega_delta_c_omega_derivative"})
        out.diagnostics.availability[quantity] = "computed";
    detail::finish(out.diagnostics, false);
    return out;
}
}} // namespace disper26::api
