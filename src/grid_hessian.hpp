#pragma once
// Grid Mode Weak-Form Hessian with Romberg Acceleration
// Requires: disper26.cpp, sensitivity_utils.hpp, romberg_engine.hpp all included first.
//
// Key design: all internal computation uses VTI (elastic-moduli) parameterization,
// which is linear in the Mesher's linear interpolation of moduli. This ensures the
// chain rule J^T g_sub → g_node is exact. If isotropic output is requested, a
// Jacobian-based conversion (J^T H J + second-derivative correction) is applied at
// the end, evaluated at NODE parameters.

// ---------------------------------------------------------------------------
// VTI → Isotropic conversion helpers
// ---------------------------------------------------------------------------

// Build block-diagonal Jacobian J of shape (M*6) × (M*3).
// Column order for isotropic: [vs, vp, rho] per node.
// Row order for VTI:          [rho, c11(A), c33(C), c13(F), c44(L), c66(N)] per node.
inline Eigen::MatrixXd build_iso_to_vti_jacobian(
    const std::vector<PhysicalParameters<double>>& nodes)
{
    int M = (int)nodes.size();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(M * 6, M * 3);
    for (int i = 0; i < M; ++i) {
        if (isLiquidLayer(nodes[i])) {
            continue;
        }
        double vs  = nodes[i].vs(), vp = nodes[i].vp(), rho = nodes[i].rho;
        // Rho = rho
        J(i*6+0, i*3+2) = 1.0;
        // A = c11 = rho*vp²
        J(i*6+1, i*3+1) = 2*rho*vp;  J(i*6+1, i*3+2) = vp*vp;
        // C = c33 = rho*vp²
        J(i*6+2, i*3+1) = 2*rho*vp;  J(i*6+2, i*3+2) = vp*vp;
        // F = c13 = rho*(vp²-2vs²)
        J(i*6+3, i*3+0) = -4*rho*vs; J(i*6+3, i*3+1) = 2*rho*vp;
        J(i*6+3, i*3+2) = vp*vp - 2*vs*vs;
        // L = c44 = rho*vs²
        J(i*6+4, i*3+0) = 2*rho*vs;  J(i*6+4, i*3+2) = vs*vs;
        // N = c66 = rho*vs²
        J(i*6+5, i*3+0) = 2*rho*vs;  J(i*6+5, i*3+2) = vs*vs;
    }
    return J;
}

// Convert VTI gradient (M*6) and Hessian (M*6 × M*6) to isotropic (M*3 / M*3 × M*3).
// H_iso = J^T H_vti J + second-derivative correction (only diagonal blocks, same node).
inline void convert_vti_to_iso(
    const Eigen::VectorXd& g_vti,
    const Eigen::MatrixXd& H_vti,
    const std::vector<PhysicalParameters<double>>& nodes,
    Eigen::VectorXd& g_iso,
    Eigen::MatrixXd& H_iso)
{
    Eigen::MatrixXd J = build_iso_to_vti_jacobian(nodes);
    Eigen::VectorXd tmp_g = J.transpose() * g_vti;
    Eigen::MatrixXd tmp_H = J.transpose() * H_vti * J;
    g_iso = tmp_g;
    H_iso = tmp_H;

    int M = (int)nodes.size();
    for (int i = 0; i < M; ++i) {
        if (isLiquidLayer(nodes[i])) {
            continue;
        }
        double vs = nodes[i].vs(), vp = nodes[i].vp(), rho = nodes[i].rho;
        double gA = g_vti[i*6+1], gC = g_vti[i*6+2], gF = g_vti[i*6+3];
        double gL = g_vti[i*6+4], gN = g_vti[i*6+5];

        // d²(VTI_p)/d(vs)²: L→2ρ, N→2ρ, F→-4ρ, A=C→0
        H_iso(i*3+0, i*3+0) += gL*2*rho + gN*2*rho + gF*(-4*rho);
        // d²(VTI_p)/d(vp)²: A→2ρ, C→2ρ, F→2ρ
        H_iso(i*3+1, i*3+1) += (gA + gC + gF)*2*rho;
        // d²(VTI_p)/(dvs·drho): L→2vs, N→2vs, F→-4vs
        H_iso(i*3+0, i*3+2) += (gL + gN)*2*vs + gF*(-4*vs);
        H_iso(i*3+2, i*3+0) += (gL + gN)*2*vs + gF*(-4*vs);
        // d²(VTI_p)/(dvp·drho): A→2vp, C→2vp, F→2vp
        H_iso(i*3+1, i*3+2) += (gA + gC + gF)*2*vp;
        H_iso(i*3+2, i*3+1) += (gA + gC + gF)*2*vp;
    }
}

inline bool grid_hessian_node_is_fixed_fluid(
    const std::vector<PhysicalParameters<double>>& nodes,
    int node_idx)
{
    return isLiquidLayer(nodes[static_cast<size_t>(node_idx)]);
}

inline void validate_rayleigh_grid_fixed_fluid_model(const GridModel& model)
{
    if (model.params.empty()) return;
    if (isLiquidLayer(model.params.back())) {
        throw std::runtime_error(
            "Rayleigh grid weak-form Hessian does not support a fluid halfspace.");
    }

    bool seen_solid = false;
    for (size_t i = 0; i < model.params.size(); ++i) {
        bool is_fluid = isLiquidLayer(model.params[i]);
        if (!is_fluid) {
            seen_solid = true;
        } else if (seen_solid) {
            throw std::runtime_error(
                "Rayleigh grid weak-form Hessian supports only a top fluid stack over solid nodes.");
        }
    }

    for (size_t i = 0; i + 1 < model.params.size(); ++i) {
        bool crosses_fluid_solid =
            isLiquidLayer(model.params[i]) != isLiquidLayer(model.params[i + 1]);
        if (crosses_fluid_solid && !model.is_discontinuity[i]) {
            throw std::runtime_error(
                "Rayleigh grid weak-form Hessian with fluid requires a duplicate-depth "
                "discontinuity at the fluid-solid interface.");
        }
    }
}

inline void zero_fixed_fluid_vti_blocks(
    const std::vector<PhysicalParameters<double>>& nodes,
    Eigen::VectorXd& g_vti,
    Eigen::MatrixXd& H_vti)
{
    const int npl = 6;
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        if (!isLiquidLayer(nodes[i])) continue;
        for (int p = 0; p < npl; ++p) {
            int idx = i * npl + p;
            g_vti(idx) = 0.0;
            H_vti.row(idx).setZero();
            H_vti.col(idx).setZero();
        }
    }
}

struct GridHessianDebugResult {
    HessianResult final_result;
    std::vector<int> attempted_levels;
    std::vector<int> accepted_levels;
    std::vector<bool> accepted_mask;
    std::vector<double> c_roots_raw;
    std::vector<double> gradient_norms_raw;
    std::vector<double> hessian_norms_raw;
    std::vector<double> gradient_rel_step_raw;
    std::vector<double> hessian_rel_step_raw;
    std::vector<Eigen::VectorXd> sublayer_gradients_raw;
    std::vector<Eigen::VectorXd> gradients_raw;
    std::vector<Eigen::MatrixXd> hessians_raw;
    bool stopped_on_error_increase = false;
    bool stopped_on_tolerance = false;
    int rejected_level = -1;
};

inline HessianResult build_grid_hessian_result_from_vti(
    const GridModel& model,
    const Eigen::VectorXd& g_vti_final,
    const Eigen::MatrixXd& H_vti_final,
    SensitivityMode mode)
{
    HessianResult result;
    int M = static_cast<int>(model.params.size());

    if (mode == SensitivityMode::Isotropic) {
        Eigen::MatrixXd J = build_iso_to_vti_jacobian(model.params);
        Eigen::MatrixXd H_vti_J = H_vti_final * J;
        Eigen::MatrixXd H_iso = J.transpose() * H_vti_J;

        for (int i = 0; i < M; ++i) {
            if (isLiquidLayer(model.params[i])) {
                continue;
            }
            double vs = model.params[i].vs();
            double vp = model.params[i].vp();
            double rho = model.params[i].rho;
            double gA = g_vti_final[i*6+1], gC = g_vti_final[i*6+2], gF = g_vti_final[i*6+3];
            double gL = g_vti_final[i*6+4], gN = g_vti_final[i*6+5];
            H_iso(i*3+0, i*3+0) += gL*2*rho + gN*2*rho + gF*(-4*rho);
            H_iso(i*3+1, i*3+1) += (gA+gC+gF)*2*rho;
            H_iso(i*3+0, i*3+2) += (gL+gN)*2*vs + gF*(-4*vs);
            H_iso(i*3+2, i*3+0) += (gL+gN)*2*vs + gF*(-4*vs);
            H_iso(i*3+1, i*3+2) += (gA+gC+gF)*2*vp;
            H_iso(i*3+2, i*3+1) += (gA+gC+gF)*2*vp;
        }

        Eigen::VectorXd g_iso(M*3);
        g_iso.setZero();
        for (int i = 0; i < M; ++i) {
            if (isLiquidLayer(model.params[i])) {
                continue;
            }
            double vs = model.params[i].vs();
            double vp = model.params[i].vp();
            double rho = model.params[i].rho;
            double gRho = g_vti_final[i*6+0], gA = g_vti_final[i*6+1], gC = g_vti_final[i*6+2];
            double gF = g_vti_final[i*6+3], gL = g_vti_final[i*6+4], gN = g_vti_final[i*6+5];
            g_iso[i*3+0] = gF*(-4*rho*vs) + gL*2*rho*vs + gN*2*rho*vs;
            g_iso[i*3+1] = (gA+gC+gF)*2*rho*vp;
            g_iso[i*3+2] = gRho + (gA+gC)*vp*vp + gF*(vp*vp-2*vs*vs) + (gL+gN)*vs*vs;
        }

        result.hessian = ((H_iso + H_iso.transpose()) * 0.5).eval();
        result.gradient = g_iso;
        std::string names[] = {"Vs", "Vp", "Rho"};
        for (int i = 0; i < M; ++i)
            for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
    } else {
        result.hessian = H_vti_final;
        result.gradient = g_vti_final;
        std::string names[] = {"Rho","A","C","F","L","N"};
        for (int i = 0; i < M; ++i)
            for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Grid Mode WF Hessian — Rayleigh Wave
// ---------------------------------------------------------------------------
// Works internally in VTI parameterization (6 elastic moduli per node) so the
// Mesher's linear-in-moduli interpolation makes the chain rule exact.
// Romberg (Neville) extrapolation applied element-wise at the end.
// ---------------------------------------------------------------------------

// Seed the node-VTI parameter block [j0, j0+width) into derivative slots
// 0..width-1 of AD type T. Fixed-fluid nodes are left unseeded so their rows
// and columns come out zero, matching the `continue` of the dual loop (and
// zero_fixed_fluid_vti_blocks still runs afterwards either way).
template <typename T>
inline std::vector<PhysicalParameters<T>> seedGridNodeParamBlock(
    const std::vector<PhysicalParameters<double>>& node_params,
    int j0, int width, int num_params
) {
    const int npl = 6;  // node space is always VTI internally
    const int M = (int)node_params.size();
    std::vector<PhysicalParameters<T>> node_ad(M);
    for (int i = 0; i < M; ++i) node_ad[i] = to_ad_params<T>(node_params[i], width);

    for (int s = 0; s < width; ++s) {
        const int j = j0 + s;
        if (j >= num_params) break;
        const int node_idx = j / npl, param_idx = j % npl;
        if (grid_hessian_node_is_fixed_fluid(node_params, node_idx)) continue;
        PhysicalParameters<T>& target = node_ad[node_idx];
        switch (param_idx) {
            case 0: target.rho.derivatives()(s) = 1.0; break;
            case 1: target.c11.derivatives()(s) = 1.0; break;
            case 2: target.c33.derivatives()(s) = 1.0; break;
            case 3: target.c13.derivatives()(s) = 1.0; break;
            case 4: target.c44.derivatives()(s) = 1.0; break;
            case 5: target.c66.derivatives()(s) = 1.0; break;
        }
    }
    return node_ad;
}

// Seed a block of REDUCED-BASIS derivative directions [j0, j0+width) into AD
// type T, using the columns of an externally supplied Jacobian J (shape
// num_params_int x P, node-VTI rows in the same [node*6+modulus] layout as
// seedGridNodeParamBlock) instead of node-VTI unit vectors. Forward AD is
// linear in the seed, so seeding column p of J and running the same AD pass
// used for the node-unit-vector Hessian yields, after the mesh chain rule,
// H_a @ J directly -- the expensive part (528/16 ~= 33 blocks) collapses to
// ceil(P/16) blocks (P ~= 18 for the production velocity basis), and a single
// cheap J^T (H_a @ J) matmul afterwards gives the fully reduced P x P
// Hessian without ever materializing the 528 x 528 H_a
// (see docs/Weakform_AD_Hessian.md and the plan this implements).
// Fixed-fluid rows of J may be nonzero. They seed no material derivatives.
template <typename T>
inline std::vector<PhysicalParameters<T>> seedGridNodeParamBlockFromJacobian(
    const std::vector<PhysicalParameters<double>>& node_params,
    const Eigen::MatrixXd& J, int j0, int width, int num_params_reduced
) {
    const int npl = 6;  // node space is always VTI internally
    const int M = (int)node_params.size();
    std::vector<PhysicalParameters<T>> node_ad(M);
    for (int i = 0; i < M; ++i) node_ad[i] = to_ad_params<T>(node_params[i], width);

    for (int i = 0; i < M; ++i) {
        if (grid_hessian_node_is_fixed_fluid(node_params, i)) continue;
        PhysicalParameters<T>& target = node_ad[i];
        for (int s = 0; s < width; ++s) {
            const int col = j0 + s;
            if (col >= num_params_reduced) break;
            target.rho.derivatives()(s) = J(i * npl + 0, col);
            target.c11.derivatives()(s) = J(i * npl + 1, col);
            target.c33.derivatives()(s) = J(i * npl + 2, col);
            target.c13.derivatives()(s) = J(i * npl + 3, col);
            target.c44.derivatives()(s) = J(i * npl + 4, col);
            target.c66.derivatives()(s) = J(i * npl + 5, col);
        }
    }
    return node_ad;
}

// Copy a block of AD gradient columns into a plain (n_sub × n_cols) matrix so
// the mesh chain rule can be applied to all directions in one walk.
template <typename T>
inline Eigen::MatrixXd gridSublayerGradBlock(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& g_sub_ad,
    int n_cols
) {
    const int n_sub = (int)g_sub_ad.size();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n_sub, n_cols);
    for (int l = 0; l < n_sub; ++l) {
        const auto& d = g_sub_ad(l).derivatives();
        if ((int)d.size() < n_cols) continue;
        for (int s = 0; s < n_cols; ++s) G(l, s) = d(s);
    }
    return G;
}

inline double validated_grid_trace_root(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    double omega,
    WaveType wave_type,
    int level,
    const std::vector<double>& phase_velocity_roots,
    double residual_tolerance = kAdaptiveResidualLimit)
{
    if (level < 0 || static_cast<size_t>(level) >= phase_velocity_roots.size()) {
        throw std::invalid_argument(
            "Romberg root trace is shorter than the requested derivative level");
    }
    const double c_root = phase_velocity_roots[static_cast<size_t>(level)];
    if (!std::isfinite(c_root) || c_root <= 0.0) {
        throw std::invalid_argument(
            "Romberg root trace contains a non-finite or non-positive root");
    }
    const double residual = std::abs(calculateDispersionPhaseValueDouble(
        layers, thicknesses, halfspace, omega, c_root, wave_type));
    if (!std::isfinite(residual) || residual > residual_tolerance) {
        throw std::invalid_argument(
            "Romberg root trace is not residual-compatible with the derivative "
            "model, frequency, wave type, and refinement level (secular residual="
            + std::to_string(residual) + ")");
    }
    return c_root;
}

inline HessianResult computePhaseVelocityHessianWeakFormRayleigh_Grid(
    const GridModel& model,
    double freq,
    double c_initial_guess,
    SensitivityMode mode = SensitivityMode::Isotropic,
    int max_levels = 5,
    double tolerance = 1e-6,
    WFHessianBackend backend = WFHessianBackend::Dual,
    const std::vector<double>* phase_velocity_roots = nullptr
) {
    using adual = autodiff::dual;
    Mesher mesher(model);
    validate_rayleigh_grid_fixed_fluid_model(model);
    int M = (int)model.params.size();
    const int npl_int = 6;                                   // always VTI internally
    int num_params_int = M * npl_int;
    const double omega = 2.0 * M_PI * freq;

    std::vector<Eigen::MatrixXd> H_table;
    std::vector<Eigen::VectorXd> g_table;
    double c_guess = c_initial_guess;
    double prev_err = std::numeric_limits<double>::infinity();

    for (int k = 0; k <= max_levels; ++k) {
        auto [layers_k_dbl, thick_k_dbl] = mesher.get_refined_layers(k);
        PhysicalParameters<double> hs_dbl = mesher.get_halfspace();

        double c_root;
        if (phase_velocity_roots) {
            c_root = validated_grid_trace_root(
                layers_k_dbl, thick_k_dbl, hs_dbl, omega,
                WaveType::Rayleigh, k, *phase_velocity_roots);
        } else {
            // Legacy standalone behavior when no direct-dispersion trace is
            // supplied.
            auto f_double = [&](double cv) {
                return calculateDispersionFunction<double>(
                    layers_k_dbl, thick_k_dbl, hs_dbl, omega, cv,
                    WaveType::Rayleigh, NormalizationMode::VectorNorm)(I24);
            };
            c_root = c_guess;
            for (int it = 0; it < 15; ++it) {
                double f0 = f_double(c_root), eps = c_root * 1e-7;
                double df = (f_double(c_root + eps) - f0) / eps;
                if (std::abs(df) < 1e-30) break;
                double step = f0 / df;
                c_root -= step;
                if (std::abs(step) < 1e-11) break;
            }
        }
        c_guess = c_root;

        // Reference sublayer gradient in VTI (linear chain rule applies exactly)
        // Keep the weakform evaluation on the same Mesher-produced grid used by
        // the Romberg root solve. An extra internal adaptive split breaks that
        // level sequence and corrupts the extrapolated Hessian.
        auto g_sub_vti_dbl = computeVariationalGradientRayleigh_T<double>(
            layers_k_dbl, thick_k_dbl, hs_dbl, omega, c_root,
            SensitivityMode::VTI, 1, false, false);

        // Chain rule: g_node_vti = J^T g_sub (linear, exact for moduli interpolation)
        Eigen::VectorXd g0_node_vti = mesher.apply_sublayer_grad_to_node(k, g_sub_vti_dbl, npl_int);

        Eigen::MatrixXd H_vti = Eigen::MatrixXd::Zero(num_params_int, num_params_int);

        if (backend != WFHessianBackend::Dual) {
            // Vectorized forward AD: one pass per block of derivative directions
            // instead of one pass per node parameter. The mesh refinement and the
            // eigenfunction propagation — the expensive parts — are evaluated once
            // per block rather than once per column.
            const int width = (backend == WFHessianBackend::VectorDyn)
                                  ? num_params_int : WF_HESSIAN_CHUNK;

            auto run_block = [&](auto tag, int j0) {
                using T = typename decltype(tag)::type;
                std::vector<PhysicalParameters<T>> node_ad =
                    seedGridNodeParamBlock<T>(model.params, j0, width, num_params_int);

                auto [layers_k_ad, thick_k_ad] = mesher.get_refined_layers_ad<T>(k, node_ad);
                PhysicalParameters<T> hs_ad = mesher.get_halfspace_ad<T>(node_ad);

                Eigen::VectorXd cp_seed = Eigen::VectorXd::Zero(width);
                for (int s = 0; s < width && j0 + s < num_params_int; ++s)
                    cp_seed(s) = g0_node_vti(j0 + s);
                T ad_cp = make_ad_grad<T>(c_root, cp_seed);
                T ad_omega = make_ad_seeded<T>(omega, width);

                Eigen::Matrix<T, Eigen::Dynamic, 1> g_sub_ad =
                    computeVariationalGradientRayleigh_T<T>(
                        layers_k_ad, thick_k_ad, hs_ad, ad_omega, ad_cp,
                        SensitivityMode::VTI, 1, false, false);

                const int n_cols = std::min(width, num_params_int - j0);
                Eigen::MatrixXd G = gridSublayerGradBlock<T>(g_sub_ad, n_cols);
                H_vti.block(0, j0, num_params_int, n_cols) =
                    mesher.apply_sublayer_grad_to_node(k, G, npl_int);
            };

            if (backend == WFHessianBackend::VectorDyn) {
                run_block(TypeTag<ADVec>{}, 0);
            } else {
                for (int j0 = 0; j0 < num_params_int; j0 += width)
                    run_block(TypeTag<ADChunk<WF_HESSIAN_CHUNK>>{}, j0);
            }
        } else {
        // Build base dual node params (zero derivatives)
        std::vector<PhysicalParameters<adual>> node_dual_base(M);
        for (int i = 0; i < M; ++i) node_dual_base[i] = to_dual(model.params[i]);

        std::vector<PhysicalParameters<adual>> node_dual = node_dual_base;

        for (int j = 0; j < num_params_int; ++j) {
            int node_idx = j / npl_int, param_idx = j % npl_int;
            if (grid_hessian_node_is_fixed_fluid(model.params, node_idx)) {
                continue;
            }

            node_dual = node_dual_base;

            PhysicalParameters<adual>& target = node_dual[node_idx];

            // Seed single VTI param at node_idx
            switch (param_idx) {
                case 0: target.rho.grad = 1.0; break;
                case 1: target.c11.grad = 1.0; break;
                case 2: target.c33.grad = 1.0; break;
                case 3: target.c13.grad = 1.0; break;
                case 4: target.c44.grad = 1.0; break;
                case 5: target.c66.grad = 1.0; break;
            }

            auto [layers_k_dual, thick_k_dual] = mesher.get_refined_layers_ad<adual>(k, node_dual);
            PhysicalParameters<adual> hs_dual = mesher.get_halfspace_ad<adual>(node_dual);

            // IFT: d(c_root)/d(VTI_j) = g0_node_vti(j)
            adual d_cp    = make_dual(c_root, g0_node_vti(j));
            adual d_omega(omega);

            auto g_sub_dual = computeVariationalGradientRayleigh_T<adual>(
                layers_k_dual, thick_k_dual, hs_dual, d_omega, d_cp,
                SensitivityMode::VTI, 1, false, false);

            Eigen::VectorXd g_sub_grad(g_sub_dual.size());
            for (int l = 0; l < (int)g_sub_dual.size(); ++l) g_sub_grad(l) = g_sub_dual(l).grad;
            H_vti.col(j) = mesher.apply_sublayer_grad_to_node(k, g_sub_grad, npl_int);
        }
        }

        zero_fixed_fluid_vti_blocks(model.params, g0_node_vti, H_vti);
        H_vti = ((H_vti + H_vti.transpose()) * 0.5).eval();
        H_table.push_back(H_vti);
        g_table.push_back(g0_node_vti);

        if (k >= 1) {
            double hn = H_vti.norm();
            double err = (hn > 1e-30) ? (H_vti - H_table[k-1]).norm() / hn : 1.0;
            if (err > prev_err) { H_table.pop_back(); g_table.pop_back(); break; }
            if (err < tolerance) break;
            prev_err = err;
        }
    }

    // Neville extrapolation element-wise (VTI space)
    Eigen::MatrixXd H_vti_final = Eigen::MatrixXd::Zero(num_params_int, num_params_int);
    for (int i = 0; i < num_params_int; ++i)
        for (int j = 0; j < num_params_int; ++j) {
            std::vector<double> col;
            for (auto& Hk : H_table) col.push_back(Hk(i,j));
            H_vti_final(i,j) = neville_extrapolation(col).first;
        }

    Eigen::VectorXd g_vti_final(num_params_int);
    for (int i = 0; i < num_params_int; ++i) {
        std::vector<double> col;
        for (auto& g : g_table) col.push_back(g(i));
        g_vti_final(i) = neville_extrapolation(col).first;
    }

    zero_fixed_fluid_vti_blocks(model.params, g_vti_final, H_vti_final);
    return build_grid_hessian_result_from_vti(model, g_vti_final, H_vti_final, mode);
}

// Result of a reduced-basis grid Hessian evaluation: the raw node-VTI
// gradient (528-dim for the production model; the caller needs this
// unprojected to build the curvature correction of a nonlinear basis) and the
// Hessian already projected into the caller-supplied reduced basis (P x P).
struct GridHessianReducedResult {
    Eigen::VectorXd gradient_node;   // d c / d (node VTI modulus), length num_params_int
    Eigen::MatrixXd hessian_reduced; // J^T H_node J, P x P
};

struct GridVariationalGradientResult {
    Eigen::VectorXd gradient_node;
    std::vector<double> phase_velocity_roots_used;
    int accepted_level = -1;
    double relative_step = 1.0;
};

// First-order-only GRID variational loop.  This has its own convergence
// control: using an empty reduced basis in the Hessian loop leaves a 0x0
// Hessian whose norm cannot measure gradient convergence.
inline GridVariationalGradientResult computeKernelsVariationalGrid(
    const GridModel& model,
    double freq,
    double c_initial_guess,
    WaveType wave_type,
    int max_levels = 5,
    double tolerance = 1e-6,
    double initial_root_window_frac = 0.0,
    const std::vector<double>* phase_velocity_roots = nullptr)
{
    Mesher mesher(model);
    if (wave_type == WaveType::Rayleigh) {
        validate_rayleigh_grid_fixed_fluid_model(model);
    }
    const int npl = 6;
    const int num_params = static_cast<int>(model.params.size()) * npl;
    const double omega = 2.0 * M_PI * freq;

    std::vector<Eigen::VectorXd> gradient_table;
    std::vector<double> roots_used;
    std::vector<double> root_history;
    double c_guess = c_initial_guess;
    double previous_error = std::numeric_limits<double>::infinity();
    double accepted_error = 1.0;

    for (int k = 0; k <= max_levels; ++k) {
        auto [layers, thicknesses] = mesher.get_refined_layers(k);
        const PhysicalParameters<double> halfspace = mesher.get_halfspace();

        double c_root;
        if (phase_velocity_roots) {
            c_root = validated_grid_trace_root(
                layers, thicknesses, halfspace, omega, wave_type, k,
                *phase_velocity_roots);
        } else if (wave_type == WaveType::Love) {
            c_root = find_dispersion_root_adaptive(
                layers, thicknesses, halfspace, omega, c_guess, wave_type,
                tolerance, root_history, initial_root_window_frac);
        } else {
            auto f_double = [&](double cv) {
                return calculateDispersionFunction<double>(
                    layers, thicknesses, halfspace, omega, cv, wave_type,
                    NormalizationMode::VectorNorm)(I24);
            };
            c_root = c_guess;
            for (int iteration = 0; iteration < 15; ++iteration) {
                const double f0 = f_double(c_root);
                const double eps = c_root * 1.0e-7;
                const double df = (f_double(c_root + eps) - f0) / eps;
                if (std::abs(df) < 1.0e-30) break;
                const double step = f0 / df;
                c_root -= step;
                if (std::abs(step) < 1.0e-11) break;
            }
        }
        c_guess = c_root;
        root_history.push_back(c_root);

        Eigen::VectorXd sublayer_gradient =
            wave_type == WaveType::Love
                ? computeVariationalGradientLove_T<double>(
                      layers, thicknesses, halfspace, omega, c_root,
                      SensitivityMode::VTI, true, 1)
                : computeVariationalGradientRayleigh_T<double>(
                      layers, thicknesses, halfspace, omega, c_root,
                      SensitivityMode::VTI, 1, false, false);
        Eigen::VectorXd node_gradient =
            mesher.apply_sublayer_grad_to_node(k, sublayer_gradient, npl);
        if (node_gradient.size() != num_params || !node_gradient.allFinite()) {
            throw std::runtime_error(
                "GRID variational gradient produced a non-finite or invalid vector");
        }

        double error = 1.0;
        if (!gradient_table.empty()) {
            const double norm = node_gradient.norm();
            error = norm > 1.0e-30
                ? (node_gradient - gradient_table.back()).norm() / norm
                : 1.0;
            if (error > previous_error) break;
        }

        gradient_table.push_back(std::move(node_gradient));
        roots_used.push_back(c_root);
        accepted_error = error;
        if (k >= 1) {
            previous_error = error;
            if (error < tolerance) break;
        }
    }

    if (gradient_table.empty()) {
        throw std::runtime_error("GRID variational gradient accepted no levels");
    }
    GridVariationalGradientResult result;
    result.gradient_node = Eigen::VectorXd::Zero(num_params);
    for (int i = 0; i < num_params; ++i) {
        std::vector<double> column;
        column.reserve(gradient_table.size());
        for (const auto& gradient : gradient_table) column.push_back(gradient(i));
        result.gradient_node(i) = neville_extrapolation(column).first;
    }
    result.phase_velocity_roots_used = std::move(roots_used);
    result.accepted_level = static_cast<int>(gradient_table.size()) - 1;
    result.relative_step = accepted_error;
    return result;
}

// Reduced-basis counterpart of computePhaseVelocityHessianWeakFormRayleigh_Grid:
// instead of seeding every node-VTI unit direction (528 for the production
// model) to build the full node-space Hessian, seeds the columns of a
// caller-supplied Jacobian J (num_params_int x P, e.g. J = stiffness_jacobian
// . basis.full_log_operator() computed in Python) directly, producing
// J^T H_node J without ever materializing the 528 x 528 node-space Hessian.
// J is evaluated once by the caller at the current model parameters and reused
// across all Romberg levels here (it does not depend on the sublayer
// refinement, only on the node values, which the level loop does not change).
// mode/isotropic output conversion does not apply here -- the output space is
// whatever basis J represents.
inline GridHessianReducedResult computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced(
    const GridModel& model,
    double freq,
    double c_initial_guess,
    const Eigen::MatrixXd& J,
    int max_levels = 5,
    double tolerance = 1e-6,
    WFHessianBackend backend = WFHessianBackend::Dual,
    const std::vector<double>* phase_velocity_roots = nullptr
) {
    using adual = autodiff::dual;
    Mesher mesher(model);
    validate_rayleigh_grid_fixed_fluid_model(model);
    int M = (int)model.params.size();
    const int npl_int = 6;
    int num_params_int = M * npl_int;
    int P = (int)J.cols();
    if (J.rows() != num_params_int) {
        throw std::runtime_error(
            "computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced: J has "
            + std::to_string(J.rows()) + " rows, expected " + std::to_string(num_params_int));
    }
    const double omega = 2.0 * M_PI * freq;

    std::vector<Eigen::MatrixXd> H_table;  // each P x P
    std::vector<Eigen::VectorXd> g_table;  // each num_params_int
    double c_guess = c_initial_guess;
    double prev_err = std::numeric_limits<double>::infinity();

    for (int k = 0; k <= max_levels; ++k) {
        auto [layers_k_dbl, thick_k_dbl] = mesher.get_refined_layers(k);
        PhysicalParameters<double> hs_dbl = mesher.get_halfspace();

        double c_root;
        if (phase_velocity_roots) {
            c_root = validated_grid_trace_root(
                layers_k_dbl, thick_k_dbl, hs_dbl, omega,
                WaveType::Rayleigh, k, *phase_velocity_roots);
        } else {
            auto f_double = [&](double cv) {
                return calculateDispersionFunction<double>(
                    layers_k_dbl, thick_k_dbl, hs_dbl, omega, cv,
                    WaveType::Rayleigh, NormalizationMode::VectorNorm)(I24);
            };
            c_root = c_guess;
            for (int it = 0; it < 15; ++it) {
                double f0 = f_double(c_root), eps = c_root * 1e-7;
                double df = (f_double(c_root + eps) - f0) / eps;
                if (std::abs(df) < 1e-30) break;
                double step = f0 / df;
                c_root -= step;
                if (std::abs(step) < 1e-11) break;
            }
        }
        c_guess = c_root;

        auto g_sub_vti_dbl = computeVariationalGradientRayleigh_T<double>(
            layers_k_dbl, thick_k_dbl, hs_dbl, omega, c_root,
            SensitivityMode::VTI, 1, false, false);
        Eigen::VectorXd g0_node_vti = mesher.apply_sublayer_grad_to_node(k, g_sub_vti_dbl, npl_int);

        for (int node = 0; node < M; ++node) {
            if (grid_hessian_node_is_fixed_fluid(model.params, node))
                g0_node_vti.segment(node * npl_int, npl_int).setZero();
        }

        // HJ = H_node @ J, built block-by-block via J-column seeding.
        Eigen::MatrixXd HJ = Eigen::MatrixXd::Zero(num_params_int, P);

        const int width = (backend == WFHessianBackend::VectorDyn) ? P : WF_HESSIAN_CHUNK;

        auto run_block_vec = [&](auto tag, int j0) {
            using T = typename decltype(tag)::type;
            std::vector<PhysicalParameters<T>> node_ad =
                seedGridNodeParamBlockFromJacobian<T>(model.params, J, j0, width, P);

            auto [layers_k_ad, thick_k_ad] = mesher.get_refined_layers_ad<T>(k, node_ad);
            PhysicalParameters<T> hs_ad = mesher.get_halfspace_ad<T>(node_ad);

            // IFT c-seed: d(c_root)/d(basis_col) = g0_node_vti . J[:, col]
            // (chain rule through the same node-VTI gradient used elsewhere).
            Eigen::VectorXd cp_seed = Eigen::VectorXd::Zero(width);
            for (int s = 0; s < width && j0 + s < P; ++s)
                cp_seed(s) = g0_node_vti.dot(J.col(j0 + s));
            T ad_cp = make_ad_grad<T>(c_root, cp_seed);
            T ad_omega = make_ad_seeded<T>(omega, width);

            Eigen::Matrix<T, Eigen::Dynamic, 1> g_sub_ad =
                computeVariationalGradientRayleigh_T<T>(
                    layers_k_ad, thick_k_ad, hs_ad, ad_omega, ad_cp,
                    SensitivityMode::VTI, 1, false, false);

            const int n_cols = std::min(width, P - j0);
            Eigen::MatrixXd G = gridSublayerGradBlock<T>(g_sub_ad, n_cols);
            HJ.block(0, j0, num_params_int, n_cols) =
                mesher.apply_sublayer_grad_to_node(k, G, npl_int);
        };
        auto run_block_dual = [&](int col) {
            std::vector<PhysicalParameters<adual>> node_dual(M);
            for (int i = 0; i < M; ++i) node_dual[i] = to_dual(model.params[i]);
            for (int i = 0; i < M; ++i) {
                if (grid_hessian_node_is_fixed_fluid(model.params, i)) continue;
                PhysicalParameters<adual>& target = node_dual[i];
                target.rho.grad = J(i * npl_int + 0, col);
                target.c11.grad = J(i * npl_int + 1, col);
                target.c33.grad = J(i * npl_int + 2, col);
                target.c13.grad = J(i * npl_int + 3, col);
                target.c44.grad = J(i * npl_int + 4, col);
                target.c66.grad = J(i * npl_int + 5, col);
            }
            auto [layers_k_dual, thick_k_dual] = mesher.get_refined_layers_ad<adual>(k, node_dual);
            PhysicalParameters<adual> hs_dual = mesher.get_halfspace_ad<adual>(node_dual);

            adual d_cp = make_dual(c_root, g0_node_vti.dot(J.col(col)));
            adual d_omega(omega);

            auto g_sub_dual = computeVariationalGradientRayleigh_T<adual>(
                layers_k_dual, thick_k_dual, hs_dual, d_omega, d_cp,
                SensitivityMode::VTI, 1, false, false);

            Eigen::VectorXd g_sub_grad(g_sub_dual.size());
            for (int l = 0; l < (int)g_sub_dual.size(); ++l) g_sub_grad(l) = g_sub_dual(l).grad;
            HJ.col(col) = mesher.apply_sublayer_grad_to_node(k, g_sub_grad, npl_int);
        };

        if (backend == WFHessianBackend::Dual) {
            for (int col = 0; col < P; ++col) run_block_dual(col);
        } else if (backend == WFHessianBackend::VectorDyn) {
            run_block_vec(TypeTag<ADVec>{}, 0);
        } else {
            for (int j0 = 0; j0 < P; j0 += width)
                run_block_vec(TypeTag<ADChunk<WF_HESSIAN_CHUNK>>{}, j0);
        }

        for (int node = 0; node < M; ++node) {
            if (grid_hessian_node_is_fixed_fluid(model.params, node))
                HJ.middleRows(node * npl_int, npl_int).setZero();
        }
        Eigen::MatrixXd H_reduced = J.transpose() * HJ;
        H_reduced = ((H_reduced + H_reduced.transpose()) * 0.5).eval();
        H_table.push_back(H_reduced);
        g_table.push_back(g0_node_vti);

        if (k >= 1) {
            double hn = H_reduced.norm();
            double err = (hn > 1e-30) ? (H_reduced - H_table[k-1]).norm() / hn : 1.0;
            if (err > prev_err) { H_table.pop_back(); g_table.pop_back(); break; }
            if (err < tolerance) break;
            prev_err = err;
        }
    }

    GridHessianReducedResult result;
    result.hessian_reduced = Eigen::MatrixXd::Zero(P, P);
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < P; ++j) {
            std::vector<double> col;
            for (auto& Hk : H_table) col.push_back(Hk(i, j));
            result.hessian_reduced(i, j) = neville_extrapolation(col).first;
        }

    result.gradient_node = Eigen::VectorXd::Zero(num_params_int);
    for (int i = 0; i < num_params_int; ++i) {
        std::vector<double> col;
        for (auto& g : g_table) col.push_back(g(i));
        result.gradient_node(i) = neville_extrapolation(col).first;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Grid Mode WF Hessian — Love Wave
// ---------------------------------------------------------------------------

inline GridHessianDebugResult computePhaseVelocityHessianWeakFormLove_Grid_Debug(
    const GridModel& model,
    double freq,
    double c_initial_guess,
    SensitivityMode mode = SensitivityMode::Isotropic,
    int max_levels = 5,
    double tolerance = 1e-6,
    bool use_analytical = true,
    int num_sub_layers = 1,
    double initial_root_window_frac = 0.0,
    WFHessianBackend backend = WFHessianBackend::Dual,
    const std::vector<double>* phase_velocity_roots = nullptr
);

inline HessianResult computePhaseVelocityHessianWeakFormLove_Grid(
    const GridModel& model,
    double freq,
    double c_initial_guess,
    SensitivityMode mode = SensitivityMode::Isotropic,
    int max_levels = 5,
    double tolerance = 1e-6,
    double initial_root_window_frac = 0.0,
    WFHessianBackend backend = WFHessianBackend::Dual,
    const std::vector<double>* phase_velocity_roots = nullptr
) {
    return computePhaseVelocityHessianWeakFormLove_Grid_Debug(
        model, freq, c_initial_guess, mode, max_levels, tolerance, true, 1,
        initial_root_window_frac, backend, phase_velocity_roots).final_result;
}

inline GridHessianDebugResult computePhaseVelocityHessianWeakFormLove_Grid_Debug(
    const GridModel& model,
    double freq,
    double c_initial_guess,
    SensitivityMode mode,
    int max_levels,
    double tolerance,
    bool use_analytical,
    int num_sub_layers,
    double initial_root_window_frac,
    WFHessianBackend backend,
    const std::vector<double>* phase_velocity_roots
) {
    using adual = autodiff::dual;
    Mesher mesher(model);
    int M = static_cast<int>(model.params.size());
    const int npl_int = 6;
    int num_params_int = M * npl_int;
    const double omega = 2.0 * M_PI * freq;

    std::vector<Eigen::MatrixXd> H_table;
    std::vector<Eigen::VectorXd> g_table;
    double c_guess = c_initial_guess;
    double prev_err = std::numeric_limits<double>::infinity();
    std::vector<double> root_history;

    GridHessianDebugResult debug;

    for (int k = 0; k <= max_levels; ++k) {
        auto [layers_k_dbl, thick_k_dbl] = mesher.get_refined_layers(k);
        PhysicalParameters<double> hs_dbl = mesher.get_halfspace();

        double c_root;
        if (phase_velocity_roots) {
            c_root = validated_grid_trace_root(
                layers_k_dbl, thick_k_dbl, hs_dbl, omega,
                WaveType::Love, k, *phase_velocity_roots);
        } else {
            c_root = find_dispersion_root_adaptive(
                layers_k_dbl, thick_k_dbl, hs_dbl, omega, c_guess,
                WaveType::Love, tolerance, root_history,
                initial_root_window_frac);
        }
        c_guess = c_root;
        root_history.push_back(c_root);

        auto g_sub_vti_dbl = computeVariationalGradientLove_T<double>(
            layers_k_dbl, thick_k_dbl, hs_dbl, omega, c_root,
            SensitivityMode::VTI, use_analytical, num_sub_layers);

        Eigen::VectorXd g0_node_vti = mesher.apply_sublayer_grad_to_node(k, g_sub_vti_dbl, npl_int);
        if (!g_sub_vti_dbl.allFinite() || !g0_node_vti.allFinite()) {
            throw std::runtime_error(
                "Love grid Hessian produced a non-finite gradient at a Romberg level");
        }

        Eigen::MatrixXd H_vti = Eigen::MatrixXd::Zero(num_params_int, num_params_int);

        if (backend != WFHessianBackend::Dual) {
            // Vectorized forward AD — one pass per block of derivative directions.
            const int width = (backend == WFHessianBackend::VectorDyn)
                                  ? num_params_int : WF_HESSIAN_CHUNK;

            auto run_block = [&](auto tag, int j0) {
                using T = typename decltype(tag)::type;
                std::vector<PhysicalParameters<T>> node_ad =
                    seedGridNodeParamBlock<T>(model.params, j0, width, num_params_int);

                auto [layers_k_ad, thick_k_ad] = mesher.get_refined_layers_ad<T>(k, node_ad);
                PhysicalParameters<T> hs_ad = mesher.get_halfspace_ad<T>(node_ad);

                Eigen::VectorXd cp_seed = Eigen::VectorXd::Zero(width);
                for (int s = 0; s < width && j0 + s < num_params_int; ++s)
                    cp_seed(s) = g0_node_vti(j0 + s);
                T ad_cp = make_ad_grad<T>(c_root, cp_seed);
                T ad_omega = make_ad_seeded<T>(omega, width);

                Eigen::Matrix<T, Eigen::Dynamic, 1> g_sub_ad = computeVariationalGradientLove_T<T>(
                    layers_k_ad, thick_k_ad, hs_ad, ad_omega, ad_cp,
                    SensitivityMode::VTI, use_analytical, num_sub_layers);

                const int n_cols = std::min(width, num_params_int - j0);
                Eigen::MatrixXd G = gridSublayerGradBlock<T>(g_sub_ad, n_cols);
                if (!G.allFinite()) {
                    throw std::runtime_error(
                        "Love grid Hessian produced a non-finite AD gradient column");
                }
                H_vti.block(0, j0, num_params_int, n_cols) =
                    mesher.apply_sublayer_grad_to_node(k, G, npl_int);
            };

            if (backend == WFHessianBackend::VectorDyn) {
                run_block(TypeTag<ADVec>{}, 0);
            } else {
                for (int j0 = 0; j0 < num_params_int; j0 += width)
                    run_block(TypeTag<ADChunk<WF_HESSIAN_CHUNK>>{}, j0);
            }
        } else {
        std::vector<PhysicalParameters<adual>> node_dual_base(M);
        for (int i = 0; i < M; ++i) node_dual_base[i] = to_dual(model.params[i]);

        std::vector<PhysicalParameters<adual>> node_dual = node_dual_base;

        for (int j = 0; j < num_params_int; ++j) {
            node_dual = node_dual_base;

            int node_idx = j / npl_int, param_idx = j % npl_int;
            PhysicalParameters<adual>& target = node_dual[node_idx];

            switch (param_idx) {
                case 0: target.rho.grad = 1.0; break;
                case 1: target.c11.grad = 1.0; break;
                case 2: target.c33.grad = 1.0; break;
                case 3: target.c13.grad = 1.0; break;
                case 4: target.c44.grad = 1.0; break;
                case 5: target.c66.grad = 1.0; break;
            }

            auto [layers_k_dual, thick_k_dual] = mesher.get_refined_layers_ad<adual>(k, node_dual);
            PhysicalParameters<adual> hs_dual = mesher.get_halfspace_ad<adual>(node_dual);

            adual d_cp = make_dual(c_root, g0_node_vti(j));
            adual d_omega(omega);

            auto g_sub_dual = computeVariationalGradientLove_T<adual>(
                layers_k_dual, thick_k_dual, hs_dual, d_omega, d_cp,
                SensitivityMode::VTI, use_analytical, num_sub_layers);

            Eigen::VectorXd g_sub_grad(g_sub_dual.size());
            for (int l = 0; l < static_cast<int>(g_sub_dual.size()); ++l) g_sub_grad(l) = g_sub_dual(l).grad;
            if (!g_sub_grad.allFinite()) {
                throw std::runtime_error(
                    "Love grid Hessian produced a non-finite AD gradient column");
            }
            H_vti.col(j) = mesher.apply_sublayer_grad_to_node(k, g_sub_grad, npl_int);
        }
        }

        H_vti = ((H_vti + H_vti.transpose()) * 0.5).eval();
        if (!H_vti.allFinite()) {
            throw std::runtime_error(
                "Love grid Hessian produced a non-finite Hessian at a Romberg level");
        }

        HessianResult raw_result = build_grid_hessian_result_from_vti(model, g0_node_vti, H_vti, mode);
        debug.attempted_levels.push_back(k);
        debug.c_roots_raw.push_back(c_root);
        debug.gradient_norms_raw.push_back(raw_result.gradient.norm());
        debug.hessian_norms_raw.push_back(raw_result.hessian.norm());
        debug.sublayer_gradients_raw.push_back(g_sub_vti_dbl);
        debug.gradients_raw.push_back(raw_result.gradient);
        debug.hessians_raw.push_back(raw_result.hessian);

        double g_err = 1.0;
        double h_err = 1.0;
        if (!g_table.empty()) {
            double gn = g0_node_vti.norm();
            g_err = (gn > 1e-30) ? (g0_node_vti - g_table.back()).norm() / gn : 1.0;
        }
        if (!H_table.empty()) {
            double hn = H_vti.norm();
            h_err = (hn > 1e-30) ? (H_vti - H_table.back()).norm() / hn : 1.0;
        }
        debug.gradient_rel_step_raw.push_back(g_err);
        debug.hessian_rel_step_raw.push_back(h_err);

        bool accepted = true;
        if (!H_table.empty()) {
            if (h_err > prev_err) {
                accepted = false;
                debug.stopped_on_error_increase = true;
                debug.rejected_level = k;
            }
        }

        debug.accepted_mask.push_back(accepted);
        if (!accepted) break;

        H_table.push_back(H_vti);
        g_table.push_back(g0_node_vti);
        debug.accepted_levels.push_back(k);

        if (k >= 1) {
            prev_err = h_err;
            if (h_err < tolerance) {
                debug.stopped_on_tolerance = true;
                break;
            }
        }
    }

    Eigen::MatrixXd H_vti_final = Eigen::MatrixXd::Zero(num_params_int, num_params_int);
    for (int i = 0; i < num_params_int; ++i)
        for (int j = 0; j < num_params_int; ++j) {
            std::vector<double> col;
            for (auto& Hk : H_table) col.push_back(Hk(i, j));
            H_vti_final(i, j) = neville_extrapolation(col).first;
        }

    Eigen::VectorXd g_vti_final(num_params_int);
    for (int i = 0; i < num_params_int; ++i) {
        std::vector<double> col;
        for (auto& g : g_table) col.push_back(g(i));
        g_vti_final(i) = neville_extrapolation(col).first;
    }

    debug.final_result = build_grid_hessian_result_from_vti(model, g_vti_final, H_vti_final, mode);
    return debug;
}

// Reduced-basis counterpart of computePhaseVelocityHessianWeakFormLove_Grid.
// See computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced for the design
// (J-column seeding replaces node-VTI unit-vector seeding; J is evaluated once
// by the caller and reused across Romberg levels).
inline GridHessianReducedResult computePhaseVelocityHessianWeakFormLove_Grid_Reduced(
    const GridModel& model,
    double freq,
    double c_initial_guess,
    const Eigen::MatrixXd& J,
    int max_levels = 5,
    double tolerance = 1e-6,
    bool use_analytical = true,
    int num_sub_layers = 1,
    double initial_root_window_frac = 0.0,
    WFHessianBackend backend = WFHessianBackend::Dual,
    const std::vector<double>* phase_velocity_roots = nullptr
) {
    using adual = autodiff::dual;
    Mesher mesher(model);
    int M = static_cast<int>(model.params.size());
    const int npl_int = 6;
    int num_params_int = M * npl_int;
    int P = (int)J.cols();
    if (J.rows() != num_params_int) {
        throw std::runtime_error(
            "computePhaseVelocityHessianWeakFormLove_Grid_Reduced: J has "
            + std::to_string(J.rows()) + " rows, expected " + std::to_string(num_params_int));
    }
    const double omega = 2.0 * M_PI * freq;

    std::vector<Eigen::MatrixXd> H_table;
    std::vector<Eigen::VectorXd> g_table;
    double c_guess = c_initial_guess;
    double prev_err = std::numeric_limits<double>::infinity();
    std::vector<double> root_history;

    for (int k = 0; k <= max_levels; ++k) {
        auto [layers_k_dbl, thick_k_dbl] = mesher.get_refined_layers(k);
        PhysicalParameters<double> hs_dbl = mesher.get_halfspace();

        double c_root;
        if (phase_velocity_roots) {
            c_root = validated_grid_trace_root(
                layers_k_dbl, thick_k_dbl, hs_dbl, omega,
                WaveType::Love, k, *phase_velocity_roots);
        } else {
            c_root = find_dispersion_root_adaptive(
                layers_k_dbl, thick_k_dbl, hs_dbl, omega, c_guess,
                WaveType::Love, tolerance, root_history,
                initial_root_window_frac);
        }
        c_guess = c_root;
        root_history.push_back(c_root);

        auto g_sub_vti_dbl = computeVariationalGradientLove_T<double>(
            layers_k_dbl, thick_k_dbl, hs_dbl, omega, c_root,
            SensitivityMode::VTI, use_analytical, num_sub_layers);
        Eigen::VectorXd g0_node_vti = mesher.apply_sublayer_grad_to_node(k, g_sub_vti_dbl, npl_int);
        if (!g_sub_vti_dbl.allFinite() || !g0_node_vti.allFinite()) {
            throw std::runtime_error(
                "Love grid reduced Hessian produced a non-finite gradient at a Romberg level");
        }

        Eigen::MatrixXd HJ = Eigen::MatrixXd::Zero(num_params_int, P);
        const int width = (backend == WFHessianBackend::VectorDyn) ? P : WF_HESSIAN_CHUNK;

        auto run_block_vec = [&](auto tag, int j0) {
            using T = typename decltype(tag)::type;
            std::vector<PhysicalParameters<T>> node_ad =
                seedGridNodeParamBlockFromJacobian<T>(model.params, J, j0, width, P);

            auto [layers_k_ad, thick_k_ad] = mesher.get_refined_layers_ad<T>(k, node_ad);
            PhysicalParameters<T> hs_ad = mesher.get_halfspace_ad<T>(node_ad);

            Eigen::VectorXd cp_seed = Eigen::VectorXd::Zero(width);
            for (int s = 0; s < width && j0 + s < P; ++s)
                cp_seed(s) = g0_node_vti.dot(J.col(j0 + s));
            T ad_cp = make_ad_grad<T>(c_root, cp_seed);
            T ad_omega = make_ad_seeded<T>(omega, width);

            Eigen::Matrix<T, Eigen::Dynamic, 1> g_sub_ad = computeVariationalGradientLove_T<T>(
                layers_k_ad, thick_k_ad, hs_ad, ad_omega, ad_cp,
                SensitivityMode::VTI, use_analytical, num_sub_layers);

            const int n_cols = std::min(width, P - j0);
            Eigen::MatrixXd G = gridSublayerGradBlock<T>(g_sub_ad, n_cols);
            if (!G.allFinite()) {
                throw std::runtime_error(
                    "Love grid reduced Hessian produced a non-finite AD gradient column");
            }
            HJ.block(0, j0, num_params_int, n_cols) =
                mesher.apply_sublayer_grad_to_node(k, G, npl_int);
        };
        auto run_block_dual = [&](int col) {
            std::vector<PhysicalParameters<adual>> node_dual(M);
            for (int i = 0; i < M; ++i) node_dual[i] = to_dual(model.params[i]);
            for (int i = 0; i < M; ++i) {
                PhysicalParameters<adual>& target = node_dual[i];
                target.rho.grad = J(i * npl_int + 0, col);
                target.c11.grad = J(i * npl_int + 1, col);
                target.c33.grad = J(i * npl_int + 2, col);
                target.c13.grad = J(i * npl_int + 3, col);
                target.c44.grad = J(i * npl_int + 4, col);
                target.c66.grad = J(i * npl_int + 5, col);
            }
            auto [layers_k_dual, thick_k_dual] = mesher.get_refined_layers_ad<adual>(k, node_dual);
            PhysicalParameters<adual> hs_dual = mesher.get_halfspace_ad<adual>(node_dual);

            adual d_cp = make_dual(c_root, g0_node_vti.dot(J.col(col)));
            adual d_omega(omega);

            auto g_sub_dual = computeVariationalGradientLove_T<adual>(
                layers_k_dual, thick_k_dual, hs_dual, d_omega, d_cp,
                SensitivityMode::VTI, use_analytical, num_sub_layers);

            Eigen::VectorXd g_sub_grad(g_sub_dual.size());
            for (int l = 0; l < static_cast<int>(g_sub_dual.size()); ++l) g_sub_grad(l) = g_sub_dual(l).grad;
            if (!g_sub_grad.allFinite()) {
                throw std::runtime_error(
                    "Love grid reduced Hessian produced a non-finite AD gradient column");
            }
            HJ.col(col) = mesher.apply_sublayer_grad_to_node(k, g_sub_grad, npl_int);
        };

        if (backend == WFHessianBackend::Dual) {
            for (int col = 0; col < P; ++col) run_block_dual(col);
        } else if (backend == WFHessianBackend::VectorDyn) {
            run_block_vec(TypeTag<ADVec>{}, 0);
        } else {
            for (int j0 = 0; j0 < P; j0 += width)
                run_block_vec(TypeTag<ADChunk<WF_HESSIAN_CHUNK>>{}, j0);
        }

        Eigen::MatrixXd H_reduced = J.transpose() * HJ;
        H_reduced = ((H_reduced + H_reduced.transpose()) * 0.5).eval();
        if (!H_reduced.allFinite()) {
            throw std::runtime_error(
                "Love grid reduced Hessian produced a non-finite Hessian at a Romberg level");
        }

        double h_err = 1.0;
        if (!H_table.empty()) {
            double hn = H_reduced.norm();
            h_err = (hn > 1e-30) ? (H_reduced - H_table.back()).norm() / hn : 1.0;
        }
        bool accepted = true;
        if (!H_table.empty() && h_err > prev_err) accepted = false;
        if (!accepted) break;

        H_table.push_back(H_reduced);
        g_table.push_back(g0_node_vti);

        if (k >= 1) {
            prev_err = h_err;
            if (h_err < tolerance) break;
        }
    }

    GridHessianReducedResult result;
    result.hessian_reduced = Eigen::MatrixXd::Zero(P, P);
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < P; ++j) {
            std::vector<double> col;
            for (auto& Hk : H_table) col.push_back(Hk(i, j));
            result.hessian_reduced(i, j) = neville_extrapolation(col).first;
        }

    result.gradient_node = Eigen::VectorXd::Zero(num_params_int);
    for (int i = 0; i < num_params_int; ++i) {
        std::vector<double> col;
        for (auto& g : g_table) col.push_back(g(i));
        result.gradient_node(i) = neville_extrapolation(col).first;
    }
    return result;
}
