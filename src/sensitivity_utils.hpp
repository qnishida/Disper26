#pragma once
// Sensitivity and H/V Calculation Utilities for disper26
// Included by disper26.cpp or test files
#include <memory>

// --- Definition of Sensitivity Kernel Result ---
struct SensitivityKernel {
    std::string parameterName;
    int layerIndex; // -1 for halfspace
    double d_PhaseVelocity;
    double d_GroupVelocity;
    double d_HVRatio;
};

// --- Signed Rayleigh ellipticity ---
// z is upward, exp(i(kx-omega*t)), u_z=y1 and u_x=i*y3 (phase omitted).
// epsilon=y3/y1: negative is retrograde, positive is prograde for k>0.
// At zero shear traction the solid state is proportional to
// [Y14, Y24, -Y12, 0]. This applies at a solid free surface and on the
// solid side of a fluid/solid interface, where Y24 need not vanish.
// Unlike -Y23/Y12, this ratio remains regular when y3 crosses zero.
template <typename T>
T calculateHVRatio(const Eigen::Matrix<T, 5, 1>& Y_surface) {
    if (get_magnitude(Y_surface(I14)) < 1e-20) {
        // Preserve AD derivative dimensions and mark derivatives undefined too.
        T undefined = Y_surface(I12) + Y_surface(I14);
        undefined *= std::numeric_limits<double>::quiet_NaN();
        return undefined;
    }
    return -Y_surface(I12) / Y_surface(I14);
}

// Enum for Sensitivity Parameterization
enum class SensitivityMode {
    Isotropic, // Vs, Vp, Rho (with constraints on c44/c66/c11/c33)
    VTI        // A(c11), C(c33), F(c13), L(c44), N(c66), Rho (Independent)
};

// --- Isotropy Checker ---
template <typename T>
bool isModelIsotropic(const std::vector<PhysicalParameters<T>>& layers, const PhysicalParameters<T>& halfSpace) {
    auto check = [](const PhysicalParameters<T>& p) {
        double c11 = get_magnitude(p.c11);
        double c33 = get_magnitude(p.c33);
        double c44 = get_magnitude(p.c44);
        double c66 = get_magnitude(p.c66);
        double c13 = get_magnitude(p.c13);
        
        bool ok = true;
        if (std::abs(c11 - c33) > 1e-5 * c11) ok = false;
        if (std::abs(c44 - c66) > 1e-5 * c44) ok = false;
        if (std::abs(c13 - (c11 - 2.0*c44)) > 1e-5 * c11) ok = false;
        return ok;
    };
    
    for(const auto& l : layers) if(!check(l)) return false;
    if (!check(halfSpace)) return false;
    return true;
}

// --- Finite Difference Sensitivity Kernel Calculator (FD) ---
// Computes sensitivity of c, U, and H/V using central differences.
// This is the "Hybrid" mode from previous versions, now explicitly named FD.
template <typename T> // T is usually double
std::vector<SensitivityKernel> computeSensitivityKernelsFD(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    double freq,
    double root_c,
    WaveType type = WaveType::Rayleigh,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool computeThickness = false
) {
    // Validation
    if (mode == SensitivityMode::Isotropic) {
        if (!isModelIsotropic(layers, halfSpace)) {
            throw std::runtime_error("SensitivityMode::Isotropic requires a fully isotropic model. Use SensitivityMode::VTI for anisotropic models.");
        }
    }
    
    std::vector<SensitivityKernel> kernels;
    
    // Helper to calc state (c, U, HV) at a given model
    // Returns tuple: <c, U, HV>
    // Uses AD for U and HV calculation (Hybrid AD)
    auto calc_full_state = [&](
        const std::vector<PhysicalParameters<T>>& l_use,
        const std::vector<T>& t_use,
        const PhysicalParameters<T>& h_use,
        double f_val,
        double c_val_guess
    ) -> std::tuple<double, double, double> {
        
        // 1. Refine Root c
        // Use local root finder to get exact c for this perturbed model
        // (Assuming c_val_guess is close)
        // Helper root finder
        auto find_c = [&](double guess) {
             // Robust Local Scan + Brent's method
             double search_radius = 0.5; // Increased from 0.2
             double dz = 0.01; 
             double range_min = std::max(0.01, guess - search_radius);
             double range_max = guess + search_radius;
             
             // Define function with simple double returns for scanning
             auto func_val = [&](double c_v) {
                 using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
                 ADS w; w.value() = 2.0*M_PI*f_val; w.derivatives() = Eigen::VectorXd::Zero(0); // Constant
                 ADS c; c.value() = c_v; c.derivatives() = Eigen::VectorXd::Zero(0); // Constant
                 
                 // Manual conversion to ADS
                 auto to_ads = [](const PhysicalParameters<T>& p) {
                    PhysicalParameters<ADS> pa;
                    pa.rho = ADS(p.rho);
                    pa.c11 = ADS(p.c11); pa.c33 = ADS(p.c33);
                    pa.c44 = ADS(p.c44); pa.c66 = ADS(p.c66);
                    pa.c13 = ADS(p.c13);
                    return pa;
                };
                std::vector<PhysicalParameters<ADS>> l_ads;
                std::vector<ADS> t_ads;
                for(auto& p : l_use) l_ads.push_back(to_ads(p));
                for(auto& t : t_use) { ADS ta(t); t_ads.push_back(ta); }
                PhysicalParameters<ADS> h_ads = to_ads(h_use);
                auto res = calculateDispersionFunction(l_ads, t_ads, h_ads, w, c, type);
                return res(I24).value();
             };

             // Scan for sign change
             double c1 = range_min; double f1 = func_val(c1);
             for (double c2 = c1 + dz; c2 <= range_max; c2 += dz) {
                 double f2 = func_val(c2);
                 if (f1 * f2 <= 0.0) return brents_method(func_val, c1, c2, 1e-10, 100);
                 c1 = c2; f1 = f2;
             }
             return -1.0;
        };
        
        double c_exact = find_c(c_val_guess);
        if (c_exact < 0) {
            // Retry with wider range? Or fail?
            return std::make_tuple(NAN, NAN, NAN);
        }
        
        // 2. Calc U and HV using AD at c_exact
        using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
        ADS c_ad; c_ad.value() = c_exact; c_ad.derivatives() = Eigen::VectorXd::Unit(2, 0); // wrt c
        ADS w_ad; w_ad.value() = 2.0*M_PI*f_val; w_ad.derivatives() = Eigen::VectorXd::Unit(2, 1); // wrt w
        
        auto to_ads_der = [](const PhysicalParameters<T>& p) {
            PhysicalParameters<ADS> pa;
            pa.rho = ADS(p.rho); pa.rho.derivatives().setZero(2);
            pa.c11 = ADS(p.c11); pa.c11.derivatives().setZero(2);
            pa.c33 = ADS(p.c33); pa.c33.derivatives().setZero(2);
            pa.c44 = ADS(p.c44); pa.c44.derivatives().setZero(2);
            pa.c66 = ADS(p.c66); pa.c66.derivatives().setZero(2);
            pa.c13 = ADS(p.c13); pa.c13.derivatives().setZero(2);
            pa.gravity = ADS(p.gravity); pa.gravity.derivatives().setZero(2);
            pa.gravity_internal_scale = ADS(p.gravity_internal_scale); pa.gravity_internal_scale.derivatives().setZero(2);
            pa.gravity_surface_scale = ADS(p.gravity_surface_scale); pa.gravity_surface_scale.derivatives().setZero(2);
            return pa;
        };
        
        std::vector<PhysicalParameters<ADS>> l_ads;
        std::vector<ADS> t_ads;
        for(auto& p : l_use) l_ads.push_back(to_ads_der(p));
        for(auto& t : t_use) { ADS ta(t); ta.derivatives().setZero(2); t_ads.push_back(ta); }
        PhysicalParameters<ADS> h_ads = to_ads_der(h_use);
        
        Eigen::Matrix<ADS, 5, 1> seabedState;
        seabedState.setZero();
        auto res = calculateDispersionFunction(l_ads, t_ads, h_ads, w_ad, c_ad, type, NormalizationMode::None, &seabedState);
        ADS Delta = res(I24);
        
        double dD_dc = Delta.derivatives()[0];
        double dD_dw = Delta.derivatives()[1];
        
        if (std::abs(dD_dc) < 1e-20) return std::make_tuple(c_exact, NAN, NAN);
        
        double dc_dw = - dD_dw / dD_dc;
        double U_val = c_exact / (1.0 - (w_ad.value()/c_exact)*dc_dw);
        
        double hv_val = 0.0;
        if (type == WaveType::Rayleigh) {
             bool topIsLiquid = false;
             if (!l_use.empty() && isLiquidLayer(l_use[0])) topIsLiquid = true;

             ADS hv_ads;
             if (topIsLiquid) hv_ads = calculateHVRatio(seabedState);
             else hv_ads = calculateHVRatio(res);
             hv_val = hv_ads.value();
        } else {
             hv_val = std::numeric_limits<double>::quiet_NaN();
        }
        
        return std::make_tuple(c_exact, U_val, hv_val);
    };

    // calculate center state (technically not needed for central difference, but good for guess)
    auto state0 = calc_full_state(layers, thicknesses, halfSpace, freq, root_c);
    double c0 = std::get<0>(state0);
    if(std::isnan(c0)) throw std::runtime_error("Failed to find root for baseline model in sensitivity calc.");

    // Loop Layers + Halfspace
    int num_layers = layers.size();
    
    for(int i = 0; i <= num_layers; ++i) {
        if (mode == SensitivityMode::Isotropic) {
            std::string param_names[] = {"Vs", "Vp", "Rho"};
            PhysicalParameters<T> base_p = (i < num_layers) ? layers[i] : halfSpace;
            double rho = base_p.rho;
            double c44 = base_p.c44;
            double c11 = base_p.c11;
            double vs = std::sqrt(c44 / rho);
            double vp = std::sqrt(c11 / rho);
            
            for(int p_idx = 0; p_idx < 3; ++p_idx) {
                // Central Difference: p + delta, p - delta
                // Note: Isotropic parameterization (Vs, Vp, Rho)
                // We perturb Vs, Vp, Rho directly.
                
                double delta = 1e-4; // 0.01% perturbation - High precision for Romberg
                double p_val = (p_idx==0 ? vs : (p_idx==1 ? vp : rho));
                double perturbation = p_val * delta;
                
                auto calc_perturbed = [&](double sign) {
                    std::vector<PhysicalParameters<T>> l_pert = layers;
                    PhysicalParameters<T> h_pert = halfSpace;
                    PhysicalParameters<T>* p_ptr = (i < num_layers) ? &l_pert[i] : &h_pert;
                    
                    double vs_p = vs, vp_p = vp, rho_p = rho;
                    if (p_idx == 0) vs_p += sign * perturbation;
                    if (p_idx == 1) vp_p += sign * perturbation;
                    if (p_idx == 2) rho_p += sign * perturbation;
                    
                    // Reconstruct elastic constants
                    p_ptr->rho = rho_p;
                    p_ptr->c44 = rho_p * vs_p * vs_p;
                    p_ptr->c33 = rho_p * vp_p * vp_p;
                    p_ptr->c11 = p_ptr->c33;
                    p_ptr->c13 = p_ptr->c33 - 2.0 * p_ptr->c44; 
                    p_ptr->c66 = p_ptr->c44;
                    
                    return calc_full_state(l_pert, thicknesses, h_pert, freq, c0);
                };
                
                auto state_plus = calc_perturbed(1.0);
                auto state_minus = calc_perturbed(-1.0);
                
                double c_plus = std::get<0>(state_plus);
                double c_minus = std::get<0>(state_minus);
                double U_plus = std::get<1>(state_plus);
                double U_minus = std::get<1>(state_minus);
                double HV_plus = std::get<2>(state_plus);
                double HV_minus = std::get<2>(state_minus);
                
                if (std::isnan(c_plus) || std::isnan(c_minus)) {
                    // Fallback to 0 if calculation fails? Or throw?
                    // For now, partial result warning?
                    // Just set to 0 to avoid crash
                    c_plus=c_minus=c0; U_plus=U_minus=0; HV_plus=HV_minus=0;
                }

                SensitivityKernel k;
                k.parameterName = param_names[p_idx];
                k.layerIndex = (i < num_layers) ? i : -1;
                k.d_PhaseVelocity = (c_plus - c_minus) / (2.0 * perturbation);
                k.d_GroupVelocity = (U_plus - U_minus) / (2.0 * perturbation);
                k.d_HVRatio       = (HV_plus - HV_minus) / (2.0 * perturbation); // Note: HV sensitvity
                kernels.push_back(k);
            }
        } 
        else if (mode == SensitivityMode::VTI) {
            std::string param_names[] = {"Rho", "A", "C", "F", "L", "N"};
            PhysicalParameters<T> base_p = (i < num_layers) ? layers[i] : halfSpace;
            
            for(int p_idx = 0; p_idx < 6; ++p_idx) {
                double val_base = 0.0;
                if (p_idx==0) val_base=base_p.rho;
                else if (p_idx==1) val_base=base_p.c11;
                else if (p_idx==2) val_base=base_p.c33;
                else if (p_idx==3) val_base=base_p.c13;
                else if (p_idx==4) val_base=base_p.c44;
                else if (p_idx==5) val_base=base_p.c66;
                
                double delta = 1e-4;
                double perturbation = val_base * delta;
                
                auto calc_perturbed = [&](double sign) {
                    std::vector<PhysicalParameters<T>> l_pert = layers;
                    PhysicalParameters<T> h_pert = halfSpace;
                    PhysicalParameters<T>* p_ptr = (i < num_layers) ? &l_pert[i] : &h_pert;
                    
                    double val_new = val_base + sign * perturbation;
                    if (p_idx==0) p_ptr->rho = val_new;
                    else if (p_idx==1) p_ptr->c11 = val_new;
                    else if (p_idx==2) p_ptr->c33 = val_new;
                    else if (p_idx==3) p_ptr->c13 = val_new;
                    else if (p_idx==4) p_ptr->c44 = val_new;
                    else if (p_idx==5) p_ptr->c66 = val_new;
                    
                    return calc_full_state(l_pert, thicknesses, h_pert, freq, c0);
                };
                
                auto state_plus = calc_perturbed(1.0);
                auto state_minus = calc_perturbed(-1.0);
                
                double c_plus = std::get<0>(state_plus);
                double c_minus = std::get<0>(state_minus);
                double U_plus = std::get<1>(state_plus);
                double U_minus = std::get<1>(state_minus);
                double HV_plus = std::get<2>(state_plus);
                double HV_minus = std::get<2>(state_minus);
                
                if (std::isnan(c_plus) || std::isnan(c_minus)) {
                     c_plus=c_minus=c0; U_plus=U_minus=0; HV_plus=HV_minus=0;
                }

                SensitivityKernel k;
                k.parameterName = param_names[p_idx];
                k.layerIndex = (i < num_layers) ? i : -1;
                k.d_PhaseVelocity = (c_plus - c_minus) / (2.0 * perturbation);
                k.d_GroupVelocity = (U_plus - U_minus) / (2.0 * perturbation);
                k.d_HVRatio       = (HV_plus - HV_minus) / (2.0 * perturbation);
                kernels.push_back(k);
            }
        }
        
        // Thickness Sensitivity (Partial, for layer i < num_layers)
        if (computeThickness && i < num_layers) {
            double thickness = thicknesses[i];
            double delta = 1e-4;
            double perturbation = thickness * delta;
            if (perturbation < 1e-8) perturbation = 1e-8; // Avoid zero perturbation for thin layers

            auto calc_perturbed = [&](double sign) {
                std::vector<PhysicalParameters<T>> l_pert = layers;
                std::vector<T> t_pert = thicknesses;
                PhysicalParameters<T> h_pert = halfSpace;
                
                t_pert[i] += sign * perturbation;
                return calc_full_state(l_pert, t_pert, h_pert, freq, c0);
            };
            
            auto state_plus = calc_perturbed(1.0);
            auto state_minus = calc_perturbed(-1.0);
            
            double c_plus = std::get<0>(state_plus);
            double c_minus = std::get<0>(state_minus);
            double U_plus = std::get<1>(state_plus);
            double U_minus = std::get<1>(state_minus);
            double HV_plus = std::get<2>(state_plus);
            double HV_minus = std::get<2>(state_minus);
            
            if (std::isnan(c_plus) || std::isnan(c_minus)) {
                 c_plus=c_minus=c0; U_plus=U_minus=0; HV_plus=HV_minus=0;
            }

            SensitivityKernel k;
            k.parameterName = "Thickness";
            k.layerIndex = i;
            k.d_PhaseVelocity = (c_plus - c_minus) / (2.0 * perturbation);
            k.d_GroupVelocity = (U_plus - U_minus) / (2.0 * perturbation);
            k.d_HVRatio       = (HV_plus - HV_minus) / (2.0 * perturbation);
            kernels.push_back(k);
        }
    }
    
    
    return kernels;
}

// Helper to seed constant into AD type with proper size
template <typename T>
T seed_const(double val, int size) {
    if constexpr (std::is_same_v<T, double>) {
        return val;
    } else {
        T ad;
        ad.value() = val;
        if (size > 0) ad.derivatives() = Eigen::VectorXd::Zero(size);
        else ad.derivatives() = Eigen::VectorXd();
        return ad;
    }
}

// --- Full AD Group Velocity Calculation (Standalone) ---
// Calculates U using implicit differentiation of F(c, omega) = 0.
// Works for any AD type T (e.g., AD scalar w.r.t parameters).
// Uses VectorNorm normalization for stability.
template <typename T>
T calculateGroupVelocityFullAD(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega,
    T root_c,
    WaveType type = WaveType::Rayleigh,
    std::vector<double>* sFactors_provided = nullptr
) {
    // Nested AD type
    using InnerAD = Eigen::AutoDiffScalar<Eigen::Matrix<T, 2, 1>>;
    
    int ad_size = 0;
    if constexpr (!std::is_same_v<T, double>) {
        ad_size = omega.derivatives().size();
    }

    InnerAD w_in; 
    w_in.value() = omega; 
    w_in.derivatives() = Eigen::Matrix<T, 2, 1>::Zero();
    w_in.derivatives()(0) = seed_const<T>(1.0, ad_size); 

    InnerAD c_in; 
    c_in.value() = root_c; 
    c_in.derivatives() = Eigen::Matrix<T, 2, 1>::Zero();
    c_in.derivatives()(1) = seed_const<T>(1.0, ad_size); 
    
    auto to_inner = [ad_size](const PhysicalParameters<T>& p) {
        PhysicalParameters<InnerAD> pi;
        pi.rho.value() = p.rho; pi.rho.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.rho.derivatives()(j) = seed_const<T>(0.0, ad_size);
        
        pi.c11.value() = p.c11; pi.c11.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c11.derivatives()(j) = seed_const<T>(0.0, ad_size);
        
        pi.c33.value() = p.c33; pi.c33.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c33.derivatives()(j) = seed_const<T>(0.0, ad_size);
        
        pi.c44.value() = p.c44; pi.c44.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c44.derivatives()(j) = seed_const<T>(0.0, ad_size);
        
        pi.c66.value() = p.c66; pi.c66.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c66.derivatives()(j) = seed_const<T>(0.0, ad_size);
        
        pi.c13.value() = p.c13; pi.c13.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c13.derivatives()(j) = seed_const<T>(0.0, ad_size);
        

        // Gravity is a constant with respect to omega and c, but it has to be
        // carried across: the fluid propagator switches to a different 2x2
        // matrix when it is nonzero, and there is an optional free-surface
        // correction keyed the same way. Leaving these at their defaults
        // would silently drop gravity from the derivative for a Cowling ocean
        // model (see docs/Cowling_Kg.md).
        auto carry = [&](InnerAD& dst, const T& src) {
            dst.value() = src; dst.derivatives().setZero();
            for(int j=0; j<2; ++j) dst.derivatives()(j) = seed_const<T>(0.0, ad_size);
        };
        carry(pi.gravity, p.gravity);
        carry(pi.gravity_internal_scale, p.gravity_internal_scale);
        carry(pi.gravity_surface_scale, p.gravity_surface_scale);

        return pi;
    };
    
    std::vector<PhysicalParameters<InnerAD>> l_in;
    std::vector<InnerAD> t_in;
    if constexpr (std::is_same_v<T, double>) {
        l_in.reserve(layers.size());
        t_in.reserve(thicknesses.size());
    }
    for(auto& p : layers) l_in.push_back(to_inner(p));
    for(auto& t : thicknesses) { 
        InnerAD ti; ti.value() = t; 
        ti.derivatives().setZero(); 
        for(int j=0; j<2; ++j) ti.derivatives()(j) = seed_const<T>(0.0, ad_size);
        t_in.push_back(ti); 
    }
    PhysicalParameters<InnerAD> h_in = to_inner(halfSpace);
    
    Eigen::Matrix<InnerAD, 5, 1>* no_seabed_grp = nullptr;
    size_t sIdx_grp = 0;
    NormalizationMode grp_norm = sFactors_provided ? NormalizationMode::Provided : NormalizationMode::Vertical;
    auto res = calculateDispersionFunction(l_in, t_in, h_in, w_in, c_in, type, grp_norm,
                                           no_seabed_grp, sFactors_provided, sFactors_provided ? &sIdx_grp : nullptr);
    InnerAD Delta = res(I24);
    
    T dD_dw = Delta.derivatives()(0);
    T dD_dc = Delta.derivatives()(1);
    
    if (get_magnitude(dD_dc) < 1e-20) return seed_const<T>(std::numeric_limits<double>::quiet_NaN(), ad_size);
    
    T dc_dw = -dD_dw / dD_dc;
    T U = root_c / (seed_const<T>(1.0, ad_size) - (omega/root_c)*dc_dw);
    return U;
}

// Wrapper for dC/dOmega (or dC/dFreq) via AD
template <typename T>
T calculateDcDOmegaAD(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega,
    T root_c,
    WaveType type = WaveType::Rayleigh
) {
    // Re-uses logic from Group Velocity calculation to get dc/dw
    // See above function for comments
    using InnerAD = Eigen::AutoDiffScalar<Eigen::Matrix<T, 2, 1>>;

    int ad_size = 0;
    if constexpr (!std::is_same_v<T, double>) {
        ad_size = omega.derivatives().size();
    }

    InnerAD w_in; 
    w_in.value() = omega; 
    w_in.derivatives() = Eigen::Matrix<T, 2, 1>::Zero();
    w_in.derivatives()(0) = seed_const<T>(1.0, ad_size); 

    InnerAD c_in; 
    c_in.value() = root_c; 
    c_in.derivatives() = Eigen::Matrix<T, 2, 1>::Zero();
    c_in.derivatives()(1) = seed_const<T>(1.0, ad_size); 
    
    auto to_inner = [ad_size](const PhysicalParameters<T>& p) {
        PhysicalParameters<InnerAD> pi;
        pi.rho.value() = p.rho; pi.rho.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.rho.derivatives()(j) = seed_const<T>(0.0, ad_size);
        pi.c11.value() = p.c11; pi.c11.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c11.derivatives()(j) = seed_const<T>(0.0, ad_size);
        pi.c33.value() = p.c33; pi.c33.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c33.derivatives()(j) = seed_const<T>(0.0, ad_size);
        pi.c44.value() = p.c44; pi.c44.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c44.derivatives()(j) = seed_const<T>(0.0, ad_size);
        pi.c66.value() = p.c66; pi.c66.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c66.derivatives()(j) = seed_const<T>(0.0, ad_size);
        pi.c13.value() = p.c13; pi.c13.derivatives().setZero();
        for(int j=0; j<2; ++j) pi.c13.derivatives()(j) = seed_const<T>(0.0, ad_size);

        // Gravity is a constant with respect to omega and c, but it has to be
        // carried across: the fluid propagator switches to a different 2x2
        // matrix when it is nonzero, and there is an optional free-surface
        // correction keyed the same way. Leaving these at their defaults
        // would silently drop gravity from the derivative for a Cowling ocean
        // model (see docs/Cowling_Kg.md).
        auto carry = [&](InnerAD& dst, const T& src) {
            dst.value() = src; dst.derivatives().setZero();
            for(int j=0; j<2; ++j) dst.derivatives()(j) = seed_const<T>(0.0, ad_size);
        };
        carry(pi.gravity, p.gravity);
        carry(pi.gravity_internal_scale, p.gravity_internal_scale);
        carry(pi.gravity_surface_scale, p.gravity_surface_scale);

        return pi;
    };
    
    std::vector<PhysicalParameters<InnerAD>> l_in;
    std::vector<InnerAD> t_in;
    for(auto& p : layers) l_in.push_back(to_inner(p));
    for(auto& t : thicknesses) { 
        InnerAD ti; ti.value() = t; 
        ti.derivatives().setZero(); 
        for(int j=0; j<2; ++j) ti.derivatives()(j) = seed_const<T>(0.0, ad_size);
        t_in.push_back(ti); 
    }
    PhysicalParameters<InnerAD> h_in = to_inner(halfSpace);
    
    auto res = calculateDispersionFunction(l_in, t_in, h_in, w_in, c_in, type, NormalizationMode::VectorNorm);
    InnerAD Delta = res(I24);
    
    T dD_dw = Delta.derivatives()(0);
    T dD_dc = Delta.derivatives()(1);
    
     if (get_magnitude(dD_dc) < 1e-20) return seed_const<T>(std::numeric_limits<double>::quiet_NaN(), ad_size);
     
     return -dD_dw / dD_dc; // dc/dw
}

// --- Full AD Sensitivity Kernel Calculator ---
// Uses Vector Norm and Nested Differentiation
template <typename T>
std::vector<SensitivityKernel> computeSensitivityKernelsFullAD(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    double freq,
    double root_c_double,
    WaveType type = WaveType::Rayleigh,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool computeThickness = false
) {
    std::vector<SensitivityKernel> kernels;
    int num_layers = layers.size();
    
    double omega_val = 2.0 * M_PI * freq;
    
    int num_params_per_layer = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int layer_params_count = (layers.size() + 1) * num_params_per_layer;
    int total_params = layer_params_count + (computeThickness ? layers.size() : 0);
    
    using OuterAD = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    auto create_outer_ad_model = [&](int p_idx_start_offset) {
        std::vector<PhysicalParameters<OuterAD>> l_ad;
        std::vector<OuterAD> t_ad;
        PhysicalParameters<OuterAD> h_ad;
        
        // Helper to populate AD params
        int current_idx = 0;
        
        auto populate = [&](const PhysicalParameters<T>& source) {
            PhysicalParameters<OuterAD> dest;
             // Isotropic mapping
             if (mode == SensitivityMode::Isotropic) {
                 double rho = source.rho;
                 double vs = std::sqrt(source.c44 / rho);
                 double vp = std::sqrt(source.c11 / rho);
                 
                 OuterAD vs_ad(vs, total_params, current_idx++);
                 OuterAD vp_ad(vp, total_params, current_idx++);
                 OuterAD rho_ad(rho, total_params, current_idx++);
                 
                 dest.rho = rho_ad;
                 dest.c44 = rho_ad * vs_ad * vs_ad;
                 dest.c33 = rho_ad * vp_ad * vp_ad;
                 dest.c11 = dest.c33;
                 dest.c13 = dest.c33 - 2.0 * dest.c44;
                 dest.c66 = dest.c44;
             } else {
                 // VTI mapping (Rho, A, C, F, L, N)
                 dest.rho = OuterAD(source.rho, total_params, current_idx++);
                 dest.c11 = OuterAD(source.c11, total_params, current_idx++);
                 dest.c33 = OuterAD(source.c33, total_params, current_idx++);
                 dest.c13 = OuterAD(source.c13, total_params, current_idx++);
                 dest.c44 = OuterAD(source.c44, total_params, current_idx++);
                 dest.c66 = OuterAD(source.c66, total_params, current_idx++);
             }
             dest.gravity = seed_const<OuterAD>(source.gravity, total_params);
             dest.gravity_internal_scale = seed_const<OuterAD>(source.gravity_internal_scale, total_params);
             dest.gravity_surface_scale = seed_const<OuterAD>(source.gravity_surface_scale, total_params);
             return dest;
        };
        
        for(auto& p : layers) l_ad.push_back(populate(p));
        h_ad = populate(halfSpace);
        
        // Populate Thicknesses
        // If computeThickness is true, they get derivative indices AFTER all layer params
        int thick_base_idx = layer_params_count; 
        
        int t_idx = 0;
        for(auto& t : thicknesses) {
             OuterAD t_val;
             t_val.value() = t;
             if (computeThickness) {
                 t_val.derivatives() = Eigen::VectorXd::Zero(total_params);
                 t_val.derivatives()(thick_base_idx + t_idx) = 1.0;
                 t_idx++;
             } else {
                 t_val.derivatives() = Eigen::VectorXd::Zero(total_params);
             }
             t_ad.push_back(t_val);
        } 
        
        return std::make_tuple(l_ad, t_ad, h_ad);
    };
    
    auto [l_ad, t_ad, h_ad] = create_outer_ad_model(0);
    // Omega is constant w.r.t parameters
    OuterAD omega_ad;
    omega_ad.value() = omega_val;
    omega_ad.derivatives() = Eigen::VectorXd::Zero(total_params);
    
    // --- Pre-pass: collect normalization scale factors as double constants ---
    // Mathematical basis: normalization terms cancel at F=0 (IFT / scale-invariance),
    // so treating scale factors as constants in AD is exact (not an approximation).
    std::vector<double> sFactors_vnorm;  // for IFT AD pass (VectorNorm)
    std::vector<double> sFactors_vert;   // for group velocity (Vertical)
    calculateDispersionFunction<double>(layers, thicknesses, halfSpace,
        omega_val, root_c_double, type, NormalizationMode::VectorNorm,
        nullptr, &sFactors_vnorm, nullptr);
    calculateDispersionFunction<double>(layers, thicknesses, halfSpace,
        omega_val, root_c_double, type, NormalizationMode::Vertical,
        nullptr, &sFactors_vert, nullptr);

    // Idea 1: IFT — compute dF/dc analytically via 1D AutoDiff (T=double, 1 derivative).
    // Cheaper than the 2D NestedAD previously used; machine-precision accuracy.
    // The ratio ∂F/∂p / ∂F/∂c is normalization-independent at the root.
    double dF_dc_double;
    {
        using AD1 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;
        AD1 c_d; c_d.value() = root_c_double; c_d.derivatives()(0) = 1.0;
        AD1 w_d; w_d.value() = omega_val;      w_d.derivatives()(0) = 0.0;
        auto to_ad1 = [](const PhysicalParameters<double>& p) {
            PhysicalParameters<AD1> pa;
            auto wrap = [](double v) { AD1 a; a.value() = v; a.derivatives()(0) = 0.0; return a; };
            pa.rho=wrap(p.rho); pa.c11=wrap(p.c11); pa.c33=wrap(p.c33);
            pa.c44=wrap(p.c44); pa.c66=wrap(p.c66); pa.c13=wrap(p.c13);
            pa.gravity=wrap(p.gravity);
            pa.gravity_internal_scale=wrap(p.gravity_internal_scale);
            pa.gravity_surface_scale=wrap(p.gravity_surface_scale);
            return pa;
        };
        std::vector<PhysicalParameters<AD1>> l_d1;
        for (auto& p : layers) l_d1.push_back(to_ad1(p));
        std::vector<AD1> t_d1;
        for (auto t : thicknesses) { AD1 a; a.value()=t; a.derivatives()(0)=0.0; t_d1.push_back(a); }
        PhysicalParameters<AD1> h_d1 = to_ad1(halfSpace);
        Eigen::Matrix<AD1, 5, 1>* no_sb_d1 = nullptr;
        size_t sIdx_d1 = 0;
        auto res_d1 = calculateDispersionFunction(l_d1, t_d1, h_d1, w_d, c_d, type,
                                                  NormalizationMode::Provided, no_sb_d1,
                                                  &sFactors_vnorm, &sIdx_d1);
        dF_dc_double = res_d1(I24).derivatives()(0);
    }

    // 1. IFT: single OuterAD pass — replaces Newton loop (2 OuterAD + 2 NestedAD calls).
    // dc/dp_i = -∂F(c_root, p)/∂p_i / (∂F/∂c)_double
    // c_root is the exact double root; c_ad starts with zero derivatives (constant),
    // then IFT sets the correct implicit derivatives after the evaluation.
    OuterAD c_ad;
    c_ad.value() = root_c_double;
    c_ad.derivatives() = Eigen::VectorXd::Zero(total_params);

    bool topIsLiquid = !layers.empty() && isLiquidLayer(layers[0]);

    size_t sIdx_ift = 0;
    Eigen::Matrix<OuterAD, 5, 1>* no_seabed = nullptr;
    auto res_ift = calculateDispersionFunction(l_ad, t_ad, h_ad, omega_ad, c_ad, type,
                                               NormalizationMode::Provided, no_seabed,
                                               &sFactors_vnorm, &sIdx_ift);
    OuterAD F_ad = res_ift(I24);

    // Apply IFT: dc/dp = -∂F/∂p / (∂F/∂c)
    if (std::abs(dF_dc_double) > 1e-20) {
        c_ad.derivatives() = -F_ad.derivatives() / dF_dc_double;
    }

    // 2. Compute Group Velocity U(p) — nested AD for dc/dω (Idea 3 would remove this)
    OuterAD U_ad = calculateGroupVelocityFullAD(l_ad, t_ad, h_ad, omega_ad, c_ad, type, &sFactors_vert);

    // 3. H/V (if Rayleigh) — Idea 2: second OuterAD pass with c_ad now carrying correct dc/dp.
    // c_ad.derivatives() must be set before this call so that implicit c(p) sensitivity propagates
    // into the eigenfunction components used by calculateHVRatio.
    OuterAD HV_ad;
    if (type == WaveType::Rayleigh) {
        Eigen::Matrix<OuterAD, 5, 1> seabedState_ad;
        seabedState_ad.setZero();
        size_t sIdx_hv = 0;
        auto res_hv = calculateDispersionFunction(l_ad, t_ad, h_ad, omega_ad, c_ad, type,
                                                  NormalizationMode::Provided, &seabedState_ad,
                                                  &sFactors_vert, &sIdx_hv);
        if (topIsLiquid) {
            HV_ad = calculateHVRatio(seabedState_ad);
        } else {
            HV_ad = calculateHVRatio(res_hv);
        }
    }
    
    // 4. Extract Sensitivities
    std::string iso_names[] = {"Vs", "Vp", "Rho"};
    std::string vti_names[] = {"Rho", "A", "C", "F", "L", "N"};
    int n_names = (mode==SensitivityMode::Isotropic) ? 3 : 6;
    
    int global_idx = 0;
    for(int i=0; i<=num_layers; ++i) {
        for(int p_idx=0; p_idx<n_names; ++p_idx) {
             SensitivityKernel k;
             k.parameterName = (mode==SensitivityMode::Isotropic) ? iso_names[p_idx] : vti_names[p_idx];
             k.layerIndex = (i < num_layers) ? i : -1;
             
             int d_idx = global_idx++;
             k.d_PhaseVelocity = c_ad.derivatives()[d_idx];
             k.d_GroupVelocity = U_ad.derivatives()[d_idx];
             if (type == WaveType::Rayleigh) k.d_HVRatio = HV_ad.derivatives()[d_idx];
             else k.d_HVRatio = NAN;
             
             kernels.push_back(k);
        }
    }
    
    // Extract Thickness Sensitivities
    if (computeThickness) {
         int thick_base_idx = layer_params_count;
         for(int i=0; i < num_layers; ++i) {
             SensitivityKernel k;
             k.parameterName = "Thickness";
             k.layerIndex = i;
             
             int d_idx = thick_base_idx + i;
             k.d_PhaseVelocity = c_ad.derivatives()[d_idx];
             k.d_GroupVelocity = U_ad.derivatives()[d_idx];
             if (type == WaveType::Rayleigh) k.d_HVRatio = HV_ad.derivatives()[d_idx];
             else k.d_HVRatio = NAN;
             
             kernels.push_back(k);
         }
    }
    
    return kernels;
}

// =============================================================================
// HessianAD: Forward-mode 2nd-order AD for phase-velocity Hessian computation
// =============================================================================

struct HessianResult {
    Eigen::MatrixXd hessian;
    Eigen::VectorXd gradient;
    std::vector<std::string> parameterNames;
    std::vector<int> layerIndices;
};

const int MAX_AD_PARAMS = 80; 

template<int N>
struct HessianAD {
    // Full AD calls may run concurrently with different parameter counts.
    // Keep the active width local to the evaluating thread.
    static inline thread_local int s_an = N;

    double r, i;
    bool is_const;
    Eigen::Matrix<double, N, 1> gr, gi;
    std::shared_ptr<Eigen::MatrixXd> hr_ptr, hi_ptr;

    HessianAD(double rv = 0, double iv = 0) : r(rv), i(iv), is_const(true) {
        gr.setZero(); gi.setZero();
        int n = s_an;
        hr_ptr = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(n, n));
        hi_ptr = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(n, n));
    }
    
    HessianAD(const HessianAD& o) : r(o.r), i(o.i), is_const(o.is_const), gr(o.gr), gi(o.gi) {
        hr_ptr = std::make_shared<Eigen::MatrixXd>(*(o.hr_ptr));
        hi_ptr = std::make_shared<Eigen::MatrixXd>(*(o.hi_ptr));
    }

    HessianAD(const HessianAD& real_p, const HessianAD& imag_p)
        : r(real_p.r), i(imag_p.r), is_const(real_p.is_const && imag_p.is_const) {
        gr = real_p.gr; gi = imag_p.gr; 
        hr_ptr = std::make_shared<Eigen::MatrixXd>(*(real_p.hr_ptr));
        hi_ptr = std::make_shared<Eigen::MatrixXd>(*(imag_p.hr_ptr));
    }

    HessianAD& operator=(const HessianAD& o) {
        if(this != &o) {
            r = o.r; i = o.i; is_const = o.is_const; gr = o.gr; gi = o.gi;
            hr_ptr = std::make_shared<Eigen::MatrixXd>(*(o.hr_ptr));
            hi_ptr = std::make_shared<Eigen::MatrixXd>(*(o.hi_ptr));
        }
        return *this;
    }

    HessianAD(HessianAD&& o) noexcept = default;
    HessianAD& operator=(HessianAD&& o) noexcept = default;

    static HessianAD make_var(double v, int idx) { 
        HessianAD res(v);
        if (idx >= 0 && idx < N) { res.gr(idx) = 1.0; res.is_const = false; }
        return res; 
    }
    
    Eigen::MatrixXd& hr() { return *hr_ptr; }
    const Eigen::MatrixXd& hr() const { return *hr_ptr; }
    Eigen::MatrixXd& hi() { return *hi_ptr; }
    const Eigen::MatrixXd& hi() const { return *hi_ptr; }

    HessianAD real() const { HessianAD res(r, 0); res.is_const = is_const; res.gr = gr; res.hr_ptr = hr_ptr; return res; }
    HessianAD imag() const { HessianAD res(i, 0); res.is_const = is_const; res.gr = gi; res.hr_ptr = hi_ptr; return res; }
    explicit operator double() const { return r; }
};

// ComplexType specialization: HessianAD packs real+imag internally
template<int N> struct ComplexType<HessianAD<N>> { using type = HessianAD<N>; };

// Helper overloads for HessianAD (visible at template instantiation point)
template<int N> inline double get_scalar_val(const HessianAD<N>& x) { return x.r; }
template<int N> inline double get_val(const HessianAD<N>& x) { return x.r; }
template<int N> inline double get_magnitude(const HessianAD<N>& x) { return std::abs(x.r) + std::abs(x.i); }
template<int N> HessianAD<N> abs(const HessianAD<N>& x) {
    if (x.r < 0.0) return -x;
    HessianAD<N> res = x;
    return res;
}

// --- HessianAD Operators ---
template<int N> HessianAD<N> operator+(const HessianAD<N>& a, const HessianAD<N>& b) {
    int n = HessianAD<N>::s_an;
    HessianAD<N> res(a.r + b.r, a.i + b.i);
    res.is_const = a.is_const && b.is_const;
    res.gr.head(n) = a.gr.head(n) + b.gr.head(n);
    res.gi.head(n) = a.gi.head(n) + b.gi.head(n);
    res.hr() = a.hr() + b.hr(); res.hi() = a.hi() + b.hi(); return res;
}
template<int N> HessianAD<N> operator-(const HessianAD<N>& a, const HessianAD<N>& b) {
    int n = HessianAD<N>::s_an;
    HessianAD<N> res(a.r - b.r, a.i - b.i);
    res.is_const = a.is_const && b.is_const;
    res.gr.head(n) = a.gr.head(n) - b.gr.head(n);
    res.gi.head(n) = a.gi.head(n) - b.gi.head(n);
    res.hr() = a.hr() - b.hr(); res.hi() = a.hi() - b.hi(); return res;
}
template<int N> HessianAD<N> operator*(const HessianAD<N>& a, const HessianAD<N>& b) {
    if (b.is_const) {
        int n = HessianAD<N>::s_an;
        HessianAD<N> res(a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r);
        res.is_const = a.is_const;
        res.gr.head(n) = a.gr.head(n)*b.r - a.gi.head(n)*b.i;
        res.gi.head(n) = a.gr.head(n)*b.i + a.gi.head(n)*b.r;
        res.hr().noalias() = a.hr()*b.r - a.hi()*b.i;
        res.hi().noalias() = a.hr()*b.i + a.hi()*b.r;
        return res;
    }
    if (a.is_const) {
        int n = HessianAD<N>::s_an;
        HessianAD<N> res(a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r);
        res.is_const = false;
        res.gr.head(n) = a.r*b.gr.head(n) - a.i*b.gi.head(n);
        res.gi.head(n) = a.r*b.gi.head(n) + a.i*b.gr.head(n);
        res.hr().noalias() = a.r*b.hr() - a.i*b.hi();
        res.hi().noalias() = a.r*b.hi() + a.i*b.hr();
        return res;
    }
    int n = HessianAD<N>::s_an;
    HessianAD<N> res(a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r);
    res.is_const = false;
    res.gr.head(n) = a.gr.head(n)*b.r - a.gi.head(n)*b.i + a.r*b.gr.head(n) - a.i*b.gi.head(n);
    res.gi.head(n) = a.gr.head(n)*b.i + a.gi.head(n)*b.r + a.r*b.gi.head(n) + a.i*b.gr.head(n);
    auto ah = a.gr.head(n); auto bh = b.gr.head(n);
    auto ai = a.gi.head(n); auto bi = b.gi.head(n);
    res.hr().noalias() = a.hr()*b.r - a.hi()*b.i + a.r*b.hr() - a.i*b.hi();
    res.hr().noalias() += ah * bh.transpose() + bh * ah.transpose() - ai * bi.transpose() - bi * ai.transpose();
    res.hi().noalias() = a.hr()*b.i + a.hi()*b.r + a.r*b.hi() + a.i*b.hr();
    res.hi().noalias() += ah * bi.transpose() + bi * ah.transpose() + ai * bh.transpose() + bh * ai.transpose();
    return res;
}
template<int N> HessianAD<N> operator*(const HessianAD<N>& a, double b) {
    int n = HessianAD<N>::s_an;
    HessianAD<N> res(a.r*b, a.i*b);
    res.is_const = a.is_const;
    res.gr.head(n) = a.gr.head(n)*b; res.gi.head(n) = a.gi.head(n)*b;
    res.hr() = a.hr()*b; res.hi() = a.hi()*b; return res;
}
template<int N> HessianAD<N> operator*(double a, const HessianAD<N>& b) { return b * a; }
template<int N> HessianAD<N> operator-(const HessianAD<N>& a) {
    int n = HessianAD<N>::s_an;
    HessianAD<N> res(-a.r, -a.i);
    res.is_const = a.is_const;
    res.gr.head(n) = -a.gr.head(n); res.gi.head(n) = -a.gi.head(n);
    res.hr() = -a.hr(); res.hi() = -a.hi(); return res;
}
template<int N> HessianAD<N> operator+(const HessianAD<N>& a, double b) { HessianAD<N> res = a; res.r += b; return res; }
template<int N> HessianAD<N> operator+(double a, const HessianAD<N>& b) { return b + a; }
template<int N> HessianAD<N> operator-(const HessianAD<N>& a, double b) { HessianAD<N> res = a; res.r -= b; return res; }
template<int N> HessianAD<N> operator-(double a, const HessianAD<N>& b) { return (-b) + a; }

// Compound assignment operators (needed for template branches that Eigen/normalization code may instantiate)
template<int N> HessianAD<N>& operator+=(HessianAD<N>& a, const HessianAD<N>& b) { a = a + b; return a; }
template<int N> HessianAD<N>& operator-=(HessianAD<N>& a, const HessianAD<N>& b) { a = a - b; return a; }
template<int N> HessianAD<N>& operator*=(HessianAD<N>& a, const HessianAD<N>& b) { a = a * b; return a; }

// Division by double (needed for normalization branches)
template<int N> HessianAD<N> operator/(const HessianAD<N>& a, double b) { return a * (1.0/b); }
template<int N> HessianAD<N> operator/(double a, const HessianAD<N>& b) {
    HessianAD<N> num(a); return num / b;
}

// Real-valued sqrt for HessianAD (used by VectorNorm normalization branches)
template<int N> HessianAD<N> sqrt(const HessianAD<N>& x) {
    // Treat as real-only: sqrt(x.r), ignore imaginary part
    double sv = std::sqrt(std::abs(x.r) + 1e-30);
    HessianAD<N> res(sv, 0.0);
    if (!x.is_const) {
        int n = HessianAD<N>::s_an;
        double dfdx = 0.5 / sv;
        res.is_const = false;
        res.gr.head(n) = dfdx * x.gr.head(n);
        res.gi.head(n).setZero();
        res.hr() = dfdx * x.hr();
        double d2fdx2 = -0.25 / (sv * sv * sv);
        auto xr = x.gr.head(n);
        res.hr().noalias() += d2fdx2 * (xr * xr.transpose());
        res.hi().setZero();
    }
    return res;
}

// Eigen NumTraits for HessianAD (needed for Eigen matrix operations with HessianAD scalar)
namespace Eigen {
template<int N> struct NumTraits<HessianAD<N>> : NumTraits<double> {
    typedef HessianAD<N> Real;
    typedef HessianAD<N> NonInteger;
    typedef HessianAD<N> Nested;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};
} // namespace Eigen

// --- HessianAD Math Rules ---
template<int N> void apply_ad_rule(HessianAD<N>& res, const HessianAD<N>& z, std::complex<double> fp, std::complex<double> ffp) {
    int n = HessianAD<N>::s_an;
    res.is_const = z.is_const;
    res.gr.head(n) = fp.real() * z.gr.head(n) - fp.imag() * z.gi.head(n);
    res.gi.head(n) = fp.real() * z.gi.head(n) + fp.imag() * z.gr.head(n);
    if (z.is_const) {
        res.hr().setZero(); res.hi().setZero();
        return;
    }
    auto zr = z.gr.head(n); auto zi = z.gi.head(n);
    res.hr().noalias() = fp.real()*z.hr() - fp.imag()*z.hi();
    res.hr().noalias() += ffp.real() * (zr*zr.transpose() - zi*zi.transpose())
                        - ffp.imag() * (zr*zi.transpose() + zi*zr.transpose());
    res.hi().noalias() = fp.real()*z.hi() + fp.imag()*z.hr();
    res.hi().noalias() += ffp.imag() * (zr*zr.transpose() - zi*zi.transpose())
                        + ffp.real() * (zr*zi.transpose() + zi*zr.transpose());
}
template<int N> HessianAD<N> operator/(const HessianAD<N>& a, const HessianAD<N>& b) {
    double d2 = b.r*b.r + b.i*b.i + 1e-30; std::complex<double> z(b.r, b.i);
    HessianAD<N> inv_b(b.r/d2, -b.i/d2); apply_ad_rule(inv_b, b, -1.0/(z*z), 2.0/(z*z*z)); return a * inv_b;
}

// HessianAD overloads for math functions used in dispersion computation
template<int N> HessianAD<N> safe_complex_sqrt(const HessianAD<N>& z) {
    std::complex<double> zv(z.r, z.i); std::complex<double> wv = std::sqrt(zv + 1e-25);
    HessianAD<N> res(wv.real(), wv.imag()); apply_ad_rule(res, z, 0.5/wv, -0.25/(zv*wv + 1e-25)); return res;
}
template<int N> HessianAD<N> ad_cosh(const HessianAD<N>& z) {
    std::complex<double> zv(z.r, z.i); HessianAD<N> res(std::cosh(zv).real(), std::cosh(zv).imag());
    apply_ad_rule(res, z, std::sinh(zv), std::cosh(zv)); return res;
}
template<int N> HessianAD<N> ad_sinh(const HessianAD<N>& z) {
    std::complex<double> zv(z.r, z.i); HessianAD<N> res(std::sinh(zv).real(), std::sinh(zv).imag());
    apply_ad_rule(res, z, std::cosh(zv), std::sinh(zv)); return res;
}
template<int N> HessianAD<N> cosh(const HessianAD<N>& z) { return ad_cosh(z); }
template<int N> HessianAD<N> sinh(const HessianAD<N>& z) { return ad_sinh(z); }
template<int N> HessianAD<N> exp(const HessianAD<N>& z) {
    std::complex<double> zv(z.r, z.i);
    std::complex<double> ev = std::exp(zv);
    HessianAD<N> res(ev.real(), ev.imag());
    apply_ad_rule(res, z, ev, ev);
    return res;
}
template<int N> HessianAD<N> sin(const HessianAD<N>& z) {
    std::complex<double> zv(z.r, z.i);
    std::complex<double> sv = std::sin(zv);
    HessianAD<N> res(sv.real(), sv.imag());
    apply_ad_rule(res, z, std::cos(zv), -sv);
    return res;
}
template<int N> HessianAD<N> cos(const HessianAD<N>& z) {
    std::complex<double> zv(z.r, z.i);
    std::complex<double> cv = std::cos(zv);
    HessianAD<N> res(cv.real(), cv.imag());
    apply_ad_rule(res, z, -std::sin(zv), -cv);
    return res;
}

// toComplex for HessianAD: the general template in disper26.cpp handles this correctly
// via HessianAD(real_part, imag_part) constructor -- no specialization needed.
// The previous specialization `return HessianAD<N>(val.r, 0.0)` was WRONG:
// it dropped all derivatives (gr, gi, hr, hi).

// make_nan for HessianAD
template<int N> HessianAD<N> make_nan_hessian() { return HessianAD<N>(std::numeric_limits<double>::quiet_NaN()); }

// make_zero_like for HessianAD
template<int N> HessianAD<N> make_zero_like(const HessianAD<N>&) { return HessianAD<N>(0.0, 0.0); }

// ===========================================================================
// ADComplex<T>: Complex number wrapper for autodiff::dual compatibility
// ===========================================================================
// std::complex<autodiff::dual> is undefined behavior in C++.
// ADComplex<T> provides a drop-in replacement that carries AD derivatives
// through complex arithmetic. Used via ComplexType<autodiff::dual> specialization.
// ===========================================================================

template<typename T>
struct ADComplex {
    T re, im;

    ADComplex() : re(T(0.0)), im(T(0.0)) {}
    ADComplex(const T& r) : re(r), im(T(0.0)) {}  // implicit from real
    ADComplex(const T& r, const T& i) : re(r), im(i) {}

    T real() const { return re; }
    T imag() const { return im; }

    ADComplex operator+(const ADComplex& o) const { return {re + o.re, im + o.im}; }
    ADComplex operator-(const ADComplex& o) const { return {re - o.re, im - o.im}; }
    ADComplex operator-() const { return {-re, -im}; }

    ADComplex operator*(const ADComplex& o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
    ADComplex operator/(const ADComplex& o) const {
        T denom = o.re * o.re + o.im * o.im;
        return {(re * o.re + im * o.im) / denom, (im * o.re - re * o.im) / denom};
    }

    ADComplex& operator+=(const ADComplex& o) { re += o.re; im += o.im; return *this; }
    ADComplex& operator-=(const ADComplex& o) { re -= o.re; im -= o.im; return *this; }
    ADComplex& operator*=(const ADComplex& o) { *this = *this * o; return *this; }
    ADComplex& operator/=(const ADComplex& o) { *this = *this / o; return *this; }

    // Scalar operations
    ADComplex operator*(const T& s) const { return {re * s, im * s}; }
    ADComplex operator/(const T& s) const { return {re / s, im / s}; }
    ADComplex operator+(const T& s) const { return {re + s, im}; }
    ADComplex operator-(const T& s) const { return {re - s, im}; }

    friend ADComplex operator*(const T& s, const ADComplex& z) { return {s * z.re, s * z.im}; }
    friend ADComplex operator+(const T& s, const ADComplex& z) { return {s + z.re, z.im}; }
    friend ADComplex operator-(const T& s, const ADComplex& z) { return {s - z.re, -z.im}; }
    friend ADComplex operator/(const T& s, const ADComplex& z) {
        T denom = z.re * z.re + z.im * z.im;
        return {s * z.re / denom, -s * z.im / denom};
    }

    // double-scalar operations (needed for expressions like T(1.0)/z in existing code)
    ADComplex operator*(double s) const { return {re * s, im * s}; }
    ADComplex operator/(double s) const { return {re / s, im / s}; }
    friend ADComplex operator*(double s, const ADComplex& z) { return {z.re * s, z.im * s}; }
    friend ADComplex operator/(double s, const ADComplex& z) {
        T denom = z.re * z.re + z.im * z.im;
        return {T(s) * z.re / denom, T(-s) * z.im / denom};
    }
};

// Math functions for ADComplex (found via ADL)
template<typename T>
ADComplex<T> conj(const ADComplex<T>& z) { return {z.re, -z.im}; }

template<typename T>
ADComplex<T> exp(const ADComplex<T>& z) {
    using std::exp; using std::cos; using std::sin;
    T er = exp(z.re);
    return {er * cos(z.im), er * sin(z.im)};
}

template<typename T>
ADComplex<T> cosh(const ADComplex<T>& z) {
    // cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
    using std::cosh; using std::sinh; using std::cos; using std::sin;
    return {cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im)};
}

template<typename T>
ADComplex<T> sinh(const ADComplex<T>& z) {
    // sinh(a+bi) = sinh(a)cos(b) + i cosh(a)sin(b)
    using std::cosh; using std::sinh; using std::cos; using std::sin;
    return {sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im)};
}

// Polar form (mirrors the disper26.cpp safe_complex_sqrt specializations for
// autodiff::var and Eigen::AutoDiffScalar). The previous cartesian form here
// —hard-branching on re_v>=0 at the value level, then calling sqrt(z.re)
// directly— has an inherently singular derivative right at re≈0 (the branch
// itself is fine, but d(sqrt(x))/dx=1/(2sqrt(x)) still diverges as x→0, and
// this is the ONLY complex sqrt path autodiff::dual actually uses — the
// ComplexType<autodiff::dual> specialization routes here, not through
// std::complex — so it is what backs every ν1=sqrt(ν1_sq) call in the "dual"
// Hessian backend). Empirically, near c≈Vp (ν1²≈0) this backend disagreed
// with a Richardson-extrapolated finite-difference reference by ~2%, while
// the polar-form Eigen::AutoDiffScalar path used by "vector"/"vector_dyn"
// was ~20x closer (~0.1%) to the same reference. Reusing the polar form here
// removes that asymmetry (docs/Weakform_AD_Hessian.md §4.6).
template<typename T>
ADComplex<T> sqrt(const ADComplex<T>& z) {
    using std::sqrt; using std::atan2; using std::cos; using std::sin;
    double re_v = get_val(z.re), im_v = get_val(z.im);

    if (std::abs(re_v) < 1e-30 && std::abs(im_v) < 1e-30) {
        return {T(0.0), T(0.0)};
    }

    // |z| = sqrt(re^2+im^2), theta = atan2(im,re), sqrt(z) = sqrt(|z|)*(cos(theta/2)+i sin(theta/2))
    T r = sqrt(z.re * z.re + z.im * z.im);
    T theta = atan2(z.im, z.re);
    T half_theta = theta * T(0.5);
    T sqrt_r = sqrt(r);
    return {sqrt_r * cos(half_theta), sqrt_r * sin(half_theta)};
}

// Magnitude (returns double, for normalization — no derivative needed)
template<typename T>
double complex_magnitude(const ADComplex<T>& z) {
    double re_v = get_val(z.re), im_v = get_val(z.im);
    return std::sqrt(re_v * re_v + im_v * im_v);
}

// Overload for std::complex<T> (so we can write generic code)
template<typename T>
double complex_magnitude(const std::complex<T>& z) {
    return std::abs(z);
}

// --- ComplexType specialization for autodiff::dual ---
template<> struct ComplexType<autodiff::dual> { using type = ADComplex<autodiff::dual>; };

// --- Overloads for ad_cosh, ad_sinh, safe_complex_sqrt with ADComplex ---
// Note: toComplex<autodiff::dual> works via the generic ComplexType<T> overload in disper26.cpp

template<typename T>
ADComplex<T> ad_cosh(const ADComplex<T>& z) { return cosh(z); }

template<typename T>
ADComplex<T> ad_sinh(const ADComplex<T>& z) { return sinh(z); }

template<typename T>
ADComplex<T> safe_complex_sqrt(const ADComplex<T>& z) { return sqrt(z); }

// ad_conj: generic conjugate that works for both std::complex and ADComplex
template<typename T>
ADComplex<T> ad_conj(const ADComplex<T>& z) { return conj(z); }

template<typename T>
std::complex<T> ad_conj(const std::complex<T>& z) { return std::conj(z); }

// ad_exp: generic complex exp that works for both std::complex and ADComplex
template<typename T>
ADComplex<T> ad_exp(const ADComplex<T>& z) { return exp(z); }

template<typename T>
std::complex<T> ad_exp(const std::complex<T>& z) { return std::exp(z); }

// ===========================================================================
// Stable small-argument helpers for exponential/hyperbolic layer integrals.
//
// These replace naive (exp(x)-1)/x forms whose relative accuracy degrades as
// O(eps/|x|) near x=0, and provide the closed forms needed to make the
// Rayleigh ν₃≈0 (c≈Vs) layer integrals exact for every ν₃ instead of only to
// linear order.
//
// All switch thresholds are DIMENSIONLESS (a wavenumber-like quantity times
// the layer thickness h) and evaluated via get_val, so branch selection is
// identical across double / autodiff::dual / AutoDiffScalar<...> — required
// for AD backend parity (see docs/Weakform_AD_Hessian.md §13, §4.6).
// ===========================================================================

// L1 magnitude of a complex-capable AD scalar's *value*. Branch selection
// only; never differentiated.
template <typename CT>
inline double ad_cmag1(const CT& z) {
    return std::abs(get_val(z.real())) + std::abs(get_val(z.imag()));
}

// Builds a real CT constant from an existing, fully-initialized `zero`
// (typically `x - x` for some in-scope CT value `x`) by adding `value` to
// its real component only. `std::complex<T>` has no `operator+(T, double)`
// when T isn't itself `double` (no implicit double->T conversion is
// considered through the template), so this goes through T's own `+double`
// directly on the already-safe `zero.real()`/`zero.imag()` components rather
// than constructing a fresh CT from a bare double (see the note above
// ad_phi1 for why that is unsafe for Eigen::AutoDiffScalar).
template <typename CT>
inline CT ad_const(const CT& zero, double value) {
    return CT(zero.real() + value, zero.imag());
}

constexpr double AD_SINHC_TAYLOR_THRESHOLD = 1.0;
constexpr double AD_PSI_TAYLOR_THRESHOLD = 0.5;
constexpr double AD_KS_SERIES_THRESHOLD = 0.05;

// Dimensionless switch between the standard 4-mode Cramer solve and the exact
// cosh/sinhc ν₃≈0 formulation in computeRayleighLayerIntegrals_Analytical
// (both accurate to ≲1e-13 across |ν3|h ∈ [0.05, 3] — see that function).
constexpr double RAYLEIGH_NU3H_EXACT_THRESHOLD = 0.5;

// Same switch for the mirror ν₁≈0 (c≈Vp) branch. Verified independently
// (Python + mpmath + numerical quadrature, docs/Weakform_AD_Hessian.md §4.6)
// against the standard 4-mode path away from degeneracy and against
// numerical quadrature near/at degeneracy.
constexpr double RAYLEIGH_NU1H_EXACT_THRESHOLD = 0.5;

// φ₁(x) = (e^x − 1)/x = Σ_{j≥0} x^j/(j+1)!. Also ψ₀ in the ψ_m family below.
// A "constant" CT is built via `ad_const(zero, literal)` throughout this
// block, NEVER as a bare `CT(literal)`. Reason: `std::complex<T>`'s
// single-argument constructor defaults the imaginary part to `T()` — for
// `Eigen::AutoDiffScalar` (T here whenever no ComplexType<T> specialization
// exists, e.g. the fixed-width vectorized Hessian backend), `T()` runs the
// *empty* default constructor, leaving BOTH value and derivatives as
// uninitialized memory (unlike `T(0.0)`, which zeros them). `ADComplex<T>`
// (used for autodiff::dual) already zeros its imaginary part explicitly and
// would not need this, but the helpers here are generic over both complex
// wrappers, so they follow the pattern that is safe for both.
template <typename CT>
inline CT ad_phi1(const CT& x) {
    CT zero = x - x;
    if (ad_cmag1(x) < AD_PSI_TAYLOR_THRESHOLD) {
        CT sum = ad_const(zero, 1.0);
        CT xp = ad_const(zero, 1.0);
        double fact = 1.0;
        for (int j = 1; j <= 12; ++j) {
            xp = xp * x;
            fact *= double(j + 1);
            sum = sum + xp / ad_const(zero, fact);
        }
        return sum;
    }
    return (ad_exp(x) - ad_const(zero, 1.0)) / x;
}

// sinh(w)/w — even, entire. Taylor for |w| < 1, closed form otherwise.
template <typename CT>
inline CT ad_sinhc(const CT& w) {
    CT zero = w - w;
    if (ad_cmag1(w) < AD_SINHC_TAYLOR_THRESHOLD) {
        CT w2 = w * w;
        CT sum = ad_const(zero, 1.0);
        CT wp = ad_const(zero, 1.0);
        double fact = 1.0;  // (2n+1)! at n=0
        for (int n = 1; n <= 8; ++n) {
            wp = wp * w2;
            fact *= (2.0 * n) * (2.0 * n + 1.0);
            sum = sum + wp / ad_const(zero, fact);
        }
        return sum;
    }
    return ad_sinh(w) / w;
}

// (sinhc(w) − 1)/w² = Σ_{n≥0} w^{2n}/(2n+3)!. Removes the sinhc≈1 cancellation
// that a direct (ad_sinhc(w)-1)/(w*w) would hit for small w.
template <typename CT>
inline CT ad_sinhc_m1_w2(const CT& w) {
    CT zero = w - w;
    if (ad_cmag1(w) < AD_SINHC_TAYLOR_THRESHOLD) {
        CT w2 = w * w;
        CT sum = ad_const(zero, 1.0 / 6.0);
        CT wp = ad_const(zero, 1.0);
        double fact = 6.0;  // 3! at n=0
        for (int n = 1; n <= 8; ++n) {
            wp = wp * w2;
            fact *= (2.0 * n + 2.0) * (2.0 * n + 3.0);
            sum = sum + wp / ad_const(zero, fact);
        }
        return sum;
    }
    return (ad_sinhc(w) - ad_const(zero, 1.0)) / (w * w);
}

// ψ_m(x) = ∫₀¹ tᵐ e^{xt} dt for m = 0..7, all at once. Closed form via the
// stable recursion ψ_m = (e^x − m·ψ_{m−1})/x for |x| not small (no
// cancellation there: e^x and m·ψ_{m−1} are not close for |x| ≳ 0.5); Taylor
// series ψ_m(x) = Σ_j x^j/(j!(m+j+1)) for small |x|.
template <typename CT>
inline void ad_psi_chain(const CT& x, CT psi_out[8]) {
    CT zero = x - x;
    if (ad_cmag1(x) < AD_PSI_TAYLOR_THRESHOLD) {
        for (int m = 0; m <= 7; ++m) {
            CT sum = ad_const(zero, 1.0 / (m + 1.0));
            CT xp = ad_const(zero, 1.0);
            double fact = 1.0;
            for (int j = 1; j <= 12; ++j) {
                xp = xp * x;
                fact *= double(j);
                sum = sum + xp / ad_const(zero, fact * double(m + j + 1));
            }
            psi_out[m] = sum;
        }
    } else {
        CT ex = ad_exp(x);
        psi_out[0] = (ex - ad_const(zero, 1.0)) / x;
        for (int m = 1; m <= 7; ++m)
            psi_out[m] = (ex - ad_const(zero, double(m)) * psi_out[m - 1]) / x;
    }
}

// K_C(μ) = ∫₀ʰ e^{μz}cosh(ν₃z)dz,  K_S(μ) = ∫₀ʰ e^{μz}·z·sinhc(ν₃z)dz.
// Both are exact and stable for every μ, ν₃ (dimensionless x=μh, y=ν₃h), but
// each has a removable y→0 singularity that only shows up under
// differentiation, not at the value level:
//   K_C = (h/2)[φ₁(x+y)+φ₁(x−y)],  K_S = h²/(2y)·[φ₁(x+y)−φ₁(x−y)]
// nu3 = sqrt(nu3_sq) has dy/dp = h/(2·ν3)·d(nu3_sq)/dp, which is *large* as
// ν3→0 even though nu3_sq itself is perfectly smooth. K_S's own value formula
// divides by y and so is visibly singular; K_C's is not, but AD still
// differentiates φ₁(x+y) and φ₁(x−y) SEPARATELY, each carrying that same
// large dy/dp factor, and their sum's derivative only becomes O(1) via a
// subtraction of two O(dy/dp)-sized numbers — losing ~log10(dy/dp) digits.
// For small y this reads back as a genuine Hessian error localized entirely
// to derivatives w.r.t. the parameter controlling ν3 (confirmed empirically:
// with only the value-safe form below, every large weak-form-vs-trapezoid
// Hessian discrepancy near c≈Vs involved exactly that parameter).
//
// The fix is the same for both: for small y, evaluate from the convergent
// series K_C = h·Σy^{2n}/(2n)!·ψ_{2n}(x), K_S = h²·Σy^{2n}/(2n+1)!·ψ_{2n+1}(x)
// (n=0..3), built from a SINGLE ψ-chain call at x=μh only — nu3 enters solely
// through y² = (ν3 h)², computed by one multiplication (y·y), which absorbs
// the 1/(2ν3) factor cleanly (a benign multiplicative cancellation, not a
// subtractive one). This reproduces the existing linear-in-z (y→0) forms
// exactly and stays accurate to O(y⁸) in the switch band.
template <typename CT>
inline void ad_KCKS(const CT& mu, const CT& nu3, const CT& h, CT& K_C, CT& K_S) {
    CT x = mu * h;
    CT y = nu3 * h;
    CT zero = x - x;

    if (ad_cmag1(y) < AD_KS_SERIES_THRESHOLD) {
        CT psi[8];
        ad_psi_chain(x, psi);
        CT y2 = y * y;
        CT y4 = y2 * y2;
        CT y6 = y4 * y2;
        K_C = h * (psi[0]
                  + (y2 / ad_const(zero, 2.0)) * psi[2]
                  + (y4 / ad_const(zero, 24.0)) * psi[4]
                  + (y6 / ad_const(zero, 720.0)) * psi[6]);
        K_S = h * h * (psi[1]
                      + (y2 / ad_const(zero, 6.0)) * psi[3]
                      + (y4 / ad_const(zero, 120.0)) * psi[5]
                      + (y6 / ad_const(zero, 5040.0)) * psi[7]);
    } else {
        K_C = (h * ad_const(zero, 0.5)) * (ad_phi1(x + y) + ad_phi1(x - y));
        K_S = (h * h * ad_const(zero, 0.5) / y) * (ad_phi1(x + y) - ad_phi1(x - y));
    }
}

// CC=∫₀ʰcosh(ν1z)cosh(ν3z)dz, CS=∫₀ʰcosh(ν1z)·z·sinhc(ν3z)dz,
// SC=∫₀ʰz·sinhc(ν1z)·cosh(ν3z)dz, SS=∫₀ʰz·sinhc(ν1z)·z·sinhc(ν3z)dz.
// Needed when BOTH ν1 and ν3 are near their own degeneracies simultaneously
// (computeRayleighLayerIntegrals_Analytical's ν1≈0-and-ν3≈0 branch): naively
// building these from two ad_KCKS(±ν3,ν1,h) calls and dividing the
// difference by ν3 (as one would for the single-cosh/sinhc case) reintroduces
// exactly the "naked 1/ν division" that ad_KCKS's own K_S avoids internally
// via its ψ-chain series — confirmed empirically: with that division,
// dual/vector AD backends disagree (~1e-4 near c≈Vp) even though the VALUE
// (T=double, no AD) is fine.
//
// Since z·sinhc(νz) is itself a smooth, entire function of ν (no 1/ν
// anywhere — sinhc removes it by construction), a direct double power series
// in y1²=(ν1h)², y3²=(ν3h)² has no division at all and is exact wherever it
// converges. Only used where BOTH |ν1h|,|ν3h| < 0.5 (guaranteed by the
// caller's branch condition), so no large-argument fallback is needed —
// unlike ad_KCKS this is not a value/derivative-safe reformulation of an
// otherwise-exact closed form, it is the only form used.
// Verified against direct numerical quadrature to ≲1e-11 up to |y1h|,
// |y3h|=0.475 (docs/Weakform_AD_Hessian.md §4.6).
template <typename CT>
inline void ad_cross_cosh_sinhc(const CT& y1, const CT& y3, CT& CC, CT& CS, CT& SC, CT& SS) {
    CT zero = y1 - y1;
    CT y1_2 = y1 * y1, y3_2 = y3 * y3;
    CT cc = zero, cs = zero, sc = zero, ss = zero;
    CT y1p = ad_const(zero, 1.0);
    double fact2j = 1.0, fact2jp1 = 1.0;
    for (int j = 0; j <= 5; ++j) {
        CT y3p = ad_const(zero, 1.0);
        double fact2k = 1.0, fact2kp1 = 1.0;
        for (int k = 0; k <= 5; ++k) {
            CT term = y1p * y3p;
            cc = cc + term / ad_const(zero, fact2j * fact2k * double(2*j + 2*k + 1));
            cs = cs + term / ad_const(zero, fact2j * fact2kp1 * double(2*j + 2*k + 2));
            sc = sc + term / ad_const(zero, fact2jp1 * fact2k * double(2*j + 2*k + 2));
            ss = ss + term / ad_const(zero, fact2jp1 * fact2kp1 * double(2*j + 2*k + 3));
            y3p = y3p * y3_2;
            fact2k *= (2.0*k + 1.0) * (2.0*k + 2.0);
            fact2kp1 *= (2.0*k + 2.0) * (2.0*k + 3.0);
        }
        y1p = y1p * y1_2;
        fact2j *= (2.0*j + 1.0) * (2.0*j + 2.0);
        fact2jp1 *= (2.0*j + 2.0) * (2.0*j + 3.0);
    }
    CC = cc; CS = cs; SC = sc; SS = ss;
}

#include "energy_entire.hpp"

// Eigen NumTraits for ADComplex<autodiff::dual>
namespace Eigen {
template<> struct NumTraits<ADComplex<autodiff::dual>> : NumTraits<double> {
    typedef ADComplex<autodiff::dual> Real;
    typedef ADComplex<autodiff::dual> NonInteger;
    typedef ADComplex<autodiff::dual> Nested;
    enum {
        IsComplex = 0,  // Treat as "real" to avoid Eigen conjugate calls
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 2,
        AddCost = 4,
        MulCost = 8
    };
};
} // namespace Eigen

// --- Hessian API ---
inline HessianResult computePhaseVelocityHessianStable(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    WaveType type = WaveType::Rayleigh,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool computeThickness = false
) {
    HessianResult result;
    const double omega_v = 2.0 * M_PI * freq;
    
    auto f_double = [&](double cv) {
        return calculateDispersionFunction<double>(layers, thicknesses, halfSpace, omega_v, cv, type, NormalizationMode::VectorNorm)(I24);
    };
    double c_root = root_c_double;
    for(int it=0; it<15; ++it) {
        double f0 = f_double(c_root), eps = c_root * 1e-7;
        double df = (f_double(c_root + eps) - f0) / eps;
        if(std::abs(df) < 1e-30) break;
        double step = f0 / df;
        c_root -= step;
        if(std::abs(step) < 1e-11) break;
    }
    
    using AD = HessianAD<MAX_AD_PARAMS>;
    int npl = (mode == SensitivityMode::VTI) ? 6 : 3;
    int num_params = layers.size() * npl + npl;
    int c_idx = num_params;
    if (c_idx + 1 > MAX_AD_PARAMS) {
        throw std::invalid_argument(
            "full-AD Hessian exceeds MAX_AD_PARAMS; use a smaller fixed grid");
    }
    AD::s_an = c_idx + 1;
    
    AD c_ad = AD::make_var(c_root, c_idx);
    std::vector<PhysicalParameters<AD>> layers_ad;
    int cur = 0;
    for(size_t i=0; i<layers.size(); ++i) {
        if (mode == SensitivityMode::Isotropic) {
            AD vs = AD::make_var(layers[i].vs(), cur);
            AD vp = AD::make_var(layers[i].vp(), cur+1);
            AD rho = AD::make_var(layers[i].rho, cur+2);
            layers_ad.push_back(createIsotropicParameters(vs, vp, rho));
        } else {
            PhysicalParameters<AD> p;
            p.rho = AD::make_var(layers[i].rho, cur);
            p.c11 = AD::make_var(layers[i].c11, cur + 1);
            p.c33 = AD::make_var(layers[i].c33, cur + 2);
            p.c13 = AD::make_var(layers[i].c13, cur + 3);
            p.c44 = AD::make_var(layers[i].c44, cur + 4);
            p.c66 = AD::make_var(layers[i].c66, cur + 5);
            layers_ad.push_back(p);
        }
        cur += npl;
    }
    PhysicalParameters<AD> hs_ad;
    if (mode == SensitivityMode::Isotropic) {
        AD hsvs = AD::make_var(halfSpace.vs(), cur);
        AD hsvp = AD::make_var(halfSpace.vp(), cur+1);
        AD hsrho = AD::make_var(halfSpace.rho, cur+2);
        hs_ad = createIsotropicParameters(hsvs, hsvp, hsrho);
    } else {
        hs_ad.rho = AD::make_var(halfSpace.rho, cur);
        hs_ad.c11 = AD::make_var(halfSpace.c11, cur + 1);
        hs_ad.c33 = AD::make_var(halfSpace.c33, cur + 2);
        hs_ad.c13 = AD::make_var(halfSpace.c13, cur + 3);
        hs_ad.c44 = AD::make_var(halfSpace.c44, cur + 4);
        hs_ad.c66 = AD::make_var(halfSpace.c66, cur + 5);
    }
    std::vector<AD> thick_ad;
    for(auto t : thicknesses) thick_ad.push_back(AD(t));
    
    std::vector<double> scales;
    calculateRayleighDispersion<double>(layers, thicknesses, halfSpace, omega_v, c_root, NormalizationMode::Vertical, nullptr, &scales);
    
    size_t sIdx = 0;
    auto Y = calculateRayleighDispersion<AD>(layers_ad, thick_ad, hs_ad, AD(omega_v), c_ad, NormalizationMode::Provided, nullptr, &scales, &sIdx);
    AD Fr = Y(I24);
    
    double Fc = Fr.gr(c_idx);
    Eigen::VectorXd Fp  = Fr.gr.head(num_params);
    Eigen::MatrixXd Fpp = Fr.hr().block(0,0,num_params,num_params);
    Eigen::VectorXd Fpc = Fr.hr().block(0,c_idx,num_params,1);
    double Fcc           = Fr.hr()(c_idx, c_idx);
    
    result.gradient = -Fp / Fc;
    const Eigen::VectorXd& g = result.gradient;
    result.hessian = -(Fpp + Fpc*g.transpose() + g*Fpc.transpose() + Fcc*g*g.transpose()) / Fc;
    
    for(size_t i=0; i<layers.size(); ++i) {
        if (mode == SensitivityMode::Isotropic) {
            result.parameterNames.push_back("Vs"); result.layerIndices.push_back(i);
            result.parameterNames.push_back("Vp"); result.layerIndices.push_back(i);
            result.parameterNames.push_back("Rho"); result.layerIndices.push_back(i);
        } else {
            const char* names[] = {"Rho", "A", "C", "F", "L", "N"};
            for (const char* name : names) {
                result.parameterNames.push_back(name); result.layerIndices.push_back(i);
            }
        }
    }
    if (mode == SensitivityMode::Isotropic) {
        result.parameterNames.push_back("Vs"); result.layerIndices.push_back(-1);
        result.parameterNames.push_back("Vp"); result.layerIndices.push_back(-1);
        result.parameterNames.push_back("Rho"); result.layerIndices.push_back(-1);
    } else {
        const char* names[] = {"Rho", "A", "C", "F", "L", "N"};
        for (const char* name : names) {
            result.parameterNames.push_back(name); result.layerIndices.push_back(-1);
        }
    }
    return result;
}

// --- Love Wave HessianAD (Option B: independent implementation) ---
// Computes phase velocity gradient and Hessian for Love waves via HessianAD.
// Uses calculateLoveDispersion with NormalizationMode::Provided (scale factors as constants).
// IFT: gradient = -Fp/Fc, hessian = -(Fpp + Fpc*g' + g*Fpc' + Fcc*g*g')/Fc
inline HessianResult computePhaseVelocityHessianLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool computeThickness = false
) {
    HessianResult result;
    const double omega_v = 2.0 * M_PI * freq;

    // Newton polish to refine root
    // Use NormalizationMode::None — same as search_roots, no spurious zeros.
    // Vertical mode can create artificial zeros when displacement passes through 0.
    auto f_double = [&](double cv) {
        auto Y = calculateLoveDispersion<double>(layers, thicknesses, halfSpace,
            omega_v, cv, NormalizationMode::None);
        return Y(I24);
    };
    // Valid Love wave range: c in (0, Vs_halfspace). Guard upper bound strictly.
    double c_max = halfSpace.vs() * (1.0 - 1e-6);
    double c_root = std::min(root_c_double, c_max);
    for(int it=0; it<15; ++it) {
        double f0 = f_double(c_root);
        if(std::isnan(f0) || std::isinf(f0)) { c_root = root_c_double; break; }
        double eps = c_root * 1e-7;
        double df = (f_double(c_root + eps) - f0) / eps;
        if(std::abs(df) < 1e-30 || std::isnan(df)) break;
        double step = f0 / df;
        // Limit step size to prevent overshoot past Vs_halfspace
        double max_step = 0.1 * (c_max - 1e-3);
        if (std::abs(step) > max_step) step = std::copysign(max_step, step);
        c_root = std::min(c_max, std::max(1e-3, c_root - step));
        if(std::abs(step) < 1e-11) break;
    }

    using AD = HessianAD<MAX_AD_PARAMS>;
    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = layers.size() * npl + npl;
    int c_idx = num_params;
    if (c_idx + 1 > MAX_AD_PARAMS) {
        throw std::invalid_argument(
            "full-AD Hessian exceeds MAX_AD_PARAMS; use a smaller fixed grid");
    }
    AD::s_an = c_idx + 1;

    AD c_ad = AD::make_var(c_root, c_idx);
    std::vector<PhysicalParameters<AD>> layers_ad;
    int cur = 0;
    for(size_t i=0; i<layers.size(); ++i) {
        if (mode == SensitivityMode::Isotropic) {
            AD vs  = AD::make_var(layers[i].vs(),  cur);
            AD vp  = AD::make_var(layers[i].vp(),  cur+1);
            AD rho = AD::make_var(layers[i].rho,   cur+2);
            layers_ad.push_back(createIsotropicParameters(vs, vp, rho));
        } else {
            PhysicalParameters<AD> p;
            p.rho = AD::make_var(layers[i].rho,  cur);
            p.c11 = AD::make_var(layers[i].c11,  cur+1);
            p.c33 = AD::make_var(layers[i].c33,  cur+2);
            p.c13 = AD::make_var(layers[i].c13,  cur+3);
            p.c44 = AD::make_var(layers[i].c44,  cur+4);
            p.c66 = AD::make_var(layers[i].c66,  cur+5);
            layers_ad.push_back(p);
        }
        cur += npl;
    }
    PhysicalParameters<AD> hs_ad;
    if (mode == SensitivityMode::Isotropic) {
        AD hsvs  = AD::make_var(halfSpace.vs(),  cur);
        AD hsvp  = AD::make_var(halfSpace.vp(),  cur+1);
        AD hsrho = AD::make_var(halfSpace.rho,   cur+2);
        hs_ad = createIsotropicParameters(hsvs, hsvp, hsrho);
    } else {
        hs_ad.rho = AD::make_var(halfSpace.rho,  cur);
        hs_ad.c11 = AD::make_var(halfSpace.c11,  cur+1);
        hs_ad.c33 = AD::make_var(halfSpace.c33,  cur+2);
        hs_ad.c13 = AD::make_var(halfSpace.c13,  cur+3);
        hs_ad.c44 = AD::make_var(halfSpace.c44,  cur+4);
        hs_ad.c66 = AD::make_var(halfSpace.c66,  cur+5);
    }
    std::vector<AD> thick_ad;
    for(auto t : thicknesses) thick_ad.push_back(AD(t));

    // Pre-pass: collect normalization scale factors as double constants.
    // Treating them as constants is exact at the root F=0 (IFT argument).
    std::vector<double> scales;
    calculateLoveDispersion<double>(layers, thicknesses, halfSpace,
        omega_v, c_root, NormalizationMode::Vertical, &scales);

    size_t sIdx = 0;
    auto Y = calculateLoveDispersion<AD>(layers_ad, thick_ad, hs_ad,
        AD(omega_v), c_ad, NormalizationMode::Provided, &scales, &sIdx);
    AD Fr = Y(I24);

    double Fc = Fr.gr(c_idx);

    Eigen::VectorXd Fp  = Fr.gr.head(num_params);
    Eigen::MatrixXd Fpp = Fr.hr().block(0,0,num_params,num_params);
    Eigen::VectorXd Fpc = Fr.hr().block(0,c_idx,num_params,1);
    double Fcc           = Fr.hr()(c_idx, c_idx);

    result.gradient = -Fp / Fc;
    const Eigen::VectorXd& g = result.gradient;
    result.hessian = -(Fpp + Fpc*g.transpose() + g*Fpc.transpose() + Fcc*g*g.transpose()) / Fc;

    if (mode == SensitivityMode::Isotropic) {
        std::string names[] = {"Vs", "Vp", "Rho"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    } else {
        std::string names[] = {"Rho","A","C","F","L","N"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    }
    return result;
}

template <typename T>
struct HybridEnergyResult {
    T I1, I2, I3;
    std::vector<T> I1_layer, I2_layer, I3_layer;
    T I1_hs, I2_hs, I3_hs;
};

template <typename T>
HybridEnergyResult<T> computeLoveEnergyIntegralsHybridAD(
    const std::vector<Eigen::Matrix<T, 2, 1>>& states,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T phaseVelocity,
    const std::vector<double>& sFactors
) {
    // Boundary states have the same upward T as calculateLoveDispersion's
    // intermediateStates. Integrating upward from a layer bottom gives
    // u(s)=C*u_bottom + S*T_bottom/eta.
    int N = static_cast<int>(layers.size());
    T k = omega / phaseVelocity;
    T k_sq = k * k;

    HybridEnergyResult<T> res;
    res.I1 = T(0.0); res.I2 = T(0.0); res.I3 = T(0.0);
    res.I1_layer.resize(N, T(0.0));
    res.I2_layer.resize(N, T(0.0));
    res.I3_layer.resize(N, T(0.0));

    std::vector<double> cum_sq(N + 1, 1.0);
    for (int j = 1; j <= N; ++j)
        cum_sq[j] = cum_sq[j - 1] * sFactors[j - 1] * sFactors[j - 1];

    for (int m = 0; m < N; ++m) {
        int j_bottom = N - 1 - m;

        T u_b = states[j_bottom](0);
        T tau_b = states[j_bottom](1);

        T h_m = thicknesses[m];
        T L = layers[m].c44;
        T N_m = layers[m].c66;
        T rho = layers[m].rho;

        T nu_sq = (N_m * k_sq - rho * omega * omega) / L;
        T nu = safe_complex_sqrt(nu_sq);
        T eta = L * nu;

        T twoa = T(2.0) * nu * h_m;
        T C2 = ad_cosh(twoa);
        T S2 = ad_sinh(twoa);

        T S24nu = S2 / (T(4.0) * nu);
        T QQ = (C2 - T(1.0)) / (T(4.0) * nu);

        T q = tau_b / eta;
        T int_uu = u_b * u_b * (h_m * T(0.5) + S24nu)
                 + T(2.0) * u_b * q * QQ
                 + q * q * (T(-0.5) * h_m + S24nu);

        T Lnu2 = L * nu * nu;
        T int_tt_over_L = Lnu2 * u_b * u_b * (T(-0.5) * h_m + S24nu)
                        + u_b * tau_b * (C2 - T(1.0)) * T(0.5)
                        + tau_b * tau_b / L * (h_m * T(0.5) + S24nu);

        T int_uu_r = int_uu.real();
        T int_tt_L_r = int_tt_over_L.real();

        T inv_csq = T(1.0 / cum_sq[j_bottom]);
        res.I1_layer[m] = rho * int_uu_r * inv_csq;
        res.I2_layer[m] = (int_tt_L_r + k_sq * N_m * int_uu_r) * inv_csq;
        res.I3_layer[m] = N_m * int_uu_r * inv_csq;

        res.I1 += res.I1_layer[m];
        res.I2 += res.I2_layer[m];
        res.I3 += res.I3_layer[m];
    }

    {
        T L_hs = halfSpace.c44;
        T N_hs = halfSpace.c66;
        T rho_hs = halfSpace.rho;

        T nu_sq_hs = (N_hs * k_sq - rho_hs * omega * omega) / L_hs;
        T nu_hs = safe_complex_sqrt(nu_sq_hs);
        if (get_val(nu_hs.real()) < 0) {
            nu_hs = T(0.0) - nu_hs;
        }

        T u_0 = states[0](0);
        T u_0_sq = u_0 * u_0;

        T nu_real = nu_hs.real();
        T nu_imag = nu_hs.imag();
        T nu_norm_sq = nu_real * nu_real + nu_imag * nu_imag;

        res.I1_hs = T(0.0); res.I2_hs = T(0.0); res.I3_hs = T(0.0);
        if (get_val(nu_real) > 1e-10) {
            T inv_decay = T(1.0) / (T(2.0) * nu_real);
            res.I1_hs = rho_hs * u_0_sq * inv_decay;
            res.I2_hs = (L_hs * nu_norm_sq + k_sq * N_hs) * u_0_sq * inv_decay;
            res.I3_hs = N_hs * u_0_sq * inv_decay;
        }
        res.I1 += res.I1_hs;
        res.I2 += res.I2_hs;
        res.I3 += res.I3_hs;
    }

    return res;
}

enum class LoveBranchPolicy;
template <typename T> struct LayerIntegralResult;
template <typename T>
LayerIntegralResult<T> computeLoveLayerIntegrals(
    T top_u,
    T top_tau,
    const PhysicalParameters<T>& layer,
    T thickness,
    T omega,
    T cp,
    LoveBranchPolicy branch_policy
);

struct VariationalLoveHessianBreakdown {
    Eigen::MatrixXd hessian;
    Eigen::MatrixXd hessian_fpp;
    Eigen::MatrixXd hessian_fpc;
    Eigen::MatrixXd hessian_fcc;
    Eigen::MatrixXd hessian_fixed_unsym;
    Eigen::MatrixXd hessian_chain_unsym;
    Eigen::VectorXd gradient;
    Eigen::VectorXd Fp;
    Eigen::VectorXd Fpc;
    Eigen::MatrixXd Fpp;
    Eigen::MatrixXd I1pp;
    Eigen::MatrixXd I2pp;
    std::vector<Eigen::MatrixXd> I1pp_layer;
    std::vector<Eigen::MatrixXd> I2pp_layer;
    std::vector<Eigen::MatrixXd> I1pp_layer_fixed_state;
    std::vector<Eigen::MatrixXd> I2pp_layer_fixed_state;
    Eigen::MatrixXd I1pp_hs;
    Eigen::MatrixXd I2pp_hs;
    std::vector<double> top_state_u;
    std::vector<double> top_state_tau;
    std::vector<Eigen::VectorXd> top_state_u_grad;
    std::vector<Eigen::VectorXd> top_state_tau_grad;
    std::vector<Eigen::MatrixXd> top_state_u_hess;
    std::vector<Eigen::MatrixXd> top_state_tau_hess;
    std::vector<double> bottom_state_u;
    std::vector<double> bottom_state_tau;
    std::vector<Eigen::VectorXd> bottom_state_u_grad;
    std::vector<Eigen::VectorXd> bottom_state_tau_grad;
    std::vector<Eigen::MatrixXd> bottom_state_u_hess;
    std::vector<Eigen::MatrixXd> bottom_state_tau_hess;
    double Fc = NAN;
    double Fcc = NAN;
    double F = NAN;
    double root_c_used = NAN;
    double I1 = NAN;
    double I2 = NAN;
    double I3 = NAN;
    std::vector<double> I1_layer, I2_layer, I3_layer;
    double I1_hs = NAN, I2_hs = NAN, I3_hs = NAN;
    std::vector<std::string> parameterNames;
    std::vector<int> layerIndices;
};

// ===========================================================================
// Weak-Form Hessian Building Blocks (Love Wave)
// ===========================================================================
// Templated on scalar type T to work with autodiff::dual for 1st-order AD.
// All functions use purely real arithmetic (no std::complex<T>) so they are
// compatible with any AD type that supports basic math operations.
// ===========================================================================

// Eigenfunction state at a layer boundary (real-only, AD-compatible)
template <typename T>
struct LoveLayerState {
    T u;    // displacement u_y
    T tau;  // c44*du/dz, z positive upward, in every propagation direction
};

enum class LoveBranchPolicy {
    Auto,
    SignOnly,
    ForceEvanescent,
    ForcePropagating,
    ForceLinear
};

enum class LoveBranchRegime {
    Evanescent,
    Propagating,
    Linear
};

inline const char* loveBranchPolicyName(LoveBranchPolicy policy) {
    switch (policy) {
        case LoveBranchPolicy::Auto: return "auto";
        case LoveBranchPolicy::SignOnly: return "sign_only";
        case LoveBranchPolicy::ForceEvanescent: return "force_evanescent";
        case LoveBranchPolicy::ForcePropagating: return "force_propagating";
        case LoveBranchPolicy::ForceLinear: return "force_linear";
    }
    return "auto";
}

template <typename T>
inline LoveBranchRegime selectLoveBranchRegime(const T& nu_sq, LoveBranchPolicy policy) {
    const double nu_sq_val = get_val(nu_sq);
    switch (policy) {
        case LoveBranchPolicy::Auto:
            if (nu_sq_val > 1e-30) return LoveBranchRegime::Evanescent;
            if (nu_sq_val < -1e-30) return LoveBranchRegime::Propagating;
            return LoveBranchRegime::Linear;
        case LoveBranchPolicy::SignOnly:
            return (nu_sq_val >= 0.0) ? LoveBranchRegime::Evanescent : LoveBranchRegime::Propagating;
        case LoveBranchPolicy::ForceEvanescent:
            return LoveBranchRegime::Evanescent;
        case LoveBranchPolicy::ForcePropagating:
            return LoveBranchRegime::Propagating;
        case LoveBranchPolicy::ForceLinear:
            return LoveBranchRegime::Linear;
    }
    return LoveBranchRegime::Linear;
}

template <typename T>
inline auto loveBranchSqrtAbs(const T& x) {
    using std::abs;
    using std::sqrt;
    return sqrt(abs(x));
}

// Propagate Love wave eigenfunctions from surface downward.
// Returns boundary states at surface, each layer bottom, and halfspace entry.
// Uses real-only 2×2 layer matrices: no std::complex<T>.
template <typename T>
std::vector<LoveLayerState<T>> propagateLoveEigenstates(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    T omega, T cp,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto
) {
    T k = omega / cp;
    std::vector<LoveLayerState<T>> states;
    // Surface: u=1, τ=0
    T u = T(1.0), tau = T(0.0);
    states.push_back({u, tau});

    for (size_t i = 0; i < layers.size(); ++i) {
        T L_j = layers[i].c44;
        T N_j = layers[i].c66;
        T rho_j = layers[i].rho;
        T h = thicknesses[i];

        // ν² = (N·k² − ρ·ω²) / L
        T nu_sq = (N_j * k * k - rho_j * omega * omega) / L_j;

        // Build 2×2 propagator and apply to [u, τ]
        // Sign convention: downward propagation with z = -h
        // Matrix: [[cosh(νh),  -sinh(νh)/(Lν)],
        //          [-Lν·sinh(νh),  cosh(νh)]]
        T u_new, tau_new;
        switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
        case LoveBranchRegime::Evanescent: {
            // Evanescent: ν real
            T nu;
            if (branch_policy == LoveBranchPolicy::ForcePropagating) {
                nu = loveBranchSqrtAbs(-nu_sq);
            } else {
                nu = loveBranchSqrtAbs(nu_sq);
            }
            T ch = cosh(nu * h);
            T sh = sinh(nu * h);
            T eta = L_j * nu;
            u_new   =  ch * u - sh / eta * tau;
            tau_new = -eta * sh * u + ch * tau;
            break;
        }
        case LoveBranchRegime::Propagating: {
            // Propagating: ν = iμ, use cos/sin
            T mu;
            if (branch_policy == LoveBranchPolicy::ForceEvanescent) {
                mu = loveBranchSqrtAbs(nu_sq);
            } else {
                mu = loveBranchSqrtAbs(-nu_sq);
            }
            T co = cos(mu * h);
            T si = sin(mu * h);
            T eta = L_j * mu;
            u_new   =  co * u - si / eta * tau;
            tau_new =  eta * si * u + co * tau;
            break;
        }
        case LoveBranchRegime::Linear: {
            // ν ≈ 0: linear approximation
            u_new   = u - h / L_j * tau;
            tau_new = tau;
            break;
        }
        }
        u = u_new;
        tau = tau_new;
        states.push_back({u, tau});
    }
    return states;
}

// Per-layer analytical integrals of u² and (du/dz)².
// Returns {int_u2, int_dudz2} for a single layer.
template <typename T>
struct LayerIntegralResult { T int_u2; T int_dudz2; };

template <typename T>
LayerIntegralResult<T> computeLoveLayerIntegrals(
    T u_top, T tau_top,
    const PhysicalParameters<T>& layer,
    T h, T omega, T k,
    LoveBranchPolicy branch_policy
) {
    T L_j = layer.c44;
    T N_j = layer.c66;
    T rho_j = layer.rho;
    T nu_sq = (N_j * k * k - rho_j * omega * omega) / L_j;

    using CT=typename ComplexType<T>::type;
    CT regular_u2, regular_du2;
    if(branch_policy==LoveBranchPolicy::Auto &&
       disper26_energy::love_moments(layer,h,omega,k,toComplex(u_top),toComplex(tau_top),regular_u2,regular_du2))
        return {regular_u2.real(),regular_du2.real()};

    T int_u2, int_dudz2;

    switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
    case LoveBranchRegime::Evanescent: {
        // Evanescent: ν real, u(z) = c_up·exp(νz) + c_dn·exp(-νz)
        T nu;
        if (branch_policy == LoveBranchPolicy::ForcePropagating) {
            nu = loveBranchSqrtAbs(-nu_sq);
        } else {
            nu = loveBranchSqrtAbs(nu_sq);
        }
        T Lnu = L_j * nu;
        // Local integration depth x increases downward from the layer top;
        // tau_top=L*du/dz=-L*du/dx. Keep this conversion at the analytical
        // integral boundary (the propagated state remains z-up).
        // c_up = (u - τ/(Lν))/2, c_dn = (u + τ/(Lν))/2
        T c_up = T(0.5) * (u_top - tau_top / Lnu);
        T c_dn = T(0.5) * (u_top + tau_top / Lnu);

        // ∫₀ʰ exp((νm+νn)z) dz  for ν_m, ν_n ∈ {+ν, -ν}
        // e_pp = (e^{2νh}-1)/(2ν) = h·φ₁(2νh), e_mm = (1-e^{-2νh})/(2ν) = h·φ₁(-2νh).
        // φ₁ = ad_phi1 (stable Taylor/closed-form switch on the dimensionless
        // argument, shared with the Rayleigh ν₃≈0 fix); c_up/c_dn above are
        // O(1/ν) as ν→0, so — same mechanism as the Rayleigh fix — a 1st-order
        // Taylor here loses accuracy under differentiation well before its
        // value-level error would suggest (docs/Weakform_AD_Hessian.md §4.6, §13).
        T two_nu_h = T(2.0) * nu * h;
        T e_pp = h * ad_phi1(toComplex<T>(two_nu_h)).real();
        T e_mm = h * ad_phi1(toComplex<T>(-two_nu_h)).real();
        T e_pm = h; // exp(0) integral = h

        int_u2 = c_up * c_up * e_pp + T(2.0) * c_up * c_dn * e_pm + c_dn * c_dn * e_mm;
        int_dudz2 = nu * nu * (c_up * c_up * e_pp - T(2.0) * c_up * c_dn * e_pm + c_dn * c_dn * e_mm);
        break;
    }
    case LoveBranchRegime::Propagating: {
        // Propagating: ν = iμ, u(z) = A·cos(μz) + B·sin(μz)
        T mu;
        if (branch_policy == LoveBranchPolicy::ForceEvanescent) {
            mu = loveBranchSqrtAbs(nu_sq);
        } else {
            mu = loveBranchSqrtAbs(-nu_sq);
        }
        T Lmu = L_j * mu;
        T A = u_top;
        T B = -tau_top / Lmu; // u(x)=A*cos(mu*x)+B*sin(mu*x), x downward

        // ∫₀ʰ u² dz = ∫₀ʰ (A cos(μz) + B sin(μz))² dz
        //            = (A²+B²)h/2 + (A²-B²)sin(2μh)/(4μ) + AB(1-cos(2μh))/(2μ)
        T A2 = A * A, B2 = B * B, AB = A * B;
        T mu2h = T(2.0) * mu * h;
        T s2 = sin(mu2h), c2 = cos(mu2h);
        T inv_4mu = T(1.0) / (T(4.0) * mu);
        T inv_2mu = T(1.0) / (T(2.0) * mu);

        int_u2 = (A2 + B2) * h * T(0.5) + (A2 - B2) * s2 * inv_4mu + AB * (T(1.0) - c2) * inv_2mu;

        // ∫₀ʰ (du/dz)² dz, du/dz = -Aμ sin(μz) + Bμ cos(μz)
        // = μ²[(A²+B²)h/2 - (A²-B²)sin(2μh)/(4μ) - AB(1-cos(2μh))/(2μ)]
        int_dudz2 = mu * mu * ((A2 + B2) * h * T(0.5) - (A2 - B2) * s2 * inv_4mu - AB * (T(1.0) - c2) * inv_2mu);
        break;
    }
    case LoveBranchRegime::Linear: {
        // ν ≈ 0: u(x) ≈ u_top - (tau_top/L)*x, x positive downward
        T slope = -tau_top / L_j;
        int_u2 = u_top * u_top * h + u_top * slope * h * h + slope * slope * h * h * h / T(3.0);
        int_dudz2 = slope * slope * h;
        break;
    }
    }

    return {int_u2, int_dudz2};
}

// Energy integrals I₁, I₂, I₃ computed analytically (all-real, AD-compatible).
template <typename T>
struct LoveEnergyTerms {
    T I1, I2, I3;
    std::vector<LayerIntegralResult<T>> layer_integrals;
    T int_u2_hs, int_dudz2_hs; // halfspace integrals
};

template <typename T>
LoveEnergyTerms<T> computeLoveAnalyticalIntegrals(
    const std::vector<LoveLayerState<T>>& states,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto
) {
    T k = omega / cp;
    T k2 = k * k;
    LoveEnergyTerms<T> result;
    result.I1 = T(0.0);
    result.I2 = T(0.0);
    result.I3 = T(0.0);

    // Per-layer integrals
    for (size_t j = 0; j < layers.size(); ++j) {
        auto li = computeLoveLayerIntegrals<T>(
            states[j].u, states[j].tau, layers[j], thicknesses[j], omega, k, branch_policy);
        result.layer_integrals.push_back(li);

        result.I1 = result.I1 + layers[j].rho * li.int_u2;
        result.I2 = result.I2 + (li.int_dudz2 * layers[j].c44 + k2 * layers[j].c66 * li.int_u2);
        result.I3 = result.I3 + layers[j].c66 * li.int_u2;
    }

    // Halfspace (exponential decay)
    T u0 = states.back().u;
    T L_h = halfSpace.c44, N_h = halfSpace.c66, rho_h = halfSpace.rho;
    T nu_sq_hs = (N_h * k2 - rho_h * omega * omega) / L_h;
    result.int_u2_hs = T(0.0);
    result.int_dudz2_hs = T(0.0);

    bool include_halfspace =
        (branch_policy == LoveBranchPolicy::ForceLinear) ? false :
        (branch_policy == LoveBranchPolicy::ForceEvanescent) ? true :
        (branch_policy == LoveBranchPolicy::ForcePropagating) ? false :
        (branch_policy == LoveBranchPolicy::SignOnly) ? (get_val(nu_sq_hs) >= 0.0) :
        (get_val(nu_sq_hs) > 1e-10);
    if (include_halfspace) {
        T nu_hs;
        if (branch_policy == LoveBranchPolicy::ForcePropagating) {
            nu_hs = loveBranchSqrtAbs(-nu_sq_hs);
        } else {
            nu_hs = loveBranchSqrtAbs(nu_sq_hs);
        }
        T inv2nu = T(1.0) / (T(2.0) * nu_hs);
        result.int_u2_hs = u0 * u0 * inv2nu;
        result.int_dudz2_hs = nu_sq_hs * u0 * u0 * inv2nu;
        result.I1 = result.I1 + rho_h * result.int_u2_hs;
        result.I2 = result.I2 + (result.int_dudz2_hs * L_h + k2 * N_h * result.int_u2_hs);
        result.I3 = result.I3 + N_h * result.int_u2_hs;
    }

    return result;
}

template <typename T>
struct LoveScaledBoundaryData {
    std::vector<LoveLayerState<T>> states;
    std::vector<LoveLayerState<T>> bottom_states;
    // Energy weights are normalized to a common finite gauge.  Reconstructing
    // the original halfspace-amplitude gauge can require 1e300--1e308 and
    // causes a coefficient/raw-gradient cancellation to overflow in AD.
    std::vector<double> relative_scale_sq;
    std::vector<double> bottom_relative_scale_sq;
    std::vector<double> scale_factors;
};

inline double love_relative_scale_sq(double log_reference, double log_cumulative) {
    // log_reference is chosen as the smallest cumulative log-scale, so the
    // exponent is non-positive.  Underflow here only removes contributions
    // below double precision relative to the retained gauge; overflow is
    // impossible by construction.
    const double exponent = 2.0 * (log_reference - log_cumulative);
    if (exponent <= std::log(std::numeric_limits<double>::min())) return 0.0;
    return std::exp(exponent);
}

template <typename T>
LayerIntegralResult<T> computeLoveLayerIntegralsFromBottomStateUpward(
    T bottom_u,
    T bottom_tau,
    const PhysicalParameters<T>& layer,
    T thickness,
    T omega,
    T k,
    LoveBranchPolicy branch_policy
) {
    // bottom_tau is the physical upward stress. The top-state integral below
    // advances downward, so reflect traction when reusing it from the bottom.
    T L_j = layer.c44;
    T N_j = layer.c66;
    T rho_j = layer.rho;
    T nu_sq = (N_j * k * k - rho_j * omega * omega) / L_j;

    switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
    case LoveBranchRegime::Evanescent: {
        T nu = loveBranchSqrtAbs(nu_sq);
        T eta = L_j * nu;
        // S24nu=sinh(x)/(4ν), QQ=(cosh(x)-1)/(4ν), C2m1=cosh(x)-1, x=2νh.
        // Both sinh(x) and cosh(x)-1 are expressible via φ₁(x)=(e^x-1)/x:
        //   sinh(x)   = x·[φ₁(x)+φ₁(-x)]/2   cosh(x)-1 = x·[φ₁(x)-φ₁(-x)]/2
        // so S24nu, QQ, C2m1 reduce to sums/differences of φ₁(±x) with no
        // separate cancellation-prone Taylor branch needed (same φ₁ used by
        // the Rayleigh ν₃≈0 fix and the Evanescent e_pp/e_mm site above).
        T two_nu_h = T(2.0) * nu * thickness;
        T S24nu, QQ, C2m1;
        {
            T p_pos = ad_phi1(toComplex<T>(two_nu_h)).real();
            T p_neg = ad_phi1(toComplex<T>(-two_nu_h)).real();
            S24nu = (thickness * T(0.25)) * (p_pos + p_neg);
            QQ = (thickness * T(0.25)) * (p_pos - p_neg);
            C2m1 = (nu * thickness) * (p_pos - p_neg);
        }

        T q = bottom_tau / eta;
        T int_u2 = bottom_u * bottom_u * (thickness * T(0.5) + S24nu)
                 + T(2.0) * bottom_u * q * QQ
                 + q * q * (T(-0.5) * thickness + S24nu);

        T nu2 = nu * nu;
        T int_dudz2 = nu2 * bottom_u * bottom_u * (T(-0.5) * thickness + S24nu)
                    + bottom_u * bottom_tau * C2m1 / (T(2.0) * L_j)
                    + bottom_tau * bottom_tau / (L_j * L_j) * (thickness * T(0.5) + S24nu);
        return {int_u2, int_dudz2};
    }
    case LoveBranchRegime::Propagating:
    case LoveBranchRegime::Linear:
        return computeLoveLayerIntegrals<T>(
            bottom_u, T(-bottom_tau), layer, thickness, omega, k, branch_policy);
    }

    return computeLoveLayerIntegrals<T>(
        bottom_u, T(-bottom_tau), layer, thickness, omega, k, branch_policy);
}

template <typename T>
LoveScaledBoundaryData<T> buildLoveScaledBoundaryDataUpward(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    const std::vector<double>* provided_scale_factors = nullptr
) {
    T k = omega / cp;
    int nLayers = static_cast<int>(layers.size());
    std::vector<double> scale_factors;
    scale_factors.reserve(nLayers);

    // All live and stored states use T=L*du/dz (z upward), with initial
    // [1,+L_h*nu_hs] and upward transfer A(+h). No storage-time reflection.
    // On the ordinary path, normalize after every upward layer transfer.
    // Forming an unnormalized cosh/sinh transfer first can overflow for a
    // trial model even when the physical, gauge-invariant energy is finite.
    std::vector<LoveLayerState<T>> normalized_states(nLayers + 1);
    std::vector<LoveLayerState<T>> normalized_bottom_states(nLayers);
    std::vector<double> normalized_log_top(nLayers + 1, 0.0);
    std::vector<double> normalized_log_bottom(nLayers, 0.0);

    auto normalized_upward_transfer = [&](const LoveLayerState<T>& state,
                                          const PhysicalParameters<T>& layer,
                                          T h) {
        T L_j = layer.c44;
        T N_j = layer.c66;
        T rho_j = layer.rho;
        T nu_sq = (N_j * k * k - rho_j * omega * omega) / L_j;
        T u_next, tau_next;
        double log_scale_extra = 0.0;

        switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
        case LoveBranchRegime::Evanescent: {
            T nu = loveBranchSqrtAbs(nu_sq);
            T eta = L_j * nu;
            T x = nu * h;
            // DISPER80 LOVMRX scaling: divide the whole evanescent transfer
            // by cosh(x).  The omitted factor is common to both state
            // components, so it cannot change the dispersion root.  Unlike
            // an explicit growing/decaying modal reconstruction, tanh(x)
            // remains finite for arbitrarily large positive x.
            T th;
            if constexpr (std::is_same_v<T, double>) {
                th = T(std::tanh(get_val(x)));
            } else {
                // This helper is selected only by the double first-order
                // fallback. Keep the expression available for templated
                // instantiation without changing the legacy AD Hessian path.
                th = sinh(x) / cosh(x);
            }
            u_next = state.u + th / eta * state.tau;
            tau_next = eta * th * state.u + state.tau;
            const double xv = get_val(x);
            const double log_cosh = xv < 20.0
                ? std::log(std::cosh(xv))
                : xv - std::log(2.0) + std::log1p(std::exp(-2.0 * xv));
            log_scale_extra = -log_cosh;
            break;
        }
        case LoveBranchRegime::Propagating: {
            T mu = loveBranchSqrtAbs(-nu_sq);
            T x = mu * h;
            T co = cos(x);
            T si = sin(x);
            T eta = L_j * mu;
            u_next = co * state.u + si / eta * state.tau;
            tau_next = -eta * si * state.u + co * state.tau;
            break;
        }
        case LoveBranchRegime::Linear:
            u_next = state.u + h / L_j * state.tau;
            tau_next = state.tau;
            break;
        }

        const double magnitude =
            std::abs(get_val(u_next)) + std::abs(get_val(tau_next));
        if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
            throw std::runtime_error(
                "Love normalized upward eigenfunction became non-finite"
                " (u=" + std::to_string(get_val(state.u)) +
                ", tau=" + std::to_string(get_val(state.tau)) +
                ", u_next=" + std::to_string(get_val(u_next)) +
                ", tau_next=" + std::to_string(get_val(tau_next)) +
                ", h=" + std::to_string(get_val(h)) +
                ", cp=" + std::to_string(get_val(cp)) + ")");
        }
        return std::tuple<LoveLayerState<T>, double>(
            {u_next / T(magnitude), tau_next / T(magnitude)},
            log_scale_extra - std::log(magnitude));
    };

    T L_h = halfSpace.c44;
    T N_h = halfSpace.c66;
    T rho_h = halfSpace.rho;
    T nu_sq_hs = (N_h * k * k - rho_h * omega * omega) / L_h;
    T nu_hs = loveBranchSqrtAbs(nu_sq_hs);
    bool used_stable_path = false;

    if (provided_scale_factors) {
        scale_factors = *provided_scale_factors;
        if (scale_factors.size() != static_cast<size_t>(nLayers)) {
            throw std::runtime_error("Love analytical scale-factor count does not match layer count");
        }
        for (size_t i = 0; i < scale_factors.size(); ++i) {
            if (!(scale_factors[i] > 0.0) || !std::isfinite(scale_factors[i])) {
                throw std::runtime_error("Love analytical scale factor is non-finite or non-positive");
            }
        }
    } else if constexpr (std::is_same_v<T, double>) {
        // Preserve the legacy gauge whenever it is finite.  This keeps the
        // normal kernel values and LM behavior unchanged.  Only switch to
        // the modal/logarithmic gauge when the legacy transfer actually
        // overflows or its cumulative scale leaves double precision.
        LoveLayerState<T> state{T(1.0), L_h * nu_hs};
        double legacy_log_cumulative = 0.0;
        bool use_stable_path = false;
        for (int i = nLayers - 1; i >= 0; --i) {
            T L_j = layers[i].c44;
            T N_j = layers[i].c66;
            T rho_j = layers[i].rho;
            T h = thicknesses[i];
            T nu_sq = (N_j * k * k - rho_j * omega * omega) / L_j;
            T u_b = state.u;
            T tau_b = state.tau;
            T u_t, tau_t;
            switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
            case LoveBranchRegime::Evanescent: {
                T nu = loveBranchSqrtAbs(nu_sq);
                T ch = cosh(nu * h);
                T sh = sinh(nu * h);
                T eta = L_j * nu;
                u_t = ch * u_b + sh / eta * tau_b;
                tau_t = eta * sh * u_b + ch * tau_b;
                break;
            }
            case LoveBranchRegime::Propagating: {
                T mu = loveBranchSqrtAbs(-nu_sq);
                T co = cos(mu * h);
                T si = sin(mu * h);
                T eta = L_j * mu;
                u_t = co * u_b + si / eta * tau_b;
                tau_t = -eta * si * u_b + co * tau_b;
                break;
            }
            case LoveBranchRegime::Linear:
                u_t = u_b + h / L_j * tau_b;
                tau_t = tau_b;
                break;
            }
            const double m = std::abs(u_t) + std::abs(tau_t);
            const double scale = m > 1e-20 ? 1.0 / m : 1.0;
            legacy_log_cumulative += std::log(scale);
            if (!std::isfinite(u_t) || !std::isfinite(tau_t) ||
                !(scale > 0.0) || !std::isfinite(scale) ||
                !std::isfinite(legacy_log_cumulative)) {
                use_stable_path = true;
                break;
            }
            scale_factors.push_back(scale);
            state = {u_t * scale, tau_t * scale};
        }

        if (use_stable_path) {
            used_stable_path = true;
            scale_factors.clear();
            state = {T(1.0), L_h * nu_hs};
            normalized_states[nLayers] = {state.u, state.tau};
            double log_cumulative = 0.0;
            for (int i = nLayers - 1; i >= 0; --i) {
                normalized_bottom_states[i] = {state.u, state.tau};
                normalized_log_bottom[i] = log_cumulative;
                auto transfer = normalized_upward_transfer(
                    state, layers[i], thicknesses[i]);
                state = std::get<0>(transfer);
                log_cumulative += std::get<1>(transfer);
                if (!std::isfinite(log_cumulative)) {
                    throw std::runtime_error(
                        "Love cumulative normalized log-scale became non-finite");
                }
                normalized_states[i] = {state.u, state.tau};
                normalized_log_top[i] = log_cumulative;
            }
        }
    } else {
        // Preserve the original AD gauge for Hessian calculations.  Its
        // scale factors are value-only constants, and changing this gauge
        // changes the AD curvature even though the double-valued energy is
        // gauge invariant.  The overflow-prone double path above is used by
        // the grid phase-velocity kernel; AD Hessian callers retain this
        // established path and its regression tolerances.
        LoveLayerState<T> state{T(1.0), L_h * nu_hs};
        for (int i = nLayers - 1; i >= 0; --i) {
            T L_j = layers[i].c44;
            T N_j = layers[i].c66;
            T rho_j = layers[i].rho;
            T h = thicknesses[i];
            T nu_sq = (N_j * k * k - rho_j * omega * omega) / L_j;
            T u_b = state.u;
            T tau_b = state.tau;
            T u_t, tau_t;

            switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
            case LoveBranchRegime::Evanescent: {
                T nu = loveBranchSqrtAbs(nu_sq);
                T ch = cosh(nu * h);
                T sh = sinh(nu * h);
                T eta = L_j * nu;
                u_t = ch * u_b + sh / eta * tau_b;
                tau_t = eta * sh * u_b + ch * tau_b;
                break;
            }
            case LoveBranchRegime::Propagating: {
                T mu = loveBranchSqrtAbs(-nu_sq);
                T co = cos(mu * h);
                T si = sin(mu * h);
                T eta = L_j * mu;
                u_t = co * u_b + si / eta * tau_b;
                tau_t = -eta * si * u_b + co * tau_b;
                break;
            }
            case LoveBranchRegime::Linear:
                u_t = u_b + h / L_j * tau_b;
                tau_t = tau_b;
                break;
            }
            state = {u_t, tau_t};
            double m = get_magnitude(state.u) + get_magnitude(state.tau);
            if (!std::isfinite(m)) {
                throw std::runtime_error("Love upward eigenfunction scale became non-finite");
            }
            double scale = m > 1e-20 ? 1.0 / m : 1.0;
            if (!(scale > 0.0) || !std::isfinite(scale)) {
                throw std::runtime_error("Love upward eigenfunction scale is non-finite or non-positive");
            }
            scale_factors.push_back(scale);
            state.u = state.u * T(scale);
            state.tau = state.tau * T(scale);
        }
    }

    LoveScaledBoundaryData<T> result;
    result.states.resize(nLayers + 1);
    result.bottom_states.resize(nLayers);
    result.relative_scale_sq.resize(nLayers + 1, 1.0);
    result.bottom_relative_scale_sq.resize(nLayers, 1.0);
    result.scale_factors = scale_factors;

    if constexpr (std::is_same_v<T, double>) {
      if (!provided_scale_factors && used_stable_path) {
        result.states = normalized_states;
        result.bottom_states = normalized_bottom_states;
        double log_reference = 0.0;
        for (double value : normalized_log_top) log_reference = std::min(log_reference, value);
        for (double value : normalized_log_bottom) log_reference = std::min(log_reference, value);
        result.relative_scale_sq.back() = love_relative_scale_sq(log_reference, 0.0);
        for (int i = 0; i < nLayers; ++i) {
            result.bottom_relative_scale_sq[i] = love_relative_scale_sq(
                log_reference, normalized_log_bottom[i]);
            result.relative_scale_sq[i] = love_relative_scale_sq(
                log_reference, normalized_log_top[i]);
        }
        return result;
      }
    }

    LoveLayerState<T> scaled_state{T(1.0), L_h * nu_hs};
    result.states[nLayers] = {scaled_state.u, scaled_state.tau};

    // Store cumulative scales in log form.  The old implementation formed
    // 1/(cumulative_scale*cumulative_scale), which is the source of the
    // shallow-Toba overflow.  We later apply one common relative gauge to all
    // layer and halfspace energy terms instead.
    std::vector<double> log_cumulative_top(nLayers + 1, 0.0);
    std::vector<double> log_cumulative_bottom(nLayers, 0.0);
    double log_cumulative = 0.0;
    for (int i = nLayers - 1; i >= 0; --i) {
        T L_j = layers[i].c44;
        T N_j = layers[i].c66;
        T h = thicknesses[i];
        T nu_sq = (N_j * k * k - layers[i].rho * omega * omega) / L_j;

        T u_b = scaled_state.u;
        T tau_b = scaled_state.tau;
        result.bottom_states[i] = {u_b, tau_b};
        log_cumulative_bottom[i] = log_cumulative;
        T u_t, tau_t;

        switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
        case LoveBranchRegime::Evanescent: {
            T nu = loveBranchSqrtAbs(nu_sq);
            T ch = cosh(nu * h);
            T sh = sinh(nu * h);
            T eta = L_j * nu;
            u_t   =  ch * u_b + sh / eta * tau_b;
            tau_t = eta * sh * u_b + ch * tau_b;
            break;
        }
        case LoveBranchRegime::Propagating: {
            T mu = loveBranchSqrtAbs(-nu_sq);
            T co = cos(mu * h);
            T si = sin(mu * h);
            T eta = L_j * mu;
            u_t   = co * u_b + si / eta * tau_b;
            tau_t = -eta * si * u_b + co * tau_b;
            break;
        }
        case LoveBranchRegime::Linear:
            u_t = u_b + h / L_j * tau_b;
            tau_t = tau_b;
            break;
        }

        double scale = scale_factors[nLayers - 1 - i];
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::runtime_error("Love analytical scale factor is non-finite or non-positive");
        }
        log_cumulative += std::log(scale);
        if (!std::isfinite(log_cumulative)) {
            throw std::runtime_error("Love cumulative eigenfunction log-scale became non-finite");
        }
        scaled_state.u = u_t * T(scale);
        scaled_state.tau = tau_t * T(scale);

        result.states[i] = {scaled_state.u, scaled_state.tau};
        log_cumulative_top[i] = log_cumulative;
    }

    double log_reference = 0.0;  // halfspace boundary cumulative log-scale
    for (double value : log_cumulative_top) log_reference = std::min(log_reference, value);
    for (double value : log_cumulative_bottom) log_reference = std::min(log_reference, value);
    result.relative_scale_sq.back() = love_relative_scale_sq(log_reference, 0.0);
    for (int i = 0; i < nLayers; ++i) {
        result.bottom_relative_scale_sq[i] = love_relative_scale_sq(
            log_reference, log_cumulative_bottom[i]);
        result.relative_scale_sq[i] = love_relative_scale_sq(
            log_reference, log_cumulative_top[i]);
    }

    return result;
}

// Stable analytical Love integrals from halfspace-upward states.  The returned
// terms keep the legacy LoveEnergyTerms layout so weakform Hessian assembly can
// remain unchanged.
template <typename T>
struct LoveIntegralContribution {
    int layer_index;
    LayerIntegralResult<T> integral;
    double log_gauge;
};

// DISPER80-style first-order path. Each evanescent layer is divided until the
// local exponential argument is bounded, the state is normalized after every
// substep, and quadratic contributions retain the gauge in which they were
// evaluated. At the end all terms are moved to one common gauge using only
// non-positive exponentials. No large absolute eigenfunction is reconstructed.
template <typename T>
LoveEnergyTerms<T> computeLoveAnalyticalIntegralsDisper80(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    double max_evanescent_argument = 40.0,
    int max_substeps = 4096
) {
    if (layers.size() != thicknesses.size()) {
        throw std::runtime_error("Love DISPER80 scaling requires one thickness per layer");
    }
    T k = omega / cp;
    T k2 = k * k;
    T L_h = halfSpace.c44;
    T N_h = halfSpace.c66;
    T rho_h = halfSpace.rho;
    T nu_sq_hs = (N_h * k2 - rho_h * omega * omega) / L_h;
    if (!(get_val(nu_sq_hs) > 0.0)) {
        throw std::runtime_error(
            "Love DISPER80 scaling requires an evanescent halfspace");
    }
    T nu_hs = loveBranchSqrtAbs(nu_sq_hs);
    LoveLayerState<T> state{T(1.0), L_h * nu_hs};
    double log_gauge = 0.0;

    T inv2nu = T(1.0) / (T(2.0) * nu_hs);
    LayerIntegralResult<T> hs_integral{
        state.u * state.u * inv2nu,
        nu_sq_hs * state.u * state.u * inv2nu
    };
    std::vector<LoveIntegralContribution<T>> contributions;
    contributions.reserve(layers.size());

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        const auto& layer = layers[static_cast<size_t>(i)];
        T h = thicknesses[static_cast<size_t>(i)];
        T nu_sq = (layer.c66 * k2 - layer.rho * omega * omega) / layer.c44;
        LoveBranchRegime regime = selectLoveBranchRegime(nu_sq, branch_policy);
        int nsub = 1;
        if (regime == LoveBranchRegime::Evanescent) {
            T nu = loveBranchSqrtAbs(nu_sq);
            T total_argument = nu * h;
            double argument = std::abs(get_val(total_argument));
            if (!std::isfinite(argument)) {
                throw std::runtime_error(
                    "Love DISPER80 scaling received a non-finite evanescent argument");
            }
            nsub = std::max(1, static_cast<int>(std::ceil(
                argument / max_evanescent_argument)));
            if (nsub > max_substeps) {
                throw std::runtime_error(
                    "Love DISPER80 scaling exceeded the substep limit at layer " +
                    std::to_string(i));
            }
        }
        T hsub = h / T(nsub);
        for (int sub = 0; sub < nsub; ++sub) {
            // Preserve the historical DISPER80 fallback moment: its
            // evanescent branch used reflected bottom traction, unlike the
            // ordinary upward energy path. Correcting that pre-existing
            // energy discrepancy is separate from changing the SH coordinate.
            const T moment_tau = regime == LoveBranchRegime::Evanescent
                ? T(-state.tau) : state.tau;
            auto li = computeLoveLayerIntegralsFromBottomStateUpward<T>(
                state.u, moment_tau, layer, hsub, omega, k, branch_policy);
            contributions.push_back({i, li, log_gauge});

            T u_next, tau_next;
            switch (regime) {
            case LoveBranchRegime::Evanescent: {
                T nu = loveBranchSqrtAbs(nu_sq);
                T x = nu * hsub;
                T ch = cosh(x);
                T sh = sinh(x);
                T eta = layer.c44 * nu;
                u_next = ch * state.u + sh / eta * state.tau;
                tau_next = eta * sh * state.u + ch * state.tau;
                break;
            }
            case LoveBranchRegime::Propagating: {
                T mu = loveBranchSqrtAbs(-nu_sq);
                T x = mu * hsub;
                T co = cos(x);
                T si = sin(x);
                T eta = layer.c44 * mu;
                u_next = co * state.u + si / eta * state.tau;
                tau_next = -eta * si * state.u + co * state.tau;
                break;
            }
            case LoveBranchRegime::Linear:
                u_next = state.u + hsub / layer.c44 * state.tau;
                tau_next = state.tau;
                break;
            }
            double magnitude = std::max(
                std::abs(get_val(u_next)), std::abs(get_val(tau_next)));
            if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
                throw std::runtime_error(
                    "Love DISPER80 state became non-finite at layer " +
                    std::to_string(i) + ", substep " + std::to_string(sub));
            }
            double scale = 1.0 / magnitude;
            state = {u_next * T(scale), tau_next * T(scale)};
            log_gauge += std::log(scale);
            if (!std::isfinite(log_gauge)) {
                throw std::runtime_error(
                    "Love DISPER80 cumulative gauge became non-finite at layer " +
                    std::to_string(i));
            }
        }
    }

    double log_reference = 0.0;
    for (const auto& contribution : contributions) {
        log_reference = std::min(log_reference, contribution.log_gauge);
    }
    auto relative_weight = [&](double source_log_gauge) {
        return love_relative_scale_sq(log_reference, source_log_gauge);
    };

    LoveEnergyTerms<T> result;
    result.I1 = T(0.0);
    result.I2 = T(0.0);
    result.I3 = T(0.0);
    result.layer_integrals.assign(
        layers.size(), LayerIntegralResult<T>{T(0.0), T(0.0)});
    for (const auto& contribution : contributions) {
        T weight = T(relative_weight(contribution.log_gauge));
        auto& target = result.layer_integrals[static_cast<size_t>(contribution.layer_index)];
        target.int_u2 = target.int_u2 + weight * contribution.integral.int_u2;
        target.int_dudz2 = target.int_dudz2 + weight * contribution.integral.int_dudz2;
    }
    T hs_weight = T(relative_weight(0.0));
    result.int_u2_hs = hs_weight * hs_integral.int_u2;
    result.int_dudz2_hs = hs_weight * hs_integral.int_dudz2;

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& li = result.layer_integrals[i];
        result.I1 = result.I1 + layers[i].rho * li.int_u2;
        result.I2 = result.I2 + layers[i].c44 * li.int_dudz2
            + k2 * layers[i].c66 * li.int_u2;
        result.I3 = result.I3 + layers[i].c66 * li.int_u2;
    }
    result.I1 = result.I1 + rho_h * result.int_u2_hs;
    result.I2 = result.I2 + L_h * result.int_dudz2_hs
        + k2 * N_h * result.int_u2_hs;
    result.I3 = result.I3 + N_h * result.int_u2_hs;
    return result;
}

// O(N) synchronized-rescaling energy path for dU/dp. Unlike the per-layer
// variational-gradient assembly, group velocity needs only total I1/I2/I3.
// Scaling the live totals together with the state makes the omitted derivative
// of the arbitrary normalization a single common factor that cancels in U.
template <typename T>
LoveEnergyTerms<T> computeLoveGroupEnergyDisper80(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    double max_evanescent_argument = 40.0,
    int max_substeps = 4096
) {
    T k = omega / cp;
    T k2 = k * k;
    T L_h = halfSpace.c44;
    T N_h = halfSpace.c66;
    T rho_h = halfSpace.rho;
    T nu_sq_hs = (N_h * k2 - rho_h * omega * omega) / L_h;
    if (!(get_val(nu_sq_hs) > 0.0)) {
        throw std::runtime_error(
            "Love DISPER80 group energy requires an evanescent halfspace");
    }
    T nu_hs = loveBranchSqrtAbs(nu_sq_hs);
    LoveLayerState<T> state{T(1.0), L_h * nu_hs};
    T inv2nu = T(1.0) / (T(2.0) * nu_hs);

    LoveEnergyTerms<T> result;
    result.int_u2_hs = state.u * state.u * inv2nu;
    result.int_dudz2_hs = nu_sq_hs * state.u * state.u * inv2nu;
    result.I1 = rho_h * result.int_u2_hs;
    result.I2 = L_h * result.int_dudz2_hs + k2 * N_h * result.int_u2_hs;
    result.I3 = N_h * result.int_u2_hs;
    result.layer_integrals.clear();

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        const auto& layer = layers[static_cast<size_t>(i)];
        T h = thicknesses[static_cast<size_t>(i)];
        T nu_sq = (layer.c66 * k2 - layer.rho * omega * omega) / layer.c44;
        LoveBranchRegime regime = selectLoveBranchRegime(nu_sq, branch_policy);
        int nsub = 1;
        if (regime == LoveBranchRegime::Evanescent) {
            T nu = loveBranchSqrtAbs(nu_sq);
            T total_argument = nu * h;
            double argument = std::abs(get_val(total_argument));
            if (!std::isfinite(argument)) {
                throw std::runtime_error(
                    "Love DISPER80 group energy received a non-finite argument");
            }
            nsub = std::max(1, static_cast<int>(std::ceil(
                argument / max_evanescent_argument)));
            if (nsub > max_substeps) {
                throw std::runtime_error(
                    "Love DISPER80 group energy exceeded the substep limit at layer " +
                    std::to_string(i));
            }
        }
        T hsub = h / T(nsub);
        for (int sub = 0; sub < nsub; ++sub) {
            // Same legacy fallback moment contract as the per-layer DISPER80
            // path above; propagation itself always carries upward stress.
            const T moment_tau = regime == LoveBranchRegime::Evanescent
                ? T(-state.tau) : state.tau;
            auto li = computeLoveLayerIntegralsFromBottomStateUpward<T>(
                state.u, moment_tau, layer, hsub, omega, k, branch_policy);
            result.I1 = result.I1 + layer.rho * li.int_u2;
            result.I2 = result.I2 + layer.c44 * li.int_dudz2
                + k2 * layer.c66 * li.int_u2;
            result.I3 = result.I3 + layer.c66 * li.int_u2;

            T u_next, tau_next;
            switch (regime) {
            case LoveBranchRegime::Evanescent: {
                T nu = loveBranchSqrtAbs(nu_sq);
                T x = nu * hsub;
                T ch = cosh(x), sh = sinh(x), eta = layer.c44 * nu;
                u_next = ch * state.u + sh / eta * state.tau;
                tau_next = eta * sh * state.u + ch * state.tau;
                break;
            }
            case LoveBranchRegime::Propagating: {
                T mu = loveBranchSqrtAbs(-nu_sq);
                T x = mu * hsub;
                T co = cos(x), si = sin(x), eta = layer.c44 * mu;
                u_next = co * state.u + si / eta * state.tau;
                tau_next = -eta * si * state.u + co * state.tau;
                break;
            }
            case LoveBranchRegime::Linear:
                u_next = state.u + hsub / layer.c44 * state.tau;
                tau_next = state.tau;
                break;
            }
            double magnitude = std::max(
                std::abs(get_val(u_next)), std::abs(get_val(tau_next)));
            if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
                throw std::runtime_error(
                    "Love DISPER80 group state became non-finite at layer " +
                    std::to_string(i));
            }
            double scale = magnitude > 1e10 ? 1.0 / magnitude : 1.0;
            state = {u_next * T(scale), tau_next * T(scale)};
            if (scale != 1.0) {
                T scale_sq = T(scale * scale);
                result.I1 = result.I1 * scale_sq;
                result.I2 = result.I2 * scale_sq;
                result.I3 = result.I3 * scale_sq;
                result.int_u2_hs = result.int_u2_hs * scale_sq;
                result.int_dudz2_hs = result.int_dudz2_hs * scale_sq;
            }
        }
    }
    return result;
}

template <typename T>
LoveEnergyTerms<T> computeLoveAnalyticalIntegralsUpward(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    const std::vector<double>* provided_scale_factors = nullptr
) {
    auto scaled = buildLoveScaledBoundaryDataUpward<T>(
        layers, thicknesses, halfSpace, omega, cp, branch_policy, provided_scale_factors);

    T k = omega / cp;
    T k2 = k * k;
    LoveEnergyTerms<T> result;
    result.I1 = T(0.0);
    result.I2 = T(0.0);
    result.I3 = T(0.0);

    for (size_t j = 0; j < layers.size(); ++j) {
        LayerIntegralResult<T> li_scaled;
        T nu_sq = (layers[j].c66 * k2 - layers[j].rho * omega * omega) / layers[j].c44;
        LoveBranchRegime regime = selectLoveBranchRegime(nu_sq, branch_policy);
        T relative_scale_sq;
        if (regime == LoveBranchRegime::Evanescent) {
            li_scaled = computeLoveLayerIntegralsFromBottomStateUpward<T>(
                scaled.bottom_states[j].u, scaled.bottom_states[j].tau,
                layers[j], thicknesses[j], omega, k, branch_policy);
            relative_scale_sq = T(scaled.bottom_relative_scale_sq[j]);
        } else {
            li_scaled = computeLoveLayerIntegrals<T>(
                scaled.states[j].u, scaled.states[j].tau,
                layers[j], thicknesses[j], omega, k, branch_policy);
            relative_scale_sq = T(scaled.relative_scale_sq[j]);
        }
        LayerIntegralResult<T> li{
            li_scaled.int_u2 * relative_scale_sq,
            li_scaled.int_dudz2 * relative_scale_sq
        };
        result.layer_integrals.push_back(li);

        result.I1 = result.I1 + layers[j].rho * li.int_u2;
        result.I2 = result.I2 + (li.int_dudz2 * layers[j].c44 + k2 * layers[j].c66 * li.int_u2);
        result.I3 = result.I3 + layers[j].c66 * li.int_u2;
    }

    T u0 = scaled.states.back().u;
    T L_h = halfSpace.c44, N_h = halfSpace.c66, rho_h = halfSpace.rho;
    T nu_sq_hs = (N_h * k2 - rho_h * omega * omega) / L_h;
    result.int_u2_hs = T(0.0);
    result.int_dudz2_hs = T(0.0);

    bool include_halfspace =
        (branch_policy == LoveBranchPolicy::ForceLinear) ? false :
        (branch_policy == LoveBranchPolicy::ForceEvanescent) ? true :
        (branch_policy == LoveBranchPolicy::ForcePropagating) ? false :
        (branch_policy == LoveBranchPolicy::SignOnly) ? (get_val(nu_sq_hs) >= 0.0) :
        (get_val(nu_sq_hs) > 1e-10);
    if (include_halfspace) {
        T nu_hs_local;
        if (branch_policy == LoveBranchPolicy::ForcePropagating) {
            nu_hs_local = loveBranchSqrtAbs(-nu_sq_hs);
        } else {
            nu_hs_local = loveBranchSqrtAbs(nu_sq_hs);
        }
        T inv2nu = T(1.0) / (T(2.0) * nu_hs_local);
        T relative_scale_sq_hs = T(scaled.relative_scale_sq.back());
        result.int_u2_hs = u0 * u0 * inv2nu * relative_scale_sq_hs;
        result.int_dudz2_hs = nu_sq_hs * u0 * u0 * inv2nu * relative_scale_sq_hs;
        result.I1 = result.I1 + rho_h * result.int_u2_hs;
        result.I2 = result.I2 + (result.int_dudz2_hs * L_h + k2 * N_h * result.int_u2_hs);
        result.I3 = result.I3 + N_h * result.int_u2_hs;
    }

    return result;
}

inline VariationalLoveHessianBreakdown computePhaseVelocityHessianLoveUpwardIFTBreakdown(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool /*computeThickness*/ = false,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto
) {
    VariationalLoveHessianBreakdown result;
    const double omega_v = 2.0 * M_PI * freq;

    auto f_double = [&](double cv) {
        auto Y = calculateLoveDispersion<double>(layers, thicknesses, halfSpace,
            omega_v, cv, NormalizationMode::None);
        return Y(I24);
    };
    double c_max = halfSpace.vs() * (1.0 - 1e-6);
    double c_root = std::min(root_c_double, c_max);
    for (int it = 0; it < 15; ++it) {
        double f0 = f_double(c_root);
        if (std::isnan(f0) || std::isinf(f0)) { c_root = root_c_double; break; }
        double eps = c_root * 1e-7;
        double df = (f_double(c_root + eps) - f0) / eps;
        if (std::abs(df) < 1e-30 || std::isnan(df)) break;
        double step = f0 / df;
        double max_step = 0.1 * (c_max - 1e-3);
        if (std::abs(step) > max_step) step = std::copysign(max_step, step);
        c_root = std::min(c_max, std::max(1e-3, c_root - step));
        if (std::abs(step) < 1e-11) break;
    }
    result.root_c_used = c_root;

    using AD = HessianAD<MAX_AD_PARAMS>;
    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = static_cast<int>(layers.size()) * npl + npl;
    int c_idx = num_params;
    AD::s_an = c_idx + 1;

    AD c_ad = AD::make_var(c_root, c_idx);
    std::vector<PhysicalParameters<AD>> layers_ad;
    int cur = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (mode == SensitivityMode::Isotropic) {
            AD vs  = AD::make_var(layers[i].vs(), cur);
            AD vp  = AD::make_var(layers[i].vp(), cur + 1);
            AD rho = AD::make_var(layers[i].rho, cur + 2);
            layers_ad.push_back(createIsotropicParameters(vs, vp, rho));
        } else {
            PhysicalParameters<AD> p;
            p.rho = AD::make_var(layers[i].rho, cur);
            p.c11 = AD::make_var(layers[i].c11, cur + 1);
            p.c33 = AD::make_var(layers[i].c33, cur + 2);
            p.c13 = AD::make_var(layers[i].c13, cur + 3);
            p.c44 = AD::make_var(layers[i].c44, cur + 4);
            p.c66 = AD::make_var(layers[i].c66, cur + 5);
            layers_ad.push_back(p);
        }
        cur += npl;
    }
    PhysicalParameters<AD> hs_ad;
    if (mode == SensitivityMode::Isotropic) {
        AD hsvs  = AD::make_var(halfSpace.vs(), cur);
        AD hsvp  = AD::make_var(halfSpace.vp(), cur + 1);
        AD hsrho = AD::make_var(halfSpace.rho, cur + 2);
        hs_ad = createIsotropicParameters(hsvs, hsvp, hsrho);
    } else {
        hs_ad.rho = AD::make_var(halfSpace.rho, cur);
        hs_ad.c11 = AD::make_var(halfSpace.c11, cur + 1);
        hs_ad.c33 = AD::make_var(halfSpace.c33, cur + 2);
        hs_ad.c13 = AD::make_var(halfSpace.c13, cur + 3);
        hs_ad.c44 = AD::make_var(halfSpace.c44, cur + 4);
        hs_ad.c66 = AD::make_var(halfSpace.c66, cur + 5);
    }
    std::vector<AD> thick_ad;
    for (auto t : thicknesses) thick_ad.push_back(AD(t));

    std::vector<double> scales;
    calculateLoveDispersion<double>(layers, thicknesses, halfSpace,
        omega_v, c_root, NormalizationMode::Vertical, &scales);

    AD omega_sq = AD(omega_v * omega_v);
    auto scaled = buildLoveScaledBoundaryDataUpward<AD>(
        layers_ad, thick_ad, hs_ad, AD(omega_v), c_ad, branch_policy, &scales);
    auto energy = computeLoveAnalyticalIntegralsUpward<AD>(
        layers_ad, thick_ad, hs_ad, AD(omega_v), c_ad, branch_policy, &scales);

    AD F = energy.I2 - omega_sq * energy.I1;

    result.Fc = F.gr(c_idx);
    result.Fp = F.gr.head(num_params);
    result.Fpc = F.hr().block(0, c_idx, num_params, 1);
    result.Fcc = F.hr()(c_idx, c_idx);
    Eigen::MatrixXd Fpp = F.hr().block(0, 0, num_params, num_params);
    result.F = get_val(F);
    result.Fpp = Fpp.eval();

    auto extract_pp = [&](const auto& expr) {
        return expr.hr().block(0, 0, num_params, num_params).eval();
    };
    result.I1pp = extract_pp(energy.I1);
    result.I2pp = extract_pp(energy.I2);
    result.I1pp_layer.resize(energy.layer_integrals.size());
    result.I2pp_layer.resize(energy.layer_integrals.size());
    result.I1pp_layer_fixed_state.resize(energy.layer_integrals.size());
    result.I2pp_layer_fixed_state.resize(energy.layer_integrals.size());
    result.top_state_u.resize(energy.layer_integrals.size());
    result.top_state_tau.resize(energy.layer_integrals.size());
    result.top_state_u_grad.resize(energy.layer_integrals.size());
    result.top_state_tau_grad.resize(energy.layer_integrals.size());
    result.top_state_u_hess.resize(energy.layer_integrals.size());
    result.top_state_tau_hess.resize(energy.layer_integrals.size());
    AD omega_ad(omega_v);
    AD k_ad = omega_ad / c_ad;
    for (size_t i = 0; i < energy.layer_integrals.size(); ++i) {
        auto i1_expr = layers[i].rho * energy.layer_integrals[i].int_u2;
        auto i2_expr = layers[i].c44 * energy.layer_integrals[i].int_dudz2 +
                       omega_sq * layers[i].c66 * energy.layer_integrals[i].int_u2;
        result.I1pp_layer[i] = extract_pp(i1_expr);
        result.I2pp_layer[i] = extract_pp(i2_expr);
        const AD& u_top_expr = scaled.states[i].u;
        const AD& tau_top_expr = scaled.states[i].tau;
        result.top_state_u[i] = get_val(u_top_expr);
        result.top_state_tau[i] = get_val(tau_top_expr);
        result.top_state_u_grad[i] = u_top_expr.gr.head(num_params);
        result.top_state_tau_grad[i] = tau_top_expr.gr.head(num_params);
        result.top_state_u_hess[i] = extract_pp(u_top_expr);
        result.top_state_tau_hess[i] = extract_pp(tau_top_expr);
        AD u_fixed = AD(result.top_state_u[i]);
        AD tau_fixed = AD(result.top_state_tau[i]);
        auto li_fixed = computeLoveLayerIntegrals<AD>(
            u_fixed, tau_fixed, layers_ad[i], thick_ad[i], omega_ad, k_ad, branch_policy);
        AD relative_scale_sq = AD(scaled.relative_scale_sq[i]);
        auto i1_fixed = (layers_ad[i].rho * li_fixed.int_u2) * relative_scale_sq;
        auto i2_fixed = (layers_ad[i].c44 * li_fixed.int_dudz2 +
                        k_ad * k_ad * layers_ad[i].c66 * li_fixed.int_u2) * relative_scale_sq;
        result.I1pp_layer_fixed_state[i] = extract_pp(i1_fixed);
        result.I2pp_layer_fixed_state[i] = extract_pp(i2_fixed);
    }
    result.I1pp_hs = extract_pp(halfSpace.rho * energy.int_u2_hs);
    result.I2pp_hs = extract_pp(halfSpace.c44 * energy.int_dudz2_hs +
                                omega_sq * halfSpace.c66 * energy.int_u2_hs);
    result.bottom_state_u.resize(layers.size());
    result.bottom_state_tau.resize(layers.size());
    result.bottom_state_u_grad.resize(layers.size());
    result.bottom_state_tau_grad.resize(layers.size());
    result.bottom_state_u_hess.resize(layers.size());
    result.bottom_state_tau_hess.resize(layers.size());
    for (size_t i = 0; i < scaled.bottom_states.size(); ++i) {
        const AD& u_expr = scaled.bottom_states[i].u;
        const AD& tau_expr = scaled.bottom_states[i].tau;
        result.bottom_state_u[i] = get_val(u_expr);
        result.bottom_state_tau[i] = get_val(tau_expr);
        result.bottom_state_u_grad[i] = u_expr.gr.head(num_params);
        result.bottom_state_tau_grad[i] = tau_expr.gr.head(num_params);
        result.bottom_state_u_hess[i] = extract_pp(u_expr);
        result.bottom_state_tau_hess[i] = extract_pp(tau_expr);
    }

    result.gradient = -result.Fp / result.Fc;
    const Eigen::VectorXd& g = result.gradient;
    result.hessian_fpp = -Fpp / result.Fc;
    result.hessian_fpc = -(result.Fpc * g.transpose() + g * result.Fpc.transpose()) / result.Fc;
    result.hessian_fcc = -(result.Fcc * g * g.transpose()) / result.Fc;
    result.hessian_fixed_unsym = -Fpp / result.Fc - g * result.Fpc.transpose() / result.Fc;
    result.hessian_chain_unsym = -(result.Fpc * g.transpose() + result.Fcc * g * g.transpose()) / result.Fc;
    result.hessian = result.hessian_fixed_unsym + result.hessian_chain_unsym;
    result.hessian = ((result.hessian + result.hessian.transpose()) * 0.5).eval();
    result.hessian_fpp = ((result.hessian_fpp + result.hessian_fpp.transpose()) * 0.5).eval();
    result.hessian_fpc = ((result.hessian_fpc + result.hessian_fpc.transpose()) * 0.5).eval();
    result.hessian_fcc = ((result.hessian_fcc + result.hessian_fcc.transpose()) * 0.5).eval();
    result.hessian_fixed_unsym = result.hessian_fixed_unsym.eval();
    result.hessian_chain_unsym = result.hessian_chain_unsym.eval();

    result.I1 = get_val(energy.I1);
    result.I2 = get_val(energy.I2);
    result.I3 = get_val(energy.I3);
    result.I1_hs = get_val(halfSpace.rho * energy.int_u2_hs);
    result.I2_hs = get_val(halfSpace.c44 * energy.int_dudz2_hs + omega_sq * halfSpace.c66 * energy.int_u2_hs);
    result.I3_hs = get_val(halfSpace.c66 * energy.int_u2_hs);
    result.I1_layer.resize(energy.layer_integrals.size());
    result.I2_layer.resize(energy.layer_integrals.size());
    result.I3_layer.resize(energy.layer_integrals.size());
    for (size_t i = 0; i < energy.layer_integrals.size(); ++i) {
        result.I1_layer[i] = get_val(layers[i].rho * energy.layer_integrals[i].int_u2);
        result.I2_layer[i] = get_val(layers[i].c44 * energy.layer_integrals[i].int_dudz2 +
                                     omega_sq * layers[i].c66 * energy.layer_integrals[i].int_u2);
        result.I3_layer[i] = get_val(layers[i].c66 * energy.layer_integrals[i].int_u2);
    }

    if (mode == SensitivityMode::Isotropic) {
        std::string names[] = {"Vs", "Vp", "Rho"};
        for (size_t i = 0; i < layers.size(); ++i)
            for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    } else {
        std::string names[] = {"Rho", "A", "C", "F", "L", "N"};
        for (size_t i = 0; i < layers.size(); ++i)
            for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    }
    return result;
}

// ===========================================================================
// Love Wave Trapezoidal Energy Integrals (for Romberg/Grid mode)
// ===========================================================================
// Uses boundary states at top/bottom of each layer with trapezoidal rule.
// This maintains O(h²) convergence structure needed for Richardson extrapolation.
// ===========================================================================

template <typename T>
LoveEnergyTerms<T> computeLoveEnergyIntegrals_Trap(
    const std::vector<LoveLayerState<T>>& states,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    int numSubLayers = 1,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto
) {
    T k = omega / cp;
    T k2 = k * k;
    LoveEnergyTerms<T> result;
    result.I1 = T(0.0); result.I2 = T(0.0); result.I3 = T(0.0);

    for (size_t j = 0; j < layers.size(); ++j) {
        T u_top   = states[j].u,   tau_top   = states[j].tau;
        T u_bot   = states[j+1].u, tau_bot   = states[j+1].tau;
        T h       = thicknesses[j];
        T L_j     = layers[j].c44;
        T N_j     = layers[j].c66;
        T rho_j   = layers[j].rho;

        T int_u2, int_dudz2;
        if (numSubLayers <= 1) {
            T dudz_top  = tau_top / L_j;
            T dudz_bot  = tau_bot / L_j;
            int_u2    = h * T(0.5) * (u_top*u_top + u_bot*u_bot);
            int_dudz2 = h * T(0.5) * (dudz_top*dudz_top + dudz_bot*dudz_bot);
        } else {
            int n = std::max(2, numSubLayers);
            if (n % 2 != 0) ++n;
            T dz = h / T(n);
            T nu_sq = (N_j * k * k - rho_j * omega * omega) / L_j;

            auto state_at = [&](T z) {
                T u_z, tau_z;
                switch (selectLoveBranchRegime(nu_sq, branch_policy)) {
                case LoveBranchRegime::Evanescent: {
                    T nu;
                    if (branch_policy == LoveBranchPolicy::ForcePropagating) {
                        nu = loveBranchSqrtAbs(-nu_sq);
                    } else {
                        nu = loveBranchSqrtAbs(nu_sq);
                    }
                    T ch = cosh(nu * z);
                    T sh = sinh(nu * z);
                    T eta = L_j * nu;
                    u_z   =  ch * u_top - sh / eta * tau_top;
                    tau_z = -eta * sh * u_top + ch * tau_top;
                    break;
                }
                case LoveBranchRegime::Propagating: {
                    T mu;
                    if (branch_policy == LoveBranchPolicy::ForceEvanescent) {
                        mu = loveBranchSqrtAbs(nu_sq);
                    } else {
                        mu = loveBranchSqrtAbs(-nu_sq);
                    }
                    T co = cos(mu * z);
                    T si = sin(mu * z);
                    T eta = L_j * mu;
                    u_z   = co * u_top - si / eta * tau_top;
                    tau_z = eta * si * u_top + co * tau_top;
                    break;
                }
                case LoveBranchRegime::Linear:
                    u_z = u_top - z / L_j * tau_top;
                    tau_z = tau_top;
                    break;
                }
                return LoveLayerState<T>{u_z, tau_z};
            };

            int_u2 = T(0.0);
            int_dudz2 = T(0.0);
            for (int q = 0; q <= n; ++q) {
                T z = dz * T(q);
                auto st = state_at(z);
                T dudz = st.tau / L_j;
                T weight = (q == 0 || q == n) ? T(1.0) : ((q % 2 == 0) ? T(2.0) : T(4.0));
                int_u2 = int_u2 + weight * st.u * st.u;
                int_dudz2 = int_dudz2 + weight * dudz * dudz;
            }
            int_u2 = int_u2 * dz / T(3.0);
            int_dudz2 = int_dudz2 * dz / T(3.0);
        }

        LayerIntegralResult<T> li{int_u2, int_dudz2};
        result.layer_integrals.push_back(li);

        result.I1 = result.I1 + rho_j * int_u2;
        result.I2 = result.I2 + (int_dudz2 * L_j + k2 * N_j * int_u2);
        result.I3 = result.I3 + N_j * int_u2;
    }

    // Halfspace: analytical (exponential decay)
    T u0       = states.back().u;
    T L_h      = halfSpace.c44, N_h = halfSpace.c66, rho_h = halfSpace.rho;
    T nu_sq_hs = (N_h * k2 - rho_h * omega * omega) / L_h;
    result.int_u2_hs = T(0.0); result.int_dudz2_hs = T(0.0);
    bool include_halfspace =
        (branch_policy == LoveBranchPolicy::ForceLinear) ? false :
        (branch_policy == LoveBranchPolicy::ForceEvanescent) ? true :
        (branch_policy == LoveBranchPolicy::ForcePropagating) ? false :
        (branch_policy == LoveBranchPolicy::SignOnly) ? (get_val(nu_sq_hs) >= 0.0) :
        (get_val(nu_sq_hs) > 1e-10);
    if (include_halfspace) {
        T nu_hs;
        if (branch_policy == LoveBranchPolicy::ForcePropagating) {
            nu_hs = loveBranchSqrtAbs(-nu_sq_hs);
        } else {
            nu_hs = loveBranchSqrtAbs(nu_sq_hs);
        }
        T inv2nu = T(1.0) / (T(2.0) * nu_hs);
        result.int_u2_hs    = u0 * u0 * inv2nu;
        result.int_dudz2_hs = nu_sq_hs * u0 * u0 * inv2nu;
        result.I1 = result.I1 + rho_h * result.int_u2_hs;
        result.I2 = result.I2 + (result.int_dudz2_hs * L_h + k2 * N_h * result.int_u2_hs);
        result.I3 = result.I3 + N_h * result.int_u2_hs;
    }
    return result;
}

// ===========================================================================
// Phase 2: Templated Variational Gradient (Love Wave, AD-compatible)
// ===========================================================================
// Computes variational gradient dc/dp using purely real Phase 1 building
// blocks. Templated on T so it works with autodiff::dual for 1st-order AD.
// ===========================================================================

template <typename T>
struct LoveWeakformGradientPieces {
    LoveEnergyTerms<T> energy;
    T coeff;
    Eigen::Matrix<T, Eigen::Dynamic, 1> raw_gradient;
    Eigen::Matrix<T, Eigen::Dynamic, 1> gradient;
};

template <typename T>
LoveWeakformGradientPieces<T> computeVariationalGradientLovePieces_T(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    SensitivityMode mode,
    bool useAnalytical = true,  // false = numerical integration
    int numSubLayers = 1,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    const std::vector<double>* analytical_scale_factors = nullptr,
    bool disper80_scaling = false
) {
    LoveWeakformGradientPieces<T> result;
    if (useAnalytical) {
        if (disper80_scaling) {
            result.energy = computeLoveAnalyticalIntegralsDisper80<T>(
                layers, thicknesses, halfSpace, omega, cp, branch_policy);
        } else {
            result.energy = computeLoveAnalyticalIntegralsUpward<T>(
                layers, thicknesses, halfSpace, omega, cp, branch_policy,
                analytical_scale_factors);
        }
    } else {
        auto states = propagateLoveEigenstates<T>(layers, thicknesses, omega, cp, branch_policy);
        result.energy = computeLoveEnergyIntegrals_Trap<T>(states, layers, thicknesses, halfSpace, omega, cp, numSubLayers, branch_policy);
    }

    T k = omega / cp;
    T k2 = k * k;
    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = ((int)layers.size() + 1) * npl;
    result.raw_gradient.resize(num_params);
    result.gradient.resize(num_params);
    for (int i = 0; i < num_params; ++i) {
        result.raw_gradient(i) = T(0.0);
        result.gradient(i) = T(0.0);
    }

    result.coeff = cp * cp * cp / (T(2.0) * omega * omega * result.energy.I3);

    for (size_t j = 0; j < layers.size(); ++j) {
        T rho_j = layers[j].rho, L_j = layers[j].c44, N_j = layers[j].c66;
        T raw_drho = -omega * omega * result.energy.layer_integrals[j].int_u2;
        T raw_dL   = result.energy.layer_integrals[j].int_dudz2;
        T raw_dN   = k2 * result.energy.layer_integrals[j].int_u2;

        if (mode == SensitivityMode::Isotropic) {
            T Vs_j = sqrt(L_j / rho_j);
            result.raw_gradient[j*3 + 0] = (raw_dL + raw_dN) * T(2.0) * rho_j * Vs_j;
            result.raw_gradient[j*3 + 1] = T(0.0);
            result.raw_gradient[j*3 + 2] = raw_drho + (raw_dL + raw_dN) * Vs_j * Vs_j;
        } else {
            result.raw_gradient[j*6 + 0] = raw_drho;
            result.raw_gradient[j*6 + 4] = raw_dL;
            result.raw_gradient[j*6 + 5] = raw_dN;
        }
    }

    // Halfspace
    int hs_base = (int)layers.size() * npl;
    if (get_val(result.energy.int_u2_hs) > 1e-30) {
        T raw_drho_h = -omega * omega * result.energy.int_u2_hs;
        T raw_dL_h   = result.energy.int_dudz2_hs;
        T raw_dN_h   = k2 * result.energy.int_u2_hs;

        if (mode == SensitivityMode::Isotropic) {
            T L_h = halfSpace.c44, rho_h = halfSpace.rho;
            T Vs_h = sqrt(L_h / rho_h);
            result.raw_gradient[hs_base + 0] = (raw_dL_h + raw_dN_h) * T(2.0) * rho_h * Vs_h;
            result.raw_gradient[hs_base + 1] = T(0.0);
            result.raw_gradient[hs_base + 2] = raw_drho_h + (raw_dL_h + raw_dN_h) * Vs_h * Vs_h;
        } else {
            result.raw_gradient[hs_base + 0] = raw_drho_h;
            result.raw_gradient[hs_base + 4] = raw_dL_h;
            result.raw_gradient[hs_base + 5] = raw_dN_h;
        }
    }

    result.gradient = result.coeff * result.raw_gradient;
    return result;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> computeVariationalGradientLove_T(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    SensitivityMode mode,
    bool useAnalytical = true,  // false = numerical integration
    int numSubLayers = 1,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    bool disper80_scaling = false
) {
    return computeVariationalGradientLovePieces_T(
        layers, thicknesses, halfSpace, omega, cp, mode, useAnalytical,
        numSubLayers, branch_policy, nullptr, disper80_scaling
    ).gradient;
}

// ===========================================================================
// Phase 3: Weak-Form Hessian (Love Wave, 1st-order AD)
// ===========================================================================
// Computes exact Hessian using variational gradient + autodiff::dual.
// N-pass approach: for each parameter direction j, seeds p_j and c with
// dual derivatives, then computes all gradient components as dual values.
// Cost: O(NL) — N passes of O(L) dual arithmetic, 1 root-find.
// ===========================================================================

// Forward declarations for functions defined later in this file
inline double findLoveRootNear(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega, double c_guess);

// Helper: construct autodiff::dual with value and derivative
inline autodiff::dual make_dual(double val, double grad = 0.0) {
    autodiff::dual d(val);
    d.grad = grad;
    return d;
}

// Helper: copy PhysicalParameters<double> → PhysicalParameters<adual> with zero derivatives
inline PhysicalParameters<autodiff::dual> to_dual(const PhysicalParameters<double>& p) {
    PhysicalParameters<autodiff::dual> d;
    d.rho = autodiff::dual(p.rho);
    d.c11 = autodiff::dual(p.c11);
    d.c13 = autodiff::dual(p.c13);
    d.c33 = autodiff::dual(p.c33);
    d.c44 = autodiff::dual(p.c44);
    d.c66 = autodiff::dual(p.c66);
    return d;
}

// ===========================================================================
// Weak-form Hessian AD backends
// ===========================================================================
// Dual      — N-pass autodiff::dual. One pass per parameter, each one
//             rebuilding the eigenfunctions and energy integrals from scratch.
//             Reference implementation.
// Vector    — vectorized forward AD: each AD scalar carries a fixed-width block
//             of WF_HESSIAN_CHUNK derivative directions, so ceil(P/K) passes
//             replace P passes and the primal trajectory (the complex exp /
//             sqrt / cosh / sinh that dominate the propagation) is evaluated
//             once per block instead of once per parameter.
// VectorDyn — same idea with a runtime-sized Eigen::VectorXd derivative, i.e.
//             a genuine single pass. Kept for validation and benchmarking: it
//             measures slower than Dual because every AD temporary heap
//             allocates its derivative vector, which is what motivated the
//             fixed-width blocks above.
// All three produce the same matrix; see the comment in the vectorized branch
// of computePhaseVelocityHessianWeakFormLove for why.
// ===========================================================================
enum class WFHessianBackend { Dual, Vector, VectorDyn };

// Derivative directions carried by one vectorized pass. Fixed size keeps the
// derivative block on the stack, so no allocation happens per AD temporary.
constexpr int WF_HESSIAN_CHUNK = 16;

template <int K>
using ADChunk = Eigen::AutoDiffScalar<Eigen::Matrix<double, K, 1>>;

// Carries an AD scalar type into a generic lambda so the fixed-width and
// runtime-width passes can share one body.
template <typename T_>
struct TypeTag { using type = T_; };

// AD scalar carrying every parameter direction at once. The same scalar type
// already drives the full-AD group-velocity kernel through these very
// templates (computeGroupVelocityKernelVariational).
using ADVec = Eigen::AutoDiffScalar<Eigen::VectorXd>;

// Helper: build an AD scalar with `width` derivative slots, all zero except an
// optional unit seed at `slot`. Constants are always given a full-length zero
// vector rather than an empty one so Eigen never has to grow them through
// make_coherent inside the propagation loops.
template <typename T>
inline T make_ad_seeded(double val, int width, int slot = -1) {
    T a;
    a.value() = val;
    a.derivatives().setZero(width);
    if (slot >= 0) a.derivatives()(slot) = 1.0;
    return a;
}

template <typename T>
inline T make_ad_grad(double val, const Eigen::VectorXd& grad) {
    T a;
    a.value() = val;
    a.derivatives().setZero(grad.size());
    for (int i = 0; i < grad.size(); ++i) a.derivatives()(i) = grad(i);
    return a;
}

// Helper: copy PhysicalParameters<double> → PhysicalParameters<T> with zero
// derivatives. Mirrors to_dual, including leaving the gravity members at the
// values the solid weak-form path expects.
template <typename T>
inline PhysicalParameters<T> to_ad_params(const PhysicalParameters<double>& p, int width) {
    PhysicalParameters<T> d;
    d.rho = make_ad_seeded<T>(p.rho, width);
    d.c11 = make_ad_seeded<T>(p.c11, width);
    d.c13 = make_ad_seeded<T>(p.c13, width);
    d.c33 = make_ad_seeded<T>(p.c33, width);
    d.c44 = make_ad_seeded<T>(p.c44, width);
    d.c66 = make_ad_seeded<T>(p.c66, width);
    d.gravity = make_ad_seeded<T>(0.0, width);
    d.gravity_internal_scale = make_ad_seeded<T>(1.0, width);
    d.gravity_surface_scale = make_ad_seeded<T>(0.0, width);
    return d;
}

// Seed the parameter block [j0, j0+width) into derivative slots 0..width-1:
// global parameter j lands in slot j-j0, parameters outside the block stay
// constant. j0=0 with width=num_params seeds everything in one pass. The
// parameter ordering and the isotropic chain rule must match the per-pass
// seeding of the dual backend exactly, or the backends stop agreeing.
template <typename T>
inline void seedLayerCakeParamBlock(
    const std::vector<PhysicalParameters<double>>& layers,
    const PhysicalParameters<double>& halfSpace,
    SensitivityMode mode,
    int j0, int width,
    std::vector<PhysicalParameters<T>>& layers_ad,
    PhysicalParameters<T>& hs_ad
) {
    const int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    const int num_params = ((int)layers.size() + 1) * npl;

    layers_ad.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) layers_ad[i] = to_ad_params<T>(layers[i], width);
    hs_ad = to_ad_params<T>(halfSpace, width);

    // Slot of global parameter index j within this block, or -1 if outside.
    auto slot_of = [&](int j) {
        return (j >= j0 && j < j0 + width && j < num_params) ? j - j0 : -1;
    };

    for (size_t i = 0; i <= layers.size(); ++i) {
        const bool is_hs = (i == layers.size());
        const PhysicalParameters<double>& src = is_hs ? halfSpace : layers[i];
        PhysicalParameters<T>& target = is_hs ? hs_ad : layers_ad[i];
        const int base = (int)i * npl;
        if (base >= j0 + width || base + npl <= j0) continue;  // block misses this layer

        if (mode == SensitivityMode::Isotropic) {
            T vs_d = make_ad_seeded<T>(src.vs(), width, slot_of(base + 0));
            T vp_d = make_ad_seeded<T>(src.vp(), width, slot_of(base + 1));
            T rho_d = make_ad_seeded<T>(get_scalar_val(src.rho), width, slot_of(base + 2));

            // Same chain rule as the dual path: c44=c66=rho*vs², c11=c33=rho*vp², …
            PhysicalParameters<T> pp = createIsotropicParameters<T>(vs_d, vp_d, rho_d);
            target.rho = pp.rho;
            target.c11 = pp.c11;
            target.c13 = pp.c13;
            target.c33 = pp.c33;
            target.c44 = pp.c44;
            target.c66 = pp.c66;
        } else {
            // VTI: params are [Rho, A, C, F, L, N]
            const int s_rho = slot_of(base + 0), s_a = slot_of(base + 1), s_c = slot_of(base + 2);
            const int s_f = slot_of(base + 3), s_l = slot_of(base + 4), s_n = slot_of(base + 5);
            if (s_rho >= 0) target.rho.derivatives()(s_rho) = 1.0;
            if (s_a >= 0) target.c11.derivatives()(s_a) = 1.0;
            if (s_c >= 0) target.c33.derivatives()(s_c) = 1.0;
            if (s_f >= 0) target.c13.derivatives()(s_f) = 1.0;
            if (s_l >= 0) target.c44.derivatives()(s_l) = 1.0;
            if (s_n >= 0) target.c66.derivatives()(s_n) = 1.0;
        }
    }
}

// Helper: copy the derivative block of each gradient component into the
// matching Hessian row segment. A component the AD never touched carries no
// derivative storage (the identically-zero Love Vp gradient is the one real
// case); its entries stay zero, which is what the dual backend produces too.
template <typename T>
inline void extractHessianBlock(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& g_ad,
    int j0, int width,
    Eigen::MatrixXd& H
) {
    const int num_params = (int)H.rows();
    const int n_cols = std::min(width, num_params - j0);
    for (int i = 0; i < num_params; ++i) {
        const auto& d = g_ad(i).derivatives();
        if ((int)d.size() < n_cols) continue;
        for (int s = 0; s < n_cols; ++s) H(i, j0 + s) = d(s);
    }
}

// First-order counterpart of the gradient that computePhaseVelocityHessianLove
// produces: the same IFT gradient -Fp/Fc on the Love dispersion function, with
// the same Vertical-normalization pre-pass, but propagated by first-order
// forward AD instead of the full HessianAD<MAX_AD_PARAMS> second-order pass
// whose Hessian is then discarded.
//
// Same formula, so the seed is unchanged to round-off; cost drops from
// O(L·N²) to O(L·N) and the MAX_AD_PARAMS ceiling no longer applies.
// dF/dc comes from a single 1-direction pass and dF/dp from blocked passes,
// so no AD temporary ever heap-allocates its derivative storage.
inline Eigen::VectorXd computeLoveIFTGradientFirstOrder(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega, double c_root,
    SensitivityMode mode
) {
    const int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    const int num_params = ((int)layers.size() + 1) * npl;

    // Pre-pass: collect normalization scale factors as double constants.
    // Treating them as constants is exact at the root F=0 (IFT argument).
    // Must match computePhaseVelocityHessianLove, or the two gradients differ
    // at the 1e-12 level through the F scaling.
    std::vector<double> scales;
    calculateLoveDispersion<double>(layers, thicknesses, halfSpace,
        omega, c_root, NormalizationMode::Vertical, &scales);

    double Fc;
    {
        using AD1 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;
        auto wrap = [](double v, double d) { AD1 a; a.value() = v; a.derivatives()(0) = d; return a; };
        auto to_ad1 = [&](const PhysicalParameters<double>& p) {
            PhysicalParameters<AD1> pa;
            pa.rho = wrap(p.rho, 0.0); pa.c11 = wrap(p.c11, 0.0); pa.c33 = wrap(p.c33, 0.0);
            pa.c13 = wrap(p.c13, 0.0); pa.c44 = wrap(p.c44, 0.0); pa.c66 = wrap(p.c66, 0.0);
            pa.gravity = wrap(0.0, 0.0);
            pa.gravity_internal_scale = wrap(1.0, 0.0);
            pa.gravity_surface_scale = wrap(0.0, 0.0);
            return pa;
        };
        std::vector<PhysicalParameters<AD1>> l1;
        for (auto& p : layers) l1.push_back(to_ad1(p));
        std::vector<AD1> t1;
        for (double t : thicknesses) t1.push_back(wrap(t, 0.0));
        PhysicalParameters<AD1> h1 = to_ad1(halfSpace);

        size_t sIdx = 0;
        auto Y = calculateLoveDispersion<AD1>(l1, t1, h1, wrap(omega, 0.0), wrap(c_root, 1.0),
                                              NormalizationMode::Provided, &scales, &sIdx);
        Fc = Y(I24).derivatives()(0);
    }
    if (!(std::abs(Fc) > 1e-300) || !std::isfinite(Fc))
        return Eigen::VectorXd::Zero(num_params);

    using T = ADChunk<WF_HESSIAN_CHUNK>;
    constexpr int width = WF_HESSIAN_CHUNK;
    Eigen::VectorXd Fp = Eigen::VectorXd::Zero(num_params);

    for (int j0 = 0; j0 < num_params; j0 += width) {
        std::vector<PhysicalParameters<T>> l_ad;
        PhysicalParameters<T> h_ad;
        seedLayerCakeParamBlock<T>(layers, halfSpace, mode, j0, width, l_ad, h_ad);

        std::vector<T> t_ad(thicknesses.size());
        for (size_t i = 0; i < thicknesses.size(); ++i)
            t_ad[i] = make_ad_seeded<T>(thicknesses[i], width);

        size_t sIdx = 0;
        auto Y = calculateLoveDispersion<T>(
            l_ad, t_ad, h_ad, make_ad_seeded<T>(omega, width),
            make_ad_seeded<T>(c_root, width),
            NormalizationMode::Provided, &scales, &sIdx);

        const auto& d = Y(I24).derivatives();
        const int n_cols = std::min(width, num_params - j0);
        for (int s = 0; s < n_cols; ++s) Fp(j0 + s) = d(s);
    }

    return -Fp / Fc;
}

inline HessianResult computePhaseVelocityHessianWeakFormLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool /*computeThickness*/ = false,
    int numSubLayers = 200,
    bool useAnalytical = true,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    double cp_grad_multiplier = 1.0,
    WFHessianBackend backend = WFHessianBackend::Dual
) {
    using adual = autodiff::dual;
    HessianResult result;
    const double omega = 2.0 * M_PI * freq;

    // Newton polish
    double c_root = findLoveRootNear(layers, thicknesses, halfSpace, omega, root_c_double);

    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = ((int)layers.size() + 1) * npl;

    // Step 1: Compute the reference gradient used for the IFT c-seed.
    // The IFT seed dc/dp must be as accurate as the Hessian target. A small
    // variational-gradient residual is amplified at short periods, so this is
    // the IFT gradient of the dispersion function, not the variational one.
    Eigen::VectorXd g_seed = computeLoveIFTGradientFirstOrder(
        layers, thicknesses, halfSpace, omega, c_root, mode);

    // Step 2: AD pass(es) for the Hessian.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_params, num_params);

    if (backend != WFHessianBackend::Dual) {
        // Vectorized forward AD. Forward AD is linear in the seed, so seeding a
        // block of parameters p_j with unit directions e_j while seeding c with
        // the MATCHING block of g_seed gives, for gradient component i,
        //     g_i.derivatives()(s) = ∂g_i/∂p_{j0+s} + (∂g_i/∂c)·g_seed(j0+s)
        // which is exactly the H(i, j0+s) the N-pass builds one column at a
        // time. The c seed must carry the gradient values themselves, not a unit
        // direction — that is what turns the IFT term into the rank-1 update
        // (∂g/∂c) ⊗ g_seed. So one pass covers a whole block of columns, and
        // the primal trajectory is evaluated once per block instead of once per
        // parameter.
        const int width = (backend == WFHessianBackend::VectorDyn)
                              ? num_params : WF_HESSIAN_CHUNK;

        auto run_block = [&](auto tag, int j0) {
            using T = typename decltype(tag)::type;
            std::vector<PhysicalParameters<T>> v_layers;
            PhysicalParameters<T> v_hs;
            seedLayerCakeParamBlock<T>(layers, halfSpace, mode, j0, width, v_layers, v_hs);

            std::vector<T> v_thick(thicknesses.size());
            for (size_t i = 0; i < thicknesses.size(); ++i)
                v_thick[i] = make_ad_seeded<T>(thicknesses[i], width);

            T v_omega = make_ad_seeded<T>(omega, width);
            Eigen::VectorXd cp_seed = Eigen::VectorXd::Zero(width);
            for (int s = 0; s < width && j0 + s < num_params; ++s)
                cp_seed(s) = cp_grad_multiplier * g_seed(j0 + s);
            T v_cp = make_ad_grad<T>(c_root, cp_seed);

            Eigen::Matrix<T, Eigen::Dynamic, 1> g_ad = computeVariationalGradientLove_T<T>(
                v_layers, v_thick, v_hs, v_omega, v_cp, mode, useAnalytical, numSubLayers,
                branch_policy);

            extractHessianBlock<T>(g_ad, j0, width, H);
        };

        if (backend == WFHessianBackend::VectorDyn) {
            run_block(TypeTag<ADVec>{}, 0);
        } else {
            for (int j0 = 0; j0 < num_params; j0 += width)
                run_block(TypeTag<ADChunk<WF_HESSIAN_CHUNK>>{}, j0);
        }
    } else {
    std::vector<PhysicalParameters<adual>> d_layers_base(layers.size());
    std::vector<adual> d_thick_base(thicknesses.size());
    for (size_t i = 0; i < layers.size(); ++i)
        d_layers_base[i] = to_dual(layers[i]);
    for (size_t i = 0; i < thicknesses.size(); ++i)
        d_thick_base[i] = adual(thicknesses[i]);
    PhysicalParameters<adual> d_hs_base = to_dual(halfSpace);

    // Work buffers reused across passes to avoid repeated allocations.
    std::vector<PhysicalParameters<adual>> d_layers = d_layers_base;
    std::vector<adual> d_thick = d_thick_base;
    PhysicalParameters<adual> d_hs = d_hs_base;

    for (int j = 0; j < num_params; ++j) {
        // Reset to base dual parameters (zero derivatives)
        d_layers = d_layers_base;
        d_thick = d_thick_base;
        d_hs = d_hs_base;

        // Seed parameter j with derivative = 1
        int layer_idx = j / npl;
        int param_in_layer = j % npl;
        bool is_hs = (layer_idx >= (int)layers.size());
        PhysicalParameters<adual>& target = is_hs ? d_hs : d_layers[layer_idx];

        if (mode == SensitivityMode::Isotropic) {
            double vs_val = is_hs ? halfSpace.vs() : layers[layer_idx].vs();
            double vp_val = is_hs ? halfSpace.vp() : layers[layer_idx].vp();
            double rho_val = get_scalar_val(is_hs ? halfSpace.rho : layers[layer_idx].rho);
            adual vs_d(vs_val), vp_d(vp_val), rho_d(rho_val);

            if (param_in_layer == 0) vs_d.grad = 1.0;
            else if (param_in_layer == 1) vp_d.grad = 1.0;
            else rho_d.grad = 1.0;

            // Reconstruct with AD chain rule: c44=rho*vs², c66=rho*vs², etc.
            auto pp = createIsotropicParameters<adual>(vs_d, vp_d, rho_d);
            target.rho = pp.rho;
            target.c11 = pp.c11;
            target.c13 = pp.c13;
            target.c33 = pp.c33;
            target.c44 = pp.c44;
            target.c66 = pp.c66;
        } else {
            // VTI: params are [Rho, A, C, F, L, N]
            switch (param_in_layer) {
                case 0: target.rho.grad = 1.0; break;
                case 1: target.c11.grad = 1.0; break;
                case 2: target.c33.grad = 1.0; break;
                case 3: target.c13.grad = 1.0; break;
                case 4: target.c44.grad = 1.0; break;
                case 5: target.c66.grad = 1.0; break;
            }
        }

        // Seed cp with derivative = g_j (implicit function theorem: dc/dp_j = g_j)
        adual d_omega(omega);
        adual d_cp = make_dual(c_root, cp_grad_multiplier * g_seed(j));

        // Compute gradient as dual values
        auto g_dual = computeVariationalGradientLove_T<adual>(
            d_layers, d_thick, d_hs, d_omega, d_cp, mode, useAnalytical, numSubLayers, branch_policy);

        // Extract derivatives → H column j
        for (int i = 0; i < num_params; ++i)
            H(i, j) = g_dual(i).grad;
    }
    }

    // Symmetrize
    H = ((H + H.transpose()) * 0.5).eval();

    result.gradient = g_seed;
    result.hessian = H;

    // Parameter names
    if (mode == SensitivityMode::Isotropic) {
        std::string names[] = {"Vs", "Vp", "Rho"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    } else {
        std::string names[] = {"Rho","A","C","F","L","N"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    }
    return result;
}

struct LoveWeakformHessianBreakdown {
    Eigen::MatrixXd hessian;
    Eigen::MatrixXd hessian_unsym;
    Eigen::MatrixXd hessian_coeff;
    Eigen::MatrixXd hessian_coeff_unsym;
    Eigen::MatrixXd hessian_raw;
    Eigen::MatrixXd hessian_raw_unsym;
    Eigen::VectorXd gradient;
    Eigen::VectorXd raw_gradient;
    double coeff = NAN;
    double I3 = NAN;
    std::vector<double> I1_layer, I2_layer, I3_layer;
    double I1_hs = NAN, I2_hs = NAN, I3_hs = NAN;
    std::vector<double> int_u2_layer, int_dudz2_layer;
    double int_u2_hs = NAN, int_dudz2_hs = NAN;
    std::vector<std::string> parameterNames;
    std::vector<int> layerIndices;
};

inline LoveWeakformHessianBreakdown computePhaseVelocityHessianWeakFormLoveBreakdown(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    int numSubLayers = 200,
    bool useAnalytical = true,
    LoveBranchPolicy branch_policy = LoveBranchPolicy::Auto,
    const std::vector<double>* analytical_scale_factors = nullptr,
    double cp_grad_multiplier = 1.0
) {
    using adual = autodiff::dual;
    LoveWeakformHessianBreakdown result;
    const double omega = 2.0 * M_PI * freq;

    double c_root = findLoveRootNear(layers, thicknesses, halfSpace, omega, root_c_double);
    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = ((int)layers.size() + 1) * npl;

    auto base = computeVariationalGradientLovePieces_T<double>(
        layers, thicknesses, halfSpace, omega, c_root, mode, useAnalytical, numSubLayers, branch_policy,
        analytical_scale_factors);
    result.gradient = base.gradient;
    result.raw_gradient = base.raw_gradient;
    result.coeff = base.coeff;
    result.I3 = base.energy.I3;
    double k2 = (omega / c_root) * (omega / c_root);
    result.I1_layer.resize(base.energy.layer_integrals.size());
    result.I2_layer.resize(base.energy.layer_integrals.size());
    result.I3_layer.resize(base.energy.layer_integrals.size());
    result.int_u2_layer.resize(base.energy.layer_integrals.size());
    result.int_dudz2_layer.resize(base.energy.layer_integrals.size());
    for (size_t i = 0; i < base.energy.layer_integrals.size(); ++i) {
        result.int_u2_layer[i] = get_val(base.energy.layer_integrals[i].int_u2);
        result.int_dudz2_layer[i] = get_val(base.energy.layer_integrals[i].int_dudz2);
        result.I1_layer[i] = get_val(base.energy.layer_integrals[i].int_u2 * layers[i].rho);
        result.I2_layer[i] = get_val(base.energy.layer_integrals[i].int_dudz2 * layers[i].c44 +
                                     k2 * layers[i].c66 * base.energy.layer_integrals[i].int_u2);
        result.I3_layer[i] = get_val(base.energy.layer_integrals[i].int_u2 * layers[i].c66);
    }
    result.I1_hs = get_val(base.energy.int_u2_hs * halfSpace.rho);
    result.I2_hs = get_val(base.energy.int_dudz2_hs * halfSpace.c44 +
                           k2 * halfSpace.c66 * base.energy.int_u2_hs);
    result.I3_hs = get_val(base.energy.int_u2_hs * halfSpace.c66);
    result.int_u2_hs = get_val(base.energy.int_u2_hs);
    result.int_dudz2_hs = get_val(base.energy.int_dudz2_hs);

    Eigen::VectorXd g_seed = computeLoveIFTGradientFirstOrder(
        layers, thicknesses, halfSpace, omega, c_root, mode);

    result.hessian = Eigen::MatrixXd::Zero(num_params, num_params);
    result.hessian_unsym = Eigen::MatrixXd::Zero(num_params, num_params);
    result.hessian_coeff = Eigen::MatrixXd::Zero(num_params, num_params);
    result.hessian_coeff_unsym = Eigen::MatrixXd::Zero(num_params, num_params);
    result.hessian_raw = Eigen::MatrixXd::Zero(num_params, num_params);
    result.hessian_raw_unsym = Eigen::MatrixXd::Zero(num_params, num_params);

    std::vector<PhysicalParameters<adual>> d_layers_base(layers.size());
    std::vector<adual> d_thick_base(thicknesses.size());
    for (size_t i = 0; i < layers.size(); ++i) d_layers_base[i] = to_dual(layers[i]);
    for (size_t i = 0; i < thicknesses.size(); ++i) d_thick_base[i] = adual(thicknesses[i]);
    PhysicalParameters<adual> d_hs_base = to_dual(halfSpace);

    std::vector<PhysicalParameters<adual>> d_layers = d_layers_base;
    std::vector<adual> d_thick = d_thick_base;
    PhysicalParameters<adual> d_hs = d_hs_base;

    for (int j = 0; j < num_params; ++j) {
        d_layers = d_layers_base;
        d_thick = d_thick_base;
        d_hs = d_hs_base;

        int layer_idx = j / npl;
        int param_in_layer = j % npl;
        bool is_hs = (layer_idx >= (int)layers.size());
        PhysicalParameters<adual>& target = is_hs ? d_hs : d_layers[layer_idx];

        if (mode == SensitivityMode::Isotropic) {
            double vs_val = is_hs ? halfSpace.vs() : layers[layer_idx].vs();
            double vp_val = is_hs ? halfSpace.vp() : layers[layer_idx].vp();
            double rho_val = get_scalar_val(is_hs ? halfSpace.rho : layers[layer_idx].rho);
            adual vs_d(vs_val), vp_d(vp_val), rho_d(rho_val);

            if (param_in_layer == 0) vs_d.grad = 1.0;
            else if (param_in_layer == 1) vp_d.grad = 1.0;
            else rho_d.grad = 1.0;

            auto pp = createIsotropicParameters<adual>(vs_d, vp_d, rho_d);
            target.rho = pp.rho;
            target.c11 = pp.c11;
            target.c13 = pp.c13;
            target.c33 = pp.c33;
            target.c44 = pp.c44;
            target.c66 = pp.c66;
        } else {
            switch (param_in_layer) {
                case 0: target.rho.grad = 1.0; break;
                case 1: target.c11.grad = 1.0; break;
                case 2: target.c33.grad = 1.0; break;
                case 3: target.c13.grad = 1.0; break;
                case 4: target.c44.grad = 1.0; break;
                case 5: target.c66.grad = 1.0; break;
            }
        }

        adual d_cp = make_dual(c_root, cp_grad_multiplier * g_seed(j));
        auto pieces = computeVariationalGradientLovePieces_T<adual>(
            d_layers, d_thick, d_hs, adual(omega), d_cp, mode, useAnalytical, numSubLayers, branch_policy,
            analytical_scale_factors);
        double coeff_val = get_val(pieces.coeff);
        double coeff_grad = pieces.coeff.grad;

        for (int i = 0; i < num_params; ++i) {
            double raw_val = get_val(pieces.raw_gradient(i));
            double raw_grad = pieces.raw_gradient(i).grad;
            result.hessian_coeff_unsym(i, j) = raw_val * coeff_grad;
            result.hessian_raw_unsym(i, j) = coeff_val * raw_grad;
            result.hessian_unsym(i, j) = result.hessian_coeff_unsym(i, j) + result.hessian_raw_unsym(i, j);
        }
    }

    result.hessian = result.hessian_unsym;
    result.hessian_coeff = result.hessian_coeff_unsym;
    result.hessian_raw = result.hessian_raw_unsym;
    result.hessian = ((result.hessian + result.hessian.transpose()) * 0.5).eval();
    result.hessian_coeff = ((result.hessian_coeff + result.hessian_coeff.transpose()) * 0.5).eval();
    result.hessian_raw = ((result.hessian_raw + result.hessian_raw.transpose()) * 0.5).eval();

    if (mode == SensitivityMode::Isotropic) {
        std::string names[] = {"Vs", "Vp", "Rho"};
        for (int i = 0; i < (int)layers.size(); ++i)
            for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    } else {
        std::string names[] = {"Rho","A","C","F","L","N"};
        for (int i = 0; i < (int)layers.size(); ++i)
            for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for (auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    }

    return result;
}

// ===========================================================================
// Weak-Form Hessian Building Blocks (Rayleigh Wave)
// ===========================================================================
// Extends the Love wave approach to P-SV (Rayleigh) waves.
// Uses ADComplex<T> to handle complex arithmetic with autodiff::dual.
// Key differences from Love:
//   - 4-component state vector [uz, σzz, ur, σzr]
//   - 5×5 compound matrix (real) + 4×4 PSV propagator (complex)
//   - Surface state R = ur/uz depends on parameters (via compound matrix)
//   - 6 per-layer eigenfunction integrals for energy
// ===========================================================================

// Eigenfunction state at a depth point (CT-compatible)
template <typename T>
struct RayleighStatePoint {
    T depth;
    typename ComplexType<T>::type ur, uz, szz, szr;
};

// Per-layer eigenfunction integrals for variational gradient
template <typename T>
struct RayleighLayerIntegrals {
    T int_uz2_ur2;   // ∫(uz² + ur²) dz
    T int_szz2;      // ∫σzz² dz
    T int_szr2;      // ∫σzr² dz
    T int_ur2;       // ∫ur² dz
    T int_uz_szr;    // ∫uz·σzr dz
    T int_szz_ur;    // ∫σzz·ur dz
};

// Rayleigh energy integrals result
template <typename T>
struct RayleighEnergyResult {
    T I1, I2, I3;
    std::vector<RayleighLayerIntegrals<T>> layer_integrals;
    RayleighLayerIntegrals<T> hs_integrals;  // halfspace contribution
};

// CT-compatible 4×4 PSV layer matrix (uses ComplexType instead of std::complex)
template <typename T>
Eigen::Matrix<typename ComplexType<T>::type, 4, 4> computeLayerMatrixP_SV_CT(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v_var
) {
    return computeRayleighStatePropagatorEntire(p, v_var);
}

// Propagate Rayleigh eigenfunctions (CT-compatible version)
// Returns eigenfunction at all sublayer boundaries.
template <typename T>
std::vector<RayleighStatePoint<T>> propagateRayleighEigenstates(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega, T phaseVelocity,
    int numSubLayers = 10,
    bool useAdaptiveSplit = true
) {
    using CT = typename ComplexType<T>::type;
    using Vector5 = Eigen::Matrix<T, 5, 1>;
    std::vector<RayleighStatePoint<T>> results;
    std::vector<PhysicalParameters<T>> trace_layers_storage;
    std::vector<T> trace_thicknesses_storage;
    const std::vector<PhysicalParameters<T>>* trace_layers = &layers;
    const std::vector<T>* trace_thicknesses = &thicknesses;
    if (useAdaptiveSplit) {
        auto virtual_layers = buildAdaptiveRayleighVirtualLayers(
            layers, thicknesses, omega, numSubLayers);
        trace_layers_storage = std::move(virtual_layers.layers);
        trace_thicknesses_storage = std::move(virtual_layers.thicknesses);
        trace_layers = &trace_layers_storage;
        trace_thicknesses = &trace_thicknesses_storage;
    }
    if (isLiquidLayer(halfSpaceParams)) {
        throw std::runtime_error("Rayleigh eigenstates do not support a liquid half-space.");
    }

    T k = omega / phaseVelocity;
    WaveVariables<T> wave;
    wave.omega = omega; wave.k = k; wave.z = T(0.0);

    const int top_liquid_count = countTopLiquidLayersOrThrow(*trace_layers);
    if (top_liquid_count > 0) {
        const int solid_count = static_cast<int>(trace_layers->size()) - top_liquid_count;
        std::vector<PhysicalParameters<T>> solid_layers;
        std::vector<T> solid_thicknesses;
        solid_layers.reserve(std::max(0, solid_count));
        solid_thicknesses.reserve(std::max(0, solid_count));
        for (int i = top_liquid_count; i < static_cast<int>(trace_layers->size()); ++i) {
            solid_layers.push_back((*trace_layers)[i]);
            solid_thicknesses.push_back((*trace_thicknesses)[i]);
        }

        std::vector<Vector5> interfaceMinors(solid_layers.size() + 1);
        Vector5 Y_curr = computeInitialVector(halfSpaceParams, wave);
        double max_val = 0.0;
        for (int m = 0; m < 5; ++m) max_val = std::max(max_val, get_magnitude(Y_curr(m)));
        if (max_val > 0.0) Y_curr /= T(max_val);
        interfaceMinors[solid_layers.size()] = Y_curr;

        for (int i = static_cast<int>(solid_layers.size()) - 1; i >= 0; --i) {
            wave.z = solid_thicknesses[i];
            auto Qshifted = computeLayerMatrix(solid_layers[i], wave);
            Y_curr = Qshifted * Y_curr;
            max_val = 0.0;
            for (int m = 0; m < 5; ++m) max_val = std::max(max_val, get_magnitude(Y_curr(m)));
            if (max_val > 1e-20) Y_curr /= T(max_val);
            else Y_curr.setZero();
            interfaceMinors[i] = Y_curr;
        }

        Eigen::Matrix<CT, 2, 1> fluid_state;
        fluid_state(0) = toComplex(T(1.0));
        T surface_gravity = (*trace_layers)[0].gravity * (*trace_layers)[0].gravity_surface_scale;
        T surface_coeff = (*trace_layers)[0].rho * surface_gravity;
        fluid_state(1) = toComplex(surface_coeff) * fluid_state(0);

        T currentDepth = T(0.0);
        auto push_fluid_point = [&](const PhysicalParameters<T>& layer) {
            CT uz = fluid_state(0);
            CT szz = fluid_state(1);
            CT ur = computeFluidHorizontalDisplacement<T>(layer, uz, szz, omega, phaseVelocity);
            results.push_back({currentDepth, ur, uz, szz, toComplex(T(0.0))});
        };

        push_fluid_point((*trace_layers)[0]);
        for (int i = 0; i < top_liquid_count; ++i) {
            wave.z = -(*trace_thicknesses)[i];
            auto B_liq = computeLayerMatrixLiquid((*trace_layers)[i], wave);
            Eigen::Matrix<CT, 2, 1> next_state;
            next_state(0) = toComplex(B_liq(0,0)) * fluid_state(0) + toComplex(B_liq(0,1)) * fluid_state(1);
            next_state(1) = toComplex(B_liq(1,0)) * fluid_state(0) + toComplex(B_liq(1,1)) * fluid_state(1);
            fluid_state = next_state;
            currentDepth = currentDepth + (*trace_thicknesses)[i];
            push_fluid_point((*trace_layers)[i]);
            if (i + 1 < top_liquid_count) {
                push_fluid_point((*trace_layers)[i + 1]);
            }
        }

        Vector5 Y_seabed = interfaceMinors[0];
        CT uz = fluid_state(0);
        CT szz = fluid_state(1);

        CT ur = toComplex(T(0.0));
        if (get_magnitude(Y_seabed(I12)) > 1e-15) {
            ur = (szz * toComplex(Y_seabed(I13)) - uz * toComplex(Y_seabed(I23))) / toComplex(Y_seabed(I12));
        } else if (get_magnitude(Y_seabed(I14)) > 1e-15) {
            // szr = 0 at the seabed, so ur*M14 + uz*M12 = szr*M13 = 0.
            ur = -uz * toComplex(Y_seabed(I12)) / toComplex(Y_seabed(I14));
        }
        Eigen::Matrix<CT, 4, 1> state;
        CT ct_zero = toComplex(T(0.0));
        state << uz, szz, ur, ct_zero;
        results.push_back({currentDepth, state(2), state(0), state(1), state(3)});

        for (int i = 0; i < solid_count; ++i) {
            wave.z = -solid_thicknesses[i];
            auto PSV = computeLayerMatrixP_SV_CT<T>(solid_layers[i], wave);
            Eigen::Matrix<CT, 4, 1> new_state;
            for (int r = 0; r < 4; ++r) {
                new_state(r) = ct_zero;
                for (int c = 0; c < 4; ++c)
                    new_state(r) = new_state(r) + PSV(r,c) * state(c);
            }
            state = new_state;
            currentDepth = currentDepth + solid_thicknesses[i];

            Vector5 Y_local = interfaceMinors[i + 1];
            T M13 = Y_local(I13);
            if (get_magnitude(M13) > 1e-15) {
                T M12 = Y_local(I12);
                T M23 = Y_local(I23);
                T M14 = Y_local(I14);
                CT ur_local = state(2), uz_local = state(0);
                state(1) = (uz_local * toComplex(M23) + ur_local * toComplex(M12)) / toComplex(M13);
                state(3) = (ur_local * toComplex(M14) + uz_local * toComplex(M12)) / toComplex(M13);
            }
            results.push_back({currentDepth, state(2), state(0), state(1), state(3)});
        }

        return results;
    }

    // --- Step 1: Upward Pass (Compound matrix — already CT-compatible) ---
    int N = static_cast<int>(trace_layers->size());
    std::vector<Vector5> interfaceMinors(N + 1);

    Vector5 Y_curr = computeInitialVector(halfSpaceParams, wave);
    double max_val = 0.0;
    for(int m=0; m<5; ++m) max_val = std::max(max_val, get_magnitude(Y_curr(m)));
    if(max_val > 0) Y_curr /= T(max_val);
    interfaceMinors[N] = Y_curr;

    for (int i = N - 1; i >= 0; --i) {
        wave.z = (*trace_thicknesses)[i];
        auto Qshifted = computeLayerMatrix((*trace_layers)[i], wave);
        Y_curr = Qshifted * Y_curr;
        max_val = 0.0;
        for(int m=0; m<5; ++m) max_val = std::max(max_val, get_magnitude(Y_curr(m)));
        if(max_val > 1e-20) Y_curr /= T(max_val);
        else Y_curr.setZero();
        interfaceMinors[i] = Y_curr;
    }

    // --- Step 2: Surface Initial Condition ---
    Vector5 Y_surf = interfaceMinors[0];
    // Match the stable traction-free pivot in computeRayleighEigenfunctions.
    // The pivot only selects a common normalization, with no physical jump.
    const bool vertical_pivot = get_magnitude(Y_surf(I14)) >= get_magnitude(Y_surf(I23));
    T ur_surf_r = vertical_pivot ? T(-Y_surf(I12)) : Y_surf(I23);
    T uz_surf_r = vertical_pivot ? Y_surf(I14) : T(-Y_surf(I12));

    CT ur_s = toComplex(ur_surf_r);
    CT uz_s = toComplex(uz_surf_r);

    Eigen::Matrix<CT, 4, 1> state;
    CT ct_zero = toComplex(T(0.0));
    state << uz_s, ct_zero, ur_s, ct_zero;  // [uz, szz=0, ur, szr=0]

    double mag_ur = get_magnitude(ur_surf_r);
    double mag_uz = get_magnitude(uz_surf_r);
    if (mag_ur < 1e-20 && mag_uz < 1e-20) {
        CT ct_one(T(1.0));
        state << ct_one, ct_zero, ct_one, ct_zero;
    } else {
        if (mag_uz > mag_ur) state /= uz_s;
        else state /= ur_s;
    }

    // --- Step 3: Downward Pass ---
    T currentDepth = T(0.0);
    results.push_back({currentDepth, state(2), state(0), state(1), state(3)});

    for (int i = 0; i < N; ++i) {
        wave.z = -(*trace_thicknesses)[i];
        auto PSV = computeLayerMatrixP_SV_CT<T>((*trace_layers)[i], wave);
        Eigen::Matrix<CT, 4, 1> new_state;
        for(int r=0; r<4; ++r) {
            new_state(r) = ct_zero;
            for(int c=0; c<4; ++c)
                new_state(r) = new_state(r) + PSV(r,c) * state(c);
        }
        state = new_state;
        currentDepth = currentDepth + (*trace_thicknesses)[i];

        Vector5 Y_local = interfaceMinors[i+1];
        T M13 = Y_local(I13);
        if (get_magnitude(M13) > 1e-15) {
            T M12 = Y_local(I12);
            T M23 = Y_local(I23);
            T M14 = Y_local(I14);
            CT ur = state(2), uz = state(0);
            CT szz_recon = (uz * toComplex(M23) + ur * toComplex(M12)) / toComplex(M13);
            CT szr_recon = (ur * toComplex(M14) + uz * toComplex(M12)) / toComplex(M13);
            state(1) = szz_recon;
            state(3) = szr_recon;
        }
        results.push_back({currentDepth, state(2), state(0), state(1), state(3)});
    }

    return results;
}

// Shared analytical moments, regular in either squared vertical wavenumber.
template <typename T>
RayleighLayerIntegrals<T> computeRayleighLayerIntegrals_Analytical(
    T uz_top, T szz_top, T ur_top, T szr_top,
    const PhysicalParameters<T>& layer,
    T h, T omega, T k
) {
    using CT = typename ComplexType<T>::type;
    if (isLiquidLayer(layer)) {
        // Acoustic layer: only [U,Tzz] propagate. ur follows from Saito's
        // liquid relation (computeFluidHorizontalDisplacement) and szr = 0.
        CT uu, ut, tt;
        disper26_energy::fluid_moments(layer,h,omega,k,
            toComplex(uz_top),toComplex(szz_top),uu,ut,tt);
        const T m11 = uu.real(), m12 = ut.real(), m22 = tt.real();
        const T zero = make_zero_like(h);
        const T factor = k / (layer.rho * omega * omega);
        const T gravity_coeff = layer.rho * (layer.gravity * layer.gravity_internal_scale);
        const T ur2 = factor * factor
            * (gravity_coeff * gravity_coeff * m11 - T(2.0) * gravity_coeff * m12 + m22);
        const T szz_ur = factor * (gravity_coeff * m12 - m22);
        return {T(m11 + ur2), m22, zero, ur2, zero, szz_ur};
    }
    Eigen::Matrix<CT,4,4> moments;
    disper26_energy::rayleigh_moments(layer,h,omega,k,
        std::array<CT,4>{toComplex(uz_top),toComplex(szz_top),toComplex(ur_top),toComplex(szr_top)},moments);
    return {(moments(0,0)+moments(2,2)).real(),moments(1,1).real(),
            moments(3,3).real(),moments(2,2).real(),moments(0,3).real(),moments(1,2).real()};
}

// Analytical Rayleigh energy integrals (per-layer analytical + halfspace analytical)
template <typename T>
RayleighEnergyResult<T> computeRayleighEnergyIntegrals_Analytical(
    const std::vector<RayleighStatePoint<T>>& boundary_states,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega, T phaseVelocity,
    bool useAdaptiveSplit = true
) {
    // Liquid layers at the model top use the acoustic moments of
    // computeRayleighLayerIntegrals_Analytical; the half-space modal integral
    // below divides by c44 and has no liquid form.
    // See docs/fluid_gravity_status.md.
    if (isLiquidLayer(halfSpaceParams)) {
        throw std::runtime_error(
            "Analytical Rayleigh energy integrals are not implemented for a "
            "fluid half-space.");
    }
    RayleighEnergyResult<T> res;
    T zero = T(0.0);
    res.I1 = zero; res.I2 = zero; res.I3 = zero;

    T k = omega / phaseVelocity;
    T k_sq = k * k;
    int N = (int)layers.size();
    res.layer_integrals.resize(N + 1);
    for (int l = 0; l <= N; ++l)
        res.layer_integrals[l] = {zero, zero, zero, zero, zero, zero};

    auto add_layer_integrals = [](RayleighLayerIntegrals<T>& dst, const RayleighLayerIntegrals<T>& src) {
        dst.int_uz2_ur2 += src.int_uz2_ur2;
        dst.int_szz2 += src.int_szz2;
        dst.int_szr2 += src.int_szr2;
        dst.int_ur2 += src.int_ur2;
        dst.int_uz_szr += src.int_uz_szr;
        dst.int_szz_ur += src.int_szz_ur;
    };

    auto virtual_layers = RayleighVirtualLayerModel<T>{};
    if (useAdaptiveSplit) {
        virtual_layers = buildAdaptiveRayleighVirtualLayers(layers, thicknesses, omega, 1);
    }
    const auto expected_states = [](const std::vector<PhysicalParameters<T>>& stack) {
        return stack.size() + 1
            + (hasLiquidLayer(stack) ? countTopLiquidLayersOrThrow(stack) : 0);
    };
    if (!useAdaptiveSplit || boundary_states.size() != expected_states(virtual_layers.layers)) {
        virtual_layers.layers = layers;
        virtual_layers.thicknesses = thicknesses;
        virtual_layers.original_layer_index.resize(layers.size());
        virtual_layers.splits_per_original_layer.assign(layers.size(), 1);
        for (size_t i = 0; i < layers.size(); ++i) {
            virtual_layers.original_layer_index[i] = static_cast<int>(i);
        }
    }

    int M = static_cast<int>(virtual_layers.layers.size());
    int usable_layers = std::min(M, static_cast<int>(boundary_states.size()) - 1);
    // propagateRayleighEigenstates repeats the state at the bottom of every
    // top liquid layer (the horizontal displacement jumps there), so the top
    // of layer i is entry 2i in the liquid stack and i+L below it.
    const int liquid_count = hasLiquidLayer(virtual_layers.layers)
        ? countTopLiquidLayersOrThrow(virtual_layers.layers) : 0;
    if (liquid_count > 0) {
        if (static_cast<int>(boundary_states.size()) != M + liquid_count + 1) {
            throw std::runtime_error(
                "Analytical Rayleigh energy integrals: boundary states do not "
                "match the fluid layer stack.");
        }
        usable_layers = M;
    }
    for (int layer_i = 0; layer_i < usable_layers; ++layer_i) {
        const auto& st = boundary_states[layer_i < liquid_count ? 2 * layer_i
                                                               : layer_i + liquid_count];
        T uz_top = st.uz.real(), szz_top = st.szz.real();
        T ur_top = st.ur.real(), szr_top = st.szr.real();

        int original_i = virtual_layers.original_layer_index[layer_i];
        auto sub_lint = computeRayleighLayerIntegrals_Analytical<T>(
            uz_top, szz_top, ur_top, szr_top,
            virtual_layers.layers[layer_i], virtual_layers.thicknesses[layer_i], omega, k);
        add_layer_integrals(res.layer_integrals[original_i], sub_lint);

        const auto& p = layers[original_i];
        if (isLiquidLayer(p)) {
            res.I1 += p.rho * sub_lint.int_uz2_ur2;
            res.I2 += sub_lint.int_szz2 / p.c33;
            res.I3 += -sub_lint.int_szz_ur / k;
            continue;
        }
        T A_coeff = p.c11 - p.c13 * p.c13 / p.c33;
        res.I1 += p.rho * sub_lint.int_uz2_ur2;
        res.I2 += sub_lint.int_szz2 / p.c33 + sub_lint.int_szr2 / p.c44 + k_sq * A_coeff * sub_lint.int_ur2;
        res.I3 += A_coeff * sub_lint.int_ur2 + (sub_lint.int_uz_szr - (p.c13 / p.c33) * sub_lint.int_szz_ur) / k;
    }

    // Halfspace analytical contribution (same as existing code)
    using CT = typename ComplexType<T>::type;
    WaveVariables<T> wave; wave.omega = omega; wave.k = k; wave.z = zero;
    DerivedComplexValues<T> dv = computeDerivedValues(halfSpaceParams, wave);

    T y1_0 = boundary_states.back().uz.real();
    T y3_0 = boundary_states.back().ur.real();

    CT eps1 = dv.eps1; CT eps3 = dv.eps3;
    CT det = eps1 - eps3;
    CT c1 = (toComplex(y3_0) - eps3 * toComplex(y1_0)) / det;
    CT c3 = (eps1 * toComplex(y1_0) - toComplex(y3_0)) / det;

    CT cs[2] = {c1, c3};
    CT nus[2] = {dv.nu1, dv.nu3};
    CT e_v[2] = {eps1, eps3};
    CT d_v[2] = {dv.d1, dv.d2};
    CT h_v[2] = {dv.d3 * toComplex(halfSpaceParams.c44), dv.d4 * toComplex(halfSpaceParams.c44)};

    T rho = halfSpaceParams.rho, C = halfSpaceParams.c33, L = halfSpaceParams.c44;
    T A_hs = halfSpaceParams.c11, F = halfSpaceParams.c13;
    T A_coeff_hs = A_hs - F * F / C;

    auto& hs_lint = res.layer_integrals[N];
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            CT w = (cs[i] * ad_conj(cs[j])) / (nus[i] + ad_conj(nus[j]));
            res.I1 = res.I1 + (w * rho * (T(1.0) + e_v[i] * ad_conj(e_v[j]))).real();
            res.I2 = res.I2 + (w * (d_v[i]*ad_conj(d_v[j])/C + h_v[i]*ad_conj(h_v[j])/L
                     + T(k_sq*A_coeff_hs)*e_v[i]*ad_conj(e_v[j]))).real();
            CT cross = (toComplex(T(1.0))*ad_conj(h_v[j]) - T(F/C)*d_v[i]*ad_conj(e_v[j])) / k;
            res.I3 = res.I3 + (w * (A_coeff_hs*e_v[i]*ad_conj(e_v[j]) + cross)).real();

            hs_lint.int_uz2_ur2 = hs_lint.int_uz2_ur2 + (w * (T(1.0) + e_v[i]*ad_conj(e_v[j]))).real();
            hs_lint.int_szz2 = hs_lint.int_szz2 + (w * d_v[i]*ad_conj(d_v[j])).real();
            hs_lint.int_szr2 = hs_lint.int_szr2 + (w * h_v[i]*ad_conj(h_v[j])).real();
            hs_lint.int_ur2 = hs_lint.int_ur2 + (w * e_v[i]*ad_conj(e_v[j])).real();
            hs_lint.int_uz_szr = hs_lint.int_uz_szr + (w * toComplex(T(1.0))*ad_conj(h_v[j])).real();
            hs_lint.int_szz_ur = hs_lint.int_szz_ur + (w * d_v[i]*ad_conj(e_v[j])).real();
        }
    }

    return res;
}

// CT-compatible Rayleigh energy integrals (trapezoidal + analytical halfspace)
template <typename T>
RayleighEnergyResult<T> computeRayleighEnergyIntegrals_CT(
    const std::vector<RayleighStatePoint<T>>& eigenfunctions,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega, T phaseVelocity,
    int /*numSubLayers*/ = 10
) {
    RayleighEnergyResult<T> res;
    T zero = T(0.0);
    res.I1 = zero; res.I2 = zero; res.I3 = zero;
    if (eigenfunctions.size() < 2) return res;

    T k = omega / phaseVelocity;
    T k_sq = k * k;
    int N = (int)layers.size();
    res.layer_integrals.resize(N + 1);  // +1 for halfspace
    for (int l = 0; l <= N; ++l) {
        res.layer_integrals[l] = {zero, zero, zero, zero, zero, zero};
    }

    std::vector<double> cumulative_bottoms(N);
    double dep = 0.0;
    for (int layer_i = 0; layer_i < N; ++layer_i) {
        dep += get_scalar_val(thicknesses[layer_i]);
        cumulative_bottoms[layer_i] = dep;
    }

    int layer_i = 0;
    for (size_t pt_idx = 0; pt_idx + 1 < eigenfunctions.size(); ++pt_idx) {
        const auto& p1 = eigenfunctions[pt_idx];
        const auto& p2 = eigenfunctions[pt_idx + 1];
        T dz = p2.depth - p1.depth;
        if (get_scalar_val(dz) <= 0.0) continue;

        T z_mid_t = (p1.depth + p2.depth) * T(0.5);
        double z_mid = get_val(z_mid_t);
        while (layer_i < N - 1 && z_mid > cumulative_bottoms[layer_i] + 1e-9) {
            ++layer_i;
        }

        auto& lint = res.layer_integrals[layer_i];
        T uz1 = p1.uz.real(), ur1 = p1.ur.real(), szz1 = p1.szz.real(), szr1 = p1.szr.real();
        T uz2 = p2.uz.real(), ur2 = p2.ur.real(), szz2 = p2.szz.real(), szr2 = p2.szr.real();

        T half = T(0.5);
        lint.int_uz2_ur2 += (uz1*uz1 + ur1*ur1 + uz2*uz2 + ur2*ur2) * half * dz;
        lint.int_szz2    += (szz1*szz1 + szz2*szz2) * half * dz;
        lint.int_szr2    += (szr1*szr1 + szr2*szr2) * half * dz;
        lint.int_ur2     += (ur1*ur1 + ur2*ur2) * half * dz;
        lint.int_uz_szr  += (uz1*szr1 + uz2*szr2) * half * dz;
        lint.int_szz_ur  += (szz1*ur1 + szz2*ur2) * half * dz;
    }

    for (int i = 0; i < N; ++i) {
        const auto& p = layers[i];
        const auto& lint = res.layer_integrals[i];
        if (isLiquidLayer(p)) {
            res.I1 += p.rho * lint.int_uz2_ur2;
            res.I2 += lint.int_szz2 / p.c33;
            res.I3 += -lint.int_szz_ur / k;
            continue;
        }
        T A_coeff = p.c11 - p.c13 * p.c13 / p.c33;
        res.I1 += p.rho * lint.int_uz2_ur2;
        res.I2 += lint.int_szz2 / p.c33 + lint.int_szr2 / p.c44 + k_sq * A_coeff * lint.int_ur2;
        res.I3 += A_coeff * lint.int_ur2 + (lint.int_uz_szr - (p.c13 / p.c33) * lint.int_szz_ur) / k;
    }

    // Halfspace analytical contribution
    WaveVariables<T> wave; wave.omega = omega; wave.k = k; wave.z = zero;
    DerivedComplexValues<T> dv = computeDerivedValues(halfSpaceParams, wave);

    using CT = typename ComplexType<T>::type;
    // Last eigenfunction point = halfspace entry
    T y1_0 = eigenfunctions.back().uz.real();
    T y3_0 = eigenfunctions.back().ur.real();

    CT eps1 = dv.eps1; CT eps3 = dv.eps3;
    CT det = eps1 - eps3;

    // Modal decomposition coefficients
    CT c1 = (toComplex(y3_0) - eps3 * toComplex(y1_0)) / det;
    CT c3 = (eps1 * toComplex(y1_0) - toComplex(y3_0)) / det;

    CT cs[2] = {c1, c3};
    CT nus[2] = {dv.nu1, dv.nu3};
    CT e_v[2] = {eps1, eps3};
    CT d_v[2] = {dv.d1, dv.d2};
    CT h_v[2] = {dv.d3 * toComplex(halfSpaceParams.c44), dv.d4 * toComplex(halfSpaceParams.c44)};

    T rho = halfSpaceParams.rho, C = halfSpaceParams.c33, L = halfSpaceParams.c44;
    T A_hs = halfSpaceParams.c11, F = halfSpaceParams.c13;
    T A_coeff_hs = A_hs - F * F / C;

    // Halfspace per-layer integrals (reset)
    auto& hs_lint = res.layer_integrals[N];

    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            CT w = (cs[i] * ad_conj(cs[j])) / (nus[i] + ad_conj(nus[j]));

            // I1: ρ(uz² + ur²) — uz~1, ur~eps
            res.I1 = res.I1 + (w * rho * (T(1.0) + e_v[i] * ad_conj(e_v[j]))).real();

            // I2
            res.I2 = res.I2 + (w * (d_v[i]*ad_conj(d_v[j])/C + h_v[i]*ad_conj(h_v[j])/L
                     + T(k_sq*A_coeff_hs)*e_v[i]*ad_conj(e_v[j]))).real();

            // I3
            CT cross = (toComplex(T(1.0))*ad_conj(h_v[j]) - T(F/C)*d_v[i]*ad_conj(e_v[j])) / k;
            res.I3 = res.I3 + (w * (A_coeff_hs*e_v[i]*ad_conj(e_v[j]) + cross)).real();

            // Halfspace per-component integrals for gradient
            hs_lint.int_uz2_ur2 = hs_lint.int_uz2_ur2 + (w * (T(1.0) + e_v[i]*ad_conj(e_v[j]))).real();
            hs_lint.int_szz2 = hs_lint.int_szz2 + (w * d_v[i]*ad_conj(d_v[j])).real();
            hs_lint.int_szr2 = hs_lint.int_szr2 + (w * h_v[i]*ad_conj(h_v[j])).real();
            hs_lint.int_ur2 = hs_lint.int_ur2 + (w * e_v[i]*ad_conj(e_v[j])).real();
            hs_lint.int_uz_szr = hs_lint.int_uz_szr + (w * toComplex(T(1.0))*ad_conj(h_v[j])).real();
            hs_lint.int_szz_ur = hs_lint.int_szz_ur + (w * d_v[i]*ad_conj(e_v[j])).real();
        }
    }

    return res;
}

// Variational gradient for Rayleigh wave (templated for AD)
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> computeVariationalGradientRayleigh_T(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    SensitivityMode mode,
    int numSubLayers = 10,
    bool useAnalytical = false,
    bool useAdaptiveSplit = true
) {
    RayleighEnergyResult<T> energy;
    if (useAnalytical) {
        // Propagate with nsub=1 (boundary states only) + analytical integrals
        auto boundary_pts = propagateRayleighEigenstates<T>(
            layers, thicknesses, halfSpace, omega, cp, 1, useAdaptiveSplit);
        energy = computeRayleighEnergyIntegrals_Analytical<T>(
            boundary_pts, layers, thicknesses, halfSpace, omega, cp, useAdaptiveSplit);
    } else {
        auto eigen_pts = propagateRayleighEigenstates<T>(
            layers, thicknesses, halfSpace, omega, cp, numSubLayers, useAdaptiveSplit);
        energy = computeRayleighEnergyIntegrals_CT<T>(
            eigen_pts, layers, thicknesses, halfSpace, omega, cp, numSubLayers);
    }

    T k = omega / cp;
    T k2 = k * k;
    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int N = (int)layers.size();
    int num_params = (N + 1) * npl;
    Eigen::Matrix<T, Eigen::Dynamic, 1> gradient(num_params);
    for (int i = 0; i < num_params; ++i) gradient(i) = T(0.0);

    T coeff = cp * cp * cp / (T(2.0) * omega * omega * energy.I3);

    // Process each layer + halfspace
    for (int j = 0; j <= N; ++j) {
        const auto& lint = energy.layer_integrals[j];
        const auto& p = (j < N) ? layers[j] : halfSpace;

        if (isLiquidLayer(p)) {
            T K_rho = -omega * omega * lint.int_uz2_ur2;
            T C_j = p.c33;
            T K_C = lint.int_szz2 / (C_j * C_j);

            if (mode == SensitivityMode::Isotropic) {
                T rho_j = p.rho;
                T Vp_j = sqrt(C_j / rho_j);
                gradient[j*3 + 0] = T(0.0);
                gradient[j*3 + 1] = coeff * (T(2.0) * rho_j * Vp_j * K_C);
                gradient[j*3 + 2] = coeff * (K_rho + Vp_j * Vp_j * K_C);
            } else {
                // VTI row order: [Rho, A, C, F, L, N]. Acoustic fluid layers
                // depend only on density and compressional modulus C.
                gradient[j*6 + 0] = coeff * K_rho;
                gradient[j*6 + 1] = T(0.0);
                gradient[j*6 + 2] = coeff * K_C;
                gradient[j*6 + 3] = T(0.0);
                gradient[j*6 + 4] = T(0.0);
                gradient[j*6 + 5] = T(0.0);
            }
            continue;
        }

        // Displacement-frozen VTI sensitivity kernels (T&S convention)
        // Derived from I₂ = C(y₁'+ky₃)² - 2Fky₃(y₁'+ky₃) + L(y₃'+ky₁)² + k²Ay₃²
        // where (y₁'+ky₃) = (y₂+Fky₃)/C, (y₃'+ky₁) = y₄/L
        T K_rho = -omega * omega * lint.int_uz2_ur2;
        T K_A = k2 * lint.int_ur2;
        T C_j = p.c33, F_j = p.c13, L_j = p.c44;
        T K_C = (lint.int_szz2 + T(2.0) * F_j * k * lint.int_szz_ur
                + F_j * F_j * k2 * lint.int_ur2) / (C_j * C_j);
        T K_F = (T(-2.0) * k / C_j)
                * (lint.int_szz_ur + F_j * k * lint.int_ur2);
        T K_L = lint.int_szr2 / (L_j * L_j);

        if (mode == SensitivityMode::Isotropic) {
            T rho_j = p.rho;
            T Vs_j = sqrt(L_j / rho_j);
            T Vp_j = sqrt(C_j / rho_j);

            // Chain rule: A=C=ρVp², F=ρ(Vp²-2Vs²), L=ρVs²
            // dL/dVs = 2ρVs, dF/dVs = -4ρVs
            // dA/dVp = dC/dVp = 2ρVp, dF/dVp = 2ρVp
            // dA/dρ = dC/dρ = Vp², dF/dρ = Vp²-2Vs², dL/dρ = Vs²
            T g_Vs = coeff * (T(2.0) * rho_j * Vs_j * K_L + T(-4.0) * rho_j * Vs_j * K_F);
            T g_Vp = coeff * (T(2.0) * rho_j * Vp_j * (K_A + K_C + K_F));
            T g_rho = coeff * (K_rho + Vp_j*Vp_j*(K_A + K_C) + (Vp_j*Vp_j - T(2.0)*Vs_j*Vs_j)*K_F + Vs_j*Vs_j*K_L);

            gradient[j*3 + 0] = g_Vs;
            gradient[j*3 + 1] = g_Vp;
            gradient[j*3 + 2] = g_rho;
        } else {
            // VTI: [Rho, A, C, F, L, N]
            gradient[j*6 + 0] = coeff * K_rho;
            gradient[j*6 + 1] = coeff * K_A;
            gradient[j*6 + 2] = coeff * K_C;
            gradient[j*6 + 3] = coeff * K_F;
            gradient[j*6 + 4] = coeff * K_L;
            gradient[j*6 + 5] = T(0.0);  // N doesn't affect Rayleigh
        }
    }

    return gradient;
}

// ---------------------------------------------------------------------------
// Variational boundary kernel K_d (Rayleigh / Love)
//
// Formulation: docs/depth_sensitivity_kernels.tex, sections "The variational
// boundary kernel K_d" and "K_d must be summed over every boundary that moves".
//
// Returns the JUMP [K_d]_b at every boundary of the stack, b = 1..N, where
// boundary b separates layer b-1 from layer b (and b = N separates the last
// layer from the halfspace). Index b matches point b of
// propagateRayleighEigenstates(..., numSubLayers=1, useAdaptiveSplit=false),
// which emits the free surface at index 0 and one point per layer bottom.
//
// These are per-BOUNDARY quantities on purpose. A geometric derivative is
//
//     dc/dq = sum_b [K_d]_b * dz_b/dq
//
// and the weights dz_b/dq -- not the kernel -- are what distinguish an
// interface-depth derivative from a layer-thickness one. Leaving them to the
// caller keeps this function independent of the parameterization; supplying
// them here is what made the March 2026 implementation thickness-only.
//
// The eigenfunction products are formed as complex products reduced by
// .real(), matching computeRayleighEnergyIntegrals_CT. Terms carrying no
// material parameter cancel identically in the jump (see the docs), so the
// -2k y1 y4 coefficient is inert here.
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> computeBoundaryKernelsRayleigh_T(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp,
    bool useAnalytical = false,
    int numSubLayers = 10,
    bool useAdaptiveSplit = false
) {
    using CT = typename ComplexType<T>::type;
    const int N = (int)layers.size();

    auto pts = propagateRayleighEigenstates<T>(
        layers, thicknesses, halfSpace, omega, cp, 1,
        /*useAdaptiveSplit=*/false);
    RayleighEnergyResult<T> energy;
    if (useAnalytical) {
        energy = computeRayleighEnergyIntegrals_Analytical<T>(
            pts, layers, thicknesses, halfSpace, omega, cp, useAdaptiveSplit);
    } else {
        auto dense = propagateRayleighEigenstates<T>(
            layers, thicknesses, halfSpace, omega, cp, numSubLayers,
            useAdaptiveSplit);
        energy = computeRayleighEnergyIntegrals_CT<T>(
            dense, layers, thicknesses, halfSpace, omega, cp, numSubLayers);
    }

    const T k = omega / cp;
    const T lambda = cp * cp * cp / (T(2.0) * omega * omega * energy.I3);

    // Spatial Hamiltonian evaluated with one side's material at fixed
    // eigenfunction values (which are continuous across a welded boundary).
    // Every real coefficient is forced into a plain T before toComplex:
    // AutoDiffScalar arithmetic yields expression templates, and
    // std::complex<expression> will not multiply with std::complex<T>.
    auto Hz = [&](const PhysicalParameters<T>& p,
                  const CT& y1, const CT& y2, const CT& y3, const CT& y4) -> T {
        const T A = p.c11, C = p.c33, F = p.c13, L = p.c44;
        const T c_rho = p.rho * omega * omega;
        const T c_A   = k * k * (A - F * F / C);
        const T c_F   = T(2.0) * (F / C) * k;
        const T c_uw  = T(2.0) * k;
        CT acc = toComplex(c_rho) * (y1 * y1 + y3 * y3)
               - toComplex(c_A) * (y3 * y3)
               + (y2 * y2) / toComplex(C)
               + (y4 * y4) / toComplex(L)
               + toComplex(c_F) * (y3 * y2)
               - toComplex(c_uw) * (y1 * y4);
        return acc.real();
    };

    Eigen::Matrix<T, Eigen::Dynamic, 1> out(N);
    for (int b = 1; b <= N; ++b) {
        const auto& P = pts[std::min((size_t)b, pts.size() - 1)];
        const auto& above = layers[b - 1];
        const auto& below = (b < N) ? layers[b] : halfSpace;
        out(b - 1) = -lambda * (Hz(above, P.uz, P.szz, P.ur, P.szr)
                                - Hz(below, P.uz, P.szz, P.ur, P.szr));
    }
    return out;
}

// Love counterpart of computeBoundaryKernelsRayleigh_T. Same contract: one
// jump per stack boundary, weights left to the caller.
//
//   K_d = -Lambda ( rho w^2 y1^2 + y2^2/L - k^2 N y1^2 ),   y1 = u_y, y2 = tau_yz
//
// Note the PLUS on y2^2/L: this is the spatial-Hamiltonian form. The Lagrangian
// density T - V has a minus there, and for Love that sign error causes a
// near-total cancellation which suppressed the kernel by ~50x -- see
// docs/depth_sensitivity_kernels.tex. Love eigenstates are real, so no complex
// reduction is needed here.
// The eigenfunctions and the energy integrals MUST come from the same
// propagation. computeLoveEigenfunctions integrates upward from the halfspace;
// propagateLoveEigenstates integrates downward from a unit surface
// displacement, and the analytical Love integrals use yet another gauge. K_d is
// quadratic in the eigenfunction while the prefactor carries 1/I3, so a
// consistent gauge cancels exactly and an inconsistent one leaves a constant
// factor -- measured as 1.7533 when these were first mixed.
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> computeBoundaryKernelsLove_T(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp
) {
    const int N = (int)layers.size();
    auto ef = computeLoveEigenfunctions<T>(layers, thicknesses, halfSpace,
                                           omega, cp, 1);
    auto ei = computeLoveEnergyIntegrals<T>(ef, omega, cp, layers, thicknesses,
                                            halfSpace);
    const T lambda = cp * cp * cp / (T(2.0) * omega * omega * ei.I3);

    const T k = omega / cp;
    // Same expression-template caution as the Rayleigh version above.
    auto Hz = [&](const PhysicalParameters<T>& p,
                  const std::complex<T>& y1, const std::complex<T>& y2) -> T {
        const T c_rho = p.rho * omega * omega;
        const T L = p.c44;
        const T c_N = k * k * p.c66;
        std::complex<T> acc = toComplex(c_rho) * (y1 * y1)
                            + (y2 * y2) / toComplex(L)
                            - toComplex(c_N) * (y1 * y1);
        return acc.real();
    };

    Eigen::Matrix<T, Eigen::Dynamic, 1> out(N);
    for (int b = 1; b <= N; ++b) {
        const auto& P = ef[std::min((size_t)b, ef.size() - 1)];
        const auto& above = layers[b - 1];
        const auto& below = (b < N) ? layers[b] : halfSpace;
        out(b - 1) = -lambda * (Hz(above, P.uy, P.tyz) - Hz(below, P.uy, P.tyz));
    }
    return out;
}

// Rayleigh Weak-Form Hessian (1st-order AD via autodiff::dual)
inline HessianResult computePhaseVelocityHessianWeakFormRayleigh(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool /*computeThickness*/ = false,
    int numSubLayers = 10,
    bool useAnalytical = false,
    WFHessianBackend backend = WFHessianBackend::Dual
) {
    using adual = autodiff::dual;
    HessianResult result;
    const double omega = 2.0 * M_PI * freq;

    // Newton polish (same as Full AD version)
    auto f_double = [&](double cv) {
        return calculateDispersionFunction<double>(layers, thicknesses, halfSpace,
            omega, cv, WaveType::Rayleigh, NormalizationMode::VectorNorm)(I24);
    };
    double c_root = root_c_double;
    for(int it=0; it<15; ++it) {
        double f0 = f_double(c_root), eps = c_root * 1e-7;
        double df = (f_double(c_root + eps) - f0) / eps;
        if(std::abs(df) < 1e-30) break;
        double step = f0 / df;
        c_root -= step;
        if(std::abs(step) < 1e-11) break;
    }

    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = ((int)layers.size() + 1) * npl;

    // Step 1: Reference gradient with double
    auto grad_dbl = computeVariationalGradientRayleigh_T<double>(
        layers, thicknesses, halfSpace, omega, c_root, mode, numSubLayers, useAnalytical);
    Eigen::VectorXd g0(num_params);
    for (int i = 0; i < num_params; ++i) g0(i) = grad_dbl(i);

    // Step 2: AD pass(es) for the Hessian.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_params, num_params);

    if (backend != WFHessianBackend::Dual) {
        // Vectorized forward AD — see computePhaseVelocityHessianWeakFormLove
        // for why one blocked pass reproduces a whole block of Hessian columns.
        const int width = (backend == WFHessianBackend::VectorDyn)
                              ? num_params : WF_HESSIAN_CHUNK;

        auto run_block = [&](auto tag, int j0) {
            using T = typename decltype(tag)::type;
            std::vector<PhysicalParameters<T>> v_layers;
            PhysicalParameters<T> v_hs;
            seedLayerCakeParamBlock<T>(layers, halfSpace, mode, j0, width, v_layers, v_hs);

            std::vector<T> v_thick(thicknesses.size());
            for (size_t i = 0; i < thicknesses.size(); ++i)
                v_thick[i] = make_ad_seeded<T>(thicknesses[i], width);

            T v_omega = make_ad_seeded<T>(omega, width);
            Eigen::VectorXd cp_seed = Eigen::VectorXd::Zero(width);
            for (int s = 0; s < width && j0 + s < num_params; ++s) cp_seed(s) = g0(j0 + s);
            T v_cp = make_ad_grad<T>(c_root, cp_seed);

            Eigen::Matrix<T, Eigen::Dynamic, 1> g_ad = computeVariationalGradientRayleigh_T<T>(
                v_layers, v_thick, v_hs, v_omega, v_cp, mode, numSubLayers, useAnalytical);

            extractHessianBlock<T>(g_ad, j0, width, H);
        };

        if (backend == WFHessianBackend::VectorDyn) {
            run_block(TypeTag<ADVec>{}, 0);
        } else {
            for (int j0 = 0; j0 < num_params; j0 += width)
                run_block(TypeTag<ADChunk<WF_HESSIAN_CHUNK>>{}, j0);
        }
    } else {
    std::vector<PhysicalParameters<adual>> d_layers_base(layers.size());
    std::vector<adual> d_thick_base(thicknesses.size());
    for (size_t i = 0; i < layers.size(); ++i) d_layers_base[i] = to_dual(layers[i]);
    for (size_t i = 0; i < thicknesses.size(); ++i) d_thick_base[i] = adual(thicknesses[i]);
    PhysicalParameters<adual> d_hs_base = to_dual(halfSpace);

    // Work buffers reused across passes to avoid repeated allocations.
    std::vector<PhysicalParameters<adual>> d_layers = d_layers_base;
    std::vector<adual> d_thick = d_thick_base;
    PhysicalParameters<adual> d_hs = d_hs_base;

    for (int j = 0; j < num_params; ++j) {
        // Reset to base dual parameters (zero derivatives)
        d_layers = d_layers_base;
        d_thick = d_thick_base;
        d_hs = d_hs_base;

        // Seed parameter j
        int layer_idx = j / npl;
        int param_in_layer = j % npl;
        bool is_hs = (layer_idx >= (int)layers.size());
        PhysicalParameters<adual>& target = is_hs ? d_hs : d_layers[layer_idx];

        if (mode == SensitivityMode::Isotropic) {
            double vs_val = is_hs ? halfSpace.vs() : layers[layer_idx].vs();
            double vp_val = is_hs ? halfSpace.vp() : layers[layer_idx].vp();
            double rho_val = get_scalar_val(is_hs ? halfSpace.rho : layers[layer_idx].rho);
            adual vs_d(vs_val), vp_d(vp_val), rho_d(rho_val);

            if (param_in_layer == 0) vs_d.grad = 1.0;
            else if (param_in_layer == 1) vp_d.grad = 1.0;
            else rho_d.grad = 1.0;

            auto pp = createIsotropicParameters<adual>(vs_d, vp_d, rho_d);
            target.rho = pp.rho; target.c11 = pp.c11; target.c13 = pp.c13;
            target.c33 = pp.c33; target.c44 = pp.c44; target.c66 = pp.c66;
        } else {
            switch (param_in_layer) {
                case 0: target.rho.grad = 1.0; break;
                case 1: target.c11.grad = 1.0; break;
                case 2: target.c33.grad = 1.0; break;
                case 3: target.c13.grad = 1.0; break;
                case 4: target.c44.grad = 1.0; break;
                case 5: target.c66.grad = 1.0; break;
            }
        }

        // Seed cp with derivative = g_j
        adual d_omega(omega);
        adual d_cp = make_dual(c_root, g0(j));

        // Compute gradient as dual values
        auto g_dual = computeVariationalGradientRayleigh_T<adual>(
            d_layers, d_thick, d_hs, d_omega, d_cp, mode, numSubLayers, useAnalytical);

        // Extract derivatives → H column j
        for (int i = 0; i < num_params; ++i)
            H(i, j) = g_dual(i).grad;
    }
    }

    // Symmetrize
    H = ((H + H.transpose()) * 0.5).eval();

    result.gradient = g0;
    result.hessian = H;

    // Parameter names
    if (mode == SensitivityMode::Isotropic) {
        std::string names[] = {"Vs", "Vp", "Rho"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    } else {
        std::string names[] = {"Rho","A","C","F","L","N"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    }
    return result;
}

// ===========================================================================
// Variational Gradient + FD Hessian (Love Wave)
// ===========================================================================
// Computes phase velocity Hessian using variational gradient + finite
// differences. The variational gradient uses the Rayleigh quotient
// stationarity principle (exact for gradient, no 2nd-order AD needed):
//   dc/dp = c³/(2ω²I₃) · (∂I₂/∂p - ω²∂I₁/∂p)  at frozen eigenfunction
// The Hessian is assembled from central FD of the gradient, with
// eigenfunctions recomputed for each perturbation.
// Cost: O(NL) where N = number of parameters, L = per-model cost.
// ===========================================================================

// Find Love wave root near a known value using Newton method on the
// dispersion function. Used internally by the Weak-Form Hessian to
// track the root across parameter perturbations.
inline double findLoveRootNear(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega, double c_guess
) {
    double c_max = halfSpace.vs() * (1.0 - 1e-6);
    double c = std::min(c_guess, c_max);
    for (int it = 0; it < 20; ++it) {
        auto Y = calculateLoveDispersion<double>(layers, thicknesses, halfSpace,
            omega, c, NormalizationMode::None);
        double f = Y(I24);
        if (std::isnan(f) || std::isinf(f)) break;
        double eps = c * 1e-8;
        auto Ye = calculateLoveDispersion<double>(layers, thicknesses, halfSpace,
            omega, c + eps, NormalizationMode::None);
        double df = (Ye(I24) - f) / eps;
        if (std::abs(df) < 1e-30 || std::isnan(df)) break;
        double step = f / df;
        double max_step = 0.1 * (c_max - 1e-3);
        if (std::abs(step) > max_step) step = std::copysign(max_step, step);
        c = std::min(c_max, std::max(1e-3, c - step));
        if (std::abs(step) < 1e-13) break;
    }
    return c;
}

// Compute variational gradient for Love wave (all double, no AD).
// Uses eigenfunctions at layer boundaries + analytical integration (cosh/sinh).
// Rayleigh quotient stationarity guarantees gradient is exact
// even with frozen (numerically computed) eigenfunctions.
// No sublayer overhead: propagates once through layers, integrates analytically.
inline Eigen::VectorXd computeVariationalGradientLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega, double cp,
    SensitivityMode mode,
    int /*numSubLayers*/ = 200
) {
    using CT = std::complex<double>;
    // Get eigenfunctions at layer boundaries only (numSubLayers=1)
    auto ef = computeLoveEigenfunctions<double>(layers, thicknesses, halfSpace,
        omega, cp, 1);

    double k = omega / cp;
    double k2 = k * k;

    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = ((int)layers.size() + 1) * npl;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);

    // Analytical per-layer integrals using cosh/sinh decomposition.
    // Within each layer the eigenfunction is:
    //   u(z) = c_up exp(+νz) + c_dn exp(-νz)
    // where ν² = (N·k² − ρ·ω²)/L and c_up, c_dn from boundary state.
    double I1_total = 0.0, I2_total = 0.0, I3_total = 0.0;
    struct LayerIntegral { double int_u2; double int_dudz2; };
    std::vector<LayerIntegral> layer_integrals(layers.size());

    // ef[0] = surface (top of layer 0), ef[1] = bottom of layer 0 = top of layer 1, ...
    for (size_t j = 0; j < layers.size(); ++j) {
        double L_j = layers[j].c44;
        double N_j = layers[j].c66;
        double rho_j = layers[j].rho;
        double h = thicknesses[j];

        CT nu_sq = CT((N_j * k2 - rho_j * omega * omega) / L_j);
        CT nu = std::sqrt(nu_sq);
        if (nu.real() < 0) nu = -nu;

        // State at top of layer j
        CT uy_top = CT(ef[j].uy);
        CT tyz_top = CT(ef[j].tyz);

        // Decompose into up/down modes: u(z) = c_up exp(+νz) + c_dn exp(-νz)
        // The SH matrix convention: τ = L·∂u/∂z_up = -L·∂u/∂z_depth
        // So: u(0) = c_up + c_dn = uy_top
        //     τ(0) = Lν(-c_up + c_dn) = tyz_top
        // → c_up = (u - τ/(Lν))/2, c_dn = (u + τ/(Lν))/2
        CT Lnu = CT(L_j) * nu;
        CT c_up, c_dn;
        if (std::abs(Lnu) > 1e-30) {
            c_up = 0.5 * (uy_top - tyz_top / Lnu);
            c_dn = 0.5 * (uy_top + tyz_top / Lnu);
        } else {
            c_up = uy_top * 0.5;
            c_dn = uy_top * 0.5;
        }

        CT nus[2] = {nu, -nu};
        CT coeffs[2] = {c_up, c_dn};
        CT tyz_facs[2] = {Lnu, -Lnu};

        // Integrate u² and (du/dz)² analytically
        // u(z) = Σ c_m exp(ν_m z), τ(z) = Σ c_m·(L·ν_m)·exp(ν_m z)
        // du/dz = τ/L = Σ c_m·ν_m·exp(ν_m z)
        // ∫u² = ΣΣ c_m c_n ∫exp((ν_m+ν_n)z)dz  from 0 to h
        // ∫(du/dz)² = ΣΣ c_m c_n ν_m ν_n ∫exp((ν_m+ν_n)z)dz
        CT int_uu_c = CT(0.0), int_dudz2_c = CT(0.0);
        for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 2; ++n) {
                CT nu_sum = nus[m] + nus[n];
                CT exp_int;
                if (std::abs(nu_sum * h) < 1e-10) {
                    exp_int = CT(h) * (1.0 + nu_sum * CT(h) * 0.5);
                } else {
                    exp_int = (std::exp(nu_sum * h) - 1.0) / nu_sum;
                }
                int_uu_c += coeffs[m] * coeffs[n] * exp_int;
                int_dudz2_c += coeffs[m] * coeffs[n] * nus[m] * nus[n] * exp_int;
            }
        }
        double int_u2 = int_uu_c.real();
        double int_dudz2 = int_dudz2_c.real();
        layer_integrals[j] = {int_u2, int_dudz2};

        I1_total += rho_j * int_u2;
        I2_total += (int_dudz2 * L_j + k2 * N_j * int_u2);
        I3_total += N_j * int_u2;
    }

    // Halfspace (analytical exponential decay)
    double u0 = ef.back().uy.real();
    double L_h = halfSpace.c44, N_h = halfSpace.c66, rho_h = halfSpace.rho;
    double nu_sq_hs = (N_h * k2 - rho_h * omega * omega) / L_h;
    CT nu_hs = std::sqrt(CT(nu_sq_hs));
    if (nu_hs.real() < 0) nu_hs = -nu_hs;
    double int_u2_hs = 0.0, int_dudz2_hs = 0.0;
    if (nu_hs.real() > 1e-10) {
        double inv2nu = 1.0 / (2.0 * nu_hs.real());
        int_u2_hs = u0 * u0 * inv2nu;
        int_dudz2_hs = std::norm(nu_hs) * u0 * u0 * inv2nu;
        I1_total += rho_h * int_u2_hs;
        I2_total += (int_dudz2_hs * L_h + k2 * N_h * int_u2_hs);
        I3_total += N_h * int_u2_hs;
    }

    // Gradient: dc/dp = c³/(2ω²I₃) · (∂I₂/∂p − ω²∂I₁/∂p)
    double coeff = cp * cp * cp / (2.0 * omega * omega * I3_total);

    for (size_t j = 0; j < layers.size(); ++j) {
        double rho_j = layers[j].rho, L_j = layers[j].c44, N_j = layers[j].c66;
        double dc_drho = coeff * (-omega * omega * layer_integrals[j].int_u2);
        double dc_dL   = coeff * layer_integrals[j].int_dudz2;
        double dc_dN   = coeff * k2 * layer_integrals[j].int_u2;

        if (mode == SensitivityMode::Isotropic) {
            double Vs_j = std::sqrt(L_j / rho_j);
            gradient[j*3 + 0] = (dc_dL + dc_dN) * 2.0 * rho_j * Vs_j;
            gradient[j*3 + 1] = 0.0;
            gradient[j*3 + 2] = dc_drho + (dc_dL + dc_dN) * Vs_j * Vs_j;
        } else {
            gradient[j*6 + 0] = dc_drho;
            gradient[j*6 + 4] = dc_dL;
            gradient[j*6 + 5] = dc_dN;
        }
    }

    // Halfspace gradient
    int hs_base = (int)layers.size() * npl;
    if (nu_hs.real() > 1e-10) {
        double dc_drho_h = coeff * (-omega * omega * int_u2_hs);
        double dc_dL_h   = coeff * int_dudz2_hs;
        double dc_dN_h   = coeff * k2 * int_u2_hs;

        if (mode == SensitivityMode::Isotropic) {
            double Vs_h = std::sqrt(L_h / rho_h);
            gradient[hs_base + 0] = (dc_dL_h + dc_dN_h) * 2.0 * rho_h * Vs_h;
            gradient[hs_base + 1] = 0.0;
            gradient[hs_base + 2] = dc_drho_h + (dc_dL_h + dc_dN_h) * Vs_h * Vs_h;
        } else {
            gradient[hs_base + 0] = dc_drho_h;
            gradient[hs_base + 4] = dc_dL_h;
            gradient[hs_base + 5] = dc_dN_h;
        }
    }

    return gradient;
}

// Perturb a model parameter by eps. Returns modified layer/halfspace copies.
// For Isotropic mode: param order is [Vs, Vp, Rho] per layer.
// For VTI mode: param order is [Rho, A, C, F, L, N] per layer.
inline void perturbLoveModel(
    const std::vector<PhysicalParameters<double>>& layers,
    const PhysicalParameters<double>& halfSpace,
    int param_idx, double eps,
    SensitivityMode mode,
    std::vector<PhysicalParameters<double>>& out_layers,
    PhysicalParameters<double>& out_hs
) {
    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    out_layers = layers;
    out_hs = halfSpace;
    int layer_idx = param_idx / npl;
    int param_in_layer = param_idx % npl;
    bool is_hs = (layer_idx >= (int)layers.size());
    PhysicalParameters<double>& target = is_hs ? out_hs : out_layers[layer_idx];

    if (mode == SensitivityMode::Isotropic) {
        double vs = target.vs(), vp = target.vp(), rho = get_scalar_val(target.rho);
        switch (param_in_layer) {
            case 0: vs  += eps; break;
            case 1: vp  += eps; break;
            case 2: rho += eps; break;
        }
        target = createIsotropicParameters<double>(vs, vp, rho);
    } else {
        switch (param_in_layer) {
            case 0: target.rho += eps; break;
            case 1: target.c11 += eps; break;
            case 2: target.c33 += eps; break;
            case 3: target.c13 += eps; break;
            case 4: target.c44 += eps; break;
            case 5: target.c66 += eps; break;
        }
    }
}

// Main Weak-Form Hessian for Love wave.
// Gradient: exact variational formula (Rayleigh quotient stationarity).
// Hessian: central FD of gradient with re-computed eigenfunctions per perturbation.
inline HessianResult computePhaseVelocityHessianVariationalFDLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool computeThickness = false,
    int numSubLayers = 200,
    double fdEps = 1e-5
) {
    HessianResult result;
    const double omega = 2.0 * M_PI * freq;

    // Newton polish for the reference root
    double c_root = findLoveRootNear(layers, thicknesses, halfSpace, omega, root_c_double);

    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_params = ((int)layers.size() + 1) * npl;

    // Reference gradient
    Eigen::VectorXd g0 = computeVariationalGradientLove(
        layers, thicknesses, halfSpace, omega, c_root, mode, numSubLayers);

    // Hessian via central FD of gradient
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_params, num_params);

    for (int j = 0; j < num_params; ++j) {
        std::vector<PhysicalParameters<double>> lp, lm;
        PhysicalParameters<double> hp, hm;
        perturbLoveModel(layers, halfSpace, j, fdEps, mode, lp, hp);
        perturbLoveModel(layers, halfSpace, j, -fdEps, mode, lm, hm);

        double cp_p = findLoveRootNear(lp, thicknesses, hp, omega, c_root);
        double cp_m = findLoveRootNear(lm, thicknesses, hm, omega, c_root);

        Eigen::VectorXd gp = computeVariationalGradientLove(
            lp, thicknesses, hp, omega, cp_p, mode, numSubLayers);
        Eigen::VectorXd gm = computeVariationalGradientLove(
            lm, thicknesses, hm, omega, cp_m, mode, numSubLayers);

        H.col(j) = (gp - gm) / (2.0 * fdEps);
    }

    // Symmetrize
    H = ((H + H.transpose()) * 0.5).eval();

    result.gradient = g0;
    result.hessian = H;

    // Parameter names
    if (mode == SensitivityMode::Isotropic) {
        std::string names[] = {"Vs", "Vp", "Rho"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    } else {
        std::string names[] = {"Rho","A","C","F","L","N"};
        for(size_t i=0; i<layers.size(); ++i)
            for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(i); }
        for(auto& n : names) { result.parameterNames.push_back(n); result.layerIndices.push_back(-1); }
    }
    return result;
}

// Result of one width-1 frequency-AD pass over a fixed layered stack.  The
// material vectors use the internal VTI order [rho,A,C,F,L,N] for every finite
// layer followed by the halfspace; boundary vectors contain one entry per
// finite-layer bottom.
struct FrequencyVariationalGroupKernelLevelResult {
    double group_velocity;
    Eigen::VectorXd phase_material_vti;
    Eigen::VectorXd group_material_vti;
    Eigen::VectorXd phase_boundary;
    Eigen::VectorXd group_boundary;
};

inline FrequencyVariationalGroupKernelLevelResult
computeGroupVelocityKernelFrequencyVariationalLevel(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    double omega,
    double phase_velocity,
    WaveType type,
    bool compute_boundary = false,
    int rayleigh_num_sub_layers = 1,
    bool rayleigh_use_analytical = false,
    bool rayleigh_use_adaptive_split = false
);

inline Eigen::VectorXd mapFixedStackVtiKernelToMode(
    const std::vector<PhysicalParameters<double>>& layers,
    const PhysicalParameters<double>& halfspace,
    const Eigen::VectorXd& kernel_vti,
    SensitivityMode mode
) {
    const int n_material = static_cast<int>(layers.size()) + 1;
    if (kernel_vti.size() != n_material * 6) {
        throw std::runtime_error("fixed-stack VTI kernel size mismatch");
    }
    if (mode == SensitivityMode::VTI) return kernel_vti;

    Eigen::VectorXd out = Eigen::VectorXd::Zero(n_material * 3);
    for (int i = 0; i < n_material; ++i) {
        const auto& p = (i < static_cast<int>(layers.size()))
            ? layers[i] : halfspace;
        const double rho = p.rho;
        const double vs = p.vs();
        const double vp = p.vp();
        const double grho = kernel_vti(i * 6 + 0);
        const double gA = kernel_vti(i * 6 + 1);
        const double gC = kernel_vti(i * 6 + 2);
        const double gF = kernel_vti(i * 6 + 3);
        const double gL = kernel_vti(i * 6 + 4);
        const double gN = kernel_vti(i * 6 + 5);
        out(i * 3 + 0) = 2.0 * rho * vs * (gL + gN - 2.0 * gF);
        out(i * 3 + 1) = 2.0 * rho * vp * (gA + gC + gF);
        out(i * 3 + 2) = grho + (gA + gC) * vp * vp
                       + gF * (vp * vp - 2.0 * vs * vs)
                       + (gL + gN) * vs * vs;
    }
    return out;
}

inline Eigen::VectorXd mapBoundaryKernelToLayerCakeThickness(
    const Eigen::VectorXd& boundary_kernel
) {
    // h_j moves boundary j+1 and every deeper boundary by the same amount.
    // Therefore dQ/dh_j is the suffix sum of the independent-boundary kernel.
    Eigen::VectorXd out = Eigen::VectorXd::Zero(boundary_kernel.size());
    double suffix = 0.0;
    for (int j = boundary_kernel.size() - 1; j >= 0; --j) {
        suffix += boundary_kernel(j);
        out(j) = suffix;
    }
    return out;
}

inline std::vector<SensitivityKernel> computeSensitivityKernelsWeakFormLove(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool computeThickness = false,
    const std::string& uKernelMode = "variational_frequency_ad",
    bool computeC = true,
    bool computeU = true,
    bool computeHV = true
) {
    using adual = autodiff::dual;
    if (uKernelMode != "variational_frequency_ad" &&
        uKernelMode != "variational") {
        throw std::invalid_argument(
            "LAYER_CAKE u_kernel_mode must be \"variational_frequency_ad\" "
            "or \"variational\"");
    }
    const double omega = 2.0 * M_PI * freq;
    double c_root = findLoveRootNear(layers, thicknesses, halfSpace, omega, root_c_double);

    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_layers = (int)layers.size();
    int num_param_vars = (num_layers + 1) * npl;
    int total_params = num_param_vars + (computeThickness ? num_layers : 0);

    // dc/dp reference
    Eigen::VectorXd dc_dp(total_params); dc_dp.setZero();
    Eigen::VectorXd g0_eigen;
    try {
        g0_eigen = computeVariationalGradientLove_T<double>(
            layers, thicknesses, halfSpace, omega, c_root, mode, true);
    } catch (const std::exception&) {
        g0_eigen = computeVariationalGradientLove_T<double>(
            layers, thicknesses, halfSpace, omega, c_root, mode, true, 1,
            LoveBranchPolicy::Auto, true);
    }
    if (!g0_eigen.array().isFinite().all()) {
        g0_eigen = computeVariationalGradientLove_T<double>(
            layers, thicknesses, halfSpace, omega, c_root, mode, true, 1,
            LoveBranchPolicy::Auto, true);
    }
    for(int i=0; i<num_param_vars; ++i) dc_dp(i) = g0_eigen(i);

    if (computeThickness) {
        using AD1 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;
        for(int i=0; i<num_layers; ++i) {
            auto f_ad1 = [&](AD1 cv, int h_idx) {
                std::vector<AD1> t_ad;
                for(int k=0; k<num_layers; ++k) {
                    AD1 val(thicknesses[k]);
                    if (k == h_idx) val.derivatives()(0) = 1.0;
                    t_ad.push_back(val);
                }
                auto wrap = [](double v) { AD1 a; a.value()=v; a.derivatives()(0)=0.0; return a; };
                PhysicalParameters<AD1> h_a; h_a.rho=wrap(halfSpace.rho); h_a.c11=wrap(halfSpace.c11); h_a.c33=wrap(halfSpace.c33); h_a.c44=wrap(halfSpace.c44); h_a.c66=wrap(halfSpace.c66); h_a.c13=wrap(halfSpace.c13);
                std::vector<PhysicalParameters<AD1>> l_ad;
                for(auto& p : layers) {
                    PhysicalParameters<AD1> pa; pa.rho=wrap(p.rho); pa.c11=wrap(p.c11); pa.c33=wrap(p.c33); pa.c44=wrap(p.c44); pa.c66=wrap(p.c66); pa.c13=wrap(p.c13);
                    l_ad.push_back(pa);
                }
                auto res = calculateDispersionFunction(l_ad, t_ad, h_a, AD1(omega), cv, WaveType::Love);
                return res(I24);
            };
            AD1 cv(c_root); cv.derivatives()(0) = 1.0;
            double dF_dc = f_ad1(cv, -1).derivatives()(0);
            cv.derivatives()(0) = 0.0;
            double dF_dh = f_ad1(cv, i).derivatives()(0);
            dc_dp(num_param_vars + i) = -dF_dh / dF_dc;
        }
    }

    Eigen::VectorXd frequency_du = Eigen::VectorXd::Zero(total_params);
    bool frequency_ready = false;
    if (computeU && uKernelMode == "variational_frequency_ad") {
        try {
            const auto frequency =
                computeGroupVelocityKernelFrequencyVariationalLevel(
                    layers, thicknesses, halfSpace, omega, c_root,
                    WaveType::Love, false);
            frequency_du.head(num_param_vars) =
                mapFixedStackVtiKernelToMode(
                    layers, halfSpace, frequency.group_material_vti, mode);
            frequency_ready = frequency_du.head(num_param_vars).allFinite();
        } catch (const std::exception&) {
            frequency_ready = false;
        }
    }

    std::vector<SensitivityKernel> kernels;
    if (!computeU || (frequency_ready && !computeThickness)) {
        kernels.reserve(total_params);
        for (int j = 0; j < total_params; ++j) {
            int layer_idx;
            std::string p_name;
            if (j >= num_param_vars) {
                layer_idx = j - num_param_vars;
                p_name = "Thickness";
            } else {
                const int material_idx = j / npl;
                const int param_idx = j % npl;
                layer_idx = material_idx >= num_layers ? -1 : material_idx;
                if (mode == SensitivityMode::Isotropic) {
                    const char* names[] = {"Vs", "Vp", "Rho"};
                    p_name = names[param_idx];
                } else {
                    const char* names[] = {"Rho", "A", "C", "F", "L", "N"};
                    p_name = names[param_idx];
                }
            }
            kernels.push_back({
                p_name, layer_idx,
                computeC ? dc_dp(j) : 0.0,
                computeU ? frequency_du(j) : 0.0,
                computeHV ? NAN : 0.0});
        }
        return kernels;
    }

    std::vector<PhysicalParameters<adual>> d_layers_base(num_layers);
    for (int i = 0; i < num_layers; ++i) d_layers_base[i] = to_dual(layers[i]);
    std::vector<adual> d_thick_base(num_layers);
    for (int i = 0; i < num_layers; ++i) d_thick_base[i] = adual(thicknesses[i]);
    PhysicalParameters<adual> d_hs_base = to_dual(halfSpace);

    for (int j = 0; j < total_params; ++j) {
        auto d_layers = d_layers_base; auto d_thick = d_thick_base; auto d_hs = d_hs_base;
        int layer_idx = -1; std::string p_name;
        bool is_t = (j >= num_param_vars);
        if (is_t) {
            int ti = j - num_param_vars; d_thick[ti].grad = 1.0; layer_idx = ti; p_name = "Thickness";
        } else {
            layer_idx = j / npl; int pinl = j % npl; bool is_hs = (layer_idx >= num_layers);
            PhysicalParameters<adual>& target = is_hs ? d_hs : d_layers[layer_idx];
            if (mode == SensitivityMode::Isotropic) {
                const char* names[] = {"Vs", "Vp", "Rho"}; p_name = names[pinl];
                double vs = is_hs ? halfSpace.vs() : layers[layer_idx].vs();
                double vp = is_hs ? halfSpace.vp() : layers[layer_idx].vp();
                double rho = is_hs ? halfSpace.rho : layers[layer_idx].rho;
                adual vs_d(vs), vp_d(vp), rho_d(rho);
                if (pinl == 0) vs_d.grad = 1.0; else if (pinl == 1) vp_d.grad = 1.0; else rho_d.grad = 1.0;
                target = createIsotropicParameters<adual>(vs_d, vp_d, rho_d);
            } else {
                const char* names[] = {"Rho", "A", "C", "F", "L", "N"}; p_name = names[pinl];
                if(pinl==0) target.rho.grad=1.0; else if(pinl==1) target.c11.grad=1.0;
                else if(pinl==2) target.c33.grad=1.0; else if(pinl==3) target.c13.grad=1.0;
                else if(pinl==4) target.c44.grad=1.0; else target.c66.grad=1.0;
            }
            if (is_hs) layer_idx = -1;
        }
        double du_value = 0.0;
        if (computeU && frequency_ready && !is_t) {
            du_value = frequency_du(j);
        } else if (computeU) {
            adual d_cp = make_dual(c_root, dc_dp(j));
            adual d_U;
            try {
                auto states = propagateLoveEigenstates<adual>(
                    d_layers, d_thick, adual(omega), d_cp);
                auto energy = computeLoveAnalyticalIntegrals<adual>(
                    states, d_layers, d_thick, d_hs, adual(omega), d_cp);
                d_U = energy.I3 / (d_cp * energy.I1);
            } catch (const std::exception&) {
                auto energy = computeLoveGroupEnergyDisper80<adual>(
                    d_layers, d_thick, d_hs, adual(omega), d_cp);
                d_U = energy.I3 / (d_cp * energy.I1);
            }
            if (!std::isfinite(get_val(d_U)) || !std::isfinite(d_U.grad)) {
                auto energy = computeLoveGroupEnergyDisper80<adual>(
                    d_layers, d_thick, d_hs, adual(omega), d_cp);
                d_U = energy.I3 / (d_cp * energy.I1);
            }
            du_value = d_U.grad;
        }
        kernels.push_back({
            p_name, layer_idx,
            computeC ? dc_dp(j) : 0.0,
            du_value,
            computeHV ? NAN : 0.0});
    }
    return kernels;
}

inline std::vector<SensitivityKernel> computeSensitivityKernelsWeakFormRayleigh(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq, double root_c_double,
    SensitivityMode mode = SensitivityMode::Isotropic,
    bool computeThickness = false,
    int numSubLayers = 10,
    bool useAnalytical = false,
    const std::string& uKernelMode = "variational_frequency_ad",
    bool computeC = true,
    bool computeU = true,
    bool computeHV = true
) {
    using adual = autodiff::dual;
    if (uKernelMode != "variational_frequency_ad" &&
        uKernelMode != "variational") {
        throw std::invalid_argument(
            "LAYER_CAKE u_kernel_mode must be \"variational_frequency_ad\" "
            "or \"variational\"");
    }
    const double omega = 2.0 * M_PI * freq;
    
    // Newton polish
    auto f_double = [&](double cv) {
        return calculateDispersionFunction<double>(layers, thicknesses, halfSpace,
            omega, cv, WaveType::Rayleigh, NormalizationMode::VectorNorm)(I24);
    };
    double c_root = root_c_double;
    for(int it=0; it<10; ++it) {
        double f0 = f_double(c_root), eps = c_root * 1e-7;
        double df = (f_double(c_root + eps) - f0) / eps;
        if(std::abs(df) < 1e-30) break;
        double step = f0 / df; c_root -= step;
        if(std::abs(step) < 1e-11) break;
    }

    int npl = (mode == SensitivityMode::Isotropic) ? 3 : 6;
    int num_layers = (int)layers.size();
    int num_param_vars = (num_layers + 1) * npl;
    int total_params = num_param_vars + (computeThickness ? num_layers : 0);

    // dc/dp reference
    Eigen::VectorXd dc_dp(total_params); dc_dp.setZero();
    auto g0_eigen = computeVariationalGradientRayleigh_T<double>(layers, thicknesses, halfSpace, omega, c_root, mode, numSubLayers, useAnalytical);
    for(int i=0; i<num_param_vars; ++i) dc_dp(i) = g0_eigen(i);

    if (computeThickness) {
        using AD1 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;
        for(int i=0; i<num_layers; ++i) {
            auto f_ad1 = [&](AD1 cv, int h_idx) {
                std::vector<AD1> t_ad;
                for(int k=0; k<num_layers; ++k) {
                    AD1 val(thicknesses[k]);
                    if (k == h_idx) val.derivatives()(0) = 1.0;
                    t_ad.push_back(val);
                }
                auto wrap = [](double v) { AD1 a; a.value()=v; a.derivatives()(0)=0.0; return a; };
                PhysicalParameters<AD1> h_a; h_a.rho=wrap(halfSpace.rho); h_a.c11=wrap(halfSpace.c11); h_a.c33=wrap(halfSpace.c33); h_a.c44=wrap(halfSpace.c44); h_a.c66=wrap(halfSpace.c66); h_a.c13=wrap(halfSpace.c13);
                std::vector<PhysicalParameters<AD1>> l_ad;
                for(auto& p : layers) {
                    PhysicalParameters<AD1> pa; pa.rho=wrap(p.rho); pa.c11=wrap(p.c11); pa.c33=wrap(p.c33); pa.c44=wrap(p.c44); pa.c66=wrap(p.c66); pa.c13=wrap(p.c13);
                    l_ad.push_back(pa);
                }
                auto res = calculateDispersionFunction(l_ad, t_ad, h_a, AD1(omega), cv, WaveType::Rayleigh, NormalizationMode::VectorNorm);
                return res(I24);
            };
            AD1 cv(c_root); cv.derivatives()(0) = 1.0;
            double dF_dc = f_ad1(cv, -1).derivatives()(0);
            cv.derivatives()(0) = 0.0;
            double dF_dh = f_ad1(cv, i).derivatives()(0);
            dc_dp(num_param_vars + i) = -dF_dh / dF_dc;
        }
    }

    Eigen::VectorXd frequency_du = Eigen::VectorXd::Zero(total_params);
    bool frequency_ready = false;
    if (computeU && uKernelMode == "variational_frequency_ad" &&
        useAnalytical) {
        try {
            const auto frequency =
                computeGroupVelocityKernelFrequencyVariationalLevel(
                    layers, thicknesses, halfSpace, omega, c_root,
                    WaveType::Rayleigh, computeThickness, numSubLayers,
                    useAnalytical, true);
            frequency_du.head(num_param_vars) =
                mapFixedStackVtiKernelToMode(
                    layers, halfSpace, frequency.group_material_vti, mode);
            if (computeThickness) {
                frequency_du.tail(num_layers) =
                    mapBoundaryKernelToLayerCakeThickness(
                        frequency.group_boundary);
            }
            frequency_ready = frequency_du.allFinite();
        } catch (const std::exception&) {
            frequency_ready = false;
        }
    }

    if (!computeHV && (!computeU || frequency_ready)) {
        std::vector<SensitivityKernel> direct;
        direct.reserve(total_params);
        for (int j = 0; j < total_params; ++j) {
            int layer_idx;
            std::string p_name;
            if (j >= num_param_vars) {
                layer_idx = j - num_param_vars;
                p_name = "Thickness";
            } else {
                const int material_idx = j / npl;
                const int param_idx = j % npl;
                layer_idx = material_idx >= num_layers ? -1 : material_idx;
                if (mode == SensitivityMode::Isotropic) {
                    const char* names[] = {"Vs", "Vp", "Rho"};
                    p_name = names[param_idx];
                } else {
                    const char* names[] = {"Rho", "A", "C", "F", "L", "N"};
                    p_name = names[param_idx];
                }
            }
            direct.push_back({
                p_name, layer_idx,
                computeC ? dc_dp(j) : 0.0,
                computeU ? frequency_du(j) : 0.0,
                0.0});
        }
        return direct;
    }

    std::vector<double> sFactors;
    calculateDispersionFunction<double>(layers, thicknesses, halfSpace, omega, c_root, WaveType::Rayleigh, NormalizationMode::VectorNorm, nullptr, &sFactors);

    // Fluid ellipticity needs the complete boundary/root derivative, including
    // fixed Cowling settings. The solid energy weak form omits those terms.
    const auto fluid_hv = computeHV && hasLiquidLayer(layers)
        ? computeSensitivityKernelsFullAD(layers, thicknesses, halfSpace, freq,
            c_root, WaveType::Rayleigh, mode, computeThickness)
        : std::vector<SensitivityKernel>{};

    std::vector<SensitivityKernel> kernels;
    std::vector<PhysicalParameters<adual>> d_layers_base(num_layers);
    for (int i = 0; i < num_layers; ++i) d_layers_base[i] = to_dual(layers[i]);
    std::vector<adual> d_thick_base(num_layers);
    for (int i = 0; i < num_layers; ++i) d_thick_base[i] = adual(thicknesses[i]);
    PhysicalParameters<adual> d_hs_base = to_dual(halfSpace);

    for (int j = 0; j < total_params; ++j) {
        auto d_layers = d_layers_base; auto d_thick = d_thick_base; auto d_hs = d_hs_base;
        int layer_idx = -1; std::string p_name;
        bool is_t = (j >= num_param_vars);
        if (is_t) {
            int ti = j - num_param_vars; d_thick[ti].grad = 1.0; layer_idx = ti; p_name = "Thickness";
        } else {
            layer_idx = j / npl; int pinl = j % npl; bool is_hs = (layer_idx >= num_layers);
            PhysicalParameters<adual>& target = is_hs ? d_hs : d_layers[layer_idx];
            if (mode == SensitivityMode::Isotropic) {
                const char* names[] = {"Vs", "Vp", "Rho"}; p_name = names[pinl];
                double vs = is_hs ? halfSpace.vs() : layers[layer_idx].vs();
                double vp = is_hs ? halfSpace.vp() : layers[layer_idx].vp();
                double rho = is_hs ? halfSpace.rho : layers[layer_idx].rho;
                adual vs_d(vs), vp_d(vp), rho_d(rho);
                if (pinl == 0) vs_d.grad = 1.0; else if (pinl == 1) vp_d.grad = 1.0; else rho_d.grad = 1.0;
                target = createIsotropicParameters<adual>(vs_d, vp_d, rho_d);
            } else {
                const char* names[] = {"Rho", "A", "C", "F", "L", "N"}; p_name = names[pinl];
                if(pinl==0) target.rho.grad=1.0; else if(pinl==1) target.c11.grad=1.0;
                else if(pinl==2) target.c33.grad=1.0; else if(pinl==3) target.c13.grad=1.0;
                else if(pinl==4) target.c44.grad=1.0; else target.c66.grad=1.0;
            }
            if (is_hs) layer_idx = -1;
        }
        adual d_cp = make_dual(c_root, dc_dp(j));
        double du_value = 0.0;
        if (computeU) {
            if (frequency_ready) {
                du_value = frequency_du(j);
            } else {
                auto eigen_pts = propagateRayleighEigenstates<adual>(
                    d_layers, d_thick, d_hs, adual(omega), d_cp,
                    useAnalytical ? 1 : numSubLayers);
                RayleighEnergyResult<adual> energy;
                if (useAnalytical) {
                    energy = computeRayleighEnergyIntegrals_Analytical<adual>(
                        eigen_pts, d_layers, d_thick, d_hs,
                        adual(omega), d_cp);
                } else {
                    energy = computeRayleighEnergyIntegrals_CT<adual>(
                        eigen_pts, d_layers, d_thick, d_hs,
                        adual(omega), d_cp, numSubLayers);
                }
                const adual d_U = energy.I3 / (d_cp * energy.I1);
                du_value = d_U.grad;
            }
        }

        double dhv_value = fluid_hv.empty() ? 0.0 : fluid_hv[j].d_HVRatio;
        if (computeHV && fluid_hv.empty()) {
            Eigen::Matrix<adual, 5, 1> seabedState;
            seabedState.setZero();
            size_t sIdx = 0;
            auto Y_surf = calculateDispersionFunction(
                d_layers, d_thick, d_hs, adual(omega), d_cp,
                WaveType::Rayleigh, NormalizationMode::Provided,
                &seabedState, &sFactors, &sIdx);
            const bool topIsLiquid = !layers.empty() &&
                isLiquidLayer(layers[0]);
            adual d_HV = topIsLiquid
                ? calculateHVRatio(seabedState)
                : calculateHVRatio(Y_surf);
            dhv_value = d_HV.grad;
        }
        kernels.push_back({
            p_name, layer_idx,
            computeC ? dc_dp(j) : 0.0,
            du_value, dhv_value});
    }
    return kernels;
}

// Variational group-velocity kernel for vector-AD (GRID/node-based) models:
// one flat vector-AD pass over U = I3 / (c * I1) with the phase velocity
// IFT-seeded by dc/dp. This is the vectorized equivalent of the per-parameter
// make_dual(c_root, dc_dp(j)) seeding in computeSensitivityKernelsWeakForm*:
// c carries its total derivative, so U evaluated in AD yields the total
// dU/dp = dU/dp|_c + (dU/dc) dc/dp in a single non-nested propagation.
// The eigenfunctions from upward propagation satisfy the surface boundary
// condition only AT the root of this mesh's dispersion function, and the
// energy-integral U is extremely sensitive to that consistency, so c_guess
// is Newton-polished to machine precision on the double mesh first (the
// layer_cake weak-form drivers do the same; an extrapolated Romberg c is off
// this mesh's root by O(h^2) and would corrupt dU/dp).
// Not valid for fluid layers with the analytical integrals (division by c44);
// callers must gate fluid models to the full-AD path.
template <typename ADS>  // ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>
Eigen::VectorXd computeGroupVelocityKernelVariational(
    const std::vector<PhysicalParameters<double>>& layers_dbl,
    const std::vector<double>& thicknesses_dbl,
    const PhysicalParameters<double>& halfspace_dbl,
    const std::vector<PhysicalParameters<ADS>>& layers_ad,
    const std::vector<ADS>& thicknesses_ad,
    const PhysicalParameters<ADS>& halfspace_ad,
    double omega, double c_guess,
    const Eigen::VectorXd& dc_dp,
    WaveType type,
    double* U_value_out = nullptr,
    bool use_analytical = true, int num_sub_layers = 10,
    bool polish_root = true
) {
    // Newton-polish to the root of THIS mesh's dispersion function
    // (mirrors computeSensitivityKernelsWeakFormRayleigh, sensitivity_utils.hpp).
    auto f_double = [&](double cv) {
        return calculateDispersionFunction<double>(
            layers_dbl, thicknesses_dbl, halfspace_dbl, omega, cv, type,
            NormalizationMode::VectorNorm)(I24);
    };
    double c_root = c_guess;
    if (polish_root) {
        for (int it = 0; it < 10; ++it) {
            double f0 = f_double(c_root), eps = c_root * 1e-7;
            double df = (f_double(c_root + eps) - f0) / eps;
            if (std::abs(df) < 1e-30) break;
            double step = f0 / df;
            c_root -= step;
            if (std::abs(step) < 1e-11) break;
        }
    }

    const int num_vars = static_cast<int>(dc_dp.size());
    ADS omega_ad;
    omega_ad.value() = omega;
    omega_ad.derivatives() = Eigen::VectorXd::Zero(num_vars);
    ADS c_ad;
    c_ad.value() = c_root;
    c_ad.derivatives() = dc_dp;

    ADS U_ad;
    if (type == WaveType::Love) {
        bool use_disper80 = false;
        try {
            auto energy = computeLoveAnalyticalIntegralsUpward<ADS>(
                layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad);
            U_ad = energy.I3 / (c_ad * energy.I1);
            use_disper80 = !std::isfinite(U_ad.value()) ||
                !U_ad.derivatives().array().isFinite().all();
        } catch (const std::exception&) {
            use_disper80 = true;
        }
        if (use_disper80) {
            auto energy = computeLoveGroupEnergyDisper80<ADS>(
                layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad);
            U_ad = energy.I3 / (c_ad * energy.I1);
        }
    } else {
        auto eigen_pts = propagateRayleighEigenstates<ADS>(
            layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad,
            use_analytical ? 1 : num_sub_layers);
        RayleighEnergyResult<ADS> energy;
        if (use_analytical) {
            energy = computeRayleighEnergyIntegrals_Analytical<ADS>(
                eigen_pts, layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad);
        } else {
            energy = computeRayleighEnergyIntegrals_CT<ADS>(
                eigen_pts, layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad,
                num_sub_layers);
        }
        U_ad = energy.I3 / (c_ad * energy.I1);
    }

    if (U_value_out) *U_value_out = U_ad.value();
    return U_ad.derivatives();
}

// One-derivative AD scalar, used to take exact d/dc inside the GRID kernel
// path. Width 1, so a pass with it costs about what the plain double pass it
// replaces costs.
using ScalarADc = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;

// PhysicalParameters<double> -> a width-1 AD copy holding constants.
//
// Deliberately not to_ad_params<T>: that one pins gravity to 0/1 because the
// solid weak-form path wants it that way, which would silently drop gravity
// from the fluid Cowling cross-check. The four gravity members are carried
// across from the source instead. Same class of defect as the uninitialized
// gravity fixed in bcec09c -- these conversions are where it hides.
inline PhysicalParameters<ScalarADc> to_dc_params(const PhysicalParameters<double>& p) {
    PhysicalParameters<ScalarADc> d;
    d.rho = make_ad_seeded<ScalarADc>(p.rho, 1);
    d.c11 = make_ad_seeded<ScalarADc>(p.c11, 1);
    d.c13 = make_ad_seeded<ScalarADc>(p.c13, 1);
    d.c33 = make_ad_seeded<ScalarADc>(p.c33, 1);
    d.c44 = make_ad_seeded<ScalarADc>(p.c44, 1);
    d.c66 = make_ad_seeded<ScalarADc>(p.c66, 1);
    d.gravity = make_ad_seeded<ScalarADc>(p.gravity, 1);
    d.gravity_internal_scale = make_ad_seeded<ScalarADc>(p.gravity_internal_scale, 1);
    d.gravity_surface_scale = make_ad_seeded<ScalarADc>(p.gravity_surface_scale, 1);
    return d;
}

// Phase-variational group-velocity kernel on one fixed mesh.
//
// Let K_p(omega) = dc/dp at fixed omega.  Since k = omega/c and
// U = (dk/domega)^-1, commuting the omega and model derivatives gives
//
//   dU/dp = U^2 d/domega [ omega K_p / c^2 ].
//
// The derivative on the right follows the dispersion root c(omega).  Only
// that one frequency direction is carried by AD; the vector of model kernels
// is an ordinary output vector.  This changes the expensive propagation from
// width-P forward AD to width 1 while preserving every model-kernel component.
inline FrequencyVariationalGroupKernelLevelResult
computeGroupVelocityKernelFrequencyVariationalLevel(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    double omega,
    double phase_velocity,
    WaveType type,
    bool compute_boundary,
    int rayleigh_num_sub_layers,
    bool rayleigh_use_analytical,
    bool rayleigh_use_adaptive_split
) {
    const double U = calculateGroupVelocityFullAD(
        layers, thicknesses, halfspace, omega, phase_velocity, type);
    if (!std::isfinite(U) || std::abs(U) < 1e-20 ||
        !std::isfinite(omega) || std::abs(omega) < 1e-20 ||
        !std::isfinite(phase_velocity) || std::abs(phase_velocity) < 1e-20) {
        throw std::runtime_error(
            "frequency-variational group kernel has an invalid c/U/omega");
    }

    // dc/domega recovered from the same scalar group-velocity definition used
    // for the prefactor.  This seeds the total derivative along one root branch.
    const double dc_domega =
        (phase_velocity / omega) * (1.0 - phase_velocity / U);

    std::vector<PhysicalParameters<ScalarADc>> layers_ad;
    layers_ad.reserve(layers.size());
    for (const auto& p : layers) layers_ad.push_back(to_dc_params(p));
    std::vector<ScalarADc> thicknesses_ad;
    thicknesses_ad.reserve(thicknesses.size());
    for (double h : thicknesses)
        thicknesses_ad.push_back(make_ad_seeded<ScalarADc>(h, 1));
    const PhysicalParameters<ScalarADc> halfspace_ad = to_dc_params(halfspace);
    const ScalarADc omega_ad = make_ad_seeded<ScalarADc>(omega, 1, 0);
    ScalarADc c_ad = make_ad_seeded<ScalarADc>(phase_velocity, 1);
    c_ad.derivatives()(0) = dc_domega;

    Eigen::Matrix<ScalarADc, Eigen::Dynamic, 1> phase_ad;
    if (type == WaveType::Love) {
        try {
            phase_ad = computeVariationalGradientLove_T<ScalarADc>(
                layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad,
                SensitivityMode::VTI, true, 1, LoveBranchPolicy::Auto, false);
        } catch (const std::exception&) {
            phase_ad = computeVariationalGradientLove_T<ScalarADc>(
                layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad,
                SensitivityMode::VTI, true, 1, LoveBranchPolicy::Auto, true);
        }
    } else {
        phase_ad = computeVariationalGradientRayleigh_T<ScalarADc>(
            layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad,
            SensitivityMode::VTI, rayleigh_num_sub_layers,
            rayleigh_use_analytical, rayleigh_use_adaptive_split);
    }

    FrequencyVariationalGroupKernelLevelResult out;
    out.group_velocity = U;
    out.phase_material_vti = Eigen::VectorXd::Zero(phase_ad.size());
    out.group_material_vti = Eigen::VectorXd::Zero(phase_ad.size());
    const double U2 = U * U;
    for (int i = 0; i < phase_ad.size(); ++i) {
        const ScalarADc scaled = omega_ad * phase_ad(i) / (c_ad * c_ad);
        out.phase_material_vti(i) = phase_ad(i).value();
        out.group_material_vti(i) = U2 * scaled.derivatives()(0);
    }

    if (compute_boundary) {
        Eigen::Matrix<ScalarADc, Eigen::Dynamic, 1> boundary_ad =
            (type == WaveType::Love)
                ? computeBoundaryKernelsLove_T<ScalarADc>(
                      layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad)
                : computeBoundaryKernelsRayleigh_T<ScalarADc>(
                      layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad,
                      rayleigh_use_analytical, rayleigh_num_sub_layers,
                      rayleigh_use_adaptive_split);
        out.phase_boundary = Eigen::VectorXd::Zero(boundary_ad.size());
        out.group_boundary = Eigen::VectorXd::Zero(boundary_ad.size());
        for (int i = 0; i < boundary_ad.size(); ++i) {
            const ScalarADc scaled =
                omega_ad * boundary_ad(i) / (c_ad * c_ad);
            out.phase_boundary(i) = boundary_ad(i).value();
            out.group_boundary(i) = U2 * scaled.derivatives()(0);
        }
    }

    if (!out.phase_material_vti.allFinite() ||
        !out.group_material_vti.allFinite() ||
        (compute_boundary &&
         (!out.phase_boundary.allFinite() || !out.group_boundary.allFinite()))) {
        throw std::runtime_error(
            "frequency-variational group kernel produced a non-finite result");
    }
    return out;
}

// Frequency curvature of one fixed GRID refinement level.
//
// The outer width-1 derivative follows the dispersion root c(omega).  The
// inner two-component derivative in calculateGroupVelocityFullAD still takes
// the partial derivatives of the secular function with respect to omega and
// c, so differentiating the returned U produces the total dU/domega along the
// selected root branch.  Normalization factors are generated once in double
// precision and held fixed in both passes; differentiating a moving gauge
// would otherwise contaminate this second frequency derivative.
struct FrequencyCurvatureLevelResult {
    double group_velocity;
    double group_velocity_omega_derivative;
};

inline FrequencyCurvatureLevelResult compute_frequency_curvature_ad_level(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    double omega,
    double phase_velocity,
    WaveType type
) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(omega > 0.0) || !(phase_velocity > 0.0) ||
        !std::isfinite(omega) || !std::isfinite(phase_velocity)) {
        return {nan, nan};
    }

    std::vector<double> gauge;
    calculateDispersionFunction<double>(
        layers, thicknesses, halfspace, omega, phase_velocity, type,
        NormalizationMode::Vertical,
        static_cast<Eigen::Matrix<double, 5, 1>*>(nullptr), &gauge, nullptr);

    const double group_velocity = calculateGroupVelocityFullAD(
        layers, thicknesses, halfspace, omega, phase_velocity, type, &gauge);
    if (!std::isfinite(group_velocity) || std::abs(group_velocity) < 1e-20) {
        return {group_velocity, nan};
    }

    const double dc_domega =
        (phase_velocity / omega) * (1.0 - phase_velocity / group_velocity);
    if (!std::isfinite(dc_domega)) {
        return {group_velocity, nan};
    }

    std::vector<PhysicalParameters<ScalarADc>> layers_ad;
    layers_ad.reserve(layers.size());
    for (const auto& p : layers) layers_ad.push_back(to_dc_params(p));

    std::vector<ScalarADc> thicknesses_ad;
    thicknesses_ad.reserve(thicknesses.size());
    for (double thickness : thicknesses) {
        thicknesses_ad.push_back(make_ad_seeded<ScalarADc>(thickness, 1));
    }

    ScalarADc omega_ad = make_ad_seeded<ScalarADc>(omega, 1, 0);
    ScalarADc phase_velocity_ad =
        make_ad_seeded<ScalarADc>(phase_velocity, 1);
    phase_velocity_ad.derivatives()(0) = dc_domega;

    const ScalarADc group_velocity_ad = calculateGroupVelocityFullAD(
        layers_ad, thicknesses_ad, to_dc_params(halfspace),
        omega_ad, phase_velocity_ad, type, &gauge);

    return {
        group_velocity_ad.value(),
        group_velocity_ad.derivatives()(0)
    };
}

// dF/dc at fixed model, on the gauge the caller pins with `gauge`.
//
// This used to be a one-sided difference of two independent
// NormalizationMode::Vertical calls, which was wrong twice over. Vertical
// divides by the running Y(0), so what it returns is s(c)*F(c) with a gauge
// that depends on c; differencing two such calls leaves
// s(c)/s(c+eps) = 1 - (dln s/dc)*eps in the quotient. Since dF/dc is the
// denominator of dc/dp = -F_p/F_c, that lands on every component of the
// gradient equally -- measured at -3.85e-05, uniform to 1.4% over eighteen
// basis directions, at Love branch 1, T = 10 s. Overtones are where it bites:
// their eigenfunctions have nodes, Y(0) passes near zero and s moves fast. The
// fundamental mode looked healthy only because its gauge is slow-varying.
//
// Pinning the gauge alone still left +2.5e-06, the one-sided difference's own
// truncation, and made the fundamental worse (2.4e-08 -> 2.5e-06) because the
// two errors had been partly cancelling. Taking the derivative by AD removes
// both.
template <typename ADS>
inline double exact_dF_dc(
    const std::vector<PhysicalParameters<double>>& layers_dbl,
    const std::vector<double>& thicknesses_dbl,
    const PhysicalParameters<double>& halfspace_dbl,
    double omega, double c_root, WaveType type,
    std::vector<double>& gauge
) {
    std::vector<PhysicalParameters<ScalarADc>> layers_c;
    layers_c.reserve(layers_dbl.size());
    for (const auto& p : layers_dbl) layers_c.push_back(to_dc_params(p));
    std::vector<ScalarADc> thick_c;
    thick_c.reserve(thicknesses_dbl.size());
    for (double t : thicknesses_dbl) thick_c.push_back(make_ad_seeded<ScalarADc>(t, 1));
    PhysicalParameters<ScalarADc> hs_c = to_dc_params(halfspace_dbl);

    ScalarADc omega_c = make_ad_seeded<ScalarADc>(omega, 1);
    ScalarADc c_c = make_ad_seeded<ScalarADc>(c_root, 1, 0);   // the only seed

    size_t sIdx = 0;
    auto Y = calculateDispersionFunction(
        layers_c, thick_c, hs_c, omega_c, c_c, type,
        NormalizationMode::Provided,
        static_cast<Eigen::Matrix<ScalarADc, 5, 1>*>(nullptr), &gauge, &sIdx);
    return Y(I24).derivatives()(0);
}

// Newton-polish a GRID kernel evaluation onto the root of the current refined
// mesh.  Every kernel backend must expand about exactly this point; sharing the
// helper prevents backend-dependent O(h^2) root offsets before extrapolation.
inline double polish_grid_kernel_root(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfspace,
    double omega,
    double c_guess,
    WaveType type
) {
    auto f_double = [&](double cv) {
        return calculateDispersionFunction<double>(
            layers, thicknesses, halfspace, omega, cv, type,
            NormalizationMode::VectorNorm)(I24);
    };
    double c_root = c_guess;
    bool root_polish_finite = std::isfinite(c_root) && c_root > 0.0;
    for (int it = 0; it < 15; ++it) {
        const double f0 = f_double(c_root);
        const double epsn = c_root * 1e-7;
        const double df = (f_double(c_root + epsn) - f0) / epsn;
        if (!std::isfinite(f0) || !std::isfinite(df) ||
            std::abs(df) < 1e-30) {
            root_polish_finite = false;
            break;
        }
        const double step = f0 / df;
        const double candidate = c_root - step;
        if (!std::isfinite(step) || !std::isfinite(candidate) ||
            candidate <= 0.0) {
            root_polish_finite = false;
            break;
        }
        c_root = candidate;
        if (std::abs(step) < 1e-11) break;
    }
    if (type == WaveType::Love) {
        const double halfspace_vsh = std::sqrt(halfspace.c66 / halfspace.rho);
        if (!std::isfinite(c_root) || !(c_root > 0.0) ||
            !(c_root < halfspace_vsh)) {
            root_polish_finite = false;
        }
    }
    if (!root_polish_finite) {
        double search_min = std::max(0.1, c_guess * 0.8);
        double search_max = c_guess * 1.2;
        if (type == WaveType::Love) {
            const double halfspace_vsh = std::sqrt(halfspace.c66 / halfspace.rho);
            search_max = std::min(search_max, halfspace_vsh * (1.0 - 1e-8));
        }
        if (!(search_max > search_min) ||
            !std::isfinite(search_min) || !std::isfinite(search_max)) {
            throw std::runtime_error(
                "GRID kernel root polish became non-finite and fallback interval is invalid");
        }
        const double step = (search_max - search_min) / 100.0;
        auto roots = searchRoots(f_double, search_min, search_max, step, 1e-10);
        if (roots.empty()) {
            throw std::runtime_error(
                "GRID kernel root polish became non-finite and bracket fallback found no root");
        }
        c_root = *std::min_element(
            roots.begin(), roots.end(),
            [&](double a, double b) {
                return std::abs(a - c_guess) < std::abs(b - c_guess);
            });
    }
    return c_root;
}

// Single-level sensitivity kernels (dc/dp, dU/dp, dHV/dp) on a GRID/node-based
// mesh, evaluated at the mesh's Newton-polished root. Shared by the params and
// velocity Romberg-with-kernels paths, which call this once per refinement
// level and Neville-extrapolate the results. dc/dp uses the implicit function
// theorem on the dispersion function (-dF/dp / dF/dc); dU/dp uses the
// variational energy-integral kernel (u_variational, gated off for fluid) or
// the nested full-AD group velocity; dHV/dp (Rayleigh only) reuses the F-pass
// surface/seabed state. ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>.
template <typename ADS>
void compute_grid_level_kernels(
    const std::vector<PhysicalParameters<double>>& layers_dbl,
    const std::vector<double>& thicknesses_dbl,
    const PhysicalParameters<double>& halfspace_dbl,
    const std::vector<PhysicalParameters<ADS>>& layers_ad,
    const std::vector<ADS>& thicknesses_ad,
    const PhysicalParameters<ADS>& halfspace_ad,
    double omega, double c_guess, WaveType type,
    bool compute_hv, bool compute_u, bool u_variational,
    int num_vars,
    Eigen::VectorXd& dc_out, Eigen::VectorXd& dU_out, Eigen::VectorXd& dHV_out,
    bool polish_root = true
) {
    const double c_root = polish_root
        ? polish_grid_kernel_root(
              layers_dbl, thicknesses_dbl, halfspace_dbl, omega, c_guess, type)
        : c_guess;

    dc_out = Eigen::VectorXd::Zero(num_vars);
    dU_out = Eigen::VectorXd::Zero(num_vars);
    dHV_out = Eigen::VectorXd::Zero(num_vars);
    ADS c_ad; c_ad.value() = c_root; c_ad.derivatives() = Eigen::VectorXd::Zero(num_vars);
    ADS omega_ad; omega_ad.value() = omega; omega_ad.derivatives() = Eigen::VectorXd::Zero(num_vars);

    if (type == WaveType::Love) {
        // Preserve the established secular-function AD kernel whenever it is
        // finite.  At shallow short-period Love points its absolute
        // eigenfunction gauge can overflow; only then use the gauge-invariant
        // variational gradient as a fallback.
        // Pin the gauge once, then take both F_p and F_c on it: the parameter
        // pass by AD as before, and F_c by AD too (see exact_dF_dc).
        std::vector<double> gauge;
        calculateDispersionFunction(
            layers_dbl, thicknesses_dbl, halfspace_dbl, omega, c_root, type,
            NormalizationMode::Vertical,
            static_cast<Eigen::Matrix<double, 5, 1>*>(nullptr), &gauge);

        Eigen::Matrix<ADS, 5, 1> seabedState_ad; seabedState_ad.setZero();
        size_t sIdx_ad = 0;
        auto Y_all_ad = calculateDispersionFunction(
            layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad, type,
            NormalizationMode::Provided, &seabedState_ad, &gauge, &sIdx_ad);
        ADS F_ad = Y_all_ad(I24);
        Eigen::VectorXd dF_dp = F_ad.derivatives();
        double dF_dc = exact_dF_dc<ADS>(layers_dbl, thicknesses_dbl, halfspace_dbl,
                                        omega, c_root, type, gauge);
        bool finite_secular = std::isfinite(dF_dc) &&
            std::abs(dF_dc) >= 1e-20 &&
            dF_dp.array().isFinite().all();
        if (finite_secular) {
            dc_out = -dF_dp / dF_dc;
        } else {
            Eigen::VectorXd g_physical;
            try {
                g_physical = computeVariationalGradientLove_T<double>(
                    layers_dbl, thicknesses_dbl, halfspace_dbl, omega, c_root,
                    SensitivityMode::VTI, true);
            } catch (const std::exception&) {
                g_physical = computeVariationalGradientLove_T<double>(
                    layers_dbl, thicknesses_dbl, halfspace_dbl, omega, c_root,
                    SensitivityMode::VTI, true, 1,
                    LoveBranchPolicy::Auto, true);
            }
            if (!g_physical.array().isFinite().all()) {
                g_physical = computeVariationalGradientLove_T<double>(
                    layers_dbl, thicknesses_dbl, halfspace_dbl, omega, c_root,
                    SensitivityMode::VTI, true, 1,
                    LoveBranchPolicy::Auto, true);
            }
            const int n_physical = static_cast<int>(layers_dbl.size()) * 6;
            for (size_t i = 0; i < layers_ad.size(); ++i) {
                dc_out += g_physical(static_cast<int>(i) * 6 + 0) * layers_ad[i].rho.derivatives();
                dc_out += g_physical(static_cast<int>(i) * 6 + 4) * layers_ad[i].c44.derivatives();
                dc_out += g_physical(static_cast<int>(i) * 6 + 5) * layers_ad[i].c66.derivatives();
            }
            dc_out += g_physical(n_physical + 0) * halfspace_ad.rho.derivatives();
            dc_out += g_physical(n_physical + 4) * halfspace_ad.c44.derivatives();
            dc_out += g_physical(n_physical + 5) * halfspace_ad.c66.derivatives();
        }
    } else {

        // Same gauge pinning and exact F_c as the Love branch above.
        std::vector<double> gauge;
        calculateDispersionFunction(
            layers_dbl, thicknesses_dbl, halfspace_dbl, omega, c_root, type,
            NormalizationMode::Vertical,
            static_cast<Eigen::Matrix<double, 5, 1>*>(nullptr), &gauge);

        Eigen::Matrix<ADS, 5, 1> seabedState_ad; seabedState_ad.setZero();
        size_t sIdx_ad = 0;
        auto Y_all_ad = calculateDispersionFunction(layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad, type, NormalizationMode::Provided, &seabedState_ad, &gauge, &sIdx_ad);
        ADS F_ad = Y_all_ad(I24);
        Eigen::VectorXd dF_dp = F_ad.derivatives();
        double dF_dc = exact_dF_dc<ADS>(layers_dbl, thicknesses_dbl, halfspace_dbl,
                                        omega, c_root, type, gauge);
        if (std::abs(dF_dc) < 1e-20) throw std::runtime_error("dF/dc is near zero");
        dc_out = -dF_dp / dF_dc;

        if (type == WaveType::Rayleigh && compute_hv) {
            bool topIsLiquid = (!layers_ad.empty() && isLiquidLayer(layers_ad[0]));
            ADS HV_ad = topIsLiquid ? calculateHVRatio(seabedState_ad) : calculateHVRatio(Y_all_ad);
            Eigen::VectorXd dHV_direct = HV_ad.derivatives();
            // Third instance of the same shape: the dc/dp coupling term was a
            // one-sided difference sharing the old epsc. Width-1 AD on the
            // pinned gauge, as for F_c and U_c.
            std::vector<PhysicalParameters<ScalarADc>> layers_hv;
            layers_hv.reserve(layers_dbl.size());
            for (const auto& p : layers_dbl) layers_hv.push_back(to_dc_params(p));
            std::vector<ScalarADc> thick_hv;
            thick_hv.reserve(thicknesses_dbl.size());
            for (double t : thicknesses_dbl) thick_hv.push_back(make_ad_seeded<ScalarADc>(t, 1));
            Eigen::Matrix<ScalarADc, 5, 1> seabed_hv; seabed_hv.setZero();
            size_t sIdx_hv = 0;
            auto Y_hv_c = calculateDispersionFunction(
                layers_hv, thick_hv, to_dc_params(halfspace_dbl),
                make_ad_seeded<ScalarADc>(omega, 1),
                make_ad_seeded<ScalarADc>(c_root, 1, 0), type,
                NormalizationMode::Provided, &seabed_hv, &gauge, &sIdx_hv);
            ScalarADc HV_c = topIsLiquid ? calculateHVRatio(seabed_hv)
                                         : calculateHVRatio(Y_hv_c);
            double dHV_dc = HV_c.derivatives()(0);
            dHV_out = dHV_direct + dHV_dc * dc_out;
        }
    }

    if (compute_u) {
        bool has_fluid = hasLiquidLayer(layers_dbl) || isLiquidLayer(halfspace_dbl);
        if (u_variational && !has_fluid) {
            dU_out = computeGroupVelocityKernelVariational(
                layers_dbl, thicknesses_dbl, halfspace_dbl,
                layers_ad, thicknesses_ad, halfspace_ad,
                omega, c_root, dc_out, type, nullptr, true, 10, polish_root);
        } else {
            // dU/dp = dU/dp|_c + (dU/dc) dc/dp. The first term is AD; the
            // second used to be a one-sided difference at a fixed 1e-8*|c|
            // step, so the correction that couples in dc/dp was the least
            // accurate part of an otherwise exact pass. Width-1 AD gives it
            // exactly for about the cost of the double evaluation it replaces.
            // This is the branch fluid models take -- the variational U kernel
            // is gated off for them -- so it is their only route.
            ADS U_ad = calculateGroupVelocityFullAD(layers_ad, thicknesses_ad, halfspace_ad, omega_ad, c_ad, type);
            Eigen::VectorXd dU_direct = U_ad.derivatives();

            std::vector<PhysicalParameters<ScalarADc>> layers_c;
            layers_c.reserve(layers_dbl.size());
            for (const auto& p : layers_dbl) layers_c.push_back(to_dc_params(p));
            std::vector<ScalarADc> thick_c;
            thick_c.reserve(thicknesses_dbl.size());
            for (double t : thicknesses_dbl) thick_c.push_back(make_ad_seeded<ScalarADc>(t, 1));
            ScalarADc U_c = calculateGroupVelocityFullAD(
                layers_c, thick_c, to_dc_params(halfspace_dbl),
                make_ad_seeded<ScalarADc>(omega, 1),
                make_ad_seeded<ScalarADc>(c_root, 1, 0), type);
            double dU_dc = U_c.derivatives()(0);
            dU_out = dU_direct + dU_dc * dc_out;
        }
    }
}

inline HessianResult computePhaseVelocityHessianFullAD(
    const std::vector<PhysicalParameters<double>>& l, const std::vector<double>& t,
    const PhysicalParameters<double>& h, double f, double c, WaveType wt,
    SensitivityMode sm, bool ct
) {
    if (wt == WaveType::Love) return computePhaseVelocityHessianLove(l, t, h, f, c, sm, ct);
    return computePhaseVelocityHessianStable(l, t, h, f, c, wt, sm, ct);
}

inline HessianResult computePhaseVelocityHessianReference(
    const std::vector<PhysicalParameters<double>>& l, const std::vector<double>& t,
    const PhysicalParameters<double>& h, double f, double c, WaveType wt
) {
    if (wt == WaveType::Love) return computePhaseVelocityHessianLove(l, t, h, f, c);
    return computePhaseVelocityHessianStable(l, t, h, f, c, wt);
}

// Deprecated aliases. Use the LayerCake spellings in new code.
template <typename T>
inline void seedPancakeParamBlock(
    const std::vector<PhysicalParameters<double>>& layers,
    const PhysicalParameters<double>& halfSpace, SensitivityMode mode,
    int j0, int width, std::vector<PhysicalParameters<T>>& layers_ad,
    PhysicalParameters<T>& hs_ad) {
    seedLayerCakeParamBlock<T>(layers,halfSpace,mode,j0,width,layers_ad,hs_ad);
}
inline Eigen::VectorXd mapBoundaryKernelToPancakeThickness(const Eigen::VectorXd& kernel) {
    return mapBoundaryKernelToLayerCakeThickness(kernel);
}
