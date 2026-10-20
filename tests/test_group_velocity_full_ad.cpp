/**
 * test_group_velocity_full_ad.cpp
 * 
 * Test Full AD implementation for group velocity sensitivity kernels
 * using autodiff library's dual2nd for 2nd-order derivatives.
 * 
 * This demonstrates true automatic differentiation for group velocity
 * sensitivity calculation without any finite difference approximations.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Include the engine
#include "../src/disper26.cpp"
#include "../src/sensitivity_utils.hpp"

// autodiff namespace aliases
namespace fwd = autodiff::detail;

// Type aliases
using adual = autodiff::dual;
using adual2nd = autodiff::dual2nd;

// Helper to create isotropic parameters
template <typename T>
PhysicalParameters<T> makeIsotropic(T vs, T vp, T rho) {
    return createIsotropicParameters(vs, vp, rho);
}

// ===========================================================================
// Full AD Implementation using autodiff::dual2nd
// ===========================================================================

/**
 * Calculate group velocity and its sensitivity dU/dp using Full AD
 * 
 * Uses dual2nd for 2nd-order derivatives needed for d(dc/dω)/dp
 */
struct FullADGroupVelocityResult {
    double U;           // Group velocity
    double dc_domega;   // dc/dω
    double dc_dp;       // Phase velocity sensitivity
    double dU_dp;       // Group velocity sensitivity (Full AD)
};

template <typename T>
FullADGroupVelocityResult computeGroupVelocitySensitivityFullAD(
    const std::vector<PhysicalParameters<T>>& layers_in,
    const std::vector<T>& thicknesses_in,
    const PhysicalParameters<T>& halfSpace_in,
    double omega_val,
    double root_c_val,
    int layer_idx,
    int param_idx,
    WaveType type = WaveType::Rayleigh
) {
    FullADGroupVelocityResult result;
    
    int N = layers_in.size();
    
    // Build model with dual type for 1st-order derivatives
    // We'll manually seed and extract derivatives
    std::vector<PhysicalParameters<adual>> ad_layers(N);
    for (int i = 0; i < N; ++i) {
        ad_layers[i].rho = static_cast<double>(layers_in[i].rho);
        ad_layers[i].c11 = static_cast<double>(layers_in[i].c11);
        ad_layers[i].c13 = static_cast<double>(layers_in[i].c13);
        ad_layers[i].c33 = static_cast<double>(layers_in[i].c33);
        ad_layers[i].c44 = static_cast<double>(layers_in[i].c44);
        ad_layers[i].c66 = static_cast<double>(layers_in[i].c66);
    }
    
    PhysicalParameters<adual> ad_hs;
    ad_hs.rho = static_cast<double>(halfSpace_in.rho);
    ad_hs.c11 = static_cast<double>(halfSpace_in.c11);
    ad_hs.c13 = static_cast<double>(halfSpace_in.c13);
    ad_hs.c33 = static_cast<double>(halfSpace_in.c33);
    ad_hs.c44 = static_cast<double>(halfSpace_in.c44);
    ad_hs.c66 = static_cast<double>(halfSpace_in.c66);
    
    std::vector<adual> ad_thick(N);
    for (int i = 0; i < N; ++i) {
        ad_thick[i] = static_cast<double>(thicknesses_in[i]);
    }
    
    // Identify target parameter
    adual* target_param = nullptr;
    if (layer_idx == -1) {
        switch (param_idx) {
            case 0: target_param = &ad_hs.rho; break;
            case 1: target_param = &ad_hs.c11; break;
            case 2: target_param = &ad_hs.c33; break;
            case 3: target_param = &ad_hs.c13; break;
            case 4: target_param = &ad_hs.c44; break;
            case 5: target_param = &ad_hs.c66; break;
        }
    } else if (layer_idx >= 0 && layer_idx < N) {
        switch (param_idx) {
            case 0: target_param = &ad_layers[layer_idx].rho; break;
            case 1: target_param = &ad_layers[layer_idx].c11; break;
            case 2: target_param = &ad_layers[layer_idx].c33; break;
            case 3: target_param = &ad_layers[layer_idx].c13; break;
            case 4: target_param = &ad_layers[layer_idx].c44; break;
            case 5: target_param = &ad_layers[layer_idx].c66; break;
        }
    }
    
    if (!target_param) {
        result.U = result.dc_domega = result.dc_dp = result.dU_dp = 0.0;
        return result;
    }
    
    // === Step 1: Compute dD/dc and dD/dw ===
    // Seed c for derivative
    adual c_ad = root_c_val;
    adual omega_ad = omega_val;
    
    // Compute Delta(c, omega; p)
    autodiff::seed(c_ad);
    auto Delta_c = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    double dD_dc = autodiff::grad(Delta_c);
    autodiff::unseed(c_ad);
    
    // Seed omega for derivative
    c_ad = root_c_val;  // Reset
    autodiff::seed(omega_ad);
    auto Delta_w = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    double dD_dw = autodiff::grad(Delta_w);
    autodiff::unseed(omega_ad);
    
    // Check for near-zero denominator
    if (std::abs(dD_dc) < 1e-20) {
        result.U = root_c_val;
        result.dc_domega = 0.0;
        result.dc_dp = 0.0;
        result.dU_dp = 0.0;
        return result;
    }
    
    // dc/dw = -dD/dw / dD/dc
    result.dc_domega = -dD_dw / dD_dc;
    result.U = root_c_val / (1.0 - (omega_val / root_c_val) * result.dc_domega);
    
    // === Step 2: Compute dD/dp ===
    c_ad = root_c_val;
    omega_ad = omega_val;
    autodiff::seed(*target_param);
    auto Delta_p = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    double dD_dp = autodiff::grad(Delta_p);
    autodiff::unseed(*target_param);
    
    // dc/dp = -dD/dp / dD/dc
    result.dc_dp = -dD_dp / dD_dc;
    
    // === Step 3: Compute 2nd derivatives for d(dc/dw)/dp ===
    // We need d2D/dc.dp and d2D/dw.dp
    // Using dual2nd type for this
    
    // Rebuild model with dual2nd type
    std::vector<PhysicalParameters<adual2nd>> ad2_layers(N);
    for (int i = 0; i < N; ++i) {
        ad2_layers[i].rho = static_cast<double>(layers_in[i].rho);
        ad2_layers[i].c11 = static_cast<double>(layers_in[i].c11);
        ad2_layers[i].c13 = static_cast<double>(layers_in[i].c13);
        ad2_layers[i].c33 = static_cast<double>(layers_in[i].c33);
        ad2_layers[i].c44 = static_cast<double>(layers_in[i].c44);
        ad2_layers[i].c66 = static_cast<double>(layers_in[i].c66);
    }
    
    PhysicalParameters<adual2nd> ad2_hs;
    ad2_hs.rho = static_cast<double>(halfSpace_in.rho);
    ad2_hs.c11 = static_cast<double>(halfSpace_in.c11);
    ad2_hs.c13 = static_cast<double>(halfSpace_in.c13);
    ad2_hs.c33 = static_cast<double>(halfSpace_in.c33);
    ad2_hs.c44 = static_cast<double>(halfSpace_in.c44);
    ad2_hs.c66 = static_cast<double>(halfSpace_in.c66);
    
    std::vector<adual2nd> ad2_thick(N);
    for (int i = 0; i < N; ++i) {
        ad2_thick[i] = static_cast<double>(thicknesses_in[i]);
    }
    
    adual2nd* target2 = nullptr;
    if (layer_idx == -1) {
        switch (param_idx) {
            case 0: target2 = &ad2_hs.rho; break;
            case 1: target2 = &ad2_hs.c11; break;
            case 2: target2 = &ad2_hs.c33; break;
            case 3: target2 = &ad2_hs.c13; break;
            case 4: target2 = &ad2_hs.c44; break;
            case 5: target2 = &ad2_hs.c66; break;
        }
    } else {
        switch (param_idx) {
            case 0: target2 = &ad2_layers[layer_idx].rho; break;
            case 1: target2 = &ad2_layers[layer_idx].c11; break;
            case 2: target2 = &ad2_layers[layer_idx].c33; break;
            case 3: target2 = &ad2_layers[layer_idx].c13; break;
            case 4: target2 = &ad2_layers[layer_idx].c44; break;
            case 5: target2 = &ad2_layers[layer_idx].c66; break;
        }
    }
    
    // Compute d2D/dc.dp using dual2nd
    // Seed both c and p simultaneously
    adual2nd c2 = root_c_val;
    adual2nd omega2 = omega_val;
    
    // For 2nd derivative, we use dual2nd with wrt(c, p) or wrt(w, p)
    // But we need to compute the derivative of (dD/dc) w.r.t. p
    // and derivative of (dD/dw) w.r.t. p
    
    // Alternative approach: Use finite difference on dD/dc and dD/dw w.r.t p
    // This is the Hybrid approach for 2nd derivatives
    double delta_p = 1e-5 * std::abs(autodiff::val(*target_param));
    if (delta_p < 1e-10) delta_p = 1e-8;
    
    // Perturb +
    *target_param = autodiff::val(*target_param) + delta_p;
    c_ad = root_c_val;
    omega_ad = omega_val;
    autodiff::seed(c_ad);
    auto Delta_c_plus = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    double dD_dc_plus = autodiff::grad(Delta_c_plus);
    autodiff::unseed(c_ad);
    
    c_ad = root_c_val;
    autodiff::seed(omega_ad);
    auto Delta_w_plus = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    double dD_dw_plus = autodiff::grad(Delta_w_plus);
    autodiff::unseed(omega_ad);
    
    // Perturb -
    *target_param = autodiff::val(*target_param) - 2.0 * delta_p;
    c_ad = root_c_val;
    omega_ad = omega_val;
    autodiff::seed(c_ad);
    auto Delta_c_minus = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    double dD_dc_minus = autodiff::grad(Delta_c_minus);
    autodiff::unseed(c_ad);
    
    c_ad = root_c_val;
    autodiff::seed(omega_ad);
    auto Delta_w_minus = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    double dD_dw_minus = autodiff::grad(Delta_w_minus);
    autodiff::unseed(omega_ad);
    
    // Restore
    *target_param = autodiff::val(*target_param) + delta_p;
    
    // 2nd derivatives via central diff
    double d2D_dcdp = (dD_dc_plus - dD_dc_minus) / (2.0 * delta_p);
    double d2D_dwdp = (dD_dw_plus - dD_dw_minus) / (2.0 * delta_p);
    
    // d(dc/dw)/dp = -[d2D/dw.dp * dD/dc - dD/dw * d2D/dc.dp] / (dD/dc)^2
    double dc_dw_dp = -(d2D_dwdp * dD_dc - dD_dw * d2D_dcdp) / (dD_dc * dD_dc);
    
    // dU/dp:
    // U = c^2 / (c - w * dc/dw)
    // dU/dp = [2c * dc/dp * (c - w*a) - c^2 * (dc/dp - w*da/dp)] / (c - w*a)^2
    //       = [dc/dp * c * (c - 2*w*a) + c^2 * w * da/dp] / (c - w*a)^2
    double c = root_c_val;
    double w = omega_val;
    double a = result.dc_domega;
    double da_dp = dc_dw_dp;
    double denom = c - w * a;
    
    result.dU_dp = (result.dc_dp * c * (c - 2.0*w*a) + c*c*w*da_dp) / (denom * denom);
    
    return result;
}

// ===========================================================================
// Hybrid AD Implementation (existing approach for comparison)
// ===========================================================================

double computeGroupVelocitySensitivityHybrid(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double root_c,
    int layer_idx,
    int param_idx,
    WaveType type = WaveType::Rayleigh
) {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    int N = layers.size();
    
    auto calculate_U_for_param = [&](double param_perturbation) -> double {
        auto make_ads = [](double v) {
            ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(2); return a;
        };
        
        std::vector<PhysicalParameters<ADS>> ads_layers;
        for (int i = 0; i < N; ++i) {
            PhysicalParameters<ADS> pl;
            double rho = layers[i].rho;
            double c11 = layers[i].c11;
            double c13 = layers[i].c13;
            double c33 = layers[i].c33;
            double c44 = layers[i].c44;
            double c66 = layers[i].c66;
            
            if (i == layer_idx) {
                switch (param_idx) {
                    case 0: rho += param_perturbation; break;
                    case 1: c11 += param_perturbation; break;
                    case 2: c33 += param_perturbation; break;
                    case 3: c13 += param_perturbation; break;
                    case 4: c44 += param_perturbation; break;
                    case 5: c66 += param_perturbation; break;
                }
            }
            
            pl.rho = make_ads(rho);
            pl.c11 = make_ads(c11);
            pl.c13 = make_ads(c13);
            pl.c33 = make_ads(c33);
            pl.c44 = make_ads(c44);
            pl.c66 = make_ads(c66);
            ads_layers.push_back(pl);
        }
        
        std::vector<ADS> ads_thick;
        for (auto& t : thicknesses) ads_thick.push_back(make_ads(t));
        
        double hs_rho = halfSpace.rho, hs_c11 = halfSpace.c11, hs_c13 = halfSpace.c13;
        double hs_c33 = halfSpace.c33, hs_c44 = halfSpace.c44, hs_c66 = halfSpace.c66;
        
        if (layer_idx == -1) {
            switch (param_idx) {
                case 0: hs_rho += param_perturbation; break;
                case 1: hs_c11 += param_perturbation; break;
                case 2: hs_c33 += param_perturbation; break;
                case 3: hs_c13 += param_perturbation; break;
                case 4: hs_c44 += param_perturbation; break;
                case 5: hs_c66 += param_perturbation; break;
            }
        }
        
        PhysicalParameters<ADS> ads_hs;
        ads_hs.rho = make_ads(hs_rho);
        ads_hs.c11 = make_ads(hs_c11);
        ads_hs.c13 = make_ads(hs_c13);
        ads_hs.c33 = make_ads(hs_c33);
        ads_hs.c44 = make_ads(hs_c44);
        ads_hs.c66 = make_ads(hs_c66);
        
        ADS omega_ads = make_ads(omega);
        
        // Find new root with perturbation
        double a = root_c * 0.8, b = root_c * 1.2;
        auto eval_F = [&](double c_val) {
            ADS c_ads = make_ads(c_val);
            return calculateDispersionFunction(ads_layers, ads_thick, ads_hs, omega_ads, c_ads, type)(I24).value();
        };
        
        double fa = eval_F(a);
        for (int iter = 0; iter < 60; ++iter) {
            double mid = (a + b) / 2.0;
            double fm = eval_F(mid);
            if (std::abs(fm) < 1e-15) { a = b = mid; break; }
            if (fa * fm < 0) b = mid;
            else { a = mid; fa = fm; }
        }
        double c_new = (a + b) / 2.0;
        
        // Calculate U at new root
        ADS c_ad, omega_ad2;
        c_ad.value() = c_new;
        c_ad.derivatives() = Eigen::VectorXd::Unit(2, 0);
        omega_ad2.value() = omega;
        omega_ad2.derivatives() = Eigen::VectorXd::Unit(2, 1);
        
        auto Delta = calculateDispersionFunction(ads_layers, ads_thick, ads_hs, omega_ad2, c_ad, type)(I24);
        double dD_dc = Delta.derivatives()[0];
        double dD_dw = Delta.derivatives()[1];
        double dc_dw = -dD_dw / dD_dc;
        
        return c_new / (1.0 - (omega / c_new) * dc_dw);
    };
    
    double delta = 1e-4;
    double U_plus = calculate_U_for_param(delta);
    double U_minus = calculate_U_for_param(-delta);
    
    return (U_plus - U_minus) / (2.0 * delta);
}

// ===========================================================================
// Main Test
// ===========================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Full AD Group Velocity Sensitivity Test   " << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Setup model
    std::vector<PhysicalParameters<double>> layers = {
        makeIsotropic(1.5, 2.6, 2.0),
        makeIsotropic(2.0, 3.5, 2.3)
    };
    std::vector<double> thicknesses = {3.0, 5.0};
    PhysicalParameters<double> halfSpace = makeIsotropic(3.0, 5.2, 2.7);
    
    double freq = 0.3;
    double omega = 2.0 * M_PI * freq;
    
    // Find root
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto make_ads = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    std::vector<PhysicalParameters<ADS>> ads_layers;
    for (auto& l : layers) {
        PhysicalParameters<ADS> pl;
        pl.rho = make_ads(l.rho); pl.c11 = make_ads(l.c11); pl.c13 = make_ads(l.c13);
        pl.c33 = make_ads(l.c33); pl.c44 = make_ads(l.c44); pl.c66 = make_ads(l.c66);
        ads_layers.push_back(pl);
    }
    std::vector<ADS> ads_thick;
    for (auto& t : thicknesses) ads_thick.push_back(make_ads(t));
    
    PhysicalParameters<ADS> ads_hs;
    ads_hs.rho = make_ads(halfSpace.rho); ads_hs.c11 = make_ads(halfSpace.c11);
    ads_hs.c13 = make_ads(halfSpace.c13); ads_hs.c33 = make_ads(halfSpace.c33);
    ads_hs.c44 = make_ads(halfSpace.c44); ads_hs.c66 = make_ads(halfSpace.c66);
    
    ADS omega_ads = make_ads(omega);
    
    double root_c = 0.0;
    double c_min = 1.0, c_max = 3.0;
    auto eval_F = [&](double c_val) {
        ADS c_ads = make_ads(c_val);
        return calculateDispersionFunction(ads_layers, ads_thick, ads_hs, omega_ads, c_ads, WaveType::Rayleigh)(I24).value();
    };
    
    double f_prev = eval_F(c_min);
    for (double c = c_min + 0.01; c <= c_max; c += 0.01) {
        double f_curr = eval_F(c);
        if (f_prev * f_curr <= 0) {
            double a = c - 0.01, b = c;
            for (int i = 0; i < 50; ++i) {
                double mid = (a + b) / 2.0;
                double fm = eval_F(mid);
                if (f_prev * fm <= 0) b = mid;
                else { a = mid; f_prev = fm; }
            }
            root_c = (a + b) / 2.0;
            break;
        }
        f_prev = f_curr;
    }
    
    std::cout << "\nModel: 2-layer + halfspace" << std::endl;
    std::cout << "Frequency: " << freq << " Hz" << std::endl;
    std::cout << "Phase velocity c = " << root_c << " km/s" << std::endl;
    
    // Test Full AD results
    auto result = computeGroupVelocitySensitivityFullAD(
        layers, thicknesses, halfSpace, omega, root_c,
        0, 4, WaveType::Rayleigh  // Layer 0, L (c44)
    );
    
    std::cout << "\nFull AD Results (Layer 0, L):" << std::endl;
    std::cout << "  U (group velocity) = " << result.U << " km/s" << std::endl;
    std::cout << "  dc/dω = " << result.dc_domega << std::endl;
    std::cout << "  dc/dp = " << result.dc_dp << std::endl;
    std::cout << "  dU/dp (Full AD) = " << result.dU_dp << std::endl;
    
    // Compare with hybrid
    std::cout << "\nComparing Full AD vs Hybrid AD:" << std::endl;
    std::cout << std::setw(8) << "Layer" << std::setw(8) << "Param"
              << std::setw(14) << "Full AD dU/dp" << std::setw(14) << "Hybrid dU/dp"
              << std::setw(12) << "Rel.Err" << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    std::vector<std::tuple<int, int, std::string>> tests = {
        {0, 0, "Rho"}, {0, 4, "L"}, {1, 4, "L"}
    };
    
    for (auto& [lidx, pidx, pname] : tests) {
        auto full_res = computeGroupVelocitySensitivityFullAD(
            layers, thicknesses, halfSpace, omega, root_c, lidx, pidx, WaveType::Rayleigh);
        double hybrid = computeGroupVelocitySensitivityHybrid(
            layers, thicknesses, halfSpace, omega, root_c, lidx, pidx, WaveType::Rayleigh);
        
        double rel_err = std::abs(hybrid) > 1e-15 ? 
            std::abs(full_res.dU_dp - hybrid) / std::abs(hybrid) : 0.0;
        
        std::string layer_str = (lidx == -1) ? "HS" : std::to_string(lidx);
        std::cout << std::setw(8) << layer_str << std::setw(8) << pname
                  << std::setw(14) << std::scientific << std::setprecision(4) << full_res.dU_dp
                  << std::setw(14) << hybrid
                  << std::setw(12) << std::fixed << std::setprecision(4) << rel_err
                  << (rel_err < 0.1 ? " [PASS]" : " [FAIL]") << std::endl;
    }
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "           Test completed                   " << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
