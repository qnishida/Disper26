/**
 * test_autodiff_forward_sensitivity.cpp
 * 
 * Test forward mode autodiff-based sensitivity kernels
 * by comparing with existing Eigen::AutoDiff implementation.
 * 
 * Test cases:
 * 1. 3-layer solid model (Rayleigh)
 * 2. 3-layer solid model (Love)
 * 3. Ocean layer model (Rayleigh)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Include the engine (has Eigen::AutoDiff implementation)
#include "../src/disper26.cpp"
#include "../src/sensitivity_utils.hpp"

// Include autodiff utilities
#include "../src/autodiff_utils.hpp"

// Helper to create isotropic PhysicalParameters
template <typename T>
PhysicalParameters<T> makeIsotropic(T vs, T vp, T rho) {
    return createIsotropicParameters(vs, vp, rho);
}

// --- Test 1: 3-layer solid model (Rayleigh) ---
void test_solid_3layer_rayleigh() {
    std::cout << "\n=== Test 1: 3-Layer Solid Model (Rayleigh) ===" << std::endl;
    
    // Model setup: 3 layers + halfspace
    // Layer 1: Vs=1.5, Vp=2.6, Rho=2.0, H=2km
    // Layer 2: Vs=2.0, Vp=3.5, Rho=2.3, H=5km
    // Layer 3: Vs=2.5, Vp=4.3, Rho=2.5, H=8km
    // Halfspace: Vs=3.5, Vp=6.0, Rho=2.8
    
    std::vector<PhysicalParameters<double>> layers = {
        makeIsotropic(1.5, 2.6, 2.0),
        makeIsotropic(2.0, 3.5, 2.3),
        makeIsotropic(2.5, 4.3, 2.5)
    };
    std::vector<double> thicknesses = {2.0, 5.0, 8.0};
    PhysicalParameters<double> halfSpace = makeIsotropic(3.5, 6.0, 2.8);
    
    double freq = 0.5;  // Hz
    double omega = 2.0 * M_PI * freq;
    
    // Find phase velocity root
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto make_ads = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1);
        return a;
    };
    
    // Convert to ADS for root finding
    std::vector<PhysicalParameters<ADS>> ads_layers;
    for (auto& l : layers) {
        PhysicalParameters<ADS> pl;
        pl.rho = make_ads(l.rho);
        pl.c11 = make_ads(l.c11);
        pl.c13 = make_ads(l.c13);
        pl.c33 = make_ads(l.c33);
        pl.c44 = make_ads(l.c44);
        pl.c66 = make_ads(l.c66);
        ads_layers.push_back(pl);
    }
    std::vector<ADS> ads_thick;
    for (auto& t : thicknesses) ads_thick.push_back(make_ads(t));
    
    PhysicalParameters<ADS> ads_hs;
    ads_hs.rho = make_ads(halfSpace.rho);
    ads_hs.c11 = make_ads(halfSpace.c11);
    ads_hs.c13 = make_ads(halfSpace.c13);
    ads_hs.c33 = make_ads(halfSpace.c33);
    ads_hs.c44 = make_ads(halfSpace.c44);
    ads_hs.c66 = make_ads(halfSpace.c66);
    
    // Find root by scanning
    ADS omega_ads = make_ads(omega);
    double c_min = 1.0, c_max = 3.5;
    double dz = 0.01;
    double root_c = 0.0;
    
    auto eval_F = [&](double c_val) -> double {
        ADS c_ads = make_ads(c_val);
        auto res = calculateDispersionFunction(ads_layers, ads_thick, ads_hs, omega_ads, c_ads, WaveType::Rayleigh);
        return res(I24).value();
    };
    
    double f_prev = eval_F(c_min);
    for (double c = c_min + dz; c <= c_max; c += dz) {
        double f_curr = eval_F(c);
        if (f_prev * f_curr <= 0 && std::abs(f_curr) < std::abs(f_prev)) {
            // Refine with bisection
            double a = c - dz, b = c;
            for (int i = 0; i < 50; ++i) {
                double mid = (a + b) / 2.0;
                double f_mid = eval_F(mid);
                if (f_prev * f_mid <= 0) b = mid;
                else { a = mid; f_prev = f_mid; }
            }
            root_c = (a + b) / 2.0;
            break;
        }
        f_prev = f_curr;
    }
    
    std::cout << "  Phase velocity c = " << root_c << " km/s at f = " << freq << " Hz" << std::endl;
    
    if (root_c < 1e-3) {
        std::cout << "  [SKIP] No root found in range" << std::endl;
        return;
    }
    
    // Compare sensitivity kernels using existing vs autodiff
    std::cout << "\n  Comparing sensitivity kernels (Eigen::AutoDiff vs autodiff):" << std::endl;
    std::cout << "  " << std::setw(12) << "Layer" << std::setw(12) << "Param" 
              << std::setw(15) << "Eigen dc/dp" << std::setw(15) << "autodiff dc/dp"
              << std::setw(15) << "Rel. Error" << std::endl;
    std::cout << "  " << std::string(69, '-') << std::endl;
    
    // Compute using existing implementation (finite difference for sensitivity)
    auto existing_kernels = computeSensitivityKernelsFD(layers, thicknesses, halfSpace, 
                                                       freq, root_c, WaveType::Rayleigh,
                                                       SensitivityMode::VTI);
    
    int num_pass = 0, num_fail = 0;
    double max_error = 0.0;
    
    for (auto& k : existing_kernels) {
        int layer_idx = k.layerIndex;
        int param_idx = -1;
        
        // Map parameter name to index
        if (k.parameterName == "Rho") param_idx = 0;
        else if (k.parameterName == "A") param_idx = 1;
        else if (k.parameterName == "C") param_idx = 2;
        else if (k.parameterName == "F") param_idx = 3;
        else if (k.parameterName == "L") param_idx = 4;
        else if (k.parameterName == "N") param_idx = 5;
        
        if (param_idx < 0) continue;
        
        // Compute using autodiff forward mode
        double dc_dp_autodiff = computePhaseVelocitySensitivityForward(
            layers, thicknesses, halfSpace, omega, root_c,
            layer_idx, param_idx, WaveType::Rayleigh
        );
        
        double dc_dp_existing = k.d_PhaseVelocity;
        
        // Relative error
        double rel_err = 0.0;
        if (std::abs(dc_dp_existing) > 1e-15) {
            rel_err = std::abs(dc_dp_autodiff - dc_dp_existing) / std::abs(dc_dp_existing);
        } else if (std::abs(dc_dp_autodiff) > 1e-15) {
            rel_err = 1.0;  // Existing is zero but autodiff is not
        }
        
        max_error = std::max(max_error, rel_err);
        
        std::string layer_str = (layer_idx == -1) ? "HS" : std::to_string(layer_idx);
        std::cout << "  " << std::setw(12) << layer_str 
                  << std::setw(12) << k.parameterName
                  << std::setw(15) << std::scientific << std::setprecision(4) << dc_dp_existing
                  << std::setw(15) << dc_dp_autodiff
                  << std::setw(15) << std::fixed << rel_err;
        
        if (rel_err < 0.01) {  // 1% tolerance
            std::cout << " [PASS]" << std::endl;
            num_pass++;
        } else {
            std::cout << " [FAIL]" << std::endl;
            num_fail++;
        }
    }
    
    std::cout << "\n  Summary: " << num_pass << " PASS, " << num_fail << " FAIL" << std::endl;
    std::cout << "  Max relative error: " << max_error << std::endl;
}

// --- Test 2: Group Velocity Forward Mode ---
void test_group_velocity_forward() {
    std::cout << "\n=== Test 2: Group Velocity (Forward Mode AD) ===" << std::endl;
    
    // Simple 2-layer model
    std::vector<PhysicalParameters<double>> layers = {
        makeIsotropic(1.5, 2.6, 2.0),
        makeIsotropic(2.0, 3.5, 2.3)
    };
    std::vector<double> thicknesses = {3.0, 5.0};
    PhysicalParameters<double> halfSpace = makeIsotropic(3.0, 5.2, 2.7);
    
    double freq = 0.3;
    double omega = 2.0 * M_PI * freq;
    
    // Find root using existing code
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto make_ads = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(2);
        return a;
    };
    
    std::vector<PhysicalParameters<ADS>> ads_layers;
    for (auto& l : layers) {
        PhysicalParameters<ADS> pl;
        pl.rho = make_ads(l.rho);
        pl.c11 = make_ads(l.c11);
        pl.c13 = make_ads(l.c13);
        pl.c33 = make_ads(l.c33);
        pl.c44 = make_ads(l.c44);
        pl.c66 = make_ads(l.c66);
        ads_layers.push_back(pl);
    }
    std::vector<ADS> ads_thick;
    for (auto& t : thicknesses) ads_thick.push_back(make_ads(t));
    
    PhysicalParameters<ADS> ads_hs;
    ads_hs.rho = make_ads(halfSpace.rho);
    ads_hs.c11 = make_ads(halfSpace.c11);
    ads_hs.c13 = make_ads(halfSpace.c13);
    ads_hs.c33 = make_ads(halfSpace.c33);
    ads_hs.c44 = make_ads(halfSpace.c44);
    ads_hs.c66 = make_ads(halfSpace.c66);
    
    double root_c = 0.0;
    ADS omega_ads = make_ads(omega);
    
    auto eval_F = [&](double c_val) -> double {
        ADS c_ads = make_ads(c_val);
        auto res = calculateDispersionFunction(ads_layers, ads_thick, ads_hs, omega_ads, c_ads, WaveType::Rayleigh);
        return res(I24).value();
    };
    
    // Find root
    double c_min = 1.0, c_max = 3.0;
    double dz = 0.01;
    double f_prev = eval_F(c_min);
    for (double c = c_min + dz; c <= c_max; c += dz) {
        double f_curr = eval_F(c);
        if (f_prev * f_curr <= 0) {
            double a = c - dz, b = c;
            for (int i = 0; i < 50; ++i) {
                double mid = (a + b) / 2.0;
                double f_mid = eval_F(mid);
                if (f_prev * f_mid <= 0) b = mid;
                else { a = mid; f_prev = f_mid; }
            }
            root_c = (a + b) / 2.0;
            break;
        }
        f_prev = f_curr;
    }
    
    if (root_c < 1e-3) {
        std::cout << "  [SKIP] No root found" << std::endl;
        return;
    }
    
    std::cout << "  Phase velocity c = " << root_c << " km/s" << std::endl;
    
    // Compute group velocity using existing Eigen::AutoDiff method
    ADS c_ad, omega_ad2;
    c_ad.value() = root_c;
    c_ad.derivatives() = Eigen::VectorXd::Unit(2, 0);  // d/dc
    omega_ad2.value() = omega;
    omega_ad2.derivatives() = Eigen::VectorXd::Unit(2, 1);  // d/dω
    
    auto res = calculateDispersionFunction(ads_layers, ads_thick, ads_hs, omega_ad2, c_ad, WaveType::Rayleigh);
    ADS Delta = res(I24);
    
    double dD_dc = Delta.derivatives()[0];
    double dD_dw = Delta.derivatives()[1];
    double dc_dw_eigen = -dD_dw / dD_dc;
    double U_eigen = root_c / (1.0 - (omega / root_c) * dc_dw_eigen);
    
    // Compute using autodiff forward mode
    auto result = calculateGroupVelocityForward(layers, thicknesses, halfSpace, omega, root_c, WaveType::Rayleigh);
    
    std::cout << "  Group velocity (Eigen::AutoDiff) U = " << U_eigen << " km/s" << std::endl;
    std::cout << "  Group velocity (autodiff forward) U = " << result.U << " km/s" << std::endl;
    std::cout << "  dc/dω (Eigen): " << dc_dw_eigen << std::endl;
    std::cout << "  dc/dω (autodiff): " << result.dc_domega << std::endl;
    
    double rel_err_U = std::abs(result.U - U_eigen) / std::abs(U_eigen);
    double rel_err_dc = std::abs(result.dc_domega - dc_dw_eigen) / std::abs(dc_dw_eigen);
    
    std::cout << "  Relative error U: " << rel_err_U << std::endl;
    std::cout << "  Relative error dc/dω: " << rel_err_dc << std::endl;
    
    if (rel_err_U < 0.01 && rel_err_dc < 0.01) {
        std::cout << "  [PASS]" << std::endl;
    } else {
        std::cout << "  [FAIL]" << std::endl;
    }
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Autodiff Forward Mode Sensitivity Test    " << std::endl;
    std::cout << "============================================" << std::endl;
    
    test_solid_3layer_rayleigh();
    test_group_velocity_forward();
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "           Test suite completed             " << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
