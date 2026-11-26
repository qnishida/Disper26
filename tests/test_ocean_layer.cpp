#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Include the VTI Dispersion Engine
#include "../src/disper26.cpp"

// Create fluid layer parameters
template <typename T>
PhysicalParameters<T> createFluidParameters(T vp, T rho) {
    PhysicalParameters<T> p;
    p.rho = rho;
    p.c44 = rho * T(0.0);  // The fluid shear modulus is zero.
    p.c11 = rho * vp * vp;
    p.c33 = p.c11;
    p.c13 = p.c11;  // For acoustic: c13 = c11 (lambda)
    p.c66 = rho * T(0.0);  // No shear
    return p;
}

int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Helper to make constants
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    std::cout << "=== Ocean Layer Test ===" << std::endl;
    
    // --- Simple Ocean Model ---
    // Layer 0: ocean, 2 km thick
    // Layer 1: Sediment (solid)
    // Half-space: Basement (solid)
    
    std::vector<PhysicalParameters<ADS>> layers;
    std::vector<ADS> thicknesses;
    
    // Layer 0: Ocean (Vp = 1.5 km/s, Vs = 0, rho = 1.0 g/cm^3)
    double ocean_vp = 1.5;
    double ocean_rho = 1.0;
    layers.push_back(createFluidParameters(make(ocean_vp), make(ocean_rho)));
    thicknesses.push_back(make(2.0));  // 2 km depth
    
    // Layer 1: Sediment (Vp = 2.0, Vs = 0.5, rho = 1.8)
    double sed_vs = 0.5;
    double sed_vp = 2.0;
    double sed_rho = 1.8;
    layers.push_back(createIsotropicParameters(make(sed_vs), make(sed_vp), make(sed_rho)));
    thicknesses.push_back(make(1.0));  // 1 km
    
    // Half-space: Basement (Vp = 5.0, Vs = 3.0, rho = 2.5)
    double base_vs = 3.0;
    double base_vp = 5.0;
    double base_rho = 2.5;
    PhysicalParameters<ADS> halfSpace = createIsotropicParameters(make(base_vs), make(base_vp), make(base_rho));
    
    // Test frequency
    double freq = 0.5;  // Hz
    ADS omega = make(2.0 * M_PI * freq);
    
    std::cout << "Freq: " << freq << " Hz" << std::endl;
    std::cout << "Model: Ocean (2km) -> Sediment (1km) -> Basement" << std::endl;
    std::cout << std::endl;
    
    // Scan phase velocities
    std::cout << "c (km/s)\tI24 (secular)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    double prev_I24 = 0.0;
    for (double c = 0.3; c <= 3.0; c += 0.05) {
        ADS c_ads = make(c);
        Eigen::Matrix<ADS, 5, 1> res = calculateRayleighDispersion(layers, thicknesses, halfSpace, omega, c_ads);
        
        double I24_val = res(I24).value();
        
        // Check for zero crossing
        std::string marker = "";
        if (prev_I24 != 0.0 && prev_I24 * I24_val < 0) {
            marker = " <-- ROOT";
        }
        
        std::cout << std::fixed << std::setprecision(3) << c << "\t\t"
                  << std::scientific << std::setprecision(4) << I24_val
                  << marker << std::endl;
        
        prev_I24 = I24_val;
    }
    
    std::cout << std::endl;
    std::cout << "=== Scholte Mode Test (low velocity) ===" << std::endl;
    
    // Scholte mode root is between c=0.4 and c=0.45
    double c_low_scholte = 0.40, c_high_scholte = 0.45;
    for (int iter = 0; iter < 50; ++iter) {
        double c_mid = (c_low_scholte + c_high_scholte) / 2.0;
        ADS c_ads = make(c_mid);
        Eigen::Matrix<ADS, 5, 1> res = calculateRayleighDispersion(layers, thicknesses, halfSpace, omega, c_ads);
        if (res(I24).value() < 0) {
            c_high_scholte = c_mid;
        } else {
            c_low_scholte = c_mid;
        }
    }
    double c_scholte = (c_low_scholte + c_high_scholte) / 2.0;
    std::cout << "Scholte mode root at c = " << std::fixed << std::setprecision(6) << c_scholte << " km/s" << std::endl;
    
    // Verify convergence
    {
        ADS c_test = make(c_scholte);
        Eigen::Matrix<ADS, 5, 1> res = calculateRayleighDispersion(layers, thicknesses, halfSpace, omega, c_test);
        std::cout << "F(c_scholte) = " << std::scientific << res(I24).value() << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== AD-based Group Velocity and Sensitivity Test ===" << std::endl;
    
    // Find a root more precisely using bisection
    // Use the c=1.65 to c=1.70 range where we saw a sign change
    double c_low = 1.65, c_high = 1.70;
    for (int iter = 0; iter < 50; ++iter) {
        double c_mid = (c_low + c_high) / 2.0;
        ADS c_ads = make(c_mid);
        Eigen::Matrix<ADS, 5, 1> res = calculateRayleighDispersion(layers, thicknesses, halfSpace, omega, c_ads);
        if (res(I24).value() < 0) {
            c_high = c_mid;
        } else {
            c_low = c_mid;
        }
    }
    double c_root = (c_low + c_high) / 2.0;
    std::cout << "Found root at c = " << std::fixed << std::setprecision(6) << c_root << " km/s" << std::endl;
    
    // Now compute derivatives for group velocity
    // U = c / (1 + (omega/c) * (dF/domega) / (dF/dc))
    // F is the fluid free-surface traction residual stored in res(I24).
    
    // Set up AD variables: Index 0 = omega, Index 1 = c
    int deriv_size = 2;
    auto make_ad = [&](double v, int idx) {
        ADS a; a.value() = v; 
        a.derivatives() = Eigen::VectorXd::Zero(deriv_size);
        if (idx >= 0) a.derivatives()[idx] = 1.0;
        return a;
    };
    
    ADS omega_ad = make_ad(2.0 * M_PI * freq, 0);  // Seed d/domega
    ADS c_ad = make_ad(c_root, 1);                  // Seed d/dc
    
    // Recreate layers with AD
    std::vector<PhysicalParameters<ADS>> layers_ad;
    std::vector<ADS> thicknesses_ad;
    
    layers_ad.push_back(createFluidParameters(make_ad(ocean_vp, -1), make_ad(ocean_rho, -1)));
    thicknesses_ad.push_back(make_ad(2.0, -1));
    
    layers_ad.push_back(createIsotropicParameters(make_ad(sed_vs, -1), make_ad(sed_vp, -1), make_ad(sed_rho, -1)));
    thicknesses_ad.push_back(make_ad(1.0, -1));
    
    PhysicalParameters<ADS> halfSpace_ad = createIsotropicParameters(make_ad(base_vs, -1), make_ad(base_vp, -1), make_ad(base_rho, -1));
    
    Eigen::Matrix<ADS, 5, 1> res = calculateRayleighDispersion(layers_ad, thicknesses_ad, halfSpace_ad, omega_ad, c_ad);
    
    double F_val = res(I24).value();
    double dF_domega = res(I24).derivatives()[0];
    double dF_dc = res(I24).derivatives()[1];
    
    std::cout << "F(c_root) = " << std::scientific << F_val << std::endl;
    std::cout << "dF/domega = " << dF_domega << std::endl;
    std::cout << "dF/dc     = " << dF_dc << std::endl;
    
    // Group velocity: U = -dF/domega / dF/dc * c^2 / omega ??? No.
    // Actually: U = c / (1 - (c/omega) * (dF/dw) / (dF/dc))
    // Or more directly: U = dw/dk where F(k, w) = 0
    // dF/dk * dk + dF/dw * dw = 0 => dw/dk = -dF/dk / dF/dw
    // k = omega/c, so dF/dk = -c^2/omega * dF/dc + ...
    // Let's use: U = c * (1 + omega * dF_domega / (c * dF_dc))^(-1)
    
    double U = c_root / (1.0 + (omega.value() / c_root) * dF_domega / dF_dc);
    std::cout << "Group velocity U = " << std::fixed << std::setprecision(4) << U << " km/s" << std::endl;
    
    // === Sensitivity Kernel Test ===
    std::cout << std::endl;
    std::cout << "=== Sensitivity Kernel Test (d c / d Vp_ocean) ===" << std::endl;
    
    // Set up AD with ocean Vp as the variable
    // Index 0 = omega, Index 1 = c, Index 2 = Vp_ocean
    int sens_deriv_size = 3;
    auto make_sens = [&](double v, int idx) {
        ADS a; a.value() = v; 
        a.derivatives() = Eigen::VectorXd::Zero(sens_deriv_size);
        if (idx >= 0) a.derivatives()[idx] = 1.0;
        return a;
    };
    
    ADS omega_sens = make_sens(2.0 * M_PI * freq, 0);
    ADS c_sens = make_sens(c_root, 1);
    ADS vp_ocean_sens = make_sens(ocean_vp, 2);  // Seed d/d(Vp_ocean)
    
    std::vector<PhysicalParameters<ADS>> layers_sens;
    std::vector<ADS> thicknesses_sens;
    
    // Ocean layer with Vp as AD variable
    layers_sens.push_back(createFluidParameters(vp_ocean_sens, make_sens(ocean_rho, -1)));
    thicknesses_sens.push_back(make_sens(2.0, -1));
    
    // Other layers without sensitivity
    layers_sens.push_back(createIsotropicParameters(make_sens(sed_vs, -1), make_sens(sed_vp, -1), make_sens(sed_rho, -1)));
    thicknesses_sens.push_back(make_sens(1.0, -1));
    
    PhysicalParameters<ADS> halfSpace_sens = createIsotropicParameters(make_sens(base_vs, -1), make_sens(base_vp, -1), make_sens(base_rho, -1));
    
    Eigen::Matrix<ADS, 5, 1> res_sens = calculateRayleighDispersion(layers_sens, thicknesses_sens, halfSpace_sens, omega_sens, c_sens);
    
    double dF_dc_sens = res_sens(I24).derivatives()[1];
    double dF_dVp = res_sens(I24).derivatives()[2];
    
    // Using implicit function theorem: dc/dVp = -dF/dVp / dF/dc
    double dc_dVp = -dF_dVp / dF_dc_sens;
    
    std::cout << "dF/dVp_ocean = " << std::scientific << dF_dVp << std::endl;
    std::cout << "dc/dVp_ocean = " << dc_dVp << std::endl;
    std::cout << "(This is the phase velocity sensitivity to ocean Vp)" << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;
    
    return 0;
}
