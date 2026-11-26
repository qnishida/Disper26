#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include "../src/disper26.cpp"

// Create fluid layer parameters
template <typename T>
PhysicalParameters<T> createFluidParameters(T vp, T rho) {
    PhysicalParameters<T> p;
    p.rho = rho;
    p.c44 = rho * T(0.0);
    p.c11 = rho * vp * vp;
    p.c33 = p.c11;
    p.c13 = p.c11;
    p.c66 = rho * T(0.0);
    return p;
}

int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    std::cout << "=== Multi-Layer Ocean Test (SOFAR Channel) ===" << std::endl;
    
    // Model: Stratified ocean with SOFAR-like sound speed profile
    // Layer 0: Surface layer (warm, Vp = 1.52 km/s)
    // Layer 1: Thermocline (Vp decreases, 1.48 km/s)
    // Layer 2: SOFAR axis (minimum Vp, 1.46 km/s)
    // Layer 3: Deep water (Vp increases, 1.50 km/s)
    // Layer 4: Bottom water (Vp = 1.52 km/s)
    // Layer 5: Sediment (solid)
    // Half-space: Basement
    
    std::vector<PhysicalParameters<ADS>> layers;
    std::vector<ADS> thicknesses;
    
    // Ocean layers (5 fluid layers with varying Vp)
    double ocean_rho = 1.025;  // Standard seawater
    
    layers.push_back(createFluidParameters(make(1.52), make(ocean_rho)));  // Surface
    thicknesses.push_back(make(0.2));  // 200 m
    
    layers.push_back(createFluidParameters(make(1.48), make(ocean_rho)));  // Thermocline
    thicknesses.push_back(make(0.5));  // 500 m
    
    layers.push_back(createFluidParameters(make(1.46), make(ocean_rho)));  // SOFAR axis
    thicknesses.push_back(make(0.5));  // 500 m
    
    layers.push_back(createFluidParameters(make(1.50), make(ocean_rho)));  // Deep
    thicknesses.push_back(make(1.0));  // 1 km
    
    layers.push_back(createFluidParameters(make(1.52), make(ocean_rho)));  // Bottom water
    thicknesses.push_back(make(1.0));  // 1 km
    
    // Sediment layer (solid)
    layers.push_back(createIsotropicParameters(make(0.3), make(1.8), make(1.6)));
    thicknesses.push_back(make(0.5));  // 500 m
    
    // Half-space: Basement
    PhysicalParameters<ADS> halfSpace = createIsotropicParameters(make(2.5), make(4.5), make(2.4));
    
    std::cout << "Ocean depth: 3.2 km (5 layers)" << std::endl;
    std::cout << "SOFAR axis at 0.7-1.2 km depth (Vp_min = 1.46 km/s)" << std::endl;
    std::cout << "Sediment: 0.5 km (Vs = 0.3 km/s)" << std::endl;
    std::cout << std::endl;
    
    // Test frequency
    double freq = 0.2;  // Lower frequency for deeper structure
    ADS omega = make(2.0 * M_PI * freq);
    
    std::cout << "Freq: " << freq << " Hz" << std::endl;
    std::cout << std::endl;
    
    // Scan phase velocities
    std::cout << "c (km/s)\tI24 (secular)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    double prev_I24 = 0.0;
    int root_count = 0;
    for (double c = 0.2; c <= 2.5; c += 0.02) {
        ADS c_ads = make(c);
        Eigen::Matrix<ADS, 5, 1> res = calculateRayleighDispersion(layers, thicknesses, halfSpace, omega, c_ads);
        
        double I24_val = res(I24).value();
        
        std::string marker = "";
        if (prev_I24 != 0.0 && prev_I24 * I24_val < 0) {
            marker = " <-- ROOT";
            root_count++;
        }
        
        std::cout << std::fixed << std::setprecision(2) << c << "\t\t"
                  << std::scientific << std::setprecision(3) << I24_val
                  << marker << std::endl;
        
        prev_I24 = I24_val;
    }
    
    std::cout << std::endl;
    std::cout << "Total roots detected: " << root_count << std::endl;
    
    // Test AD at a specific root (use c=0.52-0.54 range where we have a root)
    std::cout << std::endl;
    std::cout << "=== AD Test at Scholte-like mode (c ~ 0.53 km/s) ===" << std::endl;
    
    double c_low = 0.52, c_high = 0.54;
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
    
    // Compute group velocity with AD
    int deriv_size = 2;
    auto make_ad = [&](double v, int idx) {
        ADS a; a.value() = v; 
        a.derivatives() = Eigen::VectorXd::Zero(deriv_size);
        if (idx >= 0) a.derivatives()[idx] = 1.0;
        return a;
    };
    
    ADS omega_ad = make_ad(2.0 * M_PI * freq, 0);
    ADS c_ad = make_ad(c_root, 1);
    
    std::vector<PhysicalParameters<ADS>> layers_ad;
    std::vector<ADS> thicknesses_ad;
    
    layers_ad.push_back(createFluidParameters(make_ad(1.52, -1), make_ad(ocean_rho, -1)));
    thicknesses_ad.push_back(make_ad(0.2, -1));
    layers_ad.push_back(createFluidParameters(make_ad(1.48, -1), make_ad(ocean_rho, -1)));
    thicknesses_ad.push_back(make_ad(0.5, -1));
    layers_ad.push_back(createFluidParameters(make_ad(1.46, -1), make_ad(ocean_rho, -1)));
    thicknesses_ad.push_back(make_ad(0.5, -1));
    layers_ad.push_back(createFluidParameters(make_ad(1.50, -1), make_ad(ocean_rho, -1)));
    thicknesses_ad.push_back(make_ad(1.0, -1));
    layers_ad.push_back(createFluidParameters(make_ad(1.52, -1), make_ad(ocean_rho, -1)));
    thicknesses_ad.push_back(make_ad(1.0, -1));
    layers_ad.push_back(createIsotropicParameters(make_ad(0.3, -1), make_ad(1.8, -1), make_ad(1.6, -1)));
    thicknesses_ad.push_back(make_ad(0.5, -1));
    
    PhysicalParameters<ADS> halfSpace_ad = createIsotropicParameters(make_ad(2.5, -1), make_ad(4.5, -1), make_ad(2.4, -1));
    
    Eigen::Matrix<ADS, 5, 1> res = calculateRayleighDispersion(layers_ad, thicknesses_ad, halfSpace_ad, omega_ad, c_ad);
    
    double F_val = res(I24).value();
    double dF_domega = res(I24).derivatives()[0];
    double dF_dc = res(I24).derivatives()[1];
    double U = c_root / (1.0 + (omega.value() / c_root) * dF_domega / dF_dc);
    
    std::cout << "Found root at c = " << std::fixed << std::setprecision(6) << c_root << " km/s" << std::endl;
    std::cout << "F(c_root) = " << std::scientific << F_val << std::endl;
    std::cout << "Group velocity U = " << std::fixed << std::setprecision(4) << U << " km/s" << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;
    
    return 0;
}
