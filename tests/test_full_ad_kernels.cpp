
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "disper26.cpp"
#include "sensitivity_utils.hpp"

// Test comparing Full AD Group Velocity Kernels vs Hybrid AD
int main() {
    std::cout << "Starting Full AD Kernel Verification..." << std::endl;

    // 1. Define Model (3-Layer Solid)
    // Layer 1
    PhysicalParameters<double> p1 = createIsotropicParameters(1.0, 2.0, 1.0); // Vs=1, Vp=2, Rho=1
    double h1 = 1.0; 
    
    // Layer 2
    PhysicalParameters<double> p2 = createIsotropicParameters(1.5, 3.0, 1.2); // Vs=1.5
    double h2 = 2.0;

    // Halfspace
    PhysicalParameters<double> hs = createIsotropicParameters(2.0, 4.0, 1.5); // Vs=2.0

    std::vector<PhysicalParameters<double>> layers = {p1, p2};
    std::vector<double> thicknesses = {h1, h2};
    std::vector<double> th_full = {h1, h2, 0.0}; // Used for some internal funcs

    double freq = 0.5;
    double omega = 2.0 * M_PI * freq;
    
    // Initial Root Search
    auto secular_func = [&](double c) {
        auto res = calculateDispersionFunction(layers, thicknesses, hs, omega, c, WaveType::Rayleigh, NormalizationMode::VectorNorm);
        return res(I24);
    };
    std::vector<double> roots = searchRoots(secular_func, 0.9*1.0, 1.1*2.0, 0.05);

    if (roots.empty()) {
        std::cerr << "No root found!" << std::endl;
        return 1;
    }
    double root_c = roots[0];
    std::cout << "Root c = " << root_c << " at f=" << freq << std::endl;

    // 2. Compute Hybrid AD Kernels (Baseline)
    std::cout << "\nComputing Hybrid AD Kernels..." << std::endl;
    auto hybrid_kernels = computeSensitivityKernelsFD(layers, thicknesses, hs, freq, root_c, WaveType::Rayleigh, SensitivityMode::Isotropic);

    // 3. Compute Full AD Kernels (New)
    std::cout << "Computing Full AD Kernels..." << std::endl;
    auto full_ad_kernels = computeSensitivityKernelsFullAD(layers, thicknesses, hs, freq, root_c, WaveType::Rayleigh, SensitivityMode::Isotropic);

    // 4. Compare Group Velocity Kernels
    std::cout << "\nComparison of dU/dp (Full AD vs Hybrid):" << std::endl;
    std::cout << std::left << std::setw(10) << "Param" << std::setw(30) << "Hybrid dU/dp" << std::setw(30) << "Full AD dU/dp" << std::setw(15) << "Diff" << std::endl;
    
    bool all_pass = true;
    for (size_t i = 0; i < hybrid_kernels.size(); ++i) {
        double val_hybrid = hybrid_kernels[i].d_GroupVelocity;
        double val_full = full_ad_kernels[i].d_GroupVelocity;
        double diff = std::abs(val_hybrid - val_full);
        
        std::cout << std::left << std::setw(10) << (hybrid_kernels[i].parameterName + "_" + std::to_string(hybrid_kernels[i].layerIndex))
                  << std::setw(30) << val_hybrid 
                  << std::setw(30) << val_full 
                  << std::setw(15) << diff;
                  
        if (diff > 1e-4) {
            std::cout << " [FAIL]";
            all_pass = false;
        } else {
            std::cout << " [PASS]";
        }
        std::cout << std::endl;
    }

    // 4b. Compare H/V Ratio Kernels
    std::cout << "\nComparison of d(H/V)/dp (Full AD vs Hybrid):" << std::endl;
    std::cout << std::left << std::setw(10) << "Param" << std::setw(30) << "Hybrid d(H/V)/dp" << std::setw(30) << "Full AD d(H/V)/dp" << std::setw(15) << "Diff" << std::endl;
    
    for (size_t i = 0; i < hybrid_kernels.size(); ++i) {
        double val_hybrid = hybrid_kernels[i].d_HVRatio;
        double val_full = full_ad_kernels[i].d_HVRatio;
        double diff = std::abs(val_hybrid - val_full);
        
        std::cout << std::left << std::setw(10) << (hybrid_kernels[i].parameterName + "_" + std::to_string(hybrid_kernels[i].layerIndex))
                  << std::setw(30) << val_hybrid 
                  << std::setw(30) << val_full 
                  << std::setw(15) << diff;
                  
        if (diff > 1e-4) {
            std::cout << " [FAIL]";
            all_pass = false;
        } else {
            std::cout << " [PASS]";
        }
        std::cout << std::endl;
    }

    // 5. Test dC/dOmega
    std::cout << "\nTesting dC/dOmega AD..." << std::endl;
    double dc_dw_ad = calculateDcDOmegaAD(layers, thicknesses, hs, omega, root_c, WaveType::Rayleigh);
    
    // Finite Difference Check
    double df = 1e-4;
    double w_plus = 2.0 * M_PI * (freq + df);
    double w_minus = 2.0 * M_PI * (freq - df);
    
    // Scan for new roots nearby
    auto func_p = [&](double c) { return calculateDispersionFunction(layers, thicknesses, hs, w_plus, c).operator()(I24); };
    auto func_m = [&](double c) { return calculateDispersionFunction(layers, thicknesses, hs, w_minus, c).operator()(I24); };
    
    double root_p = brents_method(func_p, root_c-0.1, root_c+0.1, 1e-9, 100);
    double root_m = brents_method(func_m, root_c-0.1, root_c+0.1, 1e-9, 100);
    
    double dc_dw_fd = (root_p - root_m) / (2.0 * 2.0 * M_PI * df);
    
    std::cout << "dC/dw (AD): " << dc_dw_ad << std::endl;
    std::cout << "dC/dw (FD): " << dc_dw_fd << std::endl;
    if (std::abs(dc_dw_ad - dc_dw_fd) < 1e-5) {
        std::cout << "dC/dw Check: PASS" << std::endl;
    } else {
        std::cout << "dC/dw Check: FAIL" << std::endl;
        all_pass = false;
    }

    return all_pass ? 0 : 1;
}
