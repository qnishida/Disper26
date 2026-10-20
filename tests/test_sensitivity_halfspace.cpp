#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Include the engine
#include "../src/disper26.cpp"

// --- Helper Functions ---

// 1. Root Finder (Bisection/Newton)
// To keep it simple, let's use a scanning + bisection for this test, or just simple bisection.
// 1. Root Finder (Bisection/Newton)
// To keep it simple, let's use a scanning + bisection for this test, or just simple bisection.
double find_root_bisection(double bracket_min, double bracket_max, double Vs, double Vp, double Rho, double freq) {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Constant AD make function (no derivatives for root finding)
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    ADS omega = make(2.0 * M_PI * freq);
    
    PhysicalParameters<ADS> halfSpace;
    halfSpace.rho = make(Rho);
    halfSpace.c44 = make(Rho * Vs * Vs);
    halfSpace.c11 = make(Rho * Vp * Vp);
    halfSpace.c33 = halfSpace.c11;
    halfSpace.c13 = make(halfSpace.c11.value() - 2.0 * halfSpace.c44.value());
    halfSpace.c66 = halfSpace.c44; // Isotropic
    
    std::vector<PhysicalParameters<ADS>> layers;
    std::vector<ADS> thicknesses; // Empty

    double low = bracket_min;
    double high = bracket_max;
    double tol = 1e-8;
    
    for (int i=0; i<100; ++i) {
        double mid = (low + high) / 2.0;
        
        ADS c_low = make(low);
        ADS c_mid = make(mid);
        
        auto res_low = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_low);
        auto res_mid = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_mid);
        
        double f_low = res_low(I24).value();
        double f_mid = res_mid(I24).value();
        
        // If mid is close to zero
        if (std::abs(f_mid) < 1e-15) return mid;
        
        if (std::abs(high - low) < tol) return mid;
        
        if (f_low * f_mid < 0) {
            high = mid;
        } else {
            low = mid;
        }
    }
    return (low + high) / 2.0;
}

int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Test Params
    double Vs0 = 1.0;
    double Vp0 = 1.732;
    double Rho0 = 1.0;
    double freq = 1.0;
    
    // 1. Find exact root for base model
    std::cout << "Finding exact root C0..." << std::endl;
    double c0 = find_root_bisection(0.9, 0.95, Vs0, Vp0, Rho0, freq);
    std::cout << "Root c0 = " << std::setprecision(8) << c0 << std::endl;
    
    // 2. AD Calculation of dC/dVs
    // We need dF/dVs and dF/dC at (Vs0, c0).
    // Let's set up AD variables where:
    // Index 0: Vs
    // Index 1: c
    
    // Create params with derivatives
    int deriv_size = 2;
    auto make_ad = [&](double v, int idx) {
        ADS a; a.value() = v; 
        a.derivatives() = Eigen::VectorXd::Zero(deriv_size);
        if (idx >= 0) a.derivatives()[idx] = 1.0;
        return a;
    };
    
    ADS Vs_ad = make_ad(Vs0, 0); // Seed d/dVs = 1
    ADS Vp_ad = make_ad(Vp0, -1);
    ADS Rho_ad = make_ad(Rho0, -1);
    
    ADS c_ad = make_ad(c0, 1);   // Seed d/dc = 1
    
    ADS omega = make_ad(2.0 * M_PI * freq, -1);
    
    // Recalculate halfspace params based on Vs_ad
    PhysicalParameters<ADS> halfSpace;
    halfSpace.rho = Rho_ad;
    // c44 = rho * Vs^2
    halfSpace.c44 = Rho_ad * Vs_ad * Vs_ad; 
    // c11 = rho * Vp^2 (Vp is constant here? User asked sensitivity to Vs only?)
    // If Vp is fixed constant while Vs changes, then partial/partial Vs.
    // Assuming Vp, Rho constant.
    halfSpace.c11 = Rho_ad * Vp_ad * Vp_ad;
    halfSpace.c33 = halfSpace.c11;
    // c13 = c11 - 2*c44
    halfSpace.c13 = halfSpace.c11 - make_ad(2.0, -1) * halfSpace.c44;
    halfSpace.c66 = halfSpace.c44; // Isotropic
    
    std::vector<PhysicalParameters<ADS>> layers;
    std::vector<ADS> thicknesses;
    
    Eigen::Matrix<ADS, 5, 1> res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_ad);
    
    ADS F = res(I24);
    
    double dF_dVs = F.derivatives()[0];
    double dF_dc  = F.derivatives()[1];
    
    // Implicit theorem: dC/dVs = - (dF/dVs) / (dF/dc)
    double dC_dVs_AD = - dF_dVs / dF_dc;
    
    std::cout << "--- AD Calculation ---" << std::endl;
    std::cout << "dF/dVs: " << dF_dVs << std::endl;
    std::cout << "dF/dc : " << dF_dc << std::endl;
    std::cout << "dC/dVs (AD): " << dC_dVs_AD << std::endl;
    
    // 3. FD Calculation of dC/dVs
    double delta = 1e-4;
    // Find root for Vs0 + delta
    double c_plus = find_root_bisection(0.9, 0.95, Vs0 + delta, Vp0, Rho0, freq);
    // Find root for Vs0 - delta
    double c_minus = find_root_bisection(0.9, 0.95, Vs0 - delta, Vp0, Rho0, freq);
    
    double dC_dVs_FD = (c_plus - c_minus) / (2.0 * delta);
    
    std::cout << "--- FD Calculation ---" << std::endl;
    std::cout << "Vs + " << delta << " -> c = " << c_plus << std::endl;
    std::cout << "Vs - " << delta << " -> c = " << c_minus << std::endl;
    std::cout << "dC/dVs (FD): " << dC_dVs_FD << std::endl;
    
    std::cout << "--- Comparison ---" << std::endl;
    double diff = std::abs(dC_dVs_AD - dC_dVs_FD);
    std::cout << "Difference: " << diff << std::endl;
    std::cout << "Rel Error : " << diff / std::abs(dC_dVs_FD) << std::endl;
    
    return 0;
}
