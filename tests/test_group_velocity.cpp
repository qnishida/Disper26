#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Include the engine
#include "../src/disper26.cpp"

// --- Root Finder ---
double find_root_bisection(double bracket_min, double bracket_max, 
                           const std::vector<PhysicalParameters<Eigen::AutoDiffScalar<Eigen::VectorXd>>>& layers,
                           const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>>& thicknesses,
                           const PhysicalParameters<Eigen::AutoDiffScalar<Eigen::VectorXd>>& halfSpace,
                           double freq) {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Constant AD make function (no derivatives for root finding)
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(2); return a;
    };
    
    ADS omega = make(2.0 * M_PI * freq);
    
    double low = bracket_min;
    double high = bracket_max;
    double tol = 1e-10;
    
    for (int i=0; i<100; ++i) {
        double mid = (low + high) / 2.0;
        
        ADS c_low = make(low);
        ADS c_mid = make(mid);
        
        auto res_low = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_low);
        auto res_mid = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_mid);
        
        double f_low = res_low(I24).value();
        double f_mid = res_mid(I24).value();
        
        if (std::abs(f_mid) < 1e-20) return mid; // Strict tolerance
        if (std::abs(high - low) < tol) return mid;
        
        // Handle large values/NaNs by checking sign changes properly
        // Assuming smooth function in bracket
        if (std::isnan(f_low) || std::isnan(f_mid)) {
             // scan forward if nan
        }
        
        // Debug
        // std::cout << "Bisect: " << low << " " << high << " f:" << f_mid << std::endl;

        if (f_low * f_mid <= 0) {
            high = mid;
        } else {
            low = mid;
        }
    }
    return (low + high) / 2.0;
}

int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;

    // Helper to make constants with derivative size 2 (Index 0: c, Index 1: omega)
    // Actually for physical params they are constants w.r.t (c, omega)
    int deriv_size = 2;
    auto make_const = [&](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(deriv_size); return a;
    };

    // 20-Layer Model Configuration
    std::vector<PhysicalParameters<ADS>> layers;
    std::vector<ADS> thicknesses;
    
    for (int i = 1; i <= 19; ++i) {
        double vs_val = 0.0;
        double h_val = 1.0;
        
        if (i <= 2) {
            vs_val = 1.5;
        } else if (i <= 5) { // 3, 4, 5
            vs_val = 2.5 + (i - 3.0) * 0.16;
        } else { // 6 to 19
            vs_val = 3.4;
        }
        
        double vp_val = vs_val * 1.72;
        double rho_val = 0.9473 * 0.2966 * vp_val;
        
        // Isotropic -> VTI
        PhysicalParameters<ADS> p;
        p.rho = make_const(rho_val);
        // c44 = rho * Vs^2
        p.c44 = make_const(rho_val * vs_val * vs_val);
        // c11 = rho * Vp^2
        p.c11 = make_const(rho_val * vp_val * vp_val);
        // c33 = c11
        p.c33 = p.c11;
        // c13 = c11 - 2*c44
        p.c13 = make_const(p.c11.value() - 2.0 * p.c44.value());
        
        layers.push_back(p);
        thicknesses.push_back(make_const(h_val));
    }
    
    // Half Space (Layer 20)
    PhysicalParameters<ADS> halfSpace;
    {
        double vs_val = 4.6;
        double vp_val = vs_val * 1.72;
        double rho_val = 0.9473 * 0.2966 * vp_val;
        
        halfSpace.rho = make_const(rho_val);
        halfSpace.c44 = make_const(rho_val * vs_val * vs_val);
        halfSpace.c11 = make_const(rho_val * vp_val * vp_val);
        halfSpace.c33 = halfSpace.c11;
        halfSpace.c13 = make_const(halfSpace.c11.value() - 2.0 * halfSpace.c44.value());
    }

    // Target Frequency
    double freq = 0.4;
    double omega_val = 2.0 * M_PI * freq;
    
    // 1. Find Phase Velocity Root (c)
    // From previous test, root is approx 1.53023
    std::cout << "Finding root..." << std::endl;
    // Widen bracket to ensure capture
    double c_root = find_root_bisection(1.40, 1.70, layers, thicknesses, halfSpace, freq);
    std::cout << "Phase Velocity c = " << std::setprecision(8) << c_root << " km/s" << std::endl;
    
    // 2. Compute Derivatives using AD
    // Variables: Index 0 = c, Index 1 = omega
    ADS c_ad;
    c_ad.value() = c_root;
    c_ad.derivatives() = Eigen::VectorXd::Zero(deriv_size);
    c_ad.derivatives()[0] = 1.0; // d/dc
    
    ADS omega_ad;
    omega_ad.value() = omega_val;
    omega_ad.derivatives() = Eigen::VectorXd::Zero(deriv_size);
    omega_ad.derivatives()[1] = 1.0; // d/domega
    
    // Run dispersion function with derivatives
    Eigen::Matrix<ADS, 5, 1> res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega_ad, c_ad);
    
    // Extract characteristic function Delta = Y_24
    ADS Delta = res(I24);
    
    double Delta_val = Delta.value();
    double dDelta_dc = Delta.derivatives()[0];
    double dDelta_dw = Delta.derivatives()[1];
    
    std::cout << "Delta Val: " << Delta_val << " (Should be near 0)" << std::endl;
    std::cout << "dDelta/dc: " << dDelta_dc << std::endl;
    std::cout << "dDelta/dw: " << dDelta_dw << std::endl;
    
    // 3. Compute Group Velocity U
    // Implicit Theorem: dc/dw = - (dDelta/dw) / (dDelta/dc)
    // U = c / (1 - (w/c)*(dc/dw))  [Standard definition U = dw/dk]
    //   = c / (1 - (w/c)*(-dDelta/dw / dDelta/dc))
    //   = c / (1 + (w/c)*(dDelta/dw / dDelta/dc))
    
    double dc_dw = - dDelta_dw / dDelta_dc;
    double U = c_root / (1.0 - (omega_val / c_root) * dc_dw);
    
    std::cout << "dc/dw: " << dc_dw << std::endl;
    std::cout << "Group Velocity U = " << U << " km/s" << std::endl;
    
    // For comparison with Saito's formula style:
    // YC = c * dDelta/dc
    // YW = w * dDelta/dw
    // U = c / (1 + YW/YC)
    double YC = c_root * dDelta_dc;
    double YW = omega_val * dDelta_dw;
    double U_saito = c_root / (1.0 + YW/YC);
    
    std::cout << "U (Saito Formula Check) = " << U_saito << " km/s" << std::endl;

    return 0;
}
