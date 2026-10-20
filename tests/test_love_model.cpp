#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include "../src/disper26.cpp"

// Verification against dispLove.c results
// Model:
// 1. h=5, vs=2.9, rho=1.8 (Vp=4.988)
// 2. h=35, vs=3.9, rho=2.8 (Vp=6.708)
// 3. Half: vs=4.6, rho=3.3 (Vp=7.912)

int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Helper to make constants
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    std::vector<PhysicalParameters<ADS>> layers;
    std::vector<ADS> thicknesses;
    
    // Layer 1
    {
        double h=5.0; double vs=2.9; double rho=1.8; double vp=vs*1.72;
        PhysicalParameters<ADS> p;
        p.rho = make(rho);
        p.c44 = make(rho*vs*vs);
        p.c11 = make(rho*vp*vp);
        p.c33 = p.c11;
        p.c66 = p.c44; // Isotropic
        p.c13 = make(p.c11.value() - 2.0*p.c44.value());
        layers.push_back(p);
        thicknesses.push_back(make(h));
    }
    
    // Layer 2
    {
        double h=35.0; double vs=3.9; double rho=2.8; double vp=vs*1.72;
        PhysicalParameters<ADS> p;
        p.rho = make(rho);
        p.c44 = make(rho*vs*vs);
        p.c11 = make(rho*vp*vp);
        p.c33 = p.c11;
        p.c66 = p.c44; // Isotropic
        p.c13 = make(p.c11.value() - 2.0*p.c44.value());
        layers.push_back(p);
        thicknesses.push_back(make(h));
    }
    
    // Half Space
    PhysicalParameters<ADS> halfSpace;
    {
        double vs=4.6; double rho=3.3; double vp=vs*1.72;
        halfSpace.rho = make(rho);
        halfSpace.c44 = make(rho*vs*vs);
        halfSpace.c11 = make(rho*vp*vp);
        halfSpace.c33 = halfSpace.c11;
        halfSpace.c66 = halfSpace.c44; // Isotropic
        halfSpace.c13 = make(halfSpace.c11.value() - 2.0*halfSpace.c44.value());
    }
    
    double freq = 0.1;
    ADS omega = make(2.0 * M_PI * freq);
    
    // Reference: C = 3.79362
    
    // Scan around target
    std::cout << "--- Love Wave Verification (0.1 Hz) ---" << std::endl;
    std::cout << "Target C = 3.79362" << std::endl;
    
    // Use brents method to find root
    auto func = [&](double c_val) {
        ADS c_ads = make(c_val);
        auto res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_ads, WaveType::Love);
        return res(I24).value(); // Check Stress (Index 4)
    };
    
    std::cout << "Scanning C from 3.5 to 4.0:" << std::endl;
    for(double c = 3.5; c <= 4.0; c += 0.05) {
        double val = func(c);
        std::cout << "C=" << c << " Val=" << val << std::endl;
    }
    
    double c_found = brents_method(func, 3.7, 3.8); // Narrower range around target
    
    std::cout << "Found C = " << std::fixed << std::setprecision(5) << c_found << std::endl;
    
    if (std::abs(c_found - 3.79362) < 1e-3) {
        std::cout << "SUCCESS: Matches reference." << std::endl;
    } else {
        std::cout << "FAILURE: Mismatch." << std::endl;
    }
    
    return 0;
}
