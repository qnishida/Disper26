#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Include the source directly for validaiton testing (simplest for now to get templates)
// Just for the declarations and implementations.
// In a proper CMake setup we might link against the lib, but templates often require headers.
// Since disper26.cpp contains the implementation, we can include it if we guard the main.
// PRECONDITION: src/disper26.cpp must be guarded with #ifdef BUILD_MAIN
#include "../src/disper26.cpp"

int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;

    // Helper to make constants
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };

    // Homogeneous Half-Space Test
    // Vs = 1.0 km/s, Vp = 1.732 km/s (Poisson Ratio 0.25), Rho = 1.0
    // Theoretical Rayleigh Velocity c_R approx 0.9194 * Vs
    double Vs = 1.0;
    double Vp = std::sqrt(3.0) * Vs; 
    double Rho = 1.0;
    
    PhysicalParameters<ADS> halfSpace;
    halfSpace.rho = make(Rho);
    halfSpace.c44 = make(Rho * Vs * Vs);      // mu
    halfSpace.c11 = make(Rho * Vp * Vp);      // lambda + 2mu
    halfSpace.c33 = halfSpace.c11;            // Isotropic
    halfSpace.c13 = make(halfSpace.c11.value() - 2.0 * halfSpace.c44.value());
    halfSpace.c66 = halfSpace.c44; // Isotropic
    
    // No layers above
    std::vector<PhysicalParameters<ADS>> layers;
    std::vector<ADS> thicknesses;
    
    double freq = 1.0;
    ADS omega = make(2.0 * M_PI * freq);
    
    std::cout << "--- Half-Space Rayleigh Wave Test ---" << std::endl;
    std::cout << "Params: Vs=" << Vs << " Vp=" << Vp << " Rho=" << Rho << std::endl;
    std::cout << "Target: c_R approx " << 0.9194 * Vs << std::endl;
    std::cout << "Scanning 0.8 to 1.1..." << std::endl;
    
    std::cout << "c \t I12 \t I13 \t I14 \t I23 \t I24" << std::endl;
    for (double c = 0.8; c <= 1.1; c += 0.01) {
        ADS c_ads = make(c);
        Eigen::Matrix<ADS, 5, 1> res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_ads);
        std::cout << std::fixed << std::setprecision(4) << c << " \t " 
                  << std::scientific << res(I12).value() << " \t "
                  << res(I13).value() << " \t "
                  << res(I14).value() << " \t " 
                  << res(I23).value() << " \t "
                  << res(I24).value() << std::endl;
    }

    return 0;
}
