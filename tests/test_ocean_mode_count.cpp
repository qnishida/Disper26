#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>

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
    std::cout << "=== Mode Counting for Ocean Model ===" << std::endl;
    
    // Simple ocean model: 2km ocean + sediment + basement
    std::vector<PhysicalParameters<double>> layers;
    std::vector<double> thicknesses;
    
    // Layer 0: ocean
    layers.push_back(createFluidParameters(1.5, 1.0));
    thicknesses.push_back(2.0);
    
    // Layer 1: Sediment (solid)
    layers.push_back(createIsotropicParameters(0.5, 2.0, 1.8));
    thicknesses.push_back(1.0);
    
    // Half-space
    PhysicalParameters<double> halfSpace = createIsotropicParameters(3.0, 5.0, 2.5);
    
    double freq = 0.5;
    double omega = 2.0 * M_PI * freq;
    
    std::cout << "Freq: " << freq << " Hz" << std::endl;
    std::cout << std::endl;
    
    // Scan phase velocities and show mode count
    std::cout << "c (km/s)\tMode Count\tI24" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    for (double c = 0.4; c <= 3.0; c += 0.1) {
        double mode = calculateContinuousModeCount(layers, thicknesses, halfSpace, omega, c, WaveType::Rayleigh);
        int mode_int = calculateTotalModeCount(layers, thicknesses, halfSpace, omega, c, WaveType::Rayleigh);
        
        // Also get secular function
        Eigen::Matrix<double, 5, 1> res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c, WaveType::Rayleigh);
        double I24_val = res(I24);
        
        std::cout << std::fixed << std::setprecision(2) << c << "\t\t"
                  << mode_int << " (" << std::setprecision(2) << mode << ")\t\t"
                  << std::scientific << std::setprecision(2) << I24_val << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;
    
    return 0;
}
