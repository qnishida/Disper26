#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "disper26.cpp" // Includes necessary headers and sensitivity_utils.hpp

using namespace std::chrono;

int main() {
    std::cout << "Starting AD Performance Benchmark..." << std::endl;

    // 1. Model Setup (Same as verification test)
    // 3 Layers + Halfspace
    double omega = 2.0 * M_PI * 0.5; // f = 0.5 Hz
    
    // Layer 1
    auto l1 = createIsotropicParameters<double>(3.0, 5.2, 2.5);
    double h1 = 5.0;
    
    // Layer 2
    auto l2 = createIsotropicParameters<double>(3.5, 6.0, 2.7);
    double h2 = 10.0;
    
    // Layer 3
    auto l3 = createIsotropicParameters<double>(4.0, 7.0, 2.9);
    double h3 = 15.0;
    
    // Halfspace
    auto hs = createIsotropicParameters<double>(4.5, 8.0, 3.0);
    
    std::vector<PhysicalParameters<double>> layers = {l1, l2, l3};
    std::vector<double> thicknesses = {h1, h2, h3};

    // 2. Search Root (Pre-calculate to isolate kernel time)
    double c_min = 2.5;
    double c_max = 4.5;
    double root_c = brents_method([&](double c) {
        auto res = calculateDispersionFunction(layers, thicknesses, hs, omega, c, WaveType::Rayleigh);
        return res(I24);
    }, c_min, c_max);
    
    std::cout << "Root c = " << root_c << std::endl;

    int iterations = 1000;
    std::cout << "Running " << iterations << " iterations for each method..." << std::endl;

    // --- Benchmarking Hybrid AD ---
    auto start_hybrid = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        volatile auto k = computeSensitivityKernelsFD(layers, thicknesses, hs, 0.5, root_c, WaveType::Rayleigh);
    }
    auto end_hybrid = high_resolution_clock::now();
    auto dur_hybrid = duration_cast<milliseconds>(end_hybrid - start_hybrid).count();
    
    std::cout << "Hybrid AD Time: " << dur_hybrid << " ms" << std::endl;
    std::cout << "Hybrid AD per iter: " << (double)dur_hybrid / iterations << " ms" << std::endl;

    // --- Benchmarking Full AD ---
    auto start_full = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        volatile auto k = computeSensitivityKernelsFullAD(layers, thicknesses, hs, 0.5, root_c, WaveType::Rayleigh);
    }
    auto end_full = high_resolution_clock::now();
    auto dur_full = duration_cast<milliseconds>(end_full - start_full).count();
    
    std::cout << "Full AD Time: " << dur_full << " ms" << std::endl;
    std::cout << "Full AD per iter: " << (double)dur_full / iterations << " ms" << std::endl;

    // Comparison
    double ratio = (double)dur_full / (double)dur_hybrid;
    std::cout << "Ratio (Full / Hybrid): " << ratio << "x" << std::endl;

    return 0;
}
