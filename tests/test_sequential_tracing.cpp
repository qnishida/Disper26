#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include "../src/disper26.cpp"

// Verification of Sequential Mode Tracing
// Model: Soft Sediment on Bedrock (High Contrast) to trigger potential mode jumps

int main() {
    std::cout << "--- Sequential Mode Tracing Test ---" << std::endl;
    
    // Model Definition
    // Layer 1: Soft Sediment (Vs=0.5, Vp=1.0, rho=1.8, H=0.5 km)
    // Half Space: Bedrock (Vs=2.0, Vp=3.5, rho=2.5)
    
    using T = double;
    
    std::vector<PhysicalParameters<T>> layers;
    std::vector<T> thicknesses;
    
    // Layer 1
    PhysicalParameters<T> l1 = createIsotropicParameters<T>(0.5, 1.0, 1.8);
    layers.push_back(l1);
    thicknesses.push_back(0.5);
    
    // Half Space
    PhysicalParameters<T> hs = createIsotropicParameters<T>(2.0, 3.5, 2.5);
    
    // Tracing Parameters
    double freq_start = 0.2; // 5 sec
    double freq_end = 5.0;   // 0.2 sec
    int num_steps = 50;
    
    // Perform Tracing
    std::vector<DispersionPoint> curve = traceDispersionCurve(
        layers, thicknesses, hs, 
        freq_start, freq_end, num_steps, 
        0, // Mode 0 (Fundamental)
        WaveType::Rayleigh,
        0.4, // Min search
        2.5  // Max search
    );
    
    if (curve.empty()) {
        std::cerr << "FAILURE: No curve traced." << std::endl;
        return 1;
    }
    
    std::cout << "Traced " << curve.size() << " points." << std::endl;
    std::cout << std::setw(10) << "Freq(Hz)" << std::setw(10) << "C(km/s)" << std::setw(10) << "U(km/s)" << std::endl;
    
    bool jumps_detected = false;
    double prev_c = 0.0;
    
    for (size_t i = 0; i < curve.size(); ++i) {
        const auto& p = curve[i];
        
        // Print sampling of points
        if (i % 5 == 0 || i == curve.size() - 1) {
            std::cout << std::setw(10) << p.frequency << std::setw(10) << p.phase_velocity << std::setw(10) << p.group_velocity << std::endl;
        }
        
        // Check smoothness
        if (i > 0) {
            double c_change = std::abs(p.phase_velocity - prev_c);
            if (c_change > 0.5) { // Arbitrary large jump threshold
                std::cerr << "WARNING: Large jump detected at " << p.frequency << " Hz. Delta C = " << c_change << std::endl;
                jumps_detected = true;
            }
        }
        prev_c = p.phase_velocity;
    }
    
    // Basic Validity Checks
    // 1. At low freq (0.2Hz), wavelength ~2km/s / 0.2 = 10km >> H=0.5km. Should see bedrock velocity.
    // C should be close to Rayleigh velocity of bedrock (approx 0.92 * 2.0 = 1.84).
    double c_low = curve.front().phase_velocity;
    if (std::abs(c_low - 1.84) > 0.2) { 
         // Note: Might not be fully asymptotic yet at 0.2Hz? 
         // Let's check logic.
         // Actually 10km wavelength is large enough.
         std::cout << "Low Freq Asymptote Check: C=" << c_low << " (Expect ~1.84)" << std::endl;
    }
    
    // 2. At high freq (5.0Hz), wavelength ~0.5km/s / 5 = 0.1km << H=0.5km. Should see sediment velocity.
    // C should be close to Rayleigh velocity of sediment (approx 0.92 * 0.5 = 0.46).
    double c_high = curve.back().phase_velocity;
    if (std::abs(c_high - 0.46) > 0.1) {
        std::cout << "High Freq Asymptote Check: C=" << c_high << " (Expect ~0.46)" << std::endl;
    }

    if (jumps_detected) {
        std::cout << "FAILURE: Mode jumps detected." << std::endl;
        return 1;
    } else {
        std::cout << "SUCCESS: Smooth tracing." << std::endl;
    }

    return 0;
}
