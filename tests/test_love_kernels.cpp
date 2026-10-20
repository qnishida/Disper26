#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include "../src/disper26.cpp"

// Verify Love Wave Kernels
int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    // Model: 1 Layer over HalfSpace
    // Layer 1: H=10, Vs=3.0, Rho=2.5
    // Half: Vs=4.0, Rho=3.0
    
    std::vector<PhysicalParameters<double>> layers;
    std::vector<double> thicknesses;
    
    {
        double h=10.0; double vs=3.0; double rho=2.5; double vp=vs*1.732;
        PhysicalParameters<double> p;
        p.rho = rho; p.c44 = rho*vs*vs; p.c11 = rho*vp*vp; p.c33 = p.c11; p.c66 = p.c44; p.c13 = p.c11 - 2*p.c44;
        layers.push_back(p);
        thicknesses.push_back(h);
    }
    
    PhysicalParameters<double> halfSpace;
    {
        double vs=4.0; double rho=3.0; double vp=vs*1.732;
        halfSpace.rho = rho; halfSpace.c44 = rho*vs*vs; halfSpace.c11 = rho*vp*vp; halfSpace.c33 = halfSpace.c11; halfSpace.c66 = halfSpace.c44; halfSpace.c13 = halfSpace.c11 - 2*halfSpace.c44;
    }
    
    double freq = 0.2;
    // Find initial root
    ADS omega = make(2.0 * M_PI * freq);
    
    auto func = [&](double c_val) {
         // Manual call to check root
        ADS c_ads = make(c_val);
        // We need to convert double params to ADS manually if calling calculateDispersion directly,
        // OR rely on find_root_local in kernels.
        // But for main root finding, we can use a helper.
        // Let's use brents_method with double -> ADS wrapper
        std::vector<PhysicalParameters<ADS>> l_ads;
        std::vector<ADS> t_ads;
        PhysicalParameters<ADS> h_ads;
        // ... (conversion skipped for brevity, lets just guess root)
        
        // Actually, sensitivity_utils provides find_root_local logic but it's internal.
        return 0.0;
    };
    
    // Hardcode a guess or simple scan?
    // Let's assume root is around 3.5
    double root_c = 3.5; 
    // Refine root logic? 
    // Disper26 doesn't expose a simple "solve" function yet (logic is in tests or utils).
    // I will call computeSensitivityKernels which re-finds the root anyway based on guess!
    // But I must provide a close enough root_c.
    
    // Let's implement a quick scanner
    auto finder = [&](double guess) {
         auto f = [&](double c) {
             ADS c_a = make(c);
             auto to_ads = [](const PhysicalParameters<double>& p) {
                PhysicalParameters<ADS> pa;
                pa.rho = ADS(p.rho); pa.rho.derivatives().setZero(1);
                pa.c11 = ADS(p.c11); pa.c11.derivatives().setZero(1);
                pa.c33 = ADS(p.c33); pa.c33.derivatives().setZero(1);
                pa.c44 = ADS(p.c44); pa.c44.derivatives().setZero(1);
                pa.c66 = ADS(p.c66); pa.c66.derivatives().setZero(1);
                pa.c13 = ADS(p.c13); pa.c13.derivatives().setZero(1);
                return pa;
             };
             std::vector<PhysicalParameters<ADS>> l_a;
             std::vector<ADS> t_a;
             for(auto& p: layers) l_a.push_back(to_ads(p));
             for(auto& t: thicknesses) { ADS ta; ta.value()=t; ta.derivatives().setZero(); t_a.push_back(ta); }
             PhysicalParameters<ADS> h_a = to_ads(halfSpace);
             
             return calculateDispersionFunction(l_a, t_a, h_a, omega, c_a, WaveType::Love)(I24).value();
         };
         
         // Scan
         std::cout << "Scanning 3.0 to 4.0:" << std::endl;
         for(double c=3.0; c<=4.0; c+=0.1) {
             std::cout << "C=" << c << " Val=" << f(c) << std::endl;
         }
         
         return brents_method(f, 3.1, 3.2);
    };
    
    root_c = finder(3.5);
    std::cout << "Root C: " << root_c << std::endl;
    
    // Compute Kernels (VTI Mode)
    auto kernels = computeSensitivityKernelsFD(layers, thicknesses, halfSpace, freq, root_c, WaveType::Love, SensitivityMode::VTI);
    
    std::cout << "--- Love Wave Kernels (VTI) ---" << std::endl;
    bool L_sensitive = false;
    bool N_sensitive = false;
    bool A_sensitive = false;
    
    for(const auto& k : kernels) {
        std::cout << "Layer " << k.layerIndex << " " << k.parameterName << ": dC/dp=" << k.d_PhaseVelocity << std::endl;
        
        if (k.layerIndex == 0) {
            if (k.parameterName == "L" && std::abs(k.d_PhaseVelocity) > 1e-4) L_sensitive = true;
            if (k.parameterName == "N" && std::abs(k.d_PhaseVelocity) > 1e-4) N_sensitive = true;
            if (k.parameterName == "A" && std::abs(k.d_PhaseVelocity) > 1e-6) A_sensitive = true; // Should be zero
        }
    }
    
    if (L_sensitive && N_sensitive && !A_sensitive) {
        std::cout << "SUCCESS: Detected L/N sensitivity and A independence." << std::endl;
    } else {
        std::cout << "FAILURE: Sensitivity mismatch. L=" << L_sensitive << " N=" << N_sensitive << " A=" << A_sensitive << std::endl;
    }
    
    return 0;
}
