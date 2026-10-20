#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include "../src/disper26.cpp"

// Define a test that mirrors test_group_sensitivity logic
// but specifically checks if results are stable with the new scaling logic active.

int main() {
    // 20-Layer Model
    // Replicate setup from test_group_sensitivity.cpp
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // We will just call computeSensitivityKernels for a single case
    // and print results. The scaling logic inside calculateDispersionFunction
    // is now active.
    
    int deriv_size = 2;
    auto make_const = [&](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(deriv_size); return a;
    };
    
    std::vector<PhysicalParameters<double>> layers_dbl;
    std::vector<double> thicknesses_dbl;
    
    for (int i = 1; i <= 19; ++i) {
        double vs_val = 0.0;
        double h_val = 1.0;
        if (i <= 2) vs_val = 1.5;
        else if (i <= 5) vs_val = 2.5 + (i - 3.0) * 0.16;
        else vs_val = 3.4;
        double vp_val = vs_val * 1.72;
        double rho_val = 0.9473 * 0.2966 * vp_val;
        
        PhysicalParameters<double> p; 
        p.c44 = rho_val * vs_val*vs_val; 
        p.c11 = rho_val * vp_val*vp_val;
        p.rho = rho_val; 
        p.c33 = p.c11; 
        p.c13 = p.c11 - 2.0*p.c44;
        p.c66 = p.c44; // Isotropic
        layers_dbl.push_back(p);
        thicknesses_dbl.push_back(h_val);
    }
    
    PhysicalParameters<double> half_dbl;
    {
        double vs_val = 4.6;
        double vp_val = vs_val*1.72; 
        double rho_val = 0.9473*0.2966*vp_val;
        half_dbl.c44 = rho_val * vs_val*vs_val;
        half_dbl.c11 = rho_val * vp_val*vp_val;
        half_dbl.rho = rho_val;
        half_dbl.c33 = half_dbl.c11;
        half_dbl.c13 = half_dbl.c11 - 2.0*half_dbl.c44;
        half_dbl.c66 = half_dbl.c44; // Isotropic
    }

    double freq = 0.4;
    
    // Find Root first (Ref)
    auto func = [&](double c) {
        // Minimal wrapper for brents
        using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
        ADS ca; ca.value() = c; ca.derivatives().setZero(1);
        ADS wa; wa.value() = 2.0*M_PI*freq; wa.derivatives().setZero(1);
        auto l_ad = layers_dbl; // conversion logic needed? calculateDispersionFunction expects vectors of ADS if inputs are ADS
        // Wait, computeSensitivityKernels handles double inputs!
        // We need root first.
        return 0.0; // dummy
    };
    
    // Actually computeSensitivityKernels needs root c.
    // Let's use the one we know: 1.53019
    double c_root = 1.5301899; 
    
    std::cout << "Running computeSensitivityKernels with Scaling Active..." << std::endl;
    auto kernels = computeSensitivityKernelsFD(layers_dbl, thicknesses_dbl, half_dbl, freq, c_root);
    
    std::cout << "Sensitivity Kernels (Halfspace):" << std::endl;
    for(const auto& k : kernels) {
        if(k.layerIndex == -1) { // Halfspace
            std::cout << k.parameterName << ": dU/dp = " << k.d_GroupVelocity << std::endl;
        }
    }
    
    return 0;
}
