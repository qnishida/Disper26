
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include <algorithm> // for find_if

// Include source directly to access internal helpers if needed, or just headers if properly separated.
// Since disper26 is header-only style:
#include "../src/disper26.cpp"
#include "../src/sensitivity_utils.hpp"

using namespace std;

// --- Helper to find kernel ---
double get_kernel_val(const vector<SensitivityKernel>& kernels, string param, int layer_idx, string type="c") {
    auto it = find_if(kernels.begin(), kernels.end(), [&](const SensitivityKernel& k){
        return k.parameterName == param && k.layerIndex == layer_idx;
    });
    if (it != kernels.end()) {
        if (type == "c") return it->d_PhaseVelocity;
        if (type == "U") return it->d_GroupVelocity;
        if (type == "HV") return it->d_HVRatio;
    }
    return 0.0; // Not found
}

bool has_kernel(const vector<SensitivityKernel>& kernels, string param) {
    auto it = find_if(kernels.begin(), kernels.end(), [&](const SensitivityKernel& k){
        return k.parameterName == param;
    });
    return (it != kernels.end());
}

// --- Main Test ---
int main() {
    cout << "=== Test: Optional Thickness Sensitivity Kernels ===" << endl;
    
    // Model Setup (Isotropic 1 Layer + HS)
    PhysicalParameters<double> L1, HS;
    L1 = createIsotropicParameters(1.5, 3.0, 2.0); // Vs, Vp, Rho
    HS = createIsotropicParameters(2.5, 4.5, 2.5);

    double H0 = 10.0;
    
    vector<PhysicalParameters<double>> layers = {L1};
    vector<double> thicknesses = {H0};
    
    double freq = 0.3;
    
    // Find Root
    auto func = [&](double c) {
        double w = 2.0*M_PI*freq;
        return calculateDispersionFunction(layers, thicknesses, HS, w, c, WaveType::Rayleigh)(4);
    };
    vector<double> roots = searchRoots(func, 1.5, 2.5, 0.01);
    if (roots.empty()) {
        cerr << "Error: No roots found." << endl;
        return 1;
    }
    double root_c = roots[0];
    cout << "Root c: " << root_c << endl;
    
    // 1. Test Default Behavior (No Thickness)
    {
        cout << "\n--- Test 1: Default Behavior (computeThickness=false) ---" << endl;
        auto kernels = computeSensitivityKernelsFullAD(layers, thicknesses, HS, freq, root_c, WaveType::Rayleigh, SensitivityMode::Isotropic, false);
        
        bool found = has_kernel(kernels, "Thickness");
        cout << "Has 'Thickness' kernel? " << (found ? "YES" : "NO") << endl;
        
        if (found) {
            cerr << "FAILURE: Thickness kernel present when disabled." << endl;
            return 1;
        } else {
            cout << "SUCCESS: Thickness kernel absent as expected." << endl;
        }
    }
    
    // 2. Test Enabled Behavior (With Thickness)
    {
        cout << "\n--- Test 2: Enabled Behavior (computeThickness=true) ---" << endl;
        auto kernels = computeSensitivityKernelsFullAD(layers, thicknesses, HS, freq, root_c, WaveType::Rayleigh, SensitivityMode::Isotropic, true);
        
        bool found = has_kernel(kernels, "Thickness");
        cout << "Has 'Thickness' kernel? " << (found ? "YES" : "NO") << endl;
        
        if (!found) {
             cerr << "FAILURE: Thickness kernel missing when enabled." << endl;
             return 1;
        }
        
        double dc_dH = get_kernel_val(kernels, "Thickness", 0, "c");
        double dU_dH = get_kernel_val(kernels, "Thickness", 0, "U");
        double dHV_dH = get_kernel_val(kernels, "Thickness", 0, "HV");
        
        cout << "dc/dH: " << dc_dH << endl;
        cout << "dU/dH: " << dU_dH << endl;
        cout << "dHV/dH: " << dHV_dH << endl;
        
        // Validate against known reliable FD value from previous experiments
        // Or run a quick self-contained FD checking (sanity check)
        
        cout << "Running quick FD sanity check..." << endl;
        double dh = 1e-4;
        auto f_c = [&](double h, double c_guess) {
             // quick search
             auto func_h = [&](double c) {
                 double w = 2.0*M_PI*freq;
                 return calculateRayleighDispersion(layers, {h}, HS, w, c, NormalizationMode::VectorNorm)(4);
             };
             // Scan small range
             double c_l = c_guess - 0.05, c_r = c_guess + 0.05;
             double f_l = func_h(c_l);
             // Bisection
             for(int k=0; k<50; ++k) {
                 double mid = 0.5*(c_l+c_r);
                 double f_m = func_h(mid);
                 if(f_l*f_m <= 0) c_r=mid; else { c_l=mid; f_l=f_m; }
             }
             return 0.5*(c_l+c_r);
        };
        
        double c_p = f_c(H0+dh, root_c);
        double c_m = f_c(H0-dh, root_c);
        double dc_dH_FD = (c_p - c_m)/(2.0*dh);
        
        cout << "FD dc/dH: " << dc_dH_FD << endl;
        if (abs(dc_dH - dc_dH_FD) > 1e-4) {
             cerr << "FAILURE: AD dc/dH mismatch." << endl;
             return 1;
        } else {
             cout << "SUCCESS: AD dc/dH matches FD." << endl;
        }
    }

    return 0;
}
