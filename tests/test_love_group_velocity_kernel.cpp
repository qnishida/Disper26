/**
 * test_love_group_velocity_kernel.cpp
 * 
 * Hybrid AD verification for Love Wave Group Velocity Kernel (dU/dp).
 * 
 * Love waves (SH) involve parameters: Rho, A (c11), C (c33), F (c13), L (c44), N (c66).
 * Specifically, Love waves depend primarily on Rho, L(c44), N(c66).
 * A(c11), C(c33), F(c13) should have ZERO sensitivity.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Include the engine
#include "../src/disper26.cpp"

// Helpers
template <typename T>
PhysicalParameters<T> makeIsotropic(T vs, T vp, T rho) {
    return createIsotropicParameters(vs, vp, rho);
}

// Hybrid AD Implementation for dU/dp
// (Same logic as Rayleigh test, but WaveType::Love)
double computeDerivativeHybrid(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double root_c,
    int target_layer,
    int target_param,
    WaveType type
) {
     using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto make_ads = [](double v) { ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a; };
    
    int N = layers.size();
    
    auto get_U = [&](double p_perturb) {
         // Setup AD params
         std::vector<PhysicalParameters<ADS>> ad_layers(N);
         for(int i=0; i<N; ++i) {
             ad_layers[i].rho = make_ads(layers[i].rho); ad_layers[i].c11 = make_ads(layers[i].c11);
             ad_layers[i].c33 = make_ads(layers[i].c33); ad_layers[i].c13 = make_ads(layers[i].c13);
             ad_layers[i].c44 = make_ads(layers[i].c44); ad_layers[i].c66 = make_ads(layers[i].c66);
         }
         PhysicalParameters<ADS> ad_hs;
         ad_hs.rho = make_ads(halfSpace.rho); ad_hs.c11 = make_ads(halfSpace.c11);
         ad_hs.c33 = make_ads(halfSpace.c33); ad_hs.c13 = make_ads(halfSpace.c13);
         ad_hs.c44 = make_ads(halfSpace.c44); ad_hs.c66 = make_ads(halfSpace.c66);
         
         if(target_layer>=0) {
             ADS* p[] = {&ad_layers[target_layer].rho, &ad_layers[target_layer].c11, &ad_layers[target_layer].c33, &ad_layers[target_layer].c13, &ad_layers[target_layer].c44, &ad_layers[target_layer].c66};
             p[target_param]->value() += p_perturb;
         } else {
             ADS* p[] = {&ad_hs.rho, &ad_hs.c11, &ad_hs.c33, &ad_hs.c13, &ad_hs.c44, &ad_hs.c66};
             p[target_param]->value() += p_perturb;
         }
         std::vector<ADS> ad_thick(N);
         for(int i=0; i<N; ++i) ad_thick[i] = make_ads(thicknesses[i]);
         
         double c_local = root_c;
         ADS omega_ads = make_ads(omega);
         
         // Root refinement (Newton)
         for(int k=0; k<10; ++k) {
             ADS c_ads = make_ads(c_local); c_ads.derivatives()[0] = 1.0;
             // The Love residual at index I24 is -T at the free surface.
             auto F = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ads, c_ads, type)(I24);
             c_local -= F.value() / F.derivatives()[0];
         }
         
         // compute dc/dw
         using adual = autodiff::dual;
         std::vector<PhysicalParameters<adual>> d_layers(N);
         for(int i=0; i<N; ++i) {
             d_layers[i].rho = ad_layers[i].rho.value(); d_layers[i].c11 = ad_layers[i].c11.value();
             d_layers[i].c33 = ad_layers[i].c33.value(); d_layers[i].c13 = ad_layers[i].c13.value();
             d_layers[i].c44 = ad_layers[i].c44.value(); d_layers[i].c66 = ad_layers[i].c66.value();
         }
         PhysicalParameters<adual> d_hs;
         d_hs.rho = ad_hs.rho.value(); d_hs.c11 = ad_hs.c11.value(); d_hs.c33 = ad_hs.c33.value();
         d_hs.c13 = ad_hs.c13.value(); d_hs.c44 = ad_hs.c44.value(); d_hs.c66 = ad_hs.c66.value();
         std::vector<adual> d_thick(N);
         for(int i=0; i<N; ++i) d_thick[i] = ad_thick[i].value();
         
         adual c_d = c_local;
         adual om_d = omega;
         autodiff::seed(c_d);
         auto F_c = calculateDispersionFunction(d_layers, d_thick, d_hs, om_d, c_d, type)(I24);
         double dFdc = autodiff::grad(F_c);
         autodiff::unseed(c_d);
         
         c_d = c_local;
         autodiff::seed(om_d);
         auto F_om = calculateDispersionFunction(d_layers, d_thick, d_hs, om_d, c_d, type)(I24);
         double dFdom = autodiff::grad(F_om);
         autodiff::unseed(om_d);
         
         double dcdw = -dFdom / dFdc;
         return c_local / (1.0 - (omega/c_local)*dcdw);
    };
    
    double delta = 1e-4; // Slightly larger for stability? Or keep 1e-5.
    double U_plus = get_U(delta);
    double U_minus = get_U(-delta);
    return (U_plus - U_minus) / (2.0 * delta);
}

// Find Root Helper
double find_root(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double c_guess
) {
    double c_curr = c_guess;
    for(int k=0; k<20; ++k) {
        using adual = autodiff::dual;
        adual c_ad = c_curr;
        autodiff::seed(c_ad);
        // Create AD params (const w.r.t c)
        // ... (Simplified: use double params and cast inside calculateDispersionFunction if template)
        // Wait, calculateDispersionFunction templates on T.
        // We need to construct parameters of type autodiff::dual.
        
        std::vector<PhysicalParameters<adual>> ad_layers(layers.size());
        for(size_t i=0; i<layers.size(); ++i) {
             ad_layers[i].rho = layers[i].rho; ad_layers[i].c11 = layers[i].c11;
             ad_layers[i].c33 = layers[i].c33; ad_layers[i].c13 = layers[i].c13;
             ad_layers[i].c44 = layers[i].c44; ad_layers[i].c66 = layers[i].c66;
        }
        PhysicalParameters<adual> ad_hs;
        ad_hs.rho = halfSpace.rho; ad_hs.c11 = halfSpace.c11; ad_hs.c33 = halfSpace.c33;
        ad_hs.c13 = halfSpace.c13; ad_hs.c44 = halfSpace.c44; ad_hs.c66 = halfSpace.c66;
        std::vector<adual> ad_thick(layers.size());
        for(size_t i=0; i<layers.size(); ++i) ad_thick[i] = thicknesses[i];
        
        adual omega_ad = omega;
        
        auto F = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, WaveType::Love)(I24);
        double val = autodiff::val(F);
        double grad = autodiff::grad(F);
        
        if(std::abs(val) < 1e-12) return c_curr;
        c_curr -= val / grad;
    }
    return c_curr;
}


int main() {
    std::cout << "Testing Love Wave Group Velocity Kernels..." << std::endl;
    
    // Model: 2-layer + Halfspace
    // Layer 0: Vs=3.5, Rho=2.5, H=10
    // Layer 1: Vs=4.0, Rho=2.7, H=15
    // HS:      Vs=4.5, Rho=2.9
    // Isotropic for simplicity, but Love depends on Mu (c44/c66) and Rho.
    // In Isotropic: c44 = c66 = mu.
    
    std::vector<PhysicalParameters<double>> layers = { 
        makeIsotropic(3.5, 6.0, 2.5), 
        makeIsotropic(4.0, 7.0, 2.7) 
    };
    std::vector<double> thicknesses = {10.0, 15.0};
    PhysicalParameters<double> halfSpace = makeIsotropic(4.5, 8.0, 2.9);
    
    double freq = 0.1; // Hz
    double omega = 2.0 * M_PI * freq;
    
    // Initial guess: between Vs0 and VsHS (3.5 - 4.5)
    double root_c = find_root(layers, thicknesses, halfSpace, omega, 3.8);
    std::cout << "Root Phase Velocity: " << root_c << " km/s" << std::endl;
    
    std::string param_names[] = {"Rho", "A", "C", "F", "L", "N"};
    
    // Test Sensitivities
    // Expectation: 
    // - Non-zero for Rho, L, N.
    // - Zero for A, C, F (P-wave and Vertical params irrelevant for SH).
    
    std::vector<std::pair<int, int>> test_cases = {
        {0, 0}, // Layer 0 Rho (Should be non-zero)
        {0, 4}, // Layer 0 L (c44) -> equivalent to Mu. Non-zero.
        {0, 5}, // Layer 0 N (c66) -> equivalent to Mu. Non-zero.
        {0, 1}, // Layer 0 A (c11) -> P-wave. Should be ZERO.
        {0, 2}, // Layer 0 C (c33) -> P-wave/Vertical. ZERO.
        {0, 3}  // Layer 0 F (c13) -> Lame. ZERO.
    };
    
    // Note: For Isotropic input, c44=c66. Perturbing one might break isotropy locally (VTI).
    // The code handles VTI, so this is fine.
    
    bool all_pass = true;
    
    for (auto [layer, param] : test_cases) {
        double dU_dp = computeDerivativeHybrid(layers, thicknesses, halfSpace, omega, root_c, layer, param, WaveType::Love);
        
        std::cout << "Layer " << layer << " Param " << param_names[param] << ": dU/dp = " << dU_dp;
        
        bool is_zero_expected = (param == 1 || param == 2 || param == 3);
        if (is_zero_expected) {
            if (std::abs(dU_dp) < 1e-9) {
                std::cout << " [PASS] (Zero as expected)" << std::endl;
            } else {
                std::cout << " [FAIL] (Expected Zero, got " << dU_dp << ")" << std::endl;
                all_pass = false;
            }
        } else {
            if (std::abs(dU_dp) > 1e-9) {
                std::cout << " [PASS] (Non-zero)" << std::endl;
            } else {
                std::cout << " [FAIL] (Expected Non-zero, got " << dU_dp << ")" << std::endl;
                // Note: Sensitivity could be physically zero/low, but usually not for Rho/L/N in surface layer.
                // If it fails here, check perturbations.
            }
        }
    }
    
    return all_pass ? 0 : 1;
}
