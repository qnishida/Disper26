/**
 * test_backward_mode_sensitivity.cpp
 * 
 * Test backward mode AD for sensitivity kernel calculations.
 * Backward mode (reverse AD) is efficient when #parameters >> #outputs.
 * 
 * This test verifies that the safe_complex_sqrt wrapper enables
 * backward mode to work with the dispersion calculations.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

// Include the engine
#include "../src/disper26.cpp"

// autodiff namespace
namespace rev = autodiff::reverse::detail;

using avar = autodiff::var;

// Helper to create isotropic parameters
template <typename T>
PhysicalParameters<T> makeIsotropic(T vs, T vp, T rho) {
    return createIsotropicParameters(vs, vp, rho);
}

// ===========================================================================
// Backward Mode Implementation
// ===========================================================================

struct SensitivityEntry {
    int layer_idx;
    int param_idx;
    double dc_dp;
    std::string param_name;
};

std::vector<SensitivityEntry> computeAllSensitivitiesBackwardMode(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double root_c,
    WaveType type = WaveType::Rayleigh
) {
    int N = layers.size();
    std::vector<SensitivityEntry> results;
    std::string param_names[] = {"Rho", "A", "C", "F", "L", "N"};
    
    // Build model with backward mode (var) types
    std::vector<PhysicalParameters<avar>> var_layers(N);
    for (int i = 0; i < N; ++i) {
        var_layers[i].rho = layers[i].rho;
        var_layers[i].c11 = layers[i].c11;
        var_layers[i].c13 = layers[i].c13;
        var_layers[i].c33 = layers[i].c33;
        var_layers[i].c44 = layers[i].c44;
        var_layers[i].c66 = layers[i].c66;
    }
    
    PhysicalParameters<avar> var_hs;
    var_hs.rho = halfSpace.rho;
    var_hs.c11 = halfSpace.c11;
    var_hs.c13 = halfSpace.c13;
    var_hs.c33 = halfSpace.c33;
    var_hs.c44 = halfSpace.c44;
    var_hs.c66 = halfSpace.c66;
    
    std::vector<avar> var_thick(N);
    for (int i = 0; i < N; ++i) {
        var_thick[i] = thicknesses[i];
    }
    
    avar c_var = root_c;
    avar omega_var = omega;
    
    // Compute Delta (dispersion function) - ONE forward pass
    auto res = calculateDispersionFunction(var_layers, var_thick, var_hs, omega_var, c_var, type);
    avar Delta = res(I24);
    
    // Get dDelta/dc first using a separate call
    auto [dDelta_dc] = rev::derivatives(Delta, rev::wrt(c_var));
    
    if (std::abs(dDelta_dc) < 1e-20) {
        return results;
    }
    
    // Now compute dc/dp for each parameter individually
    // Note: Each call to derivatives() is a separate backward pass, but for N+1 layers
    // with 6 params each, this is still more efficient than forward mode for large N
    for (int i = 0; i < N; ++i) {
        // Compute all 6 gradients for this layer in one backward pass
        auto [g0, g1, g2, g3, g4, g5] = rev::derivatives(Delta, rev::wrt(
            var_layers[i].rho, var_layers[i].c11, var_layers[i].c33,
            var_layers[i].c13, var_layers[i].c44, var_layers[i].c66
        ));
        
        results.push_back({i, 0, -g0 / dDelta_dc, param_names[0]});
        results.push_back({i, 1, -g1 / dDelta_dc, param_names[1]});
        results.push_back({i, 2, -g2 / dDelta_dc, param_names[2]});
        results.push_back({i, 3, -g3 / dDelta_dc, param_names[3]});
        results.push_back({i, 4, -g4 / dDelta_dc, param_names[4]});
        results.push_back({i, 5, -g5 / dDelta_dc, param_names[5]});
    }
    
    // Halfspace: all 6 params in one backward pass
    auto [hs0, hs1, hs2, hs3, hs4, hs5] = rev::derivatives(Delta, rev::wrt(
        var_hs.rho, var_hs.c11, var_hs.c33,
        var_hs.c13, var_hs.c44, var_hs.c66
    ));
    
    results.push_back({-1, 0, -hs0 / dDelta_dc, param_names[0]});
    results.push_back({-1, 1, -hs1 / dDelta_dc, param_names[1]});
    results.push_back({-1, 2, -hs2 / dDelta_dc, param_names[2]});
    results.push_back({-1, 3, -hs3 / dDelta_dc, param_names[3]});
    results.push_back({-1, 4, -hs4 / dDelta_dc, param_names[4]});
    results.push_back({-1, 5, -hs5 / dDelta_dc, param_names[5]});
    
    return results;
}

// ===========================================================================
// Forward Mode For Comparison
// ===========================================================================

std::vector<SensitivityEntry> computeAllSensitivitiesForwardMode(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double root_c,
    WaveType type = WaveType::Rayleigh
) {
    int N = layers.size();
    std::vector<SensitivityEntry> results;
    std::string param_names[] = {"Rho", "A", "C", "F", "L", "N"};
    
    using adual = autodiff::dual;
    
    for (int layer_idx = 0; layer_idx <= N; ++layer_idx) {
        for (int param_idx = 0; param_idx < 6; ++param_idx) {
            std::vector<PhysicalParameters<adual>> ad_layers(N);
            for (int i = 0; i < N; ++i) {
                ad_layers[i].rho = layers[i].rho;
                ad_layers[i].c11 = layers[i].c11;
                ad_layers[i].c13 = layers[i].c13;
                ad_layers[i].c33 = layers[i].c33;
                ad_layers[i].c44 = layers[i].c44;
                ad_layers[i].c66 = layers[i].c66;
            }
            
            PhysicalParameters<adual> ad_hs;
            ad_hs.rho = halfSpace.rho;
            ad_hs.c11 = halfSpace.c11;
            ad_hs.c13 = halfSpace.c13;
            ad_hs.c33 = halfSpace.c33;
            ad_hs.c44 = halfSpace.c44;
            ad_hs.c66 = halfSpace.c66;
            
            std::vector<adual> ad_thick(N);
            for (int i = 0; i < N; ++i) ad_thick[i] = thicknesses[i];
            
            adual c_ad = root_c;
            adual omega_ad = omega;
            
            adual* target = nullptr;
            int lidx = (layer_idx < N) ? layer_idx : -1;
            
            if (layer_idx < N) {
                adual* params[] = {
                    &ad_layers[layer_idx].rho, &ad_layers[layer_idx].c11, 
                    &ad_layers[layer_idx].c33, &ad_layers[layer_idx].c13,
                    &ad_layers[layer_idx].c44, &ad_layers[layer_idx].c66
                };
                target = params[param_idx];
            } else {
                adual* params[] = {
                    &ad_hs.rho, &ad_hs.c11, &ad_hs.c33, 
                    &ad_hs.c13, &ad_hs.c44, &ad_hs.c66
                };
                target = params[param_idx];
            }
            
            autodiff::seed(c_ad);
            auto Delta_c = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
            double dDelta_dc = autodiff::grad(Delta_c);
            autodiff::unseed(c_ad);
            
            c_ad = root_c;
            autodiff::seed(*target);
            auto Delta_p = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
            double dDelta_dp = autodiff::grad(Delta_p);
            autodiff::unseed(*target);
            
            double dc_dp = (std::abs(dDelta_dc) > 1e-20) ? -dDelta_dp / dDelta_dc : 0.0;
            results.push_back({lidx, param_idx, dc_dp, param_names[param_idx]});
        }
    }
    
    return results;
}

// ===========================================================================
// Main Test
// ===========================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Backward Mode AD Sensitivity Test         " << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Simple 2-layer model
    std::vector<PhysicalParameters<double>> layers = {
        makeIsotropic(1.5, 2.6, 2.0),
        makeIsotropic(2.0, 3.5, 2.3)
    };
    std::vector<double> thicknesses = {3.0, 5.0};
    PhysicalParameters<double> halfSpace = makeIsotropic(3.0, 5.2, 2.7);
    
    double freq = 0.3;
    double omega = 2.0 * M_PI * freq;
    
    // Find root using standard approach
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto make_ads = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    std::vector<PhysicalParameters<ADS>> ads_layers;
    for (auto& l : layers) {
        PhysicalParameters<ADS> pl;
        pl.rho = make_ads(l.rho); pl.c11 = make_ads(l.c11); pl.c13 = make_ads(l.c13);
        pl.c33 = make_ads(l.c33); pl.c44 = make_ads(l.c44); pl.c66 = make_ads(l.c66);
        ads_layers.push_back(pl);
    }
    std::vector<ADS> ads_thick;
    for (auto& t : thicknesses) ads_thick.push_back(make_ads(t));
    
    PhysicalParameters<ADS> ads_hs;
    ads_hs.rho = make_ads(halfSpace.rho); ads_hs.c11 = make_ads(halfSpace.c11);
    ads_hs.c13 = make_ads(halfSpace.c13); ads_hs.c33 = make_ads(halfSpace.c33);
    ads_hs.c44 = make_ads(halfSpace.c44); ads_hs.c66 = make_ads(halfSpace.c66);
    
    ADS omega_ads = make_ads(omega);
    
    double root_c = 0.0;
    auto eval_F = [&](double c_val) {
        ADS c_ads = make_ads(c_val);
        return calculateDispersionFunction(ads_layers, ads_thick, ads_hs, omega_ads, c_ads, WaveType::Rayleigh)(I24).value();
    };
    
    double c_min = 1.0, c_max = 3.0;
    double f_prev = eval_F(c_min);
    for (double c = c_min + 0.01; c <= c_max; c += 0.01) {
        double f_curr = eval_F(c);
        if (f_prev * f_curr <= 0) {
            double a = c - 0.01, b = c;
            for (int i = 0; i < 50; ++i) {
                double mid = (a + b) / 2.0;
                double fm = eval_F(mid);
                if (f_prev * fm <= 0) b = mid;
                else { a = mid; f_prev = fm; }
            }
            root_c = (a + b) / 2.0;
            break;
        }
        f_prev = f_curr;
    }
    
    std::cout << "\nModel: 2-layer + halfspace (" << (layers.size() + 1) * 6 << " parameters)" << std::endl;
    std::cout << "Frequency: " << freq << " Hz" << std::endl;
    std::cout << "Phase velocity c = " << root_c << " km/s" << std::endl;
    
    // Benchmark
    std::cout << "\n=== Comparing Forward vs Backward Mode ===" << std::endl;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto fwd_results = computeAllSensitivitiesForwardMode(layers, thicknesses, halfSpace, omega, root_c, WaveType::Rayleigh);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto fwd_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    
    t1 = std::chrono::high_resolution_clock::now();
    auto bwd_results = computeAllSensitivitiesBackwardMode(layers, thicknesses, halfSpace, omega, root_c, WaveType::Rayleigh);
    t2 = std::chrono::high_resolution_clock::now();
    auto bwd_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Forward mode:  " << fwd_time << " μs" << std::endl;
    std::cout << "  Backward mode: " << bwd_time << " μs" << std::endl;
    
    // Compare
    std::cout << "\nResults comparison:" << std::endl;
    std::cout << std::setw(8) << "Layer" << std::setw(8) << "Param"
              << std::setw(14) << "Forward" << std::setw(14) << "Backward"
              << std::setw(12) << "Rel.Err" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    int num_pass = 0, num_fail = 0;
    for (size_t i = 0; i < fwd_results.size() && i < bwd_results.size(); ++i) {
        auto& f = fwd_results[i];
        auto& b = bwd_results[i];
        
        double rel_err = 0.0;
        if (std::abs(f.dc_dp) > 1e-15) {
            rel_err = std::abs(f.dc_dp - b.dc_dp) / std::abs(f.dc_dp);
        }
        
        std::string layer_str = (f.layer_idx == -1) ? "HS" : std::to_string(f.layer_idx);
        bool pass = rel_err < 0.01 || (std::abs(f.dc_dp) < 1e-12 && std::abs(b.dc_dp) < 1e-12);
        
        std::cout << std::setw(8) << layer_str << std::setw(8) << f.param_name
                  << std::setw(14) << std::scientific << std::setprecision(4) << f.dc_dp
                  << std::setw(14) << b.dc_dp
                  << std::setw(12) << std::fixed << std::setprecision(4) << rel_err
                  << (pass ? " [PASS]" : " [FAIL]") << std::endl;
        
        if (pass) num_pass++; else num_fail++;
    }
    
    std::cout << "\nSummary: " << num_pass << " PASS, " << num_fail << " FAIL" << std::endl;
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "           Test completed                   " << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
