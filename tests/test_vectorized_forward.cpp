/**
 * test_vectorized_forward.cpp
 * 
 * Vectorized Forward Mode AD for sensitivity kernel calculations.
 * Uses Eigen::AutoDiffScalar<VectorXd> to compute ALL parameter gradients
 * in a SINGLE forward pass.
 * 
 * This is the most efficient approach for our use case:
 * - O(1) forward passes regardless of parameter count
 * - Uses existing Eigen::AutoDiff infrastructure
 * - SIMD-friendly memory layout
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

// Include the engine
#include "../src/disper26.cpp"

// Vectorized AD type: VectorXd holds derivatives for ALL parameters
using ADVec = Eigen::AutoDiffScalar<Eigen::VectorXd>;

// Helper to create isotropic parameters
template <typename T>
PhysicalParameters<T> makeIsotropic(T vs, T vp, T rho) {
    return createIsotropicParameters(vs, vp, rho);
}

// ===========================================================================
// Vectorized Forward Mode Implementation
// ===========================================================================

struct SensitivityEntry {
    int layer_idx;
    int param_idx;
    double dc_dp;
    std::string param_name;
};

std::vector<SensitivityEntry> computeAllSensitivitiesVectorizedForward(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double root_c,
    WaveType type = WaveType::Rayleigh
) {
    int N = layers.size();
    int num_params = (N + 1) * 6;  // 6 VTI params per layer + halfspace
    
    std::vector<SensitivityEntry> results;
    std::string param_names[] = {"Rho", "A", "C", "F", "L", "N"};
    
    // Helper to create AD variable with specific parameter index seeded
    auto make_ad = [num_params](double val, int idx = -1) {
        ADVec ad;
        ad.value() = val;
        ad.derivatives() = Eigen::VectorXd::Zero(num_params + 1);  // +1 for c
        if (idx >= 0) {
            ad.derivatives()[idx] = 1.0;
        }
        return ad;
    };
    
    // Build model with vectorized AD type
    // Seed each parameter at its corresponding index
    std::vector<PhysicalParameters<ADVec>> ad_layers(N);
    int param_idx = 0;
    for (int i = 0; i < N; ++i) {
        ad_layers[i].rho = make_ad(layers[i].rho, param_idx++);    // rho
        ad_layers[i].c11 = make_ad(layers[i].c11, param_idx++);    // A (c11)
        ad_layers[i].c33 = make_ad(layers[i].c33, param_idx++);    // C (c33)
        ad_layers[i].c13 = make_ad(layers[i].c13, param_idx++);    // F (c13)
        ad_layers[i].c44 = make_ad(layers[i].c44, param_idx++);    // L (c44)
        ad_layers[i].c66 = make_ad(layers[i].c66, param_idx++);    // N (c66)
    }
    
    PhysicalParameters<ADVec> ad_hs;
    ad_hs.rho = make_ad(halfSpace.rho, param_idx++);
    ad_hs.c11 = make_ad(halfSpace.c11, param_idx++);
    ad_hs.c33 = make_ad(halfSpace.c33, param_idx++);
    ad_hs.c13 = make_ad(halfSpace.c13, param_idx++);
    ad_hs.c44 = make_ad(halfSpace.c44, param_idx++);
    ad_hs.c66 = make_ad(halfSpace.c66, param_idx++);
    
    std::vector<ADVec> ad_thick(N);
    for (int i = 0; i < N; ++i) {
        ad_thick[i] = make_ad(thicknesses[i], -1);  // Not a parameter we differentiate
    }
    
    ADVec c_ad = make_ad(root_c, num_params);  // c is at index num_params
    ADVec omega_ad = make_ad(omega, -1);
    
    // Single forward pass computes Delta AND all its derivatives
    auto Delta = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
    
    // Extract dDelta/dc and dDelta/dp_i
    double dDelta_dc = Delta.derivatives()[num_params];
    
    if (std::abs(dDelta_dc) < 1e-20) {
        return results;
    }
    
    // Compute dc/dp = -dDelta/dp / dDelta/dc for all parameters
    param_idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            double dDelta_dp = Delta.derivatives()[param_idx];
            double dc_dp = -dDelta_dp / dDelta_dc;
            results.push_back({i, j, dc_dp, param_names[j]});
            param_idx++;
        }
    }
    
    // Halfspace
    for (int j = 0; j < 6; ++j) {
        double dDelta_dp = Delta.derivatives()[param_idx];
        double dc_dp = -dDelta_dp / dDelta_dc;
        results.push_back({-1, j, dc_dp, param_names[j]});
        param_idx++;
    }
    
    return results;
}

// ===========================================================================
// Per-Parameter Forward Mode (for comparison)
// ===========================================================================

std::vector<SensitivityEntry> computeAllSensitivitiesPerParamForward(
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
    
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Helper
    auto make_ads = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    // For each parameter, compute separately
    for (int layer_idx = 0; layer_idx <= N; ++layer_idx) {
        for (int p_idx = 0; p_idx < 6; ++p_idx) {
            std::vector<PhysicalParameters<ADS>> ad_layers(N);
            for (int i = 0; i < N; ++i) {
                ad_layers[i].rho = make_ads(layers[i].rho);
                ad_layers[i].c11 = make_ads(layers[i].c11);
                ad_layers[i].c13 = make_ads(layers[i].c13);
                ad_layers[i].c33 = make_ads(layers[i].c33);
                ad_layers[i].c44 = make_ads(layers[i].c44);
                ad_layers[i].c66 = make_ads(layers[i].c66);
            }
            
            PhysicalParameters<ADS> ad_hs;
            ad_hs.rho = make_ads(halfSpace.rho);
            ad_hs.c11 = make_ads(halfSpace.c11);
            ad_hs.c13 = make_ads(halfSpace.c13);
            ad_hs.c33 = make_ads(halfSpace.c33);
            ad_hs.c44 = make_ads(halfSpace.c44);
            ad_hs.c66 = make_ads(halfSpace.c66);
            
            std::vector<ADS> ad_thick(N);
            for (int i = 0; i < N; ++i) ad_thick[i] = make_ads(thicknesses[i]);
            
            ADS c_ad = make_ads(root_c);
            ADS omega_ad = make_ads(omega);
            
            ADS* target = nullptr;
            int lidx = (layer_idx < N) ? layer_idx : -1;
            
            if (layer_idx < N) {
                ADS* params[] = {
                    &ad_layers[layer_idx].rho, &ad_layers[layer_idx].c11, 
                    &ad_layers[layer_idx].c33, &ad_layers[layer_idx].c13,
                    &ad_layers[layer_idx].c44, &ad_layers[layer_idx].c66
                };
                target = params[p_idx];
            } else {
                ADS* params[] = {
                    &ad_hs.rho, &ad_hs.c11, &ad_hs.c33, 
                    &ad_hs.c13, &ad_hs.c44, &ad_hs.c66
                };
                target = params[p_idx];
            }
            
            // Compute dDelta/dc
            c_ad.derivatives()[0] = 1.0;
            auto Delta_c = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
            double dDelta_dc = Delta_c.derivatives()[0];
            c_ad.derivatives()[0] = 0.0;
            
            // Compute dDelta/dp
            c_ad = make_ads(root_c);
            target->derivatives()[0] = 1.0;
            auto Delta_p = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ad, c_ad, type)(I24);
            double dDelta_dp = Delta_p.derivatives()[0];
            
            double dc_dp = (std::abs(dDelta_dc) > 1e-20) ? -dDelta_dp / dDelta_dc : 0.0;
            results.push_back({lidx, p_idx, dc_dp, param_names[p_idx]});
        }
    }
    
    return results;
}

// ===========================================================================
// Main Test
// ===========================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Vectorized Forward Mode AD Test           " << std::endl;
    std::cout << "============================================" << std::endl;
    
    // 2-layer model
    std::vector<PhysicalParameters<double>> layers = {
        makeIsotropic(1.5, 2.6, 2.0),
        makeIsotropic(2.0, 3.5, 2.3)
    };
    std::vector<double> thicknesses = {3.0, 5.0};
    PhysicalParameters<double> halfSpace = makeIsotropic(3.0, 5.2, 2.7);
    
    double freq = 0.3;
    double omega = 2.0 * M_PI * freq;
    
    // Find root
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
    
    int num_params = (layers.size() + 1) * 6;
    std::cout << "\nModel: " << layers.size() << "-layer + halfspace (" << num_params << " parameters)" << std::endl;
    std::cout << "Frequency: " << freq << " Hz" << std::endl;
    std::cout << "Phase velocity c = " << root_c << " km/s" << std::endl;
    
    // Benchmark
    std::cout << "\n=== Comparing Per-Parameter vs Vectorized Forward Mode ===" << std::endl;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto per_param = computeAllSensitivitiesPerParamForward(layers, thicknesses, halfSpace, omega, root_c, WaveType::Rayleigh);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto per_param_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    
    t1 = std::chrono::high_resolution_clock::now();
    auto vectorized = computeAllSensitivitiesVectorizedForward(layers, thicknesses, halfSpace, omega, root_c, WaveType::Rayleigh);
    t2 = std::chrono::high_resolution_clock::now();
    auto vec_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Per-param forward:   " << per_param_time << " μs (" << num_params * 2 << " forward passes)" << std::endl;
    std::cout << "  Vectorized forward:  " << vec_time << " μs (1 forward pass)" << std::endl;
    std::cout << "  Speedup:             " << std::fixed << std::setprecision(2) << (double)per_param_time / vec_time << "x" << std::endl;
    
    // Compare
    std::cout << "\nResults comparison:" << std::endl;
    std::cout << std::setw(8) << "Layer" << std::setw(8) << "Param"
              << std::setw(14) << "Per-Param" << std::setw(14) << "Vectorized"
              << std::setw(12) << "Rel.Err" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    int num_pass = 0, num_fail = 0;
    for (size_t i = 0; i < per_param.size() && i < vectorized.size(); ++i) {
        auto& p = per_param[i];
        auto& v = vectorized[i];
        
        double rel_err = 0.0;
        if (std::abs(p.dc_dp) > 1e-15) {
            rel_err = std::abs(p.dc_dp - v.dc_dp) / std::abs(p.dc_dp);
        }
        
        std::string layer_str = (p.layer_idx == -1) ? "HS" : std::to_string(p.layer_idx);
        bool pass = rel_err < 0.0001 || (std::abs(p.dc_dp) < 1e-12 && std::abs(v.dc_dp) < 1e-12);
        
        std::cout << std::setw(8) << layer_str << std::setw(8) << p.param_name
                  << std::setw(14) << std::scientific << std::setprecision(4) << p.dc_dp
                  << std::setw(14) << v.dc_dp
                  << std::setw(12) << std::fixed << std::setprecision(6) << rel_err
                  << (pass ? " [PASS]" : " [FAIL]") << std::endl;
        
        if (pass) num_pass++; else num_fail++;
    }
    
    std::cout << "\nSummary: " << num_pass << " PASS, " << num_fail << " FAIL" << std::endl;
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "           Test completed                   " << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
