/**
 * test_romberg_stability.cpp
 * 
 * Verifies the numerical stability of Hybrid AD Sensitivity Kernels for Romberg Integration.
 * 
 * Romberg Integration requires the integrand to be smooth (C^2k for order k).
 * Hybrid AD uses finite differences, which introduces high-frequency noise O(delta^2/h^2).
 * We need to confirm that this noise allows for higher-order convergence (at least Simpson/Order 4).
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <functional>

// Include engine
#include "../src/disper26.cpp"

// Helpers
template <typename T>
PhysicalParameters<T> makeIsotropic(T vs, T vp, T rho) {
    return createIsotropicParameters(vs, vp, rho);
}

// Convert PhysicalParameters<double> to PhysicalParameters<adual>
template <typename T>
PhysicalParameters<autodiff::dual> toAutodiffForward(const PhysicalParameters<T>& p) {
    PhysicalParameters<autodiff::dual> pa;
    pa.rho = static_cast<double>(p.rho);
    pa.c11 = static_cast<double>(p.c11);
    pa.c13 = static_cast<double>(p.c13);
    pa.c33 = static_cast<double>(p.c33);
    pa.c44 = static_cast<double>(p.c44);
    pa.c66 = static_cast<double>(p.c66);
    return pa;
}

// --- Hybrid AD Implementation (Local Copy for Testing) ---
double computeDerivativeHybrid(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double omega,
    double root_c,
    int target_layer,
    int target_param,
    WaveType type = WaveType::Rayleigh
) {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    auto make_ads = [](double v) { ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a; };
    
    int N = layers.size();
    
    // Function to compute U at perturbed physical parameter p
    auto get_U = [&](double p_perturb) -> double {
        // 1. Setup AD Parameters
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
        
        // Apply perturbation
        if(target_layer >= 0) {
            ADS* p[] = {&ad_layers[target_layer].rho, &ad_layers[target_layer].c11, &ad_layers[target_layer].c33, &ad_layers[target_layer].c13, &ad_layers[target_layer].c44, &ad_layers[target_layer].c66};
            p[target_param]->value() += p_perturb;
        } else {
            ADS* p[] = {&ad_hs.rho, &ad_hs.c11, &ad_hs.c33, &ad_hs.c13, &ad_hs.c44, &ad_hs.c66};
            p[target_param]->value() += p_perturb;
        }
        
        std::vector<ADS> ad_thick(N);
        for(int i=0; i<N; ++i) ad_thick[i] = make_ads(thicknesses[i]);
        
        // 2. Find Root c (Phase Velocity)
        double c_local = root_c;
        ADS omega_ads = make_ads(omega);
        
        // Simple Newton Refinement (Assumption: root_c is close)
        for(int k=0; k<10; ++k) {
            ADS c_ads = make_ads(c_local); c_ads.derivatives()[0] = 1.0;
            // Use I24 for Rayleigh, Y(1) -> I24 for Love (via wrapper)
            auto F = calculateDispersionFunction(ad_layers, ad_thick, ad_hs, omega_ads, c_ads, type)(I24);
            c_local -= F.value() / F.derivatives()[0];
        }
        
        // 3. Compute dc/dω using Autodiff (Implicit Function Theorem)
        // Re-setup with autodiff::dual (forward mode) which is potentially faster/cleaner for scalar derivatives
        // But here we need to map from ADS to dual? Or just reuse logic?
        // Let's use autodiff::dual for the derivative part to match previous robust tests.
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
        
        // ∂F/∂c
        autodiff::seed(c_d);
        auto F_c = calculateDispersionFunction(d_layers, d_thick, d_hs, om_d, c_d, type)(I24);
        double dFdc = autodiff::grad(F_c);
        autodiff::unseed(c_d);
        
        // ∂F/∂ω
        c_d = c_local;
        autodiff::seed(om_d);
        auto F_om = calculateDispersionFunction(d_layers, d_thick, d_hs, om_d, c_d, type)(I24);
        double dFdom = autodiff::grad(F_om);
        autodiff::unseed(om_d);
        
        double dcdw = -dFdom / dFdc;
        
        // 4. Calculate Group Velocity U = c / (1 - (w/c) * dc/dw)
        return c_local / (1.0 - (omega/c_local)*dcdw);
    };
    
    // Finite Difference for dU/dp
    double delta = 1e-5; // Perturbation size
    double U_plus = get_U(delta);
    double U_minus = get_U(-delta);
    
    return (U_plus - U_minus) / (2.0 * delta);
}


// --- Robust Root Finder (Bisection) ---
double find_root_bisection(double bracket_min, double bracket_max, 
                           const std::vector<PhysicalParameters<Eigen::AutoDiffScalar<Eigen::VectorXd>>>& layers,
                           const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>>& thicknesses,
                           const PhysicalParameters<Eigen::AutoDiffScalar<Eigen::VectorXd>>& halfSpace,
                           double omega_val,
                           WaveType type) {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Constant AD make function
    auto make_const = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a;
    };
    
    ADS omega = make_const(omega_val);
    
    double low = bracket_min;
    double high = bracket_max;
    double tol = 1e-10;
    
    // Check brackets
    ADS c_low = make_const(low);
    auto res_low = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_low, type);
    double f_low = res_low(I24).value();
    
    ADS c_high = make_const(high);
    auto res_high = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_high, type);
    double f_high = res_high(I24).value();
    
    if (f_low * f_high > 0) {
        // Bracketing failed? Try to expand or search?
        // Fallback: Check if one is closer to zero or return nan
        // For testing, just return nan to signal failure
        // std::cerr << "Root logic: Bracket failure [" << low << "," << high << "] f_low=" << f_low << " f_high=" << f_high << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    for (int i=0; i<100; ++i) {
        double mid = (low + high) / 2.0;
        ADS c_mid = make_const(mid);
        auto res_mid = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_mid, type);
        double f_mid = res_mid(I24).value();
        
        if (std::abs(f_mid) < 1e-15) return mid;
        if (std::abs(high - low) < tol) return mid;
        
        if (f_low * f_mid <= 0) {
            high = mid;
            f_high = f_mid; // though we don't use f_high loop var
        } else {
            low = mid;
            f_low = f_mid;
        }
    }
    return (low + high) / 2.0;
}

// --- Romberg Integration ---

// Evaluates the Kernel K(omega) for a specific parameter
struct KernelEvaluator {
    std::vector<PhysicalParameters<double>> layers;
    std::vector<double> thicknesses;
    PhysicalParameters<double> halfSpace;
    int target_layer;
    int target_param;
    WaveType type;
    
    // Cache for root finding
    mutable double last_root_c; 
    
    KernelEvaluator() : last_root_c(0) {}
    
    double operator()(double omega) const {
        // Need to find root c at this omega first
        // Simple search around last_root_c
        double c_guess = (last_root_c > 0) ? last_root_c : 2.5; 
        
        // Use Bisection for robustness
        // Create ADS params
        using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
        auto make = [](double v) { ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(1); return a; };
        
        std::vector<PhysicalParameters<ADS>> al(layers.size());
        // Manual conversion AD
        for(size_t i=0; i<layers.size(); ++i) {
             al[i].rho = make(layers[i].rho); al[i].c11 = make(layers[i].c11); al[i].c33 = make(layers[i].c33); 
             al[i].c13 = make(layers[i].c13); al[i].c44 = make(layers[i].c44); al[i].c66 = make(layers[i].c66);
        }
        PhysicalParameters<ADS> ah;
        ah.rho = make(halfSpace.rho); ah.c11 = make(halfSpace.c11); ah.c33 = make(halfSpace.c33);
        ah.c13 = make(halfSpace.c13); ah.c44 = make(halfSpace.c44); ah.c66 = make(halfSpace.c66);
        std::vector<ADS> at(layers.size());
        for(size_t i=0; i<layers.size(); ++i) at[i] = make(thicknesses[i]);
        
        // Bracket search range: [0.9*c, 1.1*c] usually sufficient for small steps
        // But if jump is large, [0.5*c, 1.5*c]
        double bracket_width = 0.5; // km/s
        double min_c = std::max(0.5, c_guess - bracket_width);
        double max_c = c_guess + bracket_width;
        
        // Robust scan if bracket fails?
        // Basic bisection
        double c_curr = find_root_bisection(min_c, max_c, al, at, ah, omega, type);
        
        if (std::isnan(c_curr)) {
            // Try wider search?
             c_curr = find_root_bisection(0.5, 4.5, al, at, ah, omega, type);
        }

        if (!std::isnan(c_curr)) {
            last_root_c = c_curr;
        }
        
        // Compute Kernel
        // dU/dp
        double K = computeDerivativeHybrid(layers, thicknesses, halfSpace, omega, c_curr, target_layer, target_param, type);
        
        // Debug output for trace
        if (std::isnan(K) || std::isnan(c_curr)) {
            std::cerr << "NaN detected at Freq=" << omega/(2*M_PI) << " Hz, c_root=" << c_curr << ", K=" << K << std::endl;
        } else {
            // std::cout << "F=" << omega/(2*M_PI) << " c=" << c_curr << " K=" << K << std::endl;
        }
        
        return K;
    }
};

void run_romberg_test() {
    // Defined Model
    std::vector<PhysicalParameters<double>> layers = { makeIsotropic(2.5, 4.3, 2.2), makeIsotropic(3.2, 5.5, 2.45) };
    std::vector<double> thicknesses = {5.0, 10.0};
    PhysicalParameters<double> halfSpace = makeIsotropic(4.0, 6.9, 2.7);
    
    // Parameter: Layer 0, L (Shear Modulus) - usually sensitive
    KernelEvaluator eval;
    eval.layers = layers; eval.thicknesses = thicknesses; eval.halfSpace = halfSpace;
    eval.target_layer = 0; eval.target_param = 4; // L
    eval.type = WaveType::Rayleigh;
    
    // Frequency Range (Safe range)
    double f_start = 0.2; // Avoid very low freq instability
    double f_end = 0.5;
    double w_start = 2.0 * M_PI * f_start;
    double w_end = 2.0 * M_PI * f_end;
    
    // Initialize root guess
    eval.last_root_c = 2.4; // rough guess
    
    // Romberg Integration
    // R[i][j] where i is refinement level, j is order of extrapolation
    // level 0: 1 interval (h)
    // level 1: 2 intervals (h/2)
    // level 2: 4 intervals (h/4)
    // ...
    // level 6: 64 intervals
    
    const int MAX_LEVEL = 7;
    double R[MAX_LEVEL][MAX_LEVEL];
    
    double h = w_end - w_start;
    
    // Level 0 (Trapezoidal with 1 interval)
    R[0][0] = 0.5 * h * (eval(w_start) + eval(w_end));
    std::cout << "Level 0: " << R[0][0] << std::endl;
    
    int n = 1;
    for (int i = 1; i < MAX_LEVEL; ++i) {
        double sum = 0.0;
        // Compute new points (midpoints)
        double h_i = h / n;      // previous step size
        double h_next = h_i / 2.0; // new step size
        
        double w_curr = w_start + h_next;
        for (int k = 0; k < n; ++k) {
            sum += eval(w_curr);
            w_curr += h_i;
        }
        
        // Trapezoidal recurrence
        R[i][0] = 0.5 * R[i-1][0] + h_next * sum;
        
        // Richardson Extrapolation
        double power_of_4 = 4.0;
        for (int j = 1; j <= i; ++j) {
            R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (power_of_4 - 1.0);
            power_of_4 *= 4.0;
        }
        
        n *= 2;
        
        std::cout << "Level " << i << " (" << n << " intervals): ";
        for(int j=0; j<=i; ++j) {
            std::cout << std::fixed << std::setprecision(6) << R[i][j] << " ";
        }
        
        // Convergence Check (Relative error between highest order and previous)
        double diff = std::abs(R[i][i] - R[i-1][i-1]);
        double rel_err = diff / std::abs(R[i][i]);
        std::cout << " | RelErr: " << std::scientific << rel_err;
        
        if(rel_err < 1e-6) std::cout << " [CONVERGED]";
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "=== Romberg Integration Stability Test ===" << std::endl;
    run_romberg_test();
    return 0;
}
