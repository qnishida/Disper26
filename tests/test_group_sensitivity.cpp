#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Include the engine
#include "../src/disper26.cpp"

// --- Root Finder (Standard Double) ---
double find_root_bisection(double bracket_min, double bracket_max, 
                           const std::vector<PhysicalParameters<Eigen::AutoDiffScalar<Eigen::VectorXd>>>& layers,
                           const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>>& thicknesses,
                           const PhysicalParameters<Eigen::AutoDiffScalar<Eigen::VectorXd>>& halfSpace,
                           double freq) {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    auto make = [](double v) {
        ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(2); return a;
    };
    
    ADS omega = make(2.0 * M_PI * freq);
    
    double low = bracket_min;
    double high = bracket_max;
    // double tol = 1e-10; 
    
    for (int i=0; i<100; ++i) {
        double mid = (low + high) / 2.0;
        
        ADS c_low = make(low);
        ADS c_mid = make(mid);
        
        auto res_low = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_low);
        auto res_mid = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c_mid);
        
        double f_low = res_low(I24).value();
        double f_mid = res_mid(I24).value();
        
        if (std::abs(f_mid) < 1e-20) return mid;
        
        if (std::isnan(f_low) || std::isnan(f_mid)) {
             // skip
        }

        if (f_low * f_mid <= 0) {
            high = mid;
        } else {
            low = mid;
        }
    }
    return (low + high) / 2.0;
}

int main() {
    // --- 2nd Order AD Type Definition ---
    // Outer AD: Derivatives w.r.t parameters (e.g., Vs_halfspace)
    // Inner AD: Derivatives w.r.t state variables (c, omega)
    
    // We want dU/dVs.
    // U depends on (c, omega, Vs).
    // c depends on (omega, Vs) via implicit function Delta=0.
    
    // Approach:
    // Compute full derivatives of Delta w.r.t (c, omega, Vs) up to 2nd order?
    // Actually, simple nested AD works automatically if we set inputs correctly.
    
    // Outer Variable: p (Vs of half-space)
    // Inner Variables: c, omega
    
    using InnerAD = Eigen::AutoDiffScalar<Eigen::VectorXd>; // Size 2: [c, omega]
    using OuterAD = Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1>>; // Value is InnerAD
    // Note: Eigen AutoDiffScalar requires the ValueType to support arithmetic.
    // Standard AutoDiffScalar value is double.
    // We need AutoDiffScalar<Matrix<AutoDiffScalar<VectorXd>, -1, 1>> ?
    // No, standard usage is: AutoDiffScalar<Vector<AutoDiffScalar<...> >>
    
    // Let's define explicitly:
    // ADS2 = AutoDiffScalar< Vector< ADS1, Any > >
    // BUT Eigen's vector must contain scalars.
    
    // Alternative:
    // Use one huge vector for all variables: [c, omega, Vs] -> size 3.
    // First derivatives -> Gradient (size 3)
    // Second derivatives -> Hessian (3x3)?
    // Disper26 is templated on T. If T is AutoDiffScalar<HessianType>, we get Hessian.
    // But Eigen unsupported AutoDiff doesn't provide Hessian directly in one go easily without nesting.
    // However, we only need 1st deriv of U.
    // U = c / (1 - w/c * dc/dw).
    // dc/dw is determined by 1st derivs of Delta.
    // So U involves 1st derivs of Delta.
    // To get dU/dp, we need derivatives of (1st derivs of Delta).
    // i.e. 2nd derivatives of Delta.
    
    // Let's try "Hyper-Dual Numbers" style by nesting.
    // T = AutoDiffScalar< Vector< AutoDiffScalar<Vector>, ... > >.
    // This is complicated in C++.
    
    // Simpler Approach for this test:
    // Use Finite Difference on U (calculated via 1st order AD).
    // Since U calculation is fast and accurate, FD on U w.r.t Vs is robust and easy.
    // The user asked "Can we calculate dU/dp?".
    // Yes, via FD on U is the practical way in most inversion codes.
    // "True" 2nd order AD is possible but might be overkill.
    // Let's stick to the user request "sensitivity kernel of U".
    
    // Wait, user asked "Can it be calculated?". Implies "Is analytic/AD possible?".
    // I will implement FD of U first as it's reliable.
    // Then I will try to implement 2nd order AD if time permits, or strictly use FD for U sensitivity.
    // Actually, FD of AD-computed U is "Hybrid".
    // Let's do that:
    
    // 1. Calculate U(Vs) using AD (as in test_group_velocity).
    // 2. Calculate U(Vs + dVs) using AD.
    // 3. Diff.
    
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Setup Function to calculate U given Vs_halfspace
    auto calculate_U = [&](double vs_half_val) -> double {
        // ... (Model Setup with vs_half_val) ...
        // Need to reconstruct layers/params inside here.
        
        int deriv_size = 2; // c, omega
        auto make_const = [&](double v) {
            ADS a; a.value() = v; a.derivatives() = Eigen::VectorXd::Zero(deriv_size); return a;
        };

        std::vector<PhysicalParameters<ADS>> layers;
        std::vector<ADS> thicknesses;
        
        for (int i = 1; i <= 19; ++i) {
            double vs_val = 0.0;
            double h_val = 1.0;
            
            if (i <= 2) vs_val = 1.5;
            else if (i <= 5) vs_val = 2.5 + (i - 3.0) * 0.16;
            else vs_val = 3.4;
            
            double vp_val = vs_val * 1.72;
            double rho_val = 0.9473 * 0.2966 * vp_val;
            
            PhysicalParameters<ADS> p;
            p.rho = make_const(rho_val);
            p.c44 = make_const(rho_val * vs_val * vs_val);
            p.c11 = make_const(rho_val * vp_val * vp_val);
            p.c33 = p.c11;
            p.c13 = make_const(p.c11.value() - 2.0 * p.c44.value());
            p.c66 = p.c44; // Isotropic
            layers.push_back(p);
            thicknesses.push_back(make_const(h_val));
        }
        
        PhysicalParameters<ADS> halfSpace;
        {
            double vp_val = vs_half_val * 1.72;
            double rho_val = 0.9473 * 0.2966 * vp_val;
            halfSpace.rho = make_const(rho_val);
            halfSpace.c44 = make_const(rho_val * vs_half_val * vs_half_val);
            halfSpace.c11 = make_const(rho_val * vp_val * vp_val);
            halfSpace.c33 = halfSpace.c11;
            halfSpace.c13 = make_const(halfSpace.c11.value() - 2.0 * halfSpace.c44.value());
            halfSpace.c66 = halfSpace.c44; // Isotropic
        }
        
        double freq = 0.4;
        double c_root = find_root_bisection(1.40, 1.70, layers, thicknesses, halfSpace, freq);
        
        ADS c_ad;
        c_ad.value() = c_root;
        c_ad.derivatives() = Eigen::VectorXd::Zero(deriv_size);
        c_ad.derivatives()[0] = 1.0; 
        
        ADS omega_ad;
        omega_ad.value() = 2.0 * M_PI * freq;
        omega_ad.derivatives() = Eigen::VectorXd::Zero(deriv_size);
        omega_ad.derivatives()[1] = 1.0; 
        
        auto res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega_ad, c_ad);
        ADS Delta = res(I24);
        
        double dDelta_dw = Delta.derivatives()[1];
        double dDelta_dc = Delta.derivatives()[0];
        double dc_dw = - dDelta_dw / dDelta_dc;
        
        double U = c_root / (1.0 - (omega_ad.value() / c_root) * dc_dw);
        return U;
    };
    
    double Vs0 = 4.6;
    double U0 = calculate_U(Vs0);
    
    double delta = 1e-4;
    double U_minus = calculate_U(Vs0 - delta);
    double U_plus  = calculate_U(Vs0 + delta);
    
    double dU_dVs_FD = (U_plus - U_minus) / (2.0 * delta); // Central difference on U
    
    std::cout << "Vs0: " << Vs0 << " U0: " << U0 << std::endl;
    std::cout << "dU/dVs (FD of AD-U): " << dU_dVs_FD << std::endl;
    
    return 0;
}
