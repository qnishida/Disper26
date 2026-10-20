#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include <limits>

#include "../src/disper26.cpp"

// Re-using the logic from sensitivity_utils (simplified for test) to verify stability

// 1. Pure FD (Double Difference)
// dU/dp = (U(p+dp) - U(p-dp)) / 2dp
// U itself calculated via Finite Difference of c?
// U = c / (1 - (w/c) * dc/dw).
// PureFD means dc/dw is ALSO calculated by difference.
// U_pure_fd = c / (1 - (w/c) * (c(w+dw) - c(w-dw))/2dw)

// 2. Hybrid AD
// U_hybrid_ad = c / (1 - (w/c) * (dc/dw)_AD)
// dU/dp = (U_hybrid(p+dp) - U_hybrid(p-dp)) / 2dp

// 3. (Theoretical Ref)
// If we had Full AD, it would be exact.
// We check stability by varying dp. Use Richardson Extrapolation or step size check.

double get_U_pure_fd(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq,
    double c_guess
) {
    // 1. Solve c(w)
    // Helper to solve c at f
    auto solve_c = [&](double f_v) {
        // Simple search
        double w_v = 2.0*M_PI*f_v;
        
        // Scan around guess
        double c_min = c_guess - 0.1;
        double steps = 50;
        double dc = 0.2/steps;
        
        auto func = [&](double c) {
             using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
             // Standard double calc needed but we reuse template with constant AD
             ADS c_a; c_a.value()=c; c_a.derivatives().setZero(1);
             ADS w_a; w_a.value()=w_v; w_a.derivatives().setZero(1);
             auto to_ads = [](const PhysicalParameters<double>& p) {
                PhysicalParameters<ADS> pa; pa.rho=ADS(p.rho); pa.rho.derivatives().setZero(1);
                pa.c11=ADS(p.c11); pa.c11.derivatives().setZero(1); pa.c33=ADS(p.c33); pa.c33.derivatives().setZero(1);
                pa.c44=ADS(p.c44); pa.c44.derivatives().setZero(1); pa.c66=ADS(p.c66); pa.c66.derivatives().setZero(1);
                pa.c13=ADS(p.c13); pa.c13.derivatives().setZero(1); return pa;
             };
             std::vector<PhysicalParameters<ADS>> l_a;
             std::vector<ADS> t_a;
             for(auto& p:layers) l_a.push_back(to_ads(p));
             for(double t:thicknesses) { ADS ta; ta.value()=t; ta.derivatives().setZero(1); t_a.push_back(ta); }
             PhysicalParameters<ADS> h_a = to_ads(halfSpace);
             return calculateDispersionFunction(l_a, t_a, h_a, w_a, c_a, WaveType::Rayleigh)(I24).value();
        };
        return brents_method(func, c_guess-0.05, c_guess+0.05, 1e-10, 100);
    };

    double dw = 1e-4 * (2.0*M_PI*freq);
    double c_mid = solve_c(freq);
    double c_plus = solve_c(freq * (1.0 + 1e-4));
    double c_minus = solve_c(freq * (1.0 - 1e-4));
    
    if(c_mid < 0) return 0.0;
    
    // dc/dw
    double dc_dw = (c_plus - c_minus) / (2.0 * (dw/ (2.0*M_PI) * 2.0*M_PI) * 1e4 * 1e-4 * 2.0*M_PI); 
    // Wait, f * (1+eps). w_plus - w_minus = 2*pi*f*2eps.
    double w_plus = 2.0*M_PI*freq*(1.0 + 1e-4);
    double w_minus = 2.0*M_PI*freq*(1.0 - 1e-4);
    dc_dw = (c_plus - c_minus) / (w_plus - w_minus);
    
    return c_mid / (1.0 - (2.0*M_PI*freq / c_mid) * dc_dw);
}

double get_U_hybrid_ad(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpace,
    double freq,
    double c_guess
) {
    // 1. Solve c
     auto solve_c = [&](double f_v) {
        // (Same logic as above, condensed)
        auto func = [&](double c) {
             using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
             ADS c_a; c_a.value()=c; c_a.derivatives().setZero(1);
             ADS w_a; w_a.value()=2.0*M_PI*f_v; w_a.derivatives().setZero(1);
             // ... construct objects ...
             // Skip full reconstruction for brevity, assume we have it
             // Actually need to construct them
             PhysicalParameters<ADS> hs_a; hs_a.rho=ADS(halfSpace.rho); hs_a.rho.derivatives().setZero(1);
             hs_a.c11=ADS(halfSpace.c11); hs_a.c11.derivatives().setZero(1);
             hs_a.c33=ADS(halfSpace.c33); hs_a.c33.derivatives().setZero(1);
             hs_a.c44=ADS(halfSpace.c44); hs_a.c44.derivatives().setZero(1);
             hs_a.c66=ADS(halfSpace.c66); hs_a.c66.derivatives().setZero(1);
             hs_a.c13=ADS(halfSpace.c13); hs_a.c13.derivatives().setZero(1);
             
             PhysicalParameters<ADS> l1_a; l1_a.rho=ADS(layers[0].rho); l1_a.rho.derivatives().setZero(1);
             l1_a.c11=ADS(layers[0].c11); l1_a.c11.derivatives().setZero(1);
             l1_a.c33=ADS(layers[0].c33); l1_a.c33.derivatives().setZero(1);
             l1_a.c44=ADS(layers[0].c44); l1_a.c44.derivatives().setZero(1);
             l1_a.c66=ADS(layers[0].c66); l1_a.c66.derivatives().setZero(1);
             l1_a.c13=ADS(layers[0].c13); l1_a.c13.derivatives().setZero(1);
             
             std::vector<PhysicalParameters<ADS>> l_as={l1_a};
             ADS t_a; t_a.value()=thicknesses[0]; t_a.derivatives().setZero(1);
             std::vector<ADS> t_as={t_a};
             
             return calculateDispersionFunction(l_as, t_as, hs_a, w_a, c_a, WaveType::Rayleigh)(I24).value();
        };
        return brents_method(func, c_guess-0.05, c_guess+0.05, 1e-10, 100);
    };
    
    double c_exact = solve_c(freq);
    
    // 2. AD for dc/dw
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    ADS c_a; c_a.value()=c_exact; c_a.derivatives() = Eigen::VectorXd::Unit(2, 0);
    ADS w_a; w_a.value()=2.0*M_PI*freq; w_a.derivatives() = Eigen::VectorXd::Unit(2, 1);
    
    // Construct ADS params with size 2
    auto to_ads = [](const PhysicalParameters<double>& p) {
        PhysicalParameters<ADS> pa;
        pa.rho=ADS(p.rho); pa.rho.derivatives().setZero(2);
        pa.c11=ADS(p.c11); pa.c11.derivatives().setZero(2);
        pa.c33=ADS(p.c33); pa.c33.derivatives().setZero(2);
        pa.c44=ADS(p.c44); pa.c44.derivatives().setZero(2);
        pa.c66=ADS(p.c66); pa.c66.derivatives().setZero(2);
        pa.c13=ADS(p.c13); pa.c13.derivatives().setZero(2);
        return pa;
    };
    std::vector<PhysicalParameters<ADS>> l_as; for(auto& p:layers) l_as.push_back(to_ads(p));
    std::vector<ADS> t_as; for(double t:thicknesses) { ADS ta; ta.value()=t; ta.derivatives().setZero(2); t_as.push_back(ta); }
    PhysicalParameters<ADS> h_as = to_ads(halfSpace);
    
    auto res = calculateDispersionFunction(l_as, t_as, h_as, w_a, c_a, WaveType::Rayleigh);
    ADS Delta = res(I24);
    double dc_dw = - Delta.derivatives()[1] / Delta.derivatives()[0];
    
    return c_exact / (1.0 - (w_a.value()/c_exact)*dc_dw);
}


int main() {
    // Model Setup
    double H = 10.0;
    PhysicalParameters<double> L1, HS;
    double vs1 = 3.0, rho1 = 2.5, vp1 = vs1*1.732;
    L1.rho = rho1; L1.c44 = rho1*vs1*vs1; L1.c11 = rho1*vp1*vp1; L1.c33=L1.c11; L1.c66=L1.c44; L1.c13=L1.c11-2*L1.c44;
    double vs2 = 4.0, rho2 = 3.0, vp2 = vs2*1.732;
    HS.rho = rho2; HS.c44 = rho2*vs2*vs2; HS.c11 = rho2*vp2*vp2; HS.c33=HS.c11; HS.c66=HS.c44; HS.c13=HS.c11-2*HS.c44;
    
    std::vector<PhysicalParameters<double>> layers = {L1};
    std::vector<double> thicknesses = {H};
    double freq = 0.2;
    double root_c = 3.615;

    std::cout << "--- Romberg Feasibility Check (Error Scaling) ---" << std::endl;
    std::cout << "Delta | Val | Diff | log2(Diff_ratio)" << std::endl;
    
    // Check if error scales as O(h^2) (which is required for Romberg)
    // Central Difference Error ~ O(h^2).
    // Hybrid AD calculates U precisely.
    // So (U(p+h) - U(p-h))/(2h) should have O(h^2) error.
    
    double prev_val = 0.0;
    double prev_diff = 0.0;
    
    for(int k=2; k<=10; ++k) {
        double d = std::pow(0.5, k); // Halving step size
        
        // Perturb
        layers[0].rho = rho1 * (1.0 + d);
        layers[0].c44 = layers[0].rho * vs1 * vs1; layers[0].c11=layers[0].rho*vp1*vp1; layers[0].c33=layers[0].c11; layers[0].c66=layers[0].c44; layers[0].c13=layers[0].c11-2*layers[0].c44;
        double U_plus = get_U_hybrid_ad(layers, thicknesses, HS, freq, root_c);
        
        layers[0].rho = rho1 * (1.0 - d);
        layers[0].c44 = layers[0].rho * vs1 * vs1; layers[0].c11=layers[0].rho*vp1*vp1; layers[0].c33=layers[0].c11; layers[0].c66=layers[0].c44; layers[0].c13=layers[0].c11-2*layers[0].c44;
        double U_minus = get_U_hybrid_ad(layers, thicknesses, HS, freq, root_c);
        
        layers[0] = L1; // Reset
        
        // Finite Difference Approximation
        double val = (U_plus - U_minus) / (2.0 * rho1 * d);
        
        double diff = std::abs(val - prev_val);
        double ratio = (prev_diff > 1e-20) ? std::log2(prev_diff / diff) : 0.0;
        
        std::cout << std::scientific << d << " | " << val << " | " << diff << " | " << ratio << std::endl;
        
        prev_val = val;
        prev_diff = diff;
    }
    
    std::cout << "Note: If Ratio approaches 2.0 (order 2), Romberg will work perfectly." << std::endl;
    
    return 0;
}
