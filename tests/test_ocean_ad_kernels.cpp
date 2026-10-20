#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "disper26.cpp"
#include "sensitivity_utils.hpp"

// Setup Ocean Model
// Layer 0: Water (2 km)
// Layer 1: Sediment (1 km)
// Halfspace: Basement
void setupOceanModel(
    std::vector<PhysicalParameters<double>>& layers,
    std::vector<double>& thicknesses,
    PhysicalParameters<double>& halfSpace
) {
    // Water
    PhysicalParameters<double> water;
    double vp_w = 1.5;
    double rho_w = 1.0;
    water.rho = rho_w;
    water.c11 = rho_w * vp_w * vp_w;
    water.c33 = water.c11;
    water.c13 = water.c11;
    water.c44 = 0.0;
    water.c66 = 0.0;
    
    // Sediment
    auto sed = createIsotropicParameters<double>(1.0, 2.0, 1.8);
    
    // Basement
    halfSpace = createIsotropicParameters<double>(3.0, 5.2, 2.7);
    
    layers = {water, sed};
    thicknesses = {2.0, 1.0};
}

// Helper to ensure AD size consistency
template <typename T>
T seed_constant(double val, int size) {
    if constexpr (std::is_same_v<T, double>) {
        return val;
    } else {
        T ad;
        ad.value() = val;
        ad.derivatives() = Eigen::VectorXd::Zero(size);
        return ad;
    }
}

int main() {
    std::cout << "Starting Ocean Layer Full AD Kernel Verification..." << std::endl;

    std::vector<PhysicalParameters<double>> layers;
    std::vector<double> thicknesses;
    PhysicalParameters<double> halfSpace;
    setupOceanModel(layers, thicknesses, halfSpace);

    double freq = 1.0; // 1.0 Hz (better for ocean model)
    double omega = 2.0 * M_PI * freq;

    // Search for root (Scholte/Rayleigh wave)
    double c_min = 1.8;
    double c_max = 1.9;
    double root_c = brents_method([&](double c) {
        auto res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, c, WaveType::Rayleigh);
        return res(I24);
    }, c_min, c_max);

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Root c = " << root_c << " at f=" << freq << std::endl << std::endl;

    std::cout << "Computing Hybrid AD Kernels..." << std::endl;
    auto hybrid_kernels = computeSensitivityKernelsFD(layers, thicknesses, halfSpace, freq, root_c, WaveType::Rayleigh);

    std::cout << "Computing Full AD Kernels..." << std::endl;
    // We can get U from OuterAD version of calculateGroupVelocity
    using OuterAD = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    int total_params = (layers.size() + 1) * 3;
    OuterAD omega_ad = seed_constant<OuterAD>(omega, total_params);
    OuterAD c_ad = seed_constant<OuterAD>(root_c, total_params);
    
    // Create l_ad, t_ad, h_ad manually for consistency check
    std::vector<PhysicalParameters<OuterAD>> l_ad;
    for(auto& layer : layers) {
        PhysicalParameters<OuterAD> pa;
        pa.rho = seed_constant<OuterAD>(layer.rho, total_params);
        pa.c11 = seed_constant<OuterAD>(layer.c11, total_params);
        pa.c33 = seed_constant<OuterAD>(layer.c33, total_params);
        pa.c44 = seed_constant<OuterAD>(layer.c44, total_params);
        pa.c66 = seed_constant<OuterAD>(layer.c66, total_params);
        pa.c13 = seed_constant<OuterAD>(layer.c13, total_params);
        l_ad.push_back(pa);
    }
    std::vector<OuterAD> t_ad;
    for(auto& t : thicknesses) t_ad.push_back(seed_constant<OuterAD>(t, total_params));
    PhysicalParameters<OuterAD> h_ad;
    h_ad.rho = seed_constant<OuterAD>(halfSpace.rho, total_params);
    h_ad.c11 = seed_constant<OuterAD>(halfSpace.c11, total_params);
    h_ad.c33 = seed_constant<OuterAD>(halfSpace.c33, total_params);
    h_ad.c44 = seed_constant<OuterAD>(halfSpace.c44, total_params);
    h_ad.c66 = seed_constant<OuterAD>(halfSpace.c66, total_params);
    h_ad.c13 = seed_constant<OuterAD>(halfSpace.c13, total_params);

    OuterAD U_ad = calculateGroupVelocityFullAD(l_ad, t_ad, h_ad, omega_ad, c_ad, WaveType::Rayleigh);
    
    // Compute H/V ratio at the seabed (using the new seabedState capability)
    Eigen::Matrix<double, 5, 1> seabedState;
    seabedState.setZero();
    auto res_surface = calculateDispersionFunction(layers, thicknesses, halfSpace, omega, root_c, WaveType::Rayleigh, NormalizationMode::Vertical, &seabedState);
    double hv_seabed = calculateHVRatio(seabedState);

    std::cout << "U (Hybrid/Baseline): " << calculateGroupVelocity(layers, thicknesses, halfSpace, omega, root_c) << std::endl;
    std::cout << "U (Full AD Value):   " << U_ad.value() << std::endl;
    std::cout << "H/V Ratio (Seabed):  " << hv_seabed << std::endl;

    auto full_ad_kernels = computeSensitivityKernelsFullAD(layers, thicknesses, halfSpace, freq, root_c, WaveType::Rayleigh);

    std::cout << std::endl << "Comparison (Full AD):" << std::endl;
    std::cout << std::left << std::setw(10) << "Param" 
              << std::setw(15) << "dU/dp" 
              << std::setw(15) << "d(HV)/dp" << std::endl;
    
    for (size_t i = 0; i < full_ad_kernels.size(); ++i) {
        std::string label = full_ad_kernels[i].parameterName + "_" + std::to_string(full_ad_kernels[i].layerIndex);
        std::cout << std::left << std::setw(10) << label
                  << std::setw(15) << full_ad_kernels[i].d_GroupVelocity
                  << std::setw(15) << full_ad_kernels[i].d_HVRatio << std::endl;
    }
    
    // Perform manual careful FD for Vp_0 (Water Vp)
    auto calc_U_manual = [&](const std::vector<PhysicalParameters<double>>& l, const PhysicalParameters<double>& hs) {
        double r = brents_method([&](double c) {
            auto res = calculateDispersionFunction(l, thicknesses, hs, omega, c, WaveType::Rayleigh);
            return res(I24);
        }, 1.8, 2.0); // Narrow range
        return calculateGroupVelocity(l, thicknesses, hs, omega, r);
    };

    double eps_fd = 1e-4;
    double Vp0_val = layers[0].c33 / layers[0].rho; // alpha^2
    Vp0_val = std::sqrt(Vp0_val);
    double dVp = Vp0_val * eps_fd;

    layers[0].c11 = layers[0].rho * (Vp0_val + dVp) * (Vp0_val + dVp);
    layers[0].c33 = layers[0].c11; layers[0].c13 = layers[0].c11;
    double U_plus = calc_U_manual(layers, halfSpace);

    layers[0].c11 = layers[0].rho * (Vp0_val - dVp) * (Vp0_val - dVp);
    layers[0].c33 = layers[0].c11; layers[0].c13 = layers[0].c11;
    double U_minus = calc_U_manual(layers, halfSpace);
    
    double dU_dVp0_fd = (U_plus - U_minus) / (2.0 * dVp);

    // Reset model
    layers[0].c11 = layers[0].rho * Vp0_val * Vp0_val;
    layers[0].c33 = layers[0].c11; layers[0].c13 = layers[0].c11;

    std::cout << std::endl << "Manual FD Check (Vp_0):" << std::endl;
    std::cout << "Full AD dU/dVp_0: " << full_ad_kernels[1].d_GroupVelocity << std::endl;
    std::cout << "Manual FD dU/dVp_0: " << dU_dVp0_fd << std::endl;

    // Perform manual careful FD for Vs_1 (Sediment Vs)
    double Vs1_val = std::sqrt(layers[1].c44 / layers[1].rho);
    double dVs = Vs1_val * eps_fd;

    layers[1].c44 = layers[1].rho * (Vs1_val + dVs) * (Vs1_val + dVs);
    layers[1].c66 = layers[1].c44; layers[1].c13 = layers[1].c33 - 2.0 * layers[1].c44;
    double U_plus_s = calc_U_manual(layers, halfSpace);

    layers[1].c44 = layers[1].rho * (Vs1_val - dVs) * (Vs1_val - dVs);
    layers[1].c66 = layers[1].c44; layers[1].c13 = layers[1].c33 - 2.0 * layers[1].c44;
    double U_minus_s = calc_U_manual(layers, halfSpace);

    double dU_dVs1_fd = (U_plus_s - U_minus_s) / (2.0 * dVs);

    std::cout << std::endl << "Manual FD Check (Vs_1):" << std::endl;
    std::cout << "Full AD dU/dVs_1: " << full_ad_kernels[3].d_GroupVelocity << std::endl;
    std::cout << "Manual FD dU/dVs_1: " << dU_dVs1_fd << std::endl;

    bool all_pass = true;
    if (std::abs(full_ad_kernels[1].d_GroupVelocity - dU_dVp0_fd) > 1e-4) all_pass = false;
    if (std::abs(full_ad_kernels[3].d_GroupVelocity - dU_dVs1_fd) > 1e-3) all_pass = false;

    if (all_pass) std::cout << std::endl << "Overall Manual FD Verification: PASS" << std::endl;
    else std::cout << std::endl << "Overall Manual FD Verification: FAIL" << std::endl;

    return all_pass ? 0 : 1;
}
