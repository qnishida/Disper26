// Fixed-root U timings. "prepared" excludes material conversion/allocation;
// "scalar" times the same normalized propagation without derivatives.
#include <chrono>
#include <iomanip>
#include "../src/disper26.cpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: benchmark_group_velocity layers iterations Rayleigh|Love group|prepared|scalar\n";
        return 2;
    }
    const int n = std::stoi(argv[1]), iterations = std::stoi(argv[2]);
    const std::string wave_name = argv[3], operation = argv[4];
    if (n < 1 || iterations < 1 ||
        (wave_name != "Rayleigh" && wave_name != "Love") ||
        (operation != "group" && operation != "prepared" && operation != "scalar")) return 2;
    const WaveType wave = wave_name == "Love" ? WaveType::Love : WaveType::Rayleigh;
    std::vector<PhysicalParameters<double>> layers;
    for (int i = 0; i < n; ++i) {
        const double x = double(i) / n;
        layers.push_back(createIsotropicParameters(3.0 + 0.5*x, 5.2 + 0.8*x, 2.5 + 0.3*x));
    }
    const std::vector<double> thicknesses(n, 100.0/n);
    const auto hs = createIsotropicParameters(4.6, 8.0, 3.3);
    const double omega = 2.0 * std::acos(-1.0) / 20.0;
    auto residual = [&](double c) {
        return calculateDispersionFunction(layers, thicknesses, hs, omega, c, wave)(I24);
    };
    const auto roots = searchRoots(residual, 2.5, 4.5, 0.005, 1e-10);
    if (roots.empty()) return 1;
    const double root = roots.front();
    using AD = Eigen::AutoDiffScalar<Eigen::Vector2d>;
    auto convert = [](const PhysicalParameters<double>& p) {
        PhysicalParameters<AD> a;
        a.rho=p.rho; a.c11=p.c11; a.c13=p.c13; a.c33=p.c33;
        a.c44=p.c44; a.c66=p.c66; a.gravity=p.gravity;
        a.gravity_internal_scale=p.gravity_internal_scale;
        a.gravity_surface_scale=p.gravity_surface_scale;
        return a;
    };
    std::vector<PhysicalParameters<AD>> prepared;
    for (const auto& p : layers) prepared.push_back(convert(p));
    const std::vector<AD> prepared_h(thicknesses.begin(), thicknesses.end());
    const auto prepared_hs = convert(hs);
    AD w(omega), c(root);
    w.derivatives()(0)=1.0; c.derivatives()(1)=1.0;
    auto evaluate = [&]() {
        if (operation == "group")
            return calculateGroupVelocityFullAD(layers, thicknesses, hs, omega, root, wave);
        if (operation == "scalar")
            return calculateDispersionFunction(layers, thicknesses, hs, omega, root, wave,
                                              NormalizationMode::Vertical)(I24);
        const AD d = calculateDispersionFunction(prepared, prepared_h, prepared_hs, w, c, wave,
                                                 NormalizationMode::Vertical)(I24);
        return root / (1.0 + omega/root * d.derivatives()(0)/d.derivatives()(1));
    };
    volatile double warmup=0.0;
    for (int j=0; j<50; ++j) warmup=evaluate();
    (void)warmup;
    double checksum=0.0;
    const auto start=std::chrono::steady_clock::now();
    for (int j=0; j<iterations; ++j) checksum+=evaluate();
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (!std::isfinite(checksum)) return 1;
    std::cout << std::setprecision(17) << elapsed/iterations << ' ' << checksum/iterations << ' ' << root << '\n';
}
