// Fixed-velocity propagation timing, independent of root-finder iteration
// changes. Compile against an old source with -DDISPER26_BENCHMARK_SOURCE=...
// and its src include directory; use the same compiler/flags for both builds.
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#ifndef DISPER26_BENCHMARK_SOURCE
#define DISPER26_BENCHMARK_SOURCE "../src/disper26.cpp"
#endif
#include DISPER26_BENCHMARK_SOURCE

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: benchmark_dispersion_math L iterations Rayleigh|Love\n";
        return 2;
    }
    const int n = std::stoi(argv[1]);
    const int iterations = std::stoi(argv[2]);
    const std::string wave(argv[3]);
    if (n <= 0 || iterations <= 0 || (wave != "Rayleigh" && wave != "Love")) return 2;
    auto material = [](double vs, double vp, double rho) {
        PhysicalParameters<double> p;
        p.rho = rho;
        p.c11 = p.c33 = rho * vp * vp;
        p.c44 = p.c66 = rho * vs * vs;
        p.c13 = p.c11 - 2.0 * p.c44;
        return p;
    };
    std::vector<PhysicalParameters<double>> layers;
    for (int i = 0; i < n; ++i) {
        const double t = double(i) / n;
        layers.push_back(material(3.0 + 0.5*t, 5.2 + 0.8*t, 2.5 + 0.3*t));
    }
    const std::vector<double> thicknesses(n, 100.0/n);
    const auto halfspace = material(4.6, 8.0, 3.3);
    const double omega = 2.0 * std::acos(-1.0) / 20.0;
    auto evaluate = [&](int j) {
        const double c = 3.9 + 1e-9 * (j % 31);
        return wave == "Rayleigh"
            ? calculateRayleighDispersion(layers, thicknesses, halfspace, omega, c)(0)
            : calculateLoveDispersion(layers, thicknesses, halfspace, omega, c)(0);
    };
    volatile double warmup = 0.0;
    for (int j = 0; j < 50; ++j) warmup = evaluate(j);
    (void)warmup;
    const auto start = std::chrono::steady_clock::now();
    double checksum = 0.0;
    for (int j = 0; j < iterations; ++j) checksum += evaluate(j);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (!std::isfinite(checksum)) return 1;
    std::cout << std::setprecision(17) << elapsed / iterations << ' ' << checksum << '\n';
}
