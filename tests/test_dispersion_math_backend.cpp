#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "../src/disper26.cpp"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool same(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return a == b && (a != 0.0 || std::signbit(a) == std::signbit(b));
}

bool same(std::complex<double> a, std::complex<double> b) {
    return same(a.real(), b.real()) && same(a.imag(), b.imag());
}

void check_values() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (double a : {0.0, -0.0, tiny, -tiny, 0.25, -0.25, 0.5,
                     std::nextafter(0.5, 1.0), 1.0, 20.0, 710.0, inf, -inf, nan}) {
        for (const auto& z : {std::complex<double>(a, 0.0),
                              std::complex<double>(a, -0.0),
                              std::complex<double>(0.0, a),
                              std::complex<double>(-0.0, a),
                              std::complex<double>(a, 0.25)}) {
            std::complex<double> c, s;
            disper26_math::compute_cosh_sinh(z, c, s);
            require(same(c, ad_cosh(z)), "cosh value/classification/zero sign");
            require(same(s, ad_sinh(z)), "sinh value/classification/zero sign");
        }
    }
}

void check_ad() {
    using AD = Eigen::AutoDiffScalar<Eigen::Vector2d>;
    AD x(0.25), y(-0.125);
    x.derivatives() = Eigen::Vector2d(1.0, 0.0);
    y.derivatives() = Eigen::Vector2d(0.0, 1.0);
    const std::complex<AD> z(x, y);
    std::complex<AD> c, s;
    disper26_math::compute_cosh_sinh(z, c, s);
    const auto expected_c = ad_cosh(z);
    const auto expected_s = ad_sinh(z);
    for (const auto& pair : {std::make_pair(c.real(), expected_c.real()),
                            std::make_pair(c.imag(), expected_c.imag()),
                            std::make_pair(s.real(), expected_s.real()),
                            std::make_pair(s.imag(), expected_s.imag())}) {
        require(same(pair.first.value(), pair.second.value()), "AD value");
        require(pair.first.derivatives() == pair.second.derivatives(), "AD derivatives");
    }
}

void check_custom_ad_lookup() {
    // HessianAD and its overloads are declared after the backend header. This
    // catches accidental loss of ADL when moving the wrappers to a namespace.
    HessianAD<2> z(0.25, -0.125), c, s;
    z.is_const = false;
    z.gr(0) = 1.0;
    z.gi(1) = 1.0;
    disper26_math::compute_cosh_sinh(z, c, s);
    const auto expected_c = ad_cosh(z);
    const auto expected_s = ad_sinh(z);
    for (const auto& pair : {std::make_pair(c, expected_c), std::make_pair(s, expected_s)}) {
        require(same(pair.first.r, pair.second.r) && same(pair.first.i, pair.second.i),
                "HessianAD value");
        require(pair.first.gr == pair.second.gr && pair.first.gi == pair.second.gi,
                "HessianAD gradient");
        require(pair.first.hr() == pair.second.hr() && pair.first.hi() == pair.second.hi(),
                "HessianAD Hessian");
    }
}

void check_layer_dispatch() {
    PhysicalParameters<double> p;
    p.rho = 2.7;
    p.c11 = p.c33 = p.rho * 6.0 * 6.0;
    p.c44 = p.c66 = p.rho * 3.5 * 3.5;
    p.c13 = p.c11 - 2.0 * p.c44;
    const std::vector<PhysicalParameters<double>> layers(3, p);
    const std::vector<double> thicknesses{1.0, 2.0, 3.0};
#ifdef DISPER26_HYPERBOLIC_BACKEND_HEADER
    disper26_math::probe_calls = 0;
#endif
    // Outside the small-q entire-function branch, layer propagation still
    // delegates to the replaceable cosh/sinh backend.
    const auto rayleigh = calculateRayleighDispersion(layers, thicknesses, p, 10.0, 3.2);
    require(rayleigh.allFinite(), "Rayleigh finite state");
#ifdef DISPER26_HYPERBOLIC_BACKEND_HEADER
    require(disper26_math::probe_calls == 8,
            "custom backend reaches Rayleigh halfspace and three finite layers");
    disper26_math::probe_calls = 0;
#endif
    const auto love = calculateLoveDispersion(layers, thicknesses, p, 10.0, 3.2);
    require(love.allFinite(), "Love finite state");
#ifdef DISPER26_HYPERBOLIC_BACKEND_HEADER
    require(disper26_math::probe_calls == 3, "custom backend reaches Love layers");
#endif
}
}  // namespace

int main() {
    check_values();
    check_ad();
    check_custom_ad_lookup();
    check_layer_dispatch();
    std::cout << "Dispersion math backend checks passed\n";
}
