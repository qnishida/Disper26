#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#if !DISPER26_EXPERIMENT_SHARED_CS
#error "this test requires the shared hyperbolic experiment"
#endif

#include "../src/disper26.cpp"

namespace {

bool same_component(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (a != b) return false;
    return a != 0.0 || std::signbit(a) == std::signbit(b);
}

bool close_component(double got, long double ref) {
    const long double tolerance = 8.0L * std::numeric_limits<double>::epsilon()
                                * std::max(1.0L, std::abs(ref));
    return std::abs(static_cast<long double>(got) - ref) <= tolerance;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void check_real_axis(double x, double y_zero) {
    std::complex<double> c, s;
    ad_cosh_sinh_paired({x, y_zero}, c, s);
    const long double lx = x;
    require(close_component(c.real(), std::cosh(lx)), "real-axis cosh");
    require(close_component(s.real(), std::sinh(lx)), "real-axis sinh");
    if (std::abs(x) <= std::numeric_limits<double>::min()) {
        require(same_component(s.real(), x), "tiny real-axis sinh exactness");
    } else if (std::abs(x) <= 1e-12) {
        const long double ref = std::sinh(lx);
        const long double relative_error =
            std::abs((static_cast<long double>(s.real()) - ref) / ref);
        require(relative_error <= 8.0L * std::numeric_limits<double>::epsilon(),
                "small real-axis sinh relative accuracy");
    }
    require(same_component(c.imag(), std::sinh(x) * y_zero),
            "real-axis cosh signed zero");
    require(same_component(s.imag(), std::cosh(x) * y_zero),
            "real-axis sinh signed zero");
}

void check_imaginary_axis(double x_zero, double y) {
    std::complex<double> c, s;
    ad_cosh_sinh_paired({x_zero, y}, c, s);
    const long double ly = y;
    require(close_component(c.real(), std::cos(ly)), "imaginary-axis cos");
    require(close_component(s.imag(), std::sin(ly)), "imaginary-axis sin");
    require(same_component(c.imag(), std::sin(y) * x_zero),
            "imaginary-axis cosh signed zero");
    require(same_component(s.real(), std::cos(y) * x_zero),
            "imaginary-axis sinh signed zero");
}

void check_fallback(const std::complex<double>& z) {
    std::complex<double> c, s;
    ad_cosh_sinh_paired(z, c, s);
    const auto expected_c = ad_cosh(z);
    const auto expected_s = ad_sinh(z);
    require(same_component(c.real(), expected_c.real()) &&
            same_component(c.imag(), expected_c.imag()), "cosh fallback");
    require(same_component(s.real(), expected_s.real()) &&
            same_component(s.imag(), expected_s.imag()), "sinh fallback");
}

}  // namespace

int main() {
    static_assert(std::numeric_limits<long double>::digits >
                  std::numeric_limits<double>::digits,
                  "references require precision above binary64");

    const double denorm = std::nextafter(0.0, 1.0);
    const double min_normal = std::numeric_limits<double>::min();
    const double below_limit = std::nextafter(20.0, 0.0);
    for (double x : {0.0, -0.0, denorm, -denorm, min_normal, -min_normal,
                     1e-12, -1e-12,
                     0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 10.0, -10.0,
                     below_limit, -below_limit, 20.0, -20.0}) {
        check_real_axis(x, 0.0);
        check_real_axis(x, -0.0);
    }

    for (double y : {0.0, -0.0, denorm, -denorm, 0.25, -0.25,
                     1.0, -1.0, 20.0, -20.0, 1000.0, -1000.0}) {
        check_imaginary_axis(0.0, y);
        check_imaginary_axis(-0.0, y);
    }

    check_fallback({std::nextafter(20.0, 21.0), -0.0});
    check_fallback({-std::nextafter(20.0, 21.0), 0.0});
    check_fallback({710.0, -0.0});
    check_fallback({-710.0, 0.0});
    check_fallback({0.125, -0.25});
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto& z : {std::complex<double>(inf, 0.0),
                          std::complex<double>(-inf, -0.0),
                          std::complex<double>(0.0, inf),
                          std::complex<double>(nan, 0.0),
                          std::complex<double>(0.0, nan),
                          std::complex<double>(nan, 0.25),
                          std::complex<double>(0.25, nan)}) {
        check_fallback(z);
    }

    std::cout << "Shared hyperbolic tests passed (Taylor="
              << DISPER26_EXPERIMENT_TAYLOR_CS << ", TaylorPaired="
              << DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED << ")\n";
    return 0;
}
