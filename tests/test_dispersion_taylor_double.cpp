#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <string>

#if !DISPER26_EXPERIMENT_TAYLOR_CS || !DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED
#error "this test requires both Taylor experiment macros"
#endif

#include "../src/disper26.cpp"

namespace {

std::pair<double, double> high_precision_cs(double a, bool imaginary_axis) {
    const long double h = a;
    const long double z = imaginary_axis ? -(h * h) : h * h;
    long double c = 1;
    long double s = 1;
    long double c_term = 1;
    long double s_term = 1;
    for (unsigned n = 1; n < 100; ++n) {
        c_term *= z / static_cast<long double>((2 * n - 1) * (2 * n));
        s_term *= z / static_cast<long double>((2 * n) * (2 * n + 1));
        c += c_term;
        s += s_term;
        if (std::abs(c_term) < 1e-30L && std::abs(s_term) < 1e-30L) break;
    }
    return {static_cast<double>(c), static_cast<double>(s)};
}

bool close_component(double got, double ref) {
    const double tolerance = 8.0 * std::numeric_limits<double>::epsilon()
                           * std::max(1.0, std::abs(ref));
    return std::abs(got - ref) <= tolerance;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void check_axis(double a, bool imaginary_axis) {
    const auto ref = high_precision_cs(a, imaginary_axis);
    const std::complex<double> z = imaginary_axis
        ? std::complex<double>(std::copysign(0.0, a), a)
        : std::complex<double>(a, std::copysign(0.0, -a));
    const auto c = ad_cosh(z);
    const auto s = ad_sinh(z);
    if (imaginary_axis) {
        require(close_component(c.real(), ref.first), "imaginary-axis cosh real");
        require(close_component(s.imag(), a * ref.second), "imaginary-axis sinh imag");
    } else {
        require(close_component(c.real(), ref.first), "real-axis cosh real");
        require(close_component(s.real(), a * ref.second), "real-axis sinh real");
    }
}

bool same_component(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (a != b) return false;
    return a != 0.0 || std::signbit(a) == std::signbit(b);
}

bool close_with_zero_sign(double got, double ref) {
    return ref == 0.0 ? same_component(got, ref) : close_component(got, ref);
}

void check_pair_axis(const std::complex<double>& z) {
    const double x = z.real();
    const double y = z.imag();
    const bool imaginary_axis = x == 0.0 && y != 0.0;
    const auto cs = high_precision_cs(imaginary_axis ? y : x, imaginary_axis);
    const auto expected_c = imaginary_axis
        ? std::complex<double>(cs.first, y * cs.second * x)
        : std::complex<double>(cs.first, x * cs.second * y);
    const auto expected_s = imaginary_axis
        ? std::complex<double>(cs.first * x, y * cs.second)
        : std::complex<double>(x * cs.second, cs.first * y);
    std::complex<double> got_c, got_s;
    ad_cosh_sinh_paired(z, got_c, got_s);
    require(close_with_zero_sign(got_c.real(), expected_c.real()) &&
            close_with_zero_sign(got_c.imag(), expected_c.imag()),
            "paired cosh high-precision reference");
    require(close_with_zero_sign(got_s.real(), expected_s.real()) &&
            close_with_zero_sign(got_s.imag(), expected_s.imag()),
            "paired sinh high-precision reference");
    const auto separate_c = ad_cosh(z);
    const auto separate_s = ad_sinh(z);
    require(same_component(got_c.real(), separate_c.real()) &&
            same_component(got_c.imag(), separate_c.imag()),
            "paired cosh differs from separate wrapper");
    require(same_component(got_s.real(), separate_s.real()) &&
            same_component(got_s.imag(), separate_s.imag()),
            "paired sinh differs from separate wrapper");
}

void check_fallback(const std::complex<double>& z) {
    const double x = z.real();
    const double y = z.imag();
    const auto expected_c = y == 0.0
        ? std::complex<double>(std::cosh(x), std::sinh(x) * y)
        : x == 0.0 ? std::complex<double>(std::cos(y), std::sin(y) * x)
                   : std::cosh(z);
    const auto expected_s = y == 0.0
        ? std::complex<double>(std::sinh(x), std::cosh(x) * y)
        : x == 0.0 ? std::complex<double>(std::cos(y) * x, std::sin(y))
                   : std::sinh(z);
    const auto got_c = ad_cosh(z);
    const auto got_s = ad_sinh(z);
    require(same_component(got_c.real(), expected_c.real()) &&
            same_component(got_c.imag(), expected_c.imag()), "cosh fallback changed");
    require(same_component(got_s.real(), expected_s.real()) &&
            same_component(got_s.imag(), expected_s.imag()), "sinh fallback changed");
    std::complex<double> paired_c, paired_s;
    ad_cosh_sinh_paired(z, paired_c, paired_s);
    require(same_component(paired_c.real(), got_c.real()) &&
            same_component(paired_c.imag(), got_c.imag()), "paired cosh fallback changed");
    require(same_component(paired_s.real(), got_s.real()) &&
            same_component(paired_s.imag(), got_s.imag()), "paired sinh fallback changed");
}

void check_signed_axis(const std::complex<double>& z) {
    const double x = z.real();
    const double y = z.imag();
    const auto cs = high_precision_cs(y == 0.0 ? x : y, x == 0.0 && y != 0.0);
    const auto expected_c = y == 0.0
        ? std::complex<double>(cs.first, x * cs.second * y)
        : std::complex<double>(cs.first, y * cs.second * x);
    const auto expected_s = y == 0.0
        ? std::complex<double>(x * cs.second, cs.first * y)
        : std::complex<double>(cs.first * x, y * cs.second);
    const auto got_c = ad_cosh(z);
    const auto got_s = ad_sinh(z);
    require(same_component(got_c.real(), expected_c.real()) &&
            same_component(got_c.imag(), expected_c.imag()), "cosh signed axis");
    require(same_component(got_s.real(), expected_s.real()) &&
            same_component(got_s.imag(), expected_s.imag()), "sinh signed axis");
}

}  // namespace

int main() {
    static_assert(std::numeric_limits<long double>::digits >
                  std::numeric_limits<double>::digits,
                  "reference series requires precision above binary64");
    // The component bound is eight binary64 epsilons times max(1, |reference|).
    for (int i = 0; i <= 100; ++i) {
        const double a = -0.5 + i * 0.01;
        check_axis(a, false);
        check_axis(a, true);
        check_pair_axis({a, std::copysign(0.0, -a)});
        check_pair_axis({std::copysign(0.0, a), a});
    }

    const double near_zero = std::nextafter(0.0, 1.0);
    for (double a : {0.0, -0.0, near_zero, -near_zero}) {
        check_axis(a, false);
        check_axis(a, true);
        check_pair_axis({a, std::copysign(0.0, -a)});
        check_pair_axis({std::copysign(0.0, a), a});
    }

    const double inside = std::nextafter(0.5, 0.0);
    const double outside = std::nextafter(0.5, 1.0);
    for (double a : {inside, -inside, 0.5, -0.5}) {
        check_axis(a, false);
        check_axis(a, true);
        check_pair_axis({a, -0.0});
        check_pair_axis({-0.0, a});
    }
    for (double a : {outside, -outside}) {
        check_fallback({a, -0.0});
        check_fallback({-0.0, a});
    }

    for (double x : {0.0, -0.0}) {
        for (double y : {0.0, -0.0}) {
            check_signed_axis({x, y});
            check_pair_axis({x, y});
        }
    }
    for (double active : {0.25, -0.25}) {
        for (double zero : {0.0, -0.0}) {
            check_signed_axis({active, zero});
            check_signed_axis({zero, active});
            check_pair_axis({active, zero});
            check_pair_axis({zero, active});
        }
    }

    check_fallback({0.125, -0.25});
    check_fallback({20.0, -0.0});
    check_fallback({-0.0, -20.0});
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto& z : {std::complex<double>(inf, 0.0),
                          std::complex<double>(0.0, inf),
                          std::complex<double>(nan, 0.0),
                          std::complex<double>(0.0, nan),
                          std::complex<double>(nan, 0.25),
                          std::complex<double>(0.25, nan)}) {
        check_fallback(z);
    }

    std::cout << "Taylor complex-double wrapper tests passed (CS=1, PAIRED=1)\n";
    return 0;
}
