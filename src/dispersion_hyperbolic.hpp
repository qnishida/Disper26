#pragma once

#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>

// Existing scalar wrappers and disabled experiments. Keep their arithmetic,
// signed-zero behavior, and custom-AD overload lookup unchanged.
// Math wrappers that delegate to std library for std::complex<T>,
// but can be overloaded for custom AD types (e.g., HessianAD).
template<typename T> std::complex<T> ad_cosh(const std::complex<T>& z) { return std::cosh(z); }
template<typename T> std::complex<T> ad_sinh(const std::complex<T>& z) { return std::sinh(z); }

// Degenerate-argument shortcuts for the plain-double layer propagators.
//
// With real elastic parameters and a real phase velocity, every complex
// quantity in the layer kernels sits exactly on one axis: nu1, nu3 come out of
// a real square root, so nu*z is either purely real (evanescent) or purely
// imaginary (oscillatory). std::cosh/std::sinh on a complex argument cannot
// know that and always evaluate the full cosh(x)cos(y) + i sinh(x)sin(y) form,
// paying for a sin/cos pair that multiplies a zero.
//
// By default these overloads are exact case splits, and they fall
// through to the general routine whenever the argument really is off-axis
// (strongly anisotropic VTI, where the Christoffel discriminant goes negative).
// toComplex() builds imaginary parts as val*0.0, so the == 0.0 tests below hold
// exactly on the axis-aligned inputs they are meant to catch.
#ifndef DISPER26_EXPERIMENT_TAYLOR_CS
#define DISPER26_EXPERIMENT_TAYLOR_CS 0
#endif

#ifndef DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED
#define DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED 0
#endif

#ifndef DISPER26_EXPERIMENT_SHARED_CS
#define DISPER26_EXPERIMENT_SHARED_CS 0
#endif

#if DISPER26_EXPERIMENT_TAYLOR_CS
// This experiment changes only the plain-double overloads; generic and custom
// AD overloads continue to use their existing transcendental implementations.
struct TaylorCS {
    double c;
    double s;
};

inline TaylorCS dispersion_taylor_cs(double zz) {
    // Degree seven leaves the omitted terms below binary64 roundoff for |w| <= 0.5.
    const double c = (((((((1.0 / 87178291200.0 * zz
        + 1.0 / 479001600.0) * zz + 1.0 / 3628800.0) * zz
        + 1.0 / 40320.0) * zz + 1.0 / 720.0) * zz
        + 1.0 / 24.0) * zz + 1.0 / 2.0) * zz + 1.0);
    const double s = (((((((1.0 / 1307674368000.0 * zz
        + 1.0 / 6227020800.0) * zz + 1.0 / 39916800.0) * zz
        + 1.0 / 362880.0) * zz + 1.0 / 5040.0) * zz
        + 1.0 / 120.0) * zz + 1.0 / 6.0) * zz + 1.0);
    return {c, s};
}
#endif

#ifndef DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
#define DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC 0
#endif

#if DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
struct TaylorCSDiagnosticCounts {
    unsigned long long total = 0;
    unsigned long long axis = 0;
    unsigned long long applicable = 0;
    unsigned long long axis_out_of_range = 0;
    unsigned long long off_axis = 0;
    unsigned long long nonfinite = 0;
};

inline TaylorCSDiagnosticCounts dispersion_taylor_cs_diagnostic_counts;

inline void record_dispersion_taylor_cs_argument(double x, double y) {
    auto& counts = dispersion_taylor_cs_diagnostic_counts;
    ++counts.total;
    if (!std::isfinite(x) || !std::isfinite(y)) {
        ++counts.nonfinite;
    } else if (y == 0.0 || x == 0.0) {
        ++counts.axis;
        const double active = y == 0.0 ? x : y;
        if (std::abs(active) <= 0.5) ++counts.applicable;
        else ++counts.axis_out_of_range;
    } else {
        ++counts.off_axis;
    }
}

struct TaylorCSDiagnosticReporter {
    ~TaylorCSDiagnosticReporter() {
        const auto& c = dispersion_taylor_cs_diagnostic_counts;
        std::cerr << "DISPER26_TAYLOR_CS_DIAGNOSTIC total=" << c.total
                  << " axis=" << c.axis << " applicable=" << c.applicable
                  << " axis_out_of_range=" << c.axis_out_of_range
                  << " off_axis=" << c.off_axis << " nonfinite=" << c.nonfinite
                  << '\n';
    }
};

inline TaylorCSDiagnosticReporter dispersion_taylor_cs_diagnostic_reporter;
#endif

inline std::complex<double> ad_cosh(const std::complex<double>& z) {
    const double x = z.real();
    const double y = z.imag();
#if DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
    record_dispersion_taylor_cs_argument(x, y);
#endif
#if DISPER26_EXPERIMENT_TAYLOR_CS
    if (std::isfinite(x) && std::isfinite(y)) {
        if (y == 0.0 && std::abs(x) <= 0.5) {
            const TaylorCS cs = dispersion_taylor_cs(x * x);
            return std::complex<double>(cs.c, x * cs.s * y);
        }
        if (x == 0.0 && std::abs(y) <= 0.5) {
            const TaylorCS cs = dispersion_taylor_cs(-(y * y));
            return std::complex<double>(cs.c, y * cs.s * x);
        }
    }
#endif
    if (y == 0.0) return std::complex<double>(std::cosh(x), std::sinh(x) * y);
    if (x == 0.0) return std::complex<double>(std::cos(y), std::sin(y) * x);
    return std::cosh(z);
}

inline std::complex<double> ad_sinh(const std::complex<double>& z) {
    const double x = z.real();
    const double y = z.imag();
#if DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
    record_dispersion_taylor_cs_argument(x, y);
#endif
#if DISPER26_EXPERIMENT_TAYLOR_CS
    if (std::isfinite(x) && std::isfinite(y)) {
        if (y == 0.0 && std::abs(x) <= 0.5) {
            const TaylorCS cs = dispersion_taylor_cs(x * x);
            return std::complex<double>(x * cs.s, cs.c * y);
        }
        if (x == 0.0 && std::abs(y) <= 0.5) {
            const TaylorCS cs = dispersion_taylor_cs(-(y * y));
            return std::complex<double>(cs.c * x, y * cs.s);
        }
    }
#endif
    if (y == 0.0) return std::complex<double>(std::sinh(x), std::cosh(x) * y);
    if (x == 0.0) return std::complex<double>(std::cos(y) * x, std::sin(y));
    return std::sinh(z);
}

inline void ad_cosh_sinh_paired(
    const std::complex<double>& z,
    std::complex<double>& c,
    std::complex<double>& s
) {
#if DISPER26_EXPERIMENT_TAYLOR_CS && (DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED || DISPER26_EXPERIMENT_SHARED_CS)
    const double x = z.real();
    const double y = z.imag();
    if (std::isfinite(x) && std::isfinite(y)) {
        if (y == 0.0 && std::abs(x) <= 0.5) {
            const TaylorCS cs = dispersion_taylor_cs(x * x);
            c = std::complex<double>(cs.c, x * cs.s * y);
            s = std::complex<double>(x * cs.s, cs.c * y);
#if DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
            record_dispersion_taylor_cs_argument(x, y);
            record_dispersion_taylor_cs_argument(x, y);
#endif
            return;
        }
        if (x == 0.0 && std::abs(y) <= 0.5) {
            const TaylorCS cs = dispersion_taylor_cs(-(y * y));
            c = std::complex<double>(cs.c, y * cs.s * x);
            s = std::complex<double>(cs.c * x, y * cs.s);
#if DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
            record_dispersion_taylor_cs_argument(x, y);
            record_dispersion_taylor_cs_argument(x, y);
#endif
            return;
        }
    }
#endif
#if DISPER26_EXPERIMENT_SHARED_CS
    const double shared_x = z.real();
    const double shared_y = z.imag();
    if (std::isfinite(shared_x) && std::isfinite(shared_y)) {
        if (shared_y == 0.0 && std::abs(shared_x) <= 20.0) {
            // expm1 keeps the subtraction accurate near zero. With
            // t = exp(|x|) - 1, these are algebraically cosh(x) and
            // |sinh(x)|, evaluated from one transcendental call. The ordered
            // products also avoid an intermediate t*t overflow.
            const double t = std::expm1(std::abs(shared_x));
            const double inv_exp = 1.0 / (1.0 + t);
            const double cosh_x = 1.0 + 0.5 * t * (t * inv_exp);
            const double sinh_abs = t * (0.5 * ((2.0 + t) * inv_exp));
            const double sinh_x = std::copysign(sinh_abs, shared_x);
            c = std::complex<double>(cosh_x, sinh_x * shared_y);
            s = std::complex<double>(sinh_x, cosh_x * shared_y);
#if DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
            record_dispersion_taylor_cs_argument(shared_x, shared_y);
            record_dispersion_taylor_cs_argument(shared_x, shared_y);
#endif
            return;
        }
        if (shared_x == 0.0) {
            // GCC combines this adjacent pair into the platform sincos entry
            // point where available.
            const double cos_y = std::cos(shared_y);
            const double sin_y = std::sin(shared_y);
            c = std::complex<double>(cos_y, sin_y * shared_x);
            s = std::complex<double>(cos_y * shared_x, sin_y);
#if DISPER26_EXPERIMENT_TAYLOR_CS_DIAGNOSTIC
            record_dispersion_taylor_cs_argument(shared_x, shared_y);
            record_dispersion_taylor_cs_argument(shared_x, shared_y);
#endif
            return;
        }
    }
#endif
    // Keep every non-applicable case on the existing wrappers. They retain the
    // established off-axis/nonfinite behavior and own diagnostic accounting.
    c = ad_cosh(z);
    s = ad_sinh(z);
}

namespace disper26_math {

// A concrete, statically selected implementation: no per-layer dispatch or
// derivative work is added to the plain-double path. Unqualified calls retain
// ADL for ADComplex/HessianAD overloads defined by sensitivity_utils.hpp.
template<typename Complex>
struct DefaultHyperbolicBackend {
    static inline void compute_cosh_sinh(const Complex& z, Complex& c, Complex& s) {
#if (DISPER26_EXPERIMENT_TAYLOR_CS && DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED) || DISPER26_EXPERIMENT_SHARED_CS
        if constexpr (std::is_same_v<Complex, std::complex<double>>) {
            ad_cosh_sinh_paired(z, c, s);
        } else
#endif
        {
            c = ad_cosh(z);
            s = ad_sinh(z);
        }
    }
};

// Specialize only the supported complex type in a build-time backend header.
// Other scalar/AD types continue to use the existing implementation.
template<typename Complex>
struct HyperbolicBackend : DefaultHyperbolicBackend<Complex> {};

}  // namespace disper26_math

// The same definition must be used throughout one binary. The header is
// included before layer templates instantiate HyperbolicBackend.
#ifdef DISPER26_HYPERBOLIC_BACKEND_HEADER
#include DISPER26_HYPERBOLIC_BACKEND_HEADER
#endif

namespace disper26_math {

template<typename Complex>
inline void compute_cosh_sinh(const Complex& z, Complex& c, Complex& s) {
    HyperbolicBackend<Complex>::compute_cosh_sinh(z, c, s);
}

}  // namespace disper26_math
