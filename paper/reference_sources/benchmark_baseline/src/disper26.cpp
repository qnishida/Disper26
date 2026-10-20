#ifndef DISPER26_CPP
#define DISPER26_CPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include <iomanip>

// autodiff library integration (for forward and reverse mode AD)
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "eval_counters.hpp"

// --- AD Compatibility Layer ---
// ComplexType<T>::type maps to std::complex<T> for standard types,
// but can be specialized (e.g., HessianAD packs real+imag internally).
template<typename T> struct ComplexType { using type = std::complex<T>; };

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
// These overloads are exact case splits, not approximations, and they fall
// through to the general routine whenever the argument really is off-axis
// (strongly anisotropic VTI, where the Christoffel discriminant goes negative).
// toComplex() builds imaginary parts as val*0.0, so the == 0.0 tests below hold
// exactly on the axis-aligned inputs they are meant to catch.
inline std::complex<double> ad_cosh(const std::complex<double>& z) {
    const double x = z.real();
    const double y = z.imag();
    if (y == 0.0) return std::complex<double>(std::cosh(x), std::sinh(x) * y);
    if (x == 0.0) return std::complex<double>(std::cos(y), std::sin(y) * x);
    return std::cosh(z);
}

inline std::complex<double> ad_sinh(const std::complex<double>& z) {
    const double x = z.real();
    const double y = z.imag();
    if (y == 0.0) return std::complex<double>(std::sinh(x), std::cosh(x) * y);
    if (x == 0.0) return std::complex<double>(std::cos(y) * x, std::sin(y));
    return std::sinh(z);
}
template<typename Z>
auto ad_exp(const Z& z) {
    using std::exp;
    return exp(z);
}

// --- Type Definitions ---
enum class WaveType {
    Rayleigh,
    Love
};

// --- Normalization Modes for Dispersion Calculation ---
enum class NormalizationMode {
    None,            // Current behavior (Stop Gradient Max-Norm) - Default
    Vertical,        // Normalize by Y(0) (Vertical) - Differentiable
    VectorNorm,      // Normalize by Sqrt(Sum(Y^2)) - Differentiable
    Provided         // Use pre-computed double scale factors (for Hessian AD)
};

// Forward declarations for scalar extraction (defined below)
inline double get_scalar_val(double x);
template <typename DerType> inline double get_scalar_val(const Eigen::AutoDiffScalar<DerType>& x);
inline double get_scalar_val(const autodiff::dual& x);
inline double get_scalar_val(const autodiff::dual2nd& x);
inline double get_scalar_val(const autodiff::var& x);

template <typename T>
struct PhysicalParameters {
    T rho;
    T c11, c13, c33, c44, c66;
    // Constant fluid-layer background gravity in km/s^2. Zero preserves the
    // acoustic branch.
    //
    // Note the explicit {0.0} rather than {}. For an AD scalar T these are not
    // the same: Eigen's AutoDiffScalar declares a user-provided empty default
    // constructor (AutoDiffScalar.h:84), so value-initialization with {} calls
    // it and leaves the value uninitialized, whereas {0.0} reaches the scalar
    // constructor, which sets the value and zeroes the derivatives.
    //
    // This matters because gravity is read only inside the fluid branches --
    // computeLayerMatrixLiquid picks an entirely different 2x2 propagator when
    // it is nonzero, and there are seabed and free-surface corrections keyed
    // the same way. Every conversion of PhysicalParameters into an AD type in
    // this codebase copies the six moduli and leaves these four alone, so with
    // {} the AD group velocity of an ocean model was reading whatever was on
    // the stack: the same inputs returned 1.345, 2.080 or NaN depending on
    // what had been called before. Solid models never look at these fields,
    // which is why only fluid models were affected.
    T gravity{0.0};
    T gravity_internal_scale{1.0};
    T gravity_surface_scale{0.0};
    T gravity_seabed_scale{0.0};
    double vs() const { return std::sqrt(get_scalar_val(c44) / get_scalar_val(rho)); }
    double vp() const { return std::sqrt(get_scalar_val(c11) / get_scalar_val(rho)); }
};

// Helper: Create Isotropic Parameters from Vs, Vp, Rho
template <typename T>
PhysicalParameters<T> createIsotropicParameters(T vs, T vp, T rho) {
    PhysicalParameters<T> p;
    p.rho = rho;
    p.c44 = rho * vs * vs;
    p.c33 = rho * vp * vp;
    p.c11 = p.c33; // Isotropic: c11=c33
    p.c13 = p.c33 - 2.0 * p.c44;
    p.c66 = p.c44;
    p.gravity = T(0.0);
    p.gravity_internal_scale = T(1.0);
    p.gravity_surface_scale = T(0.0);
    p.gravity_seabed_scale = T(0.0);
    return p;
}

// Helper: Make NaN for generic type T (double or AD)
template<typename T>
T make_nan() {
    if constexpr (std::is_same_v<T, double>) {
        return std::numeric_limits<double>::quiet_NaN();
    } else {
        // Assume AD type, construct from value
        return T(std::numeric_limits<double>::quiet_NaN());
    }
}



template <typename T>
struct WaveVariables {
    T omega;
    T k; // wavenumber = omega / phase_velocity
    T z; // thickness of the layer
};

template <typename T>
struct DerivedComplexValues {
    typename ComplexType<T>::type d1, d2, d3, d4, d5, d6;
    typename ComplexType<T>::type C1, C3, S1, S3;
    typename ComplexType<T>::type eps1, eps3;
    typename ComplexType<T>::type nu1, nu3;
    typename ComplexType<T>::type nu1_sq, nu3_sq;
    bool nu1_sq_from_vieta;

    DerivedComplexValues() 
        : d1(), d2(), d3(), d4(), d5(), d6(),
          C1(), C3(), S1(), S3(),
          eps1(), eps3(),
          nu1(), nu3(), nu1_sq(), nu3_sq(), nu1_sq_from_vieta(false) {}
};

// Reduced indices for the 5x5 matrix
enum ReducedIndex { I12=0, I13=1, I14=2, I23=3, I24=4 };

// --- Helper Functions ---

// Helper to create a complex number with zero imaginary part but preserving derivative size
template<typename T>
typename ComplexType<T>::type toComplex(const T& val) {
    T zero = val * 0.0;
    return typename ComplexType<T>::type(val, zero);
}

// Helper to create a constant T with the same derivative size as reference, but value 0
template<typename T>
T make_zero_like(const T& ref) {
    return ref * 0.0;
}

// Helper to get magnitude (value) for scaling
// Recursive scalar extractor
inline double get_scalar_val(double x) { return x; }

template <typename DerType>
inline double get_scalar_val(const Eigen::AutoDiffScalar<DerType>& x) {
    return get_scalar_val(x.value());
}

// Overloads for autodiff library types
inline double get_scalar_val(const autodiff::dual& x) { return autodiff::val(x); }
inline double get_scalar_val(const autodiff::dual2nd& x) { return autodiff::val(x); }
inline double get_scalar_val(const autodiff::var& x) { return autodiff::val(x); }

inline double get_magnitude(double x) { return std::abs(x); }

template<typename DerType>
inline double get_magnitude(const Eigen::AutoDiffScalar<DerType>& x) {
    return std::abs(get_scalar_val(x));
}

// Autodiff library get_magnitude overloads
inline double get_magnitude(const autodiff::dual& x) { return std::abs(autodiff::val(x)); }
inline double get_magnitude(const autodiff::dual2nd& x) { return std::abs(autodiff::val(x)); }
inline double get_magnitude(const autodiff::var& x) { return std::abs(autodiff::val(x)); }

// Helper to get raw scalar value (preserving sign)
inline double get_val(double x) { return x; }

template<typename DerType>
inline double get_val(const Eigen::AutoDiffScalar<DerType>& x) {
    return get_scalar_val(x);
}

// Autodiff library get_val overloads
inline double get_val(const autodiff::dual& x) { return autodiff::val(x); }
inline double get_val(const autodiff::dual2nd& x) { return autodiff::val(x); }
inline double get_val(const autodiff::var& x) { return autodiff::val(x); }

// ===========================================================================
// Safe complex sqrt for autodiff compatibility
// Uses polar form to avoid std::complex internal ternary operators
// This enables backward mode (autodiff::var) to work with complex numbers
// ===========================================================================

// Default: use std::sqrt for types that support it (double, Eigen::AutoDiff, dual)
template<typename T>
std::complex<T> safe_complex_sqrt(const std::complex<T>& z) {
    return std::sqrt(z);
}

// Axis-aligned shortcut for plain double, same reasoning as ad_cosh/ad_sinh
// above: the layer kernels call this three times per Rayleigh layer (the
// Christoffel discriminant, then nu1 and nu3) and in every isotropic and
// mildly anisotropic case the argument is exactly real. libm's csqrt has to
// handle the general branch cut; a real argument needs one sqrt. Signed zeros
// are propagated so the result matches csqrt bit for bit.
template<>
inline std::complex<double> safe_complex_sqrt(const std::complex<double>& z) {
    const double y = z.imag();
    if (y == 0.0) {
        const double x = z.real();
        if (x > 0.0) return std::complex<double>(std::sqrt(x), y);
        // x <= 0: sqrt(x + -+0i) = 0 -+ sqrt(-x) i  (covers x == -0.0 too)
        const double r = std::sqrt(-x);
        return std::complex<double>(0.0, std::signbit(y) ? -r : r);
    }
    return std::sqrt(z);
}

// Specialization for autodiff::var - use polar form to avoid ternary operators
template<>
inline std::complex<autodiff::var> safe_complex_sqrt(const std::complex<autodiff::var>& z) {
    using avar = autodiff::var;
    avar x = z.real();
    avar y = z.imag();
    
    // Handle zero case - use get_scalar_val which works with var type
    double x_val = get_scalar_val(x);
    double y_val = get_scalar_val(y);
    if (std::abs(x_val) < 1e-30 && std::abs(y_val) < 1e-30) {
        return std::complex<avar>(avar(0.0), avar(0.0));
    }
    
    // |z| = sqrt(x^2 + y^2)
    avar r = sqrt(x*x + y*y);
    
    // theta = atan2(y, x)
    avar theta = atan2(y, x);
    
    // sqrt(r) * (cos(theta/2) + i*sin(theta/2))
    avar half_theta = theta * 0.5;
    avar sqrt_r = sqrt(r);
    
    return std::complex<avar>(sqrt_r * cos(half_theta), sqrt_r * sin(half_theta));
}

// Specialization for Eigen::AutoDiffScalar - similar to var
template<typename DerType>
inline std::complex<Eigen::AutoDiffScalar<DerType>> safe_complex_sqrt(const std::complex<Eigen::AutoDiffScalar<DerType>>& z) {
    using ADS = Eigen::AutoDiffScalar<DerType>;
    ADS x = z.real();
    ADS y = z.imag();
    
    // Handle zero case
    if (x.value() == 0.0 && y.value() == 0.0) {
        ADS zero_res;
        zero_res.value() = 0.0;
        zero_res.derivatives() = x.derivatives();
        zero_res.derivatives().setZero();
        return std::complex<ADS>(zero_res, zero_res);
    }
    
    // |z| = sqrt(x^2 + y^2)
    ADS r = sqrt(x*x + y*y);
    
    // theta = atan2(y, x)
    // Eigen supports atan2 for AutoDiffScalar
    ADS theta = atan2(y, x);
    
    // sqrt(r) * (cos(theta/2) + i*sin(theta/2))
    ADS half_theta = theta * 0.5;
    ADS sqrt_r = sqrt(r);
    
    return std::complex<ADS>(sqrt_r * cos(half_theta), sqrt_r * sin(half_theta));
}



// --- Matrix Calculation Logic (from Ikeda_Matsuoka_complex.cpp) ---

template <typename T>
DerivedComplexValues<T> computeDerivedValues(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v
) {
    using CT = typename ComplexType<T>::type;
    
    T rho_omega2 = p.rho * v.omega * v.omega;
    T c44_k2     = p.c44 * v.k * v.k;
    T c11_k2     = p.c11 * v.k * v.k;
    
    T c13_plus_c44 = p.c13 + p.c44;
    T term_k_sq = v.k * v.k;

    T ref_zero = make_zero_like(p.rho); 
    T val_4 = ref_zero + 4.0;
    T val_2 = ref_zero + 2.0;

    CT M1 = toComplex<T>(
          p.c44 * (rho_omega2 - c44_k2) 
        + p.c33 * (rho_omega2 - c11_k2) 
        + c13_plus_c44 * c13_plus_c44 * term_k_sq
    );

    CT M2 = toComplex<T>(
        (rho_omega2 - c44_k2) * (rho_omega2 - c11_k2)
    );

    // nu calculation
    CT discriminant = M1 * M1 - toComplex<T>(val_4 * p.c33 * p.c44) * M2;
    CT sqrt_discriminant = safe_complex_sqrt(discriminant);

    CT numerator1 = -M1 + sqrt_discriminant;
    CT numerator3 = -M1 - sqrt_discriminant;
    CT denominator = toComplex<T>(val_2 * p.c33 * p.c44);

    // Recover only the strongly cancellation-prone P root.  The direct-value
    // guard keeps the established c≈Vs branch unchanged even when its other
    // root happens to be small in a different layer.
    const double numerator1_mag =
        std::abs(get_val(numerator1.real())) + std::abs(get_val(numerator1.imag()));
    const double numerator3_mag =
        std::abs(get_val(numerator3.real())) + std::abs(get_val(numerator3.imag()));
    const CT direct_nu1_sq = numerator1 / denominator;
    const double direct_nu1_mag = std::sqrt(
        std::abs(get_val(direct_nu1_sq.real())) +
        std::abs(get_val(direct_nu1_sq.imag())));
    CT nu1_sq, nu3_sq;
    bool nu1_sq_from_vieta = false;
    if (direct_nu1_mag < 1e-5 &&
        numerator1_mag < 1e-4 * numerator3_mag && numerator3_mag > 1e-300) {
        CT root_product = M2 / toComplex<T>(p.c33 * p.c44);
        nu3_sq = numerator3 / denominator;
        nu1_sq = root_product / nu3_sq;
        nu1_sq_from_vieta = true;
    } else {
        nu1_sq = direct_nu1_sq;
        nu3_sq = numerator3 / denominator;
    }


    CT nu1 = safe_complex_sqrt(nu1_sq);
    CT nu3 = safe_complex_sqrt(nu3_sq);

    // epsilon calculation with c=Vp singularity handling
    auto compute_epsilon = [&](CT nu, CT nu_sq) {
        CT num = toComplex<T>((p.c13 + p.c44) * v.k) * nu;
        CT den_real_part = toComplex<T>(c11_k2 - rho_omega2);
        CT den_complex_part = toComplex<T>(p.c44) * nu_sq;
        CT denominator = den_real_part - den_complex_part;
        
        // Check for near-singularity (c ≈ Vp)
        // Use relative threshold: if |denominator| < eps * |numerator|
        double denom_mag = std::abs(get_val(denominator.real())) + std::abs(get_val(denominator.imag()));
        double num_mag = std::abs(get_val(num.real())) + std::abs(get_val(num.imag()));
        
        const double SINGULARITY_THRESHOLD = 1e-10;
        
        if (denom_mag < SINGULARITY_THRESHOLD * (num_mag + 1.0)) {
            // At c ≈ Vp (P-wave velocity), use limiting form
            // For isotropic media: epsilon ≈ k / nu
            CT k_c = toComplex(v.k);
            double nu_mag = std::abs(get_val(nu.real())) + std::abs(get_val(nu.imag()));
            if (nu_mag > 1e-15) {
                return k_c / nu;
            } else {
                // Both nu and denominator near zero - use finite value
                return toComplex<T>(T(1.0));
            }
        }
        
        return num / denominator;
    };

    CT eps1 = compute_epsilon(nu1, nu1_sq);
    CT eps3 = compute_epsilon(nu3, nu3_sq);

    DerivedComplexValues<T> res;
    // For matrix B calculation, we use z as the layer thickness
    res.C1 = ad_cosh(nu1 * toComplex(v.z));
    res.S1 = ad_sinh(nu1 * toComplex(v.z));
    res.C3 = ad_cosh(nu3 * toComplex(v.z));
    res.S3 = ad_sinh(nu3 * toComplex(v.z));

    CT c33_c = toComplex(p.c33);
    CT c13_k_c = toComplex<T>(p.c13 * v.k);

    res.d1 = c33_c * nu1 - c13_k_c * eps1;
    res.d2 = c33_c * nu3 - c13_k_c * eps3;
    res.d3 = toComplex(v.k) + nu1 * eps1;
    res.d4 = toComplex(v.k) + nu3 * eps3;
    res.d5 = nu1 * eps1 - nu3 * eps3;
    res.d6 = nu3 * eps1 - nu1 * eps3;

    res.eps1 = eps1;
    res.eps3 = eps3;
    res.nu1 = nu1;
    res.nu3 = nu3;
    res.nu1_sq = nu1_sq;
    res.nu3_sq = nu3_sq;
    res.nu1_sq_from_vieta = nu1_sq_from_vieta;

    return res;
}

// Rayleigh state propagator evaluated without the P/SV modal eigenvectors.
// The state ordering is [uz, szz, ur, szr], and the signs match the existing
// computeLayerMatrixP_SV formulas (the derivative at z=0 is G below).  This
// representation is entire in the material parameters at nu1=0 and nu3=0;
// unlike the modal formula it never forms eps1, d1, or a difference of nearly
// identical modal exponentials.
template <typename T>
Eigen::Matrix<typename ComplexType<T>::type, 4, 4>
computeRayleighStatePropagatorRegular(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v
) {
    using CT = typename ComplexType<T>::type;
    using Matrix4 = Eigen::Matrix<CT, 4, 4>;

    auto scalar_constant = [&](double value) {
        T out = make_zero_like(p.rho);
        out = out + value;
        return out;
    };

    const CT zero = toComplex(make_zero_like(p.rho));
    const CT one = toComplex(scalar_constant(1.0));
    const CT c33 = toComplex(p.c33);
    const CT c44 = toComplex(p.c44);
    const CT c13 = toComplex(p.c13);
    const CT rho = toComplex(p.rho);
    const CT omega = toComplex(v.omega);
    const CT k = toComplex(v.k);

    Matrix4 G;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            G(i,j) = zero;

    G(0,1) = one / c33;
    G(0,2) = c13 * k / c33;
    G(1,0) = -rho * omega * omega;
    G(1,3) = k;
    G(2,0) = -k;
    G(2,3) = one / c44;
    G(3,1) = -c13 * k / c33;
    G(3,2) = (toComplex(p.c11) - c13*c13/c33) * k*k - rho*omega*omega;

    Matrix4 H;
    const CT z = toComplex(v.z);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            H(i,j) = G(i,j) * z;

    // Scale from primal values only.  With ||H||_inf <= 0.5, 24 Taylor terms
    // leave a remainder far below double precision; repeated squaring then
    // restores the requested thickness while preserving AD derivatives.
    double norm_inf = 0.0;
    for (int i = 0; i < 4; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < 4; ++j)
            row_sum += std::abs(get_val(H(i,j).real())) + std::abs(get_val(H(i,j).imag()));
        norm_inf = std::max(norm_inf, row_sum);
    }
    int squarings = 0;
    if (norm_inf > 0.5)
        squarings = static_cast<int>(std::ceil(std::log2(norm_inf / 0.5)));
    const double inv_scale = std::ldexp(1.0, -squarings);
    const CT inv_scale_c = toComplex(scalar_constant(inv_scale));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            H(i,j) = H(i,j) * inv_scale_c;

    auto multiply = [&](const Matrix4& A, const Matrix4& B) {
        Matrix4 C;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                CT acc = zero;
                for (int q = 0; q < 4; ++q) acc = acc + A(i,q) * B(q,j);
                C(i,j) = acc;
            }
        }
        return C;
    };

    Matrix4 result, term;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const CT identity = (i == j) ? one : zero;
            result(i,j) = identity;
            term(i,j) = identity;
        }
    }
    for (int n = 1; n <= 24; ++n) {
        term = multiply(term, H);
        const CT divisor = toComplex(scalar_constant(double(n)));
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                term(i,j) = term(i,j) / divisor;
                result(i,j) = result(i,j) + term(i,j);
            }
    }
    for (int n = 0; n < squarings; ++n) result = multiply(result, result);
    return result;
}

// Reserve the regular state propagator for the genuinely cancellation-prone
// near-Vp root.  The threshold is on |nu1| itself (not on z): this keeps the
// z=0 identity calls and the near-Vs path on their established modal route.
constexpr double RAYLEIGH_NU1_REGULAR_PROPAGATOR_THRESHOLD = 1e-8;

// Second compound of the regular 4x4 propagator, reduced from the six minors
// [12,13,14,23,24,34] with the Rayleigh invariant Y34=-Y12.  This reproduces
// the existing five-component Dunkin state without modal divisions.
template <typename T>
Eigen::Matrix<T, 5, 5> computeRayleighReducedPropagatorRegular(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v
) {
    using CT = typename ComplexType<T>::type;
    const auto P = computeRayleighStatePropagatorRegular(p, v);
    constexpr int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    CT C2[6][6];
    for (int r = 0; r < 6; ++r) {
        const int i = pairs[r][0], j = pairs[r][1];
        for (int c = 0; c < 6; ++c) {
            const int k = pairs[c][0], l = pairs[c][1];
            C2[r][c] = P(i,k)*P(j,l) - P(i,l)*P(j,k);
        }
    }

    Eigen::Matrix<T, 5, 5> B;
    for (int r = 0; r < 5; ++r) {
        B(r,0) = (C2[r][0] - C2[r][5]).real();
        for (int c = 1; c < 5; ++c) B(r,c) = C2[r][c].real();
    }
    return B;
}

template <typename T>
Eigen::Matrix<T, 5, 5> computeLayerMatrix(
    const PhysicalParameters<T>& physParams,
    const WaveVariables<T>& waveVars
) {
    using CT = typename ComplexType<T>::type;
    
    // For layer matrix, waveVars.z should be the layer thickness
    DerivedComplexValues<T> v = computeDerivedValues(physParams, waveVars);

    const double nu1_mag = std::abs(get_val(v.nu1.real())) + std::abs(get_val(v.nu1.imag()));
    if (nu1_mag < RAYLEIGH_NU1_REGULAR_PROPAGATOR_THRESHOLD) {
        return computeRayleighReducedPropagatorRegular(physParams, waveVars);
    }

    const CT& d1 = v.d1; const CT& d2 = v.d2;
    const CT& d3 = v.d3; const CT& d4 = v.d4;
    const CT& d5 = v.d5; const CT& d6 = v.d6;
    const CT& C1 = v.C1; const CT& C3 = v.C3;
    const CT& S1 = v.S1; const CT& S3 = v.S3;

    CT C1C3 = C1 * C3;
    CT S1S3 = S1 * S3;

    T ref_zero = make_zero_like(physParams.rho);
    T val_1 = ref_zero + 1.0;
    T val_2 = ref_zero + 2.0;

    CT one = toComplex<T>(val_1);
    CT two = toComplex<T>(val_2);

    // --- Matrix Calculation Logic (Ported from Ikeda_Matsuoka.cpp) ---
    // Mapping:
    // Ikeda_Matsuoka p.d1 -> d1 (Complex)
    // Ikeda_Matsuoka v.C1 -> C1 (Complex)
    
    // Define Matrix B
    Eigen::Matrix<T, 5, 5> B;
    B.setZero();

    // Common terms
    CT inv_d5 = one / d5;
    CT inv_d6 = one / d6;
    CT inv_c33 = one / toComplex(physParams.c33);
    CT inv_c44 = one / toComplex(physParams.c44);
    
    CT k_33_d5d6 = inv_c33 * inv_d5 * inv_d6;
    CT k_33_d6   = inv_c33 * inv_d6;
    CT k_44_d5   = inv_c44 * inv_d5;
    CT k_all     = inv_c33 * inv_c44 * inv_d5 * inv_d6;

    // --- Column 1 (I12) ---
    // b1234 = k_all * (2.0 * p.d1 * p.d2 * (C1C3 - 1.0) - (p.d1*p.d1 + p.d2*p.d2) * S1S3);
    CT b1234 = k_all * (two * d1 * d2 * (C1C3 - one) - (d1*d1 + d2*d2) * S1S3);
    
    // B(I12, I12) = 1.0 - 2.0 * b1234;
    B(I12, I12) = (one - two * b1234).real();

    // b1224 = k_all * ((p.eps1 * p.d2 + p.eps3 * p.d1)*(1.0 - C1C3) + (p.eps1*p.d1 + p.eps3*p.d2)*S1S3);
    CT b1224 = k_all * ((v.eps1 * d2 + v.eps3 * d1)*(one - C1C3) + (v.eps1 * d1 + v.eps3 * d2)*S1S3);
    // B(I13, I12) = 2.0 * b1224; (Note: Ikeda_Matsuoka used b1312=b1224)
    B(I13, I12) = (two * b1224).real();

    // b1223 = k_33_d6 * (p.d1 * v.C1 * v.S3 - p.d2 * v.C3 * v.S1);
    CT b1223 = k_33_d6 * (d1 * C1 * S3 - d2 * C3 * S1);
    // B(I14, I12) = 2.0 * b1223;
    B(I14, I12) = (two * b1223).real();

    // b1214 = k_44_d5 * (-p.d2 * v.C1 * v.S3 + p.d1 * v.C3 * v.S1);
    CT b1214 = k_44_d5 * (-d2 * C1 * S3 + d1 * C3 * S1);
    // B(I23, I12) = 2.0 * b1214;
    B(I23, I12) = (two * b1214).real();

    // b1213 = k_33_d5d6 * (p.d1*p.d2*(p.d3+p.d4)*(C1C3-1.0) - (p.d2*p.d2*p.d3+p.d1*p.d1*p.d4)*S1S3);
    CT b1213 = k_33_d5d6 * (d1*d2*(d3+d4)*(C1C3-one) - (d2*d2*d3 + d1*d1*d4)*S1S3);
    // B(I24, I12) = 2.0 * b1213;
    B(I24, I12) = (two * b1213).real();

    // --- Column 2 (I13) ---
    B(I12, I13) = b1213.real();
    B(I13, I13) = (C1C3 + b1234).real();
    
    // B(I14, I13) = (p.c44 * k_33_d6) * (-p.d1*p.d4*v.C1*v.S3 + p.d2*p.d3*v.C3*v.S1);
    B(I14, I13) = (toComplex(physParams.c44) * k_33_d6 * (-d1*d4*C1*S3 + d2*d3*C3*S1)).real();
    
    // B(I23, I13) = inv_d5 * (p.d2*p.d3*v.C1*v.S3 - p.d1*p.d4*v.C3*v.S1);
    B(I23, I13) = (inv_d5 * (d2*d3*C1*S3 - d1*d4*C3*S1)).real();
    
    // B(I24, I13) = (p.c44 * k_33_d5d6) * (2.0*p.d1*p.d2*p.d3*p.d4*(1.0-C1C3) + (p.d1*p.d1*p.d4*p.d4 + p.d2*p.d2*p.d3*p.d3)*S1S3);
    B(I24, I13) = (toComplex(physParams.c44) * k_33_d5d6 * (two*d1*d2*d3*d4*(one-C1C3) + (d1*d1*d4*d4 + d2*d2*d3*d3)*S1S3)).real();

    // --- Column 3 (I14) ---
    B(I12, I14) = b1214.real();
    
    // b1314 = k_44_d5 * (-p.eps3 * v.C1 * v.S3 + p.eps1 * v.C3 * v.S1);
    CT b1314 = k_44_d5 * (-v.eps3 * C1 * S3 + v.eps1 * C3 * S1);
    B(I13, I14) = b1314.real();
    
    B(I14, I14) = C1C3.real();
    
    // B(I23, I14) = (p.c33 * p.d6 * k_44_d5) * S1S3; 
    B(I23, I14) = (toComplex(physParams.c33) * d6 * k_44_d5 * S1S3).real();
    
    B(I24, I14) = B(I23, I13); // Symmetry

    // --- Column 4 (I23) ---
    B(I12, I23) = b1223.real();
    
    // b1323 = k_33_d6 * (p.eps1 * v.C1 * v.S3 - p.eps3 * v.C3 * v.S1);
    CT b1323 = k_33_d6 * (v.eps1 * C1 * S3 - v.eps3 * C3 * S1);
    B(I13, I23) = b1323.real();
    
    // B(I14, I23) = (p.c44 * p.d5 * k_33_d6) * S1S3;
    B(I14, I23) = (toComplex(physParams.c44) * d5 * k_33_d6 * S1S3).real();
    
    B(I23, I23) = B(I14, I14); // Symmetry
    B(I24, I23) = B(I14, I13); // Symmetry

    // --- Column 5 (I24) ---
    B(I12, I24) = b1224.real();
    
    // B(I13, I24) = k_all * (2.0 * p.eps1 * p.eps3 * (1.0 - C1C3) + (p.eps1*p.eps1 + p.eps3*p.eps3) * S1S3);
    B(I13, I24) = (k_all * (two * v.eps1 * v.eps3 * (one - C1C3) + (v.eps1*v.eps1 + v.eps3*v.eps3) * S1S3)).real();
    
    B(I14, I24) = B(I13, I23); // Symmetry
    B(I23, I24) = B(I13, I14); // Symmetry
    B(I24, I24) = B(I13, I13); // Symmetry

    return B;
}

// --- Liquid (Acoustic) Layer Matrix Calculation ---

// Helper to check if layer is liquid (c44 == 0)
template <typename T>
bool isLiquidLayer(const PhysicalParameters<T>& p) {
    return get_magnitude(p.c44) < 1e-10;
}

// Compute 2x2 Layer Matrix for acoustic wave in liquid (P-wave only)
// State vector: [U, Tzz]^T (vertical displacement, Lagrangian normal stress)
// For raw (unscaled) variables.
template <typename T>
Eigen::Matrix<T, 2, 2> computeLayerMatrixLiquid(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v
) {
    using CT = typename ComplexType<T>::type;
    
    // In liquid: only P-wave propagation
    // nu^2 = k^2 (1 - c^2/alpha^2) where alpha = sqrt(c33/rho)
    // c33 = rho * Vp^2 for liquid
    
    CT k_c = toComplex(v.k);
    CT omega_c = toComplex(v.omega);
    CT rho_c = toComplex(p.rho);
    CT c33_c = toComplex(p.c33);  // = rho * Vp^2
    
    // Phase velocity
    CT c_c = omega_c / k_c;
    
    // P-wave velocity: alpha = sqrt(c33/rho)
    CT alpha_sq = c33_c / rho_c;
    
    // Vertical wavenumber squared: nu^2 = k^2 * (1 - c^2/alpha^2)
    CT nu_sq = k_c * k_c * (toComplex(T(1.0)) - c_c * c_c / alpha_sq);
    CT nu = safe_complex_sqrt(nu_sq);
    
    // Ensure we take the branch with positive real part for decay
    if (get_val(nu.real()) < 0) {
        nu = -nu;
    }
    
    // Propagator for [U, Tzz]^T over thickness z.
    // For g=0 this reduces to the acoustic fluid system:
    // dU/dz = -(nu^2 / (rho omega^2)) * Tzz
    // dTzz/dz = -rho omega^2 * U
    
    CT omega_sq = omega_c * omega_c;
    CT z_c = toComplex(v.z);

    T effective_gravity = p.gravity * p.gravity_internal_scale;
    if (get_magnitude(effective_gravity) > 0.0) {
        CT gravity_c = toComplex(effective_gravity);
        // Experimental Cowling-style constant-g fluid approximation in
        // Lagrangian stress variables Z=[U,Tzz]^T:
        //
        //   dU/dz   = (k^2 g / omega^2) U
        //            + (1/C - k^2/(rho omega^2)) Tzz
        //   dTzz/dz = (-rho omega^2 + k^2 rho g^2 / omega^2) U
        //            - (k^2 g / omega^2) Tzz
        // This is intentionally limited to liquid layers; solid P-SV propagation
        // and the g=0 acoustic branch are unchanged.
        CT kg = k_c * k_c * gravity_c / omega_sq;
        CT a = kg;
        CT b = -nu_sq / (rho_c * omega_sq);
        CT c = -rho_c * omega_sq + k_c * k_c * rho_c * gravity_c * gravity_c / omega_sq;
        CT d = -kg;
        CT tr_half = (a + d) * toComplex(T(0.5));
        CT delta = safe_complex_sqrt(((a - d) * (a - d)) * toComplex(T(0.25)) + b * c);
        CT E = ad_exp(tr_half * z_c);
        CT Cg = ad_cosh(delta * z_c);
        CT Sg_over_delta;
        if (std::abs(get_val(delta.real())) < 1e-15 && std::abs(get_val(delta.imag())) < 1e-15) {
            Sg_over_delta = z_c;
        } else {
            Sg_over_delta = ad_sinh(delta * z_c) / delta;
        }

        Eigen::Matrix<T, 2, 2> B;
        B(0, 0) = (E * (Cg + (a - tr_half) * Sg_over_delta)).real();
        B(0, 1) = (E * b * Sg_over_delta).real();
        B(1, 0) = (E * c * Sg_over_delta).real();
        B(1, 1) = (E * (Cg + (d - tr_half) * Sg_over_delta)).real();
        return B;
    }
    
    CT C = ad_cosh(nu * z_c);
    CT S = ad_sinh(nu * z_c);
    
    Eigen::Matrix<T, 2, 2> B;
    
    // B11 = cosh(nu*H)
    B(0, 0) = C.real();
    
    // B12 = -(nu / (rho omega^2)) * sinh(nu*H)
    // Handle nu ~ 0 case (critical angle)
    CT ratio = nu / (rho_c * omega_sq);
    if (std::abs(get_val(nu.real())) < 1e-15 && std::abs(get_val(nu.imag())) < 1e-15) {
        // nu -> 0: sinh(nu*z)/nu -> z, so B12 -> -z / (rho omega^2)
        B(0, 1) = (-z_c / (rho_c * omega_sq)).real();
    } else {
        B(0, 1) = (-ratio * S).real();
    }
    
    // B21 = -(rho omega^2 / nu) * sinh(nu*H)
    if (std::abs(get_val(nu.real())) < 1e-15 && std::abs(get_val(nu.imag())) < 1e-15) {
        // nu -> 0: (rho omega^2 / nu) * sinh(nu*z) -> rho omega^2 * z
        B(1, 0) = (-rho_c * omega_sq * z_c).real();
    } else {
        B(1, 0) = (-rho_c * omega_sq / nu * S).real();
    }
    
    // B22 = cosh(nu*H)
    B(1, 1) = C.real();
    
    return B;
}

// --- Half-Space Initial Vector Calculation ---

template <typename T>
Eigen::Matrix<T, 5, 1> computeInitialVector(
    const PhysicalParameters<T>& bottomParams,
    const WaveVariables<T>& waveVars // z is irrelevant here, usually 0
) {
    // For half-space, we don't multiply by a layer matrix. 
    // We compute the eigenvector components corresponding to decaying modes.
    // Plan: [Y12(0), Y13(0), Y14(0), Y23(0), Y24(0)] = 
    //       [d2-d1, eps3-eps1, -c44*d5, -c33*d6, c44*(d1*d4-d2*d3)]
    
    DerivedComplexValues<T> v = computeDerivedValues(bottomParams, waveVars);
    
    // Radiative Regime Check (User's Request):
    // If Re(nu) is non-positive, the wave does not decay at infinity.
    // This region should be skipped for surface wave searching.
    if (get_val(v.nu1.real()) <= 1e-10 || get_val(v.nu3.real()) <= 1e-10) {
        Eigen::Matrix<T, 5, 1> Y_radiative;
        Y_radiative.setConstant(make_nan<T>());
        return Y_radiative;
    }
    
    using CT = typename ComplexType<T>::type;
    
    CT val1 = v.d2 - v.d1;
    CT val2 = v.eps3 - v.eps1;
    CT val3 = -toComplex(bottomParams.c44) * v.d5;
    CT val4 = -toComplex(bottomParams.c33) * v.d6;
    CT val5 = toComplex(bottomParams.c44) * (v.d1 * v.d4 - v.d2 * v.d3);
    
    Eigen::Matrix<T, 5, 1> Y0;
    // The vector should be real for non-attenuative media in the Delta matrix formulation?
    // Usually Dunkin's vector is real. Let's assume we take the real part or it naturally becomes real.
    // Note: The terms above involve d1, d2 etc which are complex. 
    // However, for Rayleigh waves in propagative regime, they might be real or purely imaginary.
    // In the evanescent regime (half space), they are complex.
    // The plan implied "dunkin's method", where the compound matrix is real.
    // Let's store as T (real) if possible, or std::complex<T> if needed.
    // The original code used Eigen::Matrix<T, 5, 5> where T was auto-diff scalar (real-like).
    // So we likely need the real parts.
    
    // Logic Validation (User's Insight):
    // The initial vector Y0 (eigenvector in half-space) is physically constrained to be either 
    // Purely Real or Purely Imaginary (standing wave vs evanescent).
    // Since the Dunkin Matrix B describes real-valued operations (in non-attenuative media),
    // we must ensure Y0 is real. 
    // If Y0 is Purely Imaginary, we rotate it by multiplying by -1j (taking the imaginary part).
    
    // Sum magnitudes of Real vs Imag parts to detect phase
    double sum_real = std::abs(get_val(val1.real())) + std::abs(get_val(val2.real())) + std::abs(get_val(val3.real())) + std::abs(get_val(val4.real())) + std::abs(get_val(val5.real()));
    double sum_imag = std::abs(get_val(val1.imag())) + std::abs(get_val(val2.imag())) + std::abs(get_val(val3.imag())) + std::abs(get_val(val4.imag())) + std::abs(get_val(val5.imag()));


    
    if (sum_imag > sum_real) {
        // Purely imaginary -> Rotate by -i (effectively take imaginary part)
        // Y_real = Y_complex * (-i) = (i * Y_imag) * (-i) = Y_imag
        Y0(I12) = val1.imag();
        Y0(I13) = val2.imag();
        Y0(I14) = val3.imag();
        Y0(I23) = val4.imag();
        Y0(I24) = val5.imag();
    } else {
        // Purely real
        Y0(I12) = val1.real();
        Y0(I13) = val2.real();
        Y0(I14) = val3.real();
        Y0(I23) = val4.real();
        Y0(I24) = val5.real();
    }

    
    // --- Canonical Sign Enforcement ---
    // Ensure the vector doesn't arbitrarily flip sign as parameters change.
    // We use I12 (Mixed Minor) as the anchor - proven stable for monotonic counting.
    
    if (get_val(Y0(I12)) < 0) {
        Y0 = -Y0;
    }

    return Y0;
}

// --- Main Dispersion Function ---

// Returns the full vector Y at surface
// Supports both solid and liquid (ocean) layers
template <typename T>
Eigen::Matrix<T, 5, 1> calculateRayleighDispersion(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses, // Last element (half-space) thickness is ignored/infinite
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity,
    NormalizationMode normMode = NormalizationMode::None,
    Eigen::Matrix<T, 5, 1>* seabedState = nullptr,
    std::vector<double>* sFactors = nullptr,
    size_t* sIdx = nullptr
) {
    ++disper26_stats::dispersion_evals_total;
    if constexpr (std::is_same_v<T, double>) ++disper26_stats::dispersion_evals_double;

    T k = omega / phaseVelocity;

    WaveVariables<T> wave;
    wave.omega = omega;
    wave.k = k;
    wave.z = make_zero_like(omega); // Temporary
    
    // Check if we're starting from liquid half-space (not supported yet)
        if (isLiquidLayer(halfSpaceParams)) {
        Eigen::Matrix<T, 5, 1> Y_nan;
        Y_nan.setConstant(make_nan<T>());
        return Y_nan;
    }
    
    // 1. Initial Vector from Solid Half-Space
    Eigen::Matrix<T, 5, 1> Y = computeInitialVector(halfSpaceParams, wave);
    
    // 2. Propagate up through layers (Bottom to Top)
    // Track if we're currently in liquid mode
    bool inLiquidMode = false;
    Eigen::Matrix<T, 2, 1> Z;  // 2D liquid state [w, P]^T
    
    for (int i = layers.size() - 1; i >= 0; --i) {
        wave.z = thicknesses[i];
        bool currentLayerIsLiquid = isLiquidLayer(layers[i]);
        
        if (!inLiquidMode && !currentLayerIsLiquid) {
            // Solid to Solid: Normal 5x5 propagation
            Eigen::Matrix<T, 5, 5> B = computeLayerMatrix(layers[i], wave);
            Y = B * Y;
            
            // --- Scaling / Normalization ---
            if (sFactors) {
                // Hessian-compatible path: unified double-valued scaling
                double scale = 1.0;
                if (normMode == NormalizationMode::Provided && sIdx) {
                    if (*sIdx < sFactors->size()) scale = (*sFactors)[(*sIdx)++];
                } else {
                    double m = 0;
                    for(int j=0; j<5; ++j) m += get_magnitude(Y(j));
                    if (m > 1e-20) scale = 1.0 / m;
                    if (!sIdx) sFactors->push_back(scale);
                }
                Y = Y * T(scale);
            } else if (normMode == NormalizationMode::None) {
                // Original Max-Norm (Stop Gradient)
                double max_val = 0.0;
                for(int m=0; m<5; ++m) {
                    double v = get_magnitude(Y(m)); 
                    if(v > max_val) max_val = v;
                }
                const double THRESHOLD = 1e10;
                if (max_val > THRESHOLD) {
                    double scale = 1.0 / max_val;
                    Y = Y * T(scale);
                }
            } else if (normMode == NormalizationMode::Vertical) {
                 // Propagate gradients through normalization (Y(0) = U)
                 T val = Y(0);
                 if (get_magnitude(val) > 1e-20) {
                     T scale = T(1.0) / val;
                     Y = Y * scale;
                 }
            } else if (normMode == NormalizationMode::VectorNorm) {
                 // Normalize by Euclidean Norm (Sqrt(Sum(Y^2)))
                 T sum_sq = T(0.0);
                 for(int m=0; m<5; ++m) sum_sq += Y(m)*Y(m);
                 
                 // Use AD-compatible sqrt
                 using std::sqrt;
                 T norm = sqrt(sum_sq);
                 
                 if (get_magnitude(norm) > 1e-20) {
                      T scale = T(1.0) / norm;
                      Y = Y * scale;
                 }
            }
            
        } else if (!inLiquidMode && currentLayerIsLiquid) {
            // Solid to Liquid Transition
            if (seabedState) *seabedState = Y;
            
            // Extract 2D state from 5D minors:
            // Z[0] = w (displacement) <- from Y(I14)
            // Z[1] = P (pressure) <- from Y(I24)
            Z(0) = Y(I14);
            Z(1) = Y(I24);
            T seabed_gravity = layers[i].gravity * layers[i].gravity_seabed_scale;
            if (get_magnitude(seabed_gravity) > 0.0) {
                T rho_solid = (i + 1 < static_cast<int>(layers.size())) ? layers[i + 1].rho : halfSpaceParams.rho;
                T delta_rho = rho_solid - layers[i].rho;
                Z(1) = Z(1) + delta_rho * seabed_gravity * Z(0);
            }
            
            // Propagate through liquid layer
            Eigen::Matrix<T, 2, 2> B_liq = computeLayerMatrixLiquid(layers[i], wave);
            Z = B_liq * Z;
            
            // Scaling
            if (sFactors) {
                double scale = 1.0;
                if (normMode == NormalizationMode::Provided && sIdx) {
                    if (*sIdx < sFactors->size()) scale = (*sFactors)[(*sIdx)++];
                } else {
                    double m = get_magnitude(Z(0)) + get_magnitude(Z(1));
                    if (m > 1e-20) scale = 1.0 / m;
                    if (!sIdx) sFactors->push_back(scale);
                }
                Z = Z * T(scale);
            } else if (normMode == NormalizationMode::VectorNorm) {
                T norm = sqrt(Z(0)*Z(0) + Z(1)*Z(1));
                if (get_magnitude(norm) > 1e-20) {
                    Z = Z / norm;
                }
            } else if (normMode == NormalizationMode::Vertical) {
                T val = Z(0);
                if (get_magnitude(val) > 1e-20) {
                    Z = Z / val;
                }
            } else {
                double max_val = std::max(get_magnitude(Z(0)), get_magnitude(Z(1)));
                const double THRESHOLD = 1e10;
                if (max_val > THRESHOLD) {
                    double scale = 1.0 / max_val;
                    Z = Z * T(scale);
                }
            }
            
            inLiquidMode = true;
            
        } else if (inLiquidMode && currentLayerIsLiquid) {
            // Liquid to Liquid: 2x2 propagation
            Eigen::Matrix<T, 2, 2> B_liq = computeLayerMatrixLiquid(layers[i], wave);
            Z = B_liq * Z;
            
            // Scaling
            if (sFactors) {
                double scale = 1.0;
                if (normMode == NormalizationMode::Provided && sIdx) {
                    if (*sIdx < sFactors->size()) scale = (*sFactors)[(*sIdx)++];
                } else {
                    double m = get_magnitude(Z(0)) + get_magnitude(Z(1));
                    if (m > 1e-20) scale = 1.0 / m;
                    if (!sIdx) sFactors->push_back(scale);
                }
                Z = Z * T(scale);
            } else if (normMode == NormalizationMode::VectorNorm) {
                T norm = sqrt(Z(0)*Z(0) + Z(1)*Z(1));
                if (get_magnitude(norm) > 1e-20) {
                    Z = Z / norm;
                }
            } else if (normMode == NormalizationMode::Vertical) {
                T val = Z(0);
                if (get_magnitude(val) > 1e-20) {
                    Z = Z / val;
                }
            } else {
                double max_val = std::max(get_magnitude(Z(0)), get_magnitude(Z(1)));
                const double THRESHOLD = 1e10;
                if (max_val > THRESHOLD) {
                    double scale = 1.0 / max_val;
                    Z = Z * T(scale);
                }
            }
            
        } else {
            // Liquid to Solid Transition (not typically needed for ocean on top)
            Eigen::Matrix<T, 5, 1> Y_nan;
            Y_nan.setConstant(make_nan<T>());
            return Y_nan;
        }
    }
    
    // 3. Return result
    if (inLiquidMode) {
        // Map 2D liquid state to 5D result
        // For a liquid free surface, the default secular condition is Tzz = 0.
        // The optional surface scale applies an Eulerian-pressure diagnostic
        // correction only when explicitly requested.
        // Initialize with zero but preserve AD size if applicable
        Eigen::Matrix<T, 5, 1> Y_result;
        for(int j=0; j<5; ++j) Y_result(j) = Z(0) * T(0.0);
        T surface_pressure = Z(1);
        if (!layers.empty() && isLiquidLayer(layers.front())) {
            T surface_gravity = layers.front().gravity * layers.front().gravity_surface_scale;
            if (get_magnitude(surface_gravity) > 0.0) {
                surface_pressure = Z(1) - layers.front().rho * surface_gravity * Z(0);
            }
        }
        
        if (normMode == NormalizationMode::VectorNorm || normMode == NormalizationMode::Vertical || sFactors) {
             // In these modes, the vector is already normalized at the last layer
             Y_result(I14) = Z(0);
             Y_result(I24) = surface_pressure;
        } else {
            // Traditional normalization by displacement magnitude for scale-invariance
            double disp_mag = get_magnitude(Z(0));
            if (disp_mag > 1e-30) {
                Y_result(I14) = Z(0) / disp_mag;
                Y_result(I24) = surface_pressure / disp_mag;
            } else {
                Y_result(I14) = Z(0);
                Y_result(I24) = surface_pressure;
            }
        }
        return Y_result;
    } else {
        // Return solid 5D vector
        // Optionally normalize by I14 for scale-invariance in low-velocity regime
        double i14_mag = get_magnitude(Y(I14));
        if (i14_mag > 1e-30) {
            Y = Y / T(i14_mag);
        }
        return Y;
    }
}

// --- Explicit 4x4 Layer Matrix for VTI Rayleigh (P-SV) ---
// Based on formulas provided in docs/Haskell_matrix.md
// Verified to limit correctly to small z and be consistent with Disper26 variables.

template <typename T>
Eigen::Matrix<std::complex<T>, 4, 4> computeLayerMatrixP_SV_Explicit(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v_var
) {
    using CT = std::complex<T>;
    DerivedComplexValues<T> dv = computeDerivedValues(p, v_var);
    
    CT d1 = dv.d1; CT d2 = dv.d2;
    CT d3 = dv.d3; CT d4 = dv.d4;
    CT d5 = dv.d5; CT d6 = dv.d6;
    CT C1 = dv.C1; CT C3 = dv.C3;
    CT S1 = dv.S1; CT S3 = dv.S3;
    CT eps1 = dv.eps1; CT eps3 = dv.eps3;
    CT c33 = toComplex(p.c33);
    CT c44 = toComplex(p.c44);
    
    Eigen::Matrix<CT, 4, 4> A;
    
    // Row 1 (u_r)
    // A11 = 1/d5 * (-d4 C1 + d3 C3)
    A(0,0) = (T(1.0)/d5) * (-d4 * C1 + d3 * C3);
    // A12 = 1/(c33 d6) * (-eps3 S1 + eps1 S3)
    A(0,1) = (T(1.0)/(c33 * d6)) * (-eps3 * S1 + eps1 * S3);
    // A13 = 1/(c33 d6) * (d2 S1 - d1 S3)
    A(0,2) = (T(1.0)/(c33 * d6)) * (d2 * S1 - d1 * S3);
    // A14 = 1/(c44 d5) * (C1 - C3)
    A(0,3) = (T(1.0)/(c44 * d5)) * (C1 - C3);
    
    // Row 2 (u_z)
    // A21 = 1/d5 * (-d1 d4 S1 + d2 d3 S3)
    A(1,0) = (T(1.0)/d5) * (-d1 * d4 * S1 + d2 * d3 * S3);
    // A22 = A11
    A(1,1) = A(0,0);
    // A23 = d1 d2 / (c33 d6) * (C1 - C3)
    A(1,2) = (d1 * d2 / (c33 * d6)) * (C1 - C3);
    // A24 = 1/(c44 d5) * (d1 S1 - d2 S3)
    A(1,3) = (T(1.0)/(c44 * d5)) * (d1 * S1 - d2 * S3);
    
    // Row 3 (sigma_zz)
    // A31 = 1/d5 * (-eps1 d4 S1 + eps3 d3 S3)
    A(2,0) = (T(1.0)/d5) * (-eps1 * d4 * S1 + eps3 * d3 * S3); 
    // A32 = eps1 eps3 / (c33 d6) * (-C1 + C3)
    A(2,1) = (eps1 * eps3 / (c33 * d6)) * (-C1 + C3);
    // A33 = 1/(c33 d6) * (eps1 d2 C1 - eps3 d1 C3)
    A(2,2) = (T(1.0)/(c33 * d6)) * (eps1 * d2 * C1 - eps3 * d1 * C3);
    // A34 = 1/(c44 d5) * (eps1 S1 - eps3 S3)
    A(2,3) = (T(1.0)/(c44 * d5)) * (eps1 * S1 - eps3 * S3);
    
    // Row 4 (sigma_zr)
    // A41 = c44 d3 d4 / d5 * (-C1 + C3)
    A(3,0) = (c44 * d3 * d4 / d5) * (-C1 + C3);
    // A42 = c44 / (c33 d6) * (-eps3 d3 S1 + eps1 d4 S3)
    A(3,1) = (c44 / (c33 * d6)) * (-eps3 * d3 * S1 + eps1 * d4 * S3);
    // A43 = c44 / (c33 d6) * (d2 d3 S1 - d1 d4 S3)
    A(3,2) = (c44 / (c33 * d6)) * (d2 * d3 * S1 - d1 * d4 * S3);
    // A44 = 1/d5 (d3 C1 - d4 C3) -> Checks against A33 formula logic? 
    // Wait, A33 was (eps1 d2 C1...), A44 is (d3 C1...).
    // Explicit doc: A44 = A33. Let's trust user document formula A44 line.
    A(3,3) = (T(1.0)/d5) * (d3 * C1 - d4 * C3);
    
    return A;
}

// Eigenfunction Result Structure
template <typename T>
struct RayleighEigenfunctionPoint {
    T depth;
    std::complex<T> ur;
    std::complex<T> uz;
    std::complex<T> szz;
    std::complex<T> szr;
    // True for samples whose state obeys the fluid (acoustic) relations.
    // At the seabed two samples share the same depth — the fluid side and the
    // solid side — so consumers must use this flag, not the depth.
    bool is_fluid = false;
};

// Energy Integral Result Structure
template <typename T>
struct EnergyIntegralResult {
    T I1; // integral rho * |u|^2 dz
    T I2; // integral (Potential Energy related)
    T I3; // integral (Flux related)
    T phaseVelocity;
    T groupVelocity_Energy; // U = I3 / (c * I1)
};

template <typename T>
bool hasLiquidLayer(const std::vector<PhysicalParameters<T>>& layers) {
    for (const auto& layer : layers) {
        if (isLiquidLayer(layer)) return true;
    }
    return false;
}

template <typename T>
int countTopLiquidLayersOrThrow(const std::vector<PhysicalParameters<T>>& layers) {
    int liquid_count = 0;
    while (liquid_count < static_cast<int>(layers.size()) && isLiquidLayer(layers[liquid_count])) {
        ++liquid_count;
    }
    for (int i = liquid_count; i < static_cast<int>(layers.size()); ++i) {
        if (isLiquidLayer(layers[i])) {
            throw std::runtime_error("Rayleigh fluid eigenfunctions support only a contiguous liquid layer stack at the model top.");
        }
    }
    return liquid_count;
}

template <typename T>
typename ComplexType<T>::type computeFluidHorizontalDisplacement(
    const PhysicalParameters<T>& layer,
    const typename ComplexType<T>::type& uz,
    const typename ComplexType<T>::type& tzz,
    T omega,
    T phaseVelocity
) {
    using CT = typename ComplexType<T>::type;
    T k = omega / phaseVelocity;
    T omega_sq = omega * omega;
    T effective_gravity = layer.gravity * layer.gravity_internal_scale;
    T denom = layer.rho * omega_sq;
    if (get_magnitude(denom) < 1e-30) return CT(T(0.0));
    T factor = k / denom;
    T gravity_coeff = layer.rho * effective_gravity;
    // Saito's liquid relation for the Lagrangian normal-stress state:
    // y3 = k/(rho w^2) * (rho g y1 - y2).
    return CT(factor) * (CT(gravity_coeff) * uz - tzz);
}

template <typename T>
struct RayleighVirtualLayerModel {
    std::vector<PhysicalParameters<T>> layers;
    std::vector<T> thicknesses;
    std::vector<int> original_layer_index;
    std::vector<int> splits_per_original_layer;
};

template <typename T>
int computeAdaptiveRayleighSplitCount(
    const PhysicalParameters<T>& layer,
    const T& thickness,
    const T& omega,
    int min_subdivisions = 1,
    int max_virtual_splits = 64
) {
    int min_steps = std::max(1, min_subdivisions);
    double h = get_scalar_val(thickness);
    double omega_val = std::abs(get_scalar_val(omega));
    if (!(h > 0.0) || !(omega_val > 0.0)) return min_steps;

    double rho = std::abs(get_scalar_val(layer.rho));
    double c44 = std::abs(get_scalar_val(layer.c44));
    if (!(rho > 0.0) || !(c44 > 1e-12)) return min_steps;

    double vs = layer.vs();
    if (!std::isfinite(vs) || !(vs > 0.0)) return min_steps;

    double period = 2.0 * M_PI / omega_val;
    double dz_target = 0.5 * vs * period;
    if (!std::isfinite(dz_target) || !(dz_target > 0.0)) return min_steps;

    int adaptive_steps = static_cast<int>(std::ceil(h / dz_target));
    adaptive_steps = std::max(min_steps, adaptive_steps);
    return std::clamp(adaptive_steps, 1, std::max(1, max_virtual_splits));
}

template <typename T>
RayleighVirtualLayerModel<T> buildAdaptiveRayleighVirtualLayers(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const T& omega,
    int min_subdivisions = 1,
    int max_virtual_splits = 64
) {
    RayleighVirtualLayerModel<T> out;
    out.splits_per_original_layer.resize(layers.size(), 1);
    out.layers.reserve(layers.size() * std::max(1, min_subdivisions));
    out.thicknesses.reserve(layers.size() * std::max(1, min_subdivisions));
    out.original_layer_index.reserve(layers.size() * std::max(1, min_subdivisions));

    for (size_t i = 0; i < layers.size(); ++i) {
        int nsplit = computeAdaptiveRayleighSplitCount(
            layers[i], thicknesses[i], omega, min_subdivisions, max_virtual_splits);
        out.splits_per_original_layer[i] = nsplit;
        T h_sub = thicknesses[i] / T(nsplit);
        for (int j = 0; j < nsplit; ++j) {
            out.layers.push_back(layers[i]);
            out.thicknesses.push_back(h_sub);
            out.original_layer_index.push_back(static_cast<int>(i));
        }
    }
    return out;
}

// Compute Eigenfunctions (Robust 4x4 Upward Match + Downward Integration)
// Compute Rayleigh Eigenfunctions (Stabilized Two-Pass Method)
template <typename T>
std::vector<RayleighEigenfunctionPoint<T>> computeRayleighEigenfunctions(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity,
    int numSubLayers = 10
) {
    using CT = std::complex<T>;
    using Vector5 = Eigen::Matrix<CT, 5, 1>;
    std::vector<RayleighEigenfunctionPoint<T>> results;
    if (isLiquidLayer(halfSpaceParams)) {
        throw std::runtime_error("Rayleigh eigenfunctions do not support a liquid half-space.");
    }

    auto virtual_layers = buildAdaptiveRayleighVirtualLayers(
        layers, thicknesses, omega, numSubLayers);
    const auto& trace_layers = virtual_layers.layers;
    const auto& trace_thicknesses = virtual_layers.thicknesses;
    const int top_liquid_count = countTopLiquidLayersOrThrow(trace_layers);
    
    T k = omega / phaseVelocity;
    WaveVariables<T> wave;
    wave.omega = omega;
    wave.k = k;
    wave.z = make_zero_like(omega);

    if (top_liquid_count > 0) {
        const int solid_count = static_cast<int>(trace_layers.size()) - top_liquid_count;
        std::vector<PhysicalParameters<T>> solid_layers;
        std::vector<T> solid_thicknesses;
        solid_layers.reserve(std::max(0, solid_count));
        solid_thicknesses.reserve(std::max(0, solid_count));
        for (int i = top_liquid_count; i < static_cast<int>(trace_layers.size()); ++i) {
            solid_layers.push_back(trace_layers[i]);
            solid_thicknesses.push_back(trace_thicknesses[i]);
        }

        std::vector<Vector5> interfaceMinors(solid_layers.size() + 1);
        Vector5 Y_curr = computeInitialVector(halfSpaceParams, wave);
        double max_val = 0.0;
        for (int m = 0; m < 5; ++m) max_val = std::max(max_val, std::abs(Y_curr(m)));
        if (max_val > 0.0) Y_curr /= T(max_val);
        interfaceMinors[solid_layers.size()] = Y_curr;

        for (int i = static_cast<int>(solid_layers.size()) - 1; i >= 0; --i) {
            wave.z = solid_thicknesses[i];
            Eigen::Matrix<CT, 5, 5> Qshifted = computeLayerMatrix(solid_layers[i], wave);
            Y_curr = Qshifted * Y_curr;
            max_val = 0.0;
            for (int m = 0; m < 5; ++m) max_val = std::max(max_val, std::abs(Y_curr(m)));
            if (max_val > 1e-20) Y_curr /= T(max_val);
            else Y_curr.setZero();
            interfaceMinors[i] = Y_curr;
        }

        Eigen::Matrix<CT, 2, 1> fluid_state;
        fluid_state(0) = CT(T(1.0));
        T surface_gravity = trace_layers.front().gravity * trace_layers.front().gravity_surface_scale;
        T surface_coeff = trace_layers.front().rho * surface_gravity;
        fluid_state(1) = CT(surface_coeff) * fluid_state(0);

        T currentDepth = T(0.0);
        auto push_fluid_point = [&](const PhysicalParameters<T>& layer) {
            CT uz = fluid_state(0);
            CT szz = fluid_state(1);
            CT ur = computeFluidHorizontalDisplacement<T>(layer, uz, szz, omega, phaseVelocity);
            results.push_back({currentDepth, ur, uz, szz, CT(T(0.0)), true});
        };

        push_fluid_point(trace_layers.front());
        for (int i = 0; i < top_liquid_count; ++i) {
            wave.z = -trace_thicknesses[i];
            auto B_liq = computeLayerMatrixLiquid(trace_layers[i], wave);
            Eigen::Matrix<CT, 2, 1> next_state;
            next_state(0) = toComplex(B_liq(0,0)) * fluid_state(0) + toComplex(B_liq(0,1)) * fluid_state(1);
            next_state(1) = toComplex(B_liq(1,0)) * fluid_state(0) + toComplex(B_liq(1,1)) * fluid_state(1);
            fluid_state = next_state;
            currentDepth += trace_thicknesses[i];
            push_fluid_point(trace_layers[i]);
            if (i + 1 < top_liquid_count) {
                push_fluid_point(trace_layers[i + 1]);
            }
        }

        Vector5 Y_seabed = interfaceMinors[0];
        CT uz = fluid_state(0);
        CT szz = fluid_state(1);
        const PhysicalParameters<T>& seabed_solid =
            solid_layers.empty() ? halfSpaceParams : solid_layers.front();
        const PhysicalParameters<T>& bottom_fluid = trace_layers[top_liquid_count - 1];
        T seabed_gravity = bottom_fluid.gravity * bottom_fluid.gravity_seabed_scale;
        if (get_magnitude(seabed_gravity) > 0.0) {
            T delta_rho = seabed_solid.rho - bottom_fluid.rho;
            T seabed_coeff = delta_rho * seabed_gravity;
            szz = szz - CT(seabed_coeff) * uz;
        }

        CT ur = CT(T(0.0));
        if (std::abs(Y_seabed(I12)) > 1e-15) {
            ur = (szz * Y_seabed(I13) - uz * Y_seabed(I23)) / Y_seabed(I12);
        } else if (std::abs(Y_seabed(I14)) > 1e-15) {
            // szr = 0 at the seabed, so ur*M14 + uz*M12 = szr*M13 = 0.
            ur = -uz * Y_seabed(I12) / Y_seabed(I14);
        }
        Eigen::Matrix<CT, 4, 1> state;
        state << uz, szz, ur, CT(T(0.0));
        results.push_back({currentDepth, state(2), state(0), state(1), state(3)});

        for (int i = 0; i < solid_count; ++i) {
            wave.z = -solid_thicknesses[i];
            state = computeLayerMatrixP_SV_Explicit(solid_layers[i], wave) * state;
            currentDepth += solid_thicknesses[i];

            Vector5 Y_local = interfaceMinors[i + 1];
            CT M13 = Y_local(I13);
            if (std::abs(M13) > 1e-15) {
                CT M12 = Y_local(I12);
                CT M23 = Y_local(I23);
                CT M14 = Y_local(I14);
                CT ur_local = state(2);
                CT uz_local = state(0);
                state(1) = (uz_local * M23 + ur_local * M12) / M13;
                state(3) = (ur_local * M14 + uz_local * M12) / M13;
            }
            results.push_back({currentDepth, state(2), state(0), state(1), state(3)});
        }

        return results;
    }
    
    // --- Step 1: Upward Pass (Store Minors at Interfaces) ---
    // We store Y at the TOP of each layer.
    // Indexing: layers[0] is top layer. halfSpace is below layers[N-1].
    // We iterate from Bottom up.
    // interfaceMinors[i] stores Y at TOP of layer i.
    // interfaceMinors[N] stores Y at Top of Halfspace (Bottom of layer N-1).
    std::vector<Vector5> interfaceMinors(trace_layers.size() + 1);
    
    // Bottom (Halfspace) Minor Vector
    Vector5 Y_curr = computeInitialVector(halfSpaceParams, wave);
    
    // Normalize
    double max_val = 0.0;
    for(int m=0; m<5; ++m) max_val = std::max(max_val, std::abs(Y_curr(m)));
    if(max_val > 0) Y_curr /= T(max_val);
    
    interfaceMinors[trace_layers.size()] = Y_curr;
    
    // Propagate Up
    for (int i = static_cast<int>(trace_layers.size()) - 1; i >= 0; --i) {
        wave.z = trace_thicknesses[i];
        Eigen::Matrix<CT, 5, 5> Qshifted = computeLayerMatrix(trace_layers[i], wave);
        Y_curr = Qshifted * Y_curr;
        
        // Normalize
        max_val = 0.0;
        for(int m=0; m<5; ++m) max_val = std::max(max_val, std::abs(Y_curr(m)));
        if(max_val > 1e-20) Y_curr /= T(max_val);
        else Y_curr.setZero();
        
        interfaceMinors[i] = Y_curr;
    }
    
    // --- Step 2: Surface Initial Condition ---
    Vector5 Y_surf = interfaceMinors[0];
    
    // Ikeda-Derived Initial Condition Logic
    // Derived from Plucker Determinant Relations for Ikeda Variables (uz, szz, ur, szr).
    // Condition szz=0 implies: uz*Y(I23) + ur*Y(I12) = 0.
    // Solution: ur = Y(I23), uz = -Y(I12).
    
    CT ur_surf = Y_surf(I23);
    CT uz_surf = -Y_surf(I12);
    
    // Initial State (uz, szz, ur, szr)
    Eigen::Matrix<CT, 4, 1> state;
    state << uz_surf, CT(0.0), ur_surf, CT(0.0);
    
    // Check normalization
    if (std::abs(ur_surf) < 1e-20 && std::abs(uz_surf) < 1e-20) {
         state << CT(1.0), CT(0.0), CT(1.0), CT(0.0); // Dummy fallback
    } else {
        if (std::abs(uz_surf) > std::abs(ur_surf)) state /= uz_surf;
        else state /= ur_surf;
    }
    
    // --- Step 3: Downward Pass ---
    T currentDepth = T(0.0);
    results.push_back({currentDepth, state(2), state(0), state(1), state(3)}); // {depth, ur, uz, szz, szr}
    
    for (size_t i = 0; i < trace_layers.size(); ++i) {
        wave.z = -trace_thicknesses[i]; // Downward
        state = computeLayerMatrixP_SV_Explicit(trace_layers[i], wave) * state;
        currentDepth += trace_thicknesses[i];

        // --- Interface Stabilization (Ikeda Basis Reconstruction) ---
        Vector5 Y_local = interfaceMinors[i+1];
        CT M13 = Y_local(I13);
        if (std::abs(M13) > 1e-15) {
            CT M12 = Y_local(I12);
            CT M23 = Y_local(I23);
            CT M14 = Y_local(I14);

            CT ur = state(2); // y3
            CT uz = state(0); // y1

            CT szz_recon = (uz * M23 + ur * M12) / M13;
            CT szr_recon = (ur * M14 + uz * M12) / M13;

            state(1) = szz_recon;
            state(3) = szr_recon;
        }
        results.push_back({currentDepth, state(2), state(0), state(1), state(3)});
    }
    
    return results;
}

// Compute Rayleigh Energy Integrals (Standardized Ikeda Mapping)
template <typename T>
EnergyIntegralResult<T> computeRayleighEnergyIntegrals(
    const std::vector<RayleighEigenfunctionPoint<T>>& eigenfunctions,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity
) {
    EnergyIntegralResult<T> res; res.I1 = T(0.0); res.I2 = T(0.0); res.I3 = T(0.0); res.phaseVelocity = phaseVelocity;
    if (eigenfunctions.size() < 2) return res;
    T k = omega / phaseVelocity; T k_sq = k * k;

    for (size_t i = 0; i < eigenfunctions.size() - 1; ++i) {
        const auto& p1 = eigenfunctions[i]; const auto& p2 = eigenfunctions[i+1];
        T dz = p2.depth - p1.depth; T z_mid = (p1.depth + p2.depth) * 0.5;
        PhysicalParameters<T> p = halfSpaceParams; T dep = 0.0;
        for(size_t l=0; l<layers.size(); ++l) { if(z_mid < dep+thicknesses[l]) { p=layers[l]; break; } dep+=thicknesses[l]; }

        auto integrands = [&](const RayleighEigenfunctionPoint<T>& pt) {
            // Ikeda: y1=uz, y2=szz, y3=ur, y4=szr.
            // Our struct has: ur, uz, szz, szr.
            // Be careful to map correctly.
            T y1 = pt.uz.real(); T y2 = pt.szz.real(); T y3 = pt.ur.real(); T y4 = pt.szr.real();
            
            // I1 = rho * (y1^2 + y3^2)
            T i1 = p.rho * (y1*y1 + y3*y3);
            if (isLiquidLayer(p)) {
                T i2_fluid = y2*y2 / p.c33;
                T i3_fluid = -(y2*y3) / k;
                return std::vector<T>{i1, i2_fluid, i3_fluid};
            }
            // I2 = y2^2/C + y4^2/L + k^2(A - F^2/C)y3^2
            T i2 = (y2*y2/p.c33) + (y4*y4/p.c44) + k_sq * (p.c11 - p.c13*p.c13/p.c33) * y3*y3;
            // I3 = (A - F^2/C)y3^2 + (y1 y4 - F/C y2 y3)/k
            T i3 = (p.c11 - p.c13*p.c13/p.c33) * y3*y3 + (y1*y4 - (p.c13/p.c33)*y2*y3)/k;
            return std::vector<T>{i1, i2, i3};
        };
        auto v1 = integrands(p1); auto v2 = integrands(p2);
        res.I1 += (v1[0] + v2[0]) * 0.5 * dz; res.I2 += (v1[1] + v2[1]) * 0.5 * dz; res.I3 += (v1[2] + v2[2]) * 0.5 * dz;
    }
    // Half-space (Analytic)
    WaveVariables<T> wave; wave.omega = omega; wave.k = k; wave.z = T(0.0);
    DerivedComplexValues<T> dv = computeDerivedValues(halfSpaceParams, wave);
    T y1_0 = eigenfunctions.back().uz.real(); T y3_0 = eigenfunctions.back().ur.real();
    std::complex<T> eps1 = dv.eps1; std::complex<T> eps3 = dv.eps3; std::complex<T> det = eps1 - eps3;
    // c1 = (y3 - y1*eps3) / (eps1 - eps3)
    // c3 = (y1*eps1 - y3) / (eps1 - eps3)
    
    std::complex<T> c1 = (toComplex(y3_0) - eps3 * toComplex(y1_0)) / det;
    std::complex<T> c3 = (eps1 * toComplex(y1_0) - toComplex(y3_0)) / det;
    
    std::complex<T> cs[2] = {c1, c3}; std::complex<T> nus[2] = {dv.nu1, dv.nu3};
    
    std::complex<T> e_v[2] = {eps1, eps3}; 
    // d_v[0] = d1, d_v[1] = d2 (from DerivedComplexValues)
    // h_v[0] = d3*c44, h_v[1] = d4*c44 (from DerivedComplexValues)
    std::complex<T> d_v[2] = {dv.d1, dv.d2};
    std::complex<T> h_v[2] = {dv.d3 * toComplex(halfSpaceParams.c44), dv.d4 * toComplex(halfSpaceParams.c44)};
    
    T rho = halfSpaceParams.rho; T C = halfSpaceParams.c33; T L = halfSpaceParams.c44; T A = halfSpaceParams.c11; T F = halfSpaceParams.c13;

    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            // Integral of exp(-(nu_i + nu_j*)z) from 0 to inf is 1/(nu_i + nu_j*)
            std::complex<T> w = (cs[i] * std::conj(cs[j])) / (nus[i] + std::conj(nus[j]));
            
            // I1 = rho * (uz^2 + ur^2)
            // uz = c1*exp(-nu1*z) + c3*exp(-nu3*z)
            // ur = c1*eps1*exp(-nu1*z) + c3*eps3*exp(-nu3*z)
            // uz*uz* + ur*ur* = (c_i exp(-nu_i z) + c_j exp(-nu_j z)) * (c_k* exp(-nu_k* z) + c_l* exp(-nu_l* z))
            // This is sum over i,j of c_i c_j* (1 + eps_i eps_j*) exp(-(nu_i + nu_j*)z)
            res.I1 += (w * rho * (T(1.0) + e_v[i]*std::conj(e_v[j]))).real(); 
            
            // I2 = szz^2/C + szr^2/L + k^2(A - F^2/C)ur^2
            // szz = c1*d1*exp(-nu1*z) + c3*d2*exp(-nu3*z)
            // szr = c1*h1*exp(-nu1*z) + c3*h2*exp(-nu3*z)
            res.I2 += (w * (d_v[i]*std::conj(d_v[j])/C + h_v[i]*std::conj(h_v[j])/L + k_sq*(A - F*F/C)*e_v[i]*std::conj(e_v[j]))).real();
            
            // I3 = (A - F^2/C)ur^2 + (uz szr - F/C szz ur)/k
            // uz szr = (c_i exp(-nu_i z)) * (c_j* h_j* exp(-nu_j* z))
            // szz ur = (c_i d_i exp(-nu_i z)) * (c_j* e_j* exp(-nu_j* z))
            res.I3 += (w * ((A-F*F/C)*e_v[i]*std::conj(e_v[j]) + (T(1.0)*std::conj(h_v[j]) - (F/C)*d_v[i]*std::conj(e_v[j])) / k)).real();
        }
    }
    // get_magnitude, not std::abs: the latter has no AutoDiffScalar overload.
    if (get_magnitude(res.I1) > 1e-20) res.groupVelocity_Energy = res.I3 / (phaseVelocity * res.I1);
    return res;
}

// Compute 2x2 Layer Matrix for SH wave
template <typename T>
Eigen::Matrix<T, 2, 2> computeLayerMatrixSH(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v
) {
    using CT = typename ComplexType<T>::type;
    
    // Vertical Slowness/Wavenumber Calculation
    // For VTI SH: c44 (d^2 u / dz^2) + (rho w^2 - c66 k^2) u = 0
    // d^2 u / dz^2 - nu^2 u = 0
    // nu^2 = (c66 k^2 - rho w^2) / c44
    
    CT c44_c = toComplex(p.c44);
    CT c66_c = toComplex(p.c66);
    CT rho_c = toComplex(p.rho);
    CT k_c   = toComplex(v.k);
    CT w_c   = toComplex(v.omega);
    
    CT num = c66_c * k_c * k_c - rho_c * w_c * w_c;
    CT nu_sq = num / c44_c;
    CT nu = safe_complex_sqrt(nu_sq);
    
    CT eta = c44_c * nu; // "Impedance"
    
    CT C = ad_cosh(nu * toComplex(v.z));
    CT S = ad_sinh(nu * toComplex(v.z));
    
    Eigen::Matrix<T, 2, 2> A;
    
    // A11 = C
    A(0, 0) = C.real();
    
    // A12 = S / eta
    // Protect against eta ~ 0 (k ~ 0 or rho w^2 ~ c66 k^2)
    // If eta -> 0, nu -> 0. S/eta -> z.
    if (std::abs(get_val(eta.real())) < 1e-15 && std::abs(get_val(eta.imag())) < 1e-15) {
        A(0, 1) = v.z; // Limit
    } else {
        A(0, 1) = (S / eta).real();
    }
    
    // A21 = eta * S
    A(1, 0) = (eta * S).real();
    
    // A22 = C
    A(1, 1) = C.real();
    
    return A;
}

// Love Wave Eigenfunction Result Structure
template <typename T>
struct LoveEigenfunctionPoint {
    T depth;
    std::complex<T> uy;  // Anti-plane displacement
    std::complex<T> tyz; // Anti-plane stress
};

// Compute Love Eigenfunctions (Upward Integration)
template <typename T>
std::vector<LoveEigenfunctionPoint<T>> computeLoveEigenfunctions(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity,
    int numSubLayers = 10
) {
    using CT = std::complex<T>;
    std::vector<LoveEigenfunctionPoint<T>> results;
    if (numSubLayers < 1) numSubLayers = 1;
    
    T k = omega / phaseVelocity;
    WaveVariables<T> wave; wave.omega = omega; wave.k = k;
    wave.z = make_zero_like(omega);

    CT c44_c = toComplex(halfSpaceParams.c44);
    CT c66_c = toComplex(halfSpaceParams.c66);
    CT rho_c = toComplex(halfSpaceParams.rho);
    CT k_c   = toComplex(k);
    CT w_c   = toComplex(omega);
    CT nu = safe_complex_sqrt((c66_c * k_c * k_c - rho_c * w_c * w_c) / c44_c);
    if (get_val(nu.real()) < 0) nu = -nu;

    // Boundary states are stored in surface-to-bottom order:
    // boundary[0] = surface, boundary[N] = top of the halfspace.
    const int nLayers = static_cast<int>(layers.size());
    std::vector<Eigen::Matrix<CT, 2, 1>> boundary(nLayers + 1);
    boundary[nLayers](0) = T(1.0);
    boundary[nLayers](1) = -(c44_c * nu);

    auto complex_magnitude = [](const CT& value) {
        return std::hypot(
            get_magnitude(value.real()), get_magnitude(value.imag()));
    };
    auto normalize_boundary = [&](Eigen::Matrix<CT, 2, 1> state) {
        double magnitude = std::max(
            complex_magnitude(state(0)), complex_magnitude(state(1)));
        if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
            throw std::runtime_error("Love upward boundary state became non-finite");
        }
        return state / T(magnitude);
    };

    // Stable upward pass. In an evanescent layer divide the transfer by
    // cosh(nu*h), as in DISPER80 LOVMRX; only the state ratio is needed for
    // matching the physical downward solution.
    for (int i = nLayers - 1; i >= 0; --i) {
        const auto& p = layers[i];
        T h = thicknesses[i];
        T nu_sq = (p.c66 * k * k - p.rho * omega * omega) / p.c44;
        // Coefficients are forced into concrete T before toComplex: an
        // AutoDiffScalar expression yields std::complex<expression>, which will
        // not multiply with std::complex<T>.
        Eigen::Matrix<CT, 2, 1> next;
        if (get_val(nu_sq) > 1e-10) {
            const T nu_r = sqrt(nu_sq);
            const T eta = p.c44 * nu_r;
            // tanh is bounded, so it is safe to differentiate directly; Eigen's
            // AutoDiffScalar provides it. This used to read
            //   T th = T(std::tanh(get_val(nu_r * h)))
            // which discarded the dependence of tanh(nu*h) on the model and so
            // returned correct values with silently wrong derivatives under AD.
            const T th = tanh(nu_r * h);
            const T a = th / eta;
            const T b = eta * th;
            next(0) = boundary[i + 1](0) - toComplex(a) * boundary[i + 1](1);
            next(1) = -toComplex(b) * boundary[i + 1](0) + boundary[i + 1](1);
        } else if (get_val(nu_sq) < -1e-10) {
            const T mu = sqrt(-nu_sq);
            const T x = mu * h;
            const T co = cos(x), si = sin(x), eta = p.c44 * mu;
            const T a = si / eta;
            const T b = eta * si;
            next(0) = toComplex(co) * boundary[i + 1](0)
                    - toComplex(a) * boundary[i + 1](1);
            next(1) = toComplex(b) * boundary[i + 1](0)
                    + toComplex(co) * boundary[i + 1](1);
        } else {
            const T a = h / p.c44;
            next(0) = boundary[i + 1](0) - toComplex(a) * boundary[i + 1](1);
            next(1) = boundary[i + 1](1);
        }
        boundary[i] = normalize_boundary(next);
    }

    // Downward pass from the surface boundary condition. At every original
    // layer boundary reset traction from the upward state ratio, suppressing
    // roundoff excitation of the non-physical growing solution.
    Eigen::Matrix<CT, 2, 1> state;
    state(0) = T(1.0);
    state(1) = T(0.0);
    T currentDepth = T(0.0);
    results.push_back({currentDepth, state(0), -state(1)});

    for (int i = 0; i < nLayers; ++i) {
        const auto& p = layers[i];
        T thickness = thicknesses[i];
        T nu_sq = (p.c66 * k * k - p.rho * omega * omega) / p.c44;
        int adaptive_sub_layers = numSubLayers;
        if (get_val(nu_sq) > 1e-10) {
            T nu_r = sqrt(nu_sq);
            double argument = std::abs(get_val(nu_r * thickness));
            int required_sub_layers = std::max(
                1, static_cast<int>(std::ceil(argument / 40.0)));
            if (required_sub_layers > 4096) {
                throw std::runtime_error(
                    "Love eigenfunction exceeded the adaptive sublayer limit at layer " +
                    std::to_string(i));
            }
            adaptive_sub_layers = std::max(
                adaptive_sub_layers, required_sub_layers);
        }
        T dz = thickness / T(adaptive_sub_layers);
        for (int j = 1; j <= adaptive_sub_layers; ++j) {
            Eigen::Matrix<CT, 2, 1> next;
            if (get_val(nu_sq) > 1e-10) {
                const T nu_r = sqrt(nu_sq);
                const T x = nu_r * dz;
                const T ch = cosh(x), sh = sinh(x), eta = p.c44 * nu_r;
                const T a = sh / eta;
                const T b = eta * sh;
                next(0) = toComplex(ch) * state(0) + toComplex(a) * state(1);
                next(1) = toComplex(b) * state(0) + toComplex(ch) * state(1);
            } else if (get_val(nu_sq) < -1e-10) {
                const T mu = sqrt(-nu_sq);
                const T x = mu * dz;
                const T co = cos(x), si = sin(x), eta = p.c44 * mu;
                const T a = si / eta;
                const T b = eta * si;
                next(0) = toComplex(co) * state(0) + toComplex(a) * state(1);
                next(1) = -toComplex(b) * state(0) + toComplex(co) * state(1);
            } else {
                const T a = dz / p.c44;
                next(0) = state(0) + toComplex(a) * state(1);
                next(1) = state(1);
            }
            state = next;
            currentDepth += dz;
            if (j == adaptive_sub_layers) {
                const CT u_match = boundary[i + 1](0);
                if (complex_magnitude(u_match) > 1e-20) {
                    state(1) = boundary[i + 1](1) / u_match * state(0);
                }
            }
            if (!std::isfinite(complex_magnitude(state(0))) ||
                !std::isfinite(complex_magnitude(state(1)))) {
                throw std::runtime_error(
                    "Love downward eigenfunction became non-finite at layer " +
                    std::to_string(i));
            }
            // Preserve the historical outward traction sign in the API.
            results.push_back({currentDepth, state(0), -state(1)});
        }
    }
    
    return results;
}

// Compute Love Energy Integrals
template <typename T>
EnergyIntegralResult<T> computeLoveEnergyIntegrals(
    const std::vector<LoveEigenfunctionPoint<T>>& eigenfunctions,
    T omega,
    T phaseVelocity,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams
) {
    EnergyIntegralResult<T> res; res.I1 = T(0.0); res.I2 = T(0.0); res.I3 = T(0.0); res.phaseVelocity = phaseVelocity;
    if (eigenfunctions.size() < 2) return res;
    T k = omega / phaseVelocity; T k_sq = k * k;

    for (size_t i = 0; i < eigenfunctions.size() - 1; ++i) {
        const auto& p1 = eigenfunctions[i]; const auto& p2 = eigenfunctions[i+1];
        T dz = p2.depth - p1.depth; T z_mid = (p1.depth + p2.depth) * 0.5;
        PhysicalParameters<T> p = halfSpaceParams; T dep = 0.0;
        for(size_t l=0; l<layers.size(); ++l) { if(z_mid < dep+thicknesses[l]) { p=layers[l]; break; } dep+=thicknesses[l]; }
        auto integrands = [&](const LoveEigenfunctionPoint<T>& pt) {
            T uy = pt.uy.real(); T tyz = pt.tyz.real();
            T i1 = p.rho * uy * uy; 
            T i2 = (tyz*tyz/p.c44) + k_sq * p.c66 * uy*uy; 
            T i3 = p.c66 * uy * uy;
            return std::vector<T>{i1, i2, i3};
        };
        auto v1 = integrands(p1); auto v2 = integrands(p2);
        res.I1 += (v1[0] + v2[0]) * 0.5 * dz; res.I2 += (v1[1] + v2[1]) * 0.5 * dz; res.I3 += (v1[2] + v2[2]) * 0.5 * dz;
    }
    
    // Half-space (Analytic)
    T L = halfSpaceParams.c44; T N = halfSpaceParams.c66; T rho = halfSpaceParams.rho;
    // Force the argument into a concrete T: toComplex does `T zero = val*0.0`,
    // which fails when val is an AutoDiffScalar expression template.
    const T num_r = N * k_sq - rho * omega * omega;
    std::complex<T> num = toComplex(num_r);
    std::complex<T> nu = safe_complex_sqrt(num / toComplex(L)); 
    // Decay condition: Re(nu) > 0.
    if (get_val(nu.real()) < 0) nu = -nu;
    
    T uy_0 = eigenfunctions.back().uy.real(); 
    
    if (get_val(nu.real()) > 1e-10) {
        T inv_decay = 1.0 / (2.0 * nu.real());
        res.I1 += rho * uy_0 * uy_0 * inv_decay;
        res.I2 += (uy_0 * uy_0 * (L * std::norm(nu) ) + k_sq * N * uy_0 * uy_0) * inv_decay;
        res.I3 += N * uy_0 * uy_0 * inv_decay;
    }

    // get_magnitude, not std::abs: the latter has no AutoDiffScalar overload.
    if (get_magnitude(res.I1) > 1e-20) res.groupVelocity_Energy = res.I3 / (phaseVelocity * res.I1);
    return res;
}


// Compute Initial Vector for SH Halfspace (Decaying Mode)
template <typename T>
Eigen::Matrix<T, 2, 1> computeInitialVectorSH(
    const PhysicalParameters<T>& halfSpace,
    const WaveVariables<T>& wave
) {
    using CT = typename ComplexType<T>::type;
    
    // nu^2 = (c66 k^2 - rho w^2) / c44
    // Condition for decay: Re(nu) > 0.
    
    CT c44_c = toComplex(halfSpace.c44);
    CT c66_c = toComplex(halfSpace.c66);
    CT rho_c = toComplex(halfSpace.rho);
    CT k_c   = toComplex(wave.k);
    CT w_c   = toComplex(wave.omega);
    
    CT num = c66_c * k_c * k_c - rho_c * w_c * w_c;
    CT nu_sq = num / c44_c;
    CT nu = safe_complex_sqrt(nu_sq); // Polar form sqrt for AD compatibility
    
    // Radiative Check
    if (get_val(nu.real()) <= 1e-8) {
        // Not decaying (propagating or growing). return NaN to signal skip.
        Eigen::Matrix<T, 2, 1> Y_nan;
        Y_nan.setConstant(make_nan<T>());
        return Y_nan;
    }
    
    CT eta = c44_c * nu;
    
    // Y = [u, tau]^T
    // For decay e^{-nu z}, tau = -eta * u
    // Y0 = [1, -eta]^T
    
    Eigen::Matrix<T, 2, 1> Y0;
    Y0(0) = T(1.0); // u
    Y0(1) = -eta.real(); // tau
    
    return Y0;
}

// Love Wave Logic Implementation
// intermediateStates: if non-null, saves [u_y, τ_yz] at each layer boundary (bottom-up order).
//   states[0] = halfspace initial, states[1] = after bottom layer, ..., states[N] = surface.
template <typename T>
Eigen::Matrix<T, 5, 1> calculateLoveDispersion(
     const std::vector<PhysicalParameters<T>>& layers,
     const std::vector<T>& thicknesses,
     const PhysicalParameters<T>& halfSpace,
     T omega,
     T phaseVelocity,
     NormalizationMode normMode = NormalizationMode::None,
     std::vector<double>* sFactors = nullptr,
     size_t* sIdx = nullptr,
     std::vector<Eigen::Matrix<T, 2, 1>>* intermediateStates = nullptr
) {
    ++disper26_stats::dispersion_evals_total;
    if constexpr (std::is_same_v<T, double>) ++disper26_stats::dispersion_evals_double;

    T k = omega / phaseVelocity;
    
    WaveVariables<T> wave;
    wave.omega = omega;
    wave.k = k;
    wave.z = make_zero_like(omega);

    // 1. Initial Vector (Half Space)
    // Y [2x1] = [Displacement, Stress]^T
    Eigen::Matrix<T, 2, 1> Y = computeInitialVectorSH(halfSpace, wave);
    
    // Check for radiative failure (NaN)
    if (std::isnan(get_magnitude(Y(0)))) {
         Eigen::Matrix<T, 5, 1> Y_nan;
         Y_nan.setConstant(make_nan<T>());
         return Y_nan;
    }

    if (intermediateStates) {
        intermediateStates->clear();
        intermediateStates->push_back(Y);  // halfspace initial
    }

    // 2. Propagate Upward
    for (int i = layers.size() - 1; i >= 0; --i) {
        wave.z = -thicknesses[i]; // Upward
        Eigen::Matrix<T, 2, 2> A = computeLayerMatrixSH(layers[i], wave);
        Y = A * Y;
        
        // --- Scaling / Normalization ---
        if (sFactors) {
            double scale = 1.0;
            if (normMode == NormalizationMode::Provided && sIdx) {
                if (*sIdx < sFactors->size()) scale = (*sFactors)[(*sIdx)++];
            } else {
                double m = get_magnitude(Y(0)) + get_magnitude(Y(1));
                if (m > 1e-20) scale = 1.0 / m;
                if (!sIdx) sFactors->push_back(scale);
            }
            Y = Y * T(scale);
        } else if (normMode == NormalizationMode::VectorNorm) {
            using std::sqrt;
            T norm = sqrt(Y(0)*Y(0) + Y(1)*Y(1));
            if (get_magnitude(norm) > 1e-20) Y = Y / norm;
        } else if (normMode == NormalizationMode::Vertical) {
            T val = Y(0);
            if (get_magnitude(val) > 1e-20) Y = Y / val;
        } else {
            // NormalizationMode::None: overflow guard only
            double max_val = 0.0;
            for(int m=0; m<2; ++m) {
                double v = get_magnitude(Y(m));
                if(v > max_val) max_val = v;
            }
            const double THRESHOLD = 1e10;
            if (max_val > THRESHOLD) Y = Y * T(1.0 / max_val);
        }

        if (intermediateStates) {
            intermediateStates->push_back(Y);  // state after passing through layer i
        }
    }
    
    // 3. Map to 5x1 Vector Result
    // Constraint is Stress = 0 at surface.
    // In Rayleigh vector (Dunkin), I24 (Index 4) is used as the characteristic function.
    // Map Stress to I24.
    
    Eigen::Matrix<T, 5, 1> Y_full;
    Y_full.setZero();
    
    Y_full(I12) = Y(0); // Displacement (u_y)
    Y_full(I24) = Y(1); // Stress (tau_yz) -> zero-finding target
    
    return Y_full;
}

// Wrapper to dispatch based on WaveType
template <typename T>
Eigen::Matrix<T, 5, 1> calculateDispersionFunction(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity,
    WaveType type = WaveType::Rayleigh,
    NormalizationMode normMode = NormalizationMode::None,
    Eigen::Matrix<T, 5, 1>* seabedState = nullptr,
    std::vector<double>* sFactors = nullptr,
    size_t* sIdx = nullptr
) {
    if (type == WaveType::Rayleigh) {
        return calculateRayleighDispersion(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, normMode, seabedState, sFactors, sIdx);
    } else {
        return calculateLoveDispersion(layers, thicknesses, halfSpaceParams, omega, phaseVelocity, normMode, sFactors, sIdx);
    }
}

inline double calculateDispersionPhaseValueDouble(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity,
    WaveType type
) {
    return calculateDispersionFunction(
        layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type)(I24);
}

// --- Mode Counting Logic (Phase Unwrapping) ---

// Helper to normalize angle to (-PI, PI]
template <typename T>
T normalizePhase(T angle) {
    while (angle <= -M_PI) angle += 2.0 * M_PI;
    while (angle > M_PI) angle -= 2.0 * M_PI;
    return angle;
}

// --- Numerical Derivative Helper ---
template <typename T>
Eigen::Matrix<T, 5, 1> computeNumericalDerivative(
    const PhysicalParameters<T>& phys,
    const WaveVariables<T>& wave,
    const Eigen::Matrix<T, 5, 1>& Y0,
    T epsilon = 1e-5
) {
    // d/dz (B(z) * Y0) approx (B(z+h)Y0 - B(z-h)Y0) / 2h
    // But computeLayerMatrix takes total thickness z.
    // If we want derivative at local z, we compute B(z+h) and B(z-h).
    
    WaveVariables<T> w_plus = wave;
    w_plus.z = wave.z + epsilon;
    
    WaveVariables<T> w_minus = wave;
    w_minus.z = wave.z - epsilon;
    
    Eigen::Matrix<T, 5, 5> B_plus = computeLayerMatrix(phys, w_plus);
    Eigen::Matrix<T, 5, 5> B_minus = computeLayerMatrix(phys, w_minus);
    
    Eigen::Matrix<T, 5, 1> Y_plus = B_plus * Y0;
    Eigen::Matrix<T, 5, 1> Y_minus = B_minus * Y0;
    
    return (Y_plus - Y_minus) / (2.0 * epsilon);
}

// --- Mineos-style Signed Zero Crossing Counter (Cubic Hermite) ---
// Returns change in mode count (+1, -1, or 0)
// y1, y2: Value of Secular Function at start/end
// dy1, dy2: Derivative of Secular Function
// c1, c2: Value of Conjugate Variable
// dc1, dc2: Derivative of Conjugate Variable
template <typename T>
int countSignedZeroCrossings(T y1, T y2, T dy1, T dy2, T c1, T c2, T dc1, T dc2, T step_len) {
    // Map to t in [0, 1]
    // P(t) is cubic.
    // Coefficients for P(t) = a0 + a1 t + a2 t^2 + a3 t^3
    // P(0) = y1, P(1) = y2
    // P'(0) = dy1 * h, P'(1) = dy2 * h
    
    T h = step_len;
    T Y1p = dy1 * h;
    T Y2p = dy2 * h; // Derivatives w.r.t t
    
    T a0 = y1;
    T a1 = Y1p;
    T a2 = -3.0*y1 + 3.0*y2 - 2.0*Y1p - Y2p;
    T a3 = 2.0*y1 - 2.0*y2 + Y1p + Y2p;
    
    // Find roots of a3 t^3 + a2 t^2 + a1 t + a0 = 0 in [0, 1]
    // Use analytic cubic formula or Newton. Since a3 can be small, careful.
    // Mineos uses quadratic approximation if cubic term small?
    // Let's use a robust Newton/Bisection for P(t).
    
    // Find roots. There can be up to 3 roots.
    // Simple scan + bisection is robust enough for small interval.
    // Scan 10 points.
    std::vector<T> roots;
    T t_prev = 0.0;
    T val_prev = a0;
    
    int n_scan = 20;
    for(int i=1; i<=n_scan; ++i) {
        T t_curr = static_cast<T>(i) / static_cast<T>(n_scan);
        T val_curr = a0 + t_curr*(a1 + t_curr*(a2 + t_curr*a3));
        
        if (val_prev * val_curr <= 0.0 && val_prev != val_curr) {
            // Root bracketed
            T ta = t_prev;
            T tb = t_curr;
            // Bisection
            for(int k=0; k<10; ++k) {
                T tm = 0.5 * (ta + tb);
                T vm = a0 + tm*(a1 + tm*(a2 + tm*a3));
                if (val_prev * vm <= 0.0) {
                    tb = tm;
                } else {
                    ta = tm;
                    val_prev = vm; // redundant conceptually but ok
                }
            }
            roots.push_back(0.5*(ta+tb));
        }
        t_prev = t_curr;
        val_prev = val_curr;
    }
    
    int count_change = 0;
    
    // Conjugate interpolation coefficients (C(t))
    T C1p = dc1 * h;
    T C2p = dc2 * h;
    T c_a0 = c1;
    T c_a1 = C1p;
    T c_a2 = -3.0*c1 + 3.0*c2 - 2.0*C1p - C2p;
    T c_a3 = 2.0*c1 - 2.0*c2 + C1p + C2p;
    
    for (T t : roots) {
        // Evaluate Slope of P(t) at root: P'(t) / h = dY/dz
        T P_slope_t = a1 + t*(2.0*a2 + t*3.0*a3); 
        // Actual slope dY/dz = P_slope_t / h.
        
        // Evaluate Conjugate C(t)
        T Conj_val = c_a0 + t*(c_a1 + t*(c_a2 + t*c_a3));
        
        // Check Sign: Conj * Slope
        // Mineos: if (Conj * Slope < 0) kount++; if (...) kount--;
        // Actually Mineos logic varies. 
        // Logic: d(Phase)/dz > 0? 
        // Phase ~ atan(Y/C). 
        // Near Y=0, Phase ~ Y/C. dPhase ~ (Y'C - YC')/C^2 ~ Y'C.
        // So direction is sign(Y' * C).
        // Since we count "modes", and modes usually mean Phase DECREASES with frequency/depth?
        // Mineos counts "kount".
        T prod = P_slope_t * Conj_val;
        if (prod > 0) {
           // Standard direction?
           // Need to match Disper26 direction.
           // Mode decreases as c decreases.
           // Let's assume +1 first.
           count_change += 1;
        } else if (prod < 0) {
           count_change -= 1;
        }
    }
    
    return count_change;
}

// 1. Rayleigh Wave Layer Mode Count
// Uses Woodhouse (1988) phase definition: θ = atan2(I24, -(I23+I14))
// Tracks phase accumulation through the layer with adaptive subdivision
template <typename T>
std::pair<int, T> countModesLayerRayleigh(
    const Eigen::Matrix<T, 5, 1>& Y0_bot,
    const Eigen::Matrix<T, 5, 1>& Y0_top,
    T omega,
    T k,
    const PhysicalParameters<T>& layerParams,
    T thickness
) {
    // Phase Estimate (P_est)
    // r_beta = sqrt((omega/beta_v)^2 - k^2)
    // beta_v = sqrt(c44 / rho)
    T rho = get_val(layerParams.rho);
    T c44 = get_val(layerParams.c44);
    
    T beta_v_sq = (rho > 1e-10) ? c44 / rho : 0.0;
    
    T r_beta_sq = 0.0;
    if (beta_v_sq > 1e-10) {
        r_beta_sq = (omega * omega) / beta_v_sq - k * k;
    }

    // Adaptive Subdivision based on expected phase change
    // Enhanced for accuracy near critical angle (small ν)
    int num_steps = 10;  // Minimum steps per layer for accuracy
    
    if (r_beta_sq > 0.0) {
        // Propagating: use ν to estimate phase change
        T nu = std::sqrt(r_beta_sq);
        T P_est_total = nu * thickness;
        T P_est_total_mag = std::abs(P_est_total);
        
        // Standard fine subdivision
        int steps_by_phase = 1;
        if (P_est_total_mag > 0.02) { 
            steps_by_phase = static_cast<int>(P_est_total_mag / 0.02) + 1;
        }
        
        // Extra subdivision when ν is small (near critical angle)
        // When ν < 1, the phase behavior can be sensitive
        int steps_by_nu = 1;
        if (nu < 1.0) {
            // More steps when wave is nearly horizontal
            steps_by_nu = static_cast<int>(10.0 / (nu + 0.1));
        }
        
        num_steps = std::max({num_steps, steps_by_phase, steps_by_nu});
    } else {
        // Evanescent: use decay length to subdivide
        // Phase still rotates in (I24, -(I23+I14)) plane even when evanescent
        T decay_rate = std::sqrt(-r_beta_sq);  // Imaginary part of vertical wavenumber
        T decay_length = decay_rate * thickness;
        
        int steps_by_decay = 1;
        if (decay_length > 0.1) {
            steps_by_decay = static_cast<int>(decay_length / 0.1) + 1;
        }
        
        // Extra subdivision when κ is small (near radiative)
        int steps_by_kappa = 1;
        if (decay_rate < 1.0) {
            steps_by_kappa = static_cast<int>(10.0 / (decay_rate + 0.1));
        }
        
        num_steps = std::max({num_steps, steps_by_decay, steps_by_kappa});
    }

    T sub_d = thickness / static_cast<T>(num_steps);
    
    WaveVariables<T> sub_wave;
    sub_wave.omega = omega;
    sub_wave.k = k;
    sub_wave.z = sub_d;
    Eigen::Matrix<T, 5, 5> B_sub = computeLayerMatrix(layerParams, sub_wave);

    Eigen::Matrix<T, 5, 1> Y_curr = Y0_bot;
    int total_N = 0; 
    T total_delta = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        Eigen::Matrix<T, 5, 1> Y_next = B_sub * Y_curr;
        
        // Woodhouse (1988) equation 5.32 with corrected row ordering (1,3,2,4):
        // θ = -1/π * arccot((m4-m3)/m6)
        // Woodhouse uses physical ordering: (U, σxz, V, σzz) = components (1,3,2,4)
        // m3 = det(rows 1,4) = det(comp 1,4) = I14
        // m4 = det(rows 2,3) = det(comp 3,2) = -I23 (sign from permutation)
        // m6 = det(rows 3,4) = det(comp 2,4) = I24 (secular function)
        // m4-m3 = -I23 - I14 = -(I23+I14)
        // Phase = atan2(m6, m4-m3) = atan2(I24, -(I23+I14))
        
        T y_b = get_val(Y_curr(I24));  // m6 = secular function
        T x_b = -(get_val(Y_curr(I23)) + get_val(Y_curr(I14)));  // m4-m3 = -(I23+I14)
        T th_b = std::atan2(y_b, x_b);
        
        T y_t = get_val(Y_next(I24));  // m6
        T x_t = -(get_val(Y_next(I23)) + get_val(Y_next(I14)));  // m4-m3
        T th_t = std::atan2(y_t, x_t);
        
        // Simple phase difference with normalization to [-PI, PI]
        T d_raw = normalizePhase(th_t - th_b);
        
        total_delta += d_raw;
        
        // Update for next step
        Y_curr = Y_next;
        
        // Scaling
        double max_val = 0.0;
        for(int m=0; m<5; ++m) {
             double v = get_magnitude(Y_curr(m));
             if(v > max_val) max_val = v;
        }
        if (max_val > 1e10) {
             T scale = 1.0/max_val;
             Y_curr = Y_curr * scale;
        }
    }

    // Return: (integer mode count, continuous phase)
    // Integer count = floor(|total_delta| / PI)
    total_N = static_cast<int>(std::floor(std::abs(total_delta) / M_PI));
    return {total_N, total_delta};
}


// 2. Liquid Layer Mode Count (Rayleigh in Ocean)
// Uses phase θ = atan2(P, w) for [w, P]^T state vector
// P = 0 at free surface is the secular condition
template <typename T>
std::pair<int, T> countModesLayerLiquid(
    const Eigen::Matrix<T, 2, 1>& Z_bot,
    const Eigen::Matrix<T, 2, 1>& Z_top,
    T omega,
    T k,
    const PhysicalParameters<T>& layerParams,
    T thickness
) {
    int total_N = 0;
    T total_delta = 0.0;
    
    // For liquid layer, the phase is θ = atan2(P, w)
    // Zero crossing of P (with w as reference) defines modes
    
    // Number of sub-layers for phase tracking
    int n_sublayers = std::max(1, static_cast<int>(std::ceil(get_val(thickness) * get_val(k) / 0.5)));
    if (n_sublayers > 100) n_sublayers = 100;
    
    T dz = thickness / static_cast<T>(n_sublayers);
    
    WaveVariables<T> wave;
    wave.omega = omega;
    wave.k = k;
    
    Eigen::Matrix<T, 2, 1> Z_curr = Z_bot;
    T prev_phase = std::atan2(get_val(Z_curr(1)), get_val(Z_curr(0)));
    
    for (int sub = 0; sub < n_sublayers; ++sub) {
        wave.z = dz;
        Eigen::Matrix<T, 2, 2> B = computeLayerMatrixLiquid(layerParams, wave);
        Z_curr = B * Z_curr;
        
        T curr_phase = std::atan2(get_val(Z_curr(1)), get_val(Z_curr(0)));
        
        // Unwrap phase
        T delta = curr_phase - prev_phase;
        if (delta > M_PI) delta -= 2.0 * M_PI;
        if (delta < -M_PI) delta += 2.0 * M_PI;
        
        total_delta += delta;
        prev_phase = curr_phase;
        
        // Scaling
        double max_val = std::max(get_magnitude(Z_curr(0)), get_magnitude(Z_curr(1)));
        if (max_val > 1e10) {
            Z_curr = Z_curr / max_val;
        }
    }
    
    total_N = static_cast<int>(std::floor(std::abs(total_delta) / M_PI));
    return {total_N, total_delta};
}


// 2. Love Wave Layer Mode Count
template <typename T>
std::pair<int, T> countModesLayerLove(
    const Eigen::Matrix<T, 2, 1>& y_bot,
    const Eigen::Matrix<T, 2, 1>& y_top,
    T omega,
    T k,
    const PhysicalParameters<T>& layerParams,
    T thickness
) {
    // Phase Estimate (P_est)
    T rho = get_val(layerParams.rho);
    T c44 = get_val(layerParams.c44);
    T c66 = get_val(layerParams.c66);

    T num = rho * omega * omega - c66 * k * k;
    T r_sh_sq = 0.0; // Check c44 > 0
    if (c44 > 1e-10) {
        r_sh_sq = num / c44;
    }

    // Only count phase in propagating layers
    if (r_sh_sq <= 0.0) {
        return {0, 0.0};
    }

    T P_est_total = 0.0;
    if (r_sh_sq > 0.0) {
        P_est_total = std::sqrt(r_sh_sq) * thickness;
    }
    
    // Adaptive Subdivision
    int num_steps = 1;
    if (P_est_total > 0.2) { 
       num_steps = static_cast<int>(std::ceil(P_est_total / 0.2));
    }

    T sub_thickness = thickness / static_cast<T>(num_steps);
    
    // Compute Propagator ONCE
    WaveVariables<T> sub_wave;
    sub_wave.omega = omega;
    sub_wave.k = k;
    sub_wave.z = sub_thickness;
    Eigen::Matrix<T, 2, 2> B_sub = computeLayerMatrixSH(layerParams, sub_wave);
    
    Eigen::Matrix<T, 2, 1> Y_curr = y_bot;
    int total_N = 0;
    T total_delta_true = 0.0;
    
    for (int step = 0; step < num_steps; ++step) {
        Eigen::Matrix<T, 2, 1> Y_next = B_sub * Y_curr;
        
        T th_b = std::atan2(get_val(Y_curr(1)), get_val(Y_curr(0)));
        T th_t = std::atan2(get_val(Y_next(1)), get_val(Y_next(0)));
        T d_raw = normalizePhase(th_t - th_b);
        
        // Since step is small (P_est < 0.2), phase change is guaranteed < PI.
        int n_sub = 0;
        T d_true = d_raw;
        
        total_N += n_sub;
        total_delta_true += d_true;
        
        Y_curr = Y_next;
        
        // Scaling
        double max_val = std::max(get_magnitude(Y_curr(0)), get_magnitude(Y_curr(1)));
        if (max_val > 1e10) {
            Y_curr = Y_curr * (1.0/max_val);
        }
    }

    return {total_N, total_delta_true};
}

struct BranCrossingInfo {
    int layer_index = -1;
    // Cumulative height above the halfspace top (negative inside the virtual
    // halfspace segment); a single global coordinate so crossings from
    // different layers can be compared and deduplicated.
    double local_depth_km = 0.0;
    int contribution = 0;
    double secular = 0.0;
    double secular_derivative = 0.0;
    double conjugate = 0.0;
    double test_value = 0.0;
};

struct BranModeCountResult {
    int n_integer = -1;
    int signed_count = 0;
    int calibrated_signed_count = 0;
    bool unsupported_liquid = false;
    bool unsupported_love = false;
    bool nan_or_radiative = false;
    std::vector<BranCrossingInfo> crossings;
};

inline double eval_cubic(double a0, double a1, double a2, double a3, double t) {
    return a0 + t * (a1 + t * (a2 + t * a3));
}

inline double eval_cubic_derivative(double a1, double a2, double a3, double t) {
    return a1 + t * (2.0 * a2 + t * 3.0 * a3);
}

void add_unique_unit_interval_point(std::vector<double>& points, double t) {
    constexpr double eps = 1.0e-12;
    if (t < -eps || t > 1.0 + eps) return;
    t = std::min(1.0, std::max(0.0, t));
    for (double existing : points) {
        if (std::abs(existing - t) < 1.0e-10) return;
    }
    points.push_back(t);
}

int countBranHermiteCrossings(
    double y1,
    double y2,
    double dy1,
    double dy2,
    double c1,
    double c2,
    double dc1,
    double dc2,
    double step_len,
    int layer_index,
    double local_depth_start,
    double* last_crossing_depth,
    std::vector<BranCrossingInfo>* crossings
) {
    if (step_len <= 0.0) return 0;

    const double h = step_len;
    const double a0 = y1;
    const double a1 = h * dy1;
    const double a2 = -h * (2.0 * dy1 + dy2) + 3.0 * (y2 - y1);
    const double a3 = h * (dy1 + dy2) - 2.0 * (y2 - y1);

    const double b0 = c1;
    const double b1 = h * dc1;
    const double b2 = -h * (2.0 * dc1 + dc2) + 3.0 * (c2 - c1);
    const double b3 = h * (dc1 + dc2) - 2.0 * (c2 - c1);

    std::vector<double> points{0.0, 1.0};
    const double qa = 3.0 * a3;
    const double qb = 2.0 * a2;
    const double qc = a1;
    if (std::abs(qa) > 1.0e-30) {
        const double disc = qb * qb - 4.0 * qa * qc;
        if (disc >= 0.0) {
            const double sqrt_disc = std::sqrt(disc);
            add_unique_unit_interval_point(points, (-qb + sqrt_disc) / (2.0 * qa));
            add_unique_unit_interval_point(points, (-qb - sqrt_disc) / (2.0 * qa));
        }
    } else if (std::abs(qb) > 1.0e-30) {
        add_unique_unit_interval_point(points, -qc / qb);
    }
    std::sort(points.begin(), points.end());

    int signed_count = 0;
    constexpr double value_eps = 1.0e-14;
    constexpr double depth_eps = 1.0e-8;

    for (size_t i = 1; i < points.size(); ++i) {
        const double left = points[i - 1];
        const double right = points[i];
        const double v_left = eval_cubic(a0, a1, a2, a3, left);
        const double v_right = eval_cubic(a0, a1, a2, a3, right);

        bool bracketed = (v_left == 0.0 || v_right == 0.0 || v_left * v_right <= 0.0);
        if (!bracketed) continue;

        double root = 0.5 * (left + right);
        if (std::abs(v_left) < value_eps) {
            root = left;
        } else if (std::abs(v_right) < value_eps) {
            root = right;
        } else {
            double lo = left;
            double hi = right;
            double vlo = v_left;
            for (int iter = 0; iter < 30; ++iter) {
                const double vmid = eval_cubic(a0, a1, a2, a3, root);
                const double dmid = eval_cubic_derivative(a1, a2, a3, root);
                double candidate = root;
                if (std::abs(dmid) > 1.0e-30) {
                    candidate = root - vmid / dmid;
                }
                if (candidate <= lo || candidate >= hi || !std::isfinite(candidate)) {
                    candidate = 0.5 * (lo + hi);
                }
                const double vcand = eval_cubic(a0, a1, a2, a3, candidate);
                if (std::abs(vcand) < value_eps) {
                    root = candidate;
                    break;
                }
                if (vlo * vcand <= 0.0) {
                    hi = candidate;
                } else {
                    lo = candidate;
                    vlo = vcand;
                }
                root = candidate;
            }
        }

        const double local_depth = local_depth_start + root * h;
        if (last_crossing_depth && std::abs(local_depth - *last_crossing_depth) < depth_eps) {
            continue;
        }

        const double yprime_t = eval_cubic_derivative(a1, a2, a3, root);
        const double conjugate = eval_cubic(b0, b1, b2, b3, root);
        const double test_value = conjugate * yprime_t;
        int contribution = 0;
        if (test_value < 0.0) contribution = 1;
        if (test_value > 0.0) contribution = -1;
        if (contribution == 0) continue;

        signed_count += contribution;
        if (last_crossing_depth) *last_crossing_depth = local_depth;
        if (crossings) {
            BranCrossingInfo info;
            info.layer_index = layer_index;
            info.local_depth_km = local_depth;
            info.contribution = contribution;
            info.secular = eval_cubic(a0, a1, a2, a3, root);
            info.secular_derivative = yprime_t / h;
            info.conjugate = conjugate;
            info.test_value = test_value;
            crossings->push_back(info);
        }
    }

    return signed_count;
}

int estimateRayleighBranSubsteps(
    const PhysicalParameters<double>& layerParams,
    double omega,
    double k,
    double thickness
) {
    double rho = get_val(layerParams.rho);
    double c44 = get_val(layerParams.c44);
    double beta_v_sq = (rho > 1e-10) ? c44 / rho : 0.0;
    double r_beta_sq = 0.0;
    if (beta_v_sq > 1e-10) {
        r_beta_sq = (omega * omega) / beta_v_sq - k * k;
    }

    int num_steps = 10;
    if (r_beta_sq > 0.0) {
        double nu = std::sqrt(r_beta_sq);
        double phase = std::abs(nu * thickness);
        int steps_by_phase = (phase > 0.02) ? static_cast<int>(phase / 0.02) + 1 : 1;
        int steps_by_nu = (nu < 1.0) ? static_cast<int>(10.0 / (nu + 0.1)) : 1;
        num_steps = std::max({num_steps, steps_by_phase, steps_by_nu});
    } else {
        double decay_rate = std::sqrt(std::max(0.0, -r_beta_sq));
        double decay_length = decay_rate * thickness;
        int steps_by_decay = (decay_length > 0.1) ? static_cast<int>(decay_length / 0.1) + 1 : 1;
        int steps_by_kappa = (decay_rate < 1.0) ? static_cast<int>(10.0 / (decay_rate + 0.1)) : 1;
        num_steps = std::max({num_steps, steps_by_decay, steps_by_kappa});
    }
    return std::max(1, num_steps);
}

int estimateLiquidBranSubsteps(
    const PhysicalParameters<double>& layerParams,
    double omega,
    double k,
    double thickness
) {
    double rho = get_val(layerParams.rho);
    double c33 = get_val(layerParams.c33);
    double alpha_sq = (rho > 1.0e-10) ? c33 / rho : 0.0;
    double r_acoustic_sq = 0.0;
    if (alpha_sq > 1.0e-10) {
        r_acoustic_sq = (omega * omega) / alpha_sq - k * k;
    }

    int num_steps = 10;
    if (r_acoustic_sq > 0.0) {
        double nu = std::sqrt(r_acoustic_sq);
        double phase = std::abs(nu * thickness);
        int steps_by_phase = (phase > 0.02) ? static_cast<int>(phase / 0.02) + 1 : 1;
        int steps_by_nu = (nu < 1.0) ? static_cast<int>(10.0 / (nu + 0.1)) : 1;
        num_steps = std::max({num_steps, steps_by_phase, steps_by_nu});
    } else {
        double decay_rate = std::sqrt(std::max(0.0, -r_acoustic_sq));
        double decay_length = decay_rate * thickness;
        int steps_by_decay = (decay_length > 0.1) ? static_cast<int>(decay_length / 0.1) + 1 : 1;
        int steps_by_kappa = (decay_rate < 1.0) ? static_cast<int>(10.0 / (decay_rate + 0.1)) : 1;
        num_steps = std::max({num_steps, steps_by_decay, steps_by_kappa});
    }
    return std::max(1, num_steps);
}

int estimateLoveBranSubsteps(
    const PhysicalParameters<double>& layerParams,
    double omega,
    double k,
    double thickness
) {
    double rho = get_val(layerParams.rho);
    double c44 = get_val(layerParams.c44);
    double c66 = get_val(layerParams.c66);
    double r_sh_sq = 0.0;
    if (c44 > 1.0e-10) {
        r_sh_sq = (rho * omega * omega - c66 * k * k) / c44;
    }

    int num_steps = 10;
    if (r_sh_sq > 0.0) {
        double nu = std::sqrt(r_sh_sq);
        double phase = std::abs(nu * thickness);
        int steps_by_phase = (phase > 0.02) ? static_cast<int>(phase / 0.02) + 1 : 1;
        int steps_by_nu = (nu < 1.0) ? static_cast<int>(10.0 / (nu + 0.1)) : 1;
        num_steps = std::max({num_steps, steps_by_phase, steps_by_nu});
    } else {
        double decay_rate = std::sqrt(std::max(0.0, -r_sh_sq));
        double decay_length = decay_rate * thickness;
        int steps_by_decay = (decay_length > 0.1) ? static_cast<int>(decay_length / 0.1) + 1 : 1;
        int steps_by_kappa = (decay_rate < 1.0) ? static_cast<int>(10.0 / (decay_rate + 0.1)) : 1;
        num_steps = std::max({num_steps, steps_by_decay, steps_by_kappa});
    }
    return std::max(1, num_steps);
}

BranModeCountResult calculateBranModeCountRayleighSolid(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity
) {
    BranModeCountResult result;
    if (layers.size() != thicknesses.size()) {
        result.nan_or_radiative = true;
        return result;
    }
    if (isLiquidLayer(halfSpaceParams)) {
        result.unsupported_liquid = true;
        return result;
    }

    const double k = omega / phaseVelocity;
    WaveVariables<double> wave;
    wave.omega = omega;
    wave.k = k;
    wave.z = 0.0;
    Eigen::Matrix<double, 5, 1> Y = computeInitialVector(halfSpaceParams, wave);
    for (int i = 0; i < 5; ++i) {
        if (!std::isfinite(Y(i))) {
            result.nan_or_radiative = true;
            return result;
        }
    }

    using AD1 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;
    auto make_ad = [](double value, double deriv = 0.0) {
        AD1 out;
        out.value() = value;
        out.derivatives()(0) = deriv;
        return out;
    };
    auto make_ad_params = [&](const PhysicalParameters<double>& p) {
        PhysicalParameters<AD1> out;
        out.rho = make_ad(p.rho);
        out.c11 = make_ad(p.c11);
        out.c13 = make_ad(p.c13);
        out.c33 = make_ad(p.c33);
        out.c44 = make_ad(p.c44);
        out.c66 = make_ad(p.c66);
        out.gravity = make_ad(p.gravity);
        out.gravity_internal_scale = make_ad(p.gravity_internal_scale);
        out.gravity_surface_scale = make_ad(p.gravity_surface_scale);
        out.gravity_seabed_scale = make_ad(p.gravity_seabed_scale);
        return out;
    };
    // The layer propagator is the matrix exponential of a z-independent
    // generator, B(z) = exp(A z), so B'(z) = A B(z) and the endpoint derivative
    // needed by the Hermite crossing test is just A applied to the propagated
    // state.  The generator is read off once per layer as the z derivative of
    // B at z = 0, which is where automatic differentiation is still used; every
    // substep endpoint then only needs the plain double propagator and one
    // matrix-vector product.  Taking A from the same computeLayerMatrix call
    // keeps whichever propagator branch the layer selects self-consistent.
    auto make_generator = [&](const PhysicalParameters<double>& layer) {
        PhysicalParameters<AD1> p_ad = make_ad_params(layer);
        WaveVariables<AD1> wave_ad;
        wave_ad.omega = make_ad(omega);
        wave_ad.k = make_ad(k);
        wave_ad.z = make_ad(0.0, 1.0);
        Eigen::Matrix<AD1, 5, 5> B_ad = computeLayerMatrix(p_ad, wave_ad);
        Eigen::Matrix<double, 5, 5> A;
        for (int r = 0; r < 5; ++r)
            for (int c = 0; c < 5; ++c)
                A(r, c) = B_ad(r, c).derivatives()(0);
        return A;
    };
    // Neither B(z) nor A B(z) depends on the state it is applied to, and every
    // substep inside one layer uses the same two depths (0 and dz), so both are
    // loop-invariant.  Build them once per layer; what is left in the substep
    // is a matrix-vector product.
    //
    // Contract with the state last, the way the AD path does: forming the
    // derivative matrix A B(z) first and applying it to Y0 afterwards keeps
    // the near-zero secular component a result of the final cancellation
    // rather than an input that the generator would amplify.  Hoisting keeps
    // that order, so the endpoints come out bit for bit as they did when the
    // matrices were rebuilt every substep -- which the free-surface sign at a
    // supra-Rayleigh root depends on (docs/mode_counting.md).
    struct EndpointOperators {
        Eigen::Matrix<double, 5, 5> value;
        Eigen::Matrix<double, 5, 5> derivative;
    };
    auto build_endpoint = [&](const PhysicalParameters<double>& layer,
                              const Eigen::Matrix<double, 5, 5>& generator, double z) {
        WaveVariables<double> wave_z;
        wave_z.omega = omega;
        wave_z.k = k;
        wave_z.z = z;

        EndpointOperators ops;
        ops.value = computeLayerMatrix(layer, wave_z);
        ops.derivative = generator * ops.value;
        return ops;
    };
    auto apply_endpoint = [](const EndpointOperators& ops,
                             const Eigen::Matrix<double, 5, 1>& Y0) {
        std::pair<Eigen::Matrix<double, 5, 1>, Eigen::Matrix<double, 5, 1>> out;
        out.first = ops.value * Y0;
        out.second = ops.derivative * Y0;
        return out;
    };
    // Same generator argument as the solid propagator, for the [U, P] state.
    auto make_liquid_generator = [&](const PhysicalParameters<double>& layer) {
        PhysicalParameters<AD1> p_ad = make_ad_params(layer);
        WaveVariables<AD1> wave_ad;
        wave_ad.omega = make_ad(omega);
        wave_ad.k = make_ad(k);
        wave_ad.z = make_ad(0.0, 1.0);
        Eigen::Matrix<AD1, 2, 2> B_ad = computeLayerMatrixLiquid(p_ad, wave_ad);
        Eigen::Matrix<double, 2, 2> A;
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2; ++c)
                A(r, c) = B_ad(r, c).derivatives()(0);
        return A;
    };
    struct LiquidEndpointOperators {
        Eigen::Matrix<double, 2, 2> value;
        Eigen::Matrix<double, 2, 2> derivative;
    };
    auto build_liquid_endpoint = [&](const PhysicalParameters<double>& layer,
                                     const Eigen::Matrix<double, 2, 2>& generator, double z) {
        WaveVariables<double> wave_z;
        wave_z.omega = omega;
        wave_z.k = k;
        wave_z.z = z;

        LiquidEndpointOperators ops;
        ops.value = computeLayerMatrixLiquid(layer, wave_z);
        ops.derivative = generator * ops.value;
        return ops;
    };
    auto apply_liquid_endpoint = [](const LiquidEndpointOperators& ops,
                                    const Eigen::Matrix<double, 2, 1>& Z0) {
        std::pair<Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 1>> out;
        out.first = ops.value * Z0;
        out.second = ops.derivative * Z0;
        return out;
    };

    double last_crossing_depth = -std::numeric_limits<double>::infinity();
    double y_init_for_calibration = get_val(Y(I24));
    bool supra_rayleigh_for_calibration = (y_init_for_calibration < 0.0);
    bool in_liquid_mode = false;
    // Liquid-regime share of signed_count. The solid and liquid regimes use
    // different state variables whose crossing orientations can differ, so the
    // ocean label combines per-regime signed sums instead of one global sum.
    int signed_liquid = 0;
    Eigen::Matrix<double, 2, 1> Z;
    {
        double rho_hs = get_val(halfSpaceParams.rho);
        double c44_hs = get_val(halfSpaceParams.c44);
        double beta_v_sq = (rho_hs > 1e-10) ? c44_hs / rho_hs : 1e10;
        double kappa_sq = k * k - (omega * omega) / beta_v_sq;
        if (kappa_sq > 0.0 && !supra_rayleigh_for_calibration) {
            double kappa = std::sqrt(kappa_sq);
            double virtual_thickness = 5.0 / kappa;
            if (virtual_thickness > 500.0) virtual_thickness = 500.0;

            WaveVariables<double> hs_wave;
            hs_wave.omega = omega;
            hs_wave.k = k;
            hs_wave.z = virtual_thickness;
            Eigen::Matrix<double, 5, 5> B_hs = computeLayerMatrix(halfSpaceParams, hs_wave);
            Eigen::Matrix<double, 5, 1> Y_boundary = Y;
            Eigen::Matrix<double, 5, 1> Y_virtual = B_hs.inverse() * Y_boundary;

            int num_steps = estimateRayleighBranSubsteps(halfSpaceParams, omega, k, virtual_thickness);
            double dz = virtual_thickness / static_cast<double>(num_steps);
            const Eigen::Matrix<double, 5, 5> gen_hs = make_generator(halfSpaceParams);
            const EndpointOperators hs_bottom_ops = build_endpoint(halfSpaceParams, gen_hs, 0.0);
            const EndpointOperators hs_top_ops = build_endpoint(halfSpaceParams, gen_hs, dz);
            // Within one layer the state at the bottom of a substep is the state
            // at the top of the previous one: the value is carried in Y_virtual,
            // and B(z) = exp(Az) gives B'(z) = A B(z), so the endpoint z
            // derivative at the top equals the derivative at the next bottom.
            // Only the first substep of the layer has to evaluate z = 0.
            Eigen::Matrix<double, 5, 1> bottom_val = Eigen::Matrix<double, 5, 1>::Zero();
            Eigen::Matrix<double, 5, 1> bottom_der = Eigen::Matrix<double, 5, 1>::Zero();
            bool have_bottom = false;
            for (int step = 0; step < num_steps; ++step) {
                if (!have_bottom) {
                    auto bottom = apply_endpoint(hs_bottom_ops, Y_virtual);
                    bottom_val = bottom.first;
                    bottom_der = bottom.second;
                    have_bottom = true;
                }
                auto top = apply_endpoint(hs_top_ops, Y_virtual);

                double y1 = bottom_val(I24);
                double y2 = top.first(I24);
                double dy1 = bottom_der(I24);
                double dy2 = top.second(I24);
                double c1 = -(bottom_val(I23) + bottom_val(I14));
                double c2 = -(top.first(I23) + top.first(I14));
                double dc1 = -(bottom_der(I23) + bottom_der(I14));
                double dc2 = -(top.second(I23) + top.second(I14));

                result.signed_count += countBranHermiteCrossings(
                    y1, y2, dy1, dy2, c1, c2, dc1, dc2, dz, -1,
                    -virtual_thickness + step * dz, &last_crossing_depth, &result.crossings);

                Y_virtual = top.first;
                bottom_val = top.first;
                bottom_der = top.second;
                double max_val = 0.0;
                for (int m = 0; m < 5; ++m) max_val = std::max(max_val, std::abs(Y_virtual(m)));
                if (max_val > 1.0e10) {
                    // The propagation is linear in the state, so the carried
                    // derivative has to be rescaled with the state.
                    const double scale = 1.0 / max_val;
                    Y_virtual *= scale;
                    bottom_val *= scale;
                    bottom_der *= scale;
                }
            }
            Y = Y_boundary;
        }
    }

    // Crossing depths are tracked as cumulative height above the halfspace top
    // so that the dedup in countBranHermiteCrossings compares positions in one
    // global coordinate; per-layer local depths would collide across layers.
    double depth_offset = 0.0;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        bool current_layer_is_liquid = isLiquidLayer(layers[i]);

        if (!in_liquid_mode && !current_layer_is_liquid) {
            int num_steps = estimateRayleighBranSubsteps(layers[i], omega, k, thicknesses[i]);
            double dz = thicknesses[i] / static_cast<double>(num_steps);
            const Eigen::Matrix<double, 5, 5> gen_layer = make_generator(layers[i]);
            const EndpointOperators layer_bottom_ops = build_endpoint(layers[i], gen_layer, 0.0);
            const EndpointOperators layer_top_ops = build_endpoint(layers[i], gen_layer, dz);
            // See the virtual halfspace loop: only the first substep of a layer
            // needs a z = 0 evaluation, the rest reuse the previous top.
            Eigen::Matrix<double, 5, 1> bottom_val = Eigen::Matrix<double, 5, 1>::Zero();
            Eigen::Matrix<double, 5, 1> bottom_der = Eigen::Matrix<double, 5, 1>::Zero();
            bool have_bottom = false;
            for (int step = 0; step < num_steps; ++step) {
                if (!have_bottom) {
                    auto bottom = apply_endpoint(layer_bottom_ops, Y);
                    bottom_val = bottom.first;
                    bottom_der = bottom.second;
                    have_bottom = true;
                }
                auto top = apply_endpoint(layer_top_ops, Y);

                double y1 = bottom_val(I24);
                double y2 = top.first(I24);
                double dy1 = bottom_der(I24);
                double dy2 = top.second(I24);
                double c1 = -(bottom_val(I23) + bottom_val(I14));
                double c2 = -(top.first(I23) + top.first(I14));
                double dc1 = -(bottom_der(I23) + bottom_der(I14));
                double dc2 = -(top.second(I23) + top.second(I14));

                result.signed_count += countBranHermiteCrossings(
                    y1, y2, dy1, dy2, c1, c2, dc1, dc2, dz, i,
                    depth_offset + step * dz, &last_crossing_depth, &result.crossings);

                Y = top.first;
                bottom_val = top.first;
                bottom_der = top.second;
                double max_val = 0.0;
                for (int m = 0; m < 5; ++m) max_val = std::max(max_val, std::abs(Y(m)));
                if (max_val > 1.0e10) {
                    const double scale = 1.0 / max_val;
                    Y *= scale;
                    bottom_val *= scale;
                    bottom_der *= scale;
                }
            }
        } else if (!in_liquid_mode && current_layer_is_liquid) {
            Z(0) = Y(I14);
            Z(1) = Y(I24);
            double seabed_gravity = get_val(layers[i].gravity * layers[i].gravity_seabed_scale);
            if (std::abs(seabed_gravity) > 0.0) {
                double rho_solid = (i + 1 < static_cast<int>(layers.size())) ? get_val(layers[i + 1].rho) : get_val(halfSpaceParams.rho);
                double delta_rho = rho_solid - get_val(layers[i].rho);
                Z(1) = Z(1) + delta_rho * seabed_gravity * Z(0);
            }
            in_liquid_mode = true;
        } else if (in_liquid_mode && !current_layer_is_liquid) {
            result.unsupported_liquid = true;
            return result;
        }

        if (in_liquid_mode && current_layer_is_liquid) {
            int num_steps = estimateLiquidBranSubsteps(layers[i], omega, k, thicknesses[i]);
            double dz = thicknesses[i] / static_cast<double>(num_steps);
            const Eigen::Matrix<double, 2, 2> gen_liquid = make_liquid_generator(layers[i]);
            const LiquidEndpointOperators liquid_bottom_ops = build_liquid_endpoint(layers[i], gen_liquid, 0.0);
            const LiquidEndpointOperators liquid_top_ops = build_liquid_endpoint(layers[i], gen_liquid, dz);
            // Same substep-endpoint reuse as the solid loops.
            Eigen::Matrix<double, 2, 1> bottom_val = Eigen::Matrix<double, 2, 1>::Zero();
            Eigen::Matrix<double, 2, 1> bottom_der = Eigen::Matrix<double, 2, 1>::Zero();
            bool have_bottom = false;
            for (int step = 0; step < num_steps; ++step) {
                if (!have_bottom) {
                    auto bottom = apply_liquid_endpoint(liquid_bottom_ops, Z);
                    bottom_val = bottom.first;
                    bottom_der = bottom.second;
                    have_bottom = true;
                }
                auto top = apply_liquid_endpoint(liquid_top_ops, Z);

                double y1 = bottom_val(1);
                double y2 = top.first(1);
                double dy1 = bottom_der(1);
                double dy2 = top.second(1);
                double c1 = bottom_val(0);
                double c2 = top.first(0);
                double dc1 = bottom_der(0);
                double dc2 = top.second(0);

                int step_crossings = countBranHermiteCrossings(
                    y1, y2, dy1, dy2, c1, c2, dc1, dc2, dz, i,
                    depth_offset + step * dz, &last_crossing_depth, &result.crossings);
                result.signed_count += step_crossings;
                signed_liquid += step_crossings;

                Z = top.first;
                bottom_val = top.first;
                bottom_der = top.second;
                double max_val = std::max(std::abs(Z(0)), std::abs(Z(1)));
                if (max_val > 1.0e10) {
                    const double scale = 1.0 / max_val;
                    Z *= scale;
                    bottom_val *= scale;
                    bottom_der *= scale;
                }
            }
        }
        depth_offset += thicknesses[i];
    }

    int endpoint_correction = 0;
    if (!supra_rayleigh_for_calibration && !result.crossings.empty() && !layers.empty()) {
        const auto& last = result.crossings.back();
        // depth_offset now equals the total stack height, i.e. the free surface.
        if (last.layer_index == 0 && std::abs(last.local_depth_km - depth_offset) < 1.0e-6) {
            endpoint_correction = 1;
        }
    }
    if (in_liquid_mode) {
        // Combine per-regime signed sums: the raw cross-regime total can
        // cancel between the solid minors and the liquid [U, P] state, but a
        // signed sum within each regime still cancels spurious +/- crossing
        // pairs (e.g. grazing zeros) that an unsigned tally would count.
        int s_solid = result.signed_count - signed_liquid;
        int s_liquid = signed_liquid;
        if (endpoint_correction != 0 && !result.crossings.empty()) {
            // The free-surface crossing (layer 0) belongs to the liquid regime.
            s_liquid -= result.crossings.back().contribution;
        }
        int crossing_count = std::abs(s_solid) + std::abs(s_liquid);
        int sign = (result.signed_count < 0) ? -1 : 1;
        result.calibrated_signed_count = sign * crossing_count;
        result.n_integer = crossing_count;
    } else {
        result.calibrated_signed_count = result.signed_count;
        if (endpoint_correction != 0 && !result.crossings.empty()) {
            result.calibrated_signed_count -= result.crossings.back().contribution;
        }
        result.n_integer = std::abs(result.calibrated_signed_count);
    }
    return result;
}

BranModeCountResult calculateBranModeCountLoveSolid(
    const std::vector<PhysicalParameters<double>>& layers,
    const std::vector<double>& thicknesses,
    const PhysicalParameters<double>& halfSpaceParams,
    double omega,
    double phaseVelocity
) {
    BranModeCountResult result;
    if (layers.size() != thicknesses.size()) {
        result.nan_or_radiative = true;
        return result;
    }
    if (isLiquidLayer(halfSpaceParams)) {
        result.unsupported_liquid = true;
        return result;
    }
    for (const auto& layer : layers) {
        if (isLiquidLayer(layer)) {
            result.unsupported_liquid = true;
            return result;
        }
    }

    const double k = omega / phaseVelocity;
    WaveVariables<double> wave;
    wave.omega = omega;
    wave.k = k;
    wave.z = 0.0;
    Eigen::Matrix<double, 2, 1> Y = computeInitialVectorSH(halfSpaceParams, wave);
    for (int i = 0; i < 2; ++i) {
        if (!std::isfinite(Y(i))) {
            result.nan_or_radiative = true;
            return result;
        }
    }

    // The SH propagator is analytic in the layer coordinate, so the endpoint
    // derivative does not need automatic differentiation.  With
    //
    //   A(z) = [[C, S/eta], [eta*S, C]],  C = cosh(nu z), S = sinh(nu z),
    //   eta = c44 nu,
    //
    // and dC/dz = nu S, dS/dz = nu C,
    //
    //   dA/dz = [[nu S, C/c44], [c44 nu^2 C, nu S]].
    //
    // Both derivative entries stay regular as nu -> 0 (nu S -> nu^2 z and
    // c44 nu^2 C -> 0), so no separate small-nu form is needed for them.  The
    // value entries keep the exact expressions of computeLayerMatrixSH,
    // including its eta ~ 0 fallback, so the propagated values are unchanged.
    // Love propagation follows calculateLoveDispersion: upward travel uses a
    // negative matrix distance z = -s, hence d/ds = -d/dz.
    struct SHLayerCore {
        std::complex<double> nu;
        std::complex<double> eta;
        double c44 = 0.0;
        bool eta_degenerate = false;
    };
    auto make_sh_core = [](const PhysicalParameters<double>& p, double omega_in, double k_in) {
        using CT = std::complex<double>;
        SHLayerCore core;
        CT c44_c(p.c44, 0.0);
        CT c66_c(p.c66, 0.0);
        CT rho_c(p.rho, 0.0);
        CT k_c(k_in, 0.0);
        CT w_c(omega_in, 0.0);
        CT num = c66_c * k_c * k_c - rho_c * w_c * w_c;
        CT nu_sq = num / c44_c;
        core.nu = safe_complex_sqrt(nu_sq);
        core.eta = c44_c * core.nu;
        core.c44 = p.c44;
        core.eta_degenerate = (std::abs(core.eta.real()) < 1e-15 &&
                               std::abs(core.eta.imag()) < 1e-15);
        return core;
    };
    auto evaluate_endpoint = [](const SHLayerCore& core, const Eigen::Matrix<double, 2, 1>& Y0, double s) {
        using CT = std::complex<double>;
        const double z = -s;
        const CT C = ad_cosh(core.nu * CT(z, 0.0));
        const CT S = ad_sinh(core.nu * CT(z, 0.0));

        // Value matrix, matching computeLayerMatrixSH entry for entry.
        const double a00 = C.real();
        const double a01 = core.eta_degenerate ? z : (S / core.eta).real();
        const double a10 = (core.eta * S).real();
        const double a11 = a00;

        // d/ds = -d/dz.  In the degenerate branch the value entry is z itself,
        // whose z derivative is 1.
        const CT nu_S = core.nu * S;
        const double d00 = -nu_S.real();
        const double d01 = core.eta_degenerate ? -1.0 : -(C / CT(core.c44, 0.0)).real();
        const double d10 = -(core.eta * core.nu * C).real();
        const double d11 = d00;

        std::pair<Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 1>> out;
        out.first(0) = a00 * Y0(0) + a01 * Y0(1);
        out.first(1) = a10 * Y0(0) + a11 * Y0(1);
        out.second(0) = d00 * Y0(0) + d01 * Y0(1);
        out.second(1) = d10 * Y0(0) + d11 * Y0(1);
        return out;
    };

    // Closed-form crossing enumeration.
    //
    // In a homogeneous SH layer the state [W, T] = [u_y, tau_yz] obeys
    // dW/ds = -T/c44 and dT/ds = c44 r^2 W with
    //
    //   r^2 = (rho omega^2 - c66 k^2) / c44 = -nu^2,
    //
    // so T'' = -r^2 T and the secular variable is a single harmonic (or a
    // single hyperbolic) in the layer.  Its zeros are therefore known in closed
    // form and do not have to be searched for on a substep mesh.
    //
    //   r^2 > 0:  T(s) = R cos(r s - phi),   R = hypot(T0, c44 r W0),
    //                                        phi = atan2(c44 r W0, T0),
    //             zeros at s = (phi + pi/2 + m pi) / r.
    //   r^2 < 0:  T(s) = T0 cosh(kappa s) - c44 kappa W0 sinh(kappa s),
    //             one zero at s = atanh(T0 / (c44 kappa W0)) / kappa, when that
    //             argument lies in (0, tanh(kappa d)].
    //
    // The crossing orientation is fixed by the regime rather than evaluated:
    // at a zero of T, test_value = W dT/ds = c44 r^2 W^2, so every oscillatory
    // crossing contributes -1 and every evanescent one +1, matching the sign
    // convention of countBranHermiteCrossings.  W cannot vanish where T does
    // except for the trivial solution.
    //
    // Zeros are enumerated on the half-open interval (0, d]: a zero sitting on
    // a layer interface belongs to the layer below, which is the same rule the
    // substep dedup enforced.  The dedup is still applied as a guard.
    const double half_pi = 0.5 * M_PI;
    // Range control only: an evanescent layer is walked in pieces short enough
    // that cosh(kappa ds) cannot overflow the rescaling guard.  This has
    // nothing to do with resolving crossings, so the pieces are long.
    const double max_decay_per_piece = 20.0;

    double last_crossing_depth = -std::numeric_limits<double>::infinity();
    // Cumulative height above the halfspace top; see the Rayleigh counter for
    // why the dedup must not use per-layer local depths.
    double depth_offset = 0.0;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        const PhysicalParameters<double>& layer = layers[i];
        const double thickness = thicknesses[i];
        const double c44 = layer.c44;
        const SHLayerCore core = make_sh_core(layer, omega, k);
        const double r_sq = (c44 > 1.0e-10)
            ? (layer.rho * omega * omega - layer.c66 * k * k) / c44
            : 0.0;

        int pieces = 1;
        if (r_sq < 0.0) {
            const double kappa = std::sqrt(-r_sq);
            const double decay = kappa * thickness;
            if (decay > max_decay_per_piece) {
                pieces = static_cast<int>(decay / max_decay_per_piece) + 1;
            }
        }
        const double ds = thickness / static_cast<double>(pieces);

        for (int piece = 0; piece < pieces; ++piece) {
            const double piece_base = depth_offset + piece * ds;
            const double W0 = Y(0);
            const double T0 = Y(1);

            auto emit = [&](double s_star, int contribution) {
                const double local_depth = piece_base + s_star;
                if (std::abs(local_depth - last_crossing_depth) < 1.0e-8) return;
                result.signed_count += contribution;
                last_crossing_depth = local_depth;
                BranCrossingInfo info;
                info.layer_index = i;
                info.local_depth_km = local_depth;
                info.contribution = contribution;
                info.secular = 0.0;
                const Eigen::Matrix<double, 2, 1> state = evaluate_endpoint(core, Y, s_star).first;
                info.conjugate = state(0);
                info.secular_derivative = c44 * r_sq * state(0);
                info.test_value = info.conjugate * info.secular_derivative;
                result.crossings.push_back(info);
            };

            // A zero sitting on an interface must be reported by exactly one
            // side.  Testing the window strictly would let rounding push such a
            // zero just outside both neighbours and lose it, so the window is
            // opened by the dedup tolerance on both ends and the dedup in emit()
            // discards the duplicate.
            constexpr double window_tol = 1.0e-8;
            auto in_window = [&](double s_star) {
                return s_star > -window_tol && s_star <= ds + window_tol;
            };

            if (r_sq > 0.0) {
                const double r = std::sqrt(r_sq);
                const double phi = std::atan2(c44 * r * W0, T0);
                // Integers m with 0 < (phi + pi/2 + m pi) / r <= ds, widened by
                // one on each side so a boundary zero is always a candidate.
                const double lower = -(phi + half_pi) / M_PI;
                const double upper = (r * ds - phi - half_pi) / M_PI;
                const long m_begin = static_cast<long>(std::floor(lower));
                const long m_end = static_cast<long>(std::floor(upper)) + 1;
                for (long m = m_begin; m <= m_end; ++m) {
                    const double s_star = (phi + half_pi + static_cast<double>(m) * M_PI) / r;
                    if (!in_window(s_star)) continue;
                    emit(s_star, -1);
                }
            } else if (r_sq < 0.0) {
                const double kappa = std::sqrt(-r_sq);
                const double denom = c44 * kappa * W0;
                if (denom != 0.0) {
                    const double ratio = T0 / denom;
                    if (ratio >= 0.0 && ratio < 1.0) {
                        const double s_star = std::atanh(ratio) / kappa;
                        if (in_window(s_star)) emit(s_star, 1);
                    }
                }
            }
            // r_sq == 0 leaves T constant, so it has no interior zero.

            Y = evaluate_endpoint(core, Y, ds).first;
            const double max_val = std::max(std::abs(Y(0)), std::abs(Y(1)));
            if (max_val > 1.0e10) Y *= (1.0 / max_val);
        }
        depth_offset += thickness;
    }

    result.calibrated_signed_count = result.signed_count;
    if (!result.crossings.empty() && !layers.empty()) {
        const auto& last = result.crossings.back();
        // depth_offset now equals the total stack height, i.e. the free surface.
        if (last.layer_index == 0 && std::abs(last.local_depth_km - depth_offset) < 1.0e-6) {
            result.calibrated_signed_count -= last.contribution;
        }
    }
    result.n_integer = std::abs(result.calibrated_signed_count);
    return result;
}


template <typename T>
struct ModeCountDebugInfo {
    T theta_init = 0.0;
    T y_init = 0.0;
    T x_init = 0.0;
    T init_phase = 0.0;
    T halfspace_virtual_phase = 0.0;
    T layer_phase_sum = 0.0;
    T total_phase = 0.0;
    T total_phase_over_pi = 0.0;
    T kappa_sq = 0.0;
    int layer_mode_sum = 0;
    int floor_count = 0;
    int abs_floor_count = 0;
    bool supra_rayleigh_pi_correction_applied = false;
    bool used_virtual_halfspace = false;
    bool used_liquid_mode = false;
};



// 3. Total Mode Calculation
// 3. Core Phase Accumulation Helper
template <typename T>
T calculateTotalPhaseAccumulationWithDebug(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity,
    WaveType type = WaveType::Rayleigh,
    ModeCountDebugInfo<T>* debug = nullptr
) {
    T k = omega / phaseVelocity;
    
    WaveVariables<T> wave;
    wave.omega = omega;
    wave.k = k;
    wave.z = make_zero_like(omega);

    int total_N = 0;
    T total_phase = 0.0;
    
    // Current Phase Sum (optional, for debugging or advanced checking)
    // T total_phase_accum = 0.0;

    if (type == WaveType::Rayleigh) {
        // --- Rayleigh Wave ---
        Eigen::Matrix<T, 5, 1> Y = computeInitialVector(halfSpaceParams, wave);
        
        // Helper to compute analytical initial phase in half-space
        // Used to calibrate the starting phase for mode counting
        
        // --- Rayleigh Wave ---

        
        // Phase accumulation from half-space boundary
        // Two components:
        // 1. Initial Woodhouse phase at Y₀ (analytical)
        //    computeInitialVector returns Y₀ at the boundary.
        //    We compute its Woodhouse phase directly.
        // 2. Virtual half-space layer (3/κ): Captures additional phase for modes
        //    near the radiative cutoff (c → Vs)
        {
            // 1. Analytical Initial Phase from Y₀
            // Woodhouse phase = atan2(I24, -(I23+I14))
            T y_init = get_val(Y(I24));
            T x_init = -(get_val(Y(I23)) + get_val(Y(I14)));
            T theta_init = std::atan2(y_init, x_init);
            if (debug) {
                debug->y_init = y_init;
                debug->x_init = x_init;
                debug->theta_init = theta_init;
            }
            
            // Normalize to units of π and wrap to [-0.5, 0.5] relative to n*π
            // Actually, we want the absolute phase value. 
            // The analytical study showed Y₀ phase is ~ +0.5π for guided modes.
            // Mode counting expects n at the root.
            // If we start with total_phase = theta_init/π, we get n + 0.5.
            // To align with integer n, we should verify what integral value we want.
            // By convention, fundamental mode n=0 should have phase ~ 0 to 1 (or 0.5).
            // A common definition is phase integral = n.
            // If total accumulation is n + 0.5, we substract 0.5.
            //
            // But theta_init IS the physical phase. 
            // If we use it, we get count = n.5. 
            // The user requested to use the analytical expression.
            // Let's use the raw phase and see if we simply need to adjust our *expectation* of the count 
            // (e.g., target is n+0.5) or if we subtract 0.5 to make target n.
            //
            // Woodhouse (1988) defines n as the number of π intervals.
            // If total phase is 0.5π (fund) -> count 0.
            // If total phase is 1.5π (1st) -> count 1.
            // The floor(total/π) logic handles this naturally?
            // floor(0.5) = 0. floor(1.5) = 1.
            //
            // BUT: previous verification showed:
            // Mode 0 count = 0.39. floor(0.39) = 0. OK.
            // Mode 1 count = 1.39. floor(1.39) = 1. OK.
            // Mode 8 count = 8.07. floor(8.07) = 8. OK.
            //
            // This was with -0.33 offset.
            // Analytical Y₀ phase is ~ +0.51π.
            // If we use theta_init directly (~ +0.51):
            // Mode 0: layer (~0) + init (0.51) = 0.51. floor(0.51) = 0.
            // Mode 8: layer (~8) + init (~0.83) = 8.83. floor(8.83) = 8.
            //
            // Wait, Mode 8 init phase was +0.83π.
            // With -0.33 offset: 0.83 - 0.33 = 0.50. Perfect 8.0 -> 8.5? No.
            //
            // Let's rely on the previous finding:
            // "Measurements showed Mode 0-7 have phase n+0.5 without any offset."
            // "Y₀ phase is +0.5."
            // So Total = n + 0.5 + 0.5 = n + 1.0!
            // That's why we needed SUBTRACTION.
            //
            // To get count = n + fraction (where floor gives n):
            // We want Total ≈ n + 0.5.
            // Layer ≈ n + 0.5? No, Layer contribution for fundamental is small (~0).
            //
            // Actually, let's look at the "Layer Phase" logic again.
            // The loop accumulates phase change from bottom to top.
            // If we start with Y₀ phase, we just add the layer changes.
            //
            // Proposed Implementation:
            // total_phase = (theta_init / M_PI) - 0.5;
            // The -0.5 is the conversion from "physical phase" to "mode index counter".
            // Physical phase starts at ~π/2 (evanescent decay).
            // Fundamental mode wave function has no nodes, but phase rotates π/2 to surface?
            //
            // If we use: total_phase = (theta_init / M_PI) - 1.0;
            // Because Y₀ phase (+0.5) + Layer (+0.5) = 1.0 for fundamental?
            //
            // Let's use the analytical value and subtract 1.0?
            // Or simpler: The user wants us to USE the analytical expression.
            // The result should be consistent.
            //
            // Let's try: total_phase = (theta_init / M_PI) - 1.0;
            // Since -0.33 worked (approx -0.5 + small adjustment),
            // and analytical is ~0.5.
            // 0.5 - 1.0 = -0.5. Matches our empirics.
            
            // Correct logic:
            // theta_init is physical phase in [-PI, PI] (~0.5 PI for Y0).
            // Layer propagation adds delta.
            // Verification showed that using -0.5*PI centers the mode counts at n + 0.5,
            // which is optimal for stability when using floor().
            // Mode 0 count ≈ 0.5, Mode 8 count ≈ 8.5.
            total_phase = theta_init - 0.5 * M_PI; 
            if (debug) {
                debug->init_phase = total_phase;
            }
            
            bool supra_rayleigh = (y_init < 0.0);

            // Correction for c > c_R (supra-Rayleigh branches).
            // Empirically, these branches sit one full mode above the sub-Rayleigh
            // phase count after the standard Y0 offset is applied, so we need a
            // 2*PI correction here. The virtual half-space contribution is not
            // stable on this side of the Rayleigh pole and is skipped below.
            // This matches the two-layer elastic debug case where mode_index is
            // otherwise undercounted by exactly one.
            if (supra_rayleigh) {
                 total_phase += 2.0 * M_PI;
                 if (debug) {
                     debug->supra_rayleigh_pi_correction_applied = true;
                 }
            } 

            // Compute κ (decay constant) in the half-space
            T rho_hs = get_val(halfSpaceParams.rho);
            T c44_hs = get_val(halfSpaceParams.c44);
            T beta_v_sq = (rho_hs > 1e-10) ? c44_hs / rho_hs : 1e10;
            T kappa_sq = k * k - (omega * omega) / beta_v_sq;  // κ² = k² - (ω/β)²
            if (debug) {
                debug->kappa_sq = kappa_sq;
            }
            
            if (kappa_sq > 0.0 && !supra_rayleigh) {
                // Evanescent in half-space (normal case for guided modes)
                T kappa = std::sqrt(kappa_sq);
                
                // Virtual thickness: 5 decay lengths (wave amplitude → e^-5 < 1%)
                // Increased from 3.0 to 5.0 to better capture phase for modes near 
                // radiative cutoff (Mode 9) where deep penetration occurs.
                T virtual_thickness = 5.0 / kappa;
                
                // Limit virtual thickness to reasonable value
                if (virtual_thickness > 500.0) virtual_thickness = 500.0;
                
                // Compute the layer matrix for the virtual half-space layer
                WaveVariables<T> hs_wave;
                hs_wave.omega = omega;
                hs_wave.k = k;
                hs_wave.z = virtual_thickness;
                Eigen::Matrix<T, 5, 5> B_hs = computeLayerMatrix(halfSpaceParams, hs_wave);
                
                // Compute Y_deep = B_hs^(-1) * Y (inverse propagation)
                // Then count phases from Y_deep to Y
                Eigen::Matrix<T, 5, 5> B_hs_inv = B_hs.inverse();
                Eigen::Matrix<T, 5, 1> Y_deep = B_hs_inv * Y;
                
                // Count phase from Y_deep (at z=-virtual_thickness) to Y (at z=0)
                std::pair<int, T> hs_res = countModesLayerRayleigh(
                    Y_deep, Y, omega, k, halfSpaceParams, virtual_thickness);
                
                total_phase += hs_res.second;  // Add phase from virtual half-space
                if (debug) {
                    debug->used_virtual_halfspace = true;
                    debug->halfspace_virtual_phase = hs_res.second;
                }
            }
            // For radiative case (kappa_sq <= 0), just use the theta_init based value
        }

        // Track whether we're in liquid mode (similar to calculateRayleighDispersion)
        bool inLiquidMode = false;
        Eigen::Matrix<T, 2, 1> Z;  // 2D state for liquid [w, P]^T
        T layer_phase_sum = 0.0;
        
        for (int i = layers.size() - 1; i >= 0; --i) {
            bool currentLayerIsLiquid = isLiquidLayer(layers[i]);
            wave.z = thicknesses[i];
            
            if (!inLiquidMode && !currentLayerIsLiquid) {
                // Solid to Solid: standard 5x5 propagation
                Eigen::Matrix<T, 5, 5> B = computeLayerMatrix(layers[i], wave);
                Eigen::Matrix<T, 5, 1> Y_top = B * Y;
                
                std::pair<int, T> res = countModesLayerRayleigh(Y, Y_top, omega, k, layers[i], thicknesses[i]);
                total_N += res.first;
                total_phase += res.second;
                layer_phase_sum += res.second;
                
                Y = Y_top;
                
                // Scaling
                double max_val = 0.0;
                for(int m=0; m<5; ++m) {
                    double v = get_magnitude(Y(m)); 
                    if(v > max_val) max_val = v;
                }
                if (max_val > 1e10) {
                    Y = Y * (1.0 / max_val); 
                }
                
            } else if (!inLiquidMode && currentLayerIsLiquid) {
                // Solid to Liquid transition
                // Extract [w, P] from the 5D minor vector
                Z(0) = Y(I14);  // w component
                Z(1) = Y(I24);  // P component
                
                Eigen::Matrix<T, 2, 1> Z_bot = Z;
                Eigen::Matrix<T, 2, 2> B_liq = computeLayerMatrixLiquid(layers[i], wave);
                Z = B_liq * Z;
                
                std::pair<int, T> res = countModesLayerLiquid(Z_bot, Z, omega, k, layers[i], thicknesses[i]);
                total_N += res.first;
                total_phase += res.second;
                layer_phase_sum += res.second;
                
                inLiquidMode = true;
                if (debug) {
                    debug->used_liquid_mode = true;
                }
                
                // Scaling
                double max_val = std::max(get_magnitude(Z(0)), get_magnitude(Z(1)));
                if (max_val > 1e10) {
                    Z = Z / max_val;
                }
                
            } else if (inLiquidMode && currentLayerIsLiquid) {
                // Liquid to Liquid
                Eigen::Matrix<T, 2, 1> Z_bot = Z;
                Eigen::Matrix<T, 2, 2> B_liq = computeLayerMatrixLiquid(layers[i], wave);
                Z = B_liq * Z;
                
                std::pair<int, T> res = countModesLayerLiquid(Z_bot, Z, omega, k, layers[i], thicknesses[i]);
                total_N += res.first;
                total_phase += res.second;
                layer_phase_sum += res.second;
                if (debug) {
                    debug->used_liquid_mode = true;
                }
                
                // Scaling
                double max_val = std::max(get_magnitude(Z(0)), get_magnitude(Z(1)));
                if (max_val > 1e10) {
                    Z = Z / max_val;
                }
                
            } else {
                // Liquid to Solid (not typically used)
                // Return invalid
                return -1;
            }
        }
        if (debug) {
            debug->layer_phase_sum = layer_phase_sum;
            debug->layer_mode_sum = total_N;
        }
        
    } else {
        // --- Love Wave ---
        Eigen::Matrix<T, 2, 1> Y = computeInitialVectorSH(halfSpaceParams, wave);

        if (std::isnan(get_magnitude(Y(0)))) return -1;

        for (int i = layers.size() - 1; i >= 0; --i) {
            wave.z = -thicknesses[i]; // Disper26 uses negative z for upward Love propagation in `computeLayerMatrixSH`? 
                                      // Checking logic: `computeLayerMatrixSH` uses `std::cosh(nu * v.z)` which is even, so sign doesn't matter for Cosh.
                                      // But S has `std::sinh`. 
                                      // In `calculateLoveDispersion`: `wave.z = -thicknesses[i];`.
                                      // Let's match that convention.
            
            // Actually `a` layer matrix is `exp` based. 
            // Haskell matrix for layer d usually maps Y(z+d) to Y(z) or vice versa.
            // If we propagate BOTTOM to TOP, and z axis points DOWN (standard in seismology):
            // Bottom is z_k, Top is z_{k-1}. z_k > z_{k-1}.
            // Thickness h = z_k - z_{k-1}.
            // Y(z_{k-1}) = B * Y(z_k).
            // `computeLayerMatrix` in Rayleigh part uses `wave.z = thicknesses[i]` (positive).
            // `computeLayerMatrixSH` uses `wave.z` directly.
            // In `calculateLoveDispersion`: `wave.z = -thicknesses[i]`.
            // Why negative?
            // If z is positive down, propagating UP (decreasing z) might imply negative distance argument if the function expects z displacement?
            // Let's stick to what `calculateLoveDispersion` does to be safe.
            
            Eigen::Matrix<T, 2, 2> A = computeLayerMatrixSH(layers[i], wave);
            Eigen::Matrix<T, 2, 1> Y_top = A * Y;

            // Important: Count requires positive thickness for P_est
            std::pair<int, T> res = countModesLayerLove(Y, Y_top, omega, k, layers[i], thicknesses[i]); 
            total_N += res.first;
            total_phase += res.second;

            Y = Y_top;

            // Scaling
            double max_val = 0.0;
            for(int m=0; m<2; ++m) {
                double v = get_magnitude(Y(m)); 
                if(v > max_val) max_val = v;
            }
            if (max_val > 1e10) {
                double scale = 1.0 / max_val;
                Y = Y * scale; 
            }
        }
        if (debug) {
            debug->layer_phase_sum = total_phase;
            debug->layer_mode_sum = total_N;
        }
    }

    /* Debug print removed
    if (true) {
        std::cout << "DEBUG: c=" << phaseVelocity << " Total N=" << total_N << " TotalPhase=" << total_phase 
                  << " (Cycles=" << total_phase/(2.0*M_PI) << ")" << std::endl;
    }
    */
    
    // Return the raw total phase (radians)
    // Negative because phase usually decreases or wraps negatively standard convention?
    // We just return the sum.
    if (debug) {
        debug->total_phase = total_phase;
        debug->total_phase_over_pi = total_phase / M_PI;
        debug->floor_count = static_cast<int>(std::floor(total_phase / M_PI));
        debug->abs_floor_count = static_cast<int>(std::floor(std::abs(total_phase / M_PI)));
    }
    return total_phase;
}

template <typename T>
T calculateTotalPhaseAccumulation(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity,
    WaveType type = WaveType::Rayleigh
) {
    return calculateTotalPhaseAccumulationWithDebug(
        layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type,
        static_cast<ModeCountDebugInfo<T>*>(nullptr));
}

// Continuous Mode Count. The legacy integer wrapper that used to sit here is
// gone: the AD-z counter is the integer label, and for Love the legacy integer
// was wrong by more than an offset (see docs/mode_counting.md). The continuous
// value stays -- it is the only smooth mode-count diagnostic there is, and
// test_low_c.py / test_safe_range.py use it to probe the edges of the valid c
// range.
template <typename T>
T calculateContinuousModeCount(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity,
    WaveType type
) {
    ModeCountDebugInfo<T> debug;
    T total_phase = calculateTotalPhaseAccumulationWithDebug(
        layers, thicknesses, halfSpaceParams, omega, phaseVelocity, type, &debug);

    if (type == WaveType::Rayleigh && debug.used_liquid_mode) {
        T abs_count = std::abs(total_phase / M_PI);
        T fractional = abs_count - std::floor(abs_count);
        T supra_offset = debug.supra_rayleigh_pi_correction_applied ? T(1.0) : T(0.0);
        return static_cast<T>(debug.layer_mode_sum) + supra_offset + fractional;
    }

    return std::abs(total_phase / M_PI);
}




template <typename T, typename Func>
T brents_method(
    Func func,
    T min,
    T max,
    T tol = 1e-6,
    int max_iter = 100,
    T* final_function_value = nullptr
) {
    T a = min;
    T b = max;
    T fa = func(a);
    T fb = func(b);
    
    if (fa * fb > 0) {
        // For robustness, might need to scan or expand, but for now return error/nan
        // Or handle simple failures.
        std::cerr << "Root not bracketed in Brent's method: f(" << a << ")=" << fa << ", f(" << b << ")=" << fb << std::endl;
        if (final_function_value != nullptr) *final_function_value = fb;
        return -1.0; 
    }
    
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    
    T c = a;
    T fc = fa;
    bool mflag = true;
    T s = 0;
    T fs = 0;
    T d = 0;
    
    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(b - a) < tol) {
            if (final_function_value != nullptr) *final_function_value = fb;
            return b; // Converged
        }
        if (std::abs(fb) < 1e-12) {
            if (final_function_value != nullptr) *final_function_value = fb;
            return b; // Exact zero found
        }
        
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                (c * fa * fb) / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }
        
        // Conditions to force bisection
        // bool cond1 = (s < (3 * a + b) / 4) || (s > b); // Remove unused variable
        
        T lower_bound = std::min((3 * a + b) / 4, b);
        T upper_bound = std::max((3 * a + b) / 4, b);
        bool s_outside = (s < lower_bound) || (s > upper_bound);
        
        if (s_outside ||
            (mflag && std::abs(s - b) >= std::abs(b - c) / 2) ||
            (!mflag && std::abs(s - b) >= std::abs(c - d) / 2) ||
            (mflag && std::abs(b - c) < tol) ||
            (!mflag && std::abs(c - d) < tol)) {
            
            s = (a + b) / 2;
            mflag = true;
        } else {
            mflag = false;
        }
        
        fs = func(s);
        d = c;
        c = b;
        fc = fb; // c is previous b
        
        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    
    if (final_function_value != nullptr) *final_function_value = fb;
    return b;
}

// --- Sequential Mode Tracing Logic ---

// Structure for Dispersion Point
struct DispersionPoint {
    double frequency;
    double period;
    double phase_velocity;
    double group_velocity;
    int mode_index; 
};

// 1. Root Searching (Bracketing Scan)
// Scans [min_c, max_c] with step_c. 
// If sign changes, calls brents_method.
// Only keeps distinct roots.
template <typename Func, typename T>
std::vector<T> searchRoots(Func func, T min_c, T max_c, T step_c, T tol = 1e-6) {
    std::vector<T> roots;
    T c1 = min_c;
    T f1 = func(c1);
    
    // TODO: Integrate Woodhouse's mode counting to ensure no missed modes.
    // Current: Simple grid scan. Risk of missing double roots or close roots if step is too large.
    
    for (T c2 = min_c + step_c; c2 <= max_c; c2 += step_c) {
        T f2 = func(c2);
        
        if (!std::isnan(f1) && !std::isnan(f2)) {
             if (f1 * f2 < 0.0) { // Sign change found
                // Refine
                T root = brents_method(func, c1, c2, tol);
                if (root > 0.0) {
                    // Residual Check to avoid poles
                    T val = func(root);
                    if (std::abs(val) < 0.1) { // Tightened from 1.0 to 0.1 to filter spurious 0.2
                        roots.push_back(root);
                    }
                }
            }
        }
        
        c1 = c2; 
        f1 = f2;
    }
    
    // Check upper bound exact match or last segment
    if (std::abs(f1) < 1e-12) {
       // Already caught?
       if (roots.empty() || std::abs(roots.back() - c1) > 1e-4) {
           roots.push_back(c1);
       }
    }
    
    return roots;
}

// Forward declaration for Group Velocity internal usage
template <typename T>
T calculateGroupVelocityInternal(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega,
    T c_phase,
    WaveType type
);

// 2. Sequential Tracing
// Traces a specific mode (by index in initial search) across a frequency range.
// Uses Group Velocity to predict next C.
template <typename T>
std::vector<DispersionPoint> traceDispersionCurve(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T freq_start,
    T freq_end,
    int num_steps,
    int mode_rank, // 0 for Fundamental, 1 for 1st Overtone...
    WaveType type = WaveType::Rayleigh,
    T c_min_search = 0.5, // km/s
    T c_max_search = 6.0  // km/s
) {
    
    std::vector<DispersionPoint> curve;
    double df = (freq_end - freq_start) / (num_steps - 1);
    
    // Helper lambda for dispersion function at specific omega
    auto get_disp_func = [&](T omega_val) {
         return [=](T c_val) {
             // We need ADS wrapper if calculateDispersionFunction expects generic T but inputs are mostly double?
             // Actually, the templates allow T=double.
             // But we need to define T for calculateDispersionFunction.
             // The function signature is template<T>.
             // If we pass double, it uses double. 
             auto res = calculateDispersionFunction(layers, thicknesses, halfSpace, omega_val, c_val, type);
             return res(I24); // Characteristic function
         };
    };
    
    // Step 1: Initial Search at freq_start
    T current_freq = freq_start;
    auto func_init = get_disp_func(current_freq * 2.0 * M_PI);
    std::vector<T> initial_roots = searchRoots(func_init, c_min_search, c_max_search, T(0.05)); // 50m/s step
    
    // Sort roots (High C to Low C usually? Or just Low to High?)
    // Fundamental mode is usually the SLOWEST (Low C) for surface waves normal dispersion?
    // Wait, Fundamental Rayleigh mode has highest velocity at low frequency?
    // Usually "Fundamental" is labeled Mode 0.
    // In standard surface wave texts:
    // Fundamental mode has the lowest wavenumber (k) for a given frequency -> Highest Phase Velocity?
    // Or is it highest Phase Velocity at Low Freq, decreasing?
    // Convention: 
    // Fundamental (n=0): No nodal planes (except infinity).
    // Usually it is the highest velocity curve at long periods.
    // BUT at a fixed frequency, we get multiple roots c_0 > c_1 > c_2 ...
    // So Fundamental is the LARGEST c root?
    // Let's implement High-to-Low sort.
    
    std::sort(initial_roots.rbegin(), initial_roots.rend()); // Descending Order
    
    if (mode_rank >= initial_roots.size()) {
        std::cerr << "Mode " << mode_rank << " not found at start freq " << freq_start << ". Found " << initial_roots.size() << " modes." << std::endl;
        return curve;
    }
    
    T current_c = initial_roots[mode_rank];
    
    // Step 2: Loop
    for (int i = 0; i < num_steps; ++i) {
        T omega = 2.0 * M_PI * current_freq;
        
        // Refine current_c exactly (just to be sure, though searchRoots did it)
        // If i > 0, current_c is a prediction.
        auto func = get_disp_func(omega);
        
        // Search bracket? 
        // If prediction is good, use small bracket.
        T bracket_width = 0.2; // +/- 0.2 km/s
        T c_refined = brents_method(func, current_c - bracket_width, current_c + bracket_width, 1e-6);
        
        if (c_refined < 0) {
            // Brent failed? Try wider or panic
            // Maybe jumped too far?
             std::cerr << "Lost mode at f=" << current_freq << std::endl;
             break;
        }
        current_c = c_refined;
        
        // Calculate Group Velocity for next prediction
        T U = calculateGroupVelocityInternal(layers, thicknesses, halfSpace, omega, current_c, type);
        
        DispersionPoint p;
        p.frequency = current_freq;
        p.period = 1.0 / current_freq;
        p.phase_velocity = current_c;
        p.group_velocity = U;
        p.mode_index = mode_rank;
        curve.push_back(p);
        
        // Predict next
        if (i < num_steps - 1) {
            T next_freq = current_freq + df;
            T d_omega = 2.0 * M_PI * (next_freq - current_freq);
            
            // c_next = c + (dc/dw) * dw
            // dc/dw = (c/w) * (1 - c/U)
            T dc_dw = (current_c / omega) * (1.0 - current_c / U);
            T next_c_est = current_c + dc_dw * d_omega;
            
            current_c = next_c_est;
            current_freq = next_freq;
        }
    }
    
    return curve;
}

// --- Main (Verification) ---

// --- Sequential Mode Tracing ---

// Helper for ADS creation
template <typename T>
T calculateGroupVelocityInternal(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega,
    T c_phase,
    WaveType type
) {
    // To calculate Group Velocity U, we need dc/domega.
    // Implicit theorem: dc/domega = - (dF/domega) / (dF/dc)
    // U = c / (1 - (omega/c)*(dc/domega))
    
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    
    // Prepare variables with derivatives
    // 1. Calculate dF/domega (holding c constant)
    ADS omega_ad;
    omega_ad.value() = omega; // Assumes T is double-like
    omega_ad.derivatives() = Eigen::VectorXd::Unit(2, 0); // Index 0
    
    ADS c_const;
    c_const.value() = c_phase;
    c_const.derivatives() = Eigen::VectorXd::Zero(2);
    
    // AD-ify parameters
    // Helper lambda to make const AD from T
    auto mkad = [](T val, int sz) {
        ADS a; a.value() = val; a.derivatives() = Eigen::VectorXd::Zero(sz); return a;
    };
    
    PhysicalParameters<ADS> hs_ad; 
    hs_ad.rho = mkad(halfSpace.rho, 2);
    hs_ad.c11 = mkad(halfSpace.c11, 2);
    hs_ad.c13 = mkad(halfSpace.c13, 2);
    hs_ad.c33 = mkad(halfSpace.c33, 2);
    hs_ad.c44 = mkad(halfSpace.c44, 2);
    hs_ad.c66 = mkad(halfSpace.c66, 2);
    
    std::vector<PhysicalParameters<ADS>> layers_ad;
    std::vector<ADS> thick_ad;
    for(const auto& l : layers) {
        PhysicalParameters<ADS> la;
        la.rho = mkad(l.rho, 2);
        la.c11 = mkad(l.c11, 2);
        la.c13 = mkad(l.c13, 2);
        la.c33 = mkad(l.c33, 2);
        la.c44 = mkad(l.c44, 2);
        la.c66 = mkad(l.c66, 2);
        layers_ad.push_back(la);
    }
    for(const auto& t : thicknesses) {
        thick_ad.push_back(mkad(t, 2));
    }

    Eigen::Matrix<ADS, 5, 1> res_omega_vec = calculateDispersionFunction(layers_ad, thick_ad, hs_ad, omega_ad, c_const, type);
    double dF_dw = res_omega_vec(I24).derivatives()[0]; // Use I24 (Characteristic)

    // 2. Calculate dF/dc (holding omega constant)
    ADS c_ad;
    c_ad.value() = c_phase;
    c_ad.derivatives() = Eigen::VectorXd::Unit(2, 1); // Index 1
    
    ADS omega_const;
    omega_const.value() = omega;
    omega_const.derivatives() = Eigen::VectorXd::Zero(2);
    
    Eigen::Matrix<ADS, 5, 1> res_c_vec = calculateDispersionFunction(layers_ad, thick_ad, hs_ad, omega_const, c_ad, type);
    double dF_dc = res_c_vec(I24).derivatives()[1];
    
    if (std::abs(dF_dc) < 1e-15) return -1.0; 
    
    double dc_dw = -dF_dw / dF_dc;
    
    double c = c_phase;
    double w = omega;
    double U = c / (1.0 - (w/c) * dc_dw);
    
    return U;
}

// Wrapper for external API (backward compatibility or test usage)
template <typename T>
T calculateGroupVelocity(
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega,
    T c_phase
) {
    return calculateGroupVelocityInternal(layers, thicknesses, halfSpace, omega, c_phase, WaveType::Rayleigh);
}



#ifdef BUILD_MAIN
int main() {
    std::cout << "Disper26 Library. Run tests in tests/ directory." << std::endl;
    return 0;
}
#endif

#include "sensitivity_utils.hpp"
#endif // DISPER26_CPP
