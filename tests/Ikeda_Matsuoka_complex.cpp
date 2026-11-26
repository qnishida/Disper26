#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Type definitions.
template <typename T>
struct PhysicalParameters {
    T rho;
    T c11, c13, c33, c44;
};

template <typename T>
struct WaveVariables {
    T omega;
    T k;
    T z;
};

template <typename T>
struct DerivedComplexValues {
    std::complex<T> d1, d2, d3, d4, d5, d6;
    std::complex<T> C1, C3, S1, S3;
    std::complex<T> eps1, eps3; 
};

enum ReducedIndex { I12=0, I13=1, I14=2, I23=3, I24=4 };

// T(0.0) has an empty derivative vector for a dynamically sized AutoDiffScalar.
// Use val * 0.0 so the imaginary part has the same derivative size as val.
template<typename T>
std::complex<T> toComplex(const T& val) {
    return std::complex<T>(val, val * 0.0);
}

// Derived wave quantities.

template <typename T>
DerivedComplexValues<T> computeDerivedValues(
    const PhysicalParameters<T>& p,
    const WaveVariables<T>& v
) {
    using CT = std::complex<T>;
    
    T rho_omega2 = p.rho * v.omega * v.omega;
    T c44_k2     = p.c44 * v.k * v.k;
    T c11_k2     = p.c11 * v.k * v.k;
    
    T c13_plus_c44 = p.c13 + p.c44;
    T term_k_sq = v.k * v.k;

    // Start from p.rho * 0.0 so the constants inherit its derivative size.
    T ref_zero = p.rho * 0.0; 
    T val_4 = ref_zero + 4.0;
    T val_2 = ref_zero + 2.0;

    CT M1 = toComplex<T>(
          p.c44 * (rho_omega2 - c44_k2) 
        + p.c33 * (rho_omega2 - c11_k2) 
        + (c13_plus_c44 * c13_plus_c44) * term_k_sq
    );

    CT M2 = toComplex<T>(
        (rho_omega2 - c44_k2) * (rho_omega2 - c11_k2)
    );

    // Compute nu1 and nu3 using val_4 in the discriminant.
    CT discriminant = M1 * M1 - toComplex<T>(val_4 * p.c33 * p.c44) * M2;
    CT sqrt_discriminant = std::sqrt(discriminant);

    CT numerator1 = -M1 + sqrt_discriminant;
    CT numerator3 = -M1 - sqrt_discriminant;
    CT denominator = toComplex<T>(val_2 * p.c33 * p.c44);

    CT nu1_sq = numerator1 / denominator;
    CT nu3_sq = numerator3 / denominator;

    CT nu1 = std::sqrt(nu1_sq);
    CT nu3 = std::sqrt(nu3_sq);

    // Compute epsilon for each root.
    auto compute_epsilon = [&](CT nu, CT nu_sq) {
        CT num = toComplex<T>((p.c13 + p.c44) * v.k) * nu;
        CT den_real_part = toComplex<T>(c11_k2 - rho_omega2);
        CT den_complex_part = toComplex<T>(p.c44) * nu_sq;
        return num / (den_real_part - den_complex_part);
    };

    CT eps1 = compute_epsilon(nu1, nu1_sq);
    CT eps3 = compute_epsilon(nu3, nu3_sq);

    DerivedComplexValues<T> res;
    res.C1 = std::cosh(nu1 * toComplex(v.z));
    res.S1 = std::sinh(nu1 * toComplex(v.z));
    res.C3 = std::cosh(nu3 * toComplex(v.z));
    res.S3 = std::sinh(nu3 * toComplex(v.z));

    CT c33_c = toComplex(p.c33);
    CT c13_k_c = toComplex<T>(p.c13 * v.k);

    res.d1 = c33_c * nu1 - c13_k_c * eps1;
    res.d2 = c33_c * nu3 - c13_k_c * eps3;
    res.d3 = toComplex(v.k) * nu1 * eps1;
    res.d4 = toComplex(v.k) * nu3 * eps3;
    res.d5 = nu1 * eps1 - nu3 * eps3;
    res.d6 = nu3 * eps1 - nu1 * eps3;

    res.eps1 = eps1;
    res.eps3 = eps3;

    return res;
}

template <typename T>
Eigen::Matrix<T, 5, 5> computeMatrixFromPhysical(
    const PhysicalParameters<T>& physParams,
    const WaveVariables<T>& waveVars
) {
    using CT = std::complex<T>;

    DerivedComplexValues<T> v = computeDerivedValues(physParams, waveVars);

    const CT& d1 = v.d1; const CT& d2 = v.d2;
    const CT& d3 = v.d3; const CT& d4 = v.d4;
    const CT& d5 = v.d5; const CT& d6 = v.d6;
    const CT& C1 = v.C1; const CT& C3 = v.C3;
    const CT& S1 = v.S1; const CT& S3 = v.S3;

    CT C1C3 = C1 * C3;
    CT S1S3 = S1 * S3;

    // Give the constants 1.0 and 2.0 the same derivative size as physParams.rho.
    T ref_zero = physParams.rho * 0.0;
    T val_1 = ref_zero + 1.0;
    T val_2 = ref_zero + 2.0;

    CT one = toComplex<T>(val_1);
    CT two = toComplex<T>(val_2);

    CT inv_d5 = one / d5;
    CT inv_d6 = one / d6;
    CT inv_c33 = one / toComplex(physParams.c33);
    CT inv_c44 = one / toComplex(physParams.c44);

    CT k_33_d5d6 = inv_c33 * inv_d5 * inv_d6;
    CT k_all     = inv_c33 * inv_c44 * inv_d5 * inv_d6;

    CT term1 = two * d1 * d2 * (C1C3 - one);
    CT term2 = (d1*d1 + d2*d2) * S1S3;
    CT b1234 = k_all * (term1 - term2);
    
    CT b1213 = k_33_d5d6 * (
        d1 * d2 * (d3 + d4) * (C1C3 - one) -
        (d2*d2 * d3 + d1*d1 * d4) * S1S3
    );

    Eigen::Matrix<T, 5, 5> B;
    B.setZero();

    // Store the real part of each matrix entry.
    B(I12, I12) = (one - two * b1234).real(); 
    B(I24, I12) = (two * b1213).real();

    return B;
}

int main() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;

    PhysicalParameters<ADS> phys;
    // Create an AutoDiffScalar with value v and one zero derivative.
    auto make_const = [](double v) {
        ADS a; 
        a.value() = v; 
        a.derivatives() = Eigen::VectorXd::Zero(1);
        return a;
    };

    phys.rho = make_const(2.0);
    phys.c11 = make_const(20.0);
    phys.c13 = make_const(5.0);
    phys.c33 = make_const(15.0);
    phys.c44 = make_const(10.0);

    WaveVariables<ADS> wave;
    wave.z = make_const(1.5);
    wave.k = make_const(0.5);

    // Differentiate with respect to omega by seeding d(omega)/d(omega) = 1.
    wave.omega.value() = 10.0;
    wave.omega.derivatives() = Eigen::VectorXd::Ones(1);

    Eigen::Matrix<ADS, 5, 5> B = computeMatrixFromPhysical(phys, wave);

    std::cout << "B(0,0) Value: " << B(0,0).value() << std::endl;
    std::cout << "B(0,0) Derivative: " << B(0,0).derivatives()[0] << std::endl;

    return 0;
}
