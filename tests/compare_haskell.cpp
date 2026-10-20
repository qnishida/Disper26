#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <iomanip>

// Minimal types for verification
using T = double;
using CT = std::complex<T>;

struct PhysicalParameters {
    T rho;
    T c11, c13, c33, c44, c66;
};

struct WaveVariables {
    T omega;
    T k;
    T z;
};

struct DerivedComplexValues {
    CT d1, d2, d3, d4, d5, d6;
    CT C1, C3, S1, S3;
    CT eps1, eps3;
    CT nu1, nu3;
};

// --- Helper Functions (copied/adapted from disper26.cpp) ---

PhysicalParameters createIsotropicParameters(T vs, T vp, T rho) {
    PhysicalParameters p;
    p.rho = rho;
    p.c44 = rho * vs * vs;
    p.c33 = rho * vp * vp;
    p.c11 = p.c33;
    p.c13 = p.c33 - 2.0 * p.c44;
    p.c66 = p.c44;
    return p;
}

CT toComplex(T val) { return CT(val, 0.0); }
T get_val(T x) { return x; }

DerivedComplexValues computeDerivedValues(const PhysicalParameters& p, const WaveVariables& v) {
    T rho_omega2 = p.rho * v.omega * v.omega;
    T c44_k2     = p.c44 * v.k * v.k;
    T c11_k2     = p.c11 * v.k * v.k;
    T c13_plus_c44 = p.c13 + p.c44;
    T term_k_sq = v.k * v.k;

    CT M1 = toComplex(p.c44 * (rho_omega2 - c44_k2) + p.c33 * (rho_omega2 - c11_k2) + c13_plus_c44 * c13_plus_c44 * term_k_sq);
    CT M2 = toComplex((rho_omega2 - c44_k2) * (rho_omega2 - c11_k2));

    CT discriminant = M1 * M1 - 4.0 * toComplex(p.c33 * p.c44) * M2;
    CT sqrt_discriminant = std::sqrt(discriminant);

    CT numerator1 = -M1 + sqrt_discriminant;
    CT numerator3 = -M1 - sqrt_discriminant;
    CT denominator = 2.0 * toComplex(p.c33 * p.c44);

    CT nu1_sq = numerator1 / denominator;
    CT nu3_sq = numerator3 / denominator;

    CT nu1 = std::sqrt(nu1_sq);
    CT nu3 = std::sqrt(nu3_sq);
    
    // Enforce positive real part for decay definition consistency
    // (though for propagator it just needs to be consistent pair)
    if (nu1.real() < 0) nu1 = -nu1;
    if (nu3.real() < 0) nu3 = -nu3;

    auto compute_epsilon = [&](CT nu, CT nu_sq) {
        CT num = toComplex((p.c13 + p.c44) * v.k) * nu;
        CT den = toComplex(c11_k2 - rho_omega2) - toComplex(p.c44) * nu_sq;
        if (std::abs(den) < 1e-10) return toComplex(1.0); // Simplified
        return num / den;
    };

    CT eps1 = compute_epsilon(nu1, nu1_sq);
    CT eps3 = compute_epsilon(nu3, nu3_sq);

    DerivedComplexValues res;
    res.C1 = std::cosh(nu1 * v.z);
    res.S1 = std::sinh(nu1 * v.z);
    res.C3 = std::cosh(nu3 * v.z);
    res.S3 = std::sinh(nu3 * v.z);

    CT c33_c = toComplex(p.c33);
    CT c13_k_c = toComplex(p.c13 * v.k);

    res.d1 = c33_c * nu1 - c13_k_c * eps1;
    res.d2 = c33_c * nu3 - c13_k_c * eps3;
    CT k_c = toComplex(v.k);
    
    // Note: Disper26 d3/d4 definition: k + nu * eps
    // Let's verify if this matches the common VTI stress definitions
    // Sigma_zr = c44 * (du_r/dz + du_z/dr)
    // If u_r ~ exp(nu z), u_z ~ eps * exp(nu z) * i?
    // Let's stick to Disper26 definitions for now
    res.d3 = k_c + nu1 * eps1; 
    res.d4 = k_c + nu3 * eps3;
    
    res.d5 = nu1 * eps1 - nu3 * eps3;
    res.d6 = nu3 * eps1 - nu1 * eps3;

    res.eps1 = eps1;
    res.eps3 = eps3;
    res.nu1 = nu1;
    res.nu3 = nu3;
    return res;
}

// --- Dynamic Matrix Construction ---
Eigen::Matrix<CT, 4, 4> computeLayerMatrixDynamic(const PhysicalParameters& p, const WaveVariables& v_var) {
    DerivedComplexValues dv = computeDerivedValues(p, v_var);
    
    // Eigenvectors V
    // Based on d3, d4 being related to shear stress?
    // Let's deduce V columns from Disper26 d-values or Physical defs.
    // u_r (1), u_z (eps), S_zz (Sigma_zz), S_zr (Sigma_zr)
    
    // Mode 1 (+nu1):
    // u_r = 1
    // u_z = eps1
    // S_zr = c44 * (nu1 * u_r - k * u_z) ? No, typically shear strain is (du_r/dz + ik u_z) or similar
    // Check Disper26 d3? d3 = k + nu1 * eps1.
    // If S_zr = c44 * d3? Or c44 * (nu1 - k*eps1)?
    
    // Let's try to match the Explicit Formula structure:
    // A24 = 1/(c44 d5) * ...
    // This suggests inversion involves c44.
    
    // Hypothesis: The eigenvectors (unnormalized) are:
    // row 0 (u_r): 1
    // row 1 (u_z): eps
    // row 2 (S_zz): d1 (c33*nu - c13*k*eps) -> Matches Sigma_zz ~ c33 du_z/dz + c13 du_r/dr
    // row 3 (S_zr): c44 * d_shear.
    //   explicit A14 = (C1-C3)/(c44 d5).
    //   Inverse of V usually has 1/det(V).
    
    // Let's assume Sigma_zr is defined via c44 * (something).
    // Look at A41 = c44 d3 d4 / d5 ...
    // This implies row 3 is c44 * d3 (or d4?).
    // Actually, looking at explicit formulas:
    // A43 involves c44/c33...
    
    // Candidate Eigenvector for mode i (nu_i, eps_i):
    // v_i = [ 1,  eps_i,  D1_i,  C44 * D3_i ]^T
    // Where D1_i = c33 * nu_i - c13 * k * eps_i  (This matches d1/d2)
    //       D3_i = nu_i - k * eps_i  (This would be d5-like?)
    // Warning: Disper26's d3 = k + nu * eps.
    
    // Let's reverse engineer from explicit formulas for A12:
    // A12 = 1/(c33 d6) ( -eps3 S1 + eps1 S3 )
    // This looks like part of the inverse V.
    
    // Let's try the Dynamic construction using:
    // V = [ v1+, v1-, v3+, v3- ]
    // v(nu) = [ 1, eps, d_normal, d_shear ]^T
    // d_normal = c33*nu - c13*k*eps
    // d_shear = c44*(nu - k*eps) ??? (Check physics)
    //   strain_rz = du_r/dz + du_z/dr = nu*1 - k*eps (if exp(i(kx+nu z)))
    //   Yes, likely c44*(nu - k*eps).
    // Let's see if this definition matches the explicit formula.
    
    CT nu1 = dv.nu1;
    CT nu3 = dv.nu3;
    CT eps1 = dv.eps1;
    CT eps3 = dv.eps3;
    CT k = toComplex(v_var.k);
    
    auto compute_stress = [&](CT nu, CT eps) {
        // Sigma_zz
        CT s_zz = toComplex(p.c33) * nu - toComplex(p.c13) * k * eps;
        // Sigma_zr
        CT s_zr = toComplex(p.c44) * (nu + k * eps); // Wait, + or -? 
        // d3 = k + nu*eps.
        // If s_zr = c44 * d3?
        return std::make_pair(s_zz, s_zr);
    };

    // Construct V
    Eigen::Matrix<CT, 4, 4> V;
    
    // Col 0: +nu1
    auto s1 = compute_stress(nu1, eps1);
    V(0,0) = 1.0; V(1,0) = eps1; V(2,0) = s1.first; V(3,0) = toComplex(p.c44) * (nu1 + k * eps1); // Hypothesis: d3-like

    // Col 1: -nu1
    // eps(-nu) = -eps(nu) ?
    // Check compute_epsilon(-nu, nu^2):
    // num ~ -nu, den same. So eps(-nu) = -eps(nu).
    // s_zz(-nu, -eps) = c33(-nu) - c13 k (-eps) = -(c33 nu - c13 k eps) = -s_zz
    // s_zr(-nu, -eps) = c44(-nu + k(-eps)) = -c44(nu + k eps) = -s_zr
    
    // Actually, hyperbolic functions C and S come from exp(nu z) and exp(-nu z).
    // Propgator = V E V^-1.
    // Let's use ordered basis: [Reduced form]
    // V = [ v1, v3, v1_bar, v3_bar ]? No, easier to use cosh/sinh basis implicitly or explicit V.
    // Explicit V with cols: (+nu1), (-nu1), (+nu3), (-nu3).
    
    CT eps1_n = -eps1;
    CT nu1_n = -nu1;
    auto s1_n = compute_stress(nu1_n, eps1_n);
    V(0,1) = 1.0; V(1,1) = eps1_n; V(2,1) = s1_n.first; V(3,1) = toComplex(p.c44) * (nu1_n + k * eps1_n);

    // Col 2: +nu3
    auto s3 = compute_stress(nu3, eps3);
    V(0,2) = 1.0; V(1,2) = eps3; V(2,2) = s3.first; V(3,2) = toComplex(p.c44) * (nu3 + k * eps3);

    // Col 3: -nu3
    CT eps3_n = -eps3;
    CT nu3_n = -nu3;
    auto s3_n = compute_stress(nu3_n, eps3_n);
    V(0,3) = 1.0; V(1,3) = eps3_n; V(2,3) = s3_n.first; V(3,3) = toComplex(p.c44) * (nu3_n + k * eps3_n);
    
    // Propagator Lambda
    Eigen::Matrix<CT, 4, 4> Lam;
    Lam.setZero();
    Lam(0,0) = std::exp(nu1 * v_var.z);
    Lam(1,1) = std::exp(-nu1 * v_var.z);
    Lam(2,2) = std::exp(nu3 * v_var.z);
    Lam(3,3) = std::exp(-nu3 * v_var.z);
    
    Eigen::Matrix<CT, 4, 4> B = V * Lam * V.inverse();
    return B;
}

// --- Explicit Matrix Construction (from Markdown) ---
Eigen::Matrix<CT, 4, 4> computeLayerMatrixExplicit(const PhysicalParameters& p, const WaveVariables& v_var) {
    DerivedComplexValues dv = computeDerivedValues(p, v_var);
    
    CT d1 = dv.d1; CT d2 = dv.d2;
    CT d3 = dv.d3; CT d4 = dv.d4;
    CT d5 = dv.d5; CT d6 = dv.d6;
    CT C1 = dv.C1; CT C3 = dv.C3;
    CT S1 = dv.S1; CT S3 = dv.S3;
    CT eps1 = dv.eps1; CT eps3 = dv.eps3;
    CT c33 = toComplex(p.c33);
    CT c44 = toComplex(p.c44);
    
    Eigen::Matrix<CT, 4, 4> A;
    
    // A11 = 1/d5 * (-d4 C1 + d3 C3)
    A(0,0) = 1.0/d5 * (-d4 * C1 + d3 * C3);
    
    // A12 = 1/(c33 d6) * (-eps3 S1 + eps1 S3)
    A(0,1) = 1.0/(c33 * d6) * (-eps3 * S1 + eps1 * S3);
    
    // A13 = 1/(c33 d6) * (d2 S1 - d1 S3)
    A(0,2) = 1.0/(c33 * d6) * (d2 * S1 - d1 * S3);
    
    // A14 = 1/(c44 d5) * (C1 - C3)
    A(0,3) = 1.0/(c44 * d5) * (C1 - C3);
    
    // A21 = 1/d5 * (-d1 d4 S1 + d2 d3 S3)
    A(1,0) = 1.0/d5 * (-d1 * d4 * S1 + d2 * d3 * S3);
    
    // A22 = A11
    A(1,1) = A(0,0);
    
    // A23 = d1 d2 / (c33 d6) * (C1 - C3)
    A(1,2) = d1 * d2 / (c33 * d6) * (C1 - C3);
    
    // A24 = 1/(c44 d5) * (d1 S1 - d2 S3)
    A(1,3) = 1.0/(c44 * d5) * (d1 * S1 - d2 * S3);
    
    // A31 = -A24
    A(2,0) = -A(1,3); // Check: 1/d5 (-eps1 d4 S1 + eps3 d3 S3) ?
    // Markdown says: A31 = 1/d5 (-eps1 d4 S1 + eps3 d3 S3) = -A24?
    // Let's implement explicit formula to be safe
    // A31 = 1/d5 * (-eps1*d4*S1 + eps3*d3*S3)
    A(2,0) = 1.0/d5 * (-eps1 * d4 * S1 + eps3 * d3 * S3); 
    
    // A32 = -A14?
    // Markdown: A32 = eps1 eps3 / (c33 d6) * (-C1 + C3) = -A14? No.
    // Wait, A14 = 1/(c44 d5) (C1-C3).
    // Markdown formula: eps1 eps3 / (c33 d6) (-C1 + C3).
    // Unless relations between d5, d6, eps, c33, c44 exist.
    A(2,1) = eps1 * eps3 / (c33 * d6) * (-C1 + C3);
    
    // A33 = 1/(c33 d6) * (eps1 d2 C1 - eps3 d1 C3)
    A(2,2) = 1.0/(c33 * d6) * (eps1 * d2 * C1 - eps3 * d1 * C3);
    
    // A34 = 1/(c44 d5) * (eps1 S1 - eps3 S3)
    A(2,3) = 1.0/(c44 * d5) * (eps1 * S1 - eps3 * S3);
    
    // A41 = c44 d3 d4 / d5 * (-C1 + C3) = -A23?
    A(3,0) = c44 * d3 * d4 / d5 * (-C1 + C3);
    
    // A42 = c44 / (c33 d6) * (-eps3 d3 S1 + eps1 d4 S3) = -A13?
    A(3,1) = c44 / (c33 * d6) * (-eps3 * d3 * S1 + eps1 * d4 * S3);
    
    // A43 = c44 / (c33 d6) * (d2 d3 S1 - d1 d4 S3)
    A(3,2) = c44 / (c33 * d6) * (d2 * d3 * S1 - d1 * d4 * S3);
    
    // A44 = 1/d5 (d3 C1 - d4 C3) = A33?
    A(3,3) = 1.0/d5 * (d3 * C1 - d4 * C3);
    
    return A;
}

int main() {
    T vs = 1.0;
    T vp = 1.732;
    T rho = 1.0;
    PhysicalParameters p = createIsotropicParameters(vs, vp, rho);
    
    WaveVariables v;
    v.omega = 1.0;
    v.z = 1.0;
    
    std::cout << "Starting Haskell Matrix Verification..." << std::endl;
    printf("   C      Diff_Norm     |A11_d - A11_e|   Cond(V)\n");
    
    // Sweep Phase Velocity
    for (double c = 0.5; c < 2.5; c += 0.1) {
        if (std::abs(c - vs) < 0.01) continue; // Avoid singularity for now
        
        v.k = v.omega / c;
        
        Eigen::Matrix<CT, 4, 4> Adyn = computeLayerMatrixDynamic(p, v);
        Eigen::Matrix<CT, 4, 4> Aexp = computeLayerMatrixExplicit(p, v);
        
        double diff = (Adyn - Aexp).norm();
        double a11_diff = std::abs(Adyn(0,0) - Aexp(0,0));

        // Print detail for first valid case
        if (c > 1.05 && c < 1.15) {
            DerivedComplexValues dv = computeDerivedValues(p, v);
            std::cout << "--- Detail for C = " << c << " ---" << std::endl;
            std::cout << "d5: " << dv.d5 << " d6: " << dv.d6 << std::endl;
            std::cout << "A_dynamic:\n" << Adyn << std::endl;
            std::cout << "A_explicit:\n" << Aexp << std::endl;
            // Check V
             CT nu1 = dv.nu1;
            CT nu3 = dv.nu3;
            CT eps1 = dv.eps1;
            CT eps3 = dv.eps3;
            CT k = toComplex(v.k);
            std::cout << "V cols (approx):" << std::endl;
            std::cout << "Col0 (nu1): 1, " << eps1 << ", " << (toComplex(p.c33)*nu1 - toComplex(p.c13)*k*eps1) << ", " << (toComplex(p.c44)*(nu1+k*eps1)) << std::endl;
        }

        printf("%6.2f   %10.2e      %10.2e\n", c, diff, a11_diff);
    }
    
    return 0;
}
