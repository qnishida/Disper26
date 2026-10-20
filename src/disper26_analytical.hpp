#ifndef DISPER26_ANALYTICAL_HPP
#define DISPER26_ANALYTICAL_HPP

#include "disper26.cpp" // Include main file for types and templates (PhysicalParameters, etc.)

// ===========================================================================
// Analytical Energy Integral Implementation (Pancake Model)
// ===========================================================================
// This module implements exact analytical integration of energy integrals
// for homogeneous layers, eliminating the need for numerical integration (Romberg).
// Finite layers share regular quadratic moments with variational derivatives;
// noncritical and halfspace modes retain exponential analytical integration.
// ===========================================================================

namespace Analytical {

// Helper: Exact integral of Exp((nu_i + nu_j) * z) from 0 to d
template <typename T>
std::complex<T> integrate_exp(std::complex<T> nu_sum, T d) {
    return d * ad_phi1(std::complex<T>(nu_sum*d));
}

// ===========================================================================
// Love Wave Analytical Integration
// ===========================================================================
// Eigenmodes:
// Up-going:   u_y ~ exp(nu * z)    (Index 0)
// Down-going: u_y ~ exp(-nu * z)   (Index 1)
// Only the modal integrals in this section use a local depth x positive down.
// Their state is [uy,tau_depth], tau_depth=c44*du/dx=-T. Propagated SH states
// and LoveEigenfunctionPoint.tyz both use T=c44*du/dz, z positive upward.
// The z in the exponential formulas below denotes this local integration x.
// E = [ [1, 1], [mu*nu, -mu*nu] ]  (Check signs of nu)
// nu = k * sqrt(1 - c^2/beta^2)
// For Love (SH), tau = c44 * du/dz.
// If u ~ exp(nu*z), du/dz = nu*exp(nu*z), tau = c44*nu*u.
// If u ~ exp(-nu*z), du/dz = -nu*exp(-nu*z), tau = -c44*nu*u.
// E matrix (Cols: Up(0), Down(1)):
// E = | 1        1      |
//     | +mu*nu  -mu*nu  |
// Inverse E^-1:
// Det = -2*mu*nu
// E^-1 = (1/-2mu*nu) * | -mu*nu  -1 |
//                      | -mu*nu   1 |
//      = 0.5 * | 1   1/(mu*nu) |
//              | 1  -1/(mu*nu) |
// Coefficients C = [c_up, c_down]^T = E^-1 * y_top. (y_top at z=0 relative to layer)

template <typename T>
EnergyIntegralResult<T> computeLoveEnergyIntegrals_Analytical(
    const std::vector<LoveEigenfunctionPoint<T>>& eigenfunctions, // Only used for y_top at each layer
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity
) {
    using CT = std::complex<T>;
    EnergyIntegralResult<T> res; 
    res.I1 = T(0.0); res.I2 = T(0.0); res.I3 = T(0.0); 
    res.phaseVelocity = phaseVelocity;
    
    T k = omega / phaseVelocity; 
    T k_sq = k * k;

    // Loop over layers
    // Note: eigenfunctions vector has points at interfaces + sublayers if numerical computed.
    // If we only have interface points, we assume layer mapping is direct.
    // Implementation constraint: The "eigenfunctions" input must correspond to the layer boundaries.
    // The standard output stores points at depth 0, d1, d1+d2, ...
    // We access eigenfunctions[i] which corresponds to Top of Layer i.
    // Wait, standard output includes sublayers. We need to pick the correct points.
    // Strategy: Re-propagate to find y_top at each layer? Or search depth?
    // Searching depth is safer.
    
    int processed_layers = 0;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        // Find y_top from eigenfunctions list
        T top_depth = 0.0;
        for(int l=0; l<i; ++l) top_depth += thicknesses[l];
        
        // Linear search for point matches top_depth (approx)
        int idx = -1;
        for(size_t j=0; j<eigenfunctions.size(); ++j) {
            if (std::abs(eigenfunctions[j].depth - top_depth) < 1e-6) {
                idx = j; break;
            }
        }
        if (idx == -1) continue; // Should not happen
        
        auto pt = eigenfunctions[idx];
        CT uy_top = pt.uy;
        // Convert the public upward stress to the local depth-derivative stress
        // used by E. Without this sign, nonzero interface tractions exchange
        // the growing and decaying amplitudes in each finite layer.
        CT tau_depth_top = -pt.tyz;
        
        PhysicalParameters<T> p = layers[i];
        T d = thicknesses[i];
        
        CT u2, du2;
        if(disper26_energy::love_moments(p,d,omega,k,uy_top,pt.tyz,u2,du2)) {
            res.I1+=(p.rho*u2).real();
            res.I2+=(p.c44*du2+k_sq*p.c66*u2).real();
            res.I3+=(p.c66*u2).real();
            continue;
        }

        // 1. Calculate Eigenvalues (nu)
        // nu = k * sqrt(1 - c^2 * rho / c44) ? No.
        // Love disp eq: beta = sqrt(c44/rho). 
        // mu = c44. (VTI L)
        // nu = k * sqrt(1 - c^2/beta^2) is for isotropic.
        // For VTI: nu calculation.
        // wave.z = 0 implies homogeneous coeff.
        // From computeLayerMatrixSH:
        // nu = k * sqrt(1 - c^2 * rho / p.c44_or_c66?)
        // Let's copy logic from disper26.cpp strictly.
        // c44 (L, vertical shear) governs vertical decay. c66 (N) governs horizontal.
        // Wave eq: L u_zz + N k^2 u = -rho omega^2 u?
        // -rho omega^2 + L nu^2 - N k^2 = 0 ?
        // L nu^2 = N k^2 - rho omega^2
        // nu^2 = (N k^2 - rho omega^2) / L
        // nu = sqrt(...)
        
        T L = p.c44;
        T N = p.c66;
        T rho = p.rho;
        CT nu_sq_val = (N * k_sq - rho * omega * omega) / L;
        CT nu = std::sqrt(nu_sq_val); // Standard sqrt branch?
        
        // Ensure nu matches the branch used in propagation (Re(nu) >= 0 usually)
        if (nu.real() < 0) nu = -nu;
        
        // 2. Formulate E and E^-1
        // Modes: 0: Up (+nu), 1: Down (-nu)
        // Let's stick to exp(nu*z) and exp(-nu*z).
        // If Re(nu)>0, exp(-nu*z) is decaying (Down), exp(nu*z) is growing (Up).
        // E = [ [1, 1], [L*nu, -L*nu] ]
        // E^-1 = 0.5 * [ [1, 1/(L*nu)], [1, -1/(L*nu)] ]
        
        CT term2 = T(1.0) / (L * nu);
        CT c_up = T(0.5) * (uy_top + tau_depth_top * term2);
        CT c_dn = T(0.5) * (uy_top - tau_depth_top * term2);
        
        CT coeffs[2] = {c_up, c_dn};
        CT nus[2] = {nu, -nu};
        
        // 3. Analytic Integrals
        // I1 = rho * integral(uy^2)
        // I2 = integral( (tyz^2/L) + k^2 * N * uy^2 )
        // I3 = integral( N * uy^2 )  (Wait, I3 formula for Love is N * uy^2 ?)
        // Check docs/energy_integral.md or disper26.cpp computeLoveEnergyIntegrals
        // disper26.cpp:
        // I1 = rho * uy^2
        // I2 = (tyz^2/c44) + k^2 * c66 * uy^2
        // I3 = c66 * uy^2
        
        // We integrate uy^2 and tyz^2 analytically.
        // uy(z) = Sum_i c_i exp(nu_i z)
        // tyz(z) = Sum_i c_i E(1,i) exp(nu_i z) = Sum_i c_i * (L*nu_i) * exp(nu_i z) (Sign depends on i)
        // Note: E(1,0) = L*nu, E(1,1) = -L*nu. So factor is L*nu_i.
        
        CT int_uy2 = CT(0.0);
        CT int_tyz2 = CT(0.0);
        
        for(int m=0; m<2; ++m) {
            for(int n=0; n<2; ++n) {
                // Interaction between mode m and n
                CT exp_int = integrate_exp(nus[m] + nus[n], d);
                
                // Term in uy^2: c_m * c_n * exp_int
                CT term_uy = coeffs[m] * coeffs[n] * exp_int;
                int_uy2 += term_uy;
                
                // Term in tyz^2: (c_m * L*nu_m) * (c_n * L*nu_n) * exp_int
                CT fact_m = (m==0) ? (L*nu) : (-L*nu);
                CT fact_n = (n==0) ? (L*nu) : (-L*nu);
                CT term_tyz = (coeffs[m] * fact_m) * (coeffs[n] * fact_n) * exp_int;
                int_tyz2 += term_tyz;
            }
        }
        
        // Add to Totals
        // Note: Integrals are complex, but results should be real-ish.
        // Disper26 uses Real part of results?
        // computeLoveEnergyIntegrals: I1 += (v1[0] + v2[0]) ...
        // v1[0] = rho * uy.real() * uy.real().
        // WAIT.
        // The numerical integration uses [ Real(uy)^2 ].
        // The analytical integration computes integral of [ uy^2 ] (Complex squared).
        // If uy is real (standing wave?), uy^2 = Real(uy)^2.
        // But in decaying region, uy is real. In propagating, uy is complex.
        // The Energy Integral definition usually involves Conjugates?
        // "I1 = rho * |uy|^2" vs "rho * uy^2"?
        // Disper80C: F[1] = RHO*(SQR(Y[1]) + ...) -> SQR is x*x.
        // So it integrates the SQUARE (not modulus).
        // So Py_real^2 - Py_imag^2?
        // Actually disper26.cpp lines 1140:
        // T y1 = pt.uz.real(); ...
        // T i1 = p.rho * (y1*y1 + y3*y3);
        // It takes the REAL PART of the eigenfunction first, then squares it.
        // This is important. u(z,t) = Re[ U(z) exp(i(wt-kx)) ].
        // Time averaged energy involves |U|^2 if complex?
        // Disper80C functions are REAL-valued (Y array is double).
        // But my VTI code uses Complex arithmetic.
        // However, for pure propagation (no attenuation), the eigenfunctions can be made Real (or purely Imaginary phase shifted).
        // The existing numerical code does: `pt.uz.real()`.
        // So it effectively integrates `Real(u(z))^2`.
        
        // Analytic Problem: `Real( Sum c_i exp(nu_i z) )^2`.
        // = `(Sum Re(c_i e..))^2`.
        // This is Harder to integrate analytically than `u^2` complex.
        // But wait.
        // If U is properly normalized, U is Real. (Standing wave in z).
        // In the half-space, it decays (Real exponential).
        // In the layer, it might be sin/cos (Real).
        // So U SHOULD be Real-valued (or can be made so).
        // Let's check `computeRayleighEigenfunctions` output.
        // `state` is `Eigen::Matrix<CT, 4, 1>`.
        // If we force `state` to be real at surface, and matrix is real (no attenuation), state remains real.
        // So `int_uy2` (complex) should have zero imaginary part.
        // And `Real(int_uy2)` should equal `Integral(Real(u)^2)`.
        // So I can use the complex integral and take the Real part.
        
        res.I1 += rho * int_uy2.real();
        res.I2 += (int_tyz2.real() / L) + k_sq * N * int_uy2.real();
        res.I3 += N * int_uy2.real();
    }
    
    // Half-Space (Analytic)
    // Mode 1 (Down-going, decaying) only.
    // u_y = c_dn * exp(-nu * z) (z is depth into halfspace)
    // y_top_half = eigenvectors.back() (should match bottom of stack)
    // But we can just use the last computed y_bottom from the layer loop?
    // Or extract from eigenfunction list again.
    
    // Finding y_top_half
    T total_depth = 0; for(auto t:thicknesses) total_depth += t;
    int idx_half = -1;
    for(size_t j=0; j<eigenfunctions.size(); ++j) {
        if (std::abs(eigenfunctions[j].depth - total_depth) < 1e-6) {
            idx_half = j; break;
        }
    }
    
    if (idx_half != -1) {
        auto pt = eigenfunctions[idx_half];
        CT uy0 = pt.uy;
        CT tyz0 = pt.tyz;
        PhysicalParameters<T> p = halfSpaceParams;
        T L = p.c44; T N = p.c66; T rho = p.rho;
        CT nu_sq_val = (N * k_sq - rho * omega * omega) / L;
        CT nu = std::sqrt(nu_sq_val);
        if (nu.real() < 0) nu = -nu;
        
        // Only decaying mode exists.
        // u(z) = u0 * exp(-nu * z)
        // With local depth increasing downward, public tyz0=+L*nu*u0.
        // If not, it means we have bits of growing mode (numerical error).
        // We project onto decaying mode? Or just use u0?
        // Using u0 is safer.
        // Integral[ exp(-2*nu*z) ] from 0 to infinity = 1 / (2*nu).
        // Assuming nu is real positive.
        
        CT int_fac = T(1.0) / (T(2.0) * nu);
        CT int_uy2 = uy0 * uy0 * int_fac;
        CT int_tyz2 = tyz0 * tyz0 * int_fac; 
        // Ideally tyz0^2 = (L*nu*u0)^2.
        
        res.I1 += rho * int_uy2.real();
        res.I2 += (int_tyz2.real() / L) + k_sq * N * int_uy2.real();
        res.I3 += N * int_uy2.real();
    }
    
    return res;
}

// (Continued in namespace)

// ===========================================================================
// Rayleigh Wave Analytical Integration
// ===========================================================================

template <typename T>
EnergyIntegralResult<T> computeRayleighEnergyIntegrals_Analytical(
    const std::vector<RayleighEigenfunctionPoint<T>>& eigenfunctions,
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpaceParams,
    T omega,
    T phaseVelocity
) {
    // Solid layers only -- the analytical form divides by c44 and assumes the
    // P-SV eigenfunction shape, and a water layer has c44 = 0. Unguarded this
    // returned NaN with nothing to say why. See docs/fluid_gravity_status.md.
    if (hasLiquidLayer(layers) || isLiquidLayer(halfSpaceParams)) {
        throw std::runtime_error(
            "Analytical Rayleigh energy integrals are not implemented for fluid "
            "layers. Use compute_energy_integrals (trapezoidal), which handles "
            "fluid models.");
    }

    using CT = std::complex<T>;
    using Matrix4 = Eigen::Matrix<CT, 4, 4>;
    using Vector4 = Eigen::Matrix<CT, 4, 1>;

    EnergyIntegralResult<T> res;
    res.I1 = T(0.0); res.I2 = T(0.0); res.I3 = T(0.0); 
    res.phaseVelocity = phaseVelocity;
    
    T k = omega / phaseVelocity; 
    T k_sq = k * k;
    WaveVariables<T> wave; wave.omega = omega; wave.k = k; wave.z = T(0);

    for (size_t i = 0; i < layers.size(); ++i) {
        // 1. Find y_top (State at top of layer)
        T top_depth = 0.0; for(int l=0; l<i; ++l) top_depth += thicknesses[l];
        int idx = -1;
        for(size_t j=0; j<eigenfunctions.size(); ++j) {
            if (std::abs(eigenfunctions[j].depth - top_depth) < 1e-6) { idx = j; break; }
        }
        if (idx == -1) continue;
        
        auto pt = eigenfunctions[idx];
        // Ikeda: y1=uz, y2=szz, y3=ur, y4=szr
        // But the input struct is {ur, uz, szz, szr}
        // Let's stick to the struct fields and map them to physical Y.
        // Y_vec = [uz, szz, ur, szr]^T (Standard 4x4 Op order)
        Vector4 y_top;
        y_top << pt.uz, pt.szz, pt.ur, pt.szr;
        
        PhysicalParameters<T> p = layers[i];
        T d = thicknesses[i];
        
        Eigen::Matrix<CT,4,4> moments;
        disper26_energy::rayleigh_moments(p,d,omega,k,
            std::array<CT,4>{y_top(0),y_top(1),y_top(2),y_top(3)},moments);
        const T aa=p.c11-p.c13*p.c13/p.c33;
        res.I1+=(p.rho*(moments(0,0)+moments(2,2))).real();
        res.I2+=(moments(1,1)/p.c33+moments(3,3)/p.c44+k_sq*aa*moments(2,2)).real();
        res.I3+=(aa*moments(2,2)+(moments(0,3)-p.c13/p.c33*moments(1,2))/k).real();
    }
    
    // Half-Space (Analytic)
    // Only Down-going modes (0 and 1).
    // Coefficients of Up-going (2, 3) must be zero.
    // Solve for C0, C1 using y_top_half.
    // y_top = C0*V0 + C1*V1.
    // We have 4 equations, 2 unknowns?
    // This is overdetermined if we just take y_top_half from previous loop.
    // But `y_top_half` IS constructed from V0, V1 in the Eigenfunction Pass.
    // So it should be consistent.
    
    T total_depth = 0; for(auto t:thicknesses) total_depth += t;
    int idx_half = -1;
    for(size_t j=0; j<eigenfunctions.size(); ++j) {
        if (std::abs(eigenfunctions[j].depth - total_depth) < 1e-6) { idx_half = j; break; }
    }
    
    if (idx_half != -1) {
        auto pt = eigenfunctions[idx_half];
        Vector4 y_half; y_half << pt.uz, pt.szz, pt.ur, pt.szr;
        
        DerivedComplexValues<T> dv = computeDerivedValues(halfSpaceParams, wave);
        // E down (2 cols)
        Eigen::Matrix<CT, 4, 2> E_dn;
        // Col0: [1, d1, eps1, d3*c44]
        // Col1: [1, d2, eps3, d4*c44]
        E_dn << 1, 1,
                dv.d1, dv.d2,
                dv.eps1, dv.eps3,
                dv.d3*toComplex(halfSpaceParams.c44), dv.d4*toComplex(halfSpaceParams.c44);
                
        // Solve C = (E^T E)^-1 E^T y  (Least Squares)
        // Or just picking first 2 rows?
        // Let's use HouseholderQR for stability.
        Eigen::Matrix<CT, 2, 1> C = E_dn.colPivHouseholderQr().solve(y_half);
        
        CT coeffs[2] = {C(0), C(1)};
        CT nus[2] = {-dv.nu1, -dv.nu3}; 
        // nu1, nu3 are positive (sqrt branch). Exponents are -nu*z.
        // Wait, dv.nu1 is from `computeDerivedValues`.
        // Usually Re(nu) > 0.
        // Exponent is -nu*z.
        // Nus for integration formula: (-nu1, -nu3).
        
        // Reuse loop logic (m,n in 0..1)
        PhysicalParameters<T> p = halfSpaceParams;
        T term_A = p.c11 - p.c13*p.c13/p.c33;
        
        CT val_I1 = CT(0); CT val_I2 = CT(0); CT val_I3 = CT(0);
        
        for(int m=0; m<2; ++m) {
            for(int n=0; n<2; ++n) {
                // Integral from 0 to infinity of exp( (nu_m + nu_n) * z ) ??
                // Wait. Exponent is (-nu_m - nu_n) * z. (nu_m > 0).
                // Integral exp(-s z) = 1/s.
                // s = nu_m + nu_n.
                // Result: 1 / (nu_m + nu_n).
                
                CT sum_nu = dv.nu1; // Placeholder
                if (m==0) sum_nu = dv.nu1; else sum_nu = dv.nu3;
                if (n==0) sum_nu += dv.nu1; else sum_nu += dv.nu3;
                
                CT exp_int = T(1.0) / sum_nu;
                
                CT CmCn = coeffs[m] * coeffs[n];
                CT factor = CmCn * exp_int;
                 
                // Vectors from E_dn
                CT uz_m = E_dn(0,m); CT uz_n = E_dn(0,n);
                CT szz_m = E_dn(1,m); CT szz_n = E_dn(1,n);
                CT ur_m = E_dn(2,m); CT ur_n = E_dn(2,n);
                CT szr_m = E_dn(3,m); CT szr_n = E_dn(3,n);
                
                val_I1 += p.rho * (uz_m*uz_n + ur_m*ur_n) * factor;
                val_I2 += ( (szz_m*szz_n)/p.c33 + (szr_m*szr_n)/p.c44 + k_sq*term_A*ur_m*ur_n ) * factor;
                CT term_cross = (uz_m*szr_n - (p.c13/p.c33)*szz_m*ur_n) / k;
                val_I3 += ( term_A * ur_m*ur_n + term_cross ) * factor;
            }
        }
        res.I1 += val_I1.real();
        res.I2 += val_I2.real();
        res.I3 += val_I3.real();
    }

    return res;
}


// ===========================================================================
// Romberg-Grid Energy Integration
// ===========================================================================
// Integrates energy over a continuous Gradient Layer by subdividing into
// N homogeneous "pancakes", calculating the exact integral for each,
// and applying Romberg Extrapolation to the limit N -> infinity.

template <typename T>
EnergyIntegralResult<T> integrateGridLayer_Romberg(
    const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>& y_top, // State at top (4 for Rayleigh, 2 for Love)
    const PhysicalParameters<T>& p_top,
    const PhysicalParameters<T>& p_bot,
    T thickness,
    T omega,
    T phaseVelocity,
    WaveType type,
    int max_depth = 5,    // Max Romberg depth (2^5 = 32 subdivisions extrapolated)
    T tolerance = 1e-8,   // Convergence tolerance
    bool use_romberg = true // If false, returns raw staircase approximation for N=2^max_depth
) {
    using CT = std::complex<T>;

    // Helper: Compute Energy Density (I1, I2, I3) at a point
    auto compute_energies_rayleigh = [&](const Eigen::Matrix<CT, 4, 1>& y, const PhysicalParameters<T>& p, T k) -> std::vector<T> {
        T uz = y(0).real(); 
        T szz = y(1).real(); 
        T ur = y(2).real(); 
        T szr = y(3).real();
        T k_sq = k * k;
        
        // I1 = rho * (uz^2 + ur^2)
        T i1 = p.rho * (uz*uz + ur*ur);
        // I2 = szz^2/C + szr^2/L + k^2(A - F^2/C)ur^2
        T term_A_F2_C = p.c11 - p.c13*p.c13/p.c33;
        T i2 = (szz*szz/p.c33) + (szr*szr/p.c44) + k_sq * term_A_F2_C * ur*ur;
        // I3 = (A - F^2/C)ur^2 + (uz szr - F/C szz ur)/k
        T i3 = term_A_F2_C * ur*ur + (uz*szr - (p.c13/p.c33)*szz*ur)/k;
        
        return {i1, i2, i3};
    };

    auto compute_energies_love = [&](const Eigen::Matrix<CT, 2, 1>& y, const PhysicalParameters<T>& p, T k) -> std::vector<T> {
        T uy = y(0).real();
        T tyz = y(1).real();
        T k_sq = k*k;
        
        // I1 = rho * uy^2
        T i1 = p.rho * uy*uy;
        // I2 = tyz^2 / L + k^2 N uy^2
        T i2 = (tyz*tyz/p.c44) + k_sq * p.c66 * uy*uy;
        // I3 = N uy^2
        T i3 = p.c66 * uy*uy;
        return {i1, i2, i3};
    };

    // Helper to compute I(N)
    auto compute_approximation = [&](int N) -> EnergyIntegralResult<T> {
        EnergyIntegralResult<T> total_res;
        total_res.I1=0; total_res.I2=0; total_res.I3=0; total_res.phaseVelocity = phaseVelocity;
        
        T dz = thickness / T(N);
        Eigen::Matrix<CT, Eigen::Dynamic, 1> y_curr = y_top;
        
        PhysicalParameters<T> p_curr = p_top;
        std::vector<T> E_curr;
        T k_wave = omega / phaseVelocity;
        
        if (type == WaveType::Rayleigh) E_curr = compute_energies_rayleigh(y_curr, p_curr, k_wave);
        else E_curr = compute_energies_love(y_curr, p_curr, k_wave);
        
        // Loop over sub-pancakes
        for(int i=0; i<N; ++i) {
            // Midpoint Parameters for Propagation
            T alpha_mid = (T(i) + 0.5) / T(N);
            PhysicalParameters<T> p_mid;
            p_mid.rho = p_top.rho * (1-alpha_mid) + p_bot.rho * alpha_mid;
            p_mid.c11 = p_top.c11 * (1-alpha_mid) + p_bot.c11 * alpha_mid;
            p_mid.c33 = p_top.c33 * (1-alpha_mid) + p_bot.c33 * alpha_mid;
            p_mid.c44 = p_top.c44 * (1-alpha_mid) + p_bot.c44 * alpha_mid;
            p_mid.c66 = p_top.c66 * (1-alpha_mid) + p_bot.c66 * alpha_mid;
            p_mid.c13 = p_top.c13 * (1-alpha_mid) + p_bot.c13 * alpha_mid;
            
            // Next Parameters for Density
            T alpha_next = T(i+1) / T(N);
            PhysicalParameters<T> p_next;
            p_next.rho = p_top.rho * (1-alpha_next) + p_bot.rho * alpha_next;
            p_next.c11 = p_top.c11 * (1-alpha_next) + p_bot.c11 * alpha_next;
            p_next.c33 = p_top.c33 * (1-alpha_next) + p_bot.c33 * alpha_next;
            p_next.c44 = p_top.c44 * (1-alpha_next) + p_bot.c44 * alpha_next;
            p_next.c66 = p_top.c66 * (1-alpha_next) + p_bot.c66 * alpha_next;
            p_next.c13 = p_top.c13 * (1-alpha_next) + p_bot.c13 * alpha_next;
            
            Eigen::Matrix<CT, Eigen::Dynamic, 1> y_next;
            WaveVariables<T> wave; wave.omega = omega; wave.k = omega/phaseVelocity; wave.z = dz;

            if (type == WaveType::Rayleigh) {
                auto B = computeLayerMatrixP_SV_Explicit(p_mid, wave); 
                y_next = B * y_curr;
            } else {
                wave.z = -dz; // Downward depth increment, upward SH coordinate
                auto B = computeLayerMatrixSH(p_mid, wave);
                y_next = B * y_curr;
            }
            
            std::vector<T> E_next;
            if (type == WaveType::Rayleigh) E_next = compute_energies_rayleigh(y_next, p_next, k_wave);
            else E_next = compute_energies_love(y_next, p_next, k_wave);
            
            // Trapezoidal Rule
            total_res.I1 += (E_curr[0] + E_next[0]) * dz * 0.5;
            total_res.I2 += (E_curr[1] + E_next[1]) * dz * 0.5;
            total_res.I3 += (E_curr[2] + E_next[2]) * dz * 0.5;
            
            y_curr = y_next;
            p_curr = p_next;
            E_curr = E_next;
        }
        return total_res;
    };
    
    // Non-Romberg (Raw Staircase) Mode
    if (!use_romberg) {
        int N = 1 << max_depth;
        return compute_approximation(N);
    }
    
    // Romberg Loop
    // R[k][m] stores integral Estimate for 2^k steps, extrapolated m times.
    std::vector<EnergyIntegralResult<T>> T_row(max_depth + 1);
    
    // k=0 (N=1)
    T_row[0] = compute_approximation(1);
    
    for (int k = 1; k <= max_depth; ++k) {
        int N = 1 << k; // 2^k
        EnergyIntegralResult<T> val_k_0 = compute_approximation(N);
        
        // Richardson Extrapolation
        // T[m] = (4^m * T[m-1](new) - T[m-1](old)) / (4^m - 1)
        // We compute row k from left to right.
        std::vector<EnergyIntegralResult<T>> next_row(max_depth + 1);
        next_row[0] = val_k_0;
        
        for (int m = 1; m <= k; ++m) {
            double factor = std::pow(4.0, m);
            double denom = factor - 1.0;
            
            auto extrapolate = [&](double v_curr, double v_prev) {
                return (factor * v_curr - v_prev) / denom;
            };
            
            next_row[m].I1 = extrapolate(next_row[m-1].I1, T_row[m-1].I1);
            next_row[m].I2 = extrapolate(next_row[m-1].I2, T_row[m-1].I2);
            next_row[m].I3 = extrapolate(next_row[m-1].I3, T_row[m-1].I3);
            next_row[m].phaseVelocity = phaseVelocity;
        }
        
        // Check convergence (on I1, relative)
        double diff = std::abs(next_row[k].I1 - T_row[k-1].I1);
        double val = std::abs(next_row[k].I1);
        if (val > 1e-20 && diff/val < tolerance) {
            return next_row[k];
        }
        
        T_row = next_row;
    }
    
    return T_row[max_depth]; // Return best estimate
}

// Propagate State Vector through a gradient layer using Romberg Integration
// Can be used to find state at bottom of layer, or at any intermediate depth if 'thickness' is set to intermediate z.
template <typename T>
Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> propagate_romberg_vector(
    const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>& y_top,
    const PhysicalParameters<T>& p_top_layer, // Params at top of full layer
    const PhysicalParameters<T>& p_bot_layer, // Params at bottom of full layer
    T full_layer_thickness,                   // Total thickness of the gradient layer
    T target_depth,                           // Depth to propagate to (0 <= target_depth <= full_layer_thickness)
    T omega,
    T phaseVelocity,
    WaveType type,
    int max_depth = 5,
    T tolerance = 1e-8
) {
    using CT = std::complex<T>;
    
    // Interpolate p at target_depth to define the "sub-layer" bottom
    T alpha_target = target_depth / full_layer_thickness;
    if (full_layer_thickness < 1e-10) alpha_target = 0; // Safety
    
    PhysicalParameters<T> p_bot_target;
    p_bot_target.rho = p_top_layer.rho * (1-alpha_target) + p_bot_layer.rho * alpha_target;
    p_bot_target.c11 = p_top_layer.c11 * (1-alpha_target) + p_bot_layer.c11 * alpha_target;
    p_bot_target.c33 = p_top_layer.c33 * (1-alpha_target) + p_bot_layer.c33 * alpha_target;
    p_bot_target.c44 = p_top_layer.c44 * (1-alpha_target) + p_bot_layer.c44 * alpha_target;
    p_bot_target.c66 = p_top_layer.c66 * (1-alpha_target) + p_bot_layer.c66 * alpha_target;
    p_bot_target.c13 = p_top_layer.c13 * (1-alpha_target) + p_bot_layer.c13 * alpha_target;

    // Helper to compute y(N) (Staircase Propagator)
    auto compute_approximation = [&](int N) -> Eigen::Matrix<CT, Eigen::Dynamic, 1> {
        T dz = target_depth / T(N);
        Eigen::Matrix<CT, Eigen::Dynamic, 1> y_curr = y_top;
        
        for(int i=0; i<N; ++i) {
            T alpha = (T(i) + 0.5) / T(N); // Local alpha within [0, target_depth]
            // We need parameters at this midpoint relative to the FULL layer gradient?
            // No, the gradient is linear. We can interpolate between p_top_layer and p_bot_target.
            // Or just interpolate directly from p_top_layer and p_bot_layer using global alpha.
            
            T global_z_mid = (T(i) + 0.5) * dz; // Depth from top
            T alpha_global = global_z_mid / full_layer_thickness;
            
            PhysicalParameters<T> p_mid;
            p_mid.rho = p_top_layer.rho * (1-alpha_global) + p_bot_layer.rho * alpha_global;
            p_mid.c11 = p_top_layer.c11 * (1-alpha_global) + p_bot_layer.c11 * alpha_global;
            p_mid.c33 = p_top_layer.c33 * (1-alpha_global) + p_bot_layer.c33 * alpha_global;
            p_mid.c44 = p_top_layer.c44 * (1-alpha_global) + p_bot_layer.c44 * alpha_global;
            p_mid.c66 = p_top_layer.c66 * (1-alpha_global) + p_bot_layer.c66 * alpha_global;
            p_mid.c13 = p_top_layer.c13 * (1-alpha_global) + p_bot_layer.c13 * alpha_global;
            
            WaveVariables<T> wave; wave.omega = omega; wave.k = omega/phaseVelocity; wave.z = dz;
            
            if (type == WaveType::Rayleigh) {
                auto B = computeLayerMatrixP_SV_Explicit(p_mid, wave); 
                y_curr = B * y_curr;
            } else {
                wave.z = -dz; // Downward depth increment, upward SH coordinate
                auto B = computeLayerMatrixSH(p_mid, wave);
                y_curr = B * y_curr;
            }
        }
        return y_curr;
    };
    
    // Romberg Loop
    std::vector<Eigen::Matrix<CT, Eigen::Dynamic, 1>> T_row(max_depth + 1);
    T_row[0] = compute_approximation(1);
    
    for (int k = 1; k <= max_depth; ++k) {
        int N = 1 << k;
        // T_row[k] = compute_approximation(N); // This line was from a previous thought process.
        // It should be next_row[0] = compute_approximation(N);
        
        // Richardson Extrapolation
        // We update T_row in place from right to left to avoid extra storage?
        // Actually, let's use a temporary 'current row' like before.
        std::vector<Eigen::Matrix<CT, Eigen::Dynamic, 1>> next_row(max_depth + 1);
        // next_row[0] = T_row[k]; // Base value for this row is what we just computed
        
        // But wait, my previous logic was: T_row stores previous row.
        // next_row[0] is approximation(N).
        // next_row[m] combines next_row[m-1] and T_row[m-1].
        // Correct.
        
        next_row[0] = compute_approximation(N);
        
        for (int m = 1; m <= k; ++m) {
            double factor = std::pow(4.0, m);
            double denom = factor - 1.0;
            
            // Vector extrapolation
            next_row[m] = (factor * next_row[m-1] - T_row[m-1]) / denom;
        }
        
        // Convergence Check (L2 norm of difference)
        T diff_norm = (next_row[k] - T_row[k-1]).norm();
        T val_norm = next_row[k].norm();
        
        if (val_norm > 1e-20 && diff_norm/val_norm < tolerance) {
            return next_row[k];
        }
        
        T_row = next_row;
    }
    
    return T_row[max_depth];
}

// Unified State Evaluator
// Calculates state y(z) at target_depth (relative to top) given layer properties.
// mode=true -> Romberg (Grid), mode=false -> Pancake (Homogeneous approximation)
template <typename T>
Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> computeStateAtDepth(
    const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>& y_top,
    const PhysicalParameters<T>& p_top_layer,
    const PhysicalParameters<T>& p_bot_layer,
    T full_layer_thickness,
    T target_depth,
    T omega,
    T phaseVelocity,
    WaveType type,
    bool use_romberg = true,
    int max_depth = 5,
    T tolerance = 1e-8
) {
    if (use_romberg) {
        return propagate_romberg_vector(
            y_top, p_top_layer, p_bot_layer, full_layer_thickness, target_depth,
            omega, phaseVelocity, type, max_depth, tolerance
        );
    } else {
        // Pancake Mode: Treat segment [0, target_depth] as single homogeneous layer
        // Use parameters at midpoint of this segment
        T dz = target_depth;
        T z_mid_global = dz * 0.5;
        T alpha_mid = z_mid_global / full_layer_thickness;
        if (full_layer_thickness < 1e-10) alpha_mid = 0;
        
        PhysicalParameters<T> p_mid;
        p_mid.rho = p_top_layer.rho * (1-alpha_mid) + p_bot_layer.rho * alpha_mid;
        p_mid.c11 = p_top_layer.c11 * (1-alpha_mid) + p_bot_layer.c11 * alpha_mid;
        p_mid.c33 = p_top_layer.c33 * (1-alpha_mid) + p_bot_layer.c33 * alpha_mid;
        p_mid.c44 = p_top_layer.c44 * (1-alpha_mid) + p_bot_layer.c44 * alpha_mid;
        p_mid.c66 = p_top_layer.c66 * (1-alpha_mid) + p_bot_layer.c66 * alpha_mid;
        p_mid.c13 = p_top_layer.c13 * (1-alpha_mid) + p_bot_layer.c13 * alpha_mid;
        
        WaveVariables<T> wave; wave.omega = omega; wave.k = omega/phaseVelocity; wave.z = dz;
        
        if (type == WaveType::Rayleigh) {
            auto B = computeLayerMatrixP_SV_Explicit(p_mid, wave); 
            return B * y_top;
        } else {
            wave.z = -dz; // Downward depth increment, upward SH coordinate
            auto B = computeLayerMatrixSH(p_mid, wave);
            return B * y_top;
        }
    }
}

} // namespace Analytical

#endif // DISPER26_ANALYTICAL_HPP
