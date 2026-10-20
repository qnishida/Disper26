#pragma once

// Finite-layer quantities with the removable P/SV wavenumber factors already
// cancelled. Halfspace decaying modes still use computeDerivedValues.
// E1=nu1*eps1, E3=eps3/nu3, D1=nu1*d1, D2=d2/nu3,
// D5=d5, D6=(nu1/nu3)*d6, H_j=sinh(nu_j*z)/nu_j.
// All of these remain finite at the ordinary P and SV critical speeds.
template<class T, class CT = typename ComplexType<T>::type>
struct RayleighEntireValues {
    CT n1, n3, E1, E3, D1, D2, d3, d4, D5, D6;
    CT C1, C3, H1, H3, Cm1, Cdiff;
};

template<class T, class CT>
bool computeRayleighEntireValues(const PhysicalParameters<T>& p,
                                const WaveVariables<T>& v,
                                RayleighEntireValues<T, CT>& out) {
    auto divide = [](const CT& a, const CT& b) -> CT {
        if constexpr (std::is_same_v<CT,double>) return disper26_math::divide_real_like_complex(a,b);
        else return a/b;
    };
    auto cast = [](const T& a) -> CT {
        if constexpr (std::is_same_v<CT, double>) return a;
        else return toComplex<T>(a);
    };
    const CT two = cast(T(make_zero_like(p.rho)+2.0));
    const CT four = cast(T(make_zero_like(p.rho)+4.0));
    const CT C = cast(p.c33), L = cast(p.c44), F = cast(p.c13);
    const CT k = cast(v.k), z = cast(v.z);
    const CT rw2 = cast(p.rho*v.omega*v.omega);
    const CT x = rw2-L*k*k;
    const CT y = rw2-cast(p.c11)*k*k;
    const CT coupling = (F+L)*k;
    const CT m1 = L*x+C*y+coupling*coupling;
    const CT m2 = x*y;
    const CT disc = m1*m1-four*C*L*m2;
    auto magnitude = [](const CT& a) {
        if constexpr (std::is_same_v<CT, double>) return std::abs(a);
        else return std::abs(get_val(a.real()))+std::abs(get_val(a.imag()));
    };
    // Coalescing modal roots and an uncoupled system need no modal basis.
    // This guard catches a discriminant at its binary64 cancellation floor;
    // the existing matrix-exponential implementation supplies the limit.
    const double disc_scale = magnitude(CT(m1*m1))+magnitude(CT(four*C*L*m2));
    if (magnitude(coupling) == 0.0 ||
        magnitude(disc) <= 64.0*std::numeric_limits<double>::epsilon()*disc_scale)
        return false;
    CT root;
    if constexpr (std::is_same_v<CT, double>) {
        // No tolerance-based projection of complex conjugate roots onto R.
        if (!(disc > 0.0) || !std::isfinite(disc)) return false;
        root = std::sqrt(disc);
    } else {
        root = safe_complex_sqrt(disc);
    }
    const CT plus = -m1+root, minus = -m1-root;
    const CT denominator = two*C*L;
    // Recover the smaller quadratic root from their product. Unlike the
    // previous near-Vp-only guard, this covers a small SV root as well.
    // Initially keep the original +/- labels.
    if (magnitude(plus) < magnitude(minus)) {
        out.n3 = divide(minus,denominator);
        out.n1 = divide(divide(m2,CT(C*L)),out.n3);
    } else {
        out.n1 = divide(plus,denominator);
        out.n3 = divide(divide(m2,CT(C*L)),out.n1);
    }
    // Strong VTI can reverse the usual P/SV ordering. Assign a root tending
    // to zero at x=0 to the SV scaling, and one at y=0 to the P scaling.
    // Swapping the two modes leaves the finite-layer matrix invariant. This
    // avoids reintroducing a removable zero in D6 through a wrong scaling.
    const bool first_is_small = magnitude(out.n1) < magnitude(out.n3);
    const bool small_is_sv = magnitude(x) < magnitude(y);
    if (first_is_small == small_is_sv) std::swap(out.n1,out.n3);
    const CT eps3_den = -y-L*out.n3;
    if (magnitude(eps3_den) == 0.0) return false;
    // The two equivalent polarization equations provide complementary
    // scalings: no division by either nu1 or nu3 is performed here.
    out.E1 = divide(CT(C*out.n1+x),coupling);
    out.E3 = divide(coupling,eps3_den);
    out.D1 = C*out.n1-F*k*out.E1;
    out.D2 = C-F*k*out.E3;
    out.d3 = k+out.E1;
    out.d4 = k+out.n3*out.E3;
    out.D5 = out.E1-out.n3*out.E3;
    out.D6 = out.E1-out.n1*out.E3;
    if (magnitude(out.D5) == 0.0 || magnitude(out.D6) == 0.0) return false;
    auto entire_cs = [](const CT& q) {
        if constexpr (std::is_same_v<CT, double>)
            return disper26_math::compute_entire_cs_real(q);
        else return disper26_math::compute_entire_cs(q);
    };
    const auto cs1 = entire_cs(CT(out.n1*z*z));
    const auto cs3 = entire_cs(CT(out.n3*z*z));
    out.C1 = cs1.c; out.C3 = cs3.c;
    out.H1 = z*cs1.s; out.H3 = z*cs3.s;
    // C1*C3-1 without subtracting 1 from a rounded product near z=0.
    out.Cm1 = cs1.cm1+cs3.cm1+cs1.cm1*cs3.cm1;
    out.Cdiff = cs1.cm1-cs3.cm1;
    return true;
}

// One copy of the reduced formulas for both arithmetic carriers.
template<class T, class CT>
Eigen::Matrix<T,5,5> assembleRayleighReducedPropagator(
    const PhysicalParameters<T>& p, const RayleighEntireValues<T,CT>& v) {
    auto divide = [](const CT& a, const CT& b) -> CT {
        if constexpr (std::is_same_v<CT,double>) return disper26_math::divide_real_like_complex(a,b);
        else return a/b;
    };
    auto cast = [](const T& a) -> CT {
        if constexpr (std::is_same_v<CT,double>) return a;
        else return toComplex<T>(a);
    };
    auto real_part = [](const CT& a) -> T {
        if constexpr (std::is_same_v<CT,double>) return a;
        else return a.real();
    };
    const CT one = cast(T(make_zero_like(p.rho)+1.0));
    const CT two = cast(T(make_zero_like(p.rho)+2.0));
    const CT C = cast(p.c33), L = cast(p.c44);
    const CT inv5 = divide(one,v.D5), inv6 = divide(one,v.D6);
    const CT k33_6 = divide(inv6,C), k44_5 = divide(inv5,L);
    const CT k33_56 = k33_6*inv5, kall = divide(k33_56,L);
    const CT hp = v.H1*v.H3, nn = v.n1*v.n3;
    const CT cp = v.C1*v.C3, cm = v.Cm1;
    const CT c1h3 = v.C1*v.H3, c3h1 = v.C3*v.H1;
    const CT& a = v.D1; const CT& b = v.D2;
    const CT& e = v.E1; const CT& f = v.E3;
    const CT& d3 = v.d3; const CT& d4 = v.d4;
    // Algebraically the same reduced delta matrix as before, with every
    // nu1/nu3 factor cancelled BEFORE evaluation or differentiation. Forming
    // a compound of the full state matrix here would lose the established
    // high-frequency cancellations, so retain the explicit reduced formulas.
    const CT b1234 = kall*(two*a*b*cm-(a*a+nn*b*b)*hp);
    const CT b1224 = kall*(-(e*b+f*a)*cm+(e*a+nn*f*b)*hp);
    const CT b1223 = k33_6*(a*c1h3-v.n1*b*c3h1);
    const CT b1214 = k44_5*(-v.n3*b*c1h3+a*c3h1);
    const CT b1213 = k33_56*(a*b*(d3+d4)*cm-(nn*b*b*d3+a*a*d4)*hp);
    Eigen::Matrix<T,5,5> B;
    B(I12,I12) = real_part((one-two*b1234));
    B(I13,I12) = real_part((two*b1224));
    B(I14,I12) = real_part((two*b1223));
    B(I23,I12) = real_part((two*b1214));
    B(I24,I12) = real_part((two*b1213));
    B(I12,I13) = real_part(b1213);
    B(I13,I13) = real_part((cp+b1234));
    B(I14,I13) = real_part((L*k33_6*(-a*d4*c1h3+v.n1*b*d3*c3h1)));
    B(I23,I13) = real_part((inv5*(v.n3*b*d3*c1h3-a*d4*c3h1)));
    B(I24,I13) = real_part((L*k33_56*(-two*a*b*d3*d4*cm+(a*a*d4*d4+nn*b*b*d3*d3)*hp)));
    B(I12,I14) = real_part(b1214);
    B(I13,I14) = real_part((k44_5*(-v.n3*f*c1h3+e*c3h1)));
    B(I14,I14) = real_part(cp);
    B(I23,I14) = real_part((C*v.n3*v.D6*k44_5*hp));
    B(I24,I14) = B(I23,I13);
    B(I12,I23) = real_part(b1223);
    B(I13,I23) = real_part((k33_6*(e*c1h3-v.n1*f*c3h1)));
    B(I14,I23) = real_part((L*v.n1*v.D5*k33_6*hp));
    B(I23,I23) = B(I14,I14);
    B(I24,I23) = B(I14,I13);
    B(I12,I24) = real_part(b1224);
    B(I13,I24) = real_part((kall*(-two*e*f*cm+(e*e+nn*f*f)*hp)));
    B(I14,I24) = B(I13,I23);
    B(I23,I24) = B(I13,I14);
    B(I24,I24) = B(I13,I13);
    return B;
}

template<class T>
Eigen::Matrix<T,5,5> computeRayleighReducedPropagatorEntire(
    const PhysicalParameters<T>& p, const WaveVariables<T>& wave) {
    RayleighEntireValues<T> v;
    if (!computeRayleighEntireValues(p,wave,v))
        return computeRayleighReducedPropagatorRegular(p,wave);
    return assembleRayleighReducedPropagator(p,v);
}

// Returns false without modifying the caller's matrix when the real formula
// is inapplicable. Probes can inspect this result without production counters.
inline bool tryRayleighReducedPropagatorReal(
    const PhysicalParameters<double>& p, const WaveVariables<double>& wave,
    Eigen::Matrix<double,5,5>& matrix) {
    RayleighEntireValues<double,double> v;
    if (!computeRayleighEntireValues(p,wave,v)) return false;
    const auto result = assembleRayleighReducedPropagator(p,v);
    if (!result.allFinite()) return false;
    matrix = result;
    return true;
}

inline Eigen::Matrix<double,5,5> computeRayleighReducedPropagatorRealOrComplex(
    const PhysicalParameters<double>& p, const WaveVariables<double>& wave) {
    Eigen::Matrix<double,5,5> result;
    if (tryRayleighReducedPropagatorReal(p,wave,result)) return result;
    return computeRayleighReducedPropagatorEntire(p,wave);
}

template<class T>
Eigen::Matrix<typename ComplexType<T>::type,4,4> computeRayleighStatePropagatorEntire(
    const PhysicalParameters<T>& p, const WaveVariables<T>& wave) {
    using CT = typename ComplexType<T>::type;
    RayleighEntireValues<T> v;
    if (!computeRayleighEntireValues(p,wave,v))
        return computeRayleighStatePropagatorRegular(p,wave);
    const CT zero = toComplex(make_zero_like(p.rho));
    const CT one(zero.real()+1.0, zero.imag());
    const CT C = toComplex(p.c33), L = toComplex(p.c44);
    const CT inv5 = one/v.D5, inv6 = one/v.D6;
    const CT k33_6 = inv6/C, k44_5 = inv5/L;
    const CT& a = v.D1; const CT& b = v.D2;
    const CT& e = v.E1; const CT& f = v.E3;
    const CT& d3 = v.d3; const CT& d4 = v.d4;
    // cm1 differences retain the small-thickness off-diagonal entries.
    const CT& difference = v.Cdiff;
    Eigen::Matrix<CT,4,4> A;
    A(0,0) = inv5*(-d4*v.C1+d3*v.C3);
    A(0,1) = k33_6*(-v.n1*f*v.H1+e*v.H3);
    A(0,2) = k33_6*(v.n1*b*v.H1-a*v.H3);
    A(0,3) = k44_5*difference;
    A(1,0) = inv5*(-a*d4*v.H1+v.n3*b*d3*v.H3);
    A(1,1) = A(0,0);
    A(1,2) = a*b*k33_6*difference;
    A(1,3) = k44_5*(a*v.H1-v.n3*b*v.H3);
    A(2,0) = inv5*(-e*d4*v.H1+v.n3*f*d3*v.H3);
    A(2,1) = -e*f*k33_6*difference;
    A(2,2) = k33_6*(e*b*v.C1-f*a*v.C3);
    A(2,3) = k44_5*(e*v.H1-v.n3*f*v.H3);
    A(3,0) = -L*d3*d4*inv5*difference;
    A(3,1) = L*k33_6*(-v.n1*f*d3*v.H1+e*d4*v.H3);
    A(3,2) = L*k33_6*(v.n1*b*d3*v.H1-a*d4*v.H3);
    A(3,3) = inv5*(d3*v.C1-d4*v.C3);
    return A;
}
