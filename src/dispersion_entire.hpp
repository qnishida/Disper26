#pragma once

// Included after the scalar extraction and complex sqrt helpers. Calls stay
// unqualified so custom complex AD types can supply their overloads by ADL.
namespace disper26_math {

template<class Complex>
struct EntireCS {
    Complex c;       // cosh(sqrt(q))
    Complex s;       // sinh(sqrt(q))/sqrt(q)
    Complex cm1;     // cosh(sqrt(q))-1, without subtraction near zero
};

template<class Complex>
struct DefaultEntireBackend {
    static inline EntireCS<Complex> compute(const Complex& q) {
        // Preserve dynamic AD derivative sizes and initialize BOTH components.
        const Complex zero = q - q;
        auto constant = [&](double x) { return Complex(zero.real() + x, zero.imag()); };
        const Complex one = constant(1.0);
        const double magnitude = std::abs(get_val(q.real())) + std::abs(get_val(q.imag()));
        if (magnitude <= 0.25) {
            // Conservative accuracy baseline, not a speed-tuned approximation.
            // Degree 12 on |q|<=1/4 bounds the omitted value and first two
            // q-derivatives below 1e-29. Differentiate q directly, including
            // q=0: S'(0)=1/6, S''(0)=1/60, C'(0)=1/2, C''(0)=1/12.
            double fc = 1.0, fs = 1.0;
            double cc[13], ss[13];
            cc[0] = ss[0] = 1.0;
            for (int n = 1; n <= 12; ++n) {
                fc *= double((2*n-1)*(2*n));
                fs *= double((2*n)*(2*n+1));
                cc[n] = 1.0/fc;
                ss[n] = 1.0/fs;
            }
            if constexpr (std::is_same_v<Complex,std::complex<double>>) {
                if (q.imag() == 0.0) {
                    // The same polynomial on real q needs no complex
                    // arithmetic. Do not apply this shortcut to AD: a zero
                    // imaginary value can still carry nonzero derivatives.
                    const double x = q.real();
                    double c_tail = cc[12], s = ss[12];
                    for (int n = 11; n >= 1; --n) {
                        c_tail = c_tail*x + cc[n];
                        s = s*x + ss[n];
                    }
                    const double cm1 = x*c_tail;
                    return {{1.0+cm1,0.0},{s*x+1.0,0.0},{cm1,0.0}};
                }
            }
            Complex c_tail = constant(cc[12]), s = constant(ss[12]);
            for (int n = 11; n >= 1; --n) {
                c_tail = c_tail*q + constant(cc[n]);
                s = s*q + constant(ss[n]);
            }
            const Complex cm1 = q*c_tail;
            s = s*q + one;
            return {one + cm1, s, cm1};
        }
        const Complex z = safe_complex_sqrt(q);
        Complex c, sinh_z;
        compute_cosh_sinh(z, c, sinh_z);
        return {c, sinh_z/z, c-one};
    }
};

// Compile-time extension point for subsequent approximation experiments.
// Unchanged scalar/AD types keep the conservative baseline above.
template<class Complex>
struct EntireBackend : DefaultEntireBackend<Complex> {};

template<class Complex>
inline EntireCS<Complex> compute_entire_cs(const Complex& q) {
    return EntireBackend<Complex>::compute(q);
}

// Match the established complex quotient on real-axis operands. With
// -fcx-limited-range, complex division evaluates (a*b)/(b*b), not a/b.
// The rounding matters to integer counts at a root; keep it without imaginary
// arithmetic. The caller falls back if intermediate overflow is nonfinite.
inline double divide_real_like_complex(double a, double b) {
    return (a*b)/(b*b);
}

// Explicitly selected by the experimental double Rayleigh layer evaluator.
// The complex/AD backend is intentionally not specialized or redirected.
inline EntireCS<double> compute_entire_cs_real(double q) {
    if (std::abs(q) <= 0.25) {
        double fc = 1.0, fs = 1.0, cc[13], ss[13];
        cc[0] = ss[0] = 1.0;
        for (int n = 1; n <= 12; ++n) {
            fc *= double((2*n-1)*(2*n));
            fs *= double((2*n)*(2*n+1));
            cc[n] = 1.0/fc;
            ss[n] = 1.0/fs;
        }
        double c_tail = cc[12], s = ss[12];
        for (int n = 11; n >= 1; --n) {
            c_tail = c_tail*q + cc[n];
            s = s*q + ss[n];
        }
        const double cm1 = q*c_tail;
        return {1.0+cm1, s*q+1.0, cm1};
    }
    if (q > 0.0) {
        const double z = std::sqrt(q), c = std::cosh(z);
        return {c, divide_real_like_complex(std::sinh(z),z), c-1.0};
    }
    const double z = std::sqrt(-q), c = std::cos(z);
    return {c, divide_real_like_complex(std::sin(z),z), c-1.0};
}

} // namespace disper26_math
