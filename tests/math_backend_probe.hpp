#pragma once

// Test-only specialization, selected before the layer templates are defined.
namespace disper26_math {
inline unsigned probe_calls = 0;

template<>
struct HyperbolicBackend<std::complex<double>> {
    static inline void compute_cosh_sinh(const std::complex<double>& z,
                                        std::complex<double>& c,
                                        std::complex<double>& s) {
        ++probe_calls;
        DefaultHyperbolicBackend<std::complex<double>>::compute_cosh_sinh(z, c, s);
    }
};
}  // namespace disper26_math
