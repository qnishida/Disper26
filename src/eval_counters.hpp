#ifndef EVAL_COUNTERS_HPP
#define EVAL_COUNTERS_HPP

#include <cstdint>

// Lightweight instrumentation counting dispersion-function evaluations
// (propagator sweeps through calculateRayleighDispersion /
// calculateLoveDispersion) for any scalar type T.
// thread_local: no locks, and stays correct if the GIL is released and
// Python threads run compute paths concurrently.
namespace disper26_stats {

inline thread_local std::uint64_t dispersion_evals_total  = 0;
inline thread_local std::uint64_t dispersion_evals_double = 0;

inline void reset_dispersion_evals() {
    dispersion_evals_total = 0;
    dispersion_evals_double = 0;
}

} // namespace disper26_stats

#endif // EVAL_COUNTERS_HPP
