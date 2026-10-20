#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../src/disper26.cpp"
#include "../src/romberg_engine.hpp"

namespace {

RombergResult run_trace(
    const std::vector<double>& roots,
    double tolerance,
    int minimum_trace_level,
    int& group_calls,
    int& hv_calls,
    int max_steps = -1)
{
    auto mesh = [](int level) { return level; };
    auto root = [&roots](int level, double, double,
                         const std::vector<double>& history) {
        assert(static_cast<int>(history.size()) == level);
        return roots.at(static_cast<size_t>(level));
    };
    auto group = [&group_calls](int, double, double c) {
        ++group_calls;
        return 2.0 * c;
    };
    auto hv = [&hv_calls](int, double, double c) {
        ++hv_calls;
        return 0.5 * c;
    };
    const int phase_max_steps = max_steps >= 0
        ? max_steps : static_cast<int>(roots.size()) - 1;
    return compute_with_romberg_core(
        mesh, 20.0, roots.front(), root, group, hv,
        phase_max_steps, tolerance, false, 0,
        minimum_trace_level);
}

}  // namespace

int main()
{
    // Residual-aware callers reuse Brent's already evaluated f(root).  The
    // optional output must not add a fourth function evaluation here.
    int brent_calls = 0;
    auto steep_linear = [&brent_calls](double value) {
        ++brent_calls;
        return 1.0e12 * (value - 1.25);
    };
    double final_function_value = std::numeric_limits<double>::quiet_NaN();
    const double brent_root = brents_method(
        steep_linear, 1.0, 2.0, 1.0e-10, 100, &final_function_value);
    assert(std::abs(brent_root - 1.25) < 1.0e-14);
    assert(final_function_value == 0.0);
    assert(brent_calls == 3);

    const std::vector<double> smooth = {
        3.25,
        3.0625,
        3.015625,
        3.00390625,
        3.0009765625,
        3.000244140625,
        3.00006103515625,
    };

    int group_calls = 0;
    int hv_calls = 0;
    const RombergResult legacy = run_trace(smooth, 1.0e-12, 0,
                                           group_calls, hv_calls);
    assert(legacy.converged_level == 4);
    assert(legacy.root_trace.phase_accepted_level == 4);
    assert(legacy.root_trace.generated_level == 4);
    assert(legacy.root_trace.phase_velocity_roots.size() == 5);
    assert(group_calls == 5 && hv_calls == 5);

    group_calls = 0;
    hv_calls = 0;
    const RombergResult extended = run_trace(smooth, 1.0e-12, 6,
                                             group_calls, hv_calls);
    assert(extended.phase_velocity == legacy.phase_velocity);
    assert(extended.group_velocity == legacy.group_velocity);
    assert(extended.hv_ratio == legacy.hv_ratio);
    assert(extended.converged_level == legacy.converged_level);
    assert(extended.relative_error == legacy.relative_error);
    assert(extended.root_trace.phase_accepted_level == 4);
    assert(extended.root_trace.generated_level == 6);
    assert(extended.root_trace.phase_velocity_roots == smooth);
    // Extra trace-only levels must not recompute U or H/V.
    assert(group_calls == 5 && hv_calls == 5);

    std::vector<double> roundoff = smooth;
    roundoff[5] = 3.1;
    group_calls = 0;
    hv_calls = 0;
    const RombergResult reverted = run_trace(roundoff, -1.0, 0,
                                             group_calls, hv_calls);
    assert(reverted.converged_level == 4);
    assert(reverted.root_trace.generated_level == 4);
    assert(reverted.root_trace.phase_velocity_roots.size() == 5);
    assert(reverted.root_trace.phase_velocity_roots.back() == roundoff[4]);
    assert(group_calls == 6 && hv_calls == 6);

    // A phase-table rollback rejects only the newest extrapolation sample;
    // the raw root at that mesh level remains valid for derivative reuse.
    // In particular, equality at the minimum-trace boundary must retain k.
    group_calls = 0;
    hv_calls = 0;
    const RombergResult reverted_at_minimum = run_trace(
        roundoff, -1.0, 5, group_calls, hv_calls);
    assert(reverted_at_minimum.phase_velocity == reverted.phase_velocity);
    assert(reverted_at_minimum.group_velocity == reverted.group_velocity);
    assert(reverted_at_minimum.hv_ratio == reverted.hv_ratio);
    assert(reverted_at_minimum.converged_level == reverted.converged_level);
    assert(reverted_at_minimum.relative_error == reverted.relative_error);
    assert(reverted_at_minimum.root_trace.phase_accepted_level == 4);
    assert(reverted_at_minimum.root_trace.generated_level == 5);
    assert(reverted_at_minimum.root_trace.phase_velocity_roots.size() == 6);
    for (size_t k = 0; k < 6; ++k) {
        assert(reverted_at_minimum.root_trace.phase_velocity_roots[k] ==
               roundoff[k]);
    }
    assert(group_calls == 6 && hv_calls == 6);

    // When a still deeper trace was requested, continue root-only refinement
    // after freezing the reverted c/U/HV table.
    group_calls = 0;
    hv_calls = 0;
    const RombergResult reverted_then_extended = run_trace(
        roundoff, -1.0, 6, group_calls, hv_calls);
    assert(reverted_then_extended.phase_velocity == reverted.phase_velocity);
    assert(reverted_then_extended.group_velocity == reverted.group_velocity);
    assert(reverted_then_extended.hv_ratio == reverted.hv_ratio);
    assert(reverted_then_extended.converged_level == reverted.converged_level);
    assert(reverted_then_extended.relative_error == reverted.relative_error);
    assert(reverted_then_extended.root_trace.phase_accepted_level == 4);
    assert(reverted_then_extended.root_trace.generated_level == 6);
    assert(reverted_then_extended.root_trace.phase_velocity_roots == roundoff);
    assert(group_calls == 6 && hv_calls == 6);

    group_calls = 0;
    hv_calls = 0;
    const RombergResult beyond_phase_max = run_trace(
        smooth, -1.0, 6, group_calls, hv_calls, 4);
    assert(beyond_phase_max.converged_level == 4);
    assert(beyond_phase_max.root_trace.phase_accepted_level == 4);
    assert(beyond_phase_max.root_trace.generated_level == 6);
    assert(beyond_phase_max.root_trace.phase_velocity_roots == smooth);
    assert(group_calls == 5 && hv_calls == 5);

    std::vector<double> missing_trace_root = smooth;
    missing_trace_root[5] = std::numeric_limits<double>::quiet_NaN();
    bool trace_only_root_failed = false;
    try {
        group_calls = 0;
        hv_calls = 0;
        (void)run_trace(missing_trace_root, -1.0, 6,
                        group_calls, hv_calls, 4);
    } catch (const std::runtime_error& exc) {
        trace_only_root_failed =
            std::string(exc.what()).find("trace-only root solve failed") !=
            std::string::npos;
    }
    assert(trace_only_root_failed);

    bool unsafe_trace_depth_failed = false;
    try {
        group_calls = 0;
        hv_calls = 0;
        (void)run_trace(smooth, -1.0, 13, group_calls, hv_calls, 4);
    } catch (const std::invalid_argument& exc) {
        unsafe_trace_depth_failed =
            std::string(exc.what()).find("safe trace-only refinement limit") !=
            std::string::npos;
    }
    assert(unsafe_trace_depth_failed);

    std::cout << "Romberg level-root trace checks passed\n";
    return 0;
}
