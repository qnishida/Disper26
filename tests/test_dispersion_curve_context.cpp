#include "api.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace disper26::api;

namespace {
void check(bool ok) {
    if (!ok) throw std::runtime_error("Curve context changed the scalar contract");
}
void same(const ModeSolution& a, const ModeSolution& b) {
    const auto& x = a.diagnostics();
    const auto& y = b.diagnostics();
    // Different curve predictors can change final root rounding; retain the
    // public curve/scalar tolerance (rtol=1e-9, atol=1e-11).
    auto close=[](double x,double y){return std::abs(x-y)<=1e-11+1e-9*std::abs(y);};
    check(close(a.phase_velocity(),b.phase_velocity()));
    check(a.group_velocity().has_value()==b.group_velocity().has_value());
    check(a.hv_ratio().has_value()==b.hv_ratio().has_value());
    if(a.group_velocity()) check(close(*a.group_velocity(),*b.group_velocity()));
    if(a.hv_ratio()) check(close(*a.hv_ratio(),*b.hv_ratio()));
    check(a.period() == b.period() && a.wave() == b.wave() && a.mode() == b.mode());
    check(x.accepted_level == y.accepted_level && x.generated_level == y.generated_level);
    check(std::abs(x.relative_error-y.relative_error)<=1e-9 ||
          (std::isnan(x.relative_error) && std::isnan(y.relative_error)));
    check(x.availability == y.availability && x.convergence_by_quantity == y.convergence_by_quantity);
    check(x.convergence == y.convergence && x.status == y.status && x.messages == y.messages);
}
}

int main(int argc, char** argv) {
    const bool benchmark = argc > 1 && std::string(argv[1]) == "--benchmark";
    const auto a = create_isotropic_parameters(3., 5.2, 2.5);
    const auto b = create_isotropic_parameters(3.5, 6., 2.8);
    const auto h = create_isotropic_parameters(4.5, 8., 3.3);
    const std::vector<Model> models = {
        create_layer_cake_model({a,b}, {10.,20.}, h),
        create_grid_model({0.,10.,30.}, {a,b,h}, "stiffness"),
        create_grid_model({0.,10.,30.}, {a,b,h}, "velocity")};
    std::cout << std::setprecision(17);
    for (size_t m = 0; m < models.size(); ++m) {
        for (const auto wave : {"Rayleigh", "Love"}) {
            for (bool full : {false, true}) {
                Options options;
                options.max_steps = 4;
                options.tolerance = 1e-8;
                const std::vector<std::string> outputs = full
                    ? std::vector<std::string>{"phase_velocity", "group_velocity", "hv_ratio"}
                    : std::vector<std::string>{"phase_velocity"};
                const std::vector<double> periods = {30.,10.,20.,10.,40.};
                const auto curve = compute_dispersion_curve(models[m], periods, wave, 0, outputs, options);
                check(curve.periods == periods && curve.solutions.size() == periods.size());
                std::map<double,ModeSolution> serial;
                for (double p : periods) serial.emplace(p, ModeSolution());
                auto next = options;
                for (auto& entry : serial) {
                    entry.second = compute_dispersion(models[m], entry.first, wave, 0, outputs, next);
                    next.c_guess = entry.second.phase_velocity();
                }
                for (size_t i = 0; i < periods.size(); ++i) {
                    const auto& s = curve.solutions[i];
                    same(s, serial.at(periods[i]));
                    if (!full) check(!s.group_velocity() && !s.hv_ratio());
                    std::cout << "value " << m << ' ' << wave << ' ' << full << ' ' << periods[i]
                              << ' ' << s.phase_velocity() << ' ' << s.group_velocity().value_or(0)
                              << ' ' << s.hv_ratio().value_or(0) << ' ' << s.diagnostics().accepted_level
                              << ' ' << s.diagnostics().generated_level << ' ' << s.diagnostics().relative_error << '\n';
                }
                same(compute_dispersion_curve(models[m], {20.}, wave, 0, outputs, options).solutions[0],
                     compute_dispersion(models[m], 20., wave, 0, outputs, options));
                if (benchmark) {
                    const auto calibration_start = std::chrono::steady_clock::now();
                    for (int n = 0; n < 3; ++n)
                        compute_dispersion_curve(models[m], periods, wave, 0, outputs, options);
                    const double seconds_per_call = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-calibration_start).count()/3;
                    const int iterations = std::max(1, static_cast<int>(std::ceil(0.1/seconds_per_call)));
                    for (int run = 0; run < 5; ++run) {
                        const auto start = std::chrono::steady_clock::now();
                        for (int n = 0; n < iterations; ++n)
                            compute_dispersion_curve(models[m], periods, wave, 0, outputs, options);
                        const auto end = std::chrono::steady_clock::now();
                        std::cout << "time " << m << ' ' << wave << ' ' << full << ' '
                                  << std::chrono::duration<double>(end-start).count()/iterations << '\n';
                    }
                }
            }
        }
    }
    // Errors must not poison the next independent calculation.
    for (const auto& periods : std::vector<std::vector<double>>{{}, {20.,-1.}}) {
        bool rejected = false;
        try { compute_dispersion_curve(models[0], periods); }
        catch (const Error& e) { rejected = e.code == "invalid_argument"; }
        check(rejected);
    }
    same(compute_dispersion_curve(models[0], {20.}).solutions[0],
         compute_dispersion(models[0], 20.));
}
