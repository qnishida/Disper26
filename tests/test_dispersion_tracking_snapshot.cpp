// Run the same executable source against two builds and compare stdout to check
// roots, observables and tracking diagnostics across a bookkeeping optimization.
#include "api.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace disper26::api;

int main() {
    const auto a = create_isotropic_parameters(3.,5.2,2.5);
    const auto b = create_isotropic_parameters(3.5,6.,2.8);
    const auto h = create_isotropic_parameters(4.5,8.,3.3);
    const std::vector<Model> models = {
        create_pancake_model({a,b},{20.,40.},h),
        create_grid_model({0.,20.,60.},{a,b,h},"stiffness"),
        create_grid_model({0.,20.,60.},{a,b,h},"velocity")};
    // More than three unique periods exercise history eviction; include a wide
    // step, input reordering and a duplicate without changing the sorted history.
    const std::vector<double> periods = {8.8,8.,8.2,8.4,8.2,8.6,12.};
    std::cout << std::setprecision(17);
    for (size_t m=0; m<models.size(); ++m)
    for (const auto wave : {"Love","Rayleigh"})
    for (int mode : {0,1})
    for (const auto predictor : {"auto","history","group_velocity"})
    for (int method=0; method<3; ++method)
    for (bool full : {false,true}) {
        Options o;
        o.max_steps=4; o.tolerance=1e-8; o.predictor=predictor;
        o.method=method==0?"mode_order":"continuation";
        if(method==2) o.verify_mode=true;
        const std::vector<std::string> outputs=full
            ? std::vector<std::string>{"phase_velocity","group_velocity","hv_ratio"}
            : std::vector<std::string>{"phase_velocity"};
        const auto curve=compute_dispersion_curve(models[m],periods,wave,mode,outputs,o);
        if(curve.periods!=periods || curve.solutions[2].phase_velocity()!=curve.solutions[4].phase_velocity())
            throw std::runtime_error("Period order or duplicate changed");
        for(const auto& s:curve.solutions) {
            const auto& d=s.diagnostics();
            if(d.predictors.size()!=d.level_mode_verified.size() ||
               d.mode_verified!=std::all_of(d.level_mode_verified.begin(),d.level_mode_verified.end(),[](bool v){return v;}) ||
               (method!=1 && !d.mode_verified))
                throw std::runtime_error("Inconsistent tracking diagnostics");
            std::cout<<m<<' '<<wave<<' '<<mode<<' '<<predictor<<' '<<method<<' '<<full<<' '
                <<s.period()<<' '<<s.phase_velocity()<<' '<<s.group_velocity().value_or(0)<<' '<<s.hv_ratio().value_or(0)<<' '
                <<d.accepted_level<<' '<<d.generated_level<<' '<<d.relative_error<<' '
                <<d.status<<' '<<d.convergence<<' '<<d.method<<' '<<d.backend<<' '
                <<d.mode_verified<<' '<<d.mode_count_evaluations<<' '<<d.predictor_ad_evaluations<<' '<<d.fallback_count;
            for(bool v:d.level_mode_verified) std::cout<<' '<<v;
            for(const auto& v:d.predictors) std::cout<<" | "<<v;
            for(const auto& v:d.fallback_reasons) std::cout<<" | "<<v;
            for(const auto& v:d.messages) std::cout<<" | "<<v;
            for(const auto& v:d.availability) std::cout<<" | "<<v.first<<'='<<v.second;
            for(const auto& v:d.convergence_by_quantity) std::cout<<" | "<<v.first<<'='<<v.second;
            std::cout<<'\n';
        }
    }
}
