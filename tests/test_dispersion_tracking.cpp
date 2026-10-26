#include "../src/api_impl.hpp"
#include <iostream>

int main() {
    // A good continuation prediction should not scan the whole old window.
    // A wider bracket must still reject two observable neighbouring roots.
    int evaluations=0;
    auto linear=[&](double c){++evaluations;return c-3.;};
    const auto local=try_predicted_phase_velocity_root(linear,3.,0.,
        2.,4.,.03,1.e-12);
    if(!local || std::abs(*local-3.)>1.e-12 || evaluations>=8)
        throw std::runtime_error("Accurate prediction did not use direct Brent bracket");
    auto pair=[](double c){return (c-2.997)*(c-3.003);};
    if(try_predicted_phase_velocity_root(pair,3.,.001,2.,4.,.03,1.e-12))
        throw std::runtime_error("Wide predicted bracket hid neighbouring roots");
    // Residual metadata must preserve roots/evaluation counts for direct
    // Brent, scanned roots, exact endpoints, and rejected poles/multiple roots.
    for(double previous_error : {0.0,0.001}) {
        evaluations=0;
        const auto before=try_predicted_phase_velocity_root(linear,2.9995,previous_error,
            2.,4.,.03,1.e-12);
        const int calls=evaluations;
        double residual=123.;evaluations=0;
        const auto after=try_predicted_phase_velocity_root(linear,2.9995,previous_error,
            2.,4.,.03,1.e-12,&residual);
        if(before!=after || !after || evaluations!=calls || residual!=*after-3.)
            throw std::runtime_error("Cached local residual changed roots or evaluation count");
    }
    double rejected_residual=123.;
    if(try_predicted_phase_velocity_root(pair,3.,.001,2.,4.,.03,1.e-12,&rejected_residual) ||
       !std::isnan(rejected_residual))
        throw std::runtime_error("Multiple-root rejection retained a stale residual");
    auto pole=[](double c){return 1.0/(c-3.001);};
    if(try_phase_velocity_root_bracket(pole,3.,2.,4.,.01,.03,1.e-12,0.,&rejected_residual) ||
       !std::isnan(rejected_residual))
        throw std::runtime_error("Pole accepted as a cached root");
    std::vector<double> residuals={123.};
    auto quadratic=[](double x){return (x-1.1)*(x-2.0);};
    const auto roots=searchRoots(quadratic,0.5,2.0,0.25,1.e-12,&residuals);
    if(roots.size()!=2 || residuals.size()!=roots.size() || roots.back()!=2.)
        throw std::runtime_error("Scan residuals lost the exact endpoint");
    for(size_t i=0;i<roots.size();++i)
        if(residuals[i]!=quadratic(roots[i]))
            throw std::runtime_error("Scan residual does not belong to its root");
    namespace api=disper26::api;
    const auto low=api::create_isotropic_parameters(3.,5.2,2.5);
    const auto high=api::create_isotropic_parameters(4.5,8.,3.3);
    const auto model=api::create_pancake_model({low},{60.},high);
    const auto first=api::compute_dispersion(model,8.,"Rayleigh",1);
    auto data=api::Access::solution(first);
    data.mode=0; // Deliberately label an adjacent-mode root with the wrong index.
    data.diagnostics.mode_verified=false;
    data.trace_cache=std::make_shared<api::RootTraceCache>();
    const auto mislabeled=api::Access::solution(data);
    bool rejected=false;
    try { api::compute_kernels(mislabeled,"isotropic"); }
    catch(const api::Error& e) { rejected=e.code=="mode_mismatch"; }
    if(!rejected || mislabeled.phase_velocity()!=first.phase_velocity())
        throw std::runtime_error("Wrong-mode derivative was accepted or stored root was changed");
    if(data.trace_cache->mode_verified)
        throw std::runtime_error("Failed verification was cached as success");
    std::cout << "Wrong branch rejected before derivatives; saved root unchanged\n";

    // Seed a checked predictor with the *wrong* branch. A converged secular
    // root is not sufficient: the local integer certificate must reject it.
    const auto mesh=api::detail::mesh(model,0);
    const auto fundamental=api::compute_dispersion(model,8.,"Rayleigh",0);
    api::Options options;
    options.predictor="history";
    for (bool wrong : {false,true}) {
        api::detail::DispersionTracking tracking;
        const double seed=wrong?first.phase_velocity():fundamental.phase_velocity();
        tracking.histories[0].periods.push_back(7.99);
        tracking.histories[0].roots.push_back(seed);
        api::Diagnostics diagnostics;
        const double root=tracking.root(mesh,8.,WaveType::Rayleigh,0,
                                        options,seed,{},diagnostics);
        if(std::abs(root-fundamental.phase_velocity()) >
           1.e-11+1.e-9*std::abs(fundamental.phase_velocity()))
            throw std::runtime_error("Checked predictor selected the wrong mode");
        if(diagnostics.level_mode_verified != std::vector<bool>{true})
            throw std::runtime_error("Recovered root lost its mode certificate");
        if(wrong) {
            if(diagnostics.fallback_count!=1 || diagnostics.mode_count_evaluations<=2 ||
               diagnostics.fallback_reasons!=std::vector<std::string>{"local_mode_certificate"})
                throw std::runtime_error("Wrong-mode candidate did not fall back to count search");
        } else if(diagnostics.fallback_count!=0 || diagnostics.mode_count_evaluations!=2) {
            throw std::runtime_error("Correct candidate did not use two endpoint counts");
        }
    }
    std::cout << "Correct predictor certified; wrong-mode predictor recovered through count search\n";
    // Unchecked initial selection follows c_guess even with a wrong mode label.
    api::Options fast_options;
    fast_options.method="continuation";
    fast_options.c_guess=first.phase_velocity();
    const auto fast=api::compute_dispersion(model,8.,"Rayleigh",0,
                                            {"phase_velocity"},fast_options);
    if(fast.diagnostics().mode_count_evaluations!=0 || fast.diagnostics().mode_verified ||
       api::compute_mode_count(fast)!=1 || fast.mode()!=0)
        throw std::runtime_error("Explicit counting altered unchecked selection or diagnostics");
    // Metadata requested through adaptive/legacy scans must agree with a
    // fresh residual, including the legacy no-root return of the initial guess.
    for(const auto& history : std::vector<std::vector<double>>{{},
            {fundamental.phase_velocity()},
            {fundamental.phase_velocity(),fundamental.phase_velocity()}}) {
        for(double window : {0.0,0.01}) {
            double residual=123.;
            const auto before=disper26_stats::dispersion_evals_double;
            const double a=find_dispersion_root_adaptive(mesh.layers,mesh.thicknesses,
                mesh.halfspace,2*M_PI/8.,fundamental.phase_velocity(),WaveType::Rayleigh,
                1.e-12,history,window);
            const auto calls=disper26_stats::dispersion_evals_double-before;
            const auto start=disper26_stats::dispersion_evals_double;
            const double b=find_dispersion_root_adaptive(mesh.layers,mesh.thicknesses,
                mesh.halfspace,2*M_PI/8.,fundamental.phase_velocity(),WaveType::Rayleigh,
                1.e-12,history,window,api::missing_value,&residual);
            if(a!=b || disper26_stats::dispersion_evals_double-start!=calls ||
               residual!=calculateDispersionPhaseValueDouble(mesh.layers,mesh.thicknesses,
                   mesh.halfspace,2*M_PI/8.,b,WaveType::Rayleigh))
                throw std::runtime_error("Adaptive residual reuse changed root or evaluation count");
        }
    }
    double fallback_residual=123.;
    const double fallback=find_dispersion_root_near_by_scan(mesh.layers,mesh.thicknesses,
        mesh.halfspace,2*M_PI/8.,2.0,WaveType::Rayleigh,1.e-12,&fallback_residual);
    if(fallback!=2.0 || fallback_residual!=calculateDispersionPhaseValueDouble(
            mesh.layers,mesh.thicknesses,mesh.halfspace,2*M_PI/8.,fallback,WaveType::Rayleigh))
        throw std::runtime_error("No-root fallback lost its residual check");
    std::cout << "Unchecked initial root retained; explicit single count returns actual label\n";
    // Inject a jump after a genuinely certified first point. Endpoint recovery
    // must repair the intervening wrong value, not merely the last root.
    const std::vector<double> segment={7.98,7.99,8.};
    std::map<double,api::ModeSolution> corrupted;
    for(size_t i=0;i<segment.size();++i) {
        auto entry=api::Access::solution(api::compute_dispersion(model,segment[i],"Rayleigh",i?1:0));
        entry.mode=0;
        if(i) {
            entry.diagnostics.mode_verified=false;
            entry.diagnostics.level_mode_verified={false};
            entry.trace_cache=std::make_shared<api::RootTraceCache>();
        }
        corrupted.emplace(segment[i],api::Access::solution(std::move(entry)));
    }
    auto refiner=[&](int)->const VelocityGridRefinement& {return mesh;};
    for(const std::string response:{"fallback","error","missing"}) {
        auto repaired=corrupted;
        api::Options endpoint_options;
        endpoint_options.verify="endpoints";endpoint_options.on_violation=response;
        bool failed=false;
        try {
            api::detail::finish_tracking_segment(model,"Rayleigh",0,{"phase_velocity"},
                endpoint_options,refiner,nullptr,repaired,segment);
        } catch(const api::Error& e) {
            failed=e.code=="tracking_violation";
        }
        if(failed!=(response=="error")) throw std::runtime_error("Endpoint error policy differs");
        if(response=="error") continue;
        if(repaired.at(segment[0]).phase_velocity()!=corrupted.at(segment[0]).phase_velocity())
            throw std::runtime_error("Endpoint repair changed its certified anchor");
        for(size_t i=1;i<segment.size();++i) {
            const auto& solution=repaired.at(segment[i]);
            if(response=="missing") {
                if(!std::isnan(solution.phase_velocity()) || solution.diagnostics().mode_verified)
                    throw std::runtime_error("Endpoint missing policy retained a suspect value");
            } else {
                const double expected=api::compute_dispersion(model,segment[i]).phase_velocity();
                if(std::abs(solution.phase_velocity()-expected)>1e-9 || !solution.diagnostics().mode_verified)
                    throw std::runtime_error("Endpoint repair did not restore the whole segment");
            }
        }
        if(corrupted.at(segment[1]).phase_velocity()==repaired.at(segment[1]).phase_velocity())
            throw std::runtime_error("Injected wrong intermediate root was not replaced");
    }
}
