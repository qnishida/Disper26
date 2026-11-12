// Independent C++ inputs and direct calls provide the Fortran ABI oracle.
#include "../src/dispersion_interface.hpp"
#include <fstream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    std::ofstream out(argv[1]);
    if (!out) return 1;
    out << std::setprecision(17);
    const auto hs=createIsotropicParameters(4.5,8.0,3.3);
    for (int scenario=1; scenario<=7; ++scenario) {
        const double omega=2*M_PI/(scenario==7 ? 2 : 20);
        const auto wave = scenario==2 || scenario==4 ? WaveType::Love : WaveType::Rayleigh;
        auto solid=createIsotropicParameters(3.0,5.2,2.5);
        if (scenario==3 || scenario==4) solid=make_vti_from_velocities(2.5,5.4,5.1,3.2,3.0,0.85);
        std::vector<PhysicalParameters<double>> layers{solid};
        std::vector<double> h{10};
        if (scenario==5) {
            layers.insert(layers.begin(),createIsotropicParameters(0.0,1.5,1.03));
            h.insert(h.begin(),3);
        }
        if (scenario==6) { layers.clear(); h.clear(); }
        auto f=[&](double c) {
            const auto state=calculateDispersionFunction(layers,h,hs,omega,c,wave);
            return state(I24);
        };
        const auto roots=searchRoots(f,1.0,4.49,(4.49-1.0)/500);
        if (roots.empty()) return 2;
        if (scenario==7 && roots.size()<2) return 2;
        out << roots.size() << '\n';
        for (double c:roots) {
            const double u=calculateGroupVelocityFullAD(layers,h,hs,omega,c,wave);
            double hv=std::numeric_limits<double>::quiet_NaN();
            if (wave==WaveType::Rayleigh) {
                Eigen::Matrix<double,5,1> seabed=Eigen::Matrix<double,5,1>::Zero();
                const auto state=calculateDispersionFunction(layers,h,hs,omega,c,wave,NormalizationMode::Vertical,&seabed);
                hv=scenario==5 ? calculateHVRatio(seabed) : calculateHVRatio(state);
            }
            out << c << ' ' << u << ' ' << hv << '\n';
        }
    }
    for (int scenario=1; scenario<=6; ++scenario) {
        std::vector<double> z{0,3,3,23,23},rho{2.5,2.6,2.7,2.8,3.3};
        std::vector<double> vp{5.2,5.4,5.8,6.2,8},vs{3,3.1,3.2,3.5,4.5};
        std::vector<bool> disc{false,true,false,true,false};
        auto vph=vp,vpv=vp,vsh=vs;
        std::vector<double> eta(5,0.9);
        for(int i=0;i<5;++i) { vph[i]*=1.02; vpv[i]*=0.99; vsh[i]*=1.03; }
        if (scenario==5) {
            rho[0]=rho[1]=1.03; vp[0]=vp[1]=1.5; vs[0]=vs[1]=0;
        }
        auto refiner=[&](int k) {
            return scenario==3 || scenario==4
                ? refine_velocity_grid_vti(z,rho,vph,vpv,vsh,vs,eta,disc,k)
                : refine_velocity_grid_iso(z,rho,vp,vs,disc,k);
        };
        auto r=compute_dispersion_romberg_result_from_refiner(refiner,20,
            scenario==2 || scenario==4 ? "Love" : "Rayleigh",scenario==5 ? 3.7 : 4.0,
            scenario==6 ? 3 : 5,scenario==6 ? 1e-8 : 1e-10,false,scenario!=6,
            scenario==6 ? 0.02 : 0.0,scenario==6 ? 1 : 0);
        if (!std::isfinite(r.phase_velocity) || !std::isfinite(r.group_velocity)) return 3;
        // These models must exercise roots, not the legacy guess fallback.
        if (std::abs(r.phase_velocity-(scenario==5 ? 3.7 : 4.0)) < 1e-6) return 4;
        out << r.phase_velocity << ' ' << r.group_velocity << ' ' << r.hv_ratio << ' '
            << r.converged_level << ' ' << r.relative_error << '\n';
    }
    return out ? 0 : 1;
}
