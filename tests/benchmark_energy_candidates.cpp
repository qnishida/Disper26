// Fixed-root public analytical energy, with optional eigenfunction generation.
#include <chrono>
#include <iomanip>
#include "../src/disper26_analytical.hpp"

int main(int argc,char** argv) {
    if(argc!=5) return 2;
    const int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]);
    const std::string wave_name=argv[3],operation=argv[4];
    if(n<1||iterations<1||(wave_name!="Love"&&wave_name!="Rayleigh")||
        (operation!="integrals"&&operation!="fields")) return 2;
    const WaveType wave=wave_name=="Love"?WaveType::Love:WaveType::Rayleigh;
    std::vector<PhysicalParameters<double>> layers;
    for(int i=0;i<n;++i) {
        const double x=double(i)/n;
        layers.push_back(createIsotropicParameters(3.+.5*x,5.2+.8*x,2.5+.3*x));
    }
    const std::vector<double> h(n,100./n);
    const auto hs=createIsotropicParameters(4.6,8.,3.3);
    const double w=2.*std::acos(-1.)/20.;
    auto residual=[&](double c) {return calculateDispersionFunction(layers,h,hs,w,c,wave)(I24);};
    const auto roots=searchRoots(residual,2.5,4.5,.005,1e-10);
    if(roots.empty()) return 1;
    const double c=roots.front();
    std::vector<RayleighEigenfunctionPoint<double>> rayleigh;
    std::vector<LoveEigenfunctionPoint<double>> love;
    if(wave==WaveType::Love) love=computeLoveEigenfunctions(layers,h,hs,w,c,1);
    else rayleigh=computeRayleighEigenfunctions(layers,h,hs,w,c,1);
    auto evaluate=[&]() {
        if(wave==WaveType::Love) {
            if(operation=="fields") love=computeLoveEigenfunctions(layers,h,hs,w,c,1);
            return Analytical::computeLoveEnergyIntegrals_Analytical(love,layers,h,hs,w,c);
        }
        if(operation=="fields") rayleigh=computeRayleighEigenfunctions(layers,h,hs,w,c,1);
        return Analytical::computeRayleighEnergyIntegrals_Analytical(rayleigh,layers,h,hs,w,c);
    };
    double checksum=0.;
    for(int i=0;i<50;++i) {const auto r=evaluate();checksum+=r.I1+r.I2+r.I3;}
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<iterations;++i) {const auto r=evaluate();checksum+=r.I1+r.I2+r.I3;}
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto r=evaluate();
    if(!std::isfinite(checksum)) return 1;
    std::cout<<std::setprecision(17)<<elapsed/iterations<<' '<<r.I1<<' '<<r.I2<<' '<<r.I3<<' '<<c<<' '<<checksum<<'\n';
}
