// Fixed-state comparison: public analytical and variational layer integrals.
#include "../src/disper26_analytical.hpp"
#include <iomanip>

int main() {
    std::cout << std::setprecision(17);
    int wave;
    while (std::cin >> wave) {
        PhysicalParameters<double> p;
        double h,w,c;
        std::cin >> p.rho >> p.c11 >> p.c13 >> p.c33 >> p.c44 >> p.c66 >> h >> w >> c;
        double y[4]; for(auto& x:y) std::cin >> x;
        const auto hs=createIsotropicParameters(4.6,8.,3.3);
        EnergyIntegralResult<double> a;
        std::array<double,3> b{};
        if(wave==0) {
            std::vector<RayleighEigenfunctionPoint<double>> states{
                {0.,y[2],y[0],y[1],y[3]}, {h,0.,0.,0.,0.}};
            a=Analytical::computeRayleighEnergyIntegrals_Analytical(states,{p},{h},hs,w,c);
            const auto x=computeRayleighLayerIntegrals_Analytical(y[0],y[1],y[2],y[3],p,h,w,w/c);
            const double k=w/c, aa=p.c11-p.c13*p.c13/p.c33;
            b={p.rho*x.int_uz2_ur2,
               x.int_szz2/p.c33+x.int_szr2/p.c44+k*k*aa*x.int_ur2,
               aa*x.int_ur2+(x.int_uz_szr-p.c13/p.c33*x.int_szz_ur)/k};
        } else {
            std::vector<LoveEigenfunctionPoint<double>> states{
                {0.,y[0],y[1]}, {h,0.,0.}};
            a=Analytical::computeLoveEnergyIntegrals_Analytical(states,{p},{h},hs,w,c);
            const double k=w/c;
            const auto x=computeLoveLayerIntegrals(y[0],y[1],p,h,w,k,LoveBranchPolicy::Auto);
            b={p.rho*x.int_u2,p.c44*x.int_dudz2+k*k*p.c66*x.int_u2,p.c66*x.int_u2};
        }
        auto number=[](double x) { if(std::isfinite(x))std::cout<<x;else std::cout<<"null"; };
        std::cout << "{\"public\":[";number(a.I1);std::cout<<',';number(a.I2);std::cout<<',';number(a.I3);
        std::cout << "],\"variational\":[";number(b[0]);std::cout<<',';number(b[1]);std::cout<<',';number(b[2]);
        std::cout << "]}\n" << std::flush;
    }
}
