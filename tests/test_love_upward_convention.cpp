// Independent SH solutions pin coordinate/stress signs, not just root locations.
#include "../src/disper26_analytical.hpp"
#include <cstdlib>

namespace {
using State = Eigen::Vector2d;
using Material = PhysicalParameters<double>;
void require(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
void close(double actual, double expected, const char* message, double tol=1e-10) {
    if (!(std::abs(actual-expected) <= tol*(1.+std::abs(expected)))) {
        std::cerr << std::setprecision(17) << actual << " != " << expected << '\n';
        require(false, message);
    }
}

// Elementary homogeneous solution, independent of the production entire
// functions, propagators, initial vector and eigenfunction reconstruction.
State solution(const Material& p, double omega, double c, double z, State y) {
    const double k=omega/c;
    const double q=(p.c66*k*k-p.rho*omega*omega)/p.c44;
    double C=1., S=z;
    if (q>0.) { const double nu=std::sqrt(q); C=std::cosh(nu*z); S=std::sinh(nu*z)/nu; }
    if (q<0.) { const double mu=std::sqrt(-q); C=std::cos(mu*z); S=std::sin(mu*z)/mu; }
    return {C*y(0)+S*y(1)/p.c44, p.c44*q*S*y(0)+C*y(1)};
}
State halfspace(const Material& p, double omega, double c) {
    return {1.,p.c44*std::sqrt((p.c66*std::pow(omega/c,2)-p.rho*omega*omega)/p.c44)};
}
State surface(const std::vector<Material>& p, const std::vector<double>& h,
              const Material& hs, double omega, double c) {
    State y=halfspace(hs,omega,c);
    for (int i=int(p.size())-1;i>=0;--i) y=solution(p[i],omega,c,h[i],y);
    return y;
}

void initial_and_transfer() {
    const auto hs=createIsotropicParameters(4.5,8.,3.3);
    const auto p=createIsotropicParameters(.5,1.,2.);
    WaveVariables<double> wave; wave.omega=.7; wave.k=.7/3.3; wave.z=0.;
    const auto initial=computeInitialVectorSH(hs,wave);
    const auto expected=halfspace(hs,.7,3.3);
    close(initial(0),expected(0),"halfspace displacement",1e-12);
    close(initial(1),expected(1),"halfspace upward traction",1e-12);
    require(initial(1)>0.,"decaying halfspace must have positive T/W");
    for (double c : {.4,.5,.7}) for (double dz : {-.2,0.,.2}) {
        wave.k=wave.omega/c; wave.z=dz;
        const State y(.7,-.2);
        const State value=computeLayerMatrixSH(p,wave)*y;
        const State reference=solution(p,wave.omega,c,dz,y);
        close(value(0),reference(0),"signed SH displacement",1e-12);
        close(value(1),reference(1),"signed SH traction",1e-12);
        const auto plus=computeLayerMatrixSH(p,wave);
        wave.z=-dz;
        const auto minus=computeLayerMatrixSH(p,wave);
        Eigen::Matrix2d J=Eigen::Matrix2d::Identity(); J(1,1)=-1.;
        require((J*minus*J-plus).norm()<1e-12,"J*A(-h)*J=A(+h)");
    }
}

void fields_and_boundaries() {
    const std::vector<Material> p{createIsotropicParameters(3.,5.2,2.5),
        createIsotropicParameters(3.5,6.133333333333333,2.766666666666667),
        createIsotropicParameters(4.,7.066666666666667,3.033333333333333)};
    const std::vector<double> h(3,100./3.);
    const auto hs=createIsotropicParameters(4.5,8.,3.3);
    const double omega=2.*M_PI/20.;
    // Find roots by bisection of the independent elementary solution. Neither
    // production mode counting nor root finding is used for this reference.
    std::vector<double> roots;
    double left=3.00001, fleft=surface(p,h,hs,omega,left)(1);
    for (int i=1;i<=1000;++i) {
        const double right=3.00001+(4.499-3.00001)*i/1000.;
        const double fright=surface(p,h,hs,omega,right)(1);
        if (std::signbit(fleft)!=std::signbit(fright)) {
            double a=left,b=right,fa=fleft;
            for (int j=0;j<64;++j) {
                const double m=(a+b)*.5,fm=surface(p,h,hs,omega,m)(1);
                if (std::signbit(fm)==std::signbit(fa)) { a=m; fa=fm; } else b=m;
            }
            roots.push_back((a+b)*.5);
        }
        left=right; fleft=fright;
    }
    require(roots.size()>=2,"reference must include fundamental and overtone");
    for (int mode=0;mode<2;++mode) {
        const double c=roots[mode];
        std::vector<State> boundaries;
        const auto secular=calculateLoveDispersion(p,h,hs,omega,c,
            NormalizationMode::None,nullptr,nullptr,&boundaries);
        State upward=halfspace(hs,omega,c);
        close(boundaries[0](1),upward(1),"intermediate halfspace T");
        for (int i=2;i>=0;--i) {
            upward=solution(p[i],omega,c,h[i],upward);
            close(boundaries[3-i](0),upward(0),"intermediate W");
            close(boundaries[3-i](1),upward(1),"intermediate upward T");
        }
        close(secular(I24),-upward(1),"historical secular sign");
        const auto ef=computeLoveEigenfunctions(p,h,hs,omega,c,20);
        const auto down=propagateLoveEigenstates(p,h,omega,c);
        const auto scaled=buildLoveScaledBoundaryDataUpward(p,h,hs,omega,c);
        State y(1.,0.);
        for (int i=0;i<3;++i) {
            for (int j=1;j<=20;++j) {
                const auto exact=solution(p[i],omega,c,-h[i]*j/20.,y);
                const auto& pt=ef[i*20+j];
                close(pt.uy.real(),exact(0),"public displacement");
                close(pt.tyz.real(),exact(1),"public tyz=L*du/dz");
                close(pt.depth,(i+j/20.)*h[i],"saved depth positive downward");
            }
            y=solution(p[i],omega,c,-h[i],y);
            close(down[i+1].u,y(0),"downward sensitivity displacement");
            close(down[i+1].tau,y(1),"downward sensitivity upward stress");
            close(scaled.states[i+1].tau*y(0),scaled.states[i+1].u*y(1),"stored upward traction ratio");
        }
        require(std::abs(ef[20].tyz.real())>1e-3,"test a nonzero interface stress");
        close(ef.back().tyz.real(),halfspace(hs,omega,c)(1)*ef.back().uy.real(),"halfspace stress/displacement ratio");
        require(calculateBranModeCountLoveSolid(p,h,hs,omega,c).n_integer==mode,"reference-root mode label");
    }
}

void bottom_moments() {
    const auto p=createIsotropicParameters(.5,1.,2.);
    constexpr double nodes[8]={.095012509837637454,.28160355077925892,.45801677765722737,.61787624440264377,.755404408355003,.86563120238783176,.9445750230732326,.98940093499164994};
    constexpr double weights[8]={.18945061045506859,.18260341504492361,.16915651939500262,.14959598881657676,.12462897125553403,.095158511682492591,.062253523938647706,.027152459411754037};
    for (double c : {.4,.5,.7}) {
        const double omega=.7,h=.3;
        const State bottom(.7,-.2);
        double u2=0.,du2=0.;
        for (int n=0;n<8;++n) for (double sign : {-1.,1.}) {
            const auto y=solution(p,omega,c,h*.5*(1.+sign*nodes[n]),bottom);
            u2+=h*.5*weights[n]*y(0)*y(0);
            du2+=h*.5*weights[n]*std::pow(y(1)/p.c44,2);
        }
        const auto li=computeLoveLayerIntegralsFromBottomStateUpward(
            bottom(0),bottom(1),p,h,omega,omega/c,LoveBranchPolicy::Auto);
        close(li.int_u2,u2,"upward bottom displacement moment",1e-12);
        close(li.int_dudz2,du2,"upward bottom gradient moment",1e-12);
        const auto top=solution(p,omega,c,h,bottom);
        const auto down=computeLoveLayerIntegrals(top(0),top(1),p,h,omega,omega/c,LoveBranchPolicy::Auto);
        close(down.int_u2,u2,"downward top displacement moment",1e-12);
        close(down.int_dudz2,du2,"downward top gradient moment",1e-12);
    }
}

void crossing_signs() {
    const std::vector<Material> p{createIsotropicParameters(2.2,3.8,2.5),
                                createIsotropicParameters(1.,1.73,2.)};
    const auto hs=createIsotropicParameters(3.,5.2,2.7);
    const auto count=calculateBranModeCountLoveSolid(p,{3.,1.},hs,M_PI,1.2);
    require(count.crossings.size()==2,"oscillatory and evanescent crossings");
    require(count.crossings[0].contribution==-1 && count.crossings[1].contribution==1,
        "preserve Love signed crossing sequence");
    for (const auto& crossing : count.crossings) {
        const auto& layer=p[crossing.layer_index];
        const double r2=(layer.rho*M_PI*M_PI-layer.c66*std::pow(M_PI/1.2,2))/layer.c44;
        close(crossing.secular_derivative,-layer.c44*r2*crossing.conjugate,"physical dT/ds",1e-12);
        close(crossing.test_value,crossing.conjugate*crossing.secular_derivative,"physical W*dT/ds",1e-12);
        require(crossing.contribution==(crossing.test_value>0.?1:-1),"Love contribution=sign(W*dT/ds)");
    }
    close(count.crossings[0].local_depth_km,.870272833671631,"oscillatory crossing location",1e-12);
    close(count.crossings[1].local_depth_km,1.013666590492687,"evanescent crossing location",1e-12);
}
}
int main() {
    initial_and_transfer(); fields_and_boundaries(); bottom_moments(); crossing_signs();
    std::cout << "Upward Love states, public stresses, moments and crossing signs passed\n";
}
