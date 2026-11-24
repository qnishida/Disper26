#include "../src/disper26_analytical.hpp"
#include <cstdlib>

namespace {
using H=HessianAD<2>;
constexpr double nodes[8]={.095012509837637454,.28160355077925892,.45801677765722737,.61787624440264377,.755404408355003,.86563120238783176,.9445750230732326,.98940093499164994};
constexpr double weights[8]={.18945061045506859,.18260341504492361,.16915651939500262,.14959598881657676,.12462897125553403,.095158511682492591,.062253523938647706,.027152459411754037};
void close(double a,double b,const char* quantity) {
    if(!(std::abs(a-b)<=1e-10*(1.+std::abs(b)))) {
        std::cerr<<quantity<<": "<<std::setprecision(17)<<a<<" != "<<b<<'\n';std::exit(1);
    }
}
void compare(const H& a,const H& b) {
    close(a.r,b.r,"value");close(a.i,b.i,"imaginary value");
    for(int j=0;j<2;++j) {
        close(a.gr(j),b.gr(j),"gradient");close(a.gi(j),b.gi(j),"imaginary gradient");
        for(int k=0;k<2;++k) {
            close(a.hr()(j,k),b.hr()(j,k),"Hessian");close(a.hi()(j,k),b.hi()(j,k),"imaginary Hessian");
        }
    }
}
PhysicalParameters<H> convert(const PhysicalParameters<double>& p) {
    PhysicalParameters<H> r;r.rho=p.rho;r.c11=p.c11;r.c13=p.c13;
    r.c33=p.c33;r.c44=p.c44;r.c66=p.c66;return r;
}
void boundary_samples() {
    std::vector<LoveEigenfunctionPoint<double>> points(4);
    points[0].depth = 0.;
    points[1].depth = 1e-7;
    points[2].depth = 2e-7;
    points[3].depth = 1.;
    size_t cursor = 0;
    if (&Analytical::boundary_sample(points, 0., cursor) != &points[0] ||
        &Analytical::boundary_sample(points, 2e-7, cursor) != &points[2])
        throw std::runtime_error("Thin-layer boundary sampled at the wrong depth");
    try {
        Analytical::boundary_sample(points, 0.5, cursor);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Missing boundary sample accepted");
}
void rayleigh() {
    for(int model=0;model<5;++model) {
        auto raw=createIsotropicParameters(.5,1.,2.);
        if(model==1) {raw.rho=3.1;raw.c11=34.;raw.c13=26.;raw.c33=28.;raw.c44=5.;raw.c66=6.5;}
        if(model==2)raw.c13=-raw.c44; // no coupling: state-ODE limit
        if(model==3) {raw.rho=2.;raw.c11=30.;raw.c13=0.;raw.c33=20.;raw.c44=12.;raw.c66=10.;}
        if(model==4) {raw.rho=1.;raw.c11=4.;raw.c13=.5;raw.c33=1.;raw.c44=1.;raw.c66=1.;}
        auto p=convert(raw);
        const double vs=std::sqrt(raw.c44/raw.rho),vp=std::sqrt(raw.c11/raw.rho);
        for(double c:{vs*(1-1e-10),vs,vs*(1+1e-10),vp*(1-1e-10),vp,vp*(1+1e-10),.8*vs,std::sqrt(15./16.)}) {
            std::vector<double> depths{0.,.2};
            WaveVariables<double> base;base.omega=.7;base.k=.7/c;base.z=0.;
            RayleighEntireValues<double> v;
            if(computeRayleighEntireValues(raw,base,v)) {
                for(auto n:{v.n1,v.n3}) if(std::abs(n)>1e-3)
                    for(double q:{.0025,.25})for(double side:{1.-1e-8,1.+1e-8})
                        depths.push_back(std::sqrt(q/std::abs(n))*side);
            }
            for(double depth:depths) {
                std::cout<<"model "<<model<<" c "<<c<<" h "<<depth<<'\n';
                H w=H::make_var(.7,0),velocity=H::make_var(c,1),h(depth);
                Eigen::Matrix<H,4,4> actual,expected=Eigen::Matrix<H,4,4>::Constant(H(0.));
                const std::array<H,4> state{H(.7),H(-.2),H(.3),H(.4)};
                disper26_energy::rayleigh_moments(p,h,w,H(w/velocity),state,actual);
                Eigen::Matrix<H,4,1> y;for(int i=0;i<4;++i)y(i)=state[i];
                // Independent Gaussian integration of the physical state
                // exponential, with AD through the quadrature as a control.
                for(int n=0;n<8;++n)for(double sign:{-1.,1.}) {
                    WaveVariables<H> wave;wave.omega=w;wave.k=w/velocity;
                    wave.z=-h*H(.5*(1.+sign*nodes[n]));
                    const Eigen::Matrix<H,4,1> z=computeRayleighStatePropagatorRegular(p,wave)*y;
                    expected+=H(.5*weights[n])*h*(z*z.transpose());
                }
                for(int i=0;i<4;++i)for(int j=0;j<4;++j)compare(actual(i,j),expected(i,j));
            }
        }
    }
}
void love() {
    auto p=convert(createIsotropicParameters(.5,1.,2.));
    for(double c:{.5*(1.-1e-10),.5,.5*(1.+1e-10),.4,.7}) {
        H w=H::make_var(.7,0),velocity=H::make_var(c,1),h(.2);
        H u2,du2;
        disper26_energy::love_moments(p,h,w,H(w/velocity),H(.7),H(-.2),u2,du2);
        H expected_u(0.),expected_du(0.);
        for(int n=0;n<8;++n)for(double sign:{-1.,1.}) {
            WaveVariables<H> wave;wave.omega=w;wave.k=w/velocity;wave.z=-h*H(.5*(1.+sign*nodes[n]));
            Eigen::Matrix<H,2,1> y;y<<H(.7),H(-.2);
            const Eigen::Matrix<H,2,1> z=computeLayerMatrixSH(p,wave)*y;
            expected_u+=H(.5*weights[n])*h*z(0)*z(0);
            expected_du+=H(.5*weights[n])*h*z(1)*z(1)/(p.c44*p.c44);
        }
        compare(u2,expected_u);compare(du2,expected_du);
    }
}
}
int main() {boundary_samples();rayleigh();love();std::cout<<"Energy moment values/gradients/Hessians passed\n";}
