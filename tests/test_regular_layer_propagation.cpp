#include <cstdlib>
#include <iostream>
#include "../src/disper26.cpp"

namespace {
std::string context;
void require(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
void close(double a, double b, const char* message, double tol=1e-12) {
    if (!(std::abs(a-b) <= tol*(1.0+std::abs(b)))) {
        std::cerr << context << ": " << std::setprecision(17) << a << " != " << b << '\n';
        require(false,message);
    }
}
using H = HessianAD<2>;

// Independent extended-precision reference for values and q derivatives.
// More terms/precision than the production branch, also used outside it.
std::array<std::complex<long double>,3> reference_cs(std::complex<long double> q, int offset) {
    using C = std::complex<long double>;
    C value(0), first(0), second(0), power(1), previous(1), before_previous(1);
    long double factorial=1;
    for (int n=0;n<70;++n) {
        if (n) factorial*=static_cast<long double>((2*n-1+offset)*(2*n+offset));
        value+=power/factorial;
        if(n>=1) first+=static_cast<long double>(n)*previous/factorial;
        if(n>=2) second+=static_cast<long double>(n*(n-1))*before_previous/factorial;
        before_previous=previous;previous=power;power*=q;
    }
    return {value,first,second};
}
void check_entire() {
    for (double x : {-9.,-0.25000000000001,-0.25,-1e-12,0.,1e-12,0.25,0.25000000000001,25.}) {
        for (double y : {0.,0.01}) {
            H q(x,y);q.is_const=false;q.gr(0)=1.;q.gi(1)=1.;
            const auto cs=disper26_math::compute_entire_cs(q);
            using AD2=Eigen::AutoDiffScalar<Eigen::Vector2d>;
            AD2 re(x),im(y);
            re.derivatives()=Eigen::Vector2d(1.,0.);
            im.derivatives()=Eigen::Vector2d(0.,1.);
            const auto first=disper26_math::compute_entire_cs(std::complex<AD2>(re,im));
            for (int offset=0;offset<2;++offset) {
                const auto r=reference_cs({x,y},offset);const H& a=offset ? cs.s : cs.c;
                const auto& f=offset?first.s:first.c;
                close(f.real().value(),double(r[0].real()),"AD2 value real");
                close(f.imag().value(),double(r[0].imag()),"AD2 value imaginary");
                close(f.real().derivatives()(0),double(r[1].real()),"AD2 first real");
                close(f.imag().derivatives()(0),double(r[1].imag()),"AD2 first imaginary");
                close(f.real().derivatives()(1),-double(r[1].imag()),"AD2 imaginary seed real");
                close(f.imag().derivatives()(1),double(r[1].real()),"AD2 imaginary seed imaginary");
                close(a.r,double(r[0].real()),"entire value real");
                close(a.i,double(r[0].imag()),"entire value imag");
                close(a.gr(0),double(r[1].real()),"entire first real");
                close(a.gi(0),double(r[1].imag()),"entire first imag");
                close(a.gr(1),-double(r[1].imag()),"entire imaginary seed real");
                close(a.hr()(0,0),double(r[2].real()),"entire second real");
                close(a.hi()(0,0),double(r[2].imag()),"entire second imag");
                close(a.hr()(0,1),-double(r[2].imag()),"entire mixed derivative");
                close(a.hr()(1,1),-double(r[2].real()),"entire imaginary second");
            }
        }
    }
    using AD=Eigen::AutoDiffScalar<Eigen::VectorXd>;
    AD re(0.),im(0.);re.derivatives()=Eigen::VectorXd::Unit(3,0);im.derivatives()=Eigen::VectorXd::Zero(3);
    auto cs=disper26_math::compute_entire_cs(std::complex<AD>(re,im));
    require(cs.s.real().derivatives().size()==3,"dynamic derivative size");
    close(cs.s.real().value(),1.,"S(0)");close(cs.s.real().derivatives()(0),1./6.,"S'(0)");
    close(cs.c.real().derivatives()(0),.5,"C'(0)");
    H tiny=H::make_var(1e-30,0);
    close(disper26_math::compute_entire_cs(tiny).cm1.r/1e-30,.5,"C-1 small value");
}

template<class Matrix> void compare(const Matrix& actual,const Matrix& expected,const char* name) {
    for(int i=0;i<actual.rows();++i) for(int j=0;j<actual.cols();++j) {
        const H& a=actual(i,j);const H& b=expected(i,j);
        close(a.r,b.r,name);close(a.i,b.i,name);
        for(int k=0;k<2;++k) {
            close(a.gr(k),b.gr(k),name);close(a.gi(k),b.gi(k),name);
            for(int l=0;l<2;++l) {close(a.hr()(k,l),b.hr()(k,l),name);close(a.hi()(k,l),b.hi()(k,l),name);}
        }
    }
}
PhysicalParameters<H> convert(const PhysicalParameters<double>& p) {
    PhysicalParameters<H> a;
    a.rho=p.rho;a.c11=p.c11;a.c13=p.c13;a.c33=p.c33;a.c44=p.c44;a.c66=p.c66;
    a.gravity=p.gravity;return a;
}

void check_rayleigh() {
    auto iso=createIsotropicParameters(3.5,6.,2.7);
    for(int model=0;model<4;++model) {
        auto p=convert(iso);
        if(model==1) {p.c11=p.c11*1.1;p.c13=p.c13*.9;p.c66=p.c66*1.05;}
        // Strong VTI with complex-conjugate nu^2 roots.
        if(model==2) {p.rho=2.;p.c11=30.;p.c33=20.;p.c44=12.;p.c66=10.;p.c13=0.;}
        if(model==3) {p.rho=3.1;p.c11=34.;p.c13=26.;p.c33=28.;p.c44=5.;p.c66=6.5;}
        const double vs=std::sqrt(p.c44.r/p.rho.r), vp=std::sqrt(p.c11.r/p.rho.r);
        const double pivot=std::sqrt((p.c11.r+p.c44.r)/(2*p.rho.r));
        for(double speed : {.5,2.8,vs*(1.-1e-12),vs,vs*(1.+1e-12),vp*(1.-1e-12),vp,vp*(1.+1e-12),7.,pivot*(1.-1e-12),pivot,pivot*(1.+1e-12)}) {
            if(speed==.5 && model!=3)continue;
            context="model="+std::to_string(model)+", c="+std::to_string(speed);
            WaveVariables<H> v;v.omega=H::make_var(.7,0);H c=H::make_var(speed,1);v.k=v.omega/c;v.z=speed==.5 ? .02 : .2;
            compare(computeLayerMatrix(p,v),computeRayleighReducedPropagatorRegular(p,v),"Rayleigh reduced value/gradient/Hessian");
            compare(computeRayleighStatePropagatorEntire(p,v),computeRayleighStatePropagatorRegular(p,v),"Rayleigh state value/gradient/Hessian");
            v.z=0.;
            compare(computeLayerMatrix(p,v),computeRayleighReducedPropagatorRegular(p,v),"zero-thickness Rayleigh");
        }
    }
    auto p=convert(iso);p.c13=-p.c44; // uncoupled modes: regular fallback
    WaveVariables<H> v;v.omega=H::make_var(.7,0);v.k=v.omega/H::make_var(3.5,1);v.z=.2;
    compare(computeLayerMatrix(p,v),computeRayleighReducedPropagatorRegular(p,v),"uncoupled Rayleigh");
}

// Direct matrix exponential of the physical first-order 2x2 ODE. This does
// not share the production C/S functions, critical branches or modal roots.
Eigen::Matrix<H,2,2> ode_reference(const PhysicalParameters<H>& p,const WaveVariables<H>& v,bool fluid) {
    Eigen::Matrix<H,2,2> g;
    if(fluid) {
        H a=v.k*v.k*p.gravity/(v.omega*v.omega);
        H b=H(1.)/p.c33-v.k*v.k/(p.rho*v.omega*v.omega);
        H c=-p.rho*v.omega*v.omega+v.k*v.k*p.rho*p.gravity*p.gravity/(v.omega*v.omega);
        g<<a,b,c,-a;
    } else g<<H(0.),H(1.)/p.c44,p.c66*v.k*v.k-p.rho*v.omega*v.omega,H(0.);
    Eigen::Matrix<H,2,2> result,term;
    result<<H(1.),H(0.),H(0.),H(1.);term=result;
    for(int n=1;n<=32;++n) {term=(term*g).eval();term=term*(v.z/H(double(n)));result=result+term;}
    return result;
}
void check_sh_fluid() {
    for(bool fluid : {false,true}) for(double gravity : {0.,.1}) {
        if(!fluid && gravity!=0.)continue;
        auto p=convert(fluid ? createIsotropicParameters(0.,.5,2.) : createIsotropicParameters(.5,1.,2.));
        p.gravity=gravity;
        for(double c : {.5*(1.-1e-12),.5,.5*(1.+1e-12),std::sqrt(.25+gravity*gravity)}) {
            WaveVariables<H> v;v.omega=H::make_var(1.,0);v.k=v.omega/H::make_var(c,1);v.z=.1;
            const auto a=fluid ? computeLayerMatrixLiquid(p,v) : computeLayerMatrixSH(p,v);
            compare(a,ode_reference(p,v,fluid),"SH/fluid ODE value/gradient/Hessian");
            if(c==.5 && gravity==0.) {
                close(a(0,1).r,fluid ? 0. : .1/.5,"critical off-diagonal limit");
                close(a(0,0).gr(1),-.08,"critical diagonal derivative");
            }
        }
    }
}

void check_critical_love_mode_count() {
    // A critical bottom layer changes displacement by +h*T/c44 (z upward).
    // Omitting 1/c44 moved the first zero in the top layer from s<.41 to s>.41,
    // losing a mode only at the literal critical speed.
    const std::vector<PhysicalParameters<double>> p{
        createIsotropicParameters(.25,.6,2.),createIsotropicParameters(.5,1.,2.)};
    const std::vector<double> h{.41,.1};
    const auto hs=createIsotropicParameters(1.,2.,3.);
    const double traction=3.*std::sqrt(3.);
    const double displacement=1.+.1*traction/.5;
    const double r=std::sqrt(12.);
    const double crossing=std::atan(traction/(.125*r*displacement))/r;
    require(crossing<.41,"analytical Love crossing in top layer");
    for(double speed : {.5*(1.-1e-10),.5,.5*(1.+1e-10)}) {
        const auto count=calculateBranModeCountLoveSolid(p,h,hs,1.,speed);
        require(count.n_integer==1,"Love count continuous across critical layer");
        require(count.crossings.size()==1,"exactly one Love crossing");
        if(speed==.5)close(count.crossings[0].local_depth_km,.1+crossing,"analytical Love crossing depth");
    }
}
}
int main() {
    check_entire();check_sh_fluid();check_rayleigh();check_critical_love_mode_count();
    std::cout<<"Regular layer values, gradients and Hessians passed (atol=rtol=1e-12)\n";
}
