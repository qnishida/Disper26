#include <chrono>
#include <cstdlib>
#include <iostream>
#include "../src/disper26.cpp"

namespace {
std::string context;
void require(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
void compare(const Eigen::Matrix<double,5,5>& a,
             const Eigen::Matrix<double,5,5>& b, double rtol=1e-10) {
    // Existing high-precision regular-layer Frobenius criterion. Phase roots
    // have the separate, stricter relative 1e-12 equivalence requirement.
    require(a.allFinite() && b.allFinite(), "nonfinite matrix");
    const double error=(a-b).norm();
    if (error>1e-12+rtol*b.norm()) {
        std::cerr<<context<<" matrix difference "<<error<<" / "<<b.norm()<<'\n';
        require(false,"matrix mismatch");
    }
}
void verify() {
    int real=0, fallback=0, positive=0, negative=0, mixed=0;
    int critical_probes=0;
    auto iso=createIsotropicParameters(3.5,6.,2.7);
    for (int model=0; model<4; ++model) {
        auto p=iso;
        if(model==1) {p.c11*=1.1;p.c13*=.9;p.c66*=1.05;}
        if(model==2) {p.rho=2.;p.c11=30.;p.c33=20.;p.c44=12.;p.c66=10.;p.c13=0.;}
        if(model==3) {p.rho=3.1;p.c11=34.;p.c13=26.;p.c33=28.;p.c44=5.;p.c66=6.5;}
        const double vs=std::sqrt(p.c44/p.rho),vp=std::sqrt(p.c11/p.rho);
        for(double c : {.5,1.,2.8,vs*(1.-1e-12),vs,vs*(1.+1e-12),
                        vp*(1.-1e-12),vp,vp*(1.+1e-12),7.}) {
            for(double h : {0.,1e-10,.02,.2,2.}) {
                context="model="+std::to_string(model)+" c="+std::to_string(c)+" h="+std::to_string(h);
                WaveVariables<double> w;w.omega=.7;w.k=w.omega/c;w.z=h;
                Eigen::Matrix<double,5,5> matrix;
                bool used=tryRayleighReducedPropagatorReal(p,w,matrix);
                used ? ++real : ++fallback;
                RayleighEntireValues<double,double> values;
                if(used && computeRayleighEntireValues(p,w,values)) {
                    if(values.n1>0 && values.n3>0) ++positive;
                    else if(values.n1<0 && values.n3<0) ++negative;
                    else ++mixed;
                }
                const auto result=computeRayleighReducedPropagatorRealOrComplex(p,w);
                compare(result,computeLayerMatrix(p,w));
                if(h<=.2) compare(result,computeRayleighReducedPropagatorRegular(p,w),1e-10);
            }
        }
        // Sweep both sides of finite-layer critical speeds, including the
        // immediately adjacent binary64 values. The independent regular
        // matrix exponential avoids sharing the removable-singularity formula.
        for(double critical : {vs,vp}) {
            std::vector<double> speeds={critical,
                std::nextafter(critical,0.),
                std::nextafter(critical,std::numeric_limits<double>::infinity())};
            for(int exponent=2;exponent<=15;++exponent)
                for(double sign : {-1.,1.})
                    speeds.push_back(critical*(1.+sign*std::pow(10.,-exponent)));
            for(double c : speeds) for(double omega : {.001,.7,10.})
                for(double h : {0.,1e-10,.02}) {
                    context="critical sweep model="+std::to_string(model);
                    WaveVariables<double> wave;wave.omega=omega;wave.k=omega/c;wave.z=h;
                    const auto result=computeRayleighReducedPropagatorRealOrComplex(p,wave);
                    compare(result,computeLayerMatrix(p,wave));
                    compare(result,computeRayleighReducedPropagatorRegular(p,wave));
                    ++critical_probes;
                }
        }
    }
    require(real>0 && fallback>0 && positive>0 && negative>0 && mixed>0,"missing regimes");
    auto uncoupled=iso;uncoupled.c13=-uncoupled.c44;
    WaveVariables<double> w;w.omega=.7;w.k=.2;w.z=.2;
    Eigen::Matrix<double,5,5> matrix;
    require(!tryRayleighReducedPropagatorReal(uncoupled,w,matrix),"uncoupled fallback missing");
    // Coalescing roots: identical uncoupled vertical equations, then approach
    // the discriminant floor through c13+c44.
    auto degenerate=iso;
    degenerate.c11=degenerate.c33=degenerate.c44=10.;
    for(double coupling : {0.,1e-8,1e-5}) {
        degenerate.c13=-10.+coupling;
        const auto result=computeRayleighReducedPropagatorRealOrComplex(degenerate,w);
        compare(result,computeLayerMatrix(degenerate,w));
    }
    auto strong=iso;
    strong.rho=3.;strong.c11=strong.c33=30.;strong.c44=strong.c66=6.;strong.c13=24.;
    auto discriminant=[&](double c) {
        double k=.7/c,x=strong.rho*.7*.7-strong.c44*k*k;
        double y=strong.rho*.7*.7-strong.c11*k*k;
        double coupling=(strong.c13+strong.c44)*k;
        double m1=strong.c44*x+strong.c33*y+coupling*coupling;
        return m1*m1-4*strong.c33*strong.c44*x*y;
    };
    double lo=.5,hi=3.;
    require(discriminant(lo)<0 && discriminant(hi)>0,"boundary not bracketed");
    for(int j=0;j<60;++j) {
        double middle=.5*(lo+hi);
        if(discriminant(middle)<0)lo=middle;else hi=middle;
    }
    for(double c : {lo*(1.-1e-10),lo,hi,hi*(1.+1e-10)}) {
        w.k=.7/c;w.z=.02;context="discriminant boundary";
        compare(computeRayleighReducedPropagatorRealOrComplex(strong,w),
                computeRayleighReducedPropagatorRegular(strong,w),1e-10);
    }
    for(double q : {-25.,-.25000000000001,-.25,-1e-30,0.,1e-30,.25,.25000000000001,25.}) {
        auto a=disper26_math::compute_entire_cs_real(q);
        auto b=disper26_math::compute_entire_cs(std::complex<double>(q));
        require(std::abs(a.c-b.c.real())<=1e-12*(1+std::abs(b.c.real())),"C mismatch");
        require(std::abs(a.s-b.s.real())<=1e-12*(1+std::abs(b.s.real())),"S mismatch");
        require(std::abs(a.cm1-b.cm1.real())<=1e-12*(1+std::abs(b.cm1.real())),"Cm1 mismatch");
    }
    w.k=.2;w.z=1e6;
    require(!tryRayleighReducedPropagatorReal(iso,w,matrix),"nonfinite fallback missing");
    std::cout<<"PASS real="<<real<<" fallback="<<fallback
             <<" positive="<<positive<<" negative="<<negative<<" mixed="<<mixed
             <<" critical_probes="<<critical_probes<<'\n';
}
}

int main(int argc,char** argv) {
    if(argc==1) { verify();return 0; }
    if(argc==2 && std::string(argv[1])=="--probe") {
        PhysicalParameters<double> p;WaveVariables<double> w;
        while(std::cin>>p.rho>>p.c11>>p.c13>>p.c33>>p.c44>>p.c66>>w.omega>>w.k>>w.z) {
            Eigen::Matrix<double,5,5> m;
            bool used=tryRayleighReducedPropagatorReal(p,w,m);
            if(!used)m=computeLayerMatrix(p,w);
            std::cout<<std::setprecision(17)<<"{\"real\":"<<(used?"true":"false")<<",\"matrix\":[";
            for(int i=0;i<5;++i)for(int j=0;j<5;++j) {
                if(i||j)std::cout<<',';
                if(std::isfinite(m(i,j)))std::cout<<m(i,j);else std::cout<<"null";
            }
            std::cout<<"]}"<<std::endl;
        }
        return 0;
    }
    if(argc!=4)return 2;
    const std::string operation(argv[1]);
    const int n=std::stoi(argv[2]),iterations=std::stoi(argv[3]);
    if(n<1 || iterations<1)return 2;
    std::vector<PhysicalParameters<double>> layers;
    for(int i=0;i<n;++i) {
        double t=double(i)/n;
        layers.push_back(createIsotropicParameters(3.+.5*t,5.2+.8*t,2.5+.3*t));
    }
    std::vector<double> h(n,100./n);
    auto hs=createIsotropicParameters(4.6,8.,3.3);
    auto evaluate=[&](int j) {
        double omega=2*M_PI/20.,c=3.9+1e-9*(j%31);
        if(operation=="phase")return calculateDispersionPhaseValueDouble(layers,h,hs,omega,c,WaveType::Rayleigh);
        if(operation=="count")return double(calculateBranModeCountRayleighSolid(layers,h,hs,omega,c).n_integer);
        if(operation=="layer") {
            WaveVariables<double> w;w.omega=omega;w.k=omega/c;w.z=h[j%n];
#if defined(DISPER26_EXPERIMENTAL_REAL_RAYLEIGH) && DISPER26_EXPERIMENTAL_REAL_RAYLEIGH
            return computeRayleighReducedPropagatorRealOrComplex(layers[j%n],w).sum();
#else
            return computeLayerMatrix(layers[j%n],w).sum();
#endif
        }
        std::exit(2);
    };
    volatile double checksum=0.;
    for(int j=0;j<10;++j)checksum+=evaluate(j);
    auto start=std::chrono::steady_clock::now();
    for(int j=0;j<iterations;++j)checksum+=evaluate(j);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/iterations;
    std::cout<<std::setprecision(17)<<seconds<<' '<<checksum<<'\n';
}
