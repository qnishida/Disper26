// Test-only probe of production propagation and independent finite-layer
// controls. No diagnostic switch is added to the production API.
#include <iomanip>
#include <string>
#include "../src/disper26.cpp"

using AD = Eigen::AutoDiffScalar<Eigen::Vector2d>;
using Material = PhysicalParameters<AD>;

void scalar(const AD& x) {
    auto number=[](double v) { if (std::isfinite(v)) std::cout<<v; else std::cout<<"null"; };
    std::cout<<'['; number(x.value()); std::cout<<',';
    number(x.derivatives()(0)); std::cout<<','; number(x.derivatives()(1)); std::cout<<']';
}
template<class Matrix> void matrix(const Matrix& a) {
    std::cout<<'[';
    for (int i=0;i<a.rows();++i) for (int j=0;j<a.cols();++j) {
        if (i || j) std::cout<<',';
        scalar(a(i,j));
    }
    std::cout<<']';
}
void complex_scalar(const std::complex<AD>& x) {
    std::cout<<'['; scalar(x.real()); std::cout<<','; scalar(x.imag()); std::cout<<']';
}

// Smooth test control in q=(nu*h)^2. No sqrt is differentiated near zero.
// 12 terms at |q| <= 1/4 bound value/first-derivative truncation far below
// binary64 roundoff. Outside this small region use the regular closed forms.
std::pair<AD,AD> entire_pair(const AD& q) {
    if (std::abs(q.value())<=0.25) {
        AD c(1.),s(1.),tc(1.),ts(1.);
        for (int j=1;j<=12;++j) {
            tc=tc*q/double((2*j-1)*(2*j));
            ts=ts*q/double((2*j)*(2*j+1));
            c+=tc; s+=ts;
        }
        return {c,s};
    }
    if (q.value()>0.) { AD x=sqrt(q); return {cosh(x),sinh(x)/x}; }
    AD x=sqrt(AD(-q)); return {cos(x),sin(x)/x};
}

Eigen::Matrix<AD,2,2> regular_sh(const Material& p, const WaveVariables<AD>& v) {
    const AD nu2=(p.c66*v.k*v.k-p.rho*v.omega*v.omega)/p.c44;
    const auto cs=entire_pair(AD(nu2*v.z*v.z));
    Eigen::Matrix<AD,2,2> a;
    a<<cs.first, AD(v.z*cs.second/p.c44), AD(p.c44*nu2*v.z*cs.second),cs.first;
    return a;
}
Eigen::Matrix<AD,2,2> regular_fluid(const Material& p, const WaveVariables<AD>& v) {
    const AD b=1./p.c33-v.k*v.k/(p.rho*v.omega*v.omega);
    const AD d=-p.rho*v.omega*v.omega;
    const auto cs=entire_pair(AD(b*d*v.z*v.z));
    Eigen::Matrix<AD,2,2> a;
    a<<cs.first, AD(b*v.z*cs.second), AD(d*v.z*cs.second),cs.first;
    return a;
}

// Same vertical normalization and final stop-gradient scale as the core;
// only finite-layer matrices use the nonsingular diagnostic controls.
template<class Matrix> Matrix splice(const Matrix& original, const Matrix& regular, int mode) {
    Matrix result=mode==1 ? original : regular;
    for (int i=0;i<result.rows();++i) for (int j=0;j<result.cols();++j) {
        if (mode==1) result(i,j).value()=regular(i,j).value();
        if (mode==2) result(i,j).value()=original(i,j).value();
    }
    return result;
}

AD regular_residual(const std::vector<Material>& layers, const std::vector<AD>& h,
                    const Material& hs, AD w, AD c, WaveType wave, int mode=0) {
    WaveVariables<AD> v; v.omega=w; v.k=w/c; v.z=0.;
    if (wave==WaveType::Love) {
        Eigen::Matrix<AD,2,1> y=computeInitialVectorSH(hs,v);
        for (int i=int(layers.size())-1;i>=0;--i) {
            v.z=h[i]; // SH coordinate positive upward
            const auto a=splice(computeLayerMatrixSH(layers[i],v),regular_sh(layers[i],v),mode);
            y=(a*y).eval();
            if (get_magnitude(y(0))>1e-20) { AD scale=y(0); y=y/scale; }
        }
        return -y(1); // Preserve the public Love secular residual
    }
    Eigen::Matrix<AD,5,1> y=computeInitialVector(hs,v);
    Eigen::Matrix<AD,2,1> z;
    bool fluid=false;
    for (int i=int(layers.size())-1;i>=0;--i) {
        v.z=h[i];
        if (isLiquidLayer(layers[i])) {
            if (!fluid) { z<<y(I14),y(I24); fluid=true; }
            const auto a=splice(computeLayerMatrixLiquid(layers[i],v),regular_fluid(layers[i],v),mode);
            z=(a*z).eval();
            if (get_magnitude(z(0))>1e-20) { AD scale=z(0); z=z/scale; }
        } else {
            const auto a=splice(computeLayerMatrix(layers[i],v),computeRayleighReducedPropagatorRegular(layers[i],v),mode);
            y=(a*y).eval();
            if (get_magnitude(y(0))>1e-20) { AD scale=1./y(0); y=y*scale; }
        }
    }
    if (fluid) return z(1);
    const double scale=get_magnitude(y(I14));
    return scale>1e-30 ? AD(y(I24)/scale) : y(I24);
}

double group_velocity(const AD& d, double w, double c) {
    return c/(1.+w/c*d.derivatives()(0)/d.derivatives()(1));
}

int main() {
    std::cout<<std::setprecision(17);
    int n, kind, selected;
    double omega,velocity;
    while (std::cin>>n>>kind>>selected>>omega>>velocity) {
        auto read=[]() {
            double rho,a,f,c,l,nn;
            std::cin>>rho>>a>>f>>c>>l>>nn;
            Material p; p.rho=rho; p.c11=a; p.c13=f; p.c33=c; p.c44=l; p.c66=nn;
            return p;
        };
        std::vector<Material> layers;
        std::vector<AD> thicknesses;
        for (int i=0;i<n;++i) { layers.push_back(read()); double h;std::cin>>h;thicknesses.emplace_back(h); }
        Material hs=read();
        AD w(omega),c(velocity); w.derivatives()(0)=1.;c.derivatives()(1)=1.;
        const WaveType wave=kind==0 ? WaveType::Rayleigh : WaveType::Love;
        // Exercise the public U implementation as well as its AD residual.
        // In particular, a direct-chain candidate can bypass that residual.
        auto values=[](const Material& p) {
            PhysicalParameters<double> a;
            a.rho=p.rho.value();a.c11=p.c11.value();a.c13=p.c13.value();
            a.c33=p.c33.value();a.c44=p.c44.value();a.c66=p.c66.value();
            return a;
        };
        std::vector<PhysicalParameters<double>> plain_layers;
        std::vector<double> plain_h;
        for(const auto& p:layers) plain_layers.push_back(values(p));
        for(const auto& h:thicknesses) plain_h.push_back(h.value());
        const double production_u=calculateGroupVelocityFullAD(plain_layers,plain_h,
            values(hs),omega,velocity,wave);
        std::vector<double> scales;
        calculateDispersionFunction<double>(plain_layers,plain_h,values(hs),omega,
            velocity,wave,NormalizationMode::VectorNorm,nullptr,&scales);
        const double provided_u=calculateGroupVelocityFullAD(plain_layers,plain_h,
            values(hs),omega,velocity,wave,&scales);
        WaveVariables<AD> v;v.omega=w;v.k=w/c;v.z=0.;
        AD d=calculateDispersionFunction(layers,thicknesses,hs,w,c,wave,NormalizationMode::Vertical)(I24);
        AD dr=regular_residual(layers,thicknesses,hs,w,c,wave);
        std::cout<<"{\"D\":";scalar(d);std::cout<<",\"regular_D\":";scalar(dr);
        std::cout<<",\"U\":";scalar(AD(production_u));
        std::cout<<",\"AD_U\":";scalar(AD(group_velocity(d,omega,velocity)));
        std::cout<<",\"provided_U\":";scalar(AD(provided_u));
        std::cout<<",\"regular_U\":";scalar(AD(group_velocity(dr,omega,velocity)));
        const AD values_only=regular_residual(layers,thicknesses,hs,w,c,wave,1);
        const AD derivatives_only=regular_residual(layers,thicknesses,hs,w,c,wave,2);
        std::cout<<",\"regular_values_only_U\":";scalar(AD(group_velocity(values_only,omega,velocity)));
        std::cout<<",\"regular_derivatives_only_U\":";scalar(AD(group_velocity(derivatives_only,omega,velocity)));
        std::cout<<",\"initial\":";
        if (kind==0) matrix(computeInitialVector(hs,v)); else matrix(computeInitialVectorSH(hs,v));
        if (selected>=0 && selected<n) {
            const Material& p=layers[selected];v.z=thicknesses[selected];
            if (kind==0 && !isLiquidLayer(p)) {
                const auto dv=computeDerivedValues(p,v);
                std::cout<<",\"nu1_sq\":";complex_scalar(dv.nu1_sq);
                std::cout<<",\"nu3_sq\":";complex_scalar(dv.nu3_sq);
                std::cout<<",\"matrix\":";matrix(computeLayerMatrix(p,v));
                std::cout<<",\"regular_matrix\":";matrix(computeRayleighReducedPropagatorRegular(p,v));
            } else {
                std::cout<<",\"matrix\":";
                if (kind==1) matrix(computeLayerMatrixSH(p,v)); else matrix(computeLayerMatrixLiquid(p,v));
                std::cout<<",\"regular_matrix\":";
                if (kind==1) matrix(regular_sh(p,v)); else matrix(regular_fluid(p,v));
            }
        }
        std::cout<<"}\n"<<std::flush;
    }
}
