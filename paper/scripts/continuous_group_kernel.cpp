// Manuscript-only diagnostic. Uses the selected Disper26 checkout at compile time.
// stdin: N wave(0=Rayleigh,1=Love) omega c Q; N+1 rows rho A C F L N h;
// Q Gauss-Legendre fractions/weights on [0,1]. Material order: rho,A,C,F,L,N.
// Output: pointwise phase/group kernels, quadrature integrals and analytic integrals.
#include "disper26.cpp"
#include "disper26_analytical.hpp"
#include "sensitivity_utils.hpp"
#include <iomanip>
int main() {
 try {
  int n,w,q; double om,cp; std::cin>>n>>w>>om>>cp>>q;
  std::vector<PhysicalParameters<double>> layers(n); PhysicalParameters<double> hs;
  std::vector<double> h(n), x(q), weight(q);
  for(int j=0;j<=n;++j) { auto& p=j<n?layers[j]:hs; double hj;
   std::cin>>p.rho>>p.c11>>p.c33>>p.c13>>p.c44>>p.c66>>hj;
   if(j<n)h[j]=hj;
  }
  for(int i=0;i<q;++i)std::cin>>x[i]>>weight[i];
  if(!std::cin) throw std::runtime_error("invalid diagnostic input");
  const WaveType wave=w?WaveType::Love:WaveType::Rayleigh;
  const double U=calculateGroupVelocityFullAD(layers,h,hs,om,cp,wave);
  using T=ScalarADc;
  auto constant=[](double v){return make_ad_seeded<T>(v,1);};
  T omega=make_ad_seeded<T>(om,1,0), c=constant(cp);
  c.derivatives()(0)=cp/om*(1-cp/U);
  T k=omega/c;
  std::vector<PhysicalParameters<T>> p; std::vector<T> ht;
  for(int j=0;j<n;++j){p.push_back(to_dc_params(layers[j]));ht.push_back(constant(h[j]));}
  auto half=to_dc_params(hs);
  std::vector<RayleighStatePoint<T>> rp;
  LoveScaledBoundaryData<T> lp;
  T I3=constant(0);
  if(w){lp=buildLoveScaledBoundaryDataUpward<T>(p,ht,half,omega,c);
   I3=computeLoveAnalyticalIntegralsUpward<T>(p,ht,half,omega,c).I3;
  } else {rp=propagateRayleighEigenstates<T>(p,ht,half,omega,c,1,false);
   I3=computeRayleighEnergyIntegrals_Analytical<T>(rp,p,ht,half,omega,c,false).I3;
   if(rp.size()!=size_t(n+1))throw std::runtime_error("unexpected state count");
  }
  T coeff=c*c*c/(constant(2)*omega*omega*I3);
  auto analytic=computeGroupVelocityKernelFrequencyVariationalLevel(layers,h,hs,om,cp,wave,false,1,true,false);
  std::cout<<std::setprecision(17)<<"META "<<cp<<" "<<U<<"\n";
  double top=0;
  for(int j=0;j<n;++j){Eigen::VectorXd sum=Eigen::VectorXd::Zero(6);
   for(int t=0;t<q;++t){std::array<T,6> raw;for(auto& v:raw)v=constant(0);
    const auto& a=p[j];T z=constant(h[j]*x[t]);
    if(w){T ns=(a.c66*k*k-a.rho*omega*omega)/a.c44;
     T u=lp.states[j].u,tau=lp.states[j].tau,uz,tz;
     if(get_val(ns)>0){T nu=sqrt(ns),ch=cosh(nu*z),sh=sinh(nu*z),eta=a.c44*nu;
      uz=ch*u-sh/eta*tau;tz=-eta*sh*u+ch*tau;
     }else if(get_val(ns)<0){T nu=sqrt(-ns),co=cos(nu*z),si=sin(nu*z),eta=a.c44*nu;
      uz=co*u-si/eta*tau;tz=eta*si*u+co*tau;
     }else{uz=u-z/a.c44*tau;tz=tau;}
     T scale=constant(lp.relative_scale_sq[j]);
     raw[0]=-omega*omega*uz*uz*scale;raw[4]=tz*tz/(a.c44*a.c44)*scale;
     raw[5]=k*k*uz*uz*scale;
    }else{WaveVariables<T> v;v.omega=omega;v.k=k;v.z=-z;
     auto M=computeLayerMatrixP_SV_CT<T>(a,v);
     using CT=typename ComplexType<T>::type;
     Eigen::Matrix<CT,4,1> st;st<<rp[j].uz,rp[j].szz,rp[j].ur,rp[j].szr;
     auto state=(M*st).eval();T uz=state(0).real(),szz=state(1).real(),ur=state(2).real(),szr=state(3).real();
     raw[0]=-omega*omega*(uz*uz+ur*ur);raw[1]=k*k*ur*ur;
     T b=szz+a.c13*k*ur;raw[2]=b*b/(a.c33*a.c33);raw[3]=-constant(2)*k/a.c33*ur*b;
     raw[4]=szr*szr/(a.c44*a.c44);
    }
    std::array<double,6> kc,ku;
    for(int a=0;a<6;++a){T phase=coeff*raw[a],scaled=omega*phase/(c*c);
     kc[a]=phase.value();ku[a]=U*U*scaled.derivatives()(0);
     if(!std::isfinite(ku[a]))throw std::runtime_error("nonfinite kernel");
     sum(a)+=h[j]*weight[t]*ku[a];
    }
    std::cout<<"P "<<j<<" "<<top+h[j]*x[t];
    for(double v:kc)std::cout<<" "<<v;for(double v:ku)std::cout<<" "<<v;std::cout<<"\n";
   }
   std::cout<<"I "<<j;for(int a=0;a<6;++a)std::cout<<" "<<sum(a);std::cout<<"\n";
   top+=h[j];
  }
  for(int j=0;j<=n;++j){std::cout<<"A "<<j;for(int a=0;a<6;++a)std::cout<<" "<<analytic.group_material_vti(6*j+a);std::cout<<"\n";}
 } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
