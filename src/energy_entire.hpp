#pragma once

// Shared finite-layer quadratic moments. The modal coefficients are scaled
// before evaluation, as in the regular propagator. Only noncritical modes
// are resolved into growing/decaying exponentials.
namespace disper26_energy {

// Coalescing or uncoupled modes have no usable modal basis. Integrate the
// physical state equation instead: integrate E'=G^T E+E G, E(0)=W,
// followed by exact interval doubling. No inverse eigenvector matrix.
template<class T>
void rayleigh_moments_ode(const PhysicalParameters<T>& p,T h,T w,T k,
    const std::array<typename ComplexType<T>::type,4>& y,
    Eigen::Matrix<typename ComplexType<T>::type,4,4>& result) {
    using CT=typename ComplexType<T>::type;
    using M=Eigen::Matrix<CT,4,4>;
    const CT zero=toComplex(make_zero_like(p.rho)), one=ad_const(zero,1.);
    M g=M::Constant(zero), identity=M::Constant(zero);
    for(int i=0;i<4;++i)identity(i,i)=one;
    g(0,1)=-one/toComplex(p.c33);g(0,2)=-toComplex<T>(p.c13*k/p.c33);
    g(1,0)=toComplex<T>(p.rho*w*w);g(1,3)=-toComplex(k);
    g(2,0)=toComplex(k);g(2,3)=-one/toComplex(p.c44);
    g(3,1)=toComplex<T>(p.c13*k/p.c33);
    g(3,2)=-toComplex<T>((p.c11-p.c13*p.c13/p.c33)*k*k-p.rho*w*w);
    M b=g*toComplex(h);
    double norm=0.;
    for (int i = 0; i < 4; ++i) {
        double row = 0.;
        for (int j = 0; j < 4; ++j) row += ad_cmag1(b(i,j));
        if (!std::isfinite(row))
            throw std::runtime_error("Nonfinite Rayleigh energy matrix");
        norm = std::max(norm, row);
    }
    int steps=0;while(norm>.25) {norm*=.5;++steps;}
    const double scale=std::ldexp(1.,-steps);
    b=b*ad_const(zero,scale);
    const CT dh=toComplex(h)*ad_const(zero,scale);
    M term=identity, prop=identity;
    for(int n=1;n<=24;++n) {term=(term*b/ad_const(zero,n)).eval();prop+=term;}
    std::array<M,10> integrals;
    constexpr int pairs[10][2]={{0,0},{1,1},{2,2},{3,3},{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for(int a=0;a<10;++a) {
        const int i=pairs[a][0],j=pairs[a][1];
        term=M::Constant(zero);term(i,j)=term(j,i)=ad_const(zero,i==j?1.:.5);
        integrals[a]=term*dh;
        for(int n=1;n<=24;++n) {
            term=((b.transpose()*term+term*b)/ad_const(zero,n)).eval();
            integrals[a]+=term*(dh/ad_const(zero,n+1.));
        }
    }
    for(int s=0;s<steps;++s) {
        for(auto& v:integrals)v=(v+prop.transpose()*v*prop).eval();
        prop=(prop*prop).eval();
    }
    result=M::Constant(zero);
    for(int a=0;a<10;++a) {
        CT value=zero;
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)value=value+y[i]*integrals[a](i,j)*y[j];
        result(pairs[a][0],pairs[a][1])=result(pairs[a][1],pairs[a][0])=value;
    }
}

template<class CT>
CT sinhc_tail(const CT& q) {
    const CT zero=q-q, one=ad_const(zero,1.);
    if(ad_cmag1(q)<=1.) {
        double factorial=6.;
        CT sum=ad_const(zero,1./6.), power=one;
        for(int n=1;n<=12;++n) {
            factorial*=double((2*n+2)*(2*n+3));
            power=power*q;
            sum=sum+power/ad_const(zero,factorial);
        }
        return sum;
    }
    return (disper26_math::compute_entire_cs(q).s-one)/q;
}

// Dimensionless integrals of C(q*x²), x*S(q*x²) on x in [0,1].
// Both |q|<=1/4. Degree eight leaves value/first/second errors well below
// binary64 roundoff. q is used directly even when a vertical root is zero.
template<class CT>
void cross_small(const CT& q1,const CT& q3,CT& cc,CT& cs,CT& sc,CT& ss) {
    const CT zero=q1-q1, one=ad_const(zero,1.);
    cc=cs=sc=ss=zero;
    // The q3 recurrence is independent of the outer q1 index. Preserve
    // summation and division order while reusing these identical powers.
    std::array<CT,9> powers;
    powers[0]=one;
    for(int j=1;j<=8;++j) powers[j]=powers[j-1]*q3;
    CT pc=one;
    double fc=1.;
    for(int i=0;i<=8;++i) {
        double fd=1.;
        for(int j=0;j<=8;++j) {
            const CT term=pc*powers[j]/ad_const(zero,fc*fd);
            cc=cc+term/ad_const(zero,2*i+2*j+1.);
            cs=cs+term/ad_const(zero,(2*j+1.)*(2*i+2*j+2.));
            sc=sc+term/ad_const(zero,(2*i+1.)*(2*i+2*j+2.));
            ss=ss+term/ad_const(zero,(2*i+1.)*(2*j+1.)*(2*i+2*j+3.));
            fd*=double((2*j+1)*(2*j+2));
        }
        pc=pc*q1;fc*=double((2*i+1)*(2*i+2));
    }
}

// Existing exponential/C,S cross moments, with the small squared root
// supplied directly so sqrt(0)'s derivative never enters the series.
template<class CT>
void cross_exp(const CT& mu,const CT& n,const CT& h,CT& kc,CT& ks) {
    const CT x=mu*h, q=n*h*h, zero=x-x;
    if(ad_cmag1(q)<AD_KS_SERIES_THRESHOLD*AD_KS_SERIES_THRESHOLD) {
        CT psi[8];ad_psi_chain(x,psi);
        const CT q2=q*q, q3=q2*q;
        kc=h*(psi[0]+q*psi[2]/ad_const(zero,2.)+q2*psi[4]/ad_const(zero,24.)+q3*psi[6]/ad_const(zero,720.));
        ks=h*h*(psi[1]+q*psi[3]/ad_const(zero,6.)+q2*psi[5]/ad_const(zero,120.)+q3*psi[7]/ad_const(zero,5040.));
    } else {
        const CT nu=safe_complex_sqrt(n);
        ad_KCKS(mu,nu,h,kc,ks);
    }
}

template<class CT>
void same_small(const CT& q,const CT& h,CT& cc,CT& cs,CT& ss) {
    const CT zero=q-q, half=ad_const(zero,.5), four=ad_const(zero,4.);
    const auto pair=disper26_math::compute_entire_cs(q);
    cc=h*half*(ad_const(zero,1.)+disper26_math::compute_entire_cs(CT(four*q)).s);
    cs=h*h*half*pair.s*pair.s;
    ss=ad_const(zero,2.)*h*h*h*sinhc_tail(CT(four*q));
}

// Quadratic moments of the liquid state [U,Tzz] over a homogeneous layer,
// integrated over local downward depth from the state at the layer top.
// The coefficient matrix G=[[a,b],[c,-a]] is that of computeLayerMatrixLiquid;
// G^2=delta^2 I, so the state at depth x is C*y0-x*S*(G*y0) with the entire
// C,S of delta^2 x^2. No vertical-wavenumber root is formed.
template<class T>
void fluid_moments(const PhysicalParameters<T>& p,T h,T w,T k,
    const typename ComplexType<T>::type& u,const typename ComplexType<T>::type& t,
    typename ComplexType<T>::type& uu,typename ComplexType<T>::type& ut,
    typename ComplexType<T>::type& tt) {
    using CT=typename ComplexType<T>::type;
    const T gravity=p.gravity*p.gravity_internal_scale;
    const CT kc=toComplex(k), wc=toComplex(w), rho=toComplex(p.rho), g=toComplex(gravity);
    const CT w2=wc*wc, ch=toComplex(h), two=ad_const(CT(ch-ch),2.), one=ad_const(CT(ch-ch),1.);
    const CT phase=wc/kc, n=kc*kc*(one-phase*phase*rho/toComplex(p.c33));
    const CT a=kc*kc*g/w2, b=-n/(rho*w2), c=-rho*w2+kc*kc*rho*g*g/w2;
    CT cc,cs,ss;same_small(CT((a*a+b*c)*ch*ch),ch,cc,cs,ss);
    const CT du=-(a*u+b*t), dt=-(c*u-a*t);
    uu=u*u*cc+two*u*du*cs+du*du*ss;
    ut=u*t*cc+(u*dt+du*t)*cs+du*dt*ss;
    tt=t*t*cc+two*t*dt*cs+dt*dt*ss;
}

template<class T>
bool love_moments(const PhysicalParameters<T>& p,T h,T w,T k,
    const typename ComplexType<T>::type& u,const typename ComplexType<T>::type& traction,
    typename ComplexType<T>::type& u2,typename ComplexType<T>::type& du2) {
    using CT=typename ComplexType<T>::type;
    const CT n=toComplex<T>((p.c66*k*k-p.rho*w*w)/p.c44), ch=toComplex(h);
    const CT q=n*ch*ch, L=toComplex(p.c44), two=ad_const(CT(q-q),2.);
    if(ad_cmag1(q)>.25)return false;
    CT cc,cs,ss;same_small(q,ch,cc,cs,ss);
    // traction is upward T=L*du/dz; moments integrate local downward depth.
    const CT slope=-traction/L;
    u2=u*u*cc+two*u*slope*cs+slope*slope*ss;
    du2=slope*slope*cc+two*u*slope*n*cs+u*u*n*n*ss;
    return true;
}

template<class T>
void rayleigh_moments(const PhysicalParameters<T>& p,T h,T w,T k,
    const std::array<typename ComplexType<T>::type,4>& y,
    Eigen::Matrix<typename ComplexType<T>::type,4,4>& result) {
    using CT=typename ComplexType<T>::type;
    WaveVariables<T> wave;wave.omega=w;wave.k=k;wave.z=make_zero_like(h);
    RayleighEntireValues<T> v;
    if(!computeRayleighEntireValues(p,wave,v)) {
        rayleigh_moments_ode(p,h,w,k,y,result);
        return;
    }
    const CT zero=toComplex(make_zero_like(p.rho)), half=ad_const(zero,.5);
    const CT ch=toComplex(h), L=toComplex(p.c44);
    const CT ld3=L*v.d3, ld4=L*v.d4;
    const CT pc=(ld4*y[0]-y[3])/(ld4-ld3);
    const CT rc=(y[3]-ld3*y[0])/(ld4-ld3);
    const CT det=v.D1*v.E3-v.D2*v.E1;
    const CT qc=(v.E3*y[1]-v.D2*y[2])/det;
    const CT beta=(v.D1*y[2]-v.E1*y[1])/det;
    std::array<CT,4> amp[4]={
        {pc,v.D1*qc,v.E1*qc,ld3*pc},
        {-v.n1*qc,-v.D1*pc,-v.E1*pc,-ld3*v.n1*qc},
        {rc,v.D2*beta,v.E3*beta,ld4*rc},
        {-beta,-v.D2*rc*v.n3,-v.E3*rc*v.n3,-ld4*beta}};
    const CT n[2]={v.n1,v.n3}, q[2]={v.n1*ch*ch,v.n3*ch*ch};
    CT nu[2]={zero,zero}, gram[4][4];
    const bool small[2]={ad_cmag1(q[0])<=.25,ad_cmag1(q[1])<=.25};
    for(int m=0;m<2;++m) {
        const int a=2*m;
        if(small[m]) {
            same_small(q[m],ch,gram[a][a],gram[a][a+1],gram[a+1][a+1]);
            gram[a+1][a]=gram[a][a+1];
        } else {
            nu[m]=safe_complex_sqrt(n[m]);
            for(int r=0;r<4;++r) {
                const CT c=amp[a][r], s=amp[a+1][r]/nu[m];
                amp[a][r]=half*(c-s);amp[a+1][r]=half*(c+s);
            }
            gram[a][a]=ch*ad_phi1(CT(-nu[m]*ch*ad_const(zero,2.)));
            gram[a][a+1]=gram[a+1][a]=ch;
            gram[a+1][a+1]=ch*ad_phi1(CT(nu[m]*ch*ad_const(zero,2.)));
        }
    }
    if(small[0] && small[1]) {
        CT cc,cs,sc,ss;cross_small(q[0],q[1],cc,cs,sc,ss);
        gram[0][2]=ch*cc;gram[0][3]=ch*ch*cs;
        gram[1][2]=ch*ch*sc;gram[1][3]=ch*ch*ch*ss;
    } else if(small[0] || small[1]) {
        const int s=small[0]?0:1, e=1-s;
        for(int j=0;j<2;++j) {
            CT kc,ks;cross_exp(CT(j?nu[e]:-nu[e]),n[s],ch,kc,ks);
            if(s==0) {gram[0][2+j]=kc;gram[1][2+j]=ks;}
            else {gram[j][2]=kc;gram[j][3]=ks;}
        }
    } else {
        for(int i=0;i<2;++i)for(int j=0;j<2;++j)
            gram[i][2+j]=ch*ad_phi1(CT(((i?nu[0]:-nu[0])+(j?nu[1]:-nu[1]))*ch));
    }
    for(int i=0;i<2;++i)for(int j=2;j<4;++j)gram[j][i]=gram[i][j];
    // Compute Gram*amplitude once for all ten physical quadratic moments.
    // This changes association, but keeps the same Gram matrix and moments.
    CT weighted[4][4];
    for(int i=0;i<4;++i) for(int b=0;b<4;++b) {
        CT sum=zero;
        for(int j=0;j<4;++j) sum=sum+gram[i][j]*amp[j][b];
        weighted[i][b]=sum;
    }
    for(int a=0;a<4;++a) for(int b=a;b<4;++b) {
        CT sum=zero;
        for(int i=0;i<4;++i) sum=sum+amp[i][a]*weighted[i][b];
        result(a,b)=result(b,a)=sum;
    }
}

} // namespace disper26_energy
