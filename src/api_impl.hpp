#pragma once
#include "api.hpp"
#include "api_backend.hpp"
#include "dispersion_tracking.hpp"
#include <set>
#include <functional>
#include <numeric>
#include <mutex>

namespace disper26 { namespace api {
struct RootTraceCache {
    std::mutex mutex;
    std::vector<double> roots;
    bool mode_verified = false;
};
struct SolutionData {
    Model model;
    double period = 0, c = 0;
    std::optional<double> u, hv;
    std::string wave;
    int mode = 0;
    Options options;
    Diagnostics diagnostics;
    std::vector<double> roots;
    std::shared_ptr<RootTraceCache> trace_cache = std::make_shared<RootTraceCache>();
};
struct Access {
    static DispersionContextData& context(DispersionContext& context);
    static Model model(ModelData d) { return Model(std::move(d)); }
    static ModeSolution solution(SolutionData d) {
        return ModeSolution(std::make_shared<const SolutionData>(std::move(d)));
    }
    static const SolutionData& solution(const ModeSolution& s) {
        if (!s.data_) throw Error("invalid_argument", "Empty solution");
        return *s.data_;
    }
};
namespace detail {
inline void require(bool ok, const char* message) {
    if (!ok) throw Error("invalid_argument", message);
}
inline void require(bool ok, const std::string& message) {
    if (!ok) throw Error("invalid_argument", message);
}
[[noreturn]] inline void unsupported(const std::string& message) {
    throw Error("unsupported", message);
}
inline void positive(double x, const std::string& name) {
    require(std::isfinite(x) && x > 0, name + " must be finite and positive");
}
inline void check_geometry(const Geometry& g) {
    require(g.kind=="flat" || g.kind=="spherical_ba97" || g.kind=="spherical_experimental_density",
        "geometry kind must be flat, spherical_ba97 or spherical_experimental_density");
    positive(g.planet_radius,"planet_radius");
    require(std::isnan(g.flattening_exponent) || std::isfinite(g.flattening_exponent),
        "flattening_exponent must be finite or unspecified");
}
inline std::string wave_name(const std::string& wave) {
    if (wave == "Rayleigh" || wave == "rayleigh") return "Rayleigh";
    if (wave == "Love" || wave == "love") return "Love";
    throw Error("invalid_argument", "wave must be Rayleigh or Love");
}
inline void check_options(const Options& o) {
    require(o.max_steps >= 0 && o.max_steps <= 12 && o.max_levels >= 0 && o.max_levels <= 12,
            "refinement limits must be between 0 and 12");
    require(o.num_sub_layers > 0, "num_sub_layers must be positive");
    positive(o.tolerance, "tolerance"); positive(o.root_tolerance, "root_tolerance");
    positive(o.derivative_tolerance, "derivative_tolerance");
    if (o.c_guess) positive(*o.c_guess, "c_guess");
    require(o.start == "long" || o.start == "short", "start must be long or short");
    if(o.c_range) {
        positive(o.c_range->first,"c_range lower bound");
        positive(o.c_range->second,"c_range upper bound");
        require(o.c_range->first < o.c_range->second,"c_range lower bound must be smaller than upper bound");
    }
    if(o.verify) require(*o.verify=="none" || *o.verify=="endpoints" || *o.verify=="every",
        "verify must be none, endpoints or every");
    require(!(o.verify && o.verify_mode),"Specify verify or verify_mode, not both");
    if(o.on_violation) require(*o.on_violation=="fallback" || *o.on_violation=="error" || *o.on_violation=="missing",
        "on_violation must be fallback, error or missing");
    require(o.backend == "dual" || o.backend == "vector" || o.backend == "vector_dyn", "Unknown backend");
    require(o.integration == "auto" || o.integration == "analytical" || o.integration == "trapezoidal", "Unknown integration method");
}
inline Diagnostics diagnostic(std::string method, const Options& o) {
    Diagnostics d; d.method = std::move(method);
    d.backend = d.method.rfind("variational_ad",0)==0 ? o.backend : "native";
    return d;
}
inline void finish(Diagnostics& d, bool strict) {
    for(const auto& entry:d.availability) if(!d.convergence_by_quantity.count(entry.first))
        d.convergence_by_quantity[entry.first]=entry.second=="computed"?d.convergence:entry.second;
    if (d.convergence == "not_converged") {
        d.status = "warning";
        d.messages.push_back("Refinement limit reached before requested tolerance");
    }
    if (strict && d.convergence != "converged")
        throw Error("not_converged", "Requested accuracy was not established: " + d.convergence);
}
inline PhysicalParameters<double> unpack(const Material& m) {
    PhysicalParameters<double> p;
    p.rho=m.rho; p.c11=m.A; p.c33=m.C; p.c13=m.F; p.c44=m.L; p.c66=m.N;
    p.gravity=m.gravity; p.gravity_internal_scale=m.gravity_internal_scale;
    p.gravity_surface_scale=m.gravity_surface_scale;
    return p;
}
inline Material pack(const PhysicalParameters<double>& p) {
    Material m; m.rho=p.rho; m.A=p.c11; m.C=p.c33; m.F=p.c13; m.L=p.c44; m.N=p.c66;
    m.gravity=p.gravity; m.gravity_internal_scale=p.gravity_internal_scale;
    m.gravity_surface_scale=p.gravity_surface_scale;
    return m;
}
inline std::vector<PhysicalParameters<double>> unpack(const std::vector<Material>& ms) {
    std::vector<PhysicalParameters<double>> ps; for (const auto& m:ms) ps.push_back(unpack(m)); return ps;
}
inline bool fluid(const Material& p) { return isLiquidLayer(unpack(p)); }
inline bool has_fluid(const Model& m) {
    const auto& d=m.data();
    return std::any_of(d.materials.begin(),d.materials.end(),fluid) || fluid(d.halfspace);
}
inline void check_material(const Material& p) {
    for (double x:{p.rho,p.A,p.C,p.F,p.L,p.N,p.gravity,p.gravity_internal_scale,p.gravity_surface_scale})
        require(std::isfinite(x), "Material fields must be finite");
    positive(p.rho,"density"); positive(p.A,"A"); positive(p.C,"C");
    require(p.L >= 0 && p.N >= 0, "Shear moduli must be nonnegative");
    require((p.L == 0) == (p.N == 0), "Fluid requires both shear moduli to be zero");
}
inline ::GridModel grid(const Model& m) {
    const auto& d=m.data(); ::GridModel g;
    g.depths=d.depths; g.params=unpack(d.materials); g.is_discontinuity=d.discontinuities;
    g.refinement_policy=make_grid_refinement_policy(d.geometry.kind,d.geometry.planet_radius,d.geometry.flattening_exponent);
    return g;
}
struct VelocityFields { std::vector<double> rho,ph,pv,sh,sv,eta; };
inline VelocityFields velocities(const Model& m) {
    VelocityFields f;
    for (const auto& p:m.data().materials) {
        require(p.A-2*p.L != 0, "Velocity-VTI representation is undefined when A - 2 L = 0");
        f.rho.push_back(p.rho); f.ph.push_back(std::sqrt(p.A/p.rho)); f.pv.push_back(std::sqrt(p.C/p.rho));
        f.sh.push_back(std::sqrt(p.N/p.rho)); f.sv.push_back(std::sqrt(p.L/p.rho)); f.eta.push_back(p.F/(p.A-2*p.L));
    }
    return f;
}
inline VelocityGridRefinement mesh(const Model& model, int level) {
    const auto& d=model.data();
    if (d.kind == "pancake") {
        if (d.geometry.kind != "flat") unsupported("Pancake spherical input needs explicit earth_flattening_transformation");
        VelocityGridRefinement out; out.layers=unpack(d.materials); out.thicknesses=d.thicknesses; out.halfspace=unpack(d.halfspace); return out;
    }
    if (d.interpolation == "stiffness") {
        Mesher mesher(grid(model)); VelocityGridRefinement out;
        std::tie(out.layers,out.thicknesses)=mesher.get_refined_layers(level);
        out.halfspace=mesher.get_halfspace(); return out;
    }
    const auto v=velocities(model);
    return refine_velocity_grid_vti(d.depths,v.rho,v.ph,v.pv,v.sh,v.sv,v.eta,d.discontinuities,level,
        make_grid_refinement_policy(d.geometry.kind,d.geometry.planet_radius,d.geometry.flattening_exponent));
}
inline std::set<std::string> outputs(const std::vector<std::string>& out) {
    require(!out.empty(), "outputs must not be empty"); std::set<std::string> o;
    for (const auto& s:out) {
        require(s=="phase_velocity" || s=="group_velocity" || s=="hv_ratio", "Unknown output: "+s);
        require(o.insert(s).second, "Duplicate output: "+s);
    }
    return o;
}
inline void ensure_mode_verified(const ModeSolution& s) {
    const auto& a=Access::solution(s);
    if(!std::isfinite(a.c)) throw Error("missing_solution","Cannot compute from a missing dispersion solution");
    if(a.diagnostics.mode_verified) return;
    auto& cache=*a.trace_cache;
    std::lock_guard<std::mutex> lock(cache.mutex);
    if(cache.mode_verified) return;
    std::vector<double> prefix;
    for(size_t level=0;level<a.roots.size();++level) {
        const auto refined=mesh(a.model,static_cast<int>(level));
        const double stored=a.roots[level];
        const double checked=find_dispersion_root_by_mode_order(refined,2*M_PI/a.period,
            parseWaveType(a.wave),a.mode,a.options.root_tolerance,stored,prefix);
        const double tolerance=std::max(8*a.options.root_tolerance,
            64*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(stored)));
        if(!std::isfinite(checked) || std::abs(checked-stored)>tolerance)
            throw Error("mode_mismatch","Tracked root does not match the requested mode; saved solution was not changed");
        prefix.push_back(stored);
    }
    cache.mode_verified=true;
}
inline void derivative_wave(const ModeSolution& s) {
    ensure_mode_verified(s);
    if (s.wave()=="Love" && has_fluid(s.model())) unsupported("Love derivatives with fluid media are not supported");
    if(s.wave()=="Rayleigh") {
        const auto& d=s.model().data();
        if(d.halfspace.gravity!=0 || std::any_of(d.materials.begin(),d.materials.end(),
                [](const Material& p){return p.gravity!=0;}))
            unsupported("Material and interface derivatives with Cowling gravity are not validated; use a non-gravitating reference solution and the separate Kg correction");
    }
}
inline std::vector<double> trace_roots(const ModeSolution& s, int level) {
    ensure_mode_verified(s);
    const auto& a=Access::solution(s);
    if (a.model.data().kind == "pancake") return {a.c};
    require(level >= 0 && level <= 12, "Trace refinement level must be between 0 and 12");
    auto& cache = *a.trace_cache;
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.roots.empty()) cache.roots = a.roots;
    require(!cache.roots.empty(), "The solution has no GRID root trace");
    const auto type = parseWaveType(a.wave);
    const double omega = 2.0 * M_PI / a.period;
    // The accepted prefix is immutable. Derivative requests append only new
    // mesh roots, sharing them across all handles/copies of this solution.
    while (static_cast<int>(cache.roots.size()) <= level) {
        const auto refined = mesh(a.model, static_cast<int>(cache.roots.size()));
        const double root = find_dispersion_root_by_mode_order(refined, omega,
            type, a.mode, a.options.root_tolerance, cache.roots.back(), cache.roots);
        const double residual = std::abs(calculateDispersionPhaseValueDouble(
            refined.layers, refined.thicknesses, refined.halfspace, omega, root, type));
        if (!std::isfinite(root) || root <= 0.0 || !std::isfinite(residual) ||
            residual > kAdaptiveResidualLimit)
            throw Error("numerical_failure", "Trace extension produced an invalid GRID root");
        cache.roots.push_back(root);
    }
    return cache.roots;
}
inline std::vector<Material> parameter_materials(const Model& model) {
    auto p=model.data().materials;
    if (model.data().kind=="pancake") p.push_back(model.data().halfspace);
    return p;
}
inline std::string entity(const Model& m,int i) {
    return m.data().kind=="grid" ? "node" : (i==static_cast<int>(m.data().materials.size())?"halfspace":"layer");
}
inline bool analytical(const Model& m,const Options& o) {
    if(o.integration=="analytical" && has_fluid(m)) unsupported("Analytical integrals require solid media");
    return o.integration=="analytical" || (o.integration=="auto" && !has_fluid(m));
}
inline std::string unit(const std::string& name) {
    if (name=="Rho" || name=="rho") return "g/cm^3";
    if (name=="eta") return "1";
    if (name=="Depth" || name=="Thickness") return "km";
    if (name.size() && (name[0]=='V' || name[0]=='v')) return "km/s";
    return "GPa";
}
inline const std::vector<std::string>& stiffness_names() {
    static const std::vector<std::string> n={"A","C","F","L","N","Rho"}; return n;
}
inline std::vector<Parameter> axes(const Model& m,const std::string& basis,bool fixed_fluid=false) {
    std::vector<std::string> n;
    if(basis=="stiffness") n=stiffness_names();
    else if(basis=="isotropic") n={"Vs","Vp","Rho"};
    else if(basis=="velocity_vti" || basis=="log_velocity_vti") n={"rho","vph","vpv","vsh","vsv","eta"};
    else throw Error("invalid_argument","Unknown derivative basis: "+basis);
    std::vector<Parameter> out; const auto p=parameter_materials(m);
    for(int i=0;i<static_cast<int>(p.size());++i) for(const auto& name:n)
        out.push_back({name,entity(m,i),basis=="log_velocity_vti"?"1":unit(name),
            entity(m,i)=="halfspace"?0:i,!(fixed_fluid&&fluid(p[i]))});
    return out;
}
// Exact material-coordinate map. Stiffness output order A,C,F,L,N,Rho.
struct BasisMap {Eigen::MatrixXd J; std::vector<Eigen::MatrixXd> K;};
inline BasisMap basis_map(const Material& p,const std::string& basis) {
    if(basis=="stiffness") return {Eigen::MatrixXd::Identity(6,6),std::vector<Eigen::MatrixXd>(6,Eigen::MatrixXd::Zero(6,6))};
    require(p.A-2*p.L != 0,"Velocity basis is undefined when A-2L=0");
    const double r=p.rho,ph=std::sqrt(p.A/r),pv=std::sqrt(p.C/r),sh=std::sqrt(p.N/r),sv=std::sqrt(p.L/r),e=p.F/(p.A-2*p.L);
    BasisMap b{Eigen::MatrixXd::Zero(6,6),std::vector<Eigen::MatrixXd>(6,Eigen::MatrixXd::Zero(6,6))};
    auto quadratic=[&](int row,int col,double v) {
        b.J(row,0)=v*v; b.J(row,col)=2*r*v;
        b.K[row](0,col)=b.K[row](col,0)=2*v; b.K[row](col,col)=2*r;
    };
    quadratic(0,1,ph); quadratic(1,2,pv); quadratic(3,4,sv); quadratic(4,3,sh);
    b.J.row(2)=e*(b.J.row(0)-2*b.J.row(3)); b.J(2,5)=p.A-2*p.L;
    b.K[2]=e*(b.K[0]-2*b.K[3]);
    for(int j=0;j<5;++j) b.K[2](j,5)=b.K[2](5,j)=b.J(0,j)-2*b.J(3,j);
    b.J(5,0)=1;
    if(basis=="isotropic") {
        // Check using the same roundoff scale as a material constructed via rho*v^2.
        const double scale=std::max({1.0,std::abs(p.A),std::abs(p.C),std::abs(p.F),std::abs(p.L),std::abs(p.N)});
        const double eps=64*std::numeric_limits<double>::epsilon()*scale;
        require(std::abs(p.A-p.C)<=eps && std::abs(p.L-p.N)<=eps && std::abs(p.F-(p.A-2*p.L))<=eps,
            "isotropic basis requires an isotropic model; use velocity_vti or stiffness");
        Eigen::MatrixXd P=Eigen::MatrixXd::Zero(6,3); P(0,2)=1; P(1,1)=P(2,1)=1; P(3,0)=P(4,0)=1;
        b.J=(b.J*P).eval(); for(auto& k:b.K) k=(P.transpose()*k*P).eval();
    } else if(basis=="log_velocity_vti") {
        Eigen::VectorXd x(6); x<<r,ph,pv,sh,sv,e;
        require((x.array()>0).all(),"log_velocity_vti requires strictly positive solid velocity coordinates");
        for(int k=0;k<6;++k) {
            b.K[k]=(x.asDiagonal()*b.K[k]*x.asDiagonal()).eval();
            b.K[k].diagonal().array()+=b.J.row(k).transpose().array()*x.array();
        }
        b.J=(b.J*x.asDiagonal()).eval();
    } else require(basis=="velocity_vti","Unknown derivative basis");
    return b;
}
inline void check_basis(const Model& model,const std::string& basis) {
    (void)axes(model,basis);
    for(const auto& p:parameter_materials(model))
        if(model.data().kind!="grid" || !fluid(p)) (void)basis_map(p,basis);
}
inline void transform(DerivativeResult& r,const Model& model,const std::string& basis,bool fixed_fluid=false) {
    const auto ps=parameter_materials(model); const int nn=static_cast<int>(ps.size());
    const int q=basis=="isotropic"?3:6;
    Eigen::MatrixXd J=Eigen::MatrixXd::Zero(nn*6,nn*q);
    std::vector<BasisMap> maps;
    for(int i=0;i<nn;++i) {
        if(fixed_fluid && fluid(ps[i])) maps.push_back({Eigen::MatrixXd::Zero(6,q),std::vector<Eigen::MatrixXd>(6,Eigen::MatrixXd::Zero(q,q))});
        else maps.push_back(basis_map(ps[i],basis));
        J.block(i*6,i*q,6,q)=maps.back().J;
    }
    if(!r.hessian.empty()) {
        const auto& gv=r.gradients.at("phase_velocity");
        Eigen::Map<const Eigen::MatrixXd> H(r.hessian.data(),nn*6,nn*6);
        Eigen::MatrixXd target=J.transpose()*H*J;
        for(int i=0;i<nn;++i) for(int k=0;k<6;++k) target.block(i*q,i*q,q,q)+=gv[i*6+k]*maps[i].K[k];
        r.hessian.assign(target.data(),target.data()+target.size()); r.rows=r.cols=nn*q;
    }
    for(auto& kv:r.gradients) {
        Eigen::Map<const Eigen::VectorXd> g(kv.second.data(),nn*6);
        Eigen::VectorXd v=J.transpose()*g; kv.second.assign(v.data(),v.data()+v.size());
    }
    r.parameters=axes(model,basis,fixed_fluid); r.basis=basis;
}
inline DerivativeResult from_hessian(const HessianResult& h,const Model& model) {
    DerivativeResult out; const int n=static_cast<int>(parameter_materials(model).size())*6;
    std::vector<int> ix; ix.reserve(n);
    for(int i=0;i<n/6;++i) for(const auto& name:stiffness_names()) {
        const int source_index=model.data().kind=="pancake" && i==n/6-1 ? -1 : i;
        int found=-1;
        for(int j=0;j<static_cast<int>(h.parameterNames.size());++j)
            if(h.parameterNames[j]==name && h.layerIndices[j]==source_index) {found=j;break;}
        require(found>=0,"Incomplete backend parameter metadata"); ix.push_back(found);
    }
    auto& g=out.gradients["phase_velocity"]; g.resize(n); out.hessian.resize(n*n);
    for(int i=0;i<n;++i) {g[i]=h.gradient(ix[i]); for(int j=0;j<n;++j) out.hessian[i+j*n]=h.hessian(ix[i],ix[j]);}
    out.rows=out.cols=n; return out;
}
inline void ensure_finite(const DerivativeResult& r) {
    for(const auto& g:r.gradients) for(double x:g.second)
        if(!std::isfinite(x)) throw Error("numerical_failure","Nonfinite derivative result: "+g.first);
    for(double x:r.hessian) if(!std::isfinite(x)) throw Error("numerical_failure","Nonfinite Hessian result");
}
} // detail

struct DispersionContextData {
    std::mutex mutex;
    std::array<VelocityGridRefinement,13> meshes;
    std::array<bool,13> ready{};
    std::vector<GridSublayerGeometry> geometry;
    std::vector<bool> fluids;
    detail::VelocityFields fields;
    RombergWorkspace romberg;

    void begin_call() {
        ready.fill(false);
        for(auto& mesh:meshes) {mesh.layers.clear();mesh.thicknesses.clear();}
        geometry.clear(); fluids.clear();
        fields.rho.clear(); fields.ph.clear(); fields.pv.clear();
        fields.sh.clear(); fields.sv.clear(); fields.eta.clear();
        romberg.c.clear(); romberg.u.clear(); romberg.hv.clear(); romberg.roots.clear();
    }
    const VelocityGridRefinement& mesh(const Model& model,int level) {
        auto& out=meshes.at(static_cast<size_t>(level));
        if(ready.at(level)) return out;
        const auto& d=model.data();
        if(d.kind=="pancake") {
            if(d.geometry.kind!="flat") detail::unsupported("Pancake spherical input needs explicit earth_flattening_transformation");
            out.layers.reserve(d.materials.size());
            for(const auto& p:d.materials) out.layers.push_back(detail::unpack(p));
            out.thicknesses.assign(d.thicknesses.begin(),d.thicknesses.end());
            out.halfspace=detail::unpack(d.halfspace);
        } else if(d.interpolation=="stiffness") {
            Mesher mesher(detail::grid(model));
            mesher.fill_refined_layers(level,out.layers,out.thicknesses,geometry,fluids);
            out.halfspace=mesher.get_halfspace();
        } else {
            if(fields.rho.empty()) {
                for(const auto& p:d.materials) {
                    detail::require(p.A-2*p.L!=0,"Velocity-VTI representation is undefined when A - 2 L = 0");
                    fields.rho.push_back(p.rho); fields.ph.push_back(std::sqrt(p.A/p.rho));
                    fields.pv.push_back(std::sqrt(p.C/p.rho)); fields.sh.push_back(std::sqrt(p.N/p.rho));
                    fields.sv.push_back(std::sqrt(p.L/p.rho)); fields.eta.push_back(p.F/(p.A-2*p.L));
                }
            }
            fill_velocity_grid_vti_values(d.depths,fields.rho,fields.ph,fields.pv,fields.sh,fields.sv,
                fields.eta,d.discontinuities,level,
                make_grid_refinement_policy(d.geometry.kind,d.geometry.planet_radius,d.geometry.flattening_exponent),
                out,geometry,fluids);
        }
        ready[level]=true;
        return out;
    }
};
DispersionContext::DispersionContext():data_(std::make_unique<DispersionContextData>()) {}
DispersionContext::~DispersionContext()=default;
DispersionContext::DispersionContext(DispersionContext&&) noexcept=default;
DispersionContext& DispersionContext::operator=(DispersionContext&&) noexcept=default;
DispersionContext create_dispersion_context() {return DispersionContext{};}
DispersionContextData& Access::context(DispersionContext& context) {
    detail::require(bool(context.data_),"Empty dispersion context");
    return *context.data_;
}

Model::Model(ModelData data):data_(std::make_shared<const ModelData>(std::move(data))) {}
const ModelData& Model::data() const {detail::require(bool(data_),"Empty model"); return *data_;}
ModeSolution::ModeSolution(std::shared_ptr<const SolutionData> d):data_(std::move(d)) {}
double ModeSolution::phase_velocity() const {return Access::solution(*this).c;}
std::optional<double> ModeSolution::group_velocity() const {return Access::solution(*this).u;}
std::optional<double> ModeSolution::hv_ratio() const {return Access::solution(*this).hv;}
double ModeSolution::period() const {return Access::solution(*this).period;}
const std::string& ModeSolution::wave() const {return Access::solution(*this).wave;}
int ModeSolution::mode() const {return Access::solution(*this).mode;}
const Model& ModeSolution::model() const {return Access::solution(*this).model;}
const Options& ModeSolution::options() const {return Access::solution(*this).options;}
const Diagnostics& ModeSolution::diagnostics() const {return Access::solution(*this).diagnostics;}
Material create_isotropic_parameters(double vs,double vp,double rho) {
    detail::positive(vp,"vp"); detail::positive(rho,"rho"); detail::require(std::isfinite(vs)&&vs>=0,"vs must be finite and nonnegative");
    return detail::pack(::createIsotropicParameters<double>(vs,vp,rho));
}
Material create_fluid_parameters(double vp,double rho) {return create_isotropic_parameters(0,vp,rho);}
Material create_velocity_vti_parameters(double rho,double ph,double pv,double sh,double sv,double eta) {
    detail::positive(rho,"rho"); detail::positive(ph,"vph"); detail::positive(pv,"vpv");
    detail::require(std::isfinite(sh)&&sh>=0&&std::isfinite(sv)&&sv>=0&&std::isfinite(eta),"Invalid VTI coordinates");
    Material p; p.rho=rho;p.A=rho*ph*ph;p.C=rho*pv*pv;p.L=rho*sv*sv;p.N=rho*sh*sh;p.F=eta*(p.A-2*p.L);
    detail::check_material(p); return p;
}
Model create_pancake_model(const std::vector<Material>& layers,const std::vector<double>& h,const Material& hs,const Geometry& geo,const std::string& name) {
    detail::check_geometry(geo);
    detail::require(layers.size()==h.size(),"layers and thicknesses differ in length");
    detail::check_material(hs); for(const auto& p:layers) detail::check_material(p);
    for(double v:h) detail::positive(v,"thickness");
    (void)make_grid_refinement_policy(geo.kind,geo.planet_radius,geo.flattening_exponent);
    if(geo.kind!="flat") detail::require(std::accumulate(h.begin(),h.end(),0.0)<geo.planet_radius,
        "Pancake extends to planet centre");
    ModelData d;d.kind="pancake";d.interpolation="constant";d.materials=layers;d.thicknesses=h;d.halfspace=hs;d.geometry=geo;d.name=name;
    return Access::model(std::move(d));
}
Model create_grid_model(const std::vector<double>& z,const std::vector<Material>& ps,const std::string& interpolation,const Geometry& geo,const std::string& name) {
    detail::require(z.size()==ps.size() && z.size()>=2,"Grid needs equally sized depth/material arrays with at least two nodes");
    detail::require(interpolation=="stiffness" || interpolation=="velocity","Specify interpolation='stiffness' or 'velocity'");
    std::vector<bool> flags(z.size(),false);
    for(size_t i=0;i<z.size();++i) {
        detail::require(std::isfinite(z[i])&&z[i]>=0,"Depths must be finite and nonnegative"); detail::check_material(ps[i]);
        if(i) {detail::require(z[i]>=z[i-1],"Depths must be nondecreasing"); if(z[i]==z[i-1]) {
            detail::require(i<2 || z[i-2]!=z[i],"An interface must have exactly two nodes"); flags[i-1]=true;
        }}
    }
    detail::require(z.front()==0,"Grid surface depth must be zero");
    detail::check_geometry(geo);
    (void)make_grid_refinement_policy(geo.kind,geo.planet_radius,geo.flattening_exponent);
    if(geo.kind!="flat") detail::require(z.back()<geo.planet_radius,"Grid extends to planet centre");
    ModelData d;d.kind="grid";d.interpolation=interpolation;d.materials=ps;d.depths=z;d.discontinuities=flags;d.halfspace=ps.back();d.geometry=geo;d.name=name;
    auto model=Access::model(std::move(d));
    if(interpolation=="velocity") (void)detail::velocities(model);
    return model;
}

Model create_model(const std::vector<double>& depths,const std::string& kind,
    const MaterialArrays& p,const std::optional<std::string>& interpolation,
    const Geometry& geometry,const std::string& name) {
    detail::require(kind=="pancake" || kind=="grid","kind must be pancake or grid");
    detail::require(!depths.empty(),"depths must not be empty");
    detail::require(depths.front()==0,"depths must start at zero");
    for(size_t i=0;i<depths.size();++i) {
        detail::require(std::isfinite(depths[i]) && depths[i]>=0,"depths must be finite and nonnegative");
        if(i) detail::require(kind=="pancake"?depths[i]>depths[i-1]:depths[i]>=depths[i-1],
            kind=="pancake"?"Pancake depths must strictly increase; duplicate knots are forbidden":"GRID depths must be nondecreasing");
    }
    if(kind=="pancake") detail::require(!interpolation,"Pancake does not accept interpolation");
    else detail::require(interpolation && (*interpolation=="stiffness" || *interpolation=="velocity"),
        "GRID requires interpolation=stiffness or velocity");
    const bool iso=p.vs || p.vp;
    const bool velocity=p.vph || p.vpv || p.vsh || p.vsv || p.eta;
    const bool stiffness=p.A || p.C || p.F || p.L || p.N;
    detail::require(int(iso)+int(velocity)+int(stiffness)==1,
        "Supply exactly one material representation: isotropic, velocity_vti or stiffness");
    auto check=[&](const std::optional<std::vector<double>>& values,const char* field) {
        detail::require(values.has_value(),std::string("Missing material array: ")+field);
        detail::require(values->size()==depths.size(),std::string(field)+" must match depths length");
        for(double x:*values) detail::require(std::isfinite(x),std::string(field)+" must contain finite values");
    };
    check(p.rho,"rho");
    if(iso) {check(p.vs,"vs");check(p.vp,"vp");}
    if(velocity) {check(p.vph,"vph");check(p.vpv,"vpv");check(p.vsh,"vsh");check(p.vsv,"vsv");check(p.eta,"eta");}
    if(stiffness) {check(p.A,"A");check(p.C,"C");check(p.F,"F");check(p.L,"L");check(p.N,"N");}
    std::vector<Material> materials;
    for(size_t i=0;i<depths.size();++i) {
        Material m;
        if(iso) m=create_isotropic_parameters((*p.vs)[i],(*p.vp)[i],(*p.rho)[i]);
        else if(velocity) m=create_velocity_vti_parameters((*p.rho)[i],(*p.vph)[i],(*p.vpv)[i],(*p.vsh)[i],(*p.vsv)[i],(*p.eta)[i]);
        else m={(*p.rho)[i],(*p.A)[i],(*p.C)[i],(*p.F)[i],(*p.L)[i],(*p.N)[i]};
        detail::check_material(m);materials.push_back(m);
    }
    if(kind=="grid") return create_grid_model(depths,materials,*interpolation,geometry,name);
    std::vector<double> thicknesses;
    for(size_t i=1;i<depths.size();++i) thicknesses.push_back(depths[i]-depths[i-1]);
    const auto halfspace=materials.back();materials.pop_back();
    return create_pancake_model(materials,thicknesses,halfspace,geometry,name);
}

namespace detail {
inline ModeSolution missing_solution(SolutionData d, const std::string& reason,
        const std::string& message) {
    if(d.options.strict) throw Error(reason,message);
    d.c=missing_value;d.roots.clear();d.trace_cache=std::make_shared<RootTraceCache>();
    for(auto& entry:d.diagnostics.availability) if(entry.second=="computed") {
        entry.second="missing";
        if(entry.first=="group_velocity") d.u=missing_value;
        if(entry.first=="hv_ratio") d.hv=missing_value;
    }
    d.diagnostics.status="missing";d.diagnostics.convergence="not_evaluated";
    d.diagnostics.missing_reason=reason;d.diagnostics.messages.push_back(message);
    d.diagnostics.mode_verified=false;
    std::fill(d.diagnostics.level_mode_verified.begin(),d.diagnostics.level_mode_verified.end(),false);
    d.diagnostics.accepted_level=-1;d.diagnostics.generated_level=-1;
    d.diagnostics.relative_error=missing_value;
    for(const auto& entry:d.diagnostics.availability)
        d.diagnostics.convergence_by_quantity[entry.first]=entry.second;
    return Access::solution(std::move(d));
}

template<class Refiner>
ModeSolution compute_dispersion_in_context(const Model& model,double period,const std::string& wave,int mode,const std::vector<std::string>& requested,const Options& o,Refiner& refiner,DispersionTracking& tracking,RombergWorkspace* workspace=nullptr) {
    detail::positive(period,"period");detail::require(mode>=0,"mode must be nonnegative");detail::check_options(o);
    if(model.data().q_period) detail::require(period==*model.data().q_period,
        "Q-corrected models can only be evaluated at their correction period");
    detail::require(o.method=="standard" || o.method=="mode_order" || o.method=="continuation" || o.method=="continuation_recover","Dispersion method must be standard, mode_order, continuation or continuation_recover");
    detail::require(o.predictor=="auto" || o.predictor=="history" || o.predictor=="group_velocity","Unknown dispersion predictor");
    detail::require(o.verify || o.method=="continuation" || o.method=="continuation_recover" || o.verify_mode.value_or(true),"mode_order requires mode verification");
    const auto out=detail::outputs(requested); const auto w=detail::wave_name(wave); const auto type=parseWaveType(w);
    if(detail::fluid(model.data().halfspace)) detail::unsupported("Mode counting requires a solid halfspace");
    if(w=="Love"&&detail::has_fluid(model)) detail::unsupported("Love mode counting with fluids is not supported");
    SolutionData d;d.model=model;d.period=period;d.wave=w;d.mode=mode;d.options=o;
    d.diagnostics=detail::diagnostic(o.method=="standard"?"mode_order":o.method,o);
    const size_t root_levels = model.data().kind=="grid" ? o.max_steps+1 : 1;
    d.diagnostics.level_mode_verified.reserve(root_levels);
    d.diagnostics.predictors.reserve(root_levels);
    for(const auto& name:{"phase_velocity","group_velocity","hv_ratio"}) d.diagnostics.availability[name]=out.count(name)?"computed":"not_requested";
    d.diagnostics.availability["phase_velocity"]="computed";
    if(w=="Love") d.diagnostics.availability["hv_ratio"]="not_applicable";
    const auto& base=refiner(0);
    double guess=o.c_guess.value_or(0.9*std::sqrt((type==WaveType::Love?base.halfspace.c66:base.halfspace.c44)/base.halfspace.rho));
    detail::positive(guess,"initial phase velocity");
    size_t current_level=0;
    double current_u=missing_value;
    auto find_root=[&](const VelocityGridRefinement& stack,double omega,double initial,const std::vector<double>& within) {
        current_level=within.size();
        const double root=tracking.root(stack,period,type,mode,o,initial,within,d.diagnostics);
        current_u=missing_value;
        // A first-point tangent pays for itself on coarse levels. Do not
        // compute unused tangents for scalar calls, duplicate-only curves,
        // large frequency jumps, or when the requested U will supply it.
        const bool startup = o.predictor=="auto" &&
            !out.count("group_velocity") &&
            current_level < kPhaseVelocityCurvePredictedGridLevels &&
            // Checked auto uses the current coarse root at finer levels;
            // only level 0 can consume a cross-period tangent there.
            (verification_policy(o)!="every" || current_level==0) &&
            tracking.histories[current_level].roots.size()==1 &&
            std::isfinite(tracking.next_period) &&
            std::abs(period/tracking.next_period-1.0)<=0.25;
        if(o.predictor=="group_velocity" || startup) {
            current_u=calculateGroupVelocityFullAD(stack.layers,stack.thicknesses,stack.halfspace,omega,root,type);
            ++d.diagnostics.predictor_ad_evaluations;
            tracking.group_velocity(current_level,omega,root,current_u);
        }
        return root;
    };
    auto group=[&](const VelocityGridRefinement& stack,double omega,double c) {
        if(!out.count("group_velocity")) return missing_value;
        const double u=std::isfinite(current_u)?current_u:
            calculateGroupVelocityFullAD(stack.layers,stack.thicknesses,stack.halfspace,omega,c,type);
        tracking.group_velocity(current_level,omega,c,u);
        return u;
    };
    auto hv=[&](const VelocityGridRefinement& stack,double omega,double c) {
        if(!out.count("hv_ratio") || type==WaveType::Love) return missing_value;
        return py_calculateHVRatio(stack.layers,stack.thicknesses,stack.halfspace,omega,c,w);
    };
    try {
    if(model.data().kind=="grid") {
        auto r=compute_with_romberg_core(refiner,period,guess,find_root,group,hv,o.max_steps,o.tolerance,
            !out.count("group_velocity")&&!out.count("hv_ratio"),0,0,workspace);
        d.c=r.phase_velocity;d.roots=std::move(r.root_trace.phase_velocity_roots);
        if(out.count("group_velocity")) d.u=r.group_velocity;
        if(out.count("hv_ratio")&&w=="Rayleigh") d.hv=r.hv_ratio;
        d.diagnostics.accepted_level=r.converged_level;d.diagnostics.generated_level=static_cast<int>(d.roots.size())-1;
        d.diagnostics.relative_error=r.relative_error;
        d.diagnostics.convergence=std::isfinite(r.relative_error)?(r.relative_error<=o.tolerance?"converged":"not_converged"):"not_evaluated";
    } else {
        d.c=find_root(base,2*M_PI/period,guess,{});
        if(out.count("group_velocity")) d.u=group(base,2*M_PI/period,d.c);
        if(out.count("hv_ratio")&&w=="Rayleigh") d.hv=hv(base,2*M_PI/period,d.c);
        d.roots={d.c};d.diagnostics.convergence="converged";
    }
    } catch(const ModeOrderRootNotFound& e) {
        return missing_solution(std::move(d),"mode_not_found",e.what());
    } catch(const Error& e) {
        if(e.code=="tracking_violation" && o.on_violation.value_or("fallback")=="missing")
            return missing_solution(std::move(d),e.code,e.what());
        throw;
    }
    d.diagnostics.mode_verified=std::all_of(d.diagnostics.level_mode_verified.begin(),d.diagnostics.level_mode_verified.end(),[](bool x){return x;});
    if(!std::isfinite(d.c)||(d.u&&!std::isfinite(*d.u))||(d.hv&&!std::isfinite(*d.hv))) throw Error("numerical_failure","Nonfinite dispersion value");
    if(d.c>=tracking_speed_cap(base,type)) throw Error("root_not_found","Dispersion value is outside the trapped-wave speed cap");
    d.diagnostics.convergence_by_quantity["phase_velocity"]=d.diagnostics.convergence;
    for(const auto& name:{"group_velocity","hv_ratio"})
        d.diagnostics.convergence_by_quantity[name]=d.diagnostics.availability[name]=="computed"?
            "not_evaluated":d.diagnostics.availability[name];
    if(d.u || d.hv) {
        d.diagnostics.messages.push_back("Root and Romberg error estimates certify phase velocity only; group velocity and H/V accuracy are not independently estimated");
        if(d.diagnostics.convergence=="converged") d.diagnostics.convergence="not_evaluated";
    }
    detail::finish(d.diagnostics,o.strict); return Access::solution(std::move(d));
}

template<class Refiner>
void finish_tracking_segment(const Model& model,const std::string& wave,int mode,
        const std::vector<std::string>& outputs,const Options& options,Refiner& refiner,
        RombergWorkspace* workspace,std::map<double,ModeSolution>& cache,
        const std::vector<double>& segment) {
    const auto type=parseWaveType(wave_name(wave));
    if(verification_policy(options)!="endpoints" || segment.empty()) return;
    auto end=Access::solution(cache.at(segment.back()));
    if(end.diagnostics.mode_verified) return;
    bool verified=true;
    for(size_t level=0;level<end.roots.size();++level) {
        if(end.diagnostics.level_mode_verified.at(level)) continue;
        bool checked=false;
        try {
            checked=certify_tracking_root(refiner(static_cast<int>(level)),
                2*M_PI/end.period,type,mode,end.roots[level],options.root_tolerance,
                end.diagnostics.mode_count_evaluations);
        } catch(const std::runtime_error&) {
            // The selected violation policy handles a failed certificate.
        }
        end.diagnostics.level_mode_verified[level]=checked;
        if(!checked) {verified=false;break;}
    }
    end.diagnostics.mode_verified=verified;
    cache.at(segment.back())=Access::solution(std::move(end));
    if(verified) return;
    const auto response=options.on_violation.value_or("fallback");
    if(response=="error" || (options.strict && response=="missing"))
        throw Error("tracking_violation","Endpoint mode certificate failed");
    size_t begin=0;
    for(size_t i=0;i+1<segment.size();++i)
        if(cache.at(segment[i]).diagnostics().mode_verified) begin=i+1;
    DispersionTracking repair_tracking;
    repair_tracking.initial_period=false;
    Options repair=options;
    repair.verify="every";repair.verify_mode.reset();repair.c_range.reset();
    repair.on_violation="fallback";
    if(begin>0) repair.c_guess=cache.at(segment[begin-1]).phase_velocity();
    for(size_t i=begin;i<segment.size();++i) {
        const double period=segment[i];
        const auto original=Access::solution(cache.at(period));
        if(response=="missing") {
            cache.at(period)=missing_solution(original,"tracking_violation",
                "Endpoint mode certificate failed; unchecked segment discarded");
            continue;
        }
        repair_tracking.next_period=i+1<segment.size()?segment[i+1]:missing_value;
        auto corrected=compute_dispersion_in_context(model,period,wave,mode,outputs,
            repair,refiner,repair_tracking,workspace);
        auto data=Access::solution(corrected);
        data.options=original.options;
        data.diagnostics.mode_count_evaluations+=original.diagnostics.mode_count_evaluations;
        data.diagnostics.predictor_ad_evaluations+=original.diagnostics.predictor_ad_evaluations;
        data.diagnostics.fallback_count+=original.diagnostics.fallback_count+1;
        data.diagnostics.fallback_reasons.insert(data.diagnostics.fallback_reasons.begin(),
            original.diagnostics.fallback_reasons.begin(),original.diagnostics.fallback_reasons.end());
        data.diagnostics.fallback_reasons.push_back("endpoint_mode_certificate");
        cache.at(period)=Access::solution(std::move(data));
        if(std::isfinite(corrected.phase_velocity())) repair.c_guess=corrected.phase_velocity();
        else repair_tracking=DispersionTracking{};
    }
}

template<class Refiner>
DispersionCurve compute_curve_in_context(const Model& model,
        const std::vector<double>& periods,const std::string& wave,int mode,
        const std::vector<std::string>& outputs,const Options& options,
        Refiner& refiner,RombergWorkspace* workspace=nullptr) {
    require(!periods.empty(),"periods must not be empty");
    check_options(options);
    std::map<double,ModeSolution> cache;
    for(double p:periods) {positive(p,"period");cache.emplace(p,ModeSolution());}
    std::vector<double> order;
    order.reserve(cache.size());
    for(const auto& entry:cache) order.push_back(entry.first);
    if(options.start=="long") std::reverse(order.begin(),order.end());
    Options current=options;
    DispersionTracking tracking;
    bool range_used=false;
    std::vector<double> segment;
    // A missing period ends a segment. Certify the last valid root as well
    // when the final requested period is missing rather than silently skipping it.

    for(size_t i=0;i<order.size();++i) {
        const double period=order[i];
        tracking.next_period=i+1<order.size()?order[i+1]:missing_value;
        tracking.initial_period=!range_used;
        tracking.check_endpoint=segment.empty();
        auto solution=compute_dispersion_in_context(model,period,wave,mode,outputs,
            current,refiner,tracking,workspace);
        cache.at(period)=solution;
        if(!std::isfinite(solution.phase_velocity())) {
            finish_tracking_segment(model,wave,mode,outputs,options,refiner,workspace,cache,segment);
            segment.clear();tracking=DispersionTracking{};current.c_guess=options.c_guess;
            continue;
        }
        segment.push_back(period);
        range_used=true;
        current.c_guess=solution.phase_velocity();
    }
    finish_tracking_segment(model,wave,mode,outputs,options,refiner,workspace,cache,segment);
    DispersionCurve result;result.periods=periods;result.solutions.reserve(periods.size());
    for(double p:periods) result.solutions.push_back(cache.at(p));
    return result;
}
} // namespace detail

int compute_mode_count(const ModeSolution& solution, int level) {
    const auto& d=Access::solution(solution);
    if(!std::isfinite(d.c)) throw Error("missing_solution","Cannot count a missing dispersion solution");
    detail::require(level >= -1,"level must be -1 or a saved refinement level");
    if (level == -1) level=static_cast<int>(d.roots.size())-1;
    detail::require(level >= 0 && level < static_cast<int>(d.roots.size()),
                    "Requested level was not saved by the dispersion solve");
    const auto stack=detail::mesh(d.model,level);
    return calculateAdModeCount(stack.layers,stack.thicknesses,stack.halfspace,
        2*M_PI/d.period,d.roots[level],parseWaveType(d.wave));
}

ModeSolution compute_dispersion(const Model& model,double period,const std::string& wave,int mode,const std::vector<std::string>& requested,const Options& o) {
    auto refiner=disper26::detail::make_dispersion_mesh_cache<VelocityGridRefinement>(
        12,[&](int level){return detail::mesh(model,level);});
    detail::DispersionTracking tracking;
    return detail::compute_dispersion_in_context(model,period,wave,mode,requested,o,refiner,tracking);
}
ModeSolution compute_dispersion(const Model& model,double period,const std::string& wave,int mode,const std::vector<std::string>& requested,const Options& o,DispersionContext& context) {
    auto& scratch=Access::context(context);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    scratch.begin_call();
    auto refiner=[&](int level)->const VelocityGridRefinement& {return scratch.mesh(model,level);};
    detail::DispersionTracking tracking;
    return detail::compute_dispersion_in_context(model,period,wave,mode,requested,o,refiner,tracking,&scratch.romberg);
}
DispersionCurve compute_dispersion_curve(const Model& m,const std::vector<double>& periods,const std::string& wave,int mode,const std::vector<std::string>& out,const Options& options) {
    auto refiner=disper26::detail::make_dispersion_mesh_cache<VelocityGridRefinement>(
        12,[&](int level){return detail::mesh(m,level);});
    return detail::compute_curve_in_context(m,periods,wave,mode,out,options,refiner);
}
DispersionCurve compute_dispersion_curve(const Model& m,const std::vector<double>& periods,const std::string& wave,int mode,const std::vector<std::string>& out,const Options& options,DispersionContext& context) {
    auto& scratch=Access::context(context);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    scratch.begin_call();
    auto refiner=[&](int level)->const VelocityGridRefinement& {return scratch.mesh(m,level);};
    return detail::compute_curve_in_context(m,periods,wave,mode,out,options,refiner,&scratch.romberg);
}

DerivativeResult compute_kernels(const ModeSolution& s,const std::string& basis,const std::vector<std::string>& requested,const Options& o) {
    detail::check_options(o);detail::derivative_wave(s); const auto out=detail::outputs(requested);
    detail::check_basis(s.model(),basis);
    const auto& m=s.model();const auto& d=m.data();const int nn=static_cast<int>(detail::parameter_materials(m).size());
    const bool cu=out.count("group_velocity"),ch=out.count("hv_ratio")&&s.wave()=="Rayleigh",cc=out.count("phase_velocity");
    detail::require(o.method=="standard"||o.method=="variational"||o.method=="ad", "Kernel method must be standard, variational or ad");
    DerivativeResult r;r.diagnostics=detail::diagnostic(o.method,o);
    for(const auto& n:{"phase_velocity","group_velocity","hv_ratio"}) r.diagnostics.availability[n]=out.count(n)?"computed":"not_requested";
    if(s.wave()=="Love")r.diagnostics.availability["hv_ratio"]="not_applicable";
    for(const auto& n:out) if(n!="hv_ratio"||s.wave()=="Rayleigh") r.gradients[n]=std::vector<double>(nn*6,0);
    std::vector<SensitivityKernel> raw;
    if(d.kind=="grid"&&d.interpolation=="stiffness"&&!cu&&!ch&&o.method!="ad") {
        const auto roots=detail::trace_roots(s,o.max_levels);
        const auto v=computeKernelsVariationalGrid(detail::grid(m),1/s.period(),s.phase_velocity(),parseWaveType(s.wave()),
            o.max_levels,o.derivative_tolerance,0,&roots);
        if(cc) for(int i=0;i<nn;++i) for(int j=0;j<6;++j) r.gradients["phase_velocity"][i*6+j]=v.gradient_node(i*6+(j==5?0:j+1));
        r.diagnostics.method="variational";r.diagnostics.relative_error=v.relative_step;
        r.diagnostics.accepted_level=v.accepted_level;r.diagnostics.generated_level=static_cast<int>(roots.size())-1;
        r.diagnostics.convergence=std::isfinite(v.relative_step)?(v.relative_step<=o.derivative_tolerance?"converged":"not_converged"):"not_evaluated";
    } else if(d.kind=="grid") {
        if(o.method=="variational") detail::unsupported("Standalone variational GRID kernels currently provide phase-velocity material gradients for stiffness interpolation");
        const auto roots=detail::trace_roots(s,s.diagnostics().accepted_level);
        if(d.interpolation=="stiffness") {
            auto k=py_computeDispersionRombergWithKernels(d.depths,detail::unpack(d.materials),d.discontinuities,s.period(),s.wave(),s.phase_velocity(),"VTI",
                s.options().max_steps,s.options().tolerance,cc,cu,ch,!cu&&!ch,cu||ch,cu||ch,true,0,
                o.method=="ad"?"full_ad":"variational_frequency_ad",true,d.geometry.kind,d.geometry.planet_radius,d.geometry.flattening_exponent,roots,s.diagnostics().accepted_level);
            raw=std::move(std::get<5>(k));
        } else {
            auto v=detail::velocities(m);
            auto k=py_computeDispersionRombergVelocityVtiWithKernels(d.depths,v.rho,v.ph,v.pv,v.sh,v.sv,v.eta,d.discontinuities,s.period(),s.wave(),s.phase_velocity(),
                s.options().max_steps,s.options().tolerance,cc,cu,ch,!cu&&!ch,cu||ch,cu||ch,true,0,
                o.method=="ad"?"full_ad":"variational_frequency_ad",true,d.geometry.kind,d.geometry.planet_radius,d.geometry.flattening_exponent,roots,s.diagnostics().accepted_level);
            raw=std::move(std::get<5>(k));
        }
        r.diagnostics.method=cu&&o.method!="ad"?"ad+variational_frequency_ad":"ad";
        r.diagnostics.accepted_level=s.diagnostics().accepted_level;
        r.diagnostics.messages.push_back("Kernel convergence is not independently estimated by this AD path");
    } else {
        const auto mesh=detail::mesh(m,0);
        if(o.method=="ad" || (o.method=="standard" && ch)) {
            raw=py_computeSensitivityFullAD(mesh.layers,mesh.thicknesses,mesh.halfspace,2*M_PI/s.period(),s.phase_velocity(),s.wave(),"VTI",false);
            r.diagnostics.method="ad";
        } else {
            raw=py_computeSensitivityWeakForm(mesh.layers,mesh.thicknesses,mesh.halfspace,2*M_PI/s.period(),s.phase_velocity(),s.wave(),"VTI",false,
                o.num_sub_layers,detail::analytical(m,o),"variational_frequency_ad",cc,cu,ch);
            r.diagnostics.method=cu?"variational_frequency_ad":"variational";
            if(detail::has_fluid(m)&&cu)r.diagnostics.messages.push_back("Fluid Rayleigh group kernels use the established variational fallback");
        }
    }
    if(!raw.empty()) {
        const bool velocity=d.kind=="grid"&&d.interpolation=="velocity";
        const std::vector<std::string> names=velocity?std::vector<std::string>{"rho","Vph","Vpv","Vsh","Vsv","eta"}:detail::stiffness_names();
        std::vector<bool> seen(nn*6,false);
        for(const auto& k:raw) {
            auto it=std::find(names.begin(),names.end(),k.parameterName);if(it==names.end())continue;
            int i=k.layerIndex<0?nn-1:k.layerIndex;int j=static_cast<int>(it-names.begin());detail::require(i>=0&&i<nn,"Invalid kernel index");
            seen[i*6+j]=true;
            if(cc)r.gradients["phase_velocity"][i*6+j]=k.d_PhaseVelocity;
            if(cu)r.gradients["group_velocity"][i*6+j]=k.d_GroupVelocity;
            if(ch)r.gradients["hv_ratio"][i*6+j]=k.d_HVRatio;
        }
        detail::require(std::all_of(seen.begin(),seen.end(),[](bool v){return v;}),"Incomplete kernel metadata");
        if(velocity) {
            // Pull velocity-interpolation derivatives back to node stiffness,
            // then through the requested map. This never changes interpolation.
            for(int i=0;i<nn;++i) {
                auto map=detail::basis_map(d.materials[i],"velocity_vti");
                if(detail::fluid(d.materials[i])) {
                    for(auto& kv:r.gradients)std::fill(kv.second.begin()+i*6,kv.second.begin()+(i+1)*6,0);
                    continue;
                }
                for(auto& kv:r.gradients) {
                    Eigen::Map<Eigen::VectorXd> g(kv.second.data()+i*6,6);
                    Eigen::VectorXd stiffness=map.J.transpose().fullPivLu().solve(g);g=stiffness;
                }
            }
        }
    }
    detail::transform(r,m,basis,d.kind=="grid");r.diagnostics.mode_verified=true;detail::ensure_finite(r);detail::finish(r.diagnostics,o.strict);return r;
}

DerivativeResult compute_hessian(const ModeSolution& s,const std::string& basis,const Options& o) {
    detail::check_options(o);detail::derivative_wave(s);const auto& m=s.model();const auto& d=m.data();
    detail::check_basis(m,basis);
    detail::require(o.method=="standard"||o.method=="variational_ad"||o.method=="ad", "Hessian method must be standard, variational_ad or ad");
    if(d.kind=="pancake" && o.method=="ad") {
        if((d.materials.size()+1)*6+1>MAX_AD_PARAMS)
            detail::unsupported("Pancake Full AD Hessian exceeds the backend capacity of 80 AD coordinates");
        for(const auto& p:detail::parameter_materials(m)) if(p.gravity!=0)
            detail::unsupported("Pancake Full AD Hessian does not preserve Cowling gravity parameters");
    }
    HessianResult raw;
    if(d.kind=="grid") {
        if(d.interpolation!="stiffness")detail::unsupported("GRID Hessians require stiffness interpolation; the model is not converted automatically");
        if(o.method=="ad")detail::unsupported("Full AD GRID is a fixed-level validation path, not a standard extrapolated Hessian");
        auto roots=detail::trace_roots(s,o.max_levels);auto g=detail::grid(m);
        raw=s.wave()=="Rayleigh"?
            computePhaseVelocityHessianWeakFormRayleigh_Grid(g,1/s.period(),s.phase_velocity(),SensitivityMode::VTI,o.max_levels,o.derivative_tolerance,parseWFHessianBackend(o.backend),&roots):
            computePhaseVelocityHessianWeakFormLove_Grid(g,1/s.period(),s.phase_velocity(),SensitivityMode::VTI,o.max_levels,o.derivative_tolerance,0,parseWFHessianBackend(o.backend),&roots);
    } else {
        auto mesh=detail::mesh(m,0);
        if(o.method=="ad") raw=computePhaseVelocityHessianFullAD(mesh.layers,mesh.thicknesses,mesh.halfspace,1/s.period(),s.phase_velocity(),parseWaveType(s.wave()),SensitivityMode::VTI,false);
        else if(s.wave()=="Rayleigh") raw=computePhaseVelocityHessianWeakFormRayleigh(mesh.layers,mesh.thicknesses,mesh.halfspace,1/s.period(),s.phase_velocity(),SensitivityMode::VTI,false,o.num_sub_layers,detail::analytical(m,o),parseWFHessianBackend(o.backend));
        else raw=computePhaseVelocityHessianWeakFormLove(mesh.layers,mesh.thicknesses,mesh.halfspace,1/s.period(),s.phase_velocity(),SensitivityMode::VTI,false,200,detail::analytical(m,o),LoveBranchPolicy::Auto,1.0,parseWFHessianBackend(o.backend));
    }
    auto r=detail::from_hessian(raw,m);detail::transform(r,m,basis,d.kind=="grid");
    r.diagnostics=detail::diagnostic(o.method=="ad"?"ad":"variational_ad",o);
    r.diagnostics.availability["phase_velocity"]="computed";r.diagnostics.availability["hessian"]="computed";
    r.diagnostics.messages.push_back("Backend does not expose an independent Hessian error estimate");
    r.diagnostics.mode_verified=true;detail::ensure_finite(r);detail::finish(r.diagnostics,o.strict);return r;
}

DerivativeResult compute_hessian_reduced(const ModeSolution& s,const std::vector<double>& projection,int columns,const Options& o) {
    detail::check_options(o);detail::derivative_wave(s);const auto& m=s.model();
    if(m.data().kind!="grid"||m.data().interpolation!="stiffness")detail::unsupported("Reduced Hessian requires stiffness GRID; no full-matrix fallback is performed");
    detail::require(o.method=="standard"||o.method=="variational_ad","Reduced Hessian requires variational_ad");
    const int n=static_cast<int>(m.data().materials.size())*6;
    detail::require(columns>0 && projection.size()==static_cast<size_t>(n)*columns,"Projection shape must be material-stiffness axes by columns");
    Eigen::Map<const Eigen::MatrixXd> canonical(projection.data(),n,columns);detail::require(canonical.allFinite(),"Projection must be finite");
    Eigen::MatrixXd J(n,columns);
    for(int i=0;i<n/6;++i)for(int j=0;j<6;++j)J.row(i*6+(j==5?0:j+1))=canonical.row(i*6+j);
    auto roots=detail::trace_roots(s,o.max_levels);auto g=detail::grid(m);
    GridHessianReducedResult raw=s.wave()=="Rayleigh"?
        computePhaseVelocityHessianWeakFormRayleigh_Grid_Reduced(g,1/s.period(),s.phase_velocity(),J,o.max_levels,o.derivative_tolerance,parseWFHessianBackend(o.backend),&roots):
        computePhaseVelocityHessianWeakFormLove_Grid_Reduced(g,1/s.period(),s.phase_velocity(),J,o.max_levels,o.derivative_tolerance,true,1,0,parseWFHessianBackend(o.backend),&roots);
    DerivativeResult r;r.basis="linear_projection";r.rows=r.cols=columns;
    Eigen::VectorXd grad=J.transpose()*raw.gradient_node;
    r.gradients["phase_velocity"]={grad.data(),grad.data()+grad.size()};
    r.hessian={raw.hessian_reduced.data(),raw.hessian_reduced.data()+raw.hessian_reduced.size()};
    for(int i=0;i<columns;++i) {
        bool active=false;
        for(int node=0;node<n/6;++node) if(!detail::fluid(m.data().materials[node]))
            active=active || canonical.block(node*6,i,6,1).squaredNorm()!=0;
        r.parameters.push_back({"q"+std::to_string(i),"projection","user_defined",i,active});
    }
    r.diagnostics=detail::diagnostic("variational_ad_reduced",o);
    r.diagnostics.availability["phase_velocity"]="computed";
    r.diagnostics.availability["hessian"]="computed";
    r.diagnostics.messages.push_back("Projection is linear in node stiffness; nonlinear parameterizations require their map curvature term");
    r.diagnostics.mode_verified=true;detail::ensure_finite(r);detail::finish(r.diagnostics,o.strict);return r;
}

DerivativeResult compute_interface_kernels(const ModeSolution& s,const Options& o) {
    detail::check_options(o);detail::derivative_wave(s);const auto& m=s.model();const auto& d=m.data();
    detail::require(o.method=="standard"||o.method=="variational","Interface kernels require variational method");
    DerivativeResult r;r.basis="interface_depth";r.diagnostics=detail::diagnostic("variational",o);
    r.gradients["phase_velocity"]={};
    r.diagnostics.availability["phase_velocity"]="computed";
    if(d.kind=="grid") {
        if(d.interpolation!="stiffness"||d.geometry.kind!="flat")detail::unsupported("GRID interface derivatives currently require flat stiffness interpolation");
        auto roots=detail::trace_roots(s,o.max_levels);int j=0;
        for(size_t i=0;i<d.discontinuities.size();++i)if(d.discontinuities[i]) {
            const double value=py_computeInterfaceGradientVariationalExtrapolated(d.depths,detail::unpack(d.materials),d.discontinuities,s.period(),s.wave(),s.phase_velocity(),j,o.max_levels,roots,s.diagnostics().accepted_level);
            r.gradients["phase_velocity"].push_back(value);r.parameters.push_back({"Depth","interface","km",j++,true});
        }
    } else {
        auto mesh=detail::mesh(m,0);
        auto v=s.wave()=="Rayleigh"?py_computeBoundaryKernelsRayleigh(mesh.layers,mesh.thicknesses,mesh.halfspace,2*M_PI/s.period(),s.phase_velocity(),detail::analytical(m,o)):
            py_computeBoundaryKernelsLove(mesh.layers,mesh.thicknesses,mesh.halfspace,2*M_PI/s.period(),s.phase_velocity());
        r.gradients["phase_velocity"]=v;
        for(int j=0;j<static_cast<int>(v.size());++j)r.parameters.push_back({"Depth","interface","km",j,true});
        r.diagnostics.messages.push_back("Boundary jumps represent isolated boundary shifts; layer-thickness derivatives require sums over every displaced boundary");
    }
    r.diagnostics.mode_verified=true;detail::ensure_finite(r);detail::finish(r.diagnostics,o.strict);return r;
}
DerivativeResult compute_interface_hessian_row(const ModeSolution& s,int interface_index,const std::string& basis,const Options& o) {
    detail::check_options(o);detail::derivative_wave(s);const auto& d=s.model().data();
    detail::check_basis(s.model(),basis);
    detail::require(o.method=="standard"||o.method=="variational_ad","Interface Hessian requires variational_ad");
    if(d.kind!="grid"||d.interpolation!="stiffness"||d.geometry.kind!="flat")detail::unsupported("Interface Hessian requires flat stiffness GRID");
    const int ndisc=static_cast<int>(std::count(d.discontinuities.begin(),d.discontinuities.end(),true));
    detail::require(interface_index>=0&&interface_index<ndisc,"Interface index out of range");
    detail::require(basis=="stiffness"||basis=="isotropic"||basis=="velocity_vti"||basis=="log_velocity_vti","Unknown derivative basis");
    auto roots=detail::trace_roots(s,o.max_levels);
    auto raw=py_computeInterfaceHessianRowVariationalExtrapolated(d.depths,detail::unpack(d.materials),d.discontinuities,s.period(),s.wave(),s.phase_velocity(),interface_index,o.max_levels,"VTI",roots,s.diagnostics().accepted_level);
    const int nn=static_cast<int>(d.materials.size()),q=basis=="isotropic"?3:6;
    DerivativeResult r;r.basis=basis;r.rows=1;r.cols=nn*q+ndisc;r.hessian.assign(r.cols,0);r.parameters=detail::axes(s.model(),basis,true);
    // Interface wrapper VTI rows are A,C,F,L,N,Rho, followed by depths.
    for(int i=0;i<nn;++i) {
        if(detail::fluid(d.materials[i]))continue;
        auto map=detail::basis_map(d.materials[i],basis);
        Eigen::Map<const Eigen::VectorXd> row(raw.data()+i*6,6);Eigen::VectorXd transformed=map.J.transpose()*row;
        std::copy(transformed.data(),transformed.data()+q,r.hessian.begin()+i*q);
    }
    for(int j=0;j<ndisc;++j){r.hessian[nn*q+j]=raw[nn*6+j];r.parameters.push_back({"Depth","interface","km",j,true});}
    r.diagnostics=detail::diagnostic("variational_ad",o);
    r.diagnostics.availability["hessian"]="computed";
    r.diagnostics.mode_verified=true;detail::ensure_finite(r);detail::finish(r.diagnostics,o.strict);return r;
}

std::vector<Capability> get_capabilities(const Model& m,const std::string& wave) {
    const auto w=detail::wave_name(wave);
    const auto& d=m.data();
    const bool pancake=d.kind=="pancake";
    const bool stiffness=pancake||d.interpolation=="stiffness";
    std::string base_reason;
    if(pancake && d.geometry.kind!="flat")
        base_reason="Pancake spherical geometry requires explicit earth flattening";
    else if(detail::fluid(d.halfspace))
        base_reason="A fluid halfspace is not supported by the mode solver";
    else if(w=="Love" && detail::has_fluid(m))
        base_reason="Love mode counting with fluid materials is not supported";
    else {
        try {
            const auto stack=detail::mesh(m,0);
            if(w=="Rayleigh") (void)countTopLiquidLayersOrThrow(stack.layers);
        } catch(const std::exception& error) {base_reason=error.what();}
    }
    std::string grid_variational_reason;
    if(!pancake && stiffness && w=="Rayleigh") {
        try {validate_rayleigh_grid_fixed_fluid_model(detail::grid(m));}
        catch(const std::exception& error) {grid_variational_reason=error.what();}
    }
    const bool cowling=std::any_of(d.materials.begin(),d.materials.end(),
        [](const Material& material){return material.gravity!=0.0;}) || d.halfspace.gravity!=0.0;
    const std::string derivative_reason = w=="Rayleigh" && cowling
        ? "Rayleigh material/interface derivatives require a non-gravitating reference model" : "";
    const auto first_reason=[](const std::string& first,const std::string& second) {
        return first.empty()?second:first;
    };
    std::vector<Capability> result;
    auto add=[&](const std::string& operation,const std::string& basis,
                 const std::string& method,const std::string& reason) {
        result.push_back({operation,basis,method,reason,reason.empty()});
    };
    add("dispersion","","mode_order",base_reason);
    add("dispersion_curve","","mode_order",base_reason);
    add("dispersion","","continuation",base_reason);
    add("dispersion_curve","","continuation",base_reason);
    add("dispersion","","continuation_recover",base_reason);
    add("dispersion_curve","","continuation_recover",base_reason);
    for(const auto& basis:{"isotropic","stiffness","velocity_vti","log_velocity_vti"}) {
        std::string basis_reason;
        try {detail::check_basis(m,basis);}
        catch(const std::exception& error) {basis_reason=error.what();}
        const auto common=first_reason(first_reason(base_reason,derivative_reason),basis_reason);
        // Kernel capability rows describe the default phase-velocity output.
        // Explicit variational GRID kernels are limited to this output, while
        // AD supports the value-specific derivative requests as well.
        add("kernels",basis,"standard",first_reason(common,
            !pancake && stiffness?grid_variational_reason:std::string{}));
        add("kernels",basis,"ad",common);
        add("kernels",basis,"variational",first_reason(common,
            stiffness?grid_variational_reason:
                "Variational GRID phase kernels require stiffness interpolation"));
        const auto hessian_reason=first_reason(common,stiffness?
            grid_variational_reason:"GRID Hessians require stiffness interpolation");
        add("hessian",basis,"standard",hessian_reason);
        add("hessian",basis,"variational_ad",hessian_reason);
        std::string ad_reason;
        if(!pancake) ad_reason="Full AD GRID is a fixed-level validation API, not the extrapolated public Hessian";
        else if(cowling) ad_reason="Full AD Pancake Hessians do not preserve Cowling gravity fields";
        else if((d.materials.size()+1)*6+1>MAX_AD_PARAMS)
            ad_reason="Full AD Pancake Hessian exceeds MAX_AD_PARAMS="+std::to_string(MAX_AD_PARAMS);
        add("hessian",basis,"ad",first_reason(common,ad_reason));
    }
    add("hessian_reduced","linear_projection","variational_ad",first_reason(first_reason(base_reason,derivative_reason),
        !pancake && stiffness?grid_variational_reason:"Reduced Hessians require stiffness GRID"));
    const auto interface_reason=first_reason(first_reason(base_reason,derivative_reason),
        stiffness && d.geometry.kind=="flat"?std::string{}:
            "Interface derivatives require flat stiffness GRID or flat Pancake");
    add("interface_kernels","interface_depth","variational",interface_reason);
    for(const auto& basis:{"isotropic","stiffness","velocity_vti","log_velocity_vti"}) {
        std::string reason=first_reason(interface_reason,
            pancake?"Interface Hessian rows require stiffness GRID":std::string{});
        try {detail::check_basis(m,basis);}
        catch(const std::exception& error) {reason=first_reason(reason,error.what());}
        add("interface_hessian_row",basis,"variational_ad",reason);
    }
    add("eigenfunctions","","propagation",base_reason);
    add("energy_integrals","","standard",base_reason);
    add("frequency_curvature","","ad",base_reason);
    add("gravity_correction","","kg",first_reason(base_reason,
        !pancake || w!="Rayleigh"?"Kg correction requires flat Rayleigh Pancake":
        (cowling?"Kg correction requires a non-gravitating reference model":std::string{})));
    add("earth_flattening_transformation","","",!d.flattened && d.geometry.kind!="flat"?
        std::string{}:"EFT requires an unflattened spherical model");
    add("mineos_physical_dispersion","","",!d.q_period?std::string{}:"Q correction cannot be applied twice");
    return result;
}
}} // namespace disper26::api
#include "api_related_impl.hpp"
