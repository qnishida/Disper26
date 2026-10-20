#include "api_c.h"
#include "api.hpp"
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <cstdio>
#include <string_view>
#include <sstream>
#include <utility>

namespace api = disper26::api;
struct d26_api_dispersion_context { std::atomic<size_t> references{1}; api::DispersionContext value; };
struct d26_api_model { std::atomic<size_t> references{1}; api::Model value; };
struct d26_api_solution { std::atomic<size_t> references{1}; api::ModeSolution value; };
struct NumericField { std::vector<double> values; size_t rows=0, columns=1; };
struct d26_api_result {
    std::atomic<size_t> references{1};
    std::map<std::string,NumericField> numeric;
    std::map<std::string,std::string> strings;
    std::vector<api::ModeSolution> solutions;
    bool is_curve=false;
};
namespace {
struct BufferError {};
void require(bool condition,const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void message_copy(char* destination,size_t capacity,std::string_view value) noexcept {
    if (!destination || !capacity) return;
    const auto length=std::min(capacity-1,value.size());
    std::memcpy(destination,value.data(),length); destination[length]='\0';
}
template<class F> int guard(char* message,size_t capacity,F&& function) noexcept {
    message_copy(message,capacity,"");
    try { require(message || capacity==0,"NULL message with nonzero capacity"); function(); return D26_API_OK; }
    catch (const BufferError&) { message_copy(message,capacity,"Buffer is too small"); return D26_API_BUFFER_TOO_SMALL; }
    catch (const std::invalid_argument& e) { message_copy(message,capacity,e.what()); return D26_API_INVALID_ARGUMENT; }
    catch (const api::Error& e) {
        if(message && capacity) std::snprintf(message,capacity,"%s: %s",e.code.c_str(),e.what());
        if(e.code=="invalid_argument") return D26_API_INVALID_ARGUMENT;
        if(e.code=="unsupported") return D26_API_UNSUPPORTED;
        if(e.code=="not_converged") return D26_API_NOT_CONVERGED;
        return D26_API_CALCULATION_FAILED;
    }
    catch (const std::exception& e) { message_copy(message,capacity,e.what()); return D26_API_CALCULATION_FAILED; }
    catch (...) { message_copy(message,capacity,"Unknown calculation failure"); return D26_API_CALCULATION_FAILED; }
}
template<class T> void prepare(T** out) { require(out,"NULL output handle"); *out=nullptr; }
template<class T> void retain(T* p) { if(p) p->references.fetch_add(1,std::memory_order_relaxed); }
template<class T> void release(T* p) { if(p && p->references.fetch_sub(1,std::memory_order_acq_rel)==1) delete p; }
template<class T> std::vector<T> copy_array(const T* p,size_t count) {
    require(p || !count,"NULL input array with nonzero length");
    require(count<=static_cast<size_t>(INT_MAX),"Input array exceeds supported index range");
    return count ? std::vector<T>(p,p+count) : std::vector<T>{};
}
std::string text(const char* p,const char* fallback="") { return p ? p : fallback; }
api::Material material(const d26_api_material& p) {
    return {p.rho,p.A,p.C,p.F,p.L,p.N,p.gravity,p.gravity_internal_scale,p.gravity_surface_scale};
}
d26_api_material material(const api::Material& p) {
    return {p.rho,p.A,p.C,p.F,p.L,p.N,p.gravity,p.gravity_internal_scale,p.gravity_surface_scale};
}
std::vector<api::Material> materials(const d26_api_material* p,size_t n) {
    auto input=copy_array(p,n); std::vector<api::Material> output; output.reserve(n);
    for(const auto& value:input) output.push_back(material(value));
    return output;
}
api::Geometry geometry(const d26_api_geometry* g) {
    if(!g) return {};
    std::string kind;
    switch(g->kind) {
    case D26_API_FLAT: kind="flat"; break;
    case D26_API_SPHERICAL_BA97: kind="spherical_ba97"; break;
    case D26_API_SPHERICAL_EXPERIMENTAL_DENSITY: kind="spherical_experimental_density"; break;
    default: throw std::invalid_argument("Invalid geometry kind");
    }
    return {kind,g->planet_radius,g->flattening_exponent};
}
std::string fixed_string(const char (&value)[64]) {
    auto end=static_cast<const char*>(std::memchr(value,'\0',64));
    require(end,"Option string must terminate within 64 bytes"); return std::string(value,end);
}
api::Options options(const d26_api_options* p) {
    if(!p) return {};
    api::Options result;
    result.max_steps=p->max_steps; result.max_levels=p->max_levels; result.num_sub_layers=p->num_sub_layers;
    result.tolerance=p->tolerance; result.root_tolerance=p->root_tolerance; result.derivative_tolerance=p->derivative_tolerance;
    if(p->has_c_guess) result.c_guess=p->c_guess;
    result.method=fixed_string(p->method); result.backend=fixed_string(p->backend); result.integration=fixed_string(p->integration);
    result.strict=p->strict!=0; return result;
}
std::vector<std::string> outputs(const char* p) {
    if(!p) return {"phase_velocity"};
    std::vector<std::string> result; std::stringstream input(p); std::string item;
    while(std::getline(input,item,',')) {
        const auto start=item.find_first_not_of(" \t");
        const auto end=item.find_last_not_of(" \t");
        require(start!=std::string::npos,"Empty output name"); result.push_back(item.substr(start,end-start+1));
    }
    require(!result.empty(),"At least one output is required"); return result;
}
const api::Model& model(const d26_api_model* p) { require(p,"NULL model"); return p->value; }
const api::ModeSolution& solution(const d26_api_solution* p) { require(p,"NULL solution"); return p->value; }
const d26_api_result& result(const d26_api_result* p) { require(p,"NULL result"); return *p; }
void scalar(d26_api_result& r,const std::string& name,double value) { r.numeric[name]={{value},1,1}; }
void vector(d26_api_result& r,const std::string& name,std::vector<double> value) {
    auto size=value.size(); r.numeric[name]={std::move(value),size,1};
}
void diagnostics(d26_api_result& r,const api::Diagnostics& d) {
    r.strings["diagnostics.status"]=d.status; r.strings["diagnostics.convergence"]=d.convergence;
    r.strings["diagnostics.method"]=d.method; r.strings["diagnostics.backend"]=d.backend;
    std::string messages; for(const auto& message:d.messages) { if(!messages.empty()) messages+='\n'; messages+=message; }
    r.strings["diagnostics.messages"]=std::move(messages);
    for(const auto& entry:d.availability) r.strings["availability."+entry.first]=entry.second;
    for(const auto& entry:d.convergence_by_quantity) r.strings["convergence."+entry.first]=entry.second;
    scalar(r,"diagnostics.accepted_level",d.accepted_level); scalar(r,"diagnostics.generated_level",d.generated_level);
    scalar(r,"diagnostics.fallback_count",d.fallback_count); scalar(r,"diagnostics.relative_error",d.relative_error);
    scalar(r,"diagnostics.mode_verified",d.mode_verified);
    scalar(r,"diagnostics.mode_count_evaluations",d.mode_count_evaluations);
    scalar(r,"diagnostics.predictor_ad_evaluations",d.predictor_ad_evaluations);
    vector(r,"diagnostics.level_mode_verified",std::vector<double>(d.level_mode_verified.begin(),d.level_mode_verified.end()));
    auto join=[](const std::vector<std::string>& values) {
        std::string out; for(const auto& value:values) {if(!out.empty())out+='\n';out+=value;} return out;
    };
    r.strings["diagnostics.predictors"]=join(d.predictors);
    r.strings["diagnostics.fallback_reasons"]=join(d.fallback_reasons);
}
std::unique_ptr<d26_api_result> derivative(api::DerivativeResult d) {
    auto r=std::make_unique<d26_api_result>(); r->strings["basis"]=d.basis;
    std::vector<double> indices,active;
    for(size_t i=0;i<d.parameters.size();++i) {
        const auto& p=d.parameters[i]; auto prefix="parameter."+std::to_string(i)+".";
        r->strings[prefix+"name"]=p.name; r->strings[prefix+"entity"]=p.entity; r->strings[prefix+"unit"]=p.unit;
        indices.push_back(p.index); active.push_back(p.active ? 1.0 : 0.0);
    }
    vector(*r,"parameter.index",std::move(indices)); vector(*r,"parameter.active",std::move(active));
    for(auto& entry:d.gradients) vector(*r,"gradient."+entry.first,std::move(entry.second));
    if(d.rows || d.cols) r->numeric["hessian"]={std::move(d.hessian),static_cast<size_t>(d.rows),static_cast<size_t>(d.cols)};
    diagnostics(*r,d.diagnostics); return r;
}
void string_output(const std::string& value,char* buffer,size_t capacity,size_t* required) {
    require(required,"NULL required string length"); *required=value.size()+1;
    require(buffer || capacity==0,"NULL string buffer with nonzero capacity");
    if(!buffer && capacity==0) return;
    if(capacity<*required) throw BufferError{};
    std::memcpy(buffer,value.c_str(),*required);
}
}

extern "C" {
void d26_api_default_geometry(d26_api_geometry* g) { if(g) *g={0,6371.0,api::missing_value}; }
void d26_api_default_options(d26_api_options* p) {
    if(!p) return;
    // These public defaults mirror api::Options without allocating across a C boundary.
    *p={}; p->max_steps=5; p->max_levels=5; p->num_sub_layers=10;
    p->tolerance=1e-10; p->root_tolerance=1e-12; p->derivative_tolerance=1e-6;
    message_copy(p->method,64,"standard"); message_copy(p->backend,64,"dual"); message_copy(p->integration,64,"auto");
}
int d26_api_create_isotropic_parameters(double vs,double vp,double rho,d26_api_material* out,char* msg,size_t cap) {
    return guard(msg,cap,[&]{require(out,"NULL material output"); *out=material(api::create_isotropic_parameters(vs,vp,rho));});
}
int d26_api_create_fluid_parameters(double vp,double rho,d26_api_material* out,char* msg,size_t cap) {
    return guard(msg,cap,[&]{require(out,"NULL material output"); *out=material(api::create_fluid_parameters(vp,rho));});
}
int d26_api_create_velocity_vti_parameters(double rho,double vph,double vpv,double vsh,double vsv,double eta,
    d26_api_material* out,char* msg,size_t cap) {
    return guard(msg,cap,[&]{require(out,"NULL material output"); *out=material(api::create_velocity_vti_parameters(rho,vph,vpv,vsh,vsv,eta));});
}
int d26_api_create_pancake_model(size_t n,const d26_api_material* layers,const double* thicknesses,
    const d26_api_material* halfspace,const d26_api_geometry* geo,const char* name,d26_api_model** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(halfspace,"NULL halfspace");
        auto r=std::make_unique<d26_api_model>(); r->value=api::create_pancake_model(materials(layers,n),copy_array(thicknesses,n),material(*halfspace),geometry(geo),text(name)); *out=r.release();});
}
int d26_api_create_model(size_t n,const double* depths,const char* kind,
    const char* parameterization,const double* values,size_t components,
    const char* interpolation,const d26_api_geometry* geo,const char* name,
    d26_api_model** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&] {
        prepare(out);require(kind && parameterization,"kind and parameterization are required");
        const std::string basis=parameterization;
        require(basis=="isotropic" || basis=="velocity_vti" || basis=="stiffness","Unknown material parameterization");
        require(components==(basis=="isotropic"?3u:6u),"Material component count does not match parameterization");
        require(n<=static_cast<size_t>(INT_MAX)/components,"Material matrix exceeds supported size");
        const auto data=copy_array(values,n*components);
        api::MaterialArrays p;
        auto field=[&](size_t row) {std::vector<double> v;v.reserve(n);for(size_t i=0;i<n;++i)v.push_back(data[i*components+row]);return v;};
        if(basis=="isotropic") {p.vs=field(0);p.vp=field(1);p.rho=field(2);}
        else if(basis=="velocity_vti") {p.rho=field(0);p.vph=field(1);p.vpv=field(2);p.vsh=field(3);p.vsv=field(4);p.eta=field(5);}
        else {p.A=field(0);p.C=field(1);p.F=field(2);p.L=field(3);p.N=field(4);p.rho=field(5);}
        auto r=std::make_unique<d26_api_model>();
        r->value=api::create_model(copy_array(depths,n),kind,p,
            interpolation?std::optional<std::string>(interpolation):std::nullopt,geometry(geo),text(name));
        *out=r.release();
    });
}
int d26_api_create_grid_model(size_t n,const double* depths,const d26_api_material* p,const char* interpolation,
    const d26_api_geometry* geo,const char* name,d26_api_model** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(interpolation,"Interpolation must be explicit");
        auto r=std::make_unique<d26_api_model>(); r->value=api::create_grid_model(copy_array(depths,n),materials(p,n),interpolation,geometry(geo),text(name)); *out=r.release();});
}
void d26_api_model_retain(d26_api_model* p) { retain(p); }
void d26_api_model_release(d26_api_model* p) { release(p); }
void d26_api_solution_retain(d26_api_solution* p) { retain(p); }
void d26_api_solution_release(d26_api_solution* p) { release(p); }
void d26_api_result_retain(d26_api_result* p) { retain(p); }
void d26_api_result_release(d26_api_result* p) { release(p); }
int d26_api_compute_dispersion(const d26_api_model* p,double period,const char* wave,int mode,const char* quantities,
    const d26_api_options* opts,d26_api_solution** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto r=std::make_unique<d26_api_solution>();
        r->value=api::compute_dispersion(model(p),period,text(wave,"Rayleigh"),mode,outputs(quantities),options(opts)); *out=r.release();});
}
int d26_api_compute_dispersion_curve(const d26_api_model* p,size_t n,const double* periods,const char* wave,int mode,
    const char* quantities,const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto curve=api::compute_dispersion_curve(model(p),copy_array(periods,n),text(wave,"Rayleigh"),mode,outputs(quantities),options(opts));
        auto r=std::make_unique<d26_api_result>(); r->is_curve=true; vector(*r,"periods",std::move(curve.periods));
        std::vector<double> c,u,hv,u_present,hv_present;
        for(const auto& s:curve.solutions) { c.push_back(s.phase_velocity()); u.push_back(s.group_velocity().value_or(api::missing_value));
            hv.push_back(s.hv_ratio().value_or(api::missing_value)); u_present.push_back(s.group_velocity().has_value()); hv_present.push_back(s.hv_ratio().has_value()); }
        vector(*r,"phase_velocity",std::move(c)); vector(*r,"group_velocity",std::move(u)); vector(*r,"hv_ratio",std::move(hv));
        vector(*r,"group_velocity.available",std::move(u_present)); vector(*r,"hv_ratio.available",std::move(hv_present));
        r->solutions=std::move(curve.solutions); *out=r.release();});
}
int d26_api_create_dispersion_context(d26_api_dispersion_context** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out);auto value=std::make_unique<d26_api_dispersion_context>();*out=value.release();});
}
void d26_api_dispersion_context_retain(d26_api_dispersion_context* p) {retain(p);}
void d26_api_dispersion_context_release(d26_api_dispersion_context* p) {release(p);}
int d26_api_compute_dispersion_with_context(const d26_api_model* p,double period,const char* wave,int mode,const char* quantities,
    const d26_api_options* opts,d26_api_dispersion_context* ctx,d26_api_solution** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(ctx,"NULL dispersion context"); auto r=std::make_unique<d26_api_solution>();
        r->value=api::compute_dispersion(model(p),period,text(wave,"Rayleigh"),mode,outputs(quantities),options(opts),ctx->value); *out=r.release();});
}
int d26_api_compute_dispersion_curve_with_context(const d26_api_model* p,size_t n,const double* periods,const char* wave,int mode,
    const char* quantities,const d26_api_options* opts,d26_api_dispersion_context* ctx,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(ctx,"NULL dispersion context"); auto curve=api::compute_dispersion_curve(model(p),copy_array(periods,n),text(wave,"Rayleigh"),mode,outputs(quantities),options(opts),ctx->value);
        auto r=std::make_unique<d26_api_result>(); r->is_curve=true; vector(*r,"periods",std::move(curve.periods));
        std::vector<double> c,u,hv,u_present,hv_present;
        for(const auto& s:curve.solutions) { c.push_back(s.phase_velocity()); u.push_back(s.group_velocity().value_or(api::missing_value));
            hv.push_back(s.hv_ratio().value_or(api::missing_value)); u_present.push_back(s.group_velocity().has_value()); hv_present.push_back(s.hv_ratio().has_value()); }
        vector(*r,"phase_velocity",std::move(c)); vector(*r,"group_velocity",std::move(u)); vector(*r,"hv_ratio",std::move(hv));
        vector(*r,"group_velocity.available",std::move(u_present)); vector(*r,"hv_ratio.available",std::move(hv_present));
        r->solutions=std::move(curve.solutions); *out=r.release();});
}
int d26_api_curve_solution(const d26_api_result* p,size_t index,d26_api_solution** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); const auto& r=result(p); require(r.is_curve,"Result is not a dispersion curve");
        require(index<r.solutions.size(),"Curve index out of range"); auto s=std::make_unique<d26_api_solution>(); s->value=r.solutions[index]; *out=s.release();});
}
int d26_api_model_info(const d26_api_model* p,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); const auto& d=model(p).data(); auto r=std::make_unique<d26_api_result>();
        r->strings["kind"]=d.kind; r->strings["interpolation"]=d.interpolation; r->strings["name"]=d.name; r->strings["geometry"]=d.geometry.kind;
        r->strings["flattening_policy"]=d.flattening_policy;
        scalar(*r,"planet_radius",d.geometry.planet_radius); scalar(*r,"flattening_exponent",d.geometry.flattening_exponent);
        scalar(*r,"flattened",d.flattened); if(d.q_period) scalar(*r,"q_period",*d.q_period);
        if(d.q_reference_period) scalar(*r,"q_reference_period",*d.q_reference_period);
        vector(*r,"depths",d.depths); vector(*r,"thicknesses",d.thicknesses);
        std::vector<double> flags; for(bool b:d.discontinuities) flags.push_back(b); vector(*r,"discontinuities",std::move(flags));
        auto pack=[](const api::Material& m){ return std::vector<double>{m.rho,m.A,m.C,m.F,m.L,m.N,m.gravity,m.gravity_internal_scale,m.gravity_surface_scale}; };
        std::vector<double> values; for(const auto& m:d.materials) { auto row=pack(m); values.insert(values.end(),row.begin(),row.end()); }
        r->numeric["materials"]={std::move(values),9,d.materials.size()}; vector(*r,"halfspace",pack(d.halfspace)); *out=r.release();});
}
int d26_api_compute_mode_count(const d26_api_solution* p,int level,int* count,char* msg,size_t cap) {
    if(count) *count=-1;
    return guard(msg,cap,[&]{require(count,"NULL mode count output");
        *count=api::compute_mode_count(solution(p),level);});
}
int d26_api_solution_info(const d26_api_solution* p,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); const auto& s=solution(p); auto r=std::make_unique<d26_api_result>();
        scalar(*r,"phase_velocity",s.phase_velocity()); scalar(*r,"period",s.period()); scalar(*r,"mode",s.mode()); r->strings["wave"]=s.wave();
        if(s.group_velocity()) scalar(*r,"group_velocity",*s.group_velocity());
        if(s.hv_ratio()) scalar(*r,"hv_ratio",*s.hv_ratio());
        const auto& o=s.options(); scalar(*r,"options.max_steps",o.max_steps); scalar(*r,"options.max_levels",o.max_levels);
        scalar(*r,"options.num_sub_layers",o.num_sub_layers); scalar(*r,"options.tolerance",o.tolerance); scalar(*r,"options.root_tolerance",o.root_tolerance);
        scalar(*r,"options.derivative_tolerance",o.derivative_tolerance); scalar(*r,"options.strict",o.strict);
        if(o.c_guess) scalar(*r,"options.c_guess",*o.c_guess);
        r->strings["options.method"]=o.method; r->strings["options.backend"]=o.backend; r->strings["options.integration"]=o.integration;
        diagnostics(*r,s.diagnostics()); *out=r.release();});
}
int d26_api_solution_model(const d26_api_solution* p,d26_api_model** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto r=std::make_unique<d26_api_model>(); r->value=solution(p).model(); *out=r.release();});
}
int d26_api_compute_kernels(const d26_api_solution* p,const char* basis,const char* quantities,
    const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(basis,"Derivative basis must be explicit"); *out=derivative(api::compute_kernels(solution(p),basis,outputs(quantities),options(opts))).release();});
}
int d26_api_compute_hessian(const d26_api_solution* p,const char* basis,const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(basis,"Derivative basis must be explicit"); *out=derivative(api::compute_hessian(solution(p),basis,options(opts))).release();});
}
int d26_api_compute_hessian_reduced(const d26_api_solution* p,const double* projection,size_t rows,size_t columns,
    const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(rows>0 && columns>0 && rows<=INT_MAX && columns<=INT_MAX && rows<=static_cast<size_t>(INT_MAX)/columns,"Invalid projection dimensions");
        *out=derivative(api::compute_hessian_reduced(solution(p),copy_array(projection,rows*columns),static_cast<int>(columns),options(opts))).release();});
}
int d26_api_compute_interface_kernels(const d26_api_solution* p,const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); *out=derivative(api::compute_interface_kernels(solution(p),options(opts))).release();});
}
int d26_api_compute_interface_hessian_row(const d26_api_solution* p,int index,const char* basis,
    const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); require(basis,"Derivative basis must be explicit"); *out=derivative(api::compute_interface_hessian_row(solution(p),index,basis,options(opts))).release();});
}
int d26_api_compute_eigenfunctions(const d26_api_solution* p,int n,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto e=api::compute_eigenfunctions(solution(p),n); auto r=std::make_unique<d26_api_result>();
        vector(*r,"depth",std::move(e.depth)); std::vector<double> flags; for(bool b:e.is_fluid) flags.push_back(b); vector(*r,"is_fluid",std::move(flags));
        for(const auto& entry:e.components) { std::vector<double> re,im; for(auto value:entry.second) { re.push_back(value.real()); im.push_back(value.imag()); }
            vector(*r,entry.first+"_real",std::move(re)); vector(*r,entry.first+"_imag",std::move(im)); }
        diagnostics(*r,e.diagnostics); *out=r.release();});
}
int d26_api_compute_energy_integrals(const d26_api_solution* p,const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto e=api::compute_energy_integrals(solution(p),options(opts)); auto r=std::make_unique<d26_api_result>();
        scalar(*r,"I1",e.I1); scalar(*r,"I2",e.I2); scalar(*r,"I3",e.I3); scalar(*r,"phase_velocity",e.phase_velocity); scalar(*r,"group_velocity",e.group_velocity);
        diagnostics(*r,e.diagnostics); *out=r.release();});
}
int d26_api_compute_frequency_curvature(const d26_api_solution* p,const d26_api_options* opts,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto f=api::compute_frequency_curvature(solution(p),options(opts)); auto r=std::make_unique<d26_api_result>();
        scalar(*r,"phase_velocity",f.phase_velocity); scalar(*r,"group_velocity",f.group_velocity);
        scalar(*r,"group_velocity_omega_derivative",f.group_velocity_omega_derivative);
        scalar(*r,"wavenumber_omega_second_derivative",f.wavenumber_omega_second_derivative); diagnostics(*r,f.diagnostics); *out=r.release();});
}
int d26_api_compute_gravity_correction(const d26_api_solution* p,double gravity,const char* region,
    int num_sub_layers,double domega_relative,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto g=api::compute_gravity_correction(solution(p),gravity,text(region,"solid"),num_sub_layers,domega_relative);
        auto r=std::make_unique<d26_api_result>(); scalar(*r,"phase_velocity",g.phase_velocity); scalar(*r,"group_velocity",g.group_velocity);
        scalar(*r,"delta_c",g.delta_c); scalar(*r,"delta_u",g.delta_u); scalar(*r,"omega_delta_c_omega_derivative",g.omega_delta_c_omega_derivative);
        r->strings["region"]=g.region; diagnostics(*r,g.diagnostics); *out=r.release();});
}
int d26_api_earth_flattening_transformation(const d26_api_model* p,d26_api_model** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto r=std::make_unique<d26_api_model>(); r->value=api::earth_flattening_transformation(model(p)); *out=r.release();});
}
int d26_api_apply_mineos_physical_dispersion(const d26_api_model* p,size_t n,const double* qk,const double* qm,double ref,double period,
    d26_api_model** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto r=std::make_unique<d26_api_model>(); r->value=api::apply_mineos_physical_dispersion(model(p),copy_array(qk,n),copy_array(qm,n),ref,period); *out=r.release();});
}
int d26_api_get_capabilities(const d26_api_model* p,const char* wave,d26_api_result** out,char* msg,size_t cap) {
    if(out) *out=nullptr;
    return guard(msg,cap,[&]{prepare(out); auto values=api::get_capabilities(model(p),text(wave,"Rayleigh")); auto r=std::make_unique<d26_api_result>(); std::vector<double> supported;
        for(size_t i=0;i<values.size();++i) { const auto& v=values[i]; auto prefix="capability."+std::to_string(i)+".";
            r->strings[prefix+"operation"]=v.operation; r->strings[prefix+"basis"]=v.basis; r->strings[prefix+"method"]=v.method; r->strings[prefix+"reason"]=v.reason; supported.push_back(v.supported); }
        vector(*r,"supported",std::move(supported)); *out=r.release();});
}
int d26_api_result_numeric(const d26_api_result* p,const char* field,double* values,size_t capacity,size_t* rows,size_t* columns,char* msg,size_t cap) {
    return guard(msg,cap,[&]{require(rows && columns,"NULL output shape"); *rows=0; *columns=0; require(field,"NULL field name"); const auto& r=result(p);
        auto found=r.numeric.find(field); require(found!=r.numeric.end(),"Unknown numeric field"); const auto& f=found->second; *rows=f.rows; *columns=f.columns;
        require(values || capacity==0,"NULL numeric buffer with nonzero capacity"); if(!values && capacity==0) return;
        if(capacity<f.values.size()) throw BufferError{};
        std::copy(f.values.begin(),f.values.end(),values);});
}
int d26_api_result_string(const d26_api_result* p,const char* field,char* value,size_t capacity,size_t* required,char* msg,size_t cap) {
    return guard(msg,cap,[&]{require(required,"NULL required string length"); *required=0; require(field,"NULL field name"); const auto& r=result(p);
        auto found=r.strings.find(field); require(found!=r.strings.end(),"Unknown string field"); string_output(found->second,value,capacity,required);});
}
int d26_api_result_field_count(const d26_api_result* p,int numeric,size_t* count,char* msg,size_t cap) {
    return guard(msg,cap,[&]{require(count,"NULL field count"); *count=0; const auto& r=result(p); *count=numeric ? r.numeric.size() : r.strings.size();});
}
int d26_api_result_field_name(const d26_api_result* p,int numeric,size_t index,char* value,size_t capacity,size_t* required,char* msg,size_t cap) {
    return guard(msg,cap,[&]{require(required,"NULL required string length"); *required=0; const auto& r=result(p);
        std::string name;
        if(numeric) { require(index<r.numeric.size(),"Field index out of range"); auto iter=r.numeric.begin(); std::advance(iter,index); name=iter->first; }
        else { require(index<r.strings.size(),"Field index out of range"); auto iter=r.strings.begin(); std::advance(iter,index); name=iter->first; }
        string_output(name,value,capacity,required);});
}
}
