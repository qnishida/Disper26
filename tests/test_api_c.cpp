#include "api_c.h"
#include "api.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(d26_api_material)==9*sizeof(double), "Material C ABI size wrong");

namespace {
char message[2048];
void check(bool valid,const char* reason) { if(!valid) throw std::runtime_error(reason); }
void success(int code) { if(code!=D26_API_OK) throw std::runtime_error(message); }
std::vector<double> numeric(d26_api_result* r,const char* field,size_t* nrows=nullptr,size_t* ncols=nullptr) {
    size_t rows=0,cols=0;
    success(d26_api_result_numeric(r,field,nullptr,0,&rows,&cols,message,sizeof message));
    std::vector<double> values(rows*cols);
    success(d26_api_result_numeric(r,field,values.data(),values.size(),&rows,&cols,message,sizeof message));
    if(nrows) *nrows=rows;
    if(ncols) *ncols=cols;
    return values;
}
std::string text(d26_api_result* r,const char* field) {
    size_t required=0;
    success(d26_api_result_string(r,field,nullptr,0,&required,message,sizeof message));
    std::vector<char> buffer(required);
    success(d26_api_result_string(r,field,buffer.data(),buffer.size(),&required,message,sizeof message));
    return buffer.data();
}
double phase_velocity(d26_api_solution* s) {
    d26_api_result* info=nullptr;
    success(d26_api_solution_info(s,&info,message,sizeof message));
    const auto c=numeric(info,"phase_velocity").at(0);
    d26_api_result_release(info);
    return c;
}
}

int main() {
    namespace api=disper26::api;
    {
        const double depths[]={0,20,60};
        const double fields[]={3,5.2,2.5,3.5,6,2.8,4.5,8,3.3};
        d26_api_model* model=nullptr;
        d26_api_dispersion_context* context=nullptr;
        success(d26_api_create_dispersion_context(&context,message,sizeof message));
        d26_api_dispersion_context_retain(context);
        d26_api_dispersion_context_release(context);
        success(d26_api_create_model(3,depths,"grid","isotropic",fields,3,"stiffness",nullptr,nullptr,
            &model,message,sizeof message));
        const double periods[]={8.4,8.,8.2,8.2};
        d26_api_solution* saved=nullptr;
        success(d26_api_compute_dispersion_with_context(model,8.,"Rayleigh",0,nullptr,nullptr,
            context,&saved,message,sizeof message));
        const double before=phase_velocity(saved);
        for(const char* wave:{"Love","Rayleigh"}) {
            d26_api_result *reused=nullptr,*fresh=nullptr;
            success(d26_api_compute_dispersion_curve_with_context(model,4,periods,wave,0,nullptr,nullptr,
                context,&reused,message,sizeof message));
            success(d26_api_compute_dispersion_curve(model,4,periods,wave,0,nullptr,nullptr,
                &fresh,message,sizeof message));
            check(numeric(reused,"phase_velocity")==numeric(fresh,"phase_velocity"),"Context C curve differs");
            d26_api_result_release(reused);d26_api_result_release(fresh);
        }
        d26_api_solution* failed=saved;
        check(d26_api_compute_dispersion_with_context(model,8.,"Rayleigh",0,nullptr,nullptr,
            nullptr,&failed,message,sizeof message)==D26_API_INVALID_ARGUMENT && !failed,"NULL context accepted");
        d26_api_dispersion_context_release(context);
        d26_api_dispersion_context_release(nullptr);
        check(phase_velocity(saved)==before,"Saved C result changed after context destruction");
        d26_api_solution_release(saved);d26_api_model_release(model);
        auto original=api::create_dispersion_context();
        auto moved=std::move(original);
        const auto hs=api::create_isotropic_parameters(4.5,8.,3.3);
        auto m=api::create_pancake_model({}, {},hs);
        bool rejected=false;
        try {api::compute_dispersion(m,8.,"Rayleigh",0,{"phase_velocity"},{},original);}
        catch(const api::Error& e) {rejected=e.code=="invalid_argument";}
        check(rejected,"Moved-from C++ context accepted");
        check(std::isfinite(api::compute_dispersion(m,8.,"Rayleigh",0,{"phase_velocity"},{},moved).phase_velocity()),
            "Moved-to context invalid");
    }
    {
        const double depths[]={0};
        const double fields[]={4.5,8.,3.3};
        d26_api_model* homogeneous=nullptr;
        success(d26_api_create_model(1,depths,"pancake","isotropic",fields,3,
            nullptr,nullptr,nullptr,&homogeneous,message,sizeof message));
        d26_api_solution* fundamental=nullptr;
        success(d26_api_compute_dispersion(homogeneous,10.,"Rayleigh",0,nullptr,nullptr,
            &fundamental,message,sizeof message));
        // Independent isotropic Rayleigh cubic, also checked across periods
        // and identical-layer splits by test_halfspace_mode_count.py.
        check(std::abs(phase_velocity(fundamental)-4.150909460643234)<1e-8,
              "Homogeneous halfspace mode-order search missed the Rayleigh root");
        d26_api_solution_release(fundamental);
        d26_api_options tracking_options;
        d26_api_default_options(&tracking_options);
        std::strcpy(tracking_options.method,"continuation");
        const double periods[]={10.,10.1};
        d26_api_result* tracked=nullptr;
        success(d26_api_compute_dispersion_curve(homogeneous,2,periods,"Rayleigh",0,nullptr,
            &tracking_options,&tracked,message,sizeof message));
        success(d26_api_curve_solution(tracked,1,&fundamental,message,sizeof message));
        d26_api_result* info=nullptr;
        success(d26_api_solution_info(fundamental,&info,message,sizeof message));
        check(numeric(info,"diagnostics.mode_verified")[0]==0,"C tracked point falsely verified");
        check(numeric(info,"diagnostics.mode_count_evaluations")[0]==0,"C tracked point counted modes");
        int explicit_count=-1;
        success(d26_api_compute_mode_count(fundamental,-1,&explicit_count,message,sizeof message));
        check(explicit_count==0,"C explicit mode count differs");
        check(d26_api_compute_mode_count(fundamental,99,&explicit_count,message,sizeof message)!=0 && explicit_count==-1,
              "C invalid saved level was accepted");
        check(d26_api_compute_mode_count(fundamental,0,nullptr,message,sizeof message)!=0,
              "C null count output was accepted");
        d26_api_result_release(info);
        d26_api_result_release(tracked);
        d26_api_solution_release(fundamental);
        std::strcpy(tracking_options.method,"continuation_recover");
        success(d26_api_compute_dispersion_curve(homogeneous,2,periods,"Rayleigh",0,nullptr,
            &tracking_options,&tracked,message,sizeof message));
        success(d26_api_curve_solution(tracked,0,&fundamental,message,sizeof message));
        success(d26_api_solution_info(fundamental,&info,message,sizeof message));
        check(numeric(info,"diagnostics.mode_verified")[0]==1,"C recovery initial root unverified");
        check(numeric(info,"diagnostics.mode_count_evaluations")[0]>0,"C recovery initial root not counted");
        d26_api_result_release(info);
        d26_api_solution_release(fundamental);
        d26_api_result_release(tracked);
        d26_api_model_release(homogeneous);
    }
    {
        const double depths[]={0,10};
        const double fields[]={3,5.2,2.5,4.5,8,3.3}; // 3 x 2 column-major
        api::MaterialArrays arrays;
        arrays.vs=std::vector<double>{3,4.5};arrays.vp=std::vector<double>{5.2,8};
        arrays.rho=std::vector<double>{2.5,3.3};
        auto cpp=api::create_model({0,10},"pancake",arrays);
        d26_api_model* owned=nullptr;
        success(d26_api_create_model(2,depths,"pancake","isotropic",fields,3,nullptr,nullptr,nullptr,
            &owned,message,sizeof message));
        d26_api_solution* s=nullptr;
        success(d26_api_compute_dispersion(owned,20,"Rayleigh",0,nullptr,nullptr,&s,message,sizeof message));
        check(phase_velocity(s)==api::compute_dispersion(cpp,20).phase_velocity(),"Array C/C++ models differ");
        d26_api_solution_release(s);d26_api_model_release(owned);
        const double vti[]={2.5,5.3,5.2,3.1,3.0,.9,3.3,8.2,8.,4.6,4.5,.95};
        const auto top=api::create_velocity_vti_parameters(2.5,5.3,5.2,3.1,3.,.9);
        const auto bottom=api::create_velocity_vti_parameters(3.3,8.2,8.,4.6,4.5,.95);
        const auto reference=api::create_pancake_model({top},{10},bottom);
        const double stiffness[]={top.A,top.C,top.F,top.L,top.N,top.rho,
            bottom.A,bottom.C,bottom.F,bottom.L,bottom.N,bottom.rho};
        for(const char* basis:{"velocity_vti","stiffness"}) {
            success(d26_api_create_model(2,depths,"pancake",basis,
                std::string(basis)=="stiffness"?stiffness:vti,6,nullptr,nullptr,nullptr,&owned,message,sizeof message));
            success(d26_api_compute_dispersion(owned,20,"Rayleigh",0,nullptr,nullptr,&s,message,sizeof message));
            check(phase_velocity(s)==api::compute_dispersion(reference,20).phase_velocity(),"C array component order differs");
            d26_api_solution_release(s);d26_api_model_release(owned);
        }
        for(const char* interpolation:{"stiffness","velocity"}) {
            success(d26_api_create_model(2,depths,"grid","isotropic",fields,3,interpolation,nullptr,nullptr,
                &owned,message,sizeof message));
            d26_api_model_release(owned);
        }
        const double repeated[]={0,0};
        check(d26_api_create_model(2,repeated,"pancake","isotropic",fields,3,nullptr,nullptr,nullptr,
            &owned,message,sizeof message)==D26_API_INVALID_ARGUMENT && !owned,"Duplicate Pancake knot accepted");
        check(d26_api_create_model(2,depths,"pancake","isotropic",fields,6,nullptr,nullptr,nullptr,
            &owned,message,sizeof message)==D26_API_INVALID_ARGUMENT && !owned,"Wrong component count accepted");
        check(d26_api_create_model(2,depths,"grid","isotropic",fields,3,nullptr,nullptr,nullptr,
            &owned,message,sizeof message)==D26_API_INVALID_ARGUMENT && !owned,"Missing interpolation accepted");
        arrays.A=std::vector<double>{1,2};
        bool rejected=false;
        try {(void)api::create_model({0,10},"pancake",arrays);}catch(const api::Error&){rejected=true;}
        check(rejected,"Mixed C++ array representations accepted");
    }
    d26_api_geometry geo; d26_api_default_geometry(&geo);
    d26_api_options options; d26_api_default_options(&options);
    d26_api_material layer,halfspace;
    success(d26_api_create_isotropic_parameters(3,5.2,2.5,&layer,message,sizeof message));
    success(d26_api_create_isotropic_parameters(4.5,8,3.3,&halfspace,message,sizeof message));
    double thickness=10;
    d26_api_model* model=nullptr;
    success(d26_api_create_pancake_model(1,&layer,&thickness,&halfspace,&geo,"native test",&model,message,sizeof message));
    // Input ownership is independent of caller buffers.
    const double original_A=layer.A;
    layer.A=0;
    thickness=0;
    d26_api_result* info=nullptr;
    success(d26_api_model_info(model,&info,message,sizeof message));
    check(numeric(info,"materials").at(1)==original_A,"Material input was not copied");
    check(numeric(info,"thicknesses").at(0)==10,"Thickness input was not copied");
    check(text(info,"name")=="native test","Model metadata missing");
    d26_api_result_release(info);
    d26_api_solution* solution=nullptr;
    success(d26_api_compute_dispersion(model,20,"Rayleigh",0,"phase_velocity,group_velocity,hv_ratio",&options,&solution,message,sizeof message));
    d26_api_model_retain(model);
    d26_api_model_release(model);
    d26_api_model_release(model);
    model=nullptr;
    d26_api_solution_retain(solution);
    d26_api_solution_release(solution);
    // Mixed C and C++ use must link to the same core without duplicate symbols.
    const auto reference_model=api::create_pancake_model({api::create_isotropic_parameters(3,5.2,2.5)},
        {10},api::create_isotropic_parameters(4.5,8,3.3));
    const auto reference=api::compute_dispersion(reference_model,20,"Rayleigh",0,
        {"phase_velocity","group_velocity","hv_ratio"});
    // Same numerical operation and copied doubles: serialization needs no tolerance.
    check(phase_velocity(solution)==reference.phase_velocity(),"C and C++ phase velocity differ");
    success(d26_api_solution_info(solution,&info,message,sizeof message));
    check(numeric(info,"group_velocity").at(0)==*reference.group_velocity(),"Group velocity differs");
    check(numeric(info,"hv_ratio").at(0)==*reference.hv_ratio(),"H/V differs");
    check(text(info,"wave")=="Rayleigh","Wave metadata lost");
    check(numeric(info,"mode").at(0)==0,"Mode must stay zero-based");
    // Query sizes, preserve small caller buffers, and keep errors per call.
    size_t required=0,rows=0,cols=0;
    char tiny[2]={'x','y'};
    int code=d26_api_result_string(info,"wave",tiny,sizeof tiny,&required,message,sizeof message);
    check(code==D26_API_BUFFER_TOO_SMALL && required==9 && tiny[0]=='x' && tiny[1]=='y',"String size protocol broken");
    char other_message[64];
    std::string previous_message(message);
    code=d26_api_result_numeric(info,"missing",nullptr,0,&rows,&cols,other_message,sizeof other_message);
    check(code==D26_API_INVALID_ARGUMENT && rows==0 && cols==0,"Unknown field must not appear as zero");
    check(previous_message==message,"Error message must be per call");
    size_t count=0;
    success(d26_api_result_field_count(info,1,&count,message,sizeof message));
    check(count>3,"Field enumeration missing");
    success(d26_api_result_field_name(info,1,0,nullptr,0,&required,message,sizeof message));
    check(required>1,"Field name missing");
    d26_api_result_retain(info);
    d26_api_result_release(info);
    check(numeric(info,"period").at(0)==20,"Retained result invalid");
    d26_api_result_release(info);
    success(d26_api_solution_model(solution,&model,message,sizeof message));
    double periods[]={20,10,20};
    d26_api_result* curve=nullptr;
    success(d26_api_compute_dispersion_curve(model,3,periods,"Rayleigh",0,nullptr,&options,&curve,message,sizeof message));
    check(numeric(curve,"periods")==std::vector<double>({20,10,20}),"Curve order or duplicates lost");
    d26_api_solution* selected=nullptr;
    success(d26_api_curve_solution(curve,2,&selected,message,sizeof message));
    d26_api_result_release(curve);
    // Continuation changes the root bracket; compare within the configured root tolerance.
    check(std::abs(phase_velocity(selected)-reference.phase_velocity())<=
        4*options.root_tolerance*std::max(1.0,std::abs(reference.phase_velocity())),"Curve root differs beyond configured tolerance");
    d26_api_solution_release(selected);
    // Unrequested scalar is absent, with availability metadata.
    success(d26_api_compute_dispersion(model,20,"Rayleigh",0,nullptr,nullptr,&selected,message,sizeof message));
    success(d26_api_solution_info(selected,&info,message,sizeof message));
    code=d26_api_result_numeric(info,"group_velocity",nullptr,0,&rows,&cols,message,sizeof message);
    check(code==D26_API_INVALID_ARGUMENT,"Unrequested quantity must not be a computed zero");
    d26_api_result_release(info);
    d26_api_solution_release(selected);
    success(d26_api_compute_kernels(solution,"isotropic",nullptr,nullptr,&info,message,sizeof message));
    const auto reference_kernels=api::compute_kernels(reference,"isotropic");
    check(numeric(info,"gradient.phase_velocity")==reference_kernels.gradients.at("phase_velocity"),"Gradient differs from common implementation");
    check(text(info,"basis")=="isotropic","Derivative basis missing");
    check(text(info,"parameter.0.name")==reference_kernels.parameters.at(0).name,"Derivative axis missing");
    check(numeric(info,"parameter.active").size()==reference_kernels.parameters.size(),"Active mask missing");
    d26_api_result_release(info);
    success(d26_api_compute_eigenfunctions(solution,4,&info,message,sizeof message));
    const auto reference_eigen=api::compute_eigenfunctions(reference,4);
    check(numeric(info,"depth")==reference_eigen.depth,"Eigenfunction depths differ");
    for(const auto& component:reference_eigen.components) {
        const auto re=numeric(info,(component.first+"_real").c_str());
        const auto im=numeric(info,(component.first+"_imag").c_str());
        for(size_t i=0;i<re.size();++i)
            check(std::complex<double>(re[i],im[i])==component.second[i],"Complex component transfer differs");
    }
    d26_api_result_release(info);
    success(d26_api_compute_energy_integrals(solution,nullptr,&info,message,sizeof message));
    check(std::isfinite(numeric(info,"I1").at(0)),"Energy integral nonfinite");
    d26_api_result_release(info);
    success(d26_api_compute_frequency_curvature(solution,nullptr,&info,message,sizeof message));
    check(std::isfinite(numeric(info,"group_velocity_omega_derivative").at(0)),"Frequency curvature nonfinite");
    d26_api_result_release(info);
    success(d26_api_compute_interface_kernels(solution,nullptr,&info,message,sizeof message));
    check(!numeric(info,"gradient.phase_velocity").empty(),"Interface kernel missing");
    d26_api_result_release(info);
    success(d26_api_compute_gravity_correction(solution,9.80665e-3,"solid",4,5e-3,&info,message,sizeof message));
    check(std::isfinite(numeric(info,"delta_c").at(0)),"Gravity correction nonfinite");
    check(text(info,"region")=="solid","Gravity region metadata missing");
    d26_api_result_release(info);
    double qk[]={1000,1000},qm[]={600,600};
    d26_api_model* corrected=nullptr;
    success(d26_api_apply_mineos_physical_dispersion(model,2,qk,qm,1,20,&corrected,message,sizeof message));
    success(d26_api_model_info(corrected,&info,message,sizeof message));
    check(numeric(info,"q_period").at(0)==20 && numeric(info,"q_reference_period").at(0)==1,"Q period metadata missing");
    d26_api_result_release(info);
    d26_api_model_release(corrected);
    success(d26_api_get_capabilities(model,"Rayleigh",&info,message,sizeof message));
    check(!numeric(info,"supported").empty(),"Capabilities missing");
    d26_api_result_release(info);
    d26_api_solution_release(solution);
    d26_api_model_release(model);
    // Distinct fields make a 9-by-2 matrix transpose visible.
    success(d26_api_create_isotropic_parameters(3,5.2,2.5,&layer,message,sizeof message));
    d26_api_material nodes[]={layer,halfspace}; double depths[]={0,10};
    success(d26_api_create_grid_model(2,depths,nodes,"stiffness",nullptr,nullptr,&model,message,sizeof message));
    success(d26_api_model_info(model,&info,message,sizeof message));
    const auto matrix=numeric(info,"materials",&rows,&cols);
    check(rows==9 && cols==2 && matrix[0]==2.5 && matrix[9]==3.3 && matrix[10]==halfspace.A,"Nonsquare column-major matrix layout wrong");
    check(matrix[15]==halfspace.gravity && matrix[16]==halfspace.gravity_internal_scale &&
          matrix[17]==halfspace.gravity_surface_scale,"Material gravity field order wrong");
    const auto halfspace_values=numeric(info,"halfspace");
    check(halfspace_values.size()==9 && halfspace_values[8]==halfspace.gravity_surface_scale,
          "Halfspace material layout wrong");
    double sentinel[2]={123,456};
    code=d26_api_result_numeric(info,"materials",sentinel,2,&rows,&cols,message,sizeof message);
    check(code==D26_API_BUFFER_TOO_SMALL && sentinel[0]==123 && sentinel[1]==456 && rows==9 && cols==2,"Small numeric buffer modified");
    d26_api_result_release(info);
    options.max_steps=3;
    options.max_levels=2;
    success(d26_api_compute_dispersion(model,20,"Rayleigh",0,nullptr,&options,&solution,message,sizeof message));
    std::vector<double> projection(24,0.0);
    projection[0]=1; projection[5]=0.01; projection[6]=0.2;
    projection[12+3]=1; projection[12+9]=0.4;
    success(d26_api_compute_hessian_reduced(solution,projection.data(),12,2,&options,&info,message,sizeof message));
    const auto grid_reference=api::create_grid_model({0,10},
        {api::create_isotropic_parameters(3,5.2,2.5),api::create_isotropic_parameters(4.5,8,3.3)},"stiffness");
    api::Options cpp_options; cpp_options.max_steps=3; cpp_options.max_levels=2;
    const auto grid_solution=api::compute_dispersion(grid_reference,20,"Rayleigh",0,{"phase_velocity"},cpp_options);
    const auto reduced=api::compute_hessian_reduced(grid_solution,projection,2,cpp_options);
    check(numeric(info,"hessian",&rows,&cols)==reduced.hessian && rows==2 && cols==2,
        "Nonsquare projection differs from C++ reduced Hessian");
    d26_api_result_release(info);
    d26_api_solution_release(solution);
    d26_api_model_release(model);
    // Both explicit spherical policies round-trip and flatten exactly once.
    for(int policy:{D26_API_SPHERICAL_BA97,D26_API_SPHERICAL_EXPERIMENTAL_DENSITY}) {
        geo.kind=policy;
        geo.flattening_exponent=-2.275;
        success(d26_api_create_grid_model(2,depths,nodes,"stiffness",&geo,nullptr,&model,message,sizeof message));
        success(d26_api_earth_flattening_transformation(model,&corrected,message,sizeof message));
        success(d26_api_model_info(corrected,&info,message,sizeof message));
        check(numeric(info,"flattened").at(0)==1 && text(info,"geometry")=="flat","EFT metadata missing");
        check(text(info,"flattening_policy")==
            (policy==D26_API_SPHERICAL_BA97 ? "spherical_ba97" : "spherical_experimental_density"),"EFT policy changed");
        d26_api_result_release(info);
        d26_api_model* invalid=nullptr;
        code=d26_api_earth_flattening_transformation(corrected,&invalid,message,sizeof message);
        check(code==D26_API_INVALID_ARGUMENT && !invalid,"Double EFT must be rejected");
        d26_api_model_release(corrected);
        d26_api_model_release(model);
    }
    const double interface_depths[]={0,10,10,20};
    const d26_api_material interface_nodes[]={layer,layer,halfspace,halfspace};
    success(d26_api_create_grid_model(4,interface_depths,interface_nodes,"stiffness",nullptr,nullptr,&model,message,sizeof message));
    success(d26_api_compute_dispersion(model,20,"Rayleigh",0,nullptr,&options,&solution,message,sizeof message));
    success(d26_api_compute_interface_kernels(solution,&options,&info,message,sizeof message));
    check(numeric(info,"gradient.phase_velocity").size()==1,"GRID interface count incorrect");
    d26_api_result_release(info);
    success(d26_api_compute_interface_hessian_row(solution,0,"stiffness",&options,&info,message,sizeof message));
    const auto interface_row=numeric(info,"hessian",&rows,&cols);
    check(rows==1 && cols==25 && interface_row.size()==25,"Interface Hessian row dimensions incorrect");
    d26_api_result_release(info);
    d26_api_solution_release(solution);
    d26_api_model_release(model);
    model=nullptr;
    code=d26_api_create_grid_model(2,nullptr,nodes,"stiffness",nullptr,nullptr,&model,message,sizeof message);
    check(code==D26_API_INVALID_ARGUMENT && !model,"Invalid input must leave NULL output");
    code=d26_api_solution_info(nullptr,&info,message,sizeof message);
    check(code==D26_API_INVALID_ARGUMENT && !info,"NULL solution should fail safely");
    d26_api_model_release(nullptr); d26_api_solution_release(nullptr); d26_api_result_release(nullptr);
    std::cout << "Common C API ownership, fields, array layout and C++ parity passed\n";
}
