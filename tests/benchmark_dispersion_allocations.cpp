// Allocation counts are separate from timing: the instrumentation is intrusive.
// Counts operator new traffic, not third-party calls directly to malloc.
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include "api_impl.hpp"

#ifndef DISPER26_CONTEXT_AVAILABLE
#define DISPER26_CONTEXT_AVAILABLE 0
#endif
namespace {
std::atomic<bool> counting{false};
std::atomic<size_t> allocations{0}, bytes{0};
}
void* operator new(std::size_t n) {
    void* p=std::malloc(n ? n : 1);
    if(!p) throw std::bad_alloc();
    if(counting) {++allocations;bytes+=n;}
    return p;
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}

int main() {
    using namespace disper26::api;
    const auto a=create_isotropic_parameters(3.,5.2,2.5);
    const auto b=create_isotropic_parameters(3.5,6.,2.8);
    const auto h=create_isotropic_parameters(4.5,8.,3.3);
    const std::vector<Model> models={create_pancake_model({a,b},{20.,40.},h),
        create_grid_model({0.,20.,60.},{a,b,h},"stiffness"),
        create_grid_model({0.,20.,60.},{a,b,h},"velocity")};
    Options o; o.max_steps=4;o.tolerance=1e-8;
    for(size_t m=0;m<models.size();++m) for(auto wave:{"Love","Rayleigh"}) {
#if DISPER26_CONTEXT_AVAILABLE
        auto context=create_dispersion_context();
#endif
        for(int reuse=0;reuse<=DISPER26_CONTEXT_AVAILABLE;++reuse) {
            auto call=[&]() {
#if DISPER26_CONTEXT_AVAILABLE
                if(reuse) return compute_dispersion_curve(models[m],{8.,8.2,8.4},wave,0,{"phase_velocity"},o,context);
#endif
                return compute_dispersion_curve(models[m],{8.,8.2,8.4},wave,0,{"phase_velocity"},o);
            };
            call();
            allocations=0;bytes=0;counting=true;
            const auto result=call();
            counting=false;
            size_t retained=0;
#if DISPER26_CONTEXT_AVAILABLE
            if(reuse) {
                const auto& scratch=Access::context(context);
                retained=sizeof(scratch);
                for(const auto& mesh:scratch.meshes)
                    retained+=mesh.layers.capacity()*sizeof(PhysicalParameters<double>)+mesh.thicknesses.capacity()*sizeof(double);
                retained+=scratch.geometry.capacity()*sizeof(GridSublayerGeometry)+scratch.fluids.capacity()/8;
                const auto& f=scratch.fields;
                retained+=(f.rho.capacity()+f.ph.capacity()+f.pv.capacity()+f.sh.capacity()+f.sv.capacity()+f.eta.capacity())*sizeof(double);
                const auto& r=scratch.romberg;
                retained+=(r.c.capacity()+r.u.capacity()+r.hv.capacity()+r.roots.capacity())*sizeof(double);
            }
#endif
            std::cout<<m<<' '<<wave<<' '<<reuse<<' '<<allocations<<' '<<bytes<<' '<<retained<<'\n';
            if(!std::isfinite(result.solutions[0].phase_velocity())) return 1;
        }
    }
}
