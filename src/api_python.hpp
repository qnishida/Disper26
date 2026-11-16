#pragma once

#include "api.hpp"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace disper26 { namespace api { namespace python {
namespace py = pybind11;

// Each returned array owns its storage: model snapshots and cached solutions
// cannot be changed by writing to a NumPy view.
template <class T> py::array_t<T> array_copy(const std::vector<T>& values) {
    py::array_t<T> result(values.size());
    auto out = result.template mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < out.shape(0); ++i) out(i) = values[i];
    return result;
}
inline py::dict gradients_copy(const DerivativeResult& value) {
    py::dict result;
    for (const auto& entry : value.gradients)
        result[py::str(entry.first)] = array_copy(entry.second);
    return result;
}
inline py::object hessian_copy(const DerivativeResult& value) {
    if (value.hessian.empty()) return py::none();
    py::array_t<double> result({value.rows, value.cols});
    auto out = result.mutable_unchecked<2>();
    for (int j = 0; j < value.cols; ++j)
        for (int i = 0; i < value.rows; ++i)
            out(i, j) = value.hessian[i + value.rows * j];
    return std::move(result);
}
inline Options options_from_kwargs(const py::kwargs& values) {
    Options o;
    for (auto entry : values) {
        const auto key = py::cast<std::string>(entry.first);
#define DISPER26_OPTION(field) if (key == #field) { o.field = py::cast<decltype(o.field)>(entry.second); continue; }
        DISPER26_OPTION(max_steps) DISPER26_OPTION(max_levels)
        DISPER26_OPTION(num_sub_layers) DISPER26_OPTION(tolerance)
        DISPER26_OPTION(root_tolerance) DISPER26_OPTION(derivative_tolerance)
        DISPER26_OPTION(c_guess) DISPER26_OPTION(method)
        DISPER26_OPTION(predictor) DISPER26_OPTION(verify_mode)
        DISPER26_OPTION(start) DISPER26_OPTION(c_range)
        DISPER26_OPTION(verify) DISPER26_OPTION(on_violation)
        DISPER26_OPTION(backend) DISPER26_OPTION(integration) DISPER26_OPTION(strict)
#undef DISPER26_OPTION
        throw py::type_error("Unknown Options keyword: " + key);
    }
    return o;
}

inline void register_api(py::module_& parent) {
    auto m = parent.def_submodule("_api", "Shared C++ public API; use the disper26 package.");
    py::register_exception<Error>(m, "Error", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr failure) {
        try {
            if (failure) std::rethrow_exception(failure);
        } catch (const Error& error) {
            auto type = py::module_::import("disper26py").attr("_api").attr("Error");
            auto instance = type(error.what());
            instance.attr("code") = error.code;
            PyErr_SetObject(type.ptr(), instance.ptr());
        }
    });
    py::class_<Material>(m, "Material")
        .def(py::init([](double rho, double A, double C, double F, double L, double N,
                double gravity, double internal, double surface) {
            return Material{rho, A, C, F, L, N, gravity, internal, surface};
        }), py::arg("rho"), py::arg("A"), py::arg("C"), py::arg("F"),
            py::arg("L"), py::arg("N"), py::arg("gravity") = 0.,
            py::arg("gravity_internal_scale") = 1., py::arg("gravity_surface_scale") = 0.)
        .def_readonly("rho", &Material::rho).def_readonly("A", &Material::A)
        .def_readonly("C", &Material::C).def_readonly("F", &Material::F)
        .def_readonly("L", &Material::L).def_readonly("N", &Material::N)
        .def_readonly("gravity", &Material::gravity)
        .def_readonly("gravity_internal_scale", &Material::gravity_internal_scale)
        .def_readonly("gravity_surface_scale", &Material::gravity_surface_scale);
    py::class_<Geometry>(m, "Geometry")
        .def(py::init([](const std::string& kind, double radius, double exponent) {
            return Geometry{kind, radius, exponent};
        }), py::arg("kind") = "flat", py::arg("planet_radius") = 6371.,
            py::arg("flattening_exponent") = missing_value)
        .def_readonly("kind", &Geometry::kind)
        .def_readonly("planet_radius", &Geometry::planet_radius)
        .def_readonly("flattening_exponent", &Geometry::flattening_exponent);
    py::class_<Options>(m, "Options")
        .def(py::init(&options_from_kwargs))
#define DISPER26_OPTION(field) .def_readwrite(#field, &Options::field)
        DISPER26_OPTION(max_steps) DISPER26_OPTION(max_levels)
        DISPER26_OPTION(num_sub_layers) DISPER26_OPTION(tolerance)
        DISPER26_OPTION(root_tolerance) DISPER26_OPTION(derivative_tolerance)
        DISPER26_OPTION(c_guess) DISPER26_OPTION(method)
        DISPER26_OPTION(predictor) DISPER26_OPTION(verify_mode)
        DISPER26_OPTION(start) DISPER26_OPTION(c_range)
        DISPER26_OPTION(verify) DISPER26_OPTION(on_violation)
        DISPER26_OPTION(backend) DISPER26_OPTION(integration) DISPER26_OPTION(strict)
#undef DISPER26_OPTION
        ;
    py::class_<Model>(m, "Model")
        .def_property_readonly("kind", [](const Model& x) { return x.data().kind; })
        .def_property_readonly("interpolation", [](const Model& x) { return x.data().interpolation; })
        .def_property_readonly("name", [](const Model& x) { return x.data().name; })
        .def_property_readonly("materials", [](const Model& x) { return py::tuple(py::cast(x.data().materials)); })
        .def_property_readonly("depths", [](const Model& x) { return array_copy(x.data().depths); })
        .def_property_readonly("thicknesses", [](const Model& x) { return array_copy(x.data().thicknesses); })
        .def_property_readonly("discontinuities", [](const Model& x) { return array_copy(x.data().discontinuities); })
        .def_property_readonly("halfspace", [](const Model& x) { return x.data().halfspace; })
        .def_property_readonly("geometry", [](const Model& x) { return x.data().geometry; })
        .def_property_readonly("flattened", [](const Model& x) { return x.data().flattened; })
        .def_property_readonly("flattening_policy", [](const Model& x) { return x.data().flattening_policy; })
        .def_property_readonly("q_period", [](const Model& x) { return x.data().q_period; })
        .def_property_readonly("q_reference_period", [](const Model& x) { return x.data().q_reference_period; });
    py::class_<Diagnostics>(m, "Diagnostics")
#define DISPER26_FIELD(field) .def_readonly(#field, &Diagnostics::field)
        DISPER26_FIELD(status) DISPER26_FIELD(convergence)
        DISPER26_FIELD(missing_reason)
        DISPER26_FIELD(method) DISPER26_FIELD(backend) DISPER26_FIELD(messages)
        DISPER26_FIELD(availability) DISPER26_FIELD(convergence_by_quantity) DISPER26_FIELD(accepted_level)
        DISPER26_FIELD(generated_level) DISPER26_FIELD(fallback_count)
        DISPER26_FIELD(relative_error) DISPER26_FIELD(mode_verified)
        DISPER26_FIELD(level_mode_verified) DISPER26_FIELD(predictors)
        DISPER26_FIELD(fallback_reasons) DISPER26_FIELD(mode_count_evaluations)
        DISPER26_FIELD(predictor_ad_evaluations)
#undef DISPER26_FIELD
        ;
    py::class_<ModeSolution>(m, "ModeSolution")
        .def_property_readonly("phase_velocity", &ModeSolution::phase_velocity)
        .def_property_readonly("group_velocity", &ModeSolution::group_velocity)
        .def_property_readonly("hv_ratio", &ModeSolution::hv_ratio)
        .def_property_readonly("period", &ModeSolution::period)
        .def_property_readonly("wave", &ModeSolution::wave)
        .def_property_readonly("mode", &ModeSolution::mode)
        .def_property_readonly("model", &ModeSolution::model, py::return_value_policy::copy)
        .def_property_readonly("options", &ModeSolution::options, py::return_value_policy::copy)
        .def_property_readonly("diagnostics", &ModeSolution::diagnostics, py::return_value_policy::copy);
    m.def("compute_mode_count", &compute_mode_count, py::arg("solution"), py::arg("level") = -1);
    py::class_<DispersionCurve>(m, "DispersionCurve")
        .def_property_readonly("periods", [](const DispersionCurve& x) { return array_copy(x.periods); })
        .def_property_readonly("solutions", [](const DispersionCurve& x) { return py::tuple(py::cast(x.solutions)); })
        .def_property_readonly("phase_velocity", [](const DispersionCurve& x) {
            std::vector<double> values;
            for (const auto& s : x.solutions) values.push_back(s.phase_velocity());
            return array_copy(values);
        })
        .def_property_readonly("group_velocity", [](const DispersionCurve& x) -> py::object {
            if (x.solutions.empty() || !x.solutions.front().group_velocity()) return py::none();
            std::vector<double> values;
            for (const auto& s : x.solutions) values.push_back(s.group_velocity().value_or(missing_value));
            return array_copy(values);
        })
        .def_property_readonly("hv_ratio", [](const DispersionCurve& x) -> py::object {
            if (x.solutions.empty() || !x.solutions.front().hv_ratio()) return py::none();
            std::vector<double> values;
            for (const auto& s : x.solutions) values.push_back(s.hv_ratio().value_or(missing_value));
            return array_copy(values);
        });
    py::class_<Parameter>(m, "Parameter")
        .def_readonly("name", &Parameter::name).def_readonly("entity", &Parameter::entity)
        .def_readonly("unit", &Parameter::unit).def_readonly("index", &Parameter::index)
        .def_readonly("active", &Parameter::active);
    py::class_<DerivativeResult>(m, "DerivativeResult")
        .def_readonly("basis", &DerivativeResult::basis)
        .def_property_readonly("parameters", [](const DerivativeResult& x) { return py::tuple(py::cast(x.parameters)); })
        .def_property_readonly("gradients", &gradients_copy)
        .def_property_readonly("gradient", [](const DerivativeResult& x) -> py::object {
            auto it = x.gradients.find("phase_velocity");
            return it == x.gradients.end() ? py::object(py::none()) : py::object(array_copy(it->second));
        })
        .def_property_readonly("hessian", &hessian_copy)
        .def_readonly("diagnostics", &DerivativeResult::diagnostics);
    py::class_<Eigenfunctions>(m, "Eigenfunctions")
        .def_property_readonly("depth", [](const Eigenfunctions& x) { return array_copy(x.depth); })
        .def_property_readonly("is_fluid", [](const Eigenfunctions& x) { return array_copy(x.is_fluid); })
        .def_property_readonly("components", [](const Eigenfunctions& x) {
            py::dict result;
            for (const auto& entry : x.components) result[py::str(entry.first)] = array_copy(entry.second);
            return result;
        })
        .def_readonly("diagnostics", &Eigenfunctions::diagnostics);
    py::class_<EnergyIntegrals>(m, "EnergyIntegrals")
        .def_readonly("I1", &EnergyIntegrals::I1).def_readonly("I2", &EnergyIntegrals::I2)
        .def_readonly("I3", &EnergyIntegrals::I3)
        .def_readonly("phase_velocity", &EnergyIntegrals::phase_velocity)
        .def_readonly("group_velocity", &EnergyIntegrals::group_velocity)
        .def_readonly("diagnostics", &EnergyIntegrals::diagnostics);
    py::class_<FrequencyCurvature>(m, "FrequencyCurvature")
        .def_readonly("phase_velocity", &FrequencyCurvature::phase_velocity)
        .def_readonly("group_velocity", &FrequencyCurvature::group_velocity)
        .def_readonly("group_velocity_omega_derivative", &FrequencyCurvature::group_velocity_omega_derivative)
        .def_readonly("wavenumber_omega_second_derivative", &FrequencyCurvature::wavenumber_omega_second_derivative)
        .def_readonly("diagnostics", &FrequencyCurvature::diagnostics);
    py::class_<Capability>(m, "Capability")
        .def_readonly("operation", &Capability::operation)
        .def_readonly("basis", &Capability::basis).def_readonly("method", &Capability::method)
        .def_readonly("reason", &Capability::reason).def_readonly("supported", &Capability::supported);
    py::class_<GravityCorrection>(m, "GravityCorrection")
        .def_readonly("phase_velocity", &GravityCorrection::phase_velocity)
        .def_readonly("group_velocity", &GravityCorrection::group_velocity)
        .def_readonly("delta_c", &GravityCorrection::delta_c)
        .def_readonly("delta_u", &GravityCorrection::delta_u)
        .def_readonly("omega_delta_c_omega_derivative", &GravityCorrection::omega_delta_c_omega_derivative)
        .def_readonly("region", &GravityCorrection::region)
        .def_readonly("diagnostics", &GravityCorrection::diagnostics);
    m.def("create_isotropic_parameters", &create_isotropic_parameters, py::arg("vs"), py::arg("vp"), py::arg("rho"));
    m.def("create_model", [](const std::vector<double>& depths,const std::string& kind,
        const py::dict& fields,const std::optional<std::string>& interpolation,
        const Geometry& geometry,const std::string& name) {
        MaterialArrays p;
        for(auto entry:fields) {
            const auto key=py::cast<std::string>(entry.first);
#define D26_ARRAY(field) if(key==#field) {p.field=py::cast<std::vector<double>>(entry.second);continue;}
            D26_ARRAY(vs) D26_ARRAY(vp) D26_ARRAY(rho)
            D26_ARRAY(vph) D26_ARRAY(vpv) D26_ARRAY(vsh) D26_ARRAY(vsv) D26_ARRAY(eta)
            D26_ARRAY(A) D26_ARRAY(C) D26_ARRAY(F) D26_ARRAY(L) D26_ARRAY(N)
#undef D26_ARRAY
            throw py::type_error("Unknown material array: "+key);
        }
        return create_model(depths,kind,p,interpolation,geometry,name);
    },py::arg("depths"),py::arg("kind"),py::arg("fields"),py::arg("interpolation"),
        py::arg("geometry"),py::arg("name"));
    m.def("create_fluid_parameters", &create_fluid_parameters, py::arg("vp"), py::arg("rho"));
    m.def("create_velocity_vti_parameters", &create_velocity_vti_parameters,
        py::arg("rho"), py::arg("vph"), py::arg("vpv"), py::arg("vsh"), py::arg("vsv"), py::arg("eta"));
    m.def("create_layer_cake_model", &create_layer_cake_model, py::arg("layers"), py::arg("thicknesses"),
        py::arg("halfspace"), py::arg("geometry") = Geometry{}, py::arg("name") = "");
    m.def("create_pancake_model", &create_pancake_model, py::arg("layers"), py::arg("thicknesses"),
        py::arg("halfspace"), py::arg("geometry") = Geometry{}, py::arg("name") = "", "Deprecated alias. Use create_layer_cake_model; preserves kind=pancake.");
    m.def("create_grid_model", &create_grid_model, py::arg("depths"), py::arg("materials"),
        py::arg("interpolation"), py::arg("geometry") = Geometry{}, py::arg("name") = "");
    py::class_<DispersionContext>(m,"DispersionContext");
    m.def("create_dispersion_context",&create_dispersion_context);
    // Keep the existing conversion path when no context is supplied.
    m.def("compute_dispersion", py::overload_cast<const Model&,double,const std::string&,
        int,const std::vector<std::string>&,const Options&>(&compute_dispersion),
        py::arg("model"),py::arg("period"),py::arg("wave")="Rayleigh",py::arg("mode")=0,
        py::arg("outputs")=std::vector<std::string>{"phase_velocity"},py::arg("options")=Options{},
        py::call_guard<py::gil_scoped_release>());
    m.def("compute_dispersion_curve", py::overload_cast<const Model&,const std::vector<double>&,
        const std::string&,int,const std::vector<std::string>&,const Options&>(&compute_dispersion_curve),
        py::arg("model"),py::arg("periods"),py::arg("wave")="Rayleigh",py::arg("mode")=0,
        py::arg("outputs")=std::vector<std::string>{"phase_velocity"},py::arg("options")=Options{},
        py::call_guard<py::gil_scoped_release>());
    m.def("compute_dispersion", [](const Model& model,double period,const std::string& wave,
        int mode,const std::vector<std::string>& outputs,const Options& options,DispersionContext* context) {
        return context ? compute_dispersion(model,period,wave,mode,outputs,options,*context)
                       : compute_dispersion(model,period,wave,mode,outputs,options);
    },py::arg("model"),py::arg("period"),py::arg("wave")="Rayleigh",py::arg("mode")=0,
        py::arg("outputs")=std::vector<std::string>{"phase_velocity"},py::arg("options")=Options{},
        py::arg("context")=nullptr,py::call_guard<py::gil_scoped_release>());
    m.def("compute_dispersion_curve", [](const Model& model,const std::vector<double>& periods,
        const std::string& wave,int mode,const std::vector<std::string>& outputs,
        const Options& options,DispersionContext* context) {
        return context ? compute_dispersion_curve(model,periods,wave,mode,outputs,options,*context)
                       : compute_dispersion_curve(model,periods,wave,mode,outputs,options);
    },py::arg("model"),py::arg("periods"),py::arg("wave")="Rayleigh",py::arg("mode")=0,
        py::arg("outputs")=std::vector<std::string>{"phase_velocity"},py::arg("options")=Options{},
        py::arg("context")=nullptr,py::call_guard<py::gil_scoped_release>());
    m.def("compute_kernels", &compute_kernels, py::arg("solution"), py::arg("basis"),
        py::arg("outputs") = std::vector<std::string>{"phase_velocity"}, py::arg("options") = Options{},
        py::call_guard<py::gil_scoped_release>());
    m.def("compute_hessian", &compute_hessian, py::arg("solution"), py::arg("basis"),
        py::arg("options") = Options{}, py::call_guard<py::gil_scoped_release>());
    m.def("compute_hessian_reduced", [](const ModeSolution& s,
            py::array_t<double, py::array::forcecast> projection, const Options& options) {
        if (projection.ndim() != 2) throw py::value_error("projection must be a 2D stiffness-axis by reduced-axis matrix");
        auto p = projection.unchecked<2>();
        std::vector<double> values(p.shape(0) * p.shape(1));
        for (py::ssize_t j = 0; j < p.shape(1); ++j)
            for (py::ssize_t i = 0; i < p.shape(0); ++i) values[i + p.shape(0) * j] = p(i, j);
        py::gil_scoped_release release;
        return compute_hessian_reduced(s, values, static_cast<int>(p.shape(1)), options);
    }, py::arg("solution"), py::arg("projection"), py::arg("options") = Options{});
    m.def("compute_interface_kernels", &compute_interface_kernels, py::arg("solution"),
        py::arg("options") = Options{}, py::call_guard<py::gil_scoped_release>());
    m.def("compute_interface_hessian_row", &compute_interface_hessian_row, py::arg("solution"),
        py::arg("interface_index"), py::arg("basis"), py::arg("options") = Options{},
        py::call_guard<py::gil_scoped_release>());
    m.def("compute_eigenfunctions", &compute_eigenfunctions, py::arg("solution"),
        py::arg("num_sub_layers") = 10, py::call_guard<py::gil_scoped_release>());
    m.def("compute_energy_integrals", &compute_energy_integrals, py::arg("solution"),
        py::arg("options") = Options{}, py::call_guard<py::gil_scoped_release>());
    m.def("compute_frequency_curvature", &compute_frequency_curvature, py::arg("solution"),
        py::arg("options") = Options{}, py::call_guard<py::gil_scoped_release>());
    m.def("earth_flattening_transformation", &earth_flattening_transformation, py::arg("model"));
    m.def("compute_gravity_correction", &compute_gravity_correction, py::arg("solution"),
        py::arg("gravity") = 9.80665e-3, py::arg("region") = "solid", py::arg("num_sub_layers") = 40,
        py::arg("domega_relative") = 5e-3, py::call_guard<py::gil_scoped_release>());
    m.def("apply_mineos_physical_dispersion", &apply_mineos_physical_dispersion,
        py::arg("reference_model"), py::arg("qkappa"), py::arg("qmu"),
        py::arg("reference_period"), py::arg("period"));
    m.def("get_capabilities", &get_capabilities, py::arg("model"), py::arg("wave") = "Rayleigh");
}
}}} // namespace disper26::api::python
