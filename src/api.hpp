#pragma once

// Public, Python/Eigen-independent interface. Units: km, s, g/cm^3, GPa.
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace disper26 { namespace api {
inline constexpr double missing_value = std::numeric_limits<double>::quiet_NaN();

class Error : public std::runtime_error {
public:
    std::string code;
    Error(std::string code_, const std::string& message)
        : std::runtime_error(message), code(std::move(code_)) {}
};

struct Material {
    double rho = 0, A = 0, C = 0, F = 0, L = 0, N = 0;
    double gravity = 0, gravity_internal_scale = 1;
    double gravity_surface_scale = 0;
};
Material create_isotropic_parameters(double vs, double vp, double rho);
Material create_fluid_parameters(double vp, double rho);
Material create_velocity_vti_parameters(double rho, double vph, double vpv,
    double vsh, double vsv, double eta);

struct Geometry {
    std::string kind = "flat";
    double planet_radius = 6371.0;
    double flattening_exponent = missing_value;
};
struct ModelData {
    std::string kind, interpolation, name;
    std::vector<Material> materials;
    std::vector<double> depths, thicknesses;
    std::vector<bool> discontinuities;
    Material halfspace;
    Geometry geometry;
    bool flattened = false;
    std::string flattening_policy;
    std::optional<double> q_period, q_reference_period;
};
struct Access;
// Exactly one complete material representation; absent and empty are distinct.
struct MaterialArrays {
    std::optional<std::vector<double>> vs, vp, rho;
    std::optional<std::vector<double>> vph, vpv, vsh, vsv, eta;
    std::optional<std::vector<double>> A, C, F, L, N;
};
class Model {
    std::shared_ptr<const ModelData> data_;
    explicit Model(ModelData data);
    friend struct Access;
public:
    Model() = default; // Empty output holder; computation rejects it.
    const ModelData& data() const;
};
Model create_pancake_model(const std::vector<Material>& layers,
    const std::vector<double>& thicknesses, const Material& halfspace,
    const Geometry& geometry = {}, const std::string& name = "");
Model create_model(const std::vector<double>& depths, const std::string& kind,
    const MaterialArrays& parameters,
    const std::optional<std::string>& interpolation = std::nullopt,
    const Geometry& geometry = {}, const std::string& name = "");
Model create_grid_model(const std::vector<double>& depths,
    const std::vector<Material>& materials, const std::string& interpolation,
    const Geometry& geometry = {}, const std::string& name = "");

struct Options {
    int max_steps = 5, max_levels = 5, num_sub_layers = 10;
    double tolerance = 1e-10, root_tolerance = 1e-12;
    double derivative_tolerance = 1e-6;
    std::optional<double> c_guess;
    std::string method = "standard", backend = "dual", integration = "auto";
    bool strict = false;
    std::string predictor = "auto";
    std::optional<bool> verify_mode;
};
struct DispersionContextData;
// Mutable scratch capacity, never a cache of roots or model-dependent values.
// Calls sharing a context are serialized; independent contexts run concurrently.
class DispersionContext {
    std::unique_ptr<DispersionContextData> data_;
    friend struct Access;
public:
    DispersionContext();
    ~DispersionContext();
    DispersionContext(DispersionContext&&) noexcept;
    DispersionContext& operator=(DispersionContext&&) noexcept;
    DispersionContext(const DispersionContext&) = delete;
    DispersionContext& operator=(const DispersionContext&) = delete;
};
DispersionContext create_dispersion_context();

struct Diagnostics {
    std::string status = "ok", convergence = "not_evaluated";
    std::string method, backend;
    std::vector<std::string> messages;
    std::map<std::string, std::string> availability;
    std::map<std::string, std::string> convergence_by_quantity;
    int accepted_level = -1, generated_level = -1, fallback_count = 0;
    double relative_error = missing_value;
    bool mode_verified = false;
    std::vector<bool> level_mode_verified;
    std::vector<std::string> predictors, fallback_reasons;
    int mode_count_evaluations = 0, predictor_ad_evaluations = 0;
};
struct SolutionData;
class ModeSolution {
    std::shared_ptr<const SolutionData> data_;
    explicit ModeSolution(std::shared_ptr<const SolutionData> data);
    friend struct Access;
public:
    ModeSolution() = default;
    double phase_velocity() const;
    std::optional<double> group_velocity() const;
    std::optional<double> hv_ratio() const;
    double period() const;
    const std::string& wave() const;
    int mode() const;
    const Model& model() const;
    const Options& options() const;
    const Diagnostics& diagnostics() const;
};
struct DispersionCurve {
    std::vector<double> periods;
    std::vector<ModeSolution> solutions;
};
ModeSolution compute_dispersion(const Model& model, double period,
    const std::string& wave = "Rayleigh", int mode = 0,
    const std::vector<std::string>& outputs = {"phase_velocity"},
    const Options& options = {});
DispersionCurve compute_dispersion_curve(const Model& model,
    const std::vector<double>& periods, const std::string& wave = "Rayleigh",
    int mode = 0, const std::vector<std::string>& outputs = {"phase_velocity"},
    const Options& options = {});

// Additive overloads keep existing C++ signatures and symbols available.
ModeSolution compute_dispersion(const Model& model, double period,
    const std::string& wave, int mode, const std::vector<std::string>& outputs,
    const Options& options, DispersionContext& context);
DispersionCurve compute_dispersion_curve(const Model& model,
    const std::vector<double>& periods, const std::string& wave, int mode,
    const std::vector<std::string>& outputs, const Options& options,
    DispersionContext& context);

// Count once at a saved, unextrapolated root. -1 selects the last saved level.
// This does not certify other levels/periods or mutate the solution diagnostics.
int compute_mode_count(const ModeSolution& solution, int level = -1);

struct Parameter {
    std::string name, entity, unit;
    int index = 0; // C++/C/Python: zero based; halfspace has entity="halfspace".
    bool active = true;
};
struct DerivativeResult {
    std::string basis;
    std::vector<Parameter> parameters;
    std::map<std::string, std::vector<double>> gradients;
    std::vector<double> hessian; // Column-major; rows/cols describe its shape.
    int rows = 0, cols = 0;
    Diagnostics diagnostics;
};
DerivativeResult compute_kernels(const ModeSolution& solution,
    const std::string& basis,
    const std::vector<std::string>& outputs = {"phase_velocity"},
    const Options& options = {});
DerivativeResult compute_hessian(const ModeSolution& solution,
    const std::string& basis, const Options& options = {});
DerivativeResult compute_hessian_reduced(const ModeSolution& solution,
    const std::vector<double>& projection, int columns,
    const Options& options = {}); // projection: column-major stiffness axes x columns.
DerivativeResult compute_interface_kernels(const ModeSolution& solution,
    const Options& options = {});
DerivativeResult compute_interface_hessian_row(const ModeSolution& solution,
    int interface_index, const std::string& basis, const Options& options = {});

struct Eigenfunctions {
    std::vector<double> depth;
    std::map<std::string, std::vector<std::complex<double>>> components;
    std::vector<bool> is_fluid;
    Diagnostics diagnostics;
};
Eigenfunctions compute_eigenfunctions(const ModeSolution& solution,
    int num_sub_layers = 10);
struct EnergyIntegrals {
    double I1 = missing_value, I2 = missing_value, I3 = missing_value;
    double phase_velocity = missing_value, group_velocity = missing_value;
    Diagnostics diagnostics;
};
EnergyIntegrals compute_energy_integrals(const ModeSolution& solution,
    const Options& options = {});
struct FrequencyCurvature {
    double phase_velocity = missing_value, group_velocity = missing_value;
    double group_velocity_omega_derivative = missing_value;
    double wavenumber_omega_second_derivative = missing_value;
    Diagnostics diagnostics;
};
FrequencyCurvature compute_frequency_curvature(const ModeSolution& solution,
    const Options& options = {});
struct GravityCorrection {
    double phase_velocity = missing_value, group_velocity = missing_value;
    double delta_c = missing_value, delta_u = missing_value;
    double omega_delta_c_omega_derivative = missing_value;
    std::string region;
    Diagnostics diagnostics;
};
GravityCorrection compute_gravity_correction(const ModeSolution& solution,
    double gravity = 9.80665e-3, const std::string& region = "solid",
    int num_sub_layers = 40, double domega_relative = 5e-3);
Model earth_flattening_transformation(const Model& model);
Model apply_mineos_physical_dispersion(const Model& reference_model,
    const std::vector<double>& qkappa, const std::vector<double>& qmu,
    double reference_period, double period);
struct Capability {
    std::string operation, basis, method, reason;
    bool supported = false;
};
std::vector<Capability> get_capabilities(const Model& model,
    const std::string& wave = "Rayleigh");
}} // namespace disper26::api
