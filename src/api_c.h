#ifndef DISPER26_API_C_H
#define DISPER26_API_C_H

/* Additive, owning C interface to api.hpp. Units: km, s, g/cm^3, GPa.
 * All inputs are copied. Handles start with one reference; retain before
 * sharing ownership and release once per reference. NULL release is safe.
 * Models and solutions are immutable. Each call has its own error message;
 * no thread-local/global last-error storage or process termination is used.
 * Matrices use column-major order. Indices, including modes, start at zero.
 * Numeric field sizes count doubles; string capacities count bytes including
 * the terminating NUL. Query a field with NULL,0 to obtain its required size.
 * A too-small buffer is unchanged and returns D26_API_BUFFER_TOO_SMALL.
 * Every message is NUL-terminated when its nonzero capacity permits it.
 * Output handles are NULL on failure and must not already own a reference.
 */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct d26_api_model d26_api_model;
typedef struct d26_api_dispersion_context d26_api_dispersion_context;
typedef struct d26_api_solution d26_api_solution;
typedef struct d26_api_result d26_api_result;
enum { D26_API_OK=0, D26_API_INVALID_ARGUMENT=1,
       D26_API_BUFFER_TOO_SMALL=2, D26_API_CALCULATION_FAILED=3,
       D26_API_UNSUPPORTED=4, D26_API_NOT_CONVERGED=5 };

typedef struct {
    double rho, A, C, F, L, N;
    double gravity, gravity_internal_scale, gravity_surface_scale;
} d26_api_material;
enum { D26_API_FLAT=0, D26_API_SPHERICAL_BA97=1,
       D26_API_SPHERICAL_EXPERIMENTAL_DENSITY=2 };
typedef struct {
    int kind; /* One of D26_API_FLAT / D26_API_SPHERICAL_* above. */
    double planet_radius, flattening_exponent; /* NaN: use backend default. */
} d26_api_geometry;
typedef struct {
    int max_steps, max_levels, num_sub_layers;
    double tolerance, root_tolerance, derivative_tolerance;
    int has_c_guess;
    double c_guess;
    char method[64], backend[64], integration[64];
    int strict;
} d26_api_options;

/* Additive curve controls; initialize with d26_api_default_tracking_options.
 * Empty verify/on_violation means unspecified. verify_mode is -1 (unset),
 * 0 or 1. Existing d26_api_options retains its binary layout. */
typedef struct {
    size_t struct_size;
    char start[64], predictor[64], verify[64], on_violation[64];
    int has_c_range;
    double c_min, c_max;
    int verify_mode;
} d26_api_tracking_options;

void d26_api_default_geometry(d26_api_geometry* geometry);
void d26_api_default_options(d26_api_options* options);
void d26_api_default_tracking_options(d26_api_tracking_options* options);
/* Column-major values[components,count]: isotropic = Vs,Vp,Rho;
 * velocity_vti = rho,vph,vpv,vsh,vsv,eta; stiffness = A,C,F,L,N,Rho.
 * interpolation must be NULL for Layer-cake and explicit for GRID. */
int d26_api_create_model(size_t count,const double* depths,const char* kind,
    const char* parameterization,const double* values,size_t components,
    const char* interpolation,const d26_api_geometry* geometry,const char* name,
    d26_api_model** out,char* message,size_t message_capacity);
int d26_api_create_isotropic_parameters(double vs,double vp,double rho,
    d26_api_material* out,char* message,size_t message_capacity);
int d26_api_create_fluid_parameters(double vp,double rho,
    d26_api_material* out,char* message,size_t message_capacity);
int d26_api_create_velocity_vti_parameters(double rho,double vph,double vpv,
    double vsh,double vsv,double eta,d26_api_material* out,char* message,size_t message_capacity);
int d26_api_create_layer_cake_model(size_t count,const d26_api_material* layers,
    const double* thicknesses,const d26_api_material* halfspace,
    const d26_api_geometry* geometry,const char* name,d26_api_model** out,
    char* message,size_t message_capacity);
// Deprecated alias; preserves kind="pancake".
int d26_api_create_pancake_model(size_t count,const d26_api_material* layers,
    const double* thicknesses,const d26_api_material* halfspace,
    const d26_api_geometry* geometry,const char* name,d26_api_model** out,
    char* message,size_t message_capacity);
int d26_api_create_grid_model(size_t count,const double* depths,
    const d26_api_material* materials,const char* interpolation,
    const d26_api_geometry* geometry,const char* name,d26_api_model** out,
    char* message,size_t message_capacity);
void d26_api_model_retain(d26_api_model* model);
void d26_api_model_release(d26_api_model* model);
void d26_api_solution_retain(d26_api_solution* solution);
void d26_api_solution_release(d26_api_solution* solution);
void d26_api_result_retain(d26_api_result* result);
void d26_api_result_release(d26_api_result* result);

/* outputs is a comma-separated list of quantity names, or NULL for phase
 * velocity. Wave is Rayleigh or Love. NULL options uses documented defaults. */
int d26_api_compute_dispersion(const d26_api_model* model,double period,
    const char* wave,int mode,const char* outputs,const d26_api_options* options,
    d26_api_solution** out,char* message,size_t message_capacity);
int d26_api_compute_dispersion_curve(const d26_api_model* model,size_t count,
    const double* periods,const char* wave,int mode,const char* outputs,
    const d26_api_options* options,d26_api_result** out,char* message,size_t message_capacity);
/* Owned reusable capacity. The context is required by *_with_context;
 * calls sharing it are serialized. Retain/release follow other handles. */
int d26_api_create_dispersion_context(d26_api_dispersion_context** out,
    char* message,size_t message_capacity);
void d26_api_dispersion_context_retain(d26_api_dispersion_context* context);
void d26_api_dispersion_context_release(d26_api_dispersion_context* context);
int d26_api_compute_dispersion_with_context(const d26_api_model* model,double period,
    const char* wave,int mode,const char* outputs,const d26_api_options* options,
    d26_api_dispersion_context* context,d26_api_solution** out,char* message,size_t message_capacity);
int d26_api_compute_dispersion_curve_with_context(const d26_api_model* model,size_t count,
    const double* periods,const char* wave,int mode,const char* outputs,
    const d26_api_options* options,d26_api_dispersion_context* context,
    d26_api_result** out,char* message,size_t message_capacity);
/* Extended entrypoints accept optional controls and an optional context.
 * NULL controls uses the same defaults as the original entrypoints. */
int d26_api_compute_dispersion_ex(const d26_api_model* model,double period,
    const char* wave,int mode,const char* outputs,const d26_api_options* options,
    const d26_api_tracking_options* tracking,d26_api_dispersion_context* context,
    d26_api_solution** out,char* message,size_t message_capacity);
int d26_api_compute_dispersion_curve_ex(const d26_api_model* model,size_t count,
    const double* periods,const char* wave,int mode,const char* outputs,
    const d26_api_options* options,const d26_api_tracking_options* tracking,
    d26_api_dispersion_context* context,d26_api_result** out,
    char* message,size_t message_capacity);
int d26_api_curve_solution(const d26_api_result* curve,size_t index,
    d26_api_solution** out,char* message,size_t message_capacity);
int d26_api_model_info(const d26_api_model* model,d26_api_result** out,char* message,size_t message_capacity);
/* One count on one saved root; level=-1 selects the last generated level.
   No root search or diagnostic mutation. count is -1 on failure. */
int d26_api_compute_mode_count(const d26_api_solution* solution,int level,
    int* count,char* message,size_t message_capacity);
int d26_api_solution_info(const d26_api_solution* solution,d26_api_result** out,char* message,size_t message_capacity);
int d26_api_solution_model(const d26_api_solution* solution,d26_api_model** out,char* message,size_t message_capacity);
int d26_api_compute_kernels(const d26_api_solution* solution,const char* basis,
    const char* outputs,const d26_api_options* options,d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_hessian(const d26_api_solution* solution,const char* basis,
    const d26_api_options* options,d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_hessian_reduced(const d26_api_solution* solution,
    const double* projection,size_t rows,size_t columns,const d26_api_options* options,
    d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_interface_kernels(const d26_api_solution* solution,
    const d26_api_options* options,d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_interface_hessian_row(const d26_api_solution* solution,int interface_index,
    const char* basis,const d26_api_options* options,d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_eigenfunctions(const d26_api_solution* solution,int num_sub_layers,
    d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_energy_integrals(const d26_api_solution* solution,const d26_api_options* options,
    d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_frequency_curvature(const d26_api_solution* solution,const d26_api_options* options,
    d26_api_result** out,char* message,size_t message_capacity);
int d26_api_compute_gravity_correction(const d26_api_solution* solution,double gravity,const char* region,
    int num_sub_layers,double domega_relative,d26_api_result** out,char* message,size_t message_capacity);
int d26_api_earth_flattening_transformation(const d26_api_model* model,
    d26_api_model** out,char* message,size_t message_capacity);
int d26_api_apply_mineos_physical_dispersion(const d26_api_model* model,size_t count,
    const double* qkappa,const double* qmu,double reference_period,double period,
    d26_api_model** out,char* message,size_t message_capacity);
int d26_api_get_capabilities(const d26_api_model* model,const char* wave,
    d26_api_result** out,char* message,size_t message_capacity);

/* Named fields are discoverable; unknown fields are errors, never zero.
 * Scalar fields have shape (1,1); vectors (n,1), matrices (rows,columns).
 * Derivatives: gradient.<quantity>, hessian, parameter.index/active plus
 * parameter.<i>.name/entity/unit and basis. Parameter masks distinguish
 * fixed components from computed zero. Eigenfunctions: depth, is_fluid,
 * <component>_real and <component>_imag. Curve: periods and quantity arrays.
 * Metadata: diagnostics.status/convergence/method/backend/messages,
 * diagnostics.accepted_level/generated_level/fallback_count/relative_error,
 * availability.<quantity>. Unrequested scalar observables are absent.
 * get_capabilities: supported vector and capability.<i>.operation/basis/
 * method/reason. Material info matrix has fields in d26_api_material order.
 */
int d26_api_result_numeric(const d26_api_result* result,const char* field,
    double* values,size_t capacity,size_t* rows,size_t* columns,
    char* message,size_t message_capacity);
int d26_api_result_string(const d26_api_result* result,const char* field,
    char* value,size_t capacity,size_t* required,char* message,size_t message_capacity);
int d26_api_result_field_count(const d26_api_result* result,int numeric,size_t* count,
    char* message,size_t message_capacity);
int d26_api_result_field_name(const d26_api_result* result,int numeric,size_t index,
    char* value,size_t capacity,size_t* required,char* message,size_t message_capacity);

#ifdef __cplusplus
}
#endif
#endif
