#ifndef DISPER26_C_H
#define DISPER26_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* All buffers belong to the caller. No pointers are retained.
 * Layer fields are consecutive (rho,c11,c13,c33,c44,c66) per layer,
 * i.e. a Fortran array (6,n). Lengths/capacities count elements, not bytes.
 * Units: km, s, g/cm^3; moduli rho*(km/s)^2. omega is rad/s.
 * On failure numerical outputs are NaN, counts zero (except required root
 * count on BUFFER_TOO_SMALL), level -1. Messages are NUL terminated when
 * message_capacity > 0. Passing NULL is allowed only for zero-size buffers.
 * Love H/V and values skipped by phase_only are NaN on success. */
enum { DISPER26_OK = 0, DISPER26_INVALID_ARGUMENT = 1,
       DISPER26_BUFFER_TOO_SMALL = 2, DISPER26_CALCULATION_FAILED = 3 };
enum { DISPER26_RAYLEIGH = 0, DISPER26_LOVE = 1 };

int disper26_search_roots(int n, const double *layers, const double *thicknesses,
    const double *halfspace, double omega, double c_min, double c_max,
    int n_search, int wave, double *roots, int capacity, int *count,
    char *message, int message_capacity);
int disper26_compute_group_velocity_ad(int n, const double *layers,
    const double *thicknesses, const double *halfspace, double omega,
    double c_phase, int wave, double *value, char *message, int message_capacity);
int disper26_compute_hv_ratio(int n, const double *layers,
    const double *thicknesses, const double *halfspace, double omega,
    double c_phase, int wave, double *value, char *message, int message_capacity);

/* GRID fields: iso=(rho,vp,vs), vti=(rho,vph,vpv,vsh,vsv,eta), per node.
 * flags are 0 or 1; a flagged node must precede a node at the same depth.
 * The last node supplies the halfspace. Flat geometry only.
 * Returned relative_error describes the accepted phase table, not a promise
 * that tolerance was reached; level is the core's converged_level. */
int disper26_compute_dispersion_romberg_velocity_iso(int n, const double *depths,
    const double *fields, const int *flags, double period, int wave,
    double c_guess, int max_steps, double tolerance, int phase_only,
    int adaptive_root_scan, double initial_root_window_frac, int extrapolate_start_level,
    double *c, double *u, double *hv, int *level, double *relative_error,
    char *message, int message_capacity);
int disper26_compute_dispersion_romberg_velocity_vti(int n, const double *depths,
    const double *fields, const int *flags, double period, int wave,
    double c_guess, int max_steps, double tolerance, int phase_only,
    int adaptive_root_scan, double initial_root_window_frac, int extrapolate_start_level,
    double *c, double *u, double *hv, int *level, double *relative_error,
    char *message, int message_capacity);

#ifdef __cplusplus
}
#endif
#endif
