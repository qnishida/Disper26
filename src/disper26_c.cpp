#include "disper26_c.h"
#include "dispersion_interface.hpp"
#include "api_impl.hpp"
#include <cstring>
#include <climits>

namespace {
constexpr double quiet_nan = std::numeric_limits<double>::quiet_NaN();
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void positive(double value, const char* name) {
    require(std::isfinite(value) && value > 0.0, name);
}
WaveType wave_type(int wave) {
    require(wave == DISPER26_RAYLEIGH || wave == DISPER26_LOVE, "Invalid wave type");
    return wave == DISPER26_LOVE ? WaveType::Love : WaveType::Rayleigh;
}
void set_message(char* dest, int capacity, const char* message) noexcept {
    if (!dest || capacity <= 0) return;
    const size_t length = std::min(std::strlen(message), static_cast<size_t>(capacity - 1));
    std::memcpy(dest, message, length);
    dest[length] = '\0';
}
template <class F> int guarded(char* message, int capacity, F&& action) noexcept {
    set_message(message, capacity, "");
    try {
        require(capacity >= 0 && (capacity == 0 || message), "Invalid message buffer");
        return action();
    } catch (const std::invalid_argument& e) {
        set_message(message, capacity, e.what());
        return DISPER26_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        set_message(message, capacity, e.what());
        return DISPER26_CALCULATION_FAILED;
    } catch (...) {
        set_message(message, capacity, "Unknown C++ exception");
        return DISPER26_CALCULATION_FAILED;
    }
}
PhysicalParameters<double> unpack(const double* p) {
    require(p != nullptr, "Missing material parameters");
    for (int j = 0; j < 6; ++j) require(std::isfinite(p[j]), "Nonfinite material parameter");
    positive(p[0], "Density must be positive");
    require(p[1] > 0 && p[3] > 0 && p[4] >= 0 && p[5] >= 0, "Invalid elastic modulus");
    PhysicalParameters<double> out;
    out.rho=p[0]; out.c11=p[1]; out.c13=p[2]; out.c33=p[3]; out.c44=p[4]; out.c66=p[5];
    return out;
}
VelocityGridRefinement pancake(int n, const double* fields, const double* h, const double* hs) {
    require(n >= 0 && (n == 0 || (fields && h)), "Invalid layer buffers");
    VelocityGridRefinement out;
    out.halfspace = unpack(hs);
    for (int i = 0; i < n; ++i) {
        positive(h[i], "Thickness must be finite and positive");
        out.layers.push_back(unpack(fields + 6 * static_cast<size_t>(i)));
        out.thicknesses.push_back(h[i]);
    }
    return out;
}
int scalar_value(bool hv, int n, const double* fields, const double* h, const double* hs,
                 double omega, double cp, int wave, double* value, char* message, int capacity) {
    if (value) *value = quiet_nan;
    return guarded(message, capacity, [&]() {
        require(value, "Missing output buffer");
        const auto type = wave_type(wave);
        positive(omega, "omega must be finite and positive [rad/s]");
        positive(cp, "Phase velocity must be finite and positive");
        const auto model = pancake(n, fields, h, hs);
        double result;
        if (!hv) {
            result = calculateGroupVelocityFullAD(model.layers, model.thicknesses,
                                                  model.halfspace, omega, cp, type);
        } else {
            if (type == WaveType::Love) return DISPER26_OK;
            Eigen::Matrix<double, 5, 1> seabed = Eigen::Matrix<double, 5, 1>::Zero();
            const auto state = calculateDispersionFunction(model.layers, model.thicknesses,
                model.halfspace, omega, cp, type, NormalizationMode::Vertical, &seabed);
            result = !model.layers.empty() && isLiquidLayer(model.layers[0])
                ? calculateHVRatio(seabed) : calculateHVRatio(state);
        }
        if (!std::isfinite(result)) throw std::runtime_error("Nonfinite calculation result");
        *value = result;
        return DISPER26_OK;
    });
}
int grid(bool vti, int n, const double* z, const double* fields, const int* flags,
         double period, int wave, double guess, int steps, double tolerance, int phase_only,
         int adaptive, double window, int start, double* c, double* u, double* hv,
         int* level, double* error, char* message, int capacity) {
    if (c) *c=quiet_nan;
    if (u) *u=quiet_nan;
    if (hv) *hv=quiet_nan;
    if (level) *level=-1;
    if (error) *error=quiet_nan;
    return guarded(message, capacity, [&]() {
        require(c && u && hv && level && error, "Missing GRID output buffer");
        const auto type = wave_type(wave);
        require(n > 0 && z && fields && flags, "Invalid GRID buffers");
        positive(period, "period must be finite and positive [s]");
        positive(guess, "c_guess must be finite and positive");
        positive(tolerance, "tolerance must be finite and positive");
        // Geometry uses 1 << level. Reject levels that cannot be represented.
        require(steps >= 0 && steps < std::numeric_limits<int>::digits, "Invalid max_steps");
        require(start >= 0 && start <= steps, "Invalid extrapolate_start_level");
        require((phase_only == 0 || phase_only == 1) && (adaptive == 0 || adaptive == 1), "Invalid logical flag");
        require(std::isfinite(window) && window >= 0, "Invalid initial_root_window_frac");
        const int width = vti ? 6 : 3;
        std::vector<double> depths(z, z+n);
        std::vector<bool> disc;
        std::vector<std::vector<double>> p(width);
        for (int i=0; i<n; ++i) {
            require(std::isfinite(z[i]) && z[i] >= 0, "Invalid GRID depth");
            require(flags[i] == 0 || flags[i] == 1, "Invalid discontinuity flag");
            disc.push_back(flags[i] != 0);
            for (int j=0; j<width; ++j) {
                double v = fields[width * static_cast<size_t>(i) + j];
                require(std::isfinite(v), "Nonfinite GRID parameter");
                if (j == 0 || j == 1 || (vti && j == 2)) positive(v, "Density and P velocities must be positive");
                if ((!vti && j == 2) || (vti && (j == 3 || j == 4))) require(v >= 0, "Negative shear velocity");
                p[j].push_back(v);
            }
        }
        // Preserve the core's geometry rules but classify malformed inputs as arguments.
        try { validate_velocity_grid_sizes(depths, disc, {static_cast<size_t>(n)}); }
        catch (const std::runtime_error& e) { throw std::invalid_argument(e.what()); }
        auto refiner = [&](int k) {
            return vti ? refine_velocity_grid_vti(depths,p[0],p[1],p[2],p[3],p[4],p[5],disc,k)
                       : refine_velocity_grid_iso(depths,p[0],p[1],p[2],disc,k);
        };
        const auto r = compute_dispersion_romberg_result_from_refiner(refiner, period,
            type == WaveType::Love ? "Love" : "Rayleigh", guess, steps, tolerance,
            phase_only != 0, adaptive != 0, window, start);
        if (!std::isfinite(r.phase_velocity) || (!phase_only &&
            (!std::isfinite(r.group_velocity) || (type == WaveType::Rayleigh && !std::isfinite(r.hv_ratio)))))
            throw std::runtime_error("No finite GRID dispersion result");
        *c=r.phase_velocity; *u=r.group_velocity; *hv=r.hv_ratio;
        *level=r.converged_level; *error=r.relative_error;
        return DISPER26_OK;
    });
}
} // namespace

extern "C" int disper26_search_roots(int n, const double* fields, const double* h,
    const double* hs, double omega, double c_min, double c_max, int n_search, int wave,
    double* roots, int capacity, int* count, char* message, int message_capacity) {
    if (count) *count=0;
    if (roots && capacity > 0) std::fill_n(roots, capacity, quiet_nan);
    return guarded(message, message_capacity, [&]() {
        require(count && capacity >= 0 && (capacity == 0 || roots), "Invalid root output buffer");
        const auto type = wave_type(wave);
        positive(omega, "omega must be finite and positive [rad/s]");
        positive(c_min, "c_min must be finite and positive");
        require(std::isfinite(c_max) && c_max > c_min, "c_max must exceed c_min");
        require(n_search > 0, "n_search must be positive");
        const double step = (c_max-c_min)/n_search;
        require(step > 0 && c_min + step > c_min && c_max + step > c_max,
                "Root scan step is too small to advance");
        const auto model = pancake(n, fields, h, hs);
        auto function = [&](double cp) {
            return calculateDispersionPhaseValueDouble(model.layers, model.thicknesses,
                model.halfspace, omega, cp, type);
        };
        const auto result = searchRoots(function, c_min, c_max, step);
        if (result.size() > INT_MAX) throw std::runtime_error("Too many roots");
        *count = static_cast<int>(result.size());
        if (*count > capacity) {
            set_message(message, message_capacity, "Root buffer too small; count gives required capacity");
            return DISPER26_BUFFER_TOO_SMALL;
        }
        if (!result.empty()) std::copy(result.begin(), result.end(), roots);
        return DISPER26_OK;
    });
}
extern "C" int disper26_compute_group_velocity_ad(int n, const double* p, const double* h,
    const double* hs, double omega, double cp, int wave, double* value, char* msg, int cap) {
    return scalar_value(false,n,p,h,hs,omega,cp,wave,value,msg,cap);
}
extern "C" int disper26_compute_hv_ratio(int n, const double* p, const double* h,
    const double* hs, double omega, double cp, int wave, double* value, char* msg, int cap) {
    return scalar_value(true,n,p,h,hs,omega,cp,wave,value,msg,cap);
}
#define GRID_SIGNATURE int n, const double* z, const double* p, const int* flags, \
    double period, int wave, double guess, int steps, double tol, int phase_only, \
    int adaptive, double window, int start, double* c, double* u, double* hv, \
    int* level, double* error, char* msg, int cap
extern "C" int disper26_compute_dispersion_romberg_velocity_iso(GRID_SIGNATURE) {
    return grid(false,n,z,p,flags,period,wave,guess,steps,tol,phase_only,adaptive,window,start,c,u,hv,level,error,msg,cap);
}
extern "C" int disper26_compute_dispersion_romberg_velocity_vti(GRID_SIGNATURE) {
    return grid(true,n,z,p,flags,period,wave,guess,steps,tol,phase_only,adaptive,window,start,c,u,hv,level,error,msg,cap);
}
#undef GRID_SIGNATURE
