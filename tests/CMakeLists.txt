# Eigen3 is found in root CMakeLists.txt
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(test_dispersion_context test_dispersion_context.cpp)
add_test(NAME dispersion_context COMMAND test_dispersion_context)

add_executable(test_dispersion_curve_context test_dispersion_curve_context.cpp)
target_link_libraries(test_dispersion_curve_context PRIVATE disper26::api)
add_test(NAME dispersion_curve_context COMMAND test_dispersion_curve_context)

add_executable(test_dispersion_tracking_snapshot test_dispersion_tracking_snapshot.cpp)
target_link_libraries(test_dispersion_tracking_snapshot PRIVATE disper26::api)
add_test(NAME dispersion_tracking_snapshot COMMAND test_dispersion_tracking_snapshot)

add_executable(test_dispersion_tracking test_dispersion_tracking.cpp)
target_compile_options(test_dispersion_tracking PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(test_dispersion_tracking PRIVATE Eigen3::Eigen autodiff::autodiff)
add_test(NAME dispersion_tracking COMMAND test_dispersion_tracking)

add_executable(test_api_c test_api_c.cpp)
target_link_libraries(test_api_c PRIVATE disper26::api)
add_test(NAME public_api_c COMMAND test_api_c)

find_package(Threads REQUIRED)
add_executable(test_api_threads test_api_threads.cpp)
target_link_libraries(test_api_threads PRIVATE disper26::api Threads::Threads)
add_test(NAME public_api_threads COMMAND test_api_threads)

add_executable(test_brent_known_values test_brent_known_values.cpp)
target_compile_options(test_brent_known_values PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(test_brent_known_values Eigen3::Eigen autodiff::autodiff)
add_test(NAME brent_known_values COMMAND test_brent_known_values)

foreach(backend_variant default custom)
    set(backend_test test_dispersion_math_backend_${backend_variant})
    add_executable(${backend_test} test_dispersion_math_backend.cpp)
    target_compile_options(${backend_test} PRIVATE ${DISPER26_MATH_FLAGS})
    target_link_libraries(${backend_test} PRIVATE Eigen3::Eigen autodiff::autodiff)
    if(backend_variant STREQUAL "custom")
        target_compile_definitions(${backend_test} PRIVATE
            DISPER26_HYPERBOLIC_BACKEND_HEADER="${CMAKE_CURRENT_SOURCE_DIR}/math_backend_probe.hpp")
    endif()
    add_test(NAME dispersion_math_backend_${backend_variant} COMMAND ${backend_test})
endforeach()

add_executable(benchmark_dispersion_math benchmark_dispersion_math.cpp)
target_compile_options(benchmark_dispersion_math PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(benchmark_dispersion_math PRIVATE Eigen3::Eigen autodiff::autodiff)

add_executable(test_regular_layer_propagation test_regular_layer_propagation.cpp)
target_compile_options(test_regular_layer_propagation PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(test_regular_layer_propagation PRIVATE Eigen3::Eigen autodiff::autodiff)
add_test(NAME regular_layer_propagation COMMAND test_regular_layer_propagation)

add_executable(test_love_upward_convention test_love_upward_convention.cpp)
target_compile_options(test_love_upward_convention PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(test_love_upward_convention PRIVATE Eigen3::Eigen autodiff::autodiff)
add_test(NAME love_upward_convention COMMAND test_love_upward_convention)

add_executable(test_real_rayleigh test_real_rayleigh.cpp)
target_compile_options(test_real_rayleigh PRIVATE ${DISPER26_MATH_FLAGS} -march=native)
target_link_libraries(test_real_rayleigh PRIVATE Eigen3::Eigen autodiff::autodiff)
add_test(NAME real_rayleigh COMMAND test_real_rayleigh)

add_executable(probe_group_velocity_accuracy probe_group_velocity_accuracy.cpp)
target_compile_options(probe_group_velocity_accuracy PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(probe_group_velocity_accuracy PRIVATE Eigen3::Eigen autodiff::autodiff)

add_executable(test_regular_energy_integrals test_regular_energy_integrals.cpp)
target_compile_options(test_regular_energy_integrals PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(test_regular_energy_integrals PRIVATE Eigen3::Eigen autodiff::autodiff)
add_test(NAME regular_energy_integrals COMMAND test_regular_energy_integrals)

add_executable(probe_energy_integrals probe_energy_integrals.cpp)
target_compile_options(probe_energy_integrals PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(probe_energy_integrals PRIVATE Eigen3::Eigen autodiff::autodiff)

# These reference checks require more precision than binary64. Apple Silicon
# has a 53-bit long double; keep the checks intact and report their exclusion.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <limits>
static_assert(std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits, \"extended precision required\");
int main() { return 0; }" DISPER26_HAS_EXTENDED_LONG_DOUBLE)
if(DISPER26_HAS_EXTENDED_LONG_DOUBLE)
    add_executable(test_dispersion_taylor_double test_dispersion_taylor_double.cpp)
    target_compile_definitions(test_dispersion_taylor_double PRIVATE
        DISPER26_EXPERIMENT_TAYLOR_CS=1 DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED=1)
    target_compile_options(test_dispersion_taylor_double PRIVATE ${DISPER26_MATH_FLAGS})
    target_link_libraries(test_dispersion_taylor_double Eigen3::Eigen autodiff::autodiff)

    function(add_shared_hyperbolic_test target taylor_mode paired_mode)
        set(shared_test test_dispersion_shared_hyperbolic_${target})
        add_executable(${shared_test} test_dispersion_shared_hyperbolic.cpp)
        target_compile_definitions(${shared_test} PRIVATE
            DISPER26_EXPERIMENT_SHARED_CS=1
            DISPER26_EXPERIMENT_TAYLOR_CS=${taylor_mode}
            DISPER26_EXPERIMENT_TAYLOR_CS_PAIRED=${paired_mode})
        target_compile_options(${shared_test} PRIVATE ${DISPER26_MATH_FLAGS})
        target_link_libraries(${shared_test} Eigen3::Eigen autodiff::autodiff)
    endfunction()

    add_shared_hyperbolic_test(taylor_0_paired_0 0 0)
    add_shared_hyperbolic_test(taylor_1_paired_0 1 0)
    add_shared_hyperbolic_test(taylor_1_paired_1 1 1)

else()
    message(STATUS "Skipping Taylor/shared-hyperbolic reference tests: long double has no extra precision")
endif()

# Build the autodiff verification test (minimal)
add_executable(test_autodiff_basic test_autodiff_basic.cpp)
target_link_libraries(test_autodiff_basic disper26_lib Eigen3::Eigen autodiff::autodiff)

add_executable(test_autodiff_eigen_indexed test_autodiff_eigen_indexed.cpp)
target_link_libraries(test_autodiff_eigen_indexed Eigen3::Eigen autodiff::autodiff)
add_test(NAME autodiff_eigen_indexed COMMAND test_autodiff_eigen_indexed)

# Build the autodiff forward mode sensitivity test
add_executable(test_autodiff_forward_sensitivity test_autodiff_forward_sensitivity.cpp)
target_link_libraries(test_autodiff_forward_sensitivity disper26_lib Eigen3::Eigen autodiff::autodiff)

# Build the Full AD group velocity kernel test (with L2 scaling)
# add_executable(test_full_ad_group_velocity_kernel test_full_ad_group_velocity_kernel.cpp)
# target_link_libraries(test_full_ad_group_velocity_kernel disper26_lib Eigen3::Eigen autodiff::autodiff)

# Build the Love wave group velocity kernel test
add_executable(test_love_group_velocity_kernel test_love_group_velocity_kernel.cpp)
target_link_libraries(test_love_group_velocity_kernel disper26_lib Eigen3::Eigen)

# Build the Romberg stability test
add_executable(test_romberg_stability test_romberg_stability.cpp)
target_link_libraries(test_romberg_stability disper26_lib Eigen3::Eigen)

# Direct-dispersion level roots retained for derivative consumers.
add_executable(test_romberg_root_trace test_romberg_root_trace.cpp)
target_link_libraries(test_romberg_root_trace disper26_lib Eigen3::Eigen)
add_test(NAME romberg_root_trace COMMAND test_romberg_root_trace)

# Build the Full AD group velocity sensitivity test
add_executable(test_group_velocity_full_ad test_group_velocity_full_ad.cpp)
target_link_libraries(test_group_velocity_full_ad disper26_lib Eigen3::Eigen autodiff::autodiff)

# Build the backward mode sensitivity test
add_executable(test_backward_mode_sensitivity test_backward_mode_sensitivity.cpp)
target_link_libraries(test_backward_mode_sensitivity disper26_lib Eigen3::Eigen autodiff::autodiff)

# Build the vectorized forward mode test
add_executable(test_vectorized_forward test_vectorized_forward.cpp)
target_link_libraries(test_vectorized_forward disper26_lib Eigen3::Eigen)

# Build the isotropic half-space test
add_executable(test_halfspace_isotropic test_halfspace_isotropic.cpp)
target_link_libraries(test_halfspace_isotropic disper26_lib Eigen3::Eigen)

# Build the sensitivity verification test
add_executable(test_sensitivity_halfspace test_sensitivity_halfspace.cpp)
target_link_libraries(test_sensitivity_halfspace disper26_lib Eigen3::Eigen)

# Build the group velocity test
add_executable(test_group_velocity test_group_velocity.cpp)
target_link_libraries(test_group_velocity disper26_lib Eigen3::Eigen)

# Build the group sensitivity test
add_executable(test_group_sensitivity test_group_sensitivity.cpp)
target_link_libraries(test_group_sensitivity disper26_lib Eigen3::Eigen)

# Build the scaling verification test
add_executable(test_scaling test_scaling.cpp)
target_link_libraries(test_scaling disper26_lib Eigen3::Eigen)

# Build the love wave model test
add_executable(test_love_model test_love_model.cpp)
target_link_libraries(test_love_model disper26_lib Eigen3::Eigen)

# Build the love kernels test
add_executable(test_love_kernels test_love_kernels.cpp)
target_link_libraries(test_love_kernels disper26_lib Eigen3::Eigen)

# Build the isotropic helper test
add_executable(test_isotropic_helper test_isotropic_helper.cpp)
target_link_libraries(test_isotropic_helper disper26_lib Eigen3::Eigen)

# Build the Rayleigh direct-apply equivalence test



# Optional Thickness Kernels Regression Test
add_executable(test_thickness_kernels test_thickness_kernels.cpp)
target_link_libraries(test_thickness_kernels Eigen3::Eigen autodiff::autodiff)

# Per-layer phase counters (countModesLayerRayleigh / countModesLayerLove).
# Was an orphan for months -- absent from this list, so nobody noticed it had
# stopped building. Keep it here.
add_executable(verify_mode_count verify_mode_count.cpp)
target_compile_options(verify_mode_count PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(verify_mode_count Eigen3::Eigen autodiff::autodiff)
add_test(NAME legacy_layer_mode_count COMMAND verify_mode_count)

add_executable(benchmark_energy_candidates EXCLUDE_FROM_ALL benchmark_energy_candidates.cpp)
target_compile_options(benchmark_energy_candidates PRIVATE ${DISPER26_MATH_FLAGS} -march=native)
target_link_libraries(benchmark_energy_candidates PRIVATE Eigen3::Eigen autodiff::autodiff)

add_executable(benchmark_group_velocity EXCLUDE_FROM_ALL benchmark_group_velocity.cpp)
target_compile_options(benchmark_group_velocity PRIVATE ${DISPER26_MATH_FLAGS} -march=native)
target_link_libraries(benchmark_group_velocity PRIVATE Eigen3::Eigen autodiff::autodiff)

# Allocation instrumentation runs separately from wall-clock benchmarks.
add_executable(benchmark_dispersion_allocations EXCLUDE_FROM_ALL benchmark_dispersion_allocations.cpp)
target_include_directories(benchmark_dispersion_allocations PRIVATE ../src)
target_compile_definitions(benchmark_dispersion_allocations PRIVATE DISPER26_CONTEXT_AVAILABLE=1)
target_compile_options(benchmark_dispersion_allocations PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(benchmark_dispersion_allocations PRIVATE Eigen3::Eigen autodiff::autodiff Threads::Threads)

# Runtime assertions are part of the test contract, including Release builds.
# Keep production library/executable flags unchanged.
get_property(disper26_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(disper26_test_target IN LISTS disper26_test_targets)
    if(disper26_test_target MATCHES "^test_|^verify_mode_count$")
        target_compile_options(${disper26_test_target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-UNDEBUG>
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>)
    endif()
endforeach()
