cmake_minimum_required(VERSION 3.18)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/MacOSCompiler.cmake")
project(Disper26 VERSION 0.1.104 LANGUAGES CXX)

option(DISPER26_BUILD_TESTS "Build the C++ test executables" ON)
option(DISPER26_BUILD_PYTHON "Build Python bindings when pybind11 is available" ON)
option(DISPER26_REQUIRE_PYTHON "Fail if Python bindings cannot be configured" OFF)
option(DISPER26_EXPERIMENTAL_REAL_RAYLEIGH
    "Use real finite-layer arithmetic in Rayleigh phase values and integer counts" ON)
if(DISPER26_EXPERIMENTAL_REAL_RAYLEIGH)
    add_compile_definitions(DISPER26_EXPERIMENTAL_REAL_RAYLEIGH=1)
endif()

if(APPLE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR
        "Disper26 currently requires GNU GCC/libstdc++ on macOS for complex autodiff. "
        "Install Homebrew gcc and configure a new build directory, or set "
        "-DCMAKE_CXX_COMPILER=/path/to/g++-<version>. Existing compiler caches are preserved.")
endif()

option(DISPER26_BUILD_FORTRAN "Build the ISO_C_BINDING Fortran interface" OFF)
if(DISPER26_BUILD_FORTRAN)
    enable_language(Fortran)
    enable_testing()
endif()

# C++17 Standard (required for autodiff)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shared complex arithmetic options for CMake and Python builds.
#   -fcx-limited-range inlines complex multiply/divide instead of calling
#     libgcc's __muldc3/__divdc3; the NaN-recovery those helpers perform never
#     fires on finite layer parameters.
#   -fno-math-errno lets sqrt() compile to a single instruction.
# Neither relaxes real-valued IEEE semantics the way -ffast-math would.
# Defined here rather than in src/ so tests/ can apply it as well.
set(DISPER26_MATH_FLAGS -fcx-limited-range -fno-math-errno)

# macOS uses Homebrew eigen@3; Linux uses the system Eigen 3.4 package.
# Select the package directory explicitly to replace stale Eigen 5 caches
# and avoid finding another Eigen through a Conda environment.
if(APPLE)
    find_program(DISPER26_BREW brew HINTS /opt/homebrew/bin /usr/local/bin)
    if(NOT DISPER26_BREW)
        message(FATAL_ERROR "Disper26 requires Homebrew eigen@3 on macOS. Install Homebrew, then run: brew install eigen@3")
    endif()
    execute_process(COMMAND "${DISPER26_BREW}" --prefix eigen@3
        OUTPUT_VARIABLE _eigen_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _eigen_result)
    if(NOT _eigen_result EQUAL 0 OR
       NOT EXISTS "${_eigen_prefix}/share/eigen3/cmake/Eigen3Config.cmake")
        message(FATAL_ERROR "Disper26 requires Homebrew eigen@3 on macOS. Run: brew install eigen@3")
    endif()
    set(Eigen3_DIR "${_eigen_prefix}/share/eigen3/cmake" CACHE PATH
        "Eigen 3.4 package directory (Homebrew eigen@3 on macOS)" FORCE)
endif()
find_package(Eigen3 CONFIG REQUIRED)
if(NOT Eigen3_VERSION VERSION_GREATER_EQUAL "3.4" OR
   NOT Eigen3_VERSION VERSION_LESS "3.5")
    message(FATAL_ERROR
        "Disper26 requires Eigen 3.4.x; found ${Eigen3_VERSION} at ${Eigen3_DIR}. "
        "Use Homebrew eigen@3 on macOS or an Eigen 3.4.x package on Linux.")
endif()
message(STATUS "Disper26: using Eigen ${Eigen3_VERSION} from ${Eigen3_DIR}")

# --- Use the same autodiff source for native and Python builds ---
# Disable building autodiff's own tests, examples, docs, and Python bindings
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

include(FetchContent)
set(DISPER26_AUTODIFF_COMMIT b0a4feff5b2a61262e94305452ac53369fe35e75)
# An explicit source override takes precedence over the populated submodule or
# the source bundled in an sdist. Do not select a previous build's dependency.
if(NOT FETCHCONTENT_SOURCE_DIR_AUTODIFF
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/autodiff/CMakeLists.txt"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/autodiff/autodiff/forward/dual.hpp")
    set(FETCHCONTENT_SOURCE_DIR_AUTODIFF "${CMAKE_CURRENT_SOURCE_DIR}/third_party/autodiff")
endif()
if(FETCHCONTENT_SOURCE_DIR_AUTODIFF)
    if(NOT EXISTS "${FETCHCONTENT_SOURCE_DIR_AUTODIFF}/CMakeLists.txt"
       OR NOT EXISTS "${FETCHCONTENT_SOURCE_DIR_AUTODIFF}/autodiff/forward/dual.hpp")
        message(FATAL_ERROR
            "FETCHCONTENT_SOURCE_DIR_AUTODIFF must name a complete autodiff source directory.")
    endif()
endif()
FetchContent_Declare(
    autodiff
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG ${DISPER26_AUTODIFF_COMMIT}
)
FetchContent_MakeAvailable(autodiff)
message(STATUS "Disper26: autodiff source ${autodiff_SOURCE_DIR}")

# Add subdirectories
add_subdirectory(src)
if(DISPER26_BUILD_FORTRAN)
    add_subdirectory(fortran)
endif()
if(DISPER26_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
