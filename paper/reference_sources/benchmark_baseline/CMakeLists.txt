cmake_minimum_required(VERSION 3.14)
project(Disper26 VERSION 1.0 LANGUAGES CXX)

# C++17 Standard (required for autodiff)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Complex arithmetic options, kept in sync with setup.py's extra_compile_args
# so the CMake and pip builds produce the same numbers.
#   -fcx-limited-range inlines complex multiply/divide instead of calling
#     libgcc's __muldc3/__divdc3; the NaN-recovery those helpers perform never
#     fires on finite layer parameters.
#   -fno-math-errno lets sqrt() compile to a single instruction.
# Neither relaxes real-valued IEEE semantics the way -ffast-math would.
# Defined here rather than in src/ so tests/ can apply it as well.
set(DISPER26_MATH_FLAGS -fcx-limited-range -fno-math-errno)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# --- Fetch autodiff from GitHub ---
# Disable building autodiff's own tests, examples, docs, and Python bindings
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    autodiff
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG v1.1.2
)
FetchContent_MakeAvailable(autodiff)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
