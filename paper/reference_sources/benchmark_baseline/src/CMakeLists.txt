# DISPER26_MATH_FLAGS is defined in the top-level CMakeLists.txt so that
# tests/ sees it too.

# Core library with autodiff support
add_library(disper26_lib disper26.cpp)
target_compile_options(disper26_lib PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(disper26_lib Eigen3::Eigen autodiff::autodiff)

# Main Executable
add_executable(disper26_run disper26.cpp)
target_compile_definitions(disper26_run PRIVATE BUILD_MAIN)
target_compile_options(disper26_run PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(disper26_run Eigen3::Eigen autodiff::autodiff)

# --- Python Bindings ---
find_package(pybind11 CONFIG)
if(pybind11_FOUND)
    pybind11_add_module(disper26py bindings.cpp)
    # bindings.cpp includes disper26.cpp directly, so we don't link disper26_lib
    # But we need to link dependencies
    target_compile_options(disper26py PRIVATE ${DISPER26_MATH_FLAGS})
    target_link_libraries(disper26py PRIVATE Eigen3::Eigen autodiff::autodiff)
    message(STATUS "Python bindings target 'disper26py' added.")
else()
    message(WARNING "pybind11 not found. Please install it (e.g. conda install pybind11) to build Python bindings.")
endif()
