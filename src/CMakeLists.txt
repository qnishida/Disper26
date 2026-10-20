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

# Public C++ and C facade. The adapter includes the numerical implementation
# once; linking disper26_lib here would duplicate its definitions.
add_library(disper26_c STATIC disper26_c.cpp api_c.cpp)
add_library(disper26::api ALIAS disper26_c)
find_package(Threads REQUIRED)
target_compile_options(disper26_c PRIVATE ${DISPER26_MATH_FLAGS})
target_link_libraries(disper26_c PRIVATE Eigen3::Eigen autodiff::autodiff Threads::Threads)
target_include_directories(disper26_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/disper26>)
set_target_properties(disper26_c PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS disper26_c ARCHIVE DESTINATION lib)
install(FILES api.hpp api_c.h disper26_c.h DESTINATION include/disper26)

# --- Python Bindings ---
if(DISPER26_BUILD_PYTHON)
    # The wheel builder sets Python_EXECUTABLE to its own interpreter.
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module QUIET)
    if(Python_Interpreter_FOUND)
        execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _pybind11_result)
        if(_pybind11_result EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
        endif()
    endif()
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND AND Python_Development.Module_FOUND)
        # Avoid pybind11's implicit LTO: bindings.cpp is a large translation unit.
        pybind11_add_module(disper26py NO_EXTRAS bindings.cpp)
        target_compile_options(disper26py PRIVATE ${DISPER26_MATH_FLAGS})
        if(NOT APPLE)
            # Preserve the existing Linux setuptools build's CPU optimization.
            target_compile_options(disper26py PRIVATE -march=native)
        endif()
        target_link_libraries(disper26py PRIVATE Eigen3::Eigen autodiff::autodiff)
        message(STATUS "Python bindings target 'disper26py' added for ${Python_EXECUTABLE}")
    elseif(DISPER26_REQUIRE_PYTHON)
        message(FATAL_ERROR "Python development headers and pybind11 are required. Use python -m pip install .")
    else()
        message(STATUS "Python bindings skipped: install pybind11 into the selected Python environment")
    endif()
elseif(DISPER26_REQUIRE_PYTHON)
    message(FATAL_ERROR "DISPER26_REQUIRE_PYTHON requires DISPER26_BUILD_PYTHON=ON")
endif()
