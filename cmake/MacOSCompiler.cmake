# Select Homebrew GCC only for a native macOS build with no explicit toolchain.
# std::complex<Eigen::AutoDiffScalar<...>> currently requires libstdc++.
if(CMAKE_HOST_APPLE AND (NOT CMAKE_SYSTEM_NAME OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
   AND NOT CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL ""
   AND NOT CMAKE_TOOLCHAIN_FILE AND "$ENV{CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
    find_program(DISPER26_BREW brew HINTS /opt/homebrew/bin /usr/local/bin)
    if(DISPER26_BREW)
        execute_process(COMMAND "${DISPER26_BREW}" --prefix gcc
            OUTPUT_VARIABLE _gcc_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _gcc_result)
        if(_gcc_result EQUAL 0)
            file(GLOB _gcc_candidates "${_gcc_prefix}/bin/g++-[0-9]*")
            list(SORT _gcc_candidates COMPARE NATURAL ORDER DESCENDING)
            if(_gcc_candidates)
                list(GET _gcc_candidates 0 _gcc)
                set(CMAKE_CXX_COMPILER "${_gcc}" CACHE FILEPATH "C++ compiler")
                message(STATUS "Disper26: using Homebrew GCC: ${_gcc}")
            endif()
        endif()
    endif()
endif()
