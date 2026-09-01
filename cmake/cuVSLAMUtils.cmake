# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

include(CMakeParseArguments)

# Create an INTERFACE library target with common cuVSLAM settings that all targets should link to.
# This target includes compile definitions, compiler flags, and linker options.
macro(setup_cuvslam_settings)
    # Only create the interface target once
    if(TARGET cuvslam_settings)
        return()
    endif()

    # Set global CMake properties
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    # Note: we make RelWithDebInfo a *debug* config but with optimization options
    # this way we can have assertions and other debug checks enabled but have fast execution
    string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

    # Create the interface library that all targets will link to
    add_library(cuvslam_settings INTERFACE)

    # Guarantee that Eigen is licensed under the MPL2 (and other more permissive licenses)
    target_compile_definitions(cuvslam_settings INTERFACE EIGEN_MPL2_ONLY)

    # Optional feature definitions using generator expressions
    target_compile_definitions(cuvslam_settings INTERFACE
        $<$<BOOL:${CUVSLAM_LOG_ENABLE}>:CUVSLAM_LOG_ENABLE>
        $<$<BOOL:${USE_SLAM_OUTPUT}>:USE_SLAM_OUTPUT>
        $<$<BOOL:${USE_LMDB}>:USE_LMDB>
        $<$<BOOL:${SOF_USE_SMALLER_NCC}>:SOF_USE_SMALLER_NCC>
        $<$<BOOL:${USE_NVTX}>:USE_NVTX>
        $<$<BOOL:${USE_CUDA}>:USE_CUDA>
        $<$<BOOL:${ENFORCE_GPU}>:ENFORCE_GPU>
        $<$<BOOL:${USE_RERUN}>:USE_RERUN>
        $<$<BOOL:${USE_CUNLS}>:USE_CUNLS>
    )

    # Add 'libs/' as the root directory for all cuvslam includes
    target_include_directories(cuvslam_settings INTERFACE
        ${CMAKE_SOURCE_DIR}/libs
    )

    # Link threading library
    find_package(Threads REQUIRED)
    target_link_libraries(cuvslam_settings INTERFACE Threads::Threads)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(cuvslam_settings INTERFACE
            /W4              # Warning level 4
            /MP              # Multi-processor compilation
            /bigobj          # Increase number of sections in .obj file
            /Oi              # Generate intrinsic functions
            /wd4127          # Disable warning C4127: conditional expression is constant
            $<$<CONFIG:RelWithDebInfo>:/fp:except>  # Enable floating point exceptions in RelWithDebInfo
            $<$<BOOL:${TREAT_WARNINGS_AS_ERRORS}>:/WX>  # Treat warnings as errors
        )

        target_link_options(cuvslam_settings INTERFACE
            /ignore:4099     # PDB was not found
            /ignore:4049     # Locally defined symbol imported
            /ignore:4217     # Locally defined symbol imported in function
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cuvslam_settings INTERFACE
            -Wall
            -Wextra
            -Wno-unknown-pragmas
            -Wno-deprecated-copy
            $<$<BOOL:${TREAT_WARNINGS_AS_ERRORS}>:-Werror>
        )

        # Architecture-specific flags
        if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
            # Compile for compatible ARM hardware (Nano, TX2, Xavier, etc.)
            target_compile_options(cuvslam_settings INTERFACE -march=native)
        elseif (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64|amd64")
            # For x86_64 we avoid -march=haswell to maintain compatibility with AMD --
            # using default to be generic.
            # target_compile_options(cuvslam_settings INTERFACE -march=haswell)
        endif()

        target_link_options(cuvslam_settings INTERFACE
            $<$<CONFIG:Release>:-s>  # Strip symbols in release
            LINKER:--no-undefined
            LINKER:--no-allow-shlib-undefined
        )
    endif()

    message(VERBOSE "cuVSLAM Settings configured:")
    message(VERBOSE "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
    message(VERBOSE "  CMAKE_CXX_FLAGS_${_build_type_upper}: ${CMAKE_CXX_FLAGS_${_build_type_upper}}")

    # Print settings from cuvslam_settings INTERFACE target
    get_target_property(_compile_opts cuvslam_settings INTERFACE_COMPILE_OPTIONS)
    get_target_property(_compile_defs cuvslam_settings INTERFACE_COMPILE_DEFINITIONS)
    get_target_property(_link_opts cuvslam_settings INTERFACE_LINK_OPTIONS)
    get_target_property(_include_dirs cuvslam_settings INTERFACE_INCLUDE_DIRECTORIES)

    message(VERBOSE "  cuvslam_settings target properties:")
    message(VERBOSE "    INTERFACE_COMPILE_OPTIONS: ${_compile_opts}")
    message(VERBOSE "    INTERFACE_COMPILE_DEFINITIONS: ${_compile_defs}")
    message(VERBOSE "    INTERFACE_LINK_OPTIONS: ${_link_opts}")
    message(VERBOSE "    INTERFACE_INCLUDE_DIRECTORIES: ${_include_dirs}")

endmacro(setup_cuvslam_settings)

# function to set up our test projects
function(setup_test_project)
    set(oneValueArgs MODULE_NAME)
    set(multiValueArgs INCLUDE_DIRS HEADERS SOURCES LIBRARIES SANDBOX_SOURCES)
    cmake_parse_arguments(TEST "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # If user passes an argument that is not recognized, notify and exit
    if(TEST_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "setup_test_project was passed unrecognized arguments: ${TEST_UNPARSED_ARGUMENTS}")
    endif()

    set(TEST_LIBRARY "${TEST_MODULE_NAME}_test")

    # Build the target
    # ${TEST_SANDBOX_SOURCES} are ignored for now
    add_executable(${TEST_LIBRARY} ${TEST_HEADERS} ${TEST_SOURCES})

    # Include directories for the target
    if(TEST_INCLUDE_DIRS)
        target_include_directories(${TEST_LIBRARY} PRIVATE ${TEST_INCLUDE_DIRS})
    endif()

    # Link all libraries required for testing
    target_link_libraries(${TEST_LIBRARY} PRIVATE
        cuvslam_settings
        ${TEST_LIBRARIES}
        testing
        log
        common
        gflags::gflags
        GTest::gtest
        jsoncpp::jsoncpp)

    target_compile_definitions(${TEST_LIBRARY} PRIVATE
        CUVSLAM_TEST_ASSETS="${CMAKE_SOURCE_DIR}/test_data/"
    )

    # Add tests to CTest
    add_test(NAME ${TEST_LIBRARY} COMMAND ${TEST_LIBRARY})
endfunction()

# function to set up our static library projects
function(setup_staticlib_project)
    set(oneValueArgs MODULE_NAME)
    set(multiValueArgs INCLUDE_DIRS HEADERS SOURCES LIBRARIES)
    cmake_parse_arguments(LIB "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # If user passes an argument that is not recognized, notify and exit
    if(LIB_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "setup_staticlib_project was passed unrecognized arguments: ${LIB_UNPARSED_ARGUMENTS}")
    endif()

    # Adding headers below is not necessary for building the library.
    # It is added as a convenience (so that the headers appear in the IDE)
    add_library(${LIB_MODULE_NAME} STATIC ${LIB_HEADERS} ${LIB_SOURCES})

    # Include directories for the target
    if(LIB_INCLUDE_DIRS)
        target_include_directories(${LIB_MODULE_NAME} PUBLIC ${LIB_INCLUDE_DIRS})
    endif()

    # Link libraries (including common cuVSLAM settings, PUBLIC so consumers inherit them)
    target_link_libraries(${LIB_MODULE_NAME} PUBLIC
        cuvslam_settings
        ${LIB_LIBRARIES}
    )
endfunction()

# function to set up application projects
function(setup_app)
    set(oneValueArgs MODULE_NAME)
    set(multiValueArgs INCLUDE_DIRS HEADERS SOURCES LIBRARIES)
    cmake_parse_arguments(APP "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # If user passes an argument that is not recognized, notify and exit
    if(APP_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "setup_app was passed unrecognized arguments: ${APP_UNPARSED_ARGUMENTS}")
    endif()

    # Build the target
    add_executable(${APP_MODULE_NAME} ${APP_SOURCES} ${APP_HEADERS})

    # Include directories for the target
    if(APP_INCLUDE_DIRS)
        target_include_directories(${APP_MODULE_NAME} PRIVATE ${APP_INCLUDE_DIRS})
    endif()

    # Link all required libraries (including common cuVSLAM settings)
    target_link_libraries(${APP_MODULE_NAME} PRIVATE
        cuvslam_settings
        ${APP_LIBRARIES})
endfunction()
