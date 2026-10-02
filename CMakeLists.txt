cmake_minimum_required(VERSION 3.18)
project(ppdoclayout LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(USE_CUDA "build with CUDA backend" OFF)
option(PPDL_DIST "self-contained distributable DLL: static ggml, portable CPU baseline" OFF)
if(USE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # whole-graph capture+replay: removes per-kernel launch overhead (~1700 nodes)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()
if(PPDL_DIST)
    set(BUILD_SHARED_LIBS OFF)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
endif()

# init ggml submodule + apply the ppdl patch (FP32 parity + GPU custom ops,
# idempotent)
find_package(Git QUIET)
if(GIT_FOUND)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
        execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init ggml
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C ggml apply --check
                            "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-ppdl.patch"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    RESULT_VARIABLE PPDL_PATCH_CHECK
                    ERROR_QUIET)
    if(PPDL_PATCH_CHECK EQUAL 0)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C ggml apply
                                "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-ppdl.patch"
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        message(STATUS "applied patches/ggml-ppdl.patch to ggml")
    else()
        message(STATUS "ggml ppdl patch already applied (or not applicable)")
    endif()
endif()

add_subdirectory(ggml)

add_library(ppdoclayout_c SHARED src/ppdoclayout.cpp)
target_include_directories(ppdoclayout_c PUBLIC include)
target_compile_definitions(ppdoclayout_c PRIVATE PPDL_BUILD)
target_link_libraries(ppdoclayout_c PRIVATE ggml)

add_executable(ppdoclayout src/main.cpp)
target_link_libraries(ppdoclayout PRIVATE ppdoclayout_c)
