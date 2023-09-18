# @copyright (c) 2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# @file CMakeLists.txt
# This is a top-level CMake script, that is tested to work only with a
# single-configuration build systems like Ninja or Unix Makefile.
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-09-18


###############################################################################
#                       THIS IS A TOP-LEVEL CMAKELISTS.txt                    #
#                                                                             #
#        It is intended to find all dependencies (required or optional)       #
#                    and set up corresponding variables                       #
###############################################################################

# Set minimum CMake version to avoid configuration problems
cmake_minimum_required(VERSION 3.24)
message(STATUS "Using CMake ${CMAKE_VERSION}")

# Define the project
project(GEMM_REDUX VERSION 1.0.0 LANGUAGES CXX)

# For easier code navigation and interaction in editors.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++-17 standard publicly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # We are not compiling ony of CUDA sources
    #enable_language(CUDA)
    # Get cublas
    find_package(CUDAToolkit 10.1)
else()
    message(FATAL_ERROR "CUDA was not found")
endif()

# Get the pkg-config
find_package(PkgConfig REQUIRED)

# Get the StarPU
pkg_check_modules(StarPU REQUIRED starpu-1.4)

# Set a single executable
add_executable(gemm_redux "gemm_redux.cc")

# Link against CUDA
target_link_libraries(gemm_redux PRIVATE CUDA::cudart CUDA::cublas ${StarPU_LDFLAGS})
target_include_directories(gemm_redux PRIVATE ${StarPU_INCLUDE_DIRS})

