cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(keyframe-control-of-smoke-simulation VERSION 1.0.0 LANGUAGES CXX CUDA)


# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()


# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit 13.2 REQUIRED)


# ============================================================================
# Libraries
# ============================================================================
add_library(keyframe)
target_sources(
        keyframe
        PRIVATE
        keyframe.cu
        keyframe.cpp
        keyframe.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        keyframe.ixx
)
target_link_libraries(keyframe PRIVATE CUDA::cublas CUDA::cublasLt)
set_property(TARGET keyframe PROPERTY CXX_MODULE_STD ON)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE keyframe)
set_property(TARGET test PROPERTY CXX_MODULE_STD ON)