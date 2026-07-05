cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(keyframe-control-of-smoke-simulation VERSION 1.0.0 LANGUAGES CXX CUDA)

option(KEYFRAME_BUILD_APP "Build the optional Keyframe executable" ON)

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
add_library(libkeyframe)
target_sources(
        libkeyframe
        PRIVATE
        keyframe.cu
        keyframe.cpp
        keyframe.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        keyframe.ixx
)
target_link_libraries(libkeyframe PRIVATE CUDA::cublas CUDA::cublasLt)
set_property(TARGET libkeyframe PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Optional App
# ============================================================================
if (KEYFRAME_BUILD_APP)
    add_executable(keyframe main.cpp)
    target_link_libraries(keyframe PRIVATE libkeyframe)
    set_property(TARGET keyframe PROPERTY CXX_MODULE_STD ON)
endif ()