cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(keyframe-control-of-smoke-simulation VERSION 1.0.0 LANGUAGES CXX CUDA)

option(KEYFRAME_BUILD_APP "Build the optional Keyframe executable" OFF)
option(KEYFRAME_BUILD_PROJECT_PLUGIN "Build the optional Spectra project plugin" OFF)

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


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

function(download_if_needed url out_path sha256)
    string(TOLOWER "${sha256}" expected_sha256)
    if (EXISTS "${out_path}")
        file(SHA256 "${out_path}" existing_sha256)
        string(TOLOWER "${existing_sha256}" existing_sha256)
        if (NOT existing_sha256 STREQUAL expected_sha256)
            message(FATAL_ERROR "Cached file hash mismatch for ${out_path}: expected ${expected_sha256}, got ${existing_sha256}")
        endif ()
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    set(tmp_path "${out_path}.download")
    file(REMOVE "${tmp_path}")
    file(DOWNLOAD
            "${url}"
            "${tmp_path}"
            TLS_VERIFY ON
            EXPECTED_HASH "SHA256=${sha256}"
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        file(REMOVE "${tmp_path}")
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
    file(RENAME "${tmp_path}" "${out_path}")
endfunction()


# ============================================================================
# Libraries
# ============================================================================
add_library(libkeyframe)
target_sources(
        libkeyframe
        PRIVATE
        core/field.cu
        core/field.cpp
        core/field.cuh
        core/field.h
        core/boundary.cpp
        core/boundary.cu
        core/boundary.cuh
        core/boundary.h
        core/collider.cpp
        core/collider.cu
        core/collider.h
        operators/advection.cpp
        operators/advection.cu
        operators/advection.h
        operators/emitter.cpp
        operators/emitter.cu
        operators/emitter.h
        operators/projection.cpp
        operators/projection.cu
        operators/projection.h
        operators/vorticity.cpp
        operators/vorticity.cu
        operators/vorticity.h
        keyframe.solver.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        core/field.ixx
        core/geometry.ixx
        core/boundary.ixx
        core/collider.ixx
        operators/advection.ixx
        operators/emitter.ixx
        operators/projection.ixx
        operators/vorticity.ixx
        keyframe.solver.ixx
)
target_link_libraries(libkeyframe PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
set_property(TARGET libkeyframe PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Optional App
# ============================================================================
if (KEYFRAME_BUILD_APP)
    set(KEYFRAME_UTIL_GIT_TAG "bb261075e7c8a4722b23f42c3be01aa24024731a")
    set(KEYFRAME_UTIL_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/util/${KEYFRAME_UTIL_GIT_TAG}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${KEYFRAME_UTIL_GIT_TAG}/xcli.ixx"
            "${KEYFRAME_UTIL_VENDOR_DIR}/xcli.ixx"
            "2F80115AA7B739BDE4BF990EC1C0D0AFD85CF6B5610AF9A9D22A8C6686898F8D"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${KEYFRAME_UTIL_GIT_TAG}/xcli.cpp"
            "${KEYFRAME_UTIL_VENDOR_DIR}/xcli.cpp"
            "A1386CE6782CF267CD9AFB002FA66E70B7BF019CFA19ADC9C5EF0470BE445E69"
    )

    add_library(xayah-util-xcli)
    target_sources(
            xayah-util-xcli
            PRIVATE
            "${KEYFRAME_UTIL_VENDOR_DIR}/xcli.cpp"
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            "${KEYFRAME_UTIL_VENDOR_DIR}/xcli.ixx"
    )
    set_property(TARGET xayah-util-xcli PROPERTY CXX_MODULE_STD ON)

    add_executable(keyframe main.cpp)
    target_link_libraries(keyframe PRIVATE xayah-util-xcli libkeyframe)
    set_property(TARGET keyframe PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(keyframe PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()


# ============================================================================
# Optional Project Plugin
# ============================================================================
if (KEYFRAME_BUILD_PROJECT_PLUGIN)
    add_library(keyframe_project_plugin SHARED)
    target_sources(
            keyframe_project_plugin
            PRIVATE
            keyframe.project.cpp
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            keyframe.plugin.ixx
            keyframe.project.ixx
    )
    target_link_libraries(keyframe_project_plugin PRIVATE libkeyframe CUDA::cudart)
    set_property(TARGET keyframe_project_plugin PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(keyframe_project_plugin PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()
