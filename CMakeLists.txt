cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(keyframe-control-of-smoke-simulation VERSION 1.0.0 LANGUAGES CXX CUDA)

option(KEYFRAME_BUILD_APP "Build the optional Keyframe executable" ON)
option(KEYFRAME_BUILD_PROJECT_PLUGIN "Build the optional Spectra project plugin" OFF)

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

function(download_if_needed url out_path)
    if (EXISTS "${out_path}")
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    file(DOWNLOAD
            "${url}"
            "${out_path}"
            TLS_VERIFY ON
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
endfunction()


# ============================================================================
# Libraries
# ============================================================================
add_library(libkeyframe)
target_sources(
        libkeyframe
        PRIVATE
        keyframe.field.cu
        keyframe.field.cpp
        keyframe.field.h
        keyframe.solver.cu
        keyframe.solver.cpp
        keyframe.solver.h
        keyframe.inspector.cu
        keyframe.inspector.cpp
        keyframe.inspector.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        keyframe.field.ixx
        keyframe.solver.ixx
        keyframe.inspector.ixx
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
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${KEYFRAME_UTIL_GIT_TAG}/xcli.cpp"
            "${KEYFRAME_UTIL_VENDOR_DIR}/xcli.cpp"
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
endif ()
