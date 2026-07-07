#ifndef XAYAH_OPERATORS_ADVECTION_H
#define XAYAH_OPERATORS_ADVECTION_H

#include <cstdint>
#include <cuda_runtime.h>

namespace xayah::operators::advection::cuda {
    void advect_staggered_component(cudaStream_t stream, std::uint32_t axis, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint32_t* cell_indices, int nx, int ny, int nz, float h, float dt, std::uint32_t advection_mode, const std::uint32_t* boundary_modes, const float* boundary_values);
    void advect_centered_scalar(cudaStream_t stream, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint32_t* cell_indices, int nx, int ny, int nz, float h, float dt, std::uint32_t advection_mode, const std::uint32_t* scalar_boundary_modes, const float* scalar_boundary_values, const std::uint32_t* vector_boundary_modes, const float* vector_boundary_values);
} // namespace xayah::operators::advection::cuda

#endif // XAYAH_OPERATORS_ADVECTION_H
