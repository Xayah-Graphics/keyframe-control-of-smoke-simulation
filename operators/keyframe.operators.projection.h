#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_PROJECTION_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_PROJECTION_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::operators::projection {
    void reset_pressure_anchor(cudaStream_t stream, int* pressure_anchor, int value);
    void find_pressure_anchor(cudaStream_t stream, int* pressure_anchor, const std::uint32_t* cell_indices, std::uint64_t count);
    void compute_pressure_rhs(cudaStream_t stream, float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint32_t* cell_indices, const int* pressure_anchor, int nx, int ny, int nz, float h, float dt, const std::uint32_t* pressure_boundary_modes, const float* pressure_boundary_values);
    void build_pressure_matrix(cudaStream_t stream, float* values, const int* row_offsets, const int* column_indices, const std::uint32_t* cell_indices, const int* pressure_anchor, int nx, int ny, int nz, const std::uint32_t* pressure_boundary_modes);
    void compute_ratio(cudaStream_t stream, float* destination, const float* numerator, const float* denominator);
    void negate_scalar(cudaStream_t stream, float* destination, const float* source);
    void project_staggered_component(cudaStream_t stream, std::uint32_t axis, float* velocity_component, const float* pressure, const std::uint32_t* cell_indices, const float* constraint_velocity_component, int nx, int ny, int nz, float h, float dt, const std::uint32_t* velocity_boundary_modes, const float* velocity_boundary_values);
} // namespace kfs::cuda::operators::projection

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_PROJECTION_H
