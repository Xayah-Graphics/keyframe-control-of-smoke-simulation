#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::boundary {
    void enforce_staggered_boundary(cudaStream_t stream, std::uint32_t axis, float* velocity_component, const std::uint32_t* cell_indices, const float* constraint_velocity_component, int nx, int ny, int nz, const std::uint32_t* flow_types, const float* flow_velocity);
    void sync_periodic_staggered_component(cudaStream_t stream, std::uint32_t axis, float* velocity_component, int nx, int ny, int nz);
    void boundary_fill_centered_scalar(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* cell_indices, int nx, int ny, int nz, const std::uint32_t* scalar_boundary_types, const float* scalar_boundary_values);
} // namespace kfs::cuda::boundary

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H
