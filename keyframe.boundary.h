#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::boundary {
    void enforce(cudaStream_t stream, std::uint32_t axis, float* component, const std::uint32_t* cell_indices, const float* constraint_component, int nx, int ny, int nz, const std::uint32_t* boundary_modes, const float* boundary_values);
    void synchronize(cudaStream_t stream, std::uint32_t axis, float* component, int nx, int ny, int nz);
    void extrapolate(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* cell_indices, int nx, int ny, int nz, const std::uint32_t* boundary_modes, const float* boundary_values);
} // namespace kfs::cuda::boundary

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H
