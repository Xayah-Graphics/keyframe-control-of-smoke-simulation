#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_VORTICITY_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_VORTICITY_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::operators::vorticity {
    void compute_vorticity(cudaStream_t stream, float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint32_t* cell_indices, int nx, int ny, int nz, float cell_size, const std::uint32_t* flow_types, const float* flow_velocity);
    void add_confinement(cudaStream_t stream, float* destination_x, float* destination_y, float* destination_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const std::uint32_t* cell_indices, int nx, int ny, int nz, float cell_size, float confinement, const std::uint32_t* flow_types);
} // namespace kfs::cuda::operators::vorticity

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_VORTICITY_H
