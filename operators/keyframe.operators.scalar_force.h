#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_SCALAR_FORCE_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_SCALAR_FORCE_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::operators::scalar_force {
    void add_scalar_force(cudaStream_t stream, float* destination, const float* source, const std::uint8_t* cell_mask, int nx, int ny, int nz, float scale, float bias, const std::uint32_t* flow_types);
} // namespace kfs::cuda::operators::scalar_force

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_SCALAR_FORCE_H
