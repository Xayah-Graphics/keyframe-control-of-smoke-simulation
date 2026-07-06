#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_SOLID_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_SOLID_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::operators::solid {
    void apply_scalar(cudaStream_t stream, float* scalar, const float* solid_scalar, const std::uint8_t* occupancy, int nx, int ny, int nz);
} // namespace kfs::cuda::operators::solid

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_SOLID_H
