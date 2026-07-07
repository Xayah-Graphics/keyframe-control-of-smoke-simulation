#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_MASKED_SCALAR_ASSIGNMENT_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_MASKED_SCALAR_ASSIGNMENT_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::operators::masked_scalar_assignment {
    void assign_masked_scalar(cudaStream_t stream, float* destination, const float* source, const std::uint8_t* mask, int nx, int ny, int nz);
} // namespace kfs::cuda::operators::masked_scalar_assignment

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_MASKED_SCALAR_ASSIGNMENT_H
