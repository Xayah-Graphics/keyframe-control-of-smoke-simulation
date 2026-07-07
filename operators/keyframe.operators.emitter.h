#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_EMITTER_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_EMITTER_H

#include <array>
#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::operators::emitter {
    void emit_scalar(cudaStream_t stream, float* destination, const float* current, int nx, int ny, int nz, float cell_size, float delta_seconds, std::array<float, 3> center, std::array<float, 3> radius, float rate, float falloff);
} // namespace kfs::cuda::operators::emitter

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_OPERATORS_EMITTER_H
