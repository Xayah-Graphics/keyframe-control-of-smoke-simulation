#ifndef XAYAH_OPERATORS_EMITTER_H
#define XAYAH_OPERATORS_EMITTER_H

#include <array>
#include <cuda_runtime.h>

namespace xayah::operators::emitter::cuda {
    void emit_scalar(cudaStream_t stream, float* destination, const float* current, int nx, int ny, int nz, float cell_size, float delta_seconds, std::array<float, 3> center, std::array<float, 3> radius, float rate, float falloff);
} // namespace xayah::operators::emitter::cuda

#endif // XAYAH_OPERATORS_EMITTER_H
