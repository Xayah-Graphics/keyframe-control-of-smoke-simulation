#include "../keyframe.field.cuh"
#include "keyframe.operators.emitter.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
    __global__ void emit_scalar_kernel(float* destination, const float* current, const int nx, const int ny, const int nz, const float cell_size, const float delta_seconds, const float center_x, const float center_y, const float center_z, const float radius_x, const float radius_y, const float radius_z, const float rate, const float falloff) {
        const auto index               = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        const std::uint64_t cell_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (index >= cell_count) return;

        const int x    = static_cast<int>(index % static_cast<std::uint64_t>(nx));
        const int yz   = static_cast<int>(index / static_cast<std::uint64_t>(nx));
        const int y    = yz % ny;
        const int z    = yz / ny;
        const float px = (static_cast<float>(x) + 0.5f) * cell_size;
        const float py = (static_cast<float>(y) + 0.5f) * cell_size;
        const float pz = (static_cast<float>(z) + 0.5f) * cell_size;
        const float dx = (px - center_x) / radius_x;
        const float dy = (py - center_y) / radius_y;
        const float dz = (pz - center_z) / radius_z;
        const float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 1.0f) {
            destination[index] = current[index];
            return;
        }

        const float emission = expf(-falloff * r2);
        destination[index]   = current[index] + delta_seconds * rate * emission;
    }
} // namespace

namespace kfs::cuda::operators::emitter {
    void emit_scalar(const cudaStream_t stream, float* const destination, const float* const current, const int nx, const int ny, const int nz, const float cell_size, const float delta_seconds, const std::array<float, 3> center, const std::array<float, 3> radius, const float rate, const float falloff) {
        constexpr std::uint32_t block  = 256u;
        const std::uint64_t cell_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        emit_scalar_kernel<<<field::ceil_div_u32(cell_count, block), block, 0, stream>>>(destination, current, nx, ny, nz, cell_size, delta_seconds, center[0], center[1], center[2], radius[0], radius[1], radius[2], rate, falloff);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"emit_scalar_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::emitter
