#include "keyframe.operators.emitter.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {
    __global__ void emit_density_temperature_kernel(float* density_destination, const float* density_current, float* temperature_destination, const float* temperature_current, const int nx, const int ny, const int nz, const float cell_size, const float delta_seconds, const float center_x, const float center_y, const float center_z, const float radius_x, const float radius_y, const float radius_z, const float density_rate, const float temperature_rate, const float falloff) {
        const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        const std::uint32_t cell_count = static_cast<std::uint32_t>(nx) * static_cast<std::uint32_t>(ny) * static_cast<std::uint32_t>(nz);
        if (index >= cell_count) return;

        const int x    = static_cast<int>(index % static_cast<std::uint32_t>(nx));
        const int yz   = static_cast<int>(index / static_cast<std::uint32_t>(nx));
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
            density_destination[index]     = density_current[index];
            temperature_destination[index] = temperature_current[index];
            return;
        }

        const float emission = expf(-falloff * r2);
        density_destination[index]     = density_current[index] + delta_seconds * density_rate * emission;
        temperature_destination[index] = temperature_current[index] + delta_seconds * temperature_rate * emission;
    }
} // namespace

namespace kfs::cuda::operators::emitter {
    void emit_density_temperature(const cudaStream_t stream, float* const density_destination, const float* const density_current, float* const temperature_destination, const float* const temperature_current, const int nx, const int ny, const int nz, const float cell_size, const float delta_seconds, const std::array<float, 3> center, const std::array<float, 3> radius, const float density_rate, const float temperature_rate, const float falloff) {
        constexpr std::uint32_t block = 256u;
        const std::uint32_t cell_count = static_cast<std::uint32_t>(nx) * static_cast<std::uint32_t>(ny) * static_cast<std::uint32_t>(nz);
        emit_density_temperature_kernel<<<(cell_count + block - 1u) / block, block, 0, stream>>>(density_destination, density_current, temperature_destination, temperature_current, nx, ny, nz, cell_size, delta_seconds, center[0], center[1], center[2], radius[0], radius[1], radius[2], density_rate, temperature_rate, falloff);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"emit_density_temperature_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::emitter
