#include "keyframe.operators.solid.h"
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::solid {
    unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    __global__ void apply_scalar_kernel(float* scalar, const float* solid_scalar, const std::uint8_t* occupancy, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        if (occupancy[index] == 0u) return;
        scalar[index] = solid_scalar[index];
    }

    void apply_scalar(cudaStream_t stream, float* scalar, const float* solid_scalar, const std::uint8_t* occupancy, const int nx, const int ny, const int nz) {
        if (scalar == nullptr || solid_scalar == nullptr || occupancy == nullptr) throw std::runtime_error{"Solid scalar arrays must not be null"};
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error{"Solid launch resolution must be positive"};
        const auto count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (count == 0u) throw std::runtime_error{"Solid launch count must be positive"};
        constexpr unsigned block = 256u;
        const unsigned grid      = ceil_div_u32(count, block);
        apply_scalar_kernel<<<grid, block, 0, stream>>>(scalar, solid_scalar, occupancy, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"apply_scalar_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::solid
