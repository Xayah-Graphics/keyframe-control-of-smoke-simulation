#include "keyframe.operators.masked_scalar_assignment.h"
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::masked_scalar_assignment {
    unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    __global__ void assign_masked_scalar_kernel(float* destination, const float* source, const std::uint8_t* mask, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        if (mask[index] == 0u) return;
        destination[index] = source[index];
    }

    void assign_masked_scalar(cudaStream_t stream, float* destination, const float* source, const std::uint8_t* mask, const int nx, const int ny, const int nz) {
        if (destination == nullptr || source == nullptr || mask == nullptr) throw std::runtime_error{"MaskedScalarAssignment arrays must not be null"};
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error{"MaskedScalarAssignment launch resolution must be positive"};
        const auto count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (count == 0u) throw std::runtime_error{"MaskedScalarAssignment launch count must be positive"};
        constexpr unsigned block = 256u;
        const unsigned grid      = ceil_div_u32(count, block);
        assign_masked_scalar_kernel<<<grid, block, 0, stream>>>(destination, source, mask, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"assign_masked_scalar_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::masked_scalar_assignment
