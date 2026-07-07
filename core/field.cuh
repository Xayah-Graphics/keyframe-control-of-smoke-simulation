#ifndef XAYAH_CORE_FIELD_CUH
#define XAYAH_CORE_FIELD_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace xayah::core::field::cuda {
    __host__ __device__ inline unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) noexcept {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    __host__ __device__ inline std::uint64_t index(const int x, const int y, const int z, const int nx, const int ny) noexcept {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(x);
    }

    __host__ __device__ inline int extent(const std::uint32_t axis, const std::uint32_t dimension, const int nx, const int ny, const int nz) noexcept {
        const int cells = dimension == 0u ? nx : dimension == 1u ? ny : nz;
        return cells + (axis == dimension ? 1 : 0);
    }

    __host__ __device__ inline std::uint64_t index(const std::uint32_t axis, const int i, const int j, const int k, const int nx, const int ny) noexcept {
        const auto nx64 = static_cast<std::uint64_t>(nx);
        const auto ny64 = static_cast<std::uint64_t>(ny);
        if (axis == 0u) return static_cast<std::uint64_t>(k) * (nx64 + 1u) * ny64 + static_cast<std::uint64_t>(j) * (nx64 + 1u) + static_cast<std::uint64_t>(i);
        if (axis == 1u) return static_cast<std::uint64_t>(k) * nx64 * (ny64 + 1u) + static_cast<std::uint64_t>(j) * nx64 + static_cast<std::uint64_t>(i);
        return static_cast<std::uint64_t>(k) * nx64 * ny64 + static_cast<std::uint64_t>(j) * nx64 + static_cast<std::uint64_t>(i);
    }

    inline dim3 centered_grid(const int nx, const int ny, const int nz, const dim3 block) noexcept {
        return dim3{ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z)};
    }

    inline dim3 staggered_grid(const std::uint32_t axis, const int nx, const int ny, const int nz, const dim3 block) noexcept {
        return dim3{ceil_div_u32(static_cast<std::uint64_t>(extent(axis, 0u, nx, ny, nz)), block.x), ceil_div_u32(static_cast<std::uint64_t>(extent(axis, 1u, nx, ny, nz)), block.y), ceil_div_u32(static_cast<std::uint64_t>(extent(axis, 2u, nx, ny, nz)), block.z)};
    }
} // namespace xayah::core::field::cuda

#endif // XAYAH_CORE_FIELD_CUH
