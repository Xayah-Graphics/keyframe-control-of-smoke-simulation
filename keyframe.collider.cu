#include "keyframe.collider.h"
#include "keyframe.field.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
    __global__ void rasterize_ellipsoid_kernel(
            std::uint32_t* cell_indices,
            float* velocity_x,
            float* velocity_y,
            float* velocity_z,
            float* scalar,
            const int nx,
            const int ny,
            const int nz,
            const float cell_size,
            const std::uint32_t tag,
            const float center_x,
            const float center_y,
            const float center_z,
            const float radius_x,
            const float radius_y,
            const float radius_z,
            const float constraint_velocity_x,
            const float constraint_velocity_y,
            const float constraint_velocity_z,
            const float scalar_value
    ) {
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
        if (dx * dx + dy * dy + dz * dz > 1.0f) return;

        cell_indices[index] = tag;
        velocity_x[index]   = constraint_velocity_x;
        velocity_y[index]   = constraint_velocity_y;
        velocity_z[index]   = constraint_velocity_z;
        scalar[index]       = scalar_value;
    }

    __global__ void rasterize_box_kernel(
            std::uint32_t* cell_indices,
            float* velocity_x,
            float* velocity_y,
            float* velocity_z,
            float* scalar,
            const int nx,
            const int ny,
            const int nz,
            const float cell_size,
            const std::uint32_t tag,
            const float center_x,
            const float center_y,
            const float center_z,
            const float half_extent_x,
            const float half_extent_y,
            const float half_extent_z,
            const float constraint_velocity_x,
            const float constraint_velocity_y,
            const float constraint_velocity_z,
            const float scalar_value
    ) {
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
        if (fabsf(px - center_x) > half_extent_x || fabsf(py - center_y) > half_extent_y || fabsf(pz - center_z) > half_extent_z) return;

        cell_indices[index] = tag;
        velocity_x[index]   = constraint_velocity_x;
        velocity_y[index]   = constraint_velocity_y;
        velocity_z[index]   = constraint_velocity_z;
        scalar[index]       = scalar_value;
    }
} // namespace

namespace kfs::cuda::collider {
    void rasterize_ellipsoid(
            cudaStream_t stream,
            std::uint32_t* cell_indices,
            float* velocity_x,
            float* velocity_y,
            float* velocity_z,
            float* scalar,
            const int nx,
            const int ny,
            const int nz,
            const float cell_size,
            const std::uint32_t tag,
            const std::array<float, 3> center,
            const std::array<float, 3> radius,
            const std::array<float, 3> velocity,
            const float scalar_value
    ) {
        constexpr std::uint32_t block  = 256u;
        const std::uint64_t cell_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        rasterize_ellipsoid_kernel<<<field::ceil_div_u32(cell_count, block), block, 0, stream>>>(
                cell_indices,
                velocity_x,
                velocity_y,
                velocity_z,
                scalar,
                nx,
                ny,
                nz,
                cell_size,
                tag,
                center[0],
                center[1],
                center[2],
                radius[0],
                radius[1],
                radius[2],
                velocity[0],
                velocity[1],
                velocity[2],
                scalar_value
        );
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"rasterize_ellipsoid_kernel: "} + cudaGetErrorString(status)};
    }

    void rasterize_box(
            cudaStream_t stream,
            std::uint32_t* cell_indices,
            float* velocity_x,
            float* velocity_y,
            float* velocity_z,
            float* scalar,
            const int nx,
            const int ny,
            const int nz,
            const float cell_size,
            const std::uint32_t tag,
            const std::array<float, 3> center,
            const std::array<float, 3> half_extent,
            const std::array<float, 3> velocity,
            const float scalar_value
    ) {
        constexpr std::uint32_t block  = 256u;
        const std::uint64_t cell_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        rasterize_box_kernel<<<field::ceil_div_u32(cell_count, block), block, 0, stream>>>(
                cell_indices,
                velocity_x,
                velocity_y,
                velocity_z,
                scalar,
                nx,
                ny,
                nz,
                cell_size,
                tag,
                center[0],
                center[1],
                center[2],
                half_extent[0],
                half_extent[1],
                half_extent[2],
                velocity[0],
                velocity[1],
                velocity[2],
                scalar_value
        );
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"rasterize_box_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::collider
