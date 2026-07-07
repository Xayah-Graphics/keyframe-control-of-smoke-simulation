#include "keyframe.field.h"
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace kfs::cuda {
    void free_device_buffers(void** const pointers, const std::size_t count) noexcept {
        if (pointers == nullptr) return;
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] == nullptr) continue;
            cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }
} // namespace kfs::cuda

namespace kfs::cuda::field {
    namespace {
        unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
            return static_cast<unsigned>((value + divisor - 1u) / divisor);
        }

        __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
        }

        __device__ std::uint64_t index_staggered_x(const int i, const int j, const int k, const int nx, const int ny) {
            const auto nx64 = static_cast<std::uint64_t>(nx);
            const auto ny64 = static_cast<std::uint64_t>(ny);
            return static_cast<std::uint64_t>(k) * (nx64 + 1u) * ny64 + static_cast<std::uint64_t>(j) * (nx64 + 1u) + static_cast<std::uint64_t>(i);
        }

        __device__ std::uint64_t index_staggered_y(const int i, const int j, const int k, const int nx, const int ny) {
            const auto nx64 = static_cast<std::uint64_t>(nx);
            const auto ny64 = static_cast<std::uint64_t>(ny);
            return static_cast<std::uint64_t>(k) * nx64 * (ny64 + 1u) + static_cast<std::uint64_t>(j) * nx64 + static_cast<std::uint64_t>(i);
        }

        __device__ std::uint64_t index_staggered_z(const int i, const int j, const int k, const int nx, const int ny) {
            return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(i);
        }

        __global__ void fill_kernel(float* values, const float value, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            values[index] = value;
        }

        __global__ void copy_masked_kernel(float* destination, const float* source, const std::uint8_t* mask, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            if (mask[index] == 0u) return;
            destination[index] = source[index];
        }

        __global__ void add_scaled_kernel(float* destination, const float* current, const float* source, const float scale, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            destination[index] = current[index] + scale * source[index];
        }

        __global__ void center_staggered_kernel(float* cx, float* cy, float* cz, const float* sx, const float* sy, const float* sz, const int nx, const int ny, const int nz) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;
            const auto index = index_3d(x, y, z, nx, ny);
            cx[index]        = 0.5f * (sx[index_staggered_x(x, y, z, nx, ny)] + sx[index_staggered_x(x + 1, y, z, nx, ny)]);
            cy[index]        = 0.5f * (sy[index_staggered_y(x, y, z, nx, ny)] + sy[index_staggered_y(x, y + 1, z, nx, ny)]);
            cz[index]        = 0.5f * (sz[index_staggered_z(x, y, z, nx, ny)] + sz[index_staggered_z(x, y, z + 1, nx, ny)]);
        }

        __global__ void add_centered_to_staggered_x_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i > nx || j >= ny || k >= nz) return;

            float sum    = 0.0f;
            float weight = 0.0f;
            if (i > 0) {
                sum += source[index_3d(i - 1, j, k, nx, ny)];
                weight += 1.0f;
            }
            if (i < nx) {
                sum += source[index_3d(i, j, k, nx, ny)];
                weight += 1.0f;
            }
            if (weight > 0.0f) destination[index_staggered_x(i, j, k, nx, ny)] += scale * (sum / weight);
        }

        __global__ void add_centered_to_staggered_y_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j > ny || k >= nz) return;

            float sum    = 0.0f;
            float weight = 0.0f;
            if (j > 0) {
                sum += source[index_3d(i, j - 1, k, nx, ny)];
                weight += 1.0f;
            }
            if (j < ny) {
                sum += source[index_3d(i, j, k, nx, ny)];
                weight += 1.0f;
            }
            if (weight > 0.0f) destination[index_staggered_y(i, j, k, nx, ny)] += scale * (sum / weight);
        }

        __global__ void add_centered_to_staggered_z_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k > nz) return;

            float sum    = 0.0f;
            float weight = 0.0f;
            if (k > 0) {
                sum += source[index_3d(i, j, k - 1, nx, ny)];
                weight += 1.0f;
            }
            if (k < nz) {
                sum += source[index_3d(i, j, k, nx, ny)];
                weight += 1.0f;
            }
            if (weight > 0.0f) destination[index_staggered_z(i, j, k, nx, ny)] += scale * (sum / weight);
        }
    } // namespace

    void fill(cudaStream_t stream, float* const values, const std::uint64_t count, const float value) {
        fill_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(values, value, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_kernel: "} + cudaGetErrorString(status)};
    }

    void copy_masked(cudaStream_t stream, float* const destination, const float* const source, const std::uint8_t* const mask, const std::uint64_t count) {
        copy_masked_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, source, mask, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"copy_masked_kernel: "} + cudaGetErrorString(status)};
    }

    void add_scaled(cudaStream_t stream, float* const destination, const float* const current, const float* const source, const std::uint64_t count, const float scale) {
        add_scaled_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, current, source, scale, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_scaled_kernel: "} + cudaGetErrorString(status)};
    }

    void center_staggered(cudaStream_t stream, float* const cx, float* const cy, float* const cz, const float* const sx, const float* const sy, const float* const sz, const std::array<std::int32_t, 3> resolution) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid{
            ceil_div_u32(static_cast<std::uint64_t>(resolution[0]), block.x),
            ceil_div_u32(static_cast<std::uint64_t>(resolution[1]), block.y),
            ceil_div_u32(static_cast<std::uint64_t>(resolution[2]), block.z),
        };
        center_staggered_kernel<<<grid, block, 0, stream>>>(cx, cy, cz, sx, sy, sz, resolution[0], resolution[1], resolution[2]);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"center_staggered_kernel: "} + cudaGetErrorString(status)};
    }

    void add_centered_to_staggered(cudaStream_t stream, float* const destination, const float* const source, const std::uint32_t axis, const std::array<std::int32_t, 3> resolution, const float scale) {
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        constexpr dim3 block{8u, 8u, 4u};
        const auto nx   = static_cast<std::uint64_t>(resolution[0]);
        const auto ny   = static_cast<std::uint64_t>(resolution[1]);
        const auto nz   = static_cast<std::uint64_t>(resolution[2]);
        const dim3 grid = axis == 0u ? dim3(ceil_div_u32(nx + 1u, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz, block.z)) : axis == 1u ? dim3(ceil_div_u32(nx, block.x), ceil_div_u32(ny + 1u, block.y), ceil_div_u32(nz, block.z)) : dim3(ceil_div_u32(nx, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz + 1u, block.z));
        if (axis == 0u) add_centered_to_staggered_x_kernel<<<grid, block, 0, stream>>>(destination, source, resolution[0], resolution[1], resolution[2], scale);
        if (axis == 1u) add_centered_to_staggered_y_kernel<<<grid, block, 0, stream>>>(destination, source, resolution[0], resolution[1], resolution[2], scale);
        if (axis == 2u) add_centered_to_staggered_z_kernel<<<grid, block, 0, stream>>>(destination, source, resolution[0], resolution[1], resolution[2], scale);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_centered_to_staggered_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::field
