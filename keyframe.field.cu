#include "keyframe.field.h"
#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
        constexpr std::uint32_t selection_marked   = 0u;
        constexpr std::uint32_t selection_unmarked = 1u;

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

        __global__ void fill_indexed_kernel(std::uint32_t* values, const std::uint32_t value, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            values[index] = value;
        }

        __global__ void copy_index_selection_kernel(float* destination, const float* source, const std::uint32_t* indices, const std::uint32_t selection, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            const bool marked = indices[index] != 0u;
            if (selection == selection_marked && !marked) return;
            if (selection == selection_unmarked && marked) return;
            destination[index] = source[index];
        }

        __global__ void add_kernel(float* destination, const float* left, const float* right, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            destination[index] = left[index] + right[index];
        }

        __global__ void add_linear_combination_kernel(float* destination, const float* current, const float* source, const float scale, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            destination[index] = current[index] + scale * source[index];
        }

        __global__ void add_index_selection_kernel(float* destination, const float* source, const std::uint32_t* indices, const float scale, const float bias, const std::uint32_t selection, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            const bool marked = indices[index] != 0u;
            if (selection == selection_marked && !marked) return;
            if (selection == selection_unmarked && marked) return;
            destination[index] += scale * (source[index] + bias);
        }

        __global__ void subtract_kernel(float* destination, const float* left, const float* right, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            destination[index] = left[index] - right[index];
        }

        __global__ void multiply_kernel(float* destination, const float* left, const float* right, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            destination[index] = left[index] * right[index];
        }

        __global__ void scale_kernel(float* destination, const float* source, const float factor, const std::uint64_t count) {
            const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
            if (index >= count) return;
            destination[index] = factor * source[index];
        }

        __global__ void sample_kernel(float* cx, float* cy, float* cz, const float* sx, const float* sy, const float* sz, const int nx, const int ny, const int nz) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;
            const auto index = index_3d(x, y, z, nx, ny);
            cx[index]        = 0.5f * (sx[index_staggered_x(x, y, z, nx, ny)] + sx[index_staggered_x(x + 1, y, z, nx, ny)]);
            cy[index]        = 0.5f * (sy[index_staggered_y(x, y, z, nx, ny)] + sy[index_staggered_y(x, y + 1, z, nx, ny)]);
            cz[index]        = 0.5f * (sz[index_staggered_z(x, y, z, nx, ny)] + sz[index_staggered_z(x, y, z + 1, nx, ny)]);
        }

        __global__ void accumulate_centered_on_staggered_x_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
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

        __global__ void accumulate_centered_on_staggered_y_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
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

        __global__ void accumulate_centered_on_staggered_z_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
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

        constexpr unsigned stats_block_size = 256u;

        struct ScalarStatsBlock final {
            float min{0.0f};
            float max{0.0f};
            double sum{0.0};
            std::uint64_t nonzero_count{0u};
        };

        __global__ void scalar_stats_blocks_kernel(const float* values, const std::uint64_t count, ScalarStatsBlock* blocks) {
            __shared__ float block_min[stats_block_size];
            __shared__ float block_max[stats_block_size];
            __shared__ double block_sum[stats_block_size];
            __shared__ std::uint64_t block_nonzero[stats_block_size];

            const unsigned thread_index = threadIdx.x;
            const auto value_index      = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(thread_index);
            float minimum               = FLT_MAX;
            float maximum               = -FLT_MAX;
            double sum                  = 0.0;
            std::uint64_t nonzero       = 0u;
            if (value_index < count) {
                const float value = values[value_index];
                minimum           = value;
                maximum           = value;
                sum               = static_cast<double>(value);
                nonzero           = value != 0.0f ? 1u : 0u;
            }

            block_min[thread_index]     = minimum;
            block_max[thread_index]     = maximum;
            block_sum[thread_index]     = sum;
            block_nonzero[thread_index] = nonzero;
            __syncthreads();

            for (unsigned stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
                if (thread_index < stride) {
                    block_min[thread_index] = fminf(block_min[thread_index], block_min[thread_index + stride]);
                    block_max[thread_index] = fmaxf(block_max[thread_index], block_max[thread_index + stride]);
                    block_sum[thread_index] += block_sum[thread_index + stride];
                    block_nonzero[thread_index] += block_nonzero[thread_index + stride];
                }
                __syncthreads();
            }

            if (thread_index == 0u) {
                blocks[blockIdx.x] = ScalarStatsBlock{
                    .min           = block_min[0],
                    .max           = block_max[0],
                    .sum           = block_sum[0],
                    .nonzero_count = block_nonzero[0],
                };
            }
        }
    } // namespace

    void fill(cudaStream_t stream, float* const values, const std::uint64_t count, const float value) {
        fill_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(values, value, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_kernel: "} + cudaGetErrorString(status)};
    }

    void fill(cudaStream_t stream, std::uint32_t* const values, const std::uint64_t count, const std::uint32_t value) {
        fill_indexed_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(values, value, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_indexed_kernel: "} + cudaGetErrorString(status)};
    }

    void copy(cudaStream_t stream, float* const destination, const float* const source, const std::uint32_t* const indices, const std::uint64_t count, const std::uint32_t selection) {
        copy_index_selection_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, source, indices, selection, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"copy_index_selection_kernel: "} + cudaGetErrorString(status)};
    }

    void add(cudaStream_t stream, float* const destination, const float* const left, const float* const right, const std::uint64_t count) {
        add_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, left, right, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_kernel: "} + cudaGetErrorString(status)};
    }

    void add(cudaStream_t stream, float* const destination, const float* const current, const float* const source, const std::uint64_t count, const float scale) {
        add_linear_combination_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, current, source, scale, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_linear_combination_kernel: "} + cudaGetErrorString(status)};
    }

    void add(cudaStream_t stream, float* const destination, const float* const source, const std::uint32_t* const indices, const std::uint64_t count, const float scale, const float bias, const std::uint32_t selection) {
        add_index_selection_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, source, indices, scale, bias, selection, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_index_selection_kernel: "} + cudaGetErrorString(status)};
    }

    void add(cudaStream_t stream, float* const x, float* const y, float* const z, const float* const cx, const float* const cy, const float* const cz, const std::array<std::int32_t, 3> resolution, const float scale) {
        constexpr dim3 block{8u, 8u, 4u};
        const auto nx = static_cast<std::uint64_t>(resolution[0]);
        const auto ny = static_cast<std::uint64_t>(resolution[1]);
        const auto nz = static_cast<std::uint64_t>(resolution[2]);
        accumulate_centered_on_staggered_x_kernel<<<dim3{ceil_div_u32(nx + 1u, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz, block.z)}, block, 0, stream>>>(x, cx, resolution[0], resolution[1], resolution[2], scale);
        accumulate_centered_on_staggered_y_kernel<<<dim3{ceil_div_u32(nx, block.x), ceil_div_u32(ny + 1u, block.y), ceil_div_u32(nz, block.z)}, block, 0, stream>>>(y, cy, resolution[0], resolution[1], resolution[2], scale);
        accumulate_centered_on_staggered_z_kernel<<<dim3{ceil_div_u32(nx, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz + 1u, block.z)}, block, 0, stream>>>(z, cz, resolution[0], resolution[1], resolution[2], scale);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"accumulate_centered_on_staggered_kernel: "} + cudaGetErrorString(status)};
    }

    void subtract(cudaStream_t stream, float* const destination, const float* const left, const float* const right, const std::uint64_t count) {
        subtract_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, left, right, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"subtract_kernel: "} + cudaGetErrorString(status)};
    }

    void multiply(cudaStream_t stream, float* const destination, const float* const left, const float* const right, const std::uint64_t count) {
        multiply_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, left, right, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"multiply_kernel: "} + cudaGetErrorString(status)};
    }

    void scale(cudaStream_t stream, float* const destination, const float* const source, const std::uint64_t count, const float factor) {
        scale_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination, source, factor, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"scale_kernel: "} + cudaGetErrorString(status)};
    }

    void sample(cudaStream_t stream, float* const cx, float* const cy, float* const cz, const float* const sx, const float* const sy, const float* const sz, const std::array<std::int32_t, 3> resolution) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid{
            ceil_div_u32(static_cast<std::uint64_t>(resolution[0]), block.x),
            ceil_div_u32(static_cast<std::uint64_t>(resolution[1]), block.y),
            ceil_div_u32(static_cast<std::uint64_t>(resolution[2]), block.z),
        };
        sample_kernel<<<grid, block, 0, stream>>>(cx, cy, cz, sx, sy, sz, resolution[0], resolution[1], resolution[2]);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"sample_kernel: "} + cudaGetErrorString(status)};
    }

    void stats(cudaStream_t stream, const float* const values, const std::uint64_t count, ScalarStats& output) {
        output = {};
        if (count == 0u) return;

        const std::uint64_t block_count = ceil_div_u32(count, stats_block_size);
        ScalarStatsBlock* device_blocks = nullptr;
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_blocks), static_cast<std::size_t>(block_count) * sizeof(ScalarStatsBlock)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc scalar field stats: "} + cudaGetErrorString(status)};
        try {
            scalar_stats_blocks_kernel<<<static_cast<unsigned>(block_count), stats_block_size, 0, stream>>>(values, count, device_blocks);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"scalar_stats_blocks_kernel: "} + cudaGetErrorString(status)};

            std::vector<ScalarStatsBlock> blocks(static_cast<std::size_t>(block_count));
            if (const cudaError_t status = cudaMemcpyAsync(blocks.data(), device_blocks, blocks.size() * sizeof(ScalarStatsBlock), cudaMemcpyDeviceToHost, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync scalar field stats: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize scalar field stats: "} + cudaGetErrorString(status)};

            float minimum         = std::numeric_limits<float>::infinity();
            float maximum         = -std::numeric_limits<float>::infinity();
            double sum            = 0.0;
            std::uint64_t nonzero = 0u;
            for (const ScalarStatsBlock& block : blocks) {
                minimum = std::min(minimum, block.min);
                maximum = std::max(maximum, block.max);
                sum += block.sum;
                nonzero += block.nonzero_count;
            }

            output.min           = minimum;
            output.max           = maximum;
            output.sum           = sum;
            output.mean          = static_cast<float>(sum / static_cast<double>(count));
            output.nonzero_count = nonzero;
        } catch (...) {
            static_cast<void>(cudaFree(device_blocks));
            throw;
        }
        static_cast<void>(cudaFree(device_blocks));
    }
} // namespace kfs::cuda::field
