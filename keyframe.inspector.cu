#include "keyframe.inspector.h"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfs::cuda {
    namespace {
        struct ScalarStatsBlock final {
            float min{0.0f};
            float max{0.0f};
            double sum{0.0};
            std::uint64_t nonzero_count{0u};
        };

        constexpr unsigned stats_block_size = 256u;

        __global__ void scalar_stats_blocks_kernel(const float* values, const std::uint64_t count, ScalarStatsBlock* blocks) {
            __shared__ float block_min[stats_block_size];
            __shared__ float block_max[stats_block_size];
            __shared__ double block_sum[stats_block_size];
            __shared__ std::uint64_t block_nonzero[stats_block_size];

            const unsigned thread_index = threadIdx.x;
            const auto value_index       = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(thread_index);
            float minimum                = FLT_MAX;
            float maximum                = -FLT_MAX;
            double sum                   = 0.0;
            std::uint64_t nonzero        = 0u;
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

    void read_float_buffer(const float* const device_data, float* const host_data, const std::uint64_t count, cudaStream_t stream) {
        if (count == 0u) return;
        if (device_data == nullptr) throw std::runtime_error{"keyframe inspector device buffer is null."};
        if (host_data == nullptr) throw std::runtime_error{"keyframe inspector host buffer is null."};
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error{"keyframe inspector float buffer is too large."};
        const auto float_count = static_cast<std::size_t>(count);
        if (const cudaError_t status = cudaMemcpyAsync(host_data, device_data, float_count * sizeof(float), cudaMemcpyDeviceToHost, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync inspector readback: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize inspector readback: "} + cudaGetErrorString(status)};
    }

    void read_scalar_field_stats(const float* const device_data, const std::uint64_t count, cudaStream_t stream, ScalarReadbackStats& out_stats) {
        out_stats = {};
        if (count == 0u) return;
        if (device_data == nullptr) throw std::runtime_error{"keyframe inspector device buffer is null."};
        constexpr std::uint64_t block_size = stats_block_size;
        const std::uint64_t block_count    = (count + block_size - 1u) / block_size;
        if (block_count > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) throw std::runtime_error{"keyframe inspector stats grid is too large."};
        if (block_count > std::numeric_limits<std::size_t>::max() / sizeof(ScalarStatsBlock)) throw std::runtime_error{"keyframe inspector stats buffer is too large."};

        ScalarStatsBlock* device_blocks = nullptr;
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_blocks), static_cast<std::size_t>(block_count) * sizeof(ScalarStatsBlock)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc inspector scalar stats: "} + cudaGetErrorString(status)};
        try {
            scalar_stats_blocks_kernel<<<static_cast<unsigned>(block_count), stats_block_size, 0, stream>>>(device_data, count, device_blocks);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"scalar_stats_blocks_kernel: "} + cudaGetErrorString(status)};
            std::vector<ScalarStatsBlock> blocks(static_cast<std::size_t>(block_count));
            if (const cudaError_t status = cudaMemcpyAsync(blocks.data(), device_blocks, blocks.size() * sizeof(ScalarStatsBlock), cudaMemcpyDeviceToHost, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync inspector scalar stats: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize inspector scalar stats: "} + cudaGetErrorString(status)};

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

            out_stats.min           = minimum;
            out_stats.max           = maximum;
            out_stats.sum           = sum;
            out_stats.mean          = static_cast<float>(sum / static_cast<double>(count));
            out_stats.nonzero_count = nonzero;
        } catch (...) {
            static_cast<void>(cudaFree(device_blocks));
            throw;
        }
        static_cast<void>(cudaFree(device_blocks));
    }
} // namespace kfs::cuda
