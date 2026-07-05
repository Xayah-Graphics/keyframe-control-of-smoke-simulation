#include "keyframe.inspector.h"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfs::cuda {
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
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error{"keyframe inspector float buffer is too large."};
        const auto float_count = static_cast<std::size_t>(count);
        std::vector<float> values(float_count);
        read_float_buffer(device_data, values.data(), count, stream);

        float minimum            = std::numeric_limits<float>::infinity();
        float maximum            = -std::numeric_limits<float>::infinity();
        double sum               = 0.0;
        std::uint64_t nonzero    = 0u;
        for (const float value : values) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            sum += static_cast<double>(value);
            if (value != 0.0f) ++nonzero;
        }

        out_stats.min           = minimum;
        out_stats.max           = maximum;
        out_stats.sum           = sum;
        out_stats.mean          = static_cast<float>(sum / static_cast<double>(count));
        out_stats.nonzero_count = nonzero;
    }
} // namespace kfs::cuda
