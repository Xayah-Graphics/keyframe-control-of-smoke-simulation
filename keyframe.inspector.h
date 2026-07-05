#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_INSPECTOR_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_INSPECTOR_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda {
    struct ScalarReadbackStats final {
        float min{0.0f};
        float max{0.0f};
        double sum{0.0};
        float mean{0.0f};
        std::uint64_t nonzero_count{0u};
    };

    void read_float_buffer(const float* device_data, float* host_data, std::uint64_t count, cudaStream_t stream);
    void read_scalar_field_stats(const float* device_data, std::uint64_t count, cudaStream_t stream, ScalarReadbackStats& out_stats);
} // namespace kfs::cuda

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_INSPECTOR_H
