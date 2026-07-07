#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_FIELD_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_FIELD_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda {
    void free_device_buffers(void** pointers, std::size_t count) noexcept;

    template <typename... Pointers>
    void free_device_buffers(Pointers*&... pointers) noexcept {
        if constexpr (sizeof...(Pointers) > 0u) {
            void* raw[] = {static_cast<void*>(pointers)...};
            free_device_buffers(raw, sizeof...(Pointers));
            ((pointers = nullptr), ...);
        }
    }

    namespace field {
        void fill(cudaStream_t stream, float* values, std::uint64_t count, float value);
        void fill(cudaStream_t stream, std::uint32_t* values, std::uint64_t count, std::uint32_t value);

        void copy(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* indices, std::uint64_t count, std::uint32_t selection);

        void add(cudaStream_t stream, float* destination, const float* left, const float* right, std::uint64_t count);
        void add(cudaStream_t stream, float* destination, const float* current, const float* source, std::uint64_t count, float scale);
        void add(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* indices, std::uint64_t count, float scale, float bias, std::uint32_t selection);
        void add(cudaStream_t stream, float* staggered_x, float* staggered_y, float* staggered_z, const float* centered_x, const float* centered_y, const float* centered_z, std::array<std::int32_t, 3> resolution, float scale);

        void subtract(cudaStream_t stream, float* destination, const float* left, const float* right, std::uint64_t count);
        void multiply(cudaStream_t stream, float* destination, const float* left, const float* right, std::uint64_t count);
        void scale(cudaStream_t stream, float* destination, const float* source, std::uint64_t count, float factor);

        void sample(cudaStream_t stream, float* centered_x, float* centered_y, float* centered_z, const float* staggered_x, const float* staggered_y, const float* staggered_z, std::array<std::int32_t, 3> resolution);

        struct ScalarStats final {
            float min{0.0f};
            float max{0.0f};
            double sum{0.0};
            float mean{0.0f};
            std::uint64_t nonzero_count{0u};
        };

        void stats(cudaStream_t stream, const float* values, std::uint64_t count, ScalarStats& output);
    } // namespace field
} // namespace kfs::cuda

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_FIELD_H
