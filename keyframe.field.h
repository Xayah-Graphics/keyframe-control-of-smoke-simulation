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
        void copy_masked(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* indices, std::uint64_t count);
        void add_scaled(cudaStream_t stream, float* destination, const float* current, const float* source, std::uint64_t count, float scale);
        void add_unmasked_scalar_to_component(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* indices, std::uint64_t count, float scale, float bias);
        void center_staggered(cudaStream_t stream, float* cx, float* cy, float* cz, const float* sx, const float* sy, const float* sz, std::array<std::int32_t, 3> resolution);
        void add_centered_to_staggered(cudaStream_t stream, float* destination, const float* source, std::uint32_t axis, std::array<std::int32_t, 3> resolution, float scale);
    } // namespace field
} // namespace kfs::cuda

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_FIELD_H
