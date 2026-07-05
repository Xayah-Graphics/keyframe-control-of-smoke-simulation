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
        struct StaggeredVectorField3D;

        struct ScalarField3D final {
            explicit ScalarField3D(std::array<std::int32_t, 3> resolution);
            ~ScalarField3D() noexcept;
            ScalarField3D(const ScalarField3D&)            = delete;
            ScalarField3D& operator=(const ScalarField3D&) = delete;
            ScalarField3D(ScalarField3D&& other) noexcept;
            ScalarField3D& operator=(ScalarField3D&& other) noexcept;

            [[nodiscard]] std::uint64_t count() const;
            [[nodiscard]] std::size_t bytes() const;

            void resize(std::array<std::int32_t, 3> resolution);
            void fill(cudaStream_t stream, float value);

            std::array<std::int32_t, 3> resolution;
            float* data{nullptr};
        };

        struct CenteredVectorField3D final {
            explicit CenteredVectorField3D(std::array<std::int32_t, 3> resolution);
            ~CenteredVectorField3D() noexcept;
            CenteredVectorField3D(const CenteredVectorField3D&)            = delete;
            CenteredVectorField3D& operator=(const CenteredVectorField3D&) = delete;
            CenteredVectorField3D(CenteredVectorField3D&& other) noexcept;
            CenteredVectorField3D& operator=(CenteredVectorField3D&& other) noexcept;

            [[nodiscard]] std::uint64_t count() const;
            [[nodiscard]] std::size_t bytes() const;

            void resize(std::array<std::int32_t, 3> resolution);
            void fill(cudaStream_t stream, float value);

            std::array<std::int32_t, 3> resolution;
            std::array<float*, 3> data{};
        };

        struct StaggeredVectorField3D final {
            explicit StaggeredVectorField3D(std::array<std::int32_t, 3> resolution);
            ~StaggeredVectorField3D() noexcept;
            StaggeredVectorField3D(const StaggeredVectorField3D&)            = delete;
            StaggeredVectorField3D& operator=(const StaggeredVectorField3D&) = delete;
            StaggeredVectorField3D(StaggeredVectorField3D&& other) noexcept;
            StaggeredVectorField3D& operator=(StaggeredVectorField3D&& other) noexcept;

            [[nodiscard]] std::uint64_t count(std::uint32_t axis) const;
            [[nodiscard]] std::size_t bytes(std::uint32_t axis) const;

            void resize(std::array<std::int32_t, 3> resolution);
            void fill(cudaStream_t stream, float value);

            std::array<std::int32_t, 3> resolution;
            std::array<float*, 3> data{};
        };

        void copy(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source);
        void add_scaled(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& current, const ScalarField3D& source, float scale);
        void center_staggered(cudaStream_t stream, CenteredVectorField3D& destination, const StaggeredVectorField3D& source);
        void add_centered_to_staggered(cudaStream_t stream, StaggeredVectorField3D& destination, std::uint32_t axis, const CenteredVectorField3D& source, float scale);
        void copy_staggered_component(cudaStream_t stream, StaggeredVectorField3D& destination, std::uint32_t axis, const StaggeredVectorField3D& source);
    } // namespace field
} // namespace kfs::cuda

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_FIELD_H
