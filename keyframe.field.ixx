module;
#include <cuda_runtime.h>

export module keyframe.field;
import std;

export namespace kfs::field {
    enum class ScalarFieldStorageKind : std::uint32_t {
        owned    = 0u,
        external = 1u,
    };

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
        void bind_external(std::array<std::int32_t, 3> resolution, float* data);

        std::array<std::int32_t, 3> resolution;
        float* data{nullptr};
        ScalarFieldStorageKind storage_kind{ScalarFieldStorageKind::owned};
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

        std::array<std::int32_t, 3> resolution;
        std::array<float*, 3> data{};
    };

    void fill(cudaStream_t stream, ScalarField3D& values, float value);
    void fill(cudaStream_t stream, CenteredVectorField3D& values, float value);
    void fill(cudaStream_t stream, StaggeredVectorField3D& values, float value);
    void copy(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source);
    void copy_masked(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source, const std::uint8_t* mask);
    void copy(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& source);
    void copy(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& source);
    void copy_component(cudaStream_t stream, CenteredVectorField3D& destination, std::uint32_t axis, const CenteredVectorField3D& source);
    void copy_component(cudaStream_t stream, StaggeredVectorField3D& destination, std::uint32_t axis, const StaggeredVectorField3D& source);
    void add_scaled(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& current, const ScalarField3D& source, float scale);
    void add_scaled(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& current, const CenteredVectorField3D& source, float scale);
    void add_scaled(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& current, const StaggeredVectorField3D& source, float scale);
    void upload(cudaStream_t stream, ScalarField3D& destination, std::span<const float> source);
    void upload(cudaStream_t stream, CenteredVectorField3D& destination, std::array<std::span<const float>, 3> source);
    void upload(cudaStream_t stream, StaggeredVectorField3D& destination, std::array<std::span<const float>, 3> source);
    void upload_component(cudaStream_t stream, CenteredVectorField3D& destination, std::uint32_t axis, std::span<const float> source);
    void upload_component(cudaStream_t stream, StaggeredVectorField3D& destination, std::uint32_t axis, std::span<const float> source);
    void center_staggered(cudaStream_t stream, CenteredVectorField3D& destination, const StaggeredVectorField3D& source);
    void add_centered_to_staggered(cudaStream_t stream, StaggeredVectorField3D& destination, std::uint32_t axis, const CenteredVectorField3D& source, float scale);
} // namespace kfs::field
