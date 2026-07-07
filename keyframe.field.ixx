module;
#include <cuda_runtime.h>

export module keyframe.field;
import std;

export namespace kfs::field {
    enum class IndexSelection : std::uint32_t {
        marked   = 0u,
        unmarked = 1u,
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

        std::array<std::int32_t, 3> resolution;
        float* data{nullptr};
    };

    struct IndexedField3D final {
        explicit IndexedField3D(std::array<std::int32_t, 3> resolution);
        ~IndexedField3D() noexcept;
        IndexedField3D(const IndexedField3D&)            = delete;
        IndexedField3D& operator=(const IndexedField3D&) = delete;
        IndexedField3D(IndexedField3D&& other) noexcept;
        IndexedField3D& operator=(IndexedField3D&& other) noexcept;

        [[nodiscard]] std::uint64_t count() const;
        [[nodiscard]] std::size_t bytes() const;

        void resize(std::array<std::int32_t, 3> resolution);

        std::array<std::int32_t, 3> resolution;
        std::uint32_t* data{nullptr};
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

    struct ScalarFieldStats final {
        float min{0.0f};
        float max{0.0f};
        double sum{0.0};
        float mean{0.0f};
        std::uint64_t nonzero_count{0u};
    };

    void fill(cudaStream_t stream, ScalarField3D& values, float value);
    void fill(cudaStream_t stream, IndexedField3D& values, std::uint32_t value);
    void fill(cudaStream_t stream, CenteredVectorField3D& values, float value);
    void fill(cudaStream_t stream, StaggeredVectorField3D& values, float value);

    void copy(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source);
    void copy(cudaStream_t stream, IndexedField3D& destination, const IndexedField3D& source);
    void copy(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source, const IndexedField3D& indices, IndexSelection selection);
    void copy(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& source);
    void copy(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& source);

    void upload(cudaStream_t stream, ScalarField3D& destination, std::span<const float> source);
    void upload(cudaStream_t stream, IndexedField3D& destination, std::span<const std::uint32_t> source);
    void upload(cudaStream_t stream, CenteredVectorField3D& destination, std::array<std::span<const float>, 3> source);
    void upload(cudaStream_t stream, StaggeredVectorField3D& destination, std::array<std::span<const float>, 3> source);

    void download(cudaStream_t stream, std::span<float> destination, const ScalarField3D& source);
    void download(cudaStream_t stream, std::span<std::uint32_t> destination, const IndexedField3D& source);
    void download(cudaStream_t stream, std::array<std::span<float>, 3> destination, const CenteredVectorField3D& source);
    void download(cudaStream_t stream, std::array<std::span<float>, 3> destination, const StaggeredVectorField3D& source);

    void add(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& left, const ScalarField3D& right);
    void add(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& current, const ScalarField3D& source, float scale);
    void add(cudaStream_t stream, CenteredVectorField3D& destination, std::uint32_t axis, const ScalarField3D& source, float scale, float bias, const IndexedField3D& indices, IndexSelection selection);
    void add(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& left, const CenteredVectorField3D& right);
    void add(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& current, const CenteredVectorField3D& source, float scale);
    void add(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& left, const StaggeredVectorField3D& right);
    void add(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& current, const StaggeredVectorField3D& source, float scale);
    void add(cudaStream_t stream, StaggeredVectorField3D& destination, const CenteredVectorField3D& source, float scale);

    void subtract(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& left, const ScalarField3D& right);
    void subtract(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& left, const CenteredVectorField3D& right);
    void subtract(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& left, const StaggeredVectorField3D& right);

    void multiply(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& left, const ScalarField3D& right);
    void multiply(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& left, const CenteredVectorField3D& right);
    void multiply(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& left, const StaggeredVectorField3D& right);

    void scale(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source, float factor);
    void scale(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& source, float factor);
    void scale(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& source, float factor);

    void sample(cudaStream_t stream, CenteredVectorField3D& destination, const StaggeredVectorField3D& source);

    [[nodiscard]] ScalarFieldStats stats(cudaStream_t stream, const ScalarField3D& source);
} // namespace kfs::field
