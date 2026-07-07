module;
#include <cuda_runtime.h>

export module keyframe.boundary;
import std;
import keyframe.field;

export namespace kfs::boundary {
    enum class ScalarBoundaryMode : std::uint32_t {
        fixed_value   = 0,
        zero_gradient = 1,
        periodic      = 2,
    };

    enum class VectorBoundaryMode : std::uint32_t {
        fixed_value                        = 0,
        zero_gradient                      = 1,
        normal_fixed_tangent_zero_gradient = 2,
        periodic                           = 3,
    };

    struct ScalarBoundaryFace final {
        ScalarBoundaryMode mode{ScalarBoundaryMode::zero_gradient};
        float value{0.0f};
    };

    struct VectorBoundaryFace final {
        VectorBoundaryMode mode{VectorBoundaryMode::zero_gradient};
        std::array<float, 3> value{};
    };

    [[nodiscard]] ScalarBoundaryFace fixed_value(float value) noexcept;
    [[nodiscard]] ScalarBoundaryFace zero_gradient() noexcept;
    [[nodiscard]] ScalarBoundaryFace periodic_scalar() noexcept;

    [[nodiscard]] VectorBoundaryFace fixed_value(std::array<float, 3> value) noexcept;
    [[nodiscard]] VectorBoundaryFace zero_gradient_vector() noexcept;
    [[nodiscard]] VectorBoundaryFace no_slip(std::array<float, 3> value = {0.0f, 0.0f, 0.0f}) noexcept;
    [[nodiscard]] VectorBoundaryFace free_slip(std::array<float, 3> value = {0.0f, 0.0f, 0.0f}) noexcept;
    [[nodiscard]] VectorBoundaryFace outflow() noexcept;
    [[nodiscard]] VectorBoundaryFace periodic_vector() noexcept;

    struct ScalarBoundary3D final {
        ScalarBoundaryFace x_min{};
        ScalarBoundaryFace x_max{};
        ScalarBoundaryFace y_min{};
        ScalarBoundaryFace y_max{};
        ScalarBoundaryFace z_min{};
        ScalarBoundaryFace z_max{};
    };

    struct VectorBoundary3D final {
        VectorBoundaryFace x_min{};
        VectorBoundaryFace x_max{};
        VectorBoundaryFace y_min{};
        VectorBoundaryFace y_max{};
        VectorBoundaryFace z_min{};
        VectorBoundaryFace z_max{};
    };

    struct PackedScalarBoundary3D final {
        std::array<std::uint32_t, 6> modes{
            static_cast<std::uint32_t>(ScalarBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(ScalarBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(ScalarBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(ScalarBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(ScalarBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(ScalarBoundaryMode::zero_gradient),
        };
        std::array<float, 6> values{};
        std::array<bool, 3> periodic{};
    };

    struct PackedVectorBoundary3D final {
        std::array<std::uint32_t, 6> modes{
            static_cast<std::uint32_t>(VectorBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(VectorBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(VectorBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(VectorBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(VectorBoundaryMode::zero_gradient),
            static_cast<std::uint32_t>(VectorBoundaryMode::zero_gradient),
        };
        std::array<float, 18> values{};
        std::array<bool, 3> periodic{};
    };

    [[nodiscard]] PackedScalarBoundary3D pack(const ScalarBoundary3D& boundary);
    [[nodiscard]] PackedVectorBoundary3D pack(const VectorBoundary3D& boundary);

    void enforce(cudaStream_t stream, std::uint32_t axis, field::StaggeredVectorField3D& values, const field::IndexedField3D& cell_indices, const field::CenteredVectorField3D& constraint_values, const PackedVectorBoundary3D& boundary);
    void synchronize(cudaStream_t stream, std::uint32_t axis, field::StaggeredVectorField3D& values);
    void extrapolate(cudaStream_t stream, field::ScalarField3D& destination, const field::ScalarField3D& source, const field::IndexedField3D& cell_indices, const PackedScalarBoundary3D& boundary);
} // namespace kfs::boundary
