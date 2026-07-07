module;
#include <cuda_runtime.h>

export module keyframe.boundary;
import std;
import keyframe.field;

export namespace kfs::boundary {
    enum class FlowBoundaryType : std::uint32_t {
        no_slip_wall   = 0,
        free_slip_wall = 1,
        outflow        = 2,
        periodic       = 3,
    };

    enum class ScalarBoundaryType : std::uint32_t {
        fixed_value = 0,
        zero_flux   = 1,
        periodic    = 2,
    };

    struct FlowBoundaryFace final {
        FlowBoundaryType type{FlowBoundaryType::no_slip_wall};
        float velocity_x{0.0f};
        float velocity_y{0.0f};
        float velocity_z{0.0f};
        float pressure{0.0f};
    };

    struct FlowBoundary final {
        FlowBoundaryFace x_minus{FlowBoundaryType::periodic};
        FlowBoundaryFace x_plus{FlowBoundaryType::periodic};
        FlowBoundaryFace y_minus{FlowBoundaryType::no_slip_wall};
        FlowBoundaryFace y_plus{FlowBoundaryType::outflow};
        FlowBoundaryFace z_minus{FlowBoundaryType::periodic};
        FlowBoundaryFace z_plus{FlowBoundaryType::periodic};
    };

    struct ScalarBoundaryFace final {
        ScalarBoundaryType type{ScalarBoundaryType::fixed_value};
        float value{0.0f};
    };

    struct ScalarBoundary final {
        ScalarBoundaryFace x_minus{ScalarBoundaryType::periodic, 0.0f};
        ScalarBoundaryFace x_plus{ScalarBoundaryType::periodic, 0.0f};
        ScalarBoundaryFace y_minus{ScalarBoundaryType::fixed_value, 0.0f};
        ScalarBoundaryFace y_plus{ScalarBoundaryType::fixed_value, 0.0f};
        ScalarBoundaryFace z_minus{ScalarBoundaryType::periodic, 0.0f};
        ScalarBoundaryFace z_plus{ScalarBoundaryType::periodic, 0.0f};
    };

    struct DomainBoundary final {
        FlowBoundary flow{};
        ScalarBoundary density{};
        ScalarBoundary temperature{};
    };

    struct PackedFlowBoundary final {
        std::array<std::uint32_t, 6> types{};
        std::array<float, 18> velocity{};
        std::array<float, 6> pressure{};
        std::array<bool, 3> periodic{};
    };

    struct PackedScalarBoundary final {
        std::array<std::uint32_t, 6> types{};
        std::array<float, 6> values{};
        std::array<bool, 3> periodic{};
    };

    struct PackedDomainBoundary final {
        PackedFlowBoundary flow{};
        PackedScalarBoundary density{};
        PackedScalarBoundary temperature{};
    };

    void validate(const DomainBoundary& boundary);
    [[nodiscard]] PackedDomainBoundary pack(const DomainBoundary& boundary);

    void enforce_staggered_boundary(cudaStream_t stream, std::uint32_t axis, field::StaggeredVectorField3D& values, const field::IndexedField3D& cell_indices, const field::CenteredVectorField3D& solid_velocity, const PackedFlowBoundary& boundary);
    void sync_periodic_staggered_component(cudaStream_t stream, std::uint32_t axis, field::StaggeredVectorField3D& values);
    void boundary_fill_centered_scalar(cudaStream_t stream, field::ScalarField3D& destination, const field::ScalarField3D& source, const field::IndexedField3D& cell_indices, const PackedScalarBoundary& boundary);
} // namespace kfs::boundary
