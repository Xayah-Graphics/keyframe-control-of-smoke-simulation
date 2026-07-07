module;
#include "keyframe.boundary.h"

module keyframe.boundary;
import std;
import keyframe.field;

namespace kfs::boundary {
    PackedFlowBoundary pack(const FlowBoundary& boundary) {
        return PackedFlowBoundary{
            .types =
                {
                    static_cast<std::uint32_t>(boundary.x_minus.type),
                    static_cast<std::uint32_t>(boundary.x_plus.type),
                    static_cast<std::uint32_t>(boundary.y_minus.type),
                    static_cast<std::uint32_t>(boundary.y_plus.type),
                    static_cast<std::uint32_t>(boundary.z_minus.type),
                    static_cast<std::uint32_t>(boundary.z_plus.type),
                },
            .velocity =
                {
                    boundary.x_minus.velocity_x,
                    boundary.x_minus.velocity_y,
                    boundary.x_minus.velocity_z,
                    boundary.x_plus.velocity_x,
                    boundary.x_plus.velocity_y,
                    boundary.x_plus.velocity_z,
                    boundary.y_minus.velocity_x,
                    boundary.y_minus.velocity_y,
                    boundary.y_minus.velocity_z,
                    boundary.y_plus.velocity_x,
                    boundary.y_plus.velocity_y,
                    boundary.y_plus.velocity_z,
                    boundary.z_minus.velocity_x,
                    boundary.z_minus.velocity_y,
                    boundary.z_minus.velocity_z,
                    boundary.z_plus.velocity_x,
                    boundary.z_plus.velocity_y,
                    boundary.z_plus.velocity_z,
                },
            .pressure =
                {
                    boundary.x_minus.pressure,
                    boundary.x_plus.pressure,
                    boundary.y_minus.pressure,
                    boundary.y_plus.pressure,
                    boundary.z_minus.pressure,
                    boundary.z_plus.pressure,
                },
            .periodic =
                {
                    boundary.x_minus.type == FlowBoundaryType::periodic && boundary.x_plus.type == FlowBoundaryType::periodic,
                    boundary.y_minus.type == FlowBoundaryType::periodic && boundary.y_plus.type == FlowBoundaryType::periodic,
                    boundary.z_minus.type == FlowBoundaryType::periodic && boundary.z_plus.type == FlowBoundaryType::periodic,
                },
        };
    }

    PackedScalarBoundary pack(const ScalarBoundary& boundary) {
        return PackedScalarBoundary{
            .types =
                {
                    static_cast<std::uint32_t>(boundary.x_minus.type),
                    static_cast<std::uint32_t>(boundary.x_plus.type),
                    static_cast<std::uint32_t>(boundary.y_minus.type),
                    static_cast<std::uint32_t>(boundary.y_plus.type),
                    static_cast<std::uint32_t>(boundary.z_minus.type),
                    static_cast<std::uint32_t>(boundary.z_plus.type),
                },
            .values =
                {
                    boundary.x_minus.value,
                    boundary.x_plus.value,
                    boundary.y_minus.value,
                    boundary.y_plus.value,
                    boundary.z_minus.value,
                    boundary.z_plus.value,
                },
            .periodic =
                {
                    boundary.x_minus.type == ScalarBoundaryType::periodic && boundary.x_plus.type == ScalarBoundaryType::periodic,
                    boundary.y_minus.type == ScalarBoundaryType::periodic && boundary.y_plus.type == ScalarBoundaryType::periodic,
                    boundary.z_minus.type == ScalarBoundaryType::periodic && boundary.z_plus.type == ScalarBoundaryType::periodic,
                },
        };
    }

    void enforce_staggered_boundary(const cudaStream_t stream, const std::uint32_t axis, field::StaggeredVectorField3D& values, const field::IndexedField3D& cell_indices, const field::CenteredVectorField3D& constraint_velocity, const PackedFlowBoundary& boundary) {
        cuda::boundary::enforce_staggered_boundary(stream, axis, values.data[axis], cell_indices.data, constraint_velocity.data[axis], values.resolution[0], values.resolution[1], values.resolution[2], boundary.types.data(), boundary.velocity.data());
    }

    void sync_periodic_staggered_component(const cudaStream_t stream, const std::uint32_t axis, field::StaggeredVectorField3D& values) {
        cuda::boundary::sync_periodic_staggered_component(stream, axis, values.data[axis], values.resolution[0], values.resolution[1], values.resolution[2]);
    }

    void boundary_fill_centered_scalar(const cudaStream_t stream, field::ScalarField3D& destination, const field::ScalarField3D& source, const field::IndexedField3D& cell_indices, const PackedScalarBoundary& boundary) {
        cuda::boundary::boundary_fill_centered_scalar(stream, destination.data, source.data, cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], boundary.types.data(), boundary.values.data());
    }
} // namespace kfs::boundary
