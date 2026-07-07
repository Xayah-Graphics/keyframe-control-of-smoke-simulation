module;
#include "boundary.h"

module xayah.core.boundary;
import std;
import xayah.core.field;

namespace xayah::core::boundary {
    PackedScalarBoundary3D pack(const ScalarBoundary3D& boundary) {
        return PackedScalarBoundary3D{
            .modes =
                {
                    static_cast<std::uint32_t>(boundary.x_min.mode),
                    static_cast<std::uint32_t>(boundary.x_max.mode),
                    static_cast<std::uint32_t>(boundary.y_min.mode),
                    static_cast<std::uint32_t>(boundary.y_max.mode),
                    static_cast<std::uint32_t>(boundary.z_min.mode),
                    static_cast<std::uint32_t>(boundary.z_max.mode),
                },
            .values =
                {
                    boundary.x_min.value,
                    boundary.x_max.value,
                    boundary.y_min.value,
                    boundary.y_max.value,
                    boundary.z_min.value,
                    boundary.z_max.value,
                },
            .periodic =
                {
                    boundary.x_min.mode == ScalarBoundaryMode::periodic && boundary.x_max.mode == ScalarBoundaryMode::periodic,
                    boundary.y_min.mode == ScalarBoundaryMode::periodic && boundary.y_max.mode == ScalarBoundaryMode::periodic,
                    boundary.z_min.mode == ScalarBoundaryMode::periodic && boundary.z_max.mode == ScalarBoundaryMode::periodic,
                },
        };
    }

    PackedVectorBoundary3D pack(const VectorBoundary3D& boundary) {
        return PackedVectorBoundary3D{
            .modes =
                {
                    static_cast<std::uint32_t>(boundary.x_min.mode),
                    static_cast<std::uint32_t>(boundary.x_max.mode),
                    static_cast<std::uint32_t>(boundary.y_min.mode),
                    static_cast<std::uint32_t>(boundary.y_max.mode),
                    static_cast<std::uint32_t>(boundary.z_min.mode),
                    static_cast<std::uint32_t>(boundary.z_max.mode),
                },
            .values =
                {
                    boundary.x_min.value[0],
                    boundary.x_min.value[1],
                    boundary.x_min.value[2],
                    boundary.x_max.value[0],
                    boundary.x_max.value[1],
                    boundary.x_max.value[2],
                    boundary.y_min.value[0],
                    boundary.y_min.value[1],
                    boundary.y_min.value[2],
                    boundary.y_max.value[0],
                    boundary.y_max.value[1],
                    boundary.y_max.value[2],
                    boundary.z_min.value[0],
                    boundary.z_min.value[1],
                    boundary.z_min.value[2],
                    boundary.z_max.value[0],
                    boundary.z_max.value[1],
                    boundary.z_max.value[2],
                },
            .periodic =
                {
                    boundary.x_min.mode == VectorBoundaryMode::periodic && boundary.x_max.mode == VectorBoundaryMode::periodic,
                    boundary.y_min.mode == VectorBoundaryMode::periodic && boundary.y_max.mode == VectorBoundaryMode::periodic,
                    boundary.z_min.mode == VectorBoundaryMode::periodic && boundary.z_max.mode == VectorBoundaryMode::periodic,
                },
        };
    }

    void enforce(const cudaStream_t stream, const std::uint32_t axis, field::StaggeredVectorField3D& values, const field::IndexedField3D& cell_indices, const field::CenteredVectorField3D& constraint_values, const PackedVectorBoundary3D& boundary) {
        cuda::enforce(stream, axis, values.data[axis], cell_indices.data, constraint_values.data[axis], values.resolution[0], values.resolution[1], values.resolution[2], boundary.modes.data(), boundary.values.data());
    }

    void synchronize(const cudaStream_t stream, const std::uint32_t axis, field::StaggeredVectorField3D& values) {
        cuda::synchronize(stream, axis, values.data[axis], values.resolution[0], values.resolution[1], values.resolution[2]);
    }

    void extrapolate(const cudaStream_t stream, field::ScalarField3D& destination, const field::ScalarField3D& source, const field::IndexedField3D& cell_indices, const PackedScalarBoundary3D& boundary) {
        cuda::extrapolate(stream, destination.data, source.data, cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], boundary.modes.data(), boundary.values.data());
    }
} // namespace xayah::core::boundary
