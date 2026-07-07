module;
#include "keyframe.boundary.h"

module keyframe.boundary;
import std;
import keyframe.field;

namespace kfs::boundary {
    namespace {
        bool paired_periodic(const FlowBoundaryFace& minus_face, const FlowBoundaryFace& plus_face) {
            return (minus_face.type == FlowBoundaryType::periodic) == (plus_face.type == FlowBoundaryType::periodic);
        }

        bool paired_periodic(const ScalarBoundaryFace& minus_face, const ScalarBoundaryFace& plus_face) {
            return (minus_face.type == ScalarBoundaryType::periodic) == (plus_face.type == ScalarBoundaryType::periodic);
        }

        std::array<bool, 3> flow_periodic_axes(const FlowBoundary& boundary) {
            return {
                boundary.x_minus.type == FlowBoundaryType::periodic && boundary.x_plus.type == FlowBoundaryType::periodic,
                boundary.y_minus.type == FlowBoundaryType::periodic && boundary.y_plus.type == FlowBoundaryType::periodic,
                boundary.z_minus.type == FlowBoundaryType::periodic && boundary.z_plus.type == FlowBoundaryType::periodic,
            };
        }

        std::array<bool, 3> scalar_periodic_axes(const ScalarBoundary& boundary) {
            return {
                boundary.x_minus.type == ScalarBoundaryType::periodic && boundary.x_plus.type == ScalarBoundaryType::periodic,
                boundary.y_minus.type == ScalarBoundaryType::periodic && boundary.y_plus.type == ScalarBoundaryType::periodic,
                boundary.z_minus.type == ScalarBoundaryType::periodic && boundary.z_plus.type == ScalarBoundaryType::periodic,
            };
        }

        void write_flow_face(const std::size_t index, const FlowBoundaryFace& face, PackedFlowBoundary& packed) {
            packed.types[index]              = static_cast<std::uint32_t>(face.type);
            packed.velocity[index * 3u + 0u] = face.velocity_x;
            packed.velocity[index * 3u + 1u] = face.velocity_y;
            packed.velocity[index * 3u + 2u] = face.velocity_z;
            packed.pressure[index]           = face.pressure;
        }

        void write_scalar_face(const std::size_t index, const ScalarBoundaryFace& face, PackedScalarBoundary& packed) {
            packed.types[index]  = static_cast<std::uint32_t>(face.type);
            packed.values[index] = face.value;
        }

        PackedFlowBoundary pack_flow_boundary(const FlowBoundary& boundary) {
            PackedFlowBoundary packed{};
            write_flow_face(0u, boundary.x_minus, packed);
            write_flow_face(1u, boundary.x_plus, packed);
            write_flow_face(2u, boundary.y_minus, packed);
            write_flow_face(3u, boundary.y_plus, packed);
            write_flow_face(4u, boundary.z_minus, packed);
            write_flow_face(5u, boundary.z_plus, packed);
            packed.periodic = flow_periodic_axes(boundary);
            return packed;
        }

        PackedScalarBoundary pack_scalar_boundary(const ScalarBoundary& boundary) {
            PackedScalarBoundary packed{};
            write_scalar_face(0u, boundary.x_minus, packed);
            write_scalar_face(1u, boundary.x_plus, packed);
            write_scalar_face(2u, boundary.y_minus, packed);
            write_scalar_face(3u, boundary.y_plus, packed);
            write_scalar_face(4u, boundary.z_minus, packed);
            write_scalar_face(5u, boundary.z_plus, packed);
            packed.periodic = scalar_periodic_axes(boundary);
            return packed;
        }

    } // namespace

    void validate(const DomainBoundary& boundary) {
        if (!paired_periodic(boundary.flow.x_minus, boundary.flow.x_plus) || !paired_periodic(boundary.flow.y_minus, boundary.flow.y_plus) || !paired_periodic(boundary.flow.z_minus, boundary.flow.z_plus)) throw std::runtime_error{"Keyframe smoke flow periodic boundaries must be paired"};
        if (!paired_periodic(boundary.density.x_minus, boundary.density.x_plus) || !paired_periodic(boundary.density.y_minus, boundary.density.y_plus) || !paired_periodic(boundary.density.z_minus, boundary.density.z_plus)) throw std::runtime_error{"Keyframe smoke density periodic boundaries must be paired"};
        if (!paired_periodic(boundary.temperature.x_minus, boundary.temperature.x_plus) || !paired_periodic(boundary.temperature.y_minus, boundary.temperature.y_plus) || !paired_periodic(boundary.temperature.z_minus, boundary.temperature.z_plus)) throw std::runtime_error{"Keyframe smoke temperature periodic boundaries must be paired"};
    }

    PackedDomainBoundary pack(const DomainBoundary& boundary) {
        validate(boundary);
        return PackedDomainBoundary{
            .flow        = pack_flow_boundary(boundary.flow),
            .density     = pack_scalar_boundary(boundary.density),
            .temperature = pack_scalar_boundary(boundary.temperature),
        };
    }

    void enforce_staggered_boundary(const cudaStream_t stream, const std::uint32_t axis, field::StaggeredVectorField3D& values, const field::IndexedField3D& cell_indices, const field::CenteredVectorField3D& solid_velocity, const PackedFlowBoundary& boundary) {
        if (stream == nullptr) throw std::runtime_error{"Boundary stream must not be null"};
        if (axis >= 3u) throw std::runtime_error{"Boundary axis must be 0, 1, or 2"};
        if (values.resolution != solid_velocity.resolution || values.resolution != cell_indices.resolution) throw std::runtime_error{"Boundary field resolution mismatch"};
        if (values.count(axis) == 0u || values.data[axis] == nullptr) throw std::runtime_error{"velocity field component is empty"};
        if (cell_indices.count() == 0u || cell_indices.data == nullptr) throw std::runtime_error{"cell_indices field is empty"};
        if (solid_velocity.count() == 0u) throw std::runtime_error{"solid_velocity field is empty"};
        for (std::uint32_t solid_axis = 0u; solid_axis < 3u; ++solid_axis)
            if (solid_velocity.data[solid_axis] == nullptr) throw std::runtime_error{"solid_velocity field component is empty"};
        cuda::boundary::enforce_staggered_boundary(stream, axis, values.data[axis], cell_indices.data, solid_velocity.data[axis], values.resolution[0], values.resolution[1], values.resolution[2], boundary.types.data(), boundary.velocity.data());
    }

    void sync_periodic_staggered_component(const cudaStream_t stream, const std::uint32_t axis, field::StaggeredVectorField3D& values) {
        if (stream == nullptr) throw std::runtime_error{"Boundary stream must not be null"};
        if (axis >= 3u) throw std::runtime_error{"Boundary axis must be 0, 1, or 2"};
        if (values.count(axis) == 0u || values.data[axis] == nullptr) throw std::runtime_error{"velocity field component is empty"};
        cuda::boundary::sync_periodic_staggered_component(stream, axis, values.data[axis], values.resolution[0], values.resolution[1], values.resolution[2]);
    }

    void boundary_fill_centered_scalar(const cudaStream_t stream, field::ScalarField3D& destination, const field::ScalarField3D& source, const field::IndexedField3D& cell_indices, const PackedScalarBoundary& boundary) {
        if (stream == nullptr) throw std::runtime_error{"Boundary stream must not be null"};
        if (destination.resolution != source.resolution || destination.resolution != cell_indices.resolution) throw std::runtime_error{"Boundary field resolution mismatch"};
        if (destination.count() == 0u || destination.data == nullptr) throw std::runtime_error{"destination field is empty"};
        if (source.count() == 0u || source.data == nullptr) throw std::runtime_error{"source field is empty"};
        if (cell_indices.count() == 0u || cell_indices.data == nullptr) throw std::runtime_error{"cell_indices field is empty"};
        cuda::boundary::boundary_fill_centered_scalar(stream, destination.data, source.data, cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], boundary.types.data(), boundary.values.data());
    }
} // namespace kfs::boundary
