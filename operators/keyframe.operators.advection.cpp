module;
#include "keyframe.operators.advection.h"

module keyframe.operators.advection;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    Advection::Advection(const cudaStream_t stream, const float cell_size, const Advection::Scheme scheme) : stream{stream}, cell_size{cell_size}, scheme{scheme} {}

    void Advection::operator()(field::StaggeredVectorField3D& destination, const std::uint32_t axis, const field::StaggeredVectorField3D& source, const field::StaggeredVectorField3D& vector_field, const field::IndexedField3D& cell_indices, const float delta_seconds, const boundary::PackedFlowBoundary& boundary) const {
        cuda::operators::advection::advect_staggered_component(this->stream, axis, destination.data[axis], source.data[axis], vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, static_cast<std::uint32_t>(this->scheme), boundary.types.data(), boundary.velocity.data());
    }

    void Advection::operator()(field::ScalarField3D& destination, const field::ScalarField3D& source, const field::StaggeredVectorField3D& vector_field, const field::IndexedField3D& cell_indices, const float delta_seconds, const boundary::PackedScalarBoundary& scalar_boundary, const boundary::PackedFlowBoundary& vector_boundary) const {
        cuda::operators::advection::advect_centered_scalar(this->stream, destination.data, source.data, vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, static_cast<std::uint32_t>(this->scheme), scalar_boundary.types.data(), scalar_boundary.values.data(), vector_boundary.types.data(), vector_boundary.velocity.data());
    }
} // namespace kfs::operators
