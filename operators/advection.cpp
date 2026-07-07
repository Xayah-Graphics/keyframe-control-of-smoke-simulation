module;
#include "advection.h"

module xayah.operators.advection;
import std;
import xayah.core.field;
import xayah.core.boundary;

namespace xayah::operators {
    Advection::Advection(cudaStream_t stream, const float cell_size, const Advection::Scheme scheme) : stream{stream}, cell_size{cell_size}, scheme{scheme} {}

    void Advection::operator()(core::field::StaggeredVectorField3D& destination, const std::uint32_t axis, const core::field::StaggeredVectorField3D& source, const core::field::StaggeredVectorField3D& vector_field, const core::field::IndexedField3D& cell_indices, const float delta_seconds, const core::boundary::PackedVectorBoundary3D& boundary) const {
        advection::cuda::advect_staggered_component(this->stream, axis, destination.data[axis], source.data[axis], vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, static_cast<std::uint32_t>(this->scheme), boundary.modes.data(), boundary.values.data());
    }

    void Advection::operator()(core::field::ScalarField3D& destination, const core::field::ScalarField3D& source, const core::field::StaggeredVectorField3D& vector_field, const core::field::IndexedField3D& cell_indices, const float delta_seconds, const core::boundary::PackedScalarBoundary3D& scalar_boundary, const core::boundary::PackedVectorBoundary3D& vector_boundary) const {
        advection::cuda::advect_centered_scalar(this->stream, destination.data, source.data, vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, static_cast<std::uint32_t>(this->scheme), scalar_boundary.modes.data(), scalar_boundary.values.data(), vector_boundary.modes.data(), vector_boundary.values.data());
    }
} // namespace xayah::operators
