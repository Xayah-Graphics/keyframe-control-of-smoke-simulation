module;
#include "keyframe.operators.advection.h"

module keyframe.operators.advection;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    namespace {
        constexpr std::uint32_t scalar_advection_linear          = 0u;
        constexpr std::uint32_t scalar_advection_monotonic_cubic = 1u;

        std::uint32_t raw_scheme(const Advection::Scheme scheme) {
            if (scheme == Advection::Scheme::linear) return scalar_advection_linear;
            if (scheme == Advection::Scheme::monotonic_cubic) return scalar_advection_monotonic_cubic;
            throw std::runtime_error{"Advection scheme is invalid"};
        }

    } // namespace

    Advection::Advection(const cudaStream_t stream, const float cell_size, const Advection::Scheme scheme) : stream{stream}, cell_size{cell_size}, scheme{scheme} {
        if (stream == nullptr) throw std::runtime_error{"Advection stream must not be null"};
        if (!std::isfinite(cell_size) || cell_size <= 0.0f) throw std::runtime_error{"Advection cell_size must be positive"};
        raw_scheme(scheme);
    }

    void Advection::operator()(field::StaggeredVectorField3D& destination, const std::uint32_t axis, const field::StaggeredVectorField3D& source, const field::StaggeredVectorField3D& vector_field, const field::IndexedField3D& cell_indices, const float delta_seconds, const boundary::PackedFlowBoundary& boundary) const {
        if (destination.resolution != source.resolution || destination.resolution != vector_field.resolution) throw std::runtime_error{"Advection staggered field resolution mismatch"};
        if (destination.resolution != cell_indices.resolution) throw std::runtime_error{"Advection indexed field resolution mismatch"};
        if (axis >= 3u) throw std::runtime_error{"Advection axis must be 0, 1, or 2"};
        if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        if (source.count(axis) == 0u || source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        for (std::uint32_t vector_axis = 0u; vector_axis < 3u; ++vector_axis) {
            if (vector_field.count(vector_axis) == 0u || vector_field.data[vector_axis] == nullptr) throw std::runtime_error{"vector field component is empty"};
        }
        if (cell_indices.count() == 0u || cell_indices.data == nullptr) throw std::runtime_error{"Advection cell_indices field is empty"};
        if (!std::isfinite(delta_seconds) || delta_seconds <= 0.0f) throw std::runtime_error{"Advection delta_seconds must be positive"};
        cuda::operators::advection::advect_staggered_component(this->stream, axis, destination.data[axis], source.data[axis], vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, raw_scheme(this->scheme), boundary.types.data(), boundary.velocity.data());
    }

    void Advection::operator()(field::ScalarField3D& destination, const field::ScalarField3D& source, const field::StaggeredVectorField3D& vector_field, const field::IndexedField3D& cell_indices, const float delta_seconds, const boundary::PackedScalarBoundary& scalar_boundary, const boundary::PackedFlowBoundary& vector_boundary) const {
        if (destination.resolution != source.resolution || destination.resolution != vector_field.resolution) throw std::runtime_error{"Advection scalar field resolution mismatch"};
        if (destination.resolution != cell_indices.resolution) throw std::runtime_error{"Advection indexed field resolution mismatch"};
        if (destination.count() == 0u || destination.data == nullptr) throw std::runtime_error{"destination field is empty"};
        if (source.count() == 0u || source.data == nullptr) throw std::runtime_error{"source field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (vector_field.count(axis) == 0u || vector_field.data[axis] == nullptr) throw std::runtime_error{"vector field component is empty"};
        }
        if (cell_indices.count() == 0u || cell_indices.data == nullptr) throw std::runtime_error{"Advection cell_indices field is empty"};
        if (!std::isfinite(delta_seconds) || delta_seconds <= 0.0f) throw std::runtime_error{"Advection delta_seconds must be positive"};
        cuda::operators::advection::advect_centered_scalar(this->stream, destination.data, source.data, vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_indices.data, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, raw_scheme(this->scheme), scalar_boundary.types.data(), scalar_boundary.values.data(), vector_boundary.types.data(), vector_boundary.velocity.data());
    }
} // namespace kfs::operators
