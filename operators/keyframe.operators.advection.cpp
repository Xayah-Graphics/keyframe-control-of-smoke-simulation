module;
#include "keyframe.operators.advection.h"

module keyframe.operators.advection;
import std;
import keyframe.field;

namespace kfs::operators {
    namespace {
        constexpr std::array<std::int32_t, 3> empty_resolution{0, 0, 0};
        constexpr std::uint32_t scalar_advection_linear = 0u;
        constexpr std::uint32_t scalar_advection_monotonic_cubic = 1u;

        std::uint32_t raw_scheme(const Advection::Scheme scheme) {
            if (scheme == Advection::Scheme::linear) return scalar_advection_linear;
            if (scheme == Advection::Scheme::monotonic_cubic) return scalar_advection_monotonic_cubic;
            throw std::runtime_error{"Advection scheme is invalid"};
        }

        void require_scalar_field(const field::ScalarField3D& scalar, const char* name) {
            if (scalar.resolution == empty_resolution || scalar.count() == 0u || scalar.data == nullptr) throw std::runtime_error{std::string{name} + " field is empty"};
        }

        void require_staggered_component(const field::StaggeredVectorField3D& field, const std::uint32_t axis, const char* name) {
            if (axis >= 3u) throw std::runtime_error{"Advection axis must be 0, 1, or 2"};
            if (field.resolution == empty_resolution || field.count(axis) == 0u || field.data[axis] == nullptr) throw std::runtime_error{std::string{name} + " field component is empty"};
        }

        void require_staggered_field(const field::StaggeredVectorField3D& field, const char* name) {
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) require_staggered_component(field, axis, name);
        }

        void require_vector_boundary(const std::uint32_t* types, const float* values) {
            if (types == nullptr || values == nullptr) throw std::runtime_error{"Advection vector boundary arrays must not be null"};
        }

        void require_scalar_boundary(const std::uint32_t* types, const float* values) {
            if (types == nullptr || values == nullptr) throw std::runtime_error{"Advection scalar boundary arrays must not be null"};
        }

        void require_delta_seconds(const float delta_seconds) {
            if (!std::isfinite(delta_seconds) || delta_seconds <= 0.0f) throw std::runtime_error{"Advection delta_seconds must be positive"};
        }
    } // namespace

    Advection::Advection(const cudaStream_t stream, const float cell_size, const Advection::Scheme scheme) : stream{stream}, cell_size{cell_size}, scheme{scheme} {
        if (stream == nullptr) throw std::runtime_error{"Advection stream must not be null"};
        if (!std::isfinite(cell_size) || cell_size <= 0.0f) throw std::runtime_error{"Advection cell_size must be positive"};
        raw_scheme(scheme);
    }

    void Advection::operator()(field::StaggeredVectorField3D& destination, const std::uint32_t axis, const field::StaggeredVectorField3D& source, const field::StaggeredVectorField3D& vector_field, const std::uint8_t* cell_mask, const float delta_seconds, const std::uint32_t* boundary_types, const float* boundary_values) const {
        if (destination.resolution != source.resolution || destination.resolution != vector_field.resolution) throw std::runtime_error{"Advection staggered field resolution mismatch"};
        require_staggered_component(destination, axis, "destination");
        require_staggered_component(source, axis, "source");
        require_staggered_field(vector_field, "vector");
        if (cell_mask == nullptr) throw std::runtime_error{"Advection cell mask must not be null"};
        require_delta_seconds(delta_seconds);
        require_vector_boundary(boundary_types, boundary_values);
        cuda::operators::advection::advect_staggered_component(this->stream, axis, destination.data[axis], source.data[axis], vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_mask, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, raw_scheme(this->scheme), boundary_types, boundary_values);
    }

    void Advection::operator()(field::ScalarField3D& destination, const field::ScalarField3D& source, const field::StaggeredVectorField3D& vector_field, const std::uint8_t* cell_mask, const float delta_seconds, const std::uint32_t* scalar_boundary_types, const float* scalar_boundary_values, const std::uint32_t* vector_boundary_types, const float* vector_boundary_values) const {
        if (destination.resolution != source.resolution || destination.resolution != vector_field.resolution) throw std::runtime_error{"Advection scalar field resolution mismatch"};
        require_scalar_field(destination, "destination");
        require_scalar_field(source, "source");
        require_staggered_field(vector_field, "vector");
        if (cell_mask == nullptr) throw std::runtime_error{"Advection cell mask must not be null"};
        require_delta_seconds(delta_seconds);
        require_scalar_boundary(scalar_boundary_types, scalar_boundary_values);
        require_vector_boundary(vector_boundary_types, vector_boundary_values);
        cuda::operators::advection::advect_centered_scalar(this->stream, destination.data, source.data, vector_field.data[0], vector_field.data[1], vector_field.data[2], cell_mask, destination.resolution[0], destination.resolution[1], destination.resolution[2], this->cell_size, delta_seconds, raw_scheme(this->scheme), scalar_boundary_types, scalar_boundary_values, vector_boundary_types, vector_boundary_values);
    }
} // namespace kfs::operators
