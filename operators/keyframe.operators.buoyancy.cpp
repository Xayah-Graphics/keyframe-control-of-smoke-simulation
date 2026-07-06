module;
#include "keyframe.operators.buoyancy.h"

module keyframe.operators.buoyancy;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    namespace {
        constexpr std::array<std::int32_t, 3> empty_resolution{0, 0, 0};

        void require_scalar_field(const field::ScalarField3D& values, const char* name) {
            if (values.resolution == empty_resolution || values.count() == 0u || values.data == nullptr) throw std::runtime_error{std::string{name} + " field is empty"};
        }

        void require_centered_field(const field::CenteredVectorField3D& values, const char* name) {
            if (values.resolution == empty_resolution || values.count() == 0u) throw std::runtime_error{std::string{name} + " field is empty"};
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                if (values.data[axis] == nullptr) throw std::runtime_error{std::string{name} + " field component is empty"};
            }
        }
    } // namespace

    Buoyancy::Buoyancy(const cudaStream_t stream, const float ambient_temperature, const float density_factor, const float temperature_factor, const boundary::PackedFlowBoundary& boundary) : stream{stream}, ambient_temperature{ambient_temperature}, density_factor{density_factor}, temperature_factor{temperature_factor}, flow_boundary{boundary} {
        if (stream == nullptr) throw std::runtime_error{"Buoyancy stream must not be null"};
        if (!std::isfinite(ambient_temperature)) throw std::runtime_error{"Buoyancy ambient_temperature must be finite"};
        if (!std::isfinite(density_factor)) throw std::runtime_error{"Buoyancy density_factor must be finite"};
        if (!std::isfinite(temperature_factor)) throw std::runtime_error{"Buoyancy temperature_factor must be finite"};
    }

    void Buoyancy::operator()(field::CenteredVectorField3D& force, const field::ScalarField3D& density, const field::ScalarField3D& temperature, const std::uint8_t* occupancy) const {
        if (force.resolution != density.resolution || force.resolution != temperature.resolution) throw std::runtime_error{"Buoyancy field resolution mismatch"};
        require_centered_field(force, "force");
        require_scalar_field(density, "density");
        require_scalar_field(temperature, "temperature");
        if (occupancy == nullptr) throw std::runtime_error{"Buoyancy occupancy must not be null"};

        cuda::operators::buoyancy::add_buoyancy(this->stream, force.data[1], density.data, temperature.data, occupancy, force.resolution[0], force.resolution[1], force.resolution[2], this->ambient_temperature, this->density_factor, this->temperature_factor, this->flow_boundary.types.data());
    }
} // namespace kfs::operators
