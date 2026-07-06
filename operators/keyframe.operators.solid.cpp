module;
#include "keyframe.operators.solid.h"

module keyframe.operators.solid;
import std;
import keyframe.field;

namespace kfs::operators {
    namespace {
        constexpr std::array<std::int32_t, 3> empty_resolution{0, 0, 0};

        void require_scalar_field(const field::ScalarField3D& values, const char* name) {
            if (values.resolution == empty_resolution || values.count() == 0u || values.data == nullptr) throw std::runtime_error{std::string{name} + " field is empty"};
        }
    } // namespace

    Solid::Solid(const cudaStream_t stream) : stream{stream} {
        if (stream == nullptr) throw std::runtime_error{"Solid stream must not be null"};
    }

    void Solid::apply_scalar(field::ScalarField3D& scalar, const field::ScalarField3D& solid_scalar, const std::uint8_t* occupancy) const {
        if (scalar.resolution != solid_scalar.resolution) throw std::runtime_error{"Solid scalar field resolution mismatch"};
        require_scalar_field(scalar, "scalar");
        require_scalar_field(solid_scalar, "solid_scalar");
        if (occupancy == nullptr) throw std::runtime_error{"Solid occupancy must not be null"};
        cuda::operators::solid::apply_scalar(this->stream, scalar.data, solid_scalar.data, occupancy, scalar.resolution[0], scalar.resolution[1], scalar.resolution[2]);
    }
} // namespace kfs::operators
