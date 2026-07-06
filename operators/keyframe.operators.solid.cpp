module;
#include "keyframe.operators.solid.h"

module keyframe.operators.solid;
import std;
import keyframe.field;

namespace kfs::operators {
    Solid::Solid(const cudaStream_t stream) : stream{stream} {
        if (stream == nullptr) throw std::runtime_error{"Solid stream must not be null"};
    }

    void Solid::apply_scalar(field::ScalarField3D& scalar, const field::ScalarField3D& solid_scalar, const std::uint8_t* occupancy) const {
        if (scalar.resolution != solid_scalar.resolution) throw std::runtime_error{"Solid scalar field resolution mismatch"};
        if (scalar.count() == 0u || scalar.data == nullptr) throw std::runtime_error{"scalar field is empty"};
        if (solid_scalar.count() == 0u || solid_scalar.data == nullptr) throw std::runtime_error{"solid_scalar field is empty"};
        if (occupancy == nullptr) throw std::runtime_error{"Solid occupancy must not be null"};
        cuda::operators::solid::apply_scalar(this->stream, scalar.data, solid_scalar.data, occupancy, scalar.resolution[0], scalar.resolution[1], scalar.resolution[2]);
    }
} // namespace kfs::operators
