module;
#include "keyframe.operators.masked_scalar_assignment.h"

module keyframe.operators.masked_scalar_assignment;
import std;
import keyframe.field;

namespace kfs::operators {
    MaskedScalarAssignment::MaskedScalarAssignment(const cudaStream_t stream) : stream{stream} {
        if (stream == nullptr) throw std::runtime_error{"MaskedScalarAssignment stream must not be null"};
    }

    void MaskedScalarAssignment::operator()(field::ScalarField3D& destination, const field::ScalarField3D& source, const std::uint8_t* mask) const {
        if (destination.resolution != source.resolution) throw std::runtime_error{"MaskedScalarAssignment field resolution mismatch"};
        if (destination.count() == 0u || destination.data == nullptr) throw std::runtime_error{"MaskedScalarAssignment destination field is empty"};
        if (source.count() == 0u || source.data == nullptr) throw std::runtime_error{"MaskedScalarAssignment source field is empty"};
        if (mask == nullptr) throw std::runtime_error{"MaskedScalarAssignment mask must not be null"};
        cuda::operators::masked_scalar_assignment::assign_masked_scalar(this->stream, destination.data, source.data, mask, destination.resolution[0], destination.resolution[1], destination.resolution[2]);
    }
} // namespace kfs::operators
