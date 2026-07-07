module;
#include "keyframe.operators.scalar_force.h"

module keyframe.operators.scalar_force;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    ScalarForce::ScalarForce(const cudaStream_t stream, const boundary::PackedFlowBoundary& boundary) : stream{stream}, flow_boundary{boundary} {
        if (stream == nullptr) throw std::runtime_error{"ScalarForce stream must not be null"};
    }

    void ScalarForce::operator()(field::CenteredVectorField3D& destination, const std::uint32_t axis, const field::ScalarField3D& source, const float scale, const float bias, const std::uint8_t* cell_mask) const {
        if (destination.resolution != source.resolution) throw std::runtime_error{"ScalarForce field resolution mismatch"};
        if (axis >= 3u) throw std::runtime_error{"ScalarForce axis must be 0, 1, or 2"};
        if (destination.count() == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"ScalarForce destination field component is empty"};
        if (source.count() == 0u || source.data == nullptr) throw std::runtime_error{"ScalarForce source field is empty"};
        if (!std::isfinite(scale)) throw std::runtime_error{"ScalarForce scale must be finite"};
        if (!std::isfinite(bias)) throw std::runtime_error{"ScalarForce bias must be finite"};
        if (cell_mask == nullptr) throw std::runtime_error{"ScalarForce cell_mask must not be null"};

        cuda::operators::scalar_force::add_scalar_force(this->stream, destination.data[axis], source.data, cell_mask, destination.resolution[0], destination.resolution[1], destination.resolution[2], scale, bias, this->flow_boundary.types.data());
    }
} // namespace kfs::operators
