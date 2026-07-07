module;
#include "keyframe.operators.emitter.h"

module keyframe.operators.emitter;
import std;
import keyframe.field;

namespace kfs::operators {
    Emitter::Emitter(const cudaStream_t stream, const std::array<std::int32_t, 3> resolution, const float cell_size, const Source& source) : stream{stream}, resolution{resolution}, cell_size{cell_size}, source{source} {}

    void Emitter::operator()(field::ScalarField3D& destination, const field::ScalarField3D& current, const float rate, const float delta_seconds) const {
        cuda::operators::emitter::emit_scalar(this->stream, destination.data, current.data, this->resolution[0], this->resolution[1], this->resolution[2], this->cell_size, delta_seconds, this->source.region.center, this->source.region.radius, rate, this->source.falloff);
    }
} // namespace kfs::operators
