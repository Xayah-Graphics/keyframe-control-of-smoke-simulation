module;
#include "emitter.h"

module xayah.operators.emitter;
import std;
import xayah.core.field;

namespace xayah::operators {
    Emitter::Emitter(cudaStream_t stream, const std::array<std::int32_t, 3> resolution, const float cell_size, const Source& source) : stream{stream}, resolution{resolution}, cell_size{cell_size}, source{source} {}

    void Emitter::operator()(core::field::ScalarField3D& destination, const core::field::ScalarField3D& current, const float rate, const float delta_seconds) const {
        emitter::cuda::emit_scalar(this->stream, destination.data, current.data, this->resolution[0], this->resolution[1], this->resolution[2], this->cell_size, delta_seconds, this->source.region.center, this->source.region.radius, rate, this->source.falloff);
    }
} // namespace xayah::operators
