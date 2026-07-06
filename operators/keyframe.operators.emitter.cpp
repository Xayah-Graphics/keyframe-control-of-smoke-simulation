module;
#include "keyframe.operators.emitter.h"

module keyframe.operators.emitter;
import std;
import keyframe.field;
import keyframe.geometry;

namespace kfs::operators {
    Emitter::Emitter(const cudaStream_t stream, const std::array<std::int32_t, 3> resolution, const float cell_size, const Source& source) : stream{stream}, resolution{resolution}, cell_size{cell_size}, source{source} {
        if (stream == nullptr) throw std::runtime_error{"Emitter stream must not be null"};
        if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"Emitter resolution must be positive"};
        const std::uint64_t cell_count = static_cast<std::uint64_t>(resolution[0]) * static_cast<std::uint64_t>(resolution[1]) * static_cast<std::uint64_t>(resolution[2]);
        if (cell_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"Emitter cell count exceeds CUDA int range"};
        if (!std::isfinite(cell_size) || cell_size <= 0.0f) throw std::runtime_error{"Emitter cell_size must be positive"};
        geometry::validate(source.region);
        if (!std::isfinite(source.density_rate) || source.density_rate < 0.0f) throw std::runtime_error{"Emitter density_rate must be non-negative"};
        if (!std::isfinite(source.temperature_rate) || source.temperature_rate < 0.0f) throw std::runtime_error{"Emitter temperature_rate must be non-negative"};
        if (!std::isfinite(source.falloff) || source.falloff <= 0.0f) throw std::runtime_error{"Emitter falloff must be positive"};
    }

    void Emitter::operator()(field::ScalarField3D& density_destination, const field::ScalarField3D& density_current, field::ScalarField3D& temperature_destination, const field::ScalarField3D& temperature_current, const float delta_seconds) const {
        if (density_destination.resolution != this->resolution || density_current.resolution != this->resolution || temperature_destination.resolution != this->resolution || temperature_current.resolution != this->resolution) throw std::runtime_error{"Emitter field resolution mismatch"};
        if (density_destination.count() == 0u || density_destination.data == nullptr) throw std::runtime_error{"density destination field is empty"};
        if (density_current.count() == 0u || density_current.data == nullptr) throw std::runtime_error{"density current field is empty"};
        if (temperature_destination.count() == 0u || temperature_destination.data == nullptr) throw std::runtime_error{"temperature destination field is empty"};
        if (temperature_current.count() == 0u || temperature_current.data == nullptr) throw std::runtime_error{"temperature current field is empty"};
        if (!std::isfinite(delta_seconds) || delta_seconds <= 0.0f) throw std::runtime_error{"Emitter delta_seconds must be positive"};
        cuda::operators::emitter::emit_density_temperature(this->stream, density_destination.data, density_current.data, temperature_destination.data, temperature_current.data, this->resolution[0], this->resolution[1], this->resolution[2], this->cell_size, delta_seconds, this->source.region.center, this->source.region.radius, this->source.density_rate, this->source.temperature_rate, this->source.falloff);
    }
} // namespace kfs::operators
