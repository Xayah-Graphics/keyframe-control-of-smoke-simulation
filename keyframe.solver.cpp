module;
#include "core/field.h"
#include <cuda_runtime.h>

module keyframe.solver;
import std;
import xayah.core.field;
import xayah.core.boundary;
import xayah.core.collider;
import xayah.operators.advection;
import xayah.operators.emitter;
import xayah.operators.projection;
import xayah.operators.vorticity;

namespace kfs::solver {
    Solver::Solver(const std::array<std::uint32_t, 3> resolution, const float cell_size, const SmokeBoundary boundaries, cudaStream_t execution_stream) : resolution{static_cast<std::int32_t>(resolution[0]), static_cast<std::int32_t>(resolution[1]), static_cast<std::int32_t>(resolution[2])}, cell_size{cell_size}, pressure_boundary{boundaries.pressure}, stream{execution_stream}, velocity_boundary{boundaries.velocity}, density_boundary{boundaries.density}, temperature_boundary{boundaries.temperature}, device{}, advection{this->stream, this->cell_size, xayah::operators::Advection::Scheme::monotonic_cubic}, emitter{this->stream, this->resolution, this->cell_size, xayah::operators::Emitter::Source{}}, projection{this->stream, this->resolution, this->cell_size, xayah::core::boundary::pack(this->pressure_boundary), 64}, vorticity{this->stream, this->resolution, this->cell_size, 0.22f} {
        this->device.density_data.resize(this->resolution);
        this->device.density_temp.resize(this->resolution);
        this->device.temperature_data.resize(this->resolution);
        this->device.temperature_temp.resize(this->resolution);
        this->device.force.resize(this->resolution);
        this->device.constraint_velocity.resize(this->resolution);
        this->device.velocity.resize(this->resolution);
        this->device.temp_velocity.resize(this->resolution);
        this->device.centered_velocity.resize(this->resolution);
        this->device.constraint_scalar.resize(this->resolution);
        this->device.cell_indices.resize(this->resolution);

        xayah::core::field::fill(this->stream, this->device.density_data, 0.0f);
        xayah::core::field::fill(this->stream, this->device.density_temp, 0.0f);
        xayah::core::field::fill(this->stream, this->device.temperature_data, this->ambient_temperature);
        xayah::core::field::fill(this->stream, this->device.temperature_temp, this->ambient_temperature);
        xayah::core::field::fill(this->stream, this->device.force, 0.0f);
        xayah::core::field::fill(this->stream, this->device.constraint_velocity, 0.0f);
        xayah::core::field::fill(this->stream, this->device.velocity, 0.0f);
        xayah::core::field::fill(this->stream, this->device.temp_velocity, 0.0f);
        xayah::core::field::fill(this->stream, this->device.centered_velocity, 0.0f);
        xayah::core::field::fill(this->stream, this->device.constraint_scalar, 0.0f);
        xayah::core::field::fill(this->stream, this->device.cell_indices, 0u);
    }

    std::expected<StepStats, std::string> Solver::step(const StepRequest& request) {
        try {
            auto& device                    = this->device;
            const auto velocity_boundary    = xayah::core::boundary::pack(this->velocity_boundary);
            const auto density_boundary     = xayah::core::boundary::pack(this->density_boundary);
            const auto temperature_boundary = xayah::core::boundary::pack(this->temperature_boundary);
            cudaStream_t stream             = this->stream;
            const auto step_start           = std::chrono::steady_clock::now();
            const float delta_seconds       = request.delta_seconds;
            this->colliders.rasterize(stream, device.cell_indices, device.constraint_velocity, device.constraint_scalar, this->cell_size);
            if (delta_seconds > 0.0f) {
                for (std::int32_t iteration = 0; iteration < request.iterations; ++iteration) {
                    xayah::core::field::copy(stream, device.temperature_data, device.constraint_scalar, device.cell_indices, xayah::core::field::Selection::marked);
                    xayah::core::field::sample(stream, device.centered_velocity, device.velocity);
                    xayah::core::field::fill(stream, device.force, 0.0f);
                    xayah::core::field::add(stream, device.force, 1u, device.density_data, -this->buoyancy_density_factor, 0.0f, device.cell_indices, xayah::core::field::Selection::unmarked);
                    xayah::core::field::add(stream, device.force, 1u, device.temperature_data, this->buoyancy_temperature_factor, -this->ambient_temperature, device.cell_indices, xayah::core::field::Selection::unmarked);
                    this->vorticity(device.force, device.centered_velocity, device.cell_indices, velocity_boundary);
                    xayah::core::field::add(stream, device.velocity, device.force, delta_seconds);
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        xayah::core::boundary::enforce(stream, axis, device.velocity, device.cell_indices, device.constraint_velocity, velocity_boundary);
                        if (velocity_boundary.periodic[axis]) xayah::core::boundary::synchronize(stream, axis, device.velocity);
                    }
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        this->advection(device.temp_velocity, axis, device.velocity, device.velocity, device.cell_indices, delta_seconds, velocity_boundary);
                        xayah::core::boundary::enforce(stream, axis, device.temp_velocity, device.cell_indices, device.constraint_velocity, velocity_boundary);
                        if (velocity_boundary.periodic[axis]) xayah::core::boundary::synchronize(stream, axis, device.temp_velocity);
                    }
                    this->projection(device.velocity, device.temp_velocity, device.constraint_velocity, device.cell_indices, velocity_boundary, delta_seconds);
                    this->emitter(device.density_temp, device.density_data, this->density_emission_rate, delta_seconds);
                    this->emitter(device.temperature_temp, device.temperature_data, this->temperature_emission_rate, delta_seconds);
                    this->advection(device.temperature_data, device.temperature_temp, device.velocity, device.cell_indices, delta_seconds, temperature_boundary, velocity_boundary);
                    xayah::core::field::copy(stream, device.temperature_data, device.constraint_scalar, device.cell_indices, xayah::core::field::Selection::marked);
                    this->advection(device.density_data, device.density_temp, device.velocity, device.cell_indices, delta_seconds, density_boundary, velocity_boundary);
                    xayah::core::boundary::extrapolate(stream, device.density_temp, device.density_data, device.cell_indices, density_boundary);
                    xayah::core::field::copy(stream, device.density_data, device.density_temp);
                    ++this->current_step;
                }
            }
            const auto step_stop = std::chrono::steady_clock::now();
            return StepStats{
                .step       = this->current_step,
                .elapsed_ms = std::chrono::duration<float, std::milli>{step_stop - step_start}.count(),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace kfs::solver
