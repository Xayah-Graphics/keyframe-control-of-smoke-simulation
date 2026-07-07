module;
#include "keyframe.field.h"
#include <cuda_runtime.h>

module keyframe.solver;
import std;
import keyframe.field;
import keyframe.boundary;
import keyframe.operators.advection;
import keyframe.operators.emitter;
import keyframe.operators.projection;
import keyframe.operators.vorticity;

namespace kfs::solver {
    namespace {
        void initialize_host(Solver::HostData& host, const Config& config) {
            host.nx                          = static_cast<std::int32_t>(config.resolution[0]);
            host.ny                          = static_cast<std::int32_t>(config.resolution[1]);
            host.nz                          = static_cast<std::int32_t>(config.resolution[2]);
            host.cell_size                   = config.cell_size;
            host.ambient_temperature         = config.ambient_temperature;
            host.buoyancy_density_factor     = config.buoyancy_density_factor;
            host.buoyancy_temperature_factor = config.buoyancy_temperature_factor;
            host.density_emission_rate       = config.density_emission_rate;
            host.temperature_emission_rate   = config.temperature_emission_rate;
            host.boundary.velocity           = boundary::pack(config.boundary.velocity);
            host.boundary.pressure           = boundary::pack(config.boundary.pressure);
            host.boundary.density            = boundary::pack(config.boundary.density);
            host.boundary.temperature        = boundary::pack(config.boundary.temperature);
        }

        void initialize_field_buffers(const Solver::HostData& host, Solver::DeviceData& device) {
            field::fill(host.stream, device.density_data, 0.0f);
            field::fill(host.stream, device.density_temp, 0.0f);
            field::fill(host.stream, device.temperature_data, host.ambient_temperature);
            field::fill(host.stream, device.temperature_temp, host.ambient_temperature);
            field::fill(host.stream, device.force, 0.0f);
            field::fill(host.stream, device.solid_velocity, 0.0f);
            field::fill(host.stream, device.velocity, 0.0f);
            field::fill(host.stream, device.temp_velocity, 0.0f);
            field::fill(host.stream, device.centered_velocity, 0.0f);
            field::fill(host.stream, device.solid_temperature, host.ambient_temperature);
            field::fill(host.stream, device.occupancy, 0u);
        }

        void destroy_device(Solver& smoke) noexcept {
            try {
                if (smoke.host.stream != nullptr) cudaStreamSynchronize(smoke.host.stream);
                if (smoke.host.stream != nullptr) cudaStreamDestroy(smoke.host.stream);
            } catch (...) {
            }
            smoke.host.stream = nullptr;
            smoke.device      = {};
        }

        void create_device(Solver& smoke) {
            if (smoke.host.stream != nullptr) throw std::runtime_error{"Keyframe smoke device is already initialized"};
            auto& host   = smoke.host;
            auto& device = smoke.device;

            try {
                if (const cudaError_t status = cudaStreamCreateWithFlags(&host.stream, cudaStreamNonBlocking); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamCreateWithFlags: "} + cudaGetErrorString(status)};

                device = Solver::DeviceData{};
                const std::array resolution{host.nx, host.ny, host.nz};
                device.density_data.resize(resolution);
                device.density_temp.resize(resolution);
                device.temperature_data.resize(resolution);
                device.temperature_temp.resize(resolution);
                device.force.resize(resolution);
                device.solid_velocity.resize(resolution);
                device.velocity.resize(resolution);
                device.temp_velocity.resize(resolution);
                device.centered_velocity.resize(resolution);
                device.solid_temperature.resize(resolution);
                device.occupancy.resize(resolution);

                initialize_field_buffers(host, device);
            } catch (...) {
                destroy_device(smoke);
                throw;
            }
        }

    } // namespace

    Solver::Solver(const Config& config) {
        try {
            this->host   = {};
            this->device = {};
            initialize_host(this->host, config);
            create_device(*this);
            this->advection.emplace(this->host.stream, this->host.cell_size, config.advection_scheme);
            this->emitter.emplace(this->host.stream, std::array<std::int32_t, 3>{this->host.nx, this->host.ny, this->host.nz}, this->host.cell_size, config.emitter);
            this->projection.emplace(this->host.stream, std::array<std::int32_t, 3>{this->host.nx, this->host.ny, this->host.nz}, this->host.cell_size, config.pressure_iterations, this->host.boundary.velocity, this->host.boundary.pressure);
            this->vorticity.emplace(this->host.stream, std::array<std::int32_t, 3>{this->host.nx, this->host.ny, this->host.nz}, this->host.cell_size, config.vorticity_confinement, this->host.boundary.velocity);
        } catch (...) {
            this->vorticity.reset();
            this->projection.reset();
            this->emitter.reset();
            this->advection.reset();
            destroy_device(*this);
            throw;
        }
    }

    Solver::~Solver() noexcept {
        this->vorticity.reset();
        this->projection.reset();
        this->emitter.reset();
        this->advection.reset();
        destroy_device(*this);
    }

    std::expected<StepStats, std::string> Solver::step(const StepRequest& request) {
        try {
            auto& host                = this->host;
            auto& device              = this->device;
            const auto& velocity_boundary = host.boundary.velocity;
            const auto step_start     = std::chrono::steady_clock::now();
            const float delta_seconds = request.delta_seconds;
            if (delta_seconds > 0.0f) {
                for (std::int32_t iteration = 0; iteration < request.iterations; ++iteration) {
                    field::copy(host.stream, device.temperature_data, device.solid_temperature, device.occupancy, field::IndexSelection::marked);
                    field::sample(host.stream, device.centered_velocity, device.velocity);
                    field::fill(host.stream, device.force, 0.0f);
                    field::add(host.stream, device.force, 1u, device.density_data, -host.buoyancy_density_factor, 0.0f, device.occupancy, field::IndexSelection::unmarked);
                    field::add(host.stream, device.force, 1u, device.temperature_data, host.buoyancy_temperature_factor, -host.ambient_temperature, device.occupancy, field::IndexSelection::unmarked);
                    (*this->vorticity)(device.force, device.centered_velocity, device.occupancy);
                    field::add(host.stream, device.velocity, device.force, delta_seconds);
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        boundary::enforce(host.stream, axis, device.velocity, device.occupancy, device.solid_velocity, velocity_boundary);
                        if (velocity_boundary.periodic[axis]) boundary::synchronize(host.stream, axis, device.velocity);
                    }
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        (*this->advection)(device.temp_velocity, axis, device.velocity, device.velocity, device.occupancy, delta_seconds, velocity_boundary);
                        boundary::enforce(host.stream, axis, device.temp_velocity, device.occupancy, device.solid_velocity, velocity_boundary);
                        if (velocity_boundary.periodic[axis]) boundary::synchronize(host.stream, axis, device.temp_velocity);
                    }
                    (*this->projection)(device.velocity, device.temp_velocity, device.solid_velocity, device.occupancy, delta_seconds);
                    (*this->emitter)(device.density_temp, device.density_data, host.density_emission_rate, delta_seconds);
                    (*this->emitter)(device.temperature_temp, device.temperature_data, host.temperature_emission_rate, delta_seconds);
                    (*this->advection)(device.temperature_data, device.temperature_temp, device.velocity, device.occupancy, delta_seconds, host.boundary.temperature, velocity_boundary);
                    field::copy(host.stream, device.temperature_data, device.solid_temperature, device.occupancy, field::IndexSelection::marked);
                    (*this->advection)(device.density_data, device.density_temp, device.velocity, device.occupancy, delta_seconds, host.boundary.density, velocity_boundary);
                    boundary::extrapolate(host.stream, device.density_temp, device.density_data, device.occupancy, host.boundary.density);
                    field::copy(host.stream, device.density_data, device.density_temp);
                    ++this->host.current_step;
                }
            }
            const auto step_stop = std::chrono::steady_clock::now();
            return StepStats{
                .step       = this->host.current_step,
                .elapsed_ms = std::chrono::duration<float, std::milli>{step_stop - step_start}.count(),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace kfs::solver
