module;
#include "keyframe.field.h"
#include <cuda_runtime.h>

module keyframe.solver;
import std;
import keyframe.field;
import keyframe.boundary;
import keyframe.operators.advection;
import keyframe.operators.emitter;
import keyframe.operators.buoyancy;
import keyframe.operators.projection;
import keyframe.operators.solid;
import keyframe.operators.vorticity;

namespace kfs::solver {
    namespace {
        void validate_config(const Config& config) {
            if (config.resolution[0] == 0 || config.resolution[1] == 0 || config.resolution[2] == 0) throw std::runtime_error("Keyframe smoke resolution must be positive");
            if (config.resolution[0] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) || config.resolution[1] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) || config.resolution[2] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("Keyframe smoke resolution exceeds CUDA solver int range");
            if (config.cell_size <= 0.0f) throw std::runtime_error("Keyframe smoke cell_size must be positive");
            if (config.pressure_iterations <= 0) throw std::runtime_error{"Keyframe smoke pressure_iterations must be positive"};
            if (!std::isfinite(config.ambient_temperature)) throw std::runtime_error{"Keyframe smoke ambient_temperature must be finite"};
            if (!std::isfinite(config.buoyancy_density_factor)) throw std::runtime_error{"Keyframe smoke buoyancy_density_factor must be finite"};
            if (!std::isfinite(config.buoyancy_temperature_factor)) throw std::runtime_error{"Keyframe smoke buoyancy_temperature_factor must be finite"};
            if (!std::isfinite(config.density_emission_rate) || config.density_emission_rate < 0.0f) throw std::runtime_error{"Keyframe smoke density_emission_rate must be finite and non-negative"};
            if (!std::isfinite(config.temperature_emission_rate) || config.temperature_emission_rate < 0.0f) throw std::runtime_error{"Keyframe smoke temperature_emission_rate must be finite and non-negative"};

            const std::uint64_t cell_count = static_cast<std::uint64_t>(config.resolution[0]) * static_cast<std::uint64_t>(config.resolution[1]) * static_cast<std::uint64_t>(config.resolution[2]);
            if (cell_count == 0 || cell_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("Keyframe smoke cell count exceeds pressure solver int range");
        }

        void initialize_host(Solver::HostData& host, const Config& config) {
            host.nx                        = static_cast<std::int32_t>(config.resolution[0]);
            host.ny                        = static_cast<std::int32_t>(config.resolution[1]);
            host.nz                        = static_cast<std::int32_t>(config.resolution[2]);
            host.cell_size                 = config.cell_size;
            host.ambient_temperature       = config.ambient_temperature;
            host.density_emission_rate     = config.density_emission_rate;
            host.temperature_emission_rate = config.temperature_emission_rate;
            host.boundary                  = boundary::pack(config.boundary);
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
            if (const cudaError_t status = cudaMemsetAsync(device.occupancy, 0, device.density_data.count() * sizeof(std::uint8_t), host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync occupancy: "} + cudaGetErrorString(status)};
        }

        void destroy_device(Solver& smoke) noexcept {
            try {
                if (smoke.host.stream != nullptr) cudaStreamSynchronize(smoke.host.stream);
                cuda::free_device_buffers(smoke.device.occupancy);
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
                const auto cell_count = device.density_data.count();
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device.occupancy), static_cast<std::size_t>(cell_count) * sizeof(std::uint8_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc occupancy: "} + cudaGetErrorString(status)};

                initialize_field_buffers(host, device);
            } catch (...) {
                destroy_device(smoke);
                throw;
            }
        }

    } // namespace

    Solver::Solver(const Config& config) {
        try {
            validate_config(config);
            this->host   = {};
            this->device = {};
            initialize_host(this->host, config);
            create_device(*this);
            this->advection.emplace(this->host.stream, this->host.cell_size, config.advection_scheme);
            this->emitter.emplace(this->host.stream, std::array<std::int32_t, 3>{this->host.nx, this->host.ny, this->host.nz}, this->host.cell_size, config.emitter);
            this->buoyancy.emplace(this->host.stream, this->host.ambient_temperature, config.buoyancy_density_factor, config.buoyancy_temperature_factor, this->host.boundary.flow);
            this->projection.emplace(this->host.stream, std::array<std::int32_t, 3>{this->host.nx, this->host.ny, this->host.nz}, this->host.cell_size, config.pressure_iterations, this->host.boundary.flow);
            this->solid.emplace(this->host.stream);
            this->vorticity.emplace(this->host.stream, std::array<std::int32_t, 3>{this->host.nx, this->host.ny, this->host.nz}, this->host.cell_size, config.vorticity_confinement, this->host.boundary.flow);
        } catch (...) {
            this->vorticity.reset();
            this->solid.reset();
            this->projection.reset();
            this->buoyancy.reset();
            this->emitter.reset();
            this->advection.reset();
            destroy_device(*this);
            throw;
        }
    }

    Solver::~Solver() noexcept {
        this->vorticity.reset();
        this->solid.reset();
        this->projection.reset();
        this->buoyancy.reset();
        this->emitter.reset();
        this->advection.reset();
        destroy_device(*this);
    }

    std::expected<StepStats, std::string> Solver::step(const StepRequest& request) {
        try {
            if (!std::isfinite(request.delta_seconds) || request.delta_seconds < 0.0f) throw std::runtime_error{"Keyframe smoke delta_seconds must be finite and non-negative"};
            if (request.iterations < 1) throw std::runtime_error{"Keyframe smoke step iterations must be positive"};
            auto& host                = this->host;
            auto& device              = this->device;
            const auto& flow_boundary = host.boundary.flow;
            const auto step_start     = std::chrono::steady_clock::now();
            const float delta_seconds = request.delta_seconds;
            if (delta_seconds > 0.0f) {
                for (std::int32_t iteration = 0; iteration < request.iterations; ++iteration) {
                    this->solid->apply_scalar(device.temperature_data, device.solid_temperature, device.occupancy);
                    field::center_staggered(host.stream, device.centered_velocity, device.velocity);
                    field::fill(host.stream, device.force, 0.0f);
                    (*this->buoyancy)(device.force, device.density_data, device.temperature_data, device.occupancy);
                    (*this->vorticity)(device.force, device.centered_velocity, device.occupancy);
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        field::add_centered_to_staggered(host.stream, device.velocity, axis, device.force, delta_seconds);
                        boundary::enforce_staggered_boundary(host.stream, axis, device.velocity, device.occupancy, device.solid_velocity, flow_boundary);
                        if (flow_boundary.periodic[axis]) boundary::sync_periodic_staggered_component(host.stream, axis, device.velocity);
                    }
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        (*this->advection)(device.temp_velocity, axis, device.velocity, device.velocity, device.occupancy, delta_seconds, flow_boundary);
                        boundary::enforce_staggered_boundary(host.stream, axis, device.temp_velocity, device.occupancy, device.solid_velocity, flow_boundary);
                        if (flow_boundary.periodic[axis]) boundary::sync_periodic_staggered_component(host.stream, axis, device.temp_velocity);
                    }
                    (*this->projection)(device.velocity, device.temp_velocity, device.solid_velocity, device.occupancy, delta_seconds);
                    (*this->emitter)(device.density_temp, device.density_data, host.density_emission_rate, delta_seconds);
                    (*this->emitter)(device.temperature_temp, device.temperature_data, host.temperature_emission_rate, delta_seconds);
                    (*this->advection)(device.temperature_data, device.temperature_temp, device.velocity, device.occupancy, delta_seconds, host.boundary.temperature, flow_boundary);
                    this->solid->apply_scalar(device.temperature_data, device.solid_temperature, device.occupancy);
                    (*this->advection)(device.density_data, device.density_temp, device.velocity, device.occupancy, delta_seconds, host.boundary.density, flow_boundary);
                    boundary::boundary_fill_centered_scalar(host.stream, device.density_temp, device.density_data, device.occupancy, host.boundary.density);
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
