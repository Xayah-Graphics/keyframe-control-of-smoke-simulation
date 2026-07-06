module;
#include "keyframe.solver.h"

#include "keyframe.field.h"
#include <cuda_runtime.h>

module keyframe.solver;
import std;
import keyframe.field;
import keyframe.boundary;
import keyframe.operators.advection;
import keyframe.operators.projection;

namespace kfs::solver {
    namespace {
        void validate_config(const Config& config) {
            if (config.resolution[0] == 0 || config.resolution[1] == 0 || config.resolution[2] == 0) throw std::runtime_error("Keyframe smoke resolution must be positive");
            if (config.resolution[0] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) || config.resolution[1] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) || config.resolution[2] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("Keyframe smoke resolution exceeds CUDA solver int range");
            if (config.cell_size <= 0.0f) throw std::runtime_error("Keyframe smoke cell_size must be positive");
            if (config.pressure_iterations <= 0) throw std::runtime_error{"Keyframe smoke pressure_iterations must be positive"};

            const std::uint64_t cell_count = static_cast<std::uint64_t>(config.resolution[0]) * static_cast<std::uint64_t>(config.resolution[1]) * static_cast<std::uint64_t>(config.resolution[2]);
            if (cell_count == 0 || cell_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("Keyframe smoke cell count exceeds pressure solver int range");
        }

        void validate_source(const PlumeSource& source) {
            if (source.radius[0] <= 0.0f || source.radius[1] <= 0.0f || source.radius[2] <= 0.0f) throw std::runtime_error{"Keyframe smoke plume source radius must be positive"};
            if (source.density < 0.0f) throw std::runtime_error{"Keyframe smoke plume source density must be non-negative"};
            if (source.temperature < 0.0f) throw std::runtime_error{"Keyframe smoke plume source temperature must be non-negative"};
            if (source.falloff <= 0.0f) throw std::runtime_error{"Keyframe smoke plume source falloff must be positive"};
        }

        void initialize_host(Solver::HostData& host, const Config& config) {
            host.nx                          = static_cast<std::int32_t>(config.resolution[0]);
            host.ny                          = static_cast<std::int32_t>(config.resolution[1]);
            host.nz                          = static_cast<std::int32_t>(config.resolution[2]);
            host.cell_size                   = config.cell_size;
            host.ambient_temperature         = config.ambient_temperature;
            host.buoyancy_density_factor     = config.buoyancy_density_factor;
            host.buoyancy_temperature_factor = config.buoyancy_temperature_factor;
            host.vorticity_confinement       = config.vorticity_confinement;
            host.boundary = boundary::pack(config.boundary);
            const auto cell_count = static_cast<std::uint64_t>(host.nx) * static_cast<std::uint64_t>(host.ny) * static_cast<std::uint64_t>(host.nz);
            host.density_source.resize(cell_count, 0.0f);
            host.temperature_source.resize(cell_count, 0.0f);
        }

        void initialize_field_buffers(const Solver::HostData& host, Solver::DeviceData& device) {
            field::fill(host.stream, device.density_data, 0.0f);
            field::fill(host.stream, device.density_temp, 0.0f);
            field::fill(host.stream, device.density_source, 0.0f);
            field::fill(host.stream, device.temperature_data, host.ambient_temperature);
            field::fill(host.stream, device.temperature_temp, host.ambient_temperature);
            field::fill(host.stream, device.temperature_source, 0.0f);
            field::fill(host.stream, device.force, 0.0f);
            field::fill(host.stream, device.solid_velocity, 0.0f);
            field::fill(host.stream, device.velocity, 0.0f);
            field::fill(host.stream, device.temp_velocity, 0.0f);
            field::fill(host.stream, device.centered_velocity, 0.0f);
            field::fill(host.stream, device.vorticity, 0.0f);
            field::fill(host.stream, device.vorticity_magnitude, 0.0f);
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
                device.density_source.resize(resolution);
                device.temperature_data.resize(resolution);
                device.temperature_temp.resize(resolution);
                device.temperature_source.resize(resolution);
                device.force.resize(resolution);
                device.solid_velocity.resize(resolution);
                device.velocity.resize(resolution);
                device.temp_velocity.resize(resolution);
                device.centered_velocity.resize(resolution);
                device.vorticity.resize(resolution);
                device.vorticity_magnitude.resize(resolution);
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
            this->projection.emplace(this->host.stream, std::array<std::int32_t, 3>{this->host.nx, this->host.ny, this->host.nz}, this->host.cell_size, config.pressure_iterations, this->host.boundary.flow);
            this->set_plume_source(this->host.plume_source);
        } catch (...) {
            this->projection.reset();
            this->advection.reset();
            destroy_device(*this);
            throw;
        }
    }

    Solver::~Solver() noexcept {
        this->projection.reset();
        this->advection.reset();
        destroy_device(*this);
    }

    void Solver::set_plume_source(const PlumeSource& source) {
        validate_source(source);
        this->host.plume_source = source;
        std::ranges::fill(this->host.density_source, 0.0f);
        std::ranges::fill(this->host.temperature_source, 0.0f);

        const auto nx = static_cast<std::uint32_t>(this->host.nx);
        const auto ny = static_cast<std::uint32_t>(this->host.ny);
        const auto nz = static_cast<std::uint32_t>(this->host.nz);
        const std::array extent{
            static_cast<float>(nx) * this->host.cell_size,
            static_cast<float>(ny) * this->host.cell_size,
            static_cast<float>(nz) * this->host.cell_size,
        };
        const std::array center{
            source.center[0] * extent[0],
            source.center[1] * extent[1],
            source.center[2] * extent[2],
        };
        const std::array radius{
            source.radius[0] * extent[0],
            source.radius[1] * extent[1],
            source.radius[2] * extent[2],
        };

        for (std::uint32_t z = 0; z < nz; ++z) {
            for (std::uint32_t y = 0; y < ny; ++y) {
                for (std::uint32_t x = 0; x < nx; ++x) {
                    const std::size_t index = static_cast<std::size_t>(x) + static_cast<std::size_t>(nx) * (static_cast<std::size_t>(y) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(z));
                    const float px          = (static_cast<float>(x) + 0.5f) * this->host.cell_size;
                    const float py          = (static_cast<float>(y) + 0.5f) * this->host.cell_size;
                    const float pz          = (static_cast<float>(z) + 0.5f) * this->host.cell_size;
                    const float dx          = (px - center[0]) / radius[0];
                    const float dy          = (py - center[1]) / radius[1];
                    const float dz          = (pz - center[2]) / radius[2];
                    const float r2          = dx * dx + dy * dy + dz * dz;
                    if (r2 > 1.0f) continue;
                    const float plume                    = std::exp(-source.falloff * r2);
                    this->host.density_source[index]     = source.density * plume;
                    this->host.temperature_source[index] = source.temperature * plume;
                }
            }
        }

        field::upload(this->host.stream, this->device.density_source, std::span<const float>{this->host.density_source.data(), this->host.density_source.size()});
        field::upload(this->host.stream, this->device.temperature_source, std::span<const float>{this->host.temperature_source.data(), this->host.temperature_source.size()});
        if (const cudaError_t status = cudaStreamSynchronize(this->host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize plume source: "} + cudaGetErrorString(status)};
    }

    std::expected<StepStats, std::string> Solver::step(const StepRequest& request) {
        try {
            if (!std::isfinite(request.delta_seconds) || request.delta_seconds < 0.0f) throw std::runtime_error{"Keyframe smoke delta_seconds must be finite and non-negative"};
            if (request.iterations < 1) throw std::runtime_error{"Keyframe smoke step iterations must be positive"};
            auto& host   = this->host;
            auto& device = this->device;
            const auto& flow_boundary = host.boundary.flow;
            const std::uint32_t* flow_types = flow_boundary.types.data();
            const float* flow_velocity      = flow_boundary.velocity.data();
            const auto step_start     = std::chrono::steady_clock::now();
            const float delta_seconds = request.delta_seconds;
            if (delta_seconds > 0.0f) {
                for (std::int32_t iteration = 0; iteration < request.iterations; ++iteration) {
                    cuda::apply_solid_scalar(host.stream, device.temperature_data.data, device.occupancy, device.solid_temperature.data, host.nx, host.ny, host.nz, host.ambient_temperature);
                    field::center_staggered(host.stream, device.centered_velocity, device.velocity);
                    cuda::compute_vorticity(host.stream, device.vorticity.data[0], device.vorticity.data[1], device.vorticity.data[2], device.vorticity_magnitude.data, device.centered_velocity.data[0], device.centered_velocity.data[1], device.centered_velocity.data[2], device.occupancy, host.nx, host.ny, host.nz, host.cell_size, flow_types, flow_velocity);
                    field::fill(host.stream, device.force, 0.0f);
                    cuda::add_buoyancy(host.stream, device.force.data[1], device.density_data.data, device.temperature_data.data, device.occupancy, host.nx, host.ny, host.nz, host.ambient_temperature, host.buoyancy_density_factor, host.buoyancy_temperature_factor, flow_types);
                    cuda::add_vorticity_confinement(host.stream, device.force.data[0], device.force.data[1], device.force.data[2], device.vorticity.data[0], device.vorticity.data[1], device.vorticity.data[2], device.vorticity_magnitude.data, device.occupancy, host.nx, host.ny, host.nz, host.cell_size, host.vorticity_confinement, flow_types);
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
                    field::add_scaled(host.stream, device.temperature_temp, device.temperature_data, device.temperature_source, delta_seconds);
                    (*this->advection)(device.temperature_data, device.temperature_temp, device.velocity, device.occupancy, delta_seconds, host.boundary.temperature, flow_boundary);
                    cuda::apply_solid_scalar(host.stream, device.temperature_data.data, device.occupancy, device.solid_temperature.data, host.nx, host.ny, host.nz, host.ambient_temperature);
                    field::add_scaled(host.stream, device.density_temp, device.density_data, device.density_source, delta_seconds);
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
