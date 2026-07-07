module;
#include "keyframe.inspector.h"

module keyframe.inspector;
import std;
import keyframe.solver;

namespace kfs::inspector {
    Inspector::Inspector(const solver::Solver& smoke) : smoke{std::addressof(smoke)} {}

    SolverDeviceView Inspector::device_view() const {
        if (this->smoke == nullptr) throw std::runtime_error{"keyframe inspector solver is null."};
        const auto& host        = this->smoke->host;
        const auto& density     = this->smoke->device.density_data;
        const auto& temperature = this->smoke->device.temperature_data;
        const auto& velocity    = this->smoke->device.velocity;
        const auto cell_count   = density.count();
        return SolverDeviceView{
            .resolution     = {static_cast<std::uint32_t>(host.nx), static_cast<std::uint32_t>(host.ny), static_cast<std::uint32_t>(host.nz)},
            .cell_count     = cell_count,
            .velocity_count = {velocity.count(0u), velocity.count(1u), velocity.count(2u)},
            .density        = density.data,
            .temperature    = temperature.data,
            .velocity       = {velocity.data[0u], velocity.data[1u], velocity.data[2u]},
            .initialized    = this->smoke->host.stream != nullptr,
        };
    }

    FrameSnapshot Inspector::read_frame(const int frame_index) const {
        if (this->smoke == nullptr) throw std::runtime_error{"keyframe inspector solver is null."};
        if (this->smoke->host.stream == nullptr) throw std::runtime_error{"keyframe solver is not initialized."};
        const auto& density     = this->smoke->device.density_data;
        const auto& temperature = this->smoke->device.temperature_data;
        const auto& velocity    = this->smoke->device.velocity;
        if (density.data == nullptr || temperature.data == nullptr) throw std::runtime_error{"keyframe scalar fields are not initialized."};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (velocity.data[axis] == nullptr) throw std::runtime_error{"keyframe velocity field is not initialized."};
        FrameSnapshot frame{};
        frame.frame_index     = frame_index;
        frame.resolution      = {static_cast<std::uint32_t>(this->smoke->host.nx), static_cast<std::uint32_t>(this->smoke->host.ny), static_cast<std::uint32_t>(this->smoke->host.nz)};
        frame.cell_size       = this->smoke->host.cell_size;
        const auto cell_count = density.count();
        const std::array velocities{velocity.count(0u), velocity.count(1u), velocity.count(2u)};
        frame.density.resize(cell_count);
        frame.temperature.resize(cell_count);
        frame.velocity_x.resize(velocities[0u]);
        frame.velocity_y.resize(velocities[1u]);
        frame.velocity_z.resize(velocities[2u]);
        cuda::read_float_buffer(density.data, frame.density.data(), cell_count, this->smoke->host.stream);
        cuda::read_float_buffer(temperature.data, frame.temperature.data(), cell_count, this->smoke->host.stream);
        cuda::read_float_buffer(velocity.data[0u], frame.velocity_x.data(), velocities[0u], this->smoke->host.stream);
        cuda::read_float_buffer(velocity.data[1u], frame.velocity_y.data(), velocities[1u], this->smoke->host.stream);
        cuda::read_float_buffer(velocity.data[2u], frame.velocity_z.data(), velocities[2u], this->smoke->host.stream);
        return frame;
    }

    FrameStats Inspector::frame_stats(const int frame_index) const {
        if (this->smoke == nullptr) throw std::runtime_error{"keyframe inspector solver is null."};
        if (this->smoke->host.stream == nullptr) throw std::runtime_error{"keyframe solver is not initialized."};
        const auto& density     = this->smoke->device.density_data;
        const auto& temperature = this->smoke->device.temperature_data;
        const auto& velocity    = this->smoke->device.velocity;
        if (density.data == nullptr || temperature.data == nullptr) throw std::runtime_error{"keyframe scalar fields are not initialized."};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (velocity.data[axis] == nullptr) throw std::runtime_error{"keyframe velocity field is not initialized."};
        cuda::ScalarReadbackStats density_stats{};
        cuda::ScalarReadbackStats temperature_stats{};
        const auto cell_count = density.count();
        cuda::read_scalar_field_stats(density.data, cell_count, this->smoke->host.stream, density_stats);
        cuda::read_scalar_field_stats(temperature.data, cell_count, this->smoke->host.stream, temperature_stats);
        return FrameStats{
            .frame_index = frame_index,
            .resolution  = {static_cast<std::uint32_t>(this->smoke->host.nx), static_cast<std::uint32_t>(this->smoke->host.ny), static_cast<std::uint32_t>(this->smoke->host.nz)},
            .cell_size   = this->smoke->host.cell_size,
            .density     = ScalarFieldStats{.min = density_stats.min, .max = density_stats.max, .sum = density_stats.sum, .mean = density_stats.mean, .nonzero_count = density_stats.nonzero_count},
            .temperature = ScalarFieldStats{.min = temperature_stats.min, .max = temperature_stats.max, .sum = temperature_stats.sum, .mean = temperature_stats.mean, .nonzero_count = temperature_stats.nonzero_count},
        };
    }
} // namespace kfs::inspector
