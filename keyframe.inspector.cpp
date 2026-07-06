module;
#include "keyframe.inspector.h"

module keyframe.inspector;
import std;
import keyframe.solver;

namespace kfs::inspector {
namespace {
    void require_initialized(const solver::Solver& smoke) {
        if (smoke.host.stream == nullptr) throw std::runtime_error{"keyframe solver is not initialized."};
        if (smoke.device.density_data.data == nullptr || smoke.device.temperature_data.data == nullptr) throw std::runtime_error{"keyframe scalar fields are not initialized."};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (smoke.device.velocity.data[axis] == nullptr) throw std::runtime_error{"keyframe velocity field is not initialized."};
    }
} // namespace

    Inspector::Inspector(const solver::Solver& smoke) : smoke{std::addressof(smoke)} {}

    SolverDeviceView Inspector::device_view() const {
        if (this->smoke == nullptr) throw std::runtime_error{"keyframe inspector solver is null."};
        const auto& host       = this->smoke->host;
        const auto& velocity   = this->smoke->device.velocity;
        const auto cell_count  = this->smoke->device.density_data.count();
        return SolverDeviceView{
            .resolution     = {static_cast<std::uint32_t>(host.nx), static_cast<std::uint32_t>(host.ny), static_cast<std::uint32_t>(host.nz)},
            .cell_count     = cell_count,
            .velocity_count = {velocity.count(0u), velocity.count(1u), velocity.count(2u)},
            .density        = this->smoke->device.density_data.data,
            .temperature    = this->smoke->device.temperature_data.data,
            .velocity       = {velocity.data[0u], velocity.data[1u], velocity.data[2u]},
            .initialized    = this->smoke->host.stream != nullptr,
        };
    }

    FrameSnapshot Inspector::read_frame(const int frame_index) const {
        if (this->smoke == nullptr) throw std::runtime_error{"keyframe inspector solver is null."};
        require_initialized(*this->smoke);
        FrameSnapshot frame{};
        frame.frame_index = frame_index;
        frame.resolution  = {static_cast<std::uint32_t>(this->smoke->host.nx), static_cast<std::uint32_t>(this->smoke->host.ny), static_cast<std::uint32_t>(this->smoke->host.nz)};
        frame.cell_size   = this->smoke->host.cell_size;
        const auto cell_count = this->smoke->device.density_data.count();
        const std::array velocities{
            this->smoke->device.velocity.count(0u),
            this->smoke->device.velocity.count(1u),
            this->smoke->device.velocity.count(2u),
        };
        frame.density.resize(cell_count);
        frame.temperature.resize(cell_count);
        frame.velocity_x.resize(velocities[0u]);
        frame.velocity_y.resize(velocities[1u]);
        frame.velocity_z.resize(velocities[2u]);
        cuda::read_float_buffer(this->smoke->device.density_data.data, frame.density.data(), cell_count, this->smoke->host.stream);
        cuda::read_float_buffer(this->smoke->device.temperature_data.data, frame.temperature.data(), cell_count, this->smoke->host.stream);
        cuda::read_float_buffer(this->smoke->device.velocity.data[0u], frame.velocity_x.data(), velocities[0u], this->smoke->host.stream);
        cuda::read_float_buffer(this->smoke->device.velocity.data[1u], frame.velocity_y.data(), velocities[1u], this->smoke->host.stream);
        cuda::read_float_buffer(this->smoke->device.velocity.data[2u], frame.velocity_z.data(), velocities[2u], this->smoke->host.stream);
        return frame;
    }

    FrameStats Inspector::frame_stats(const int frame_index) const {
        if (this->smoke == nullptr) throw std::runtime_error{"keyframe inspector solver is null."};
        require_initialized(*this->smoke);
        cuda::ScalarReadbackStats density{};
        cuda::ScalarReadbackStats temperature{};
        const auto cell_count = this->smoke->device.density_data.count();
        cuda::read_scalar_field_stats(this->smoke->device.density_data.data, cell_count, this->smoke->host.stream, density);
        cuda::read_scalar_field_stats(this->smoke->device.temperature_data.data, cell_count, this->smoke->host.stream, temperature);
        return FrameStats{
            .frame_index = frame_index,
            .resolution  = {static_cast<std::uint32_t>(this->smoke->host.nx), static_cast<std::uint32_t>(this->smoke->host.ny), static_cast<std::uint32_t>(this->smoke->host.nz)},
            .cell_size   = this->smoke->host.cell_size,
            .density     = ScalarFieldStats{.min = density.min, .max = density.max, .sum = density.sum, .mean = density.mean, .nonzero_count = density.nonzero_count},
            .temperature = ScalarFieldStats{.min = temperature.min, .max = temperature.max, .sum = temperature.sum, .mean = temperature.mean, .nonzero_count = temperature.nonzero_count},
        };
    }
} // namespace kfs::inspector
