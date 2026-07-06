module;
#include "keyframe.operators.vorticity.h"

module keyframe.operators.vorticity;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    Vorticity::Vorticity(const cudaStream_t stream, const std::array<std::int32_t, 3> resolution, const float cell_size, const float vorticity_confinement, const boundary::PackedFlowBoundary& boundary) : stream{stream}, resolution{resolution}, cell_size{cell_size}, confinement{vorticity_confinement}, flow_boundary{boundary} {
        if (stream == nullptr) throw std::runtime_error{"Vorticity stream must not be null"};
        if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"Vorticity resolution must be positive"};
        if (!std::isfinite(cell_size) || cell_size <= 0.0f) throw std::runtime_error{"Vorticity cell_size must be positive"};
        if (!std::isfinite(vorticity_confinement)) throw std::runtime_error{"Vorticity confinement must be finite"};
        this->vorticity.resize(resolution);
        this->vorticity_magnitude.resize(resolution);
    }

    void Vorticity::operator()(field::CenteredVectorField3D& force, const field::CenteredVectorField3D& centered_velocity, const std::uint8_t* occupancy) {
        if (force.resolution != this->resolution || centered_velocity.resolution != this->resolution) throw std::runtime_error{"Vorticity field resolution mismatch"};
        if (force.count() == 0u) throw std::runtime_error{"force field is empty"};
        if (centered_velocity.count() == 0u) throw std::runtime_error{"centered_velocity field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (force.data[axis] == nullptr) throw std::runtime_error{"force field component is empty"};
            if (centered_velocity.data[axis] == nullptr) throw std::runtime_error{"centered_velocity field component is empty"};
        }
        if (occupancy == nullptr) throw std::runtime_error{"Vorticity occupancy must not be null"};

        const int nx                    = this->resolution[0];
        const int ny                    = this->resolution[1];
        const int nz                    = this->resolution[2];
        const std::uint32_t* flow_types = this->flow_boundary.types.data();
        const float* flow_velocity      = this->flow_boundary.velocity.data();
        cuda::operators::vorticity::compute_vorticity(this->stream, this->vorticity.data[0], this->vorticity.data[1], this->vorticity.data[2], this->vorticity_magnitude.data, centered_velocity.data[0], centered_velocity.data[1], centered_velocity.data[2], occupancy, nx, ny, nz, this->cell_size, flow_types, flow_velocity);
        cuda::operators::vorticity::add_confinement(this->stream, force.data[0], force.data[1], force.data[2], this->vorticity.data[0], this->vorticity.data[1], this->vorticity.data[2], this->vorticity_magnitude.data, occupancy, nx, ny, nz, this->cell_size, this->confinement, flow_types);
    }
} // namespace kfs::operators
