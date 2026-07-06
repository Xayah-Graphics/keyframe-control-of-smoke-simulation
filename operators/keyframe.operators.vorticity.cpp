module;
#include "keyframe.operators.vorticity.h"

module keyframe.operators.vorticity;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    namespace {
        constexpr std::array<std::int32_t, 3> empty_resolution{0, 0, 0};

        void require_resolution(const std::array<std::int32_t, 3> resolution) {
            if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"Vorticity resolution must be positive"};
        }

        void require_centered_field(const field::CenteredVectorField3D& values, const char* name) {
            if (values.resolution == empty_resolution || values.count() == 0u) throw std::runtime_error{std::string{name} + " field is empty"};
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                if (values.data[axis] == nullptr) throw std::runtime_error{std::string{name} + " field component is empty"};
            }
        }
    } // namespace

    Vorticity::Vorticity(const cudaStream_t stream, const std::array<std::int32_t, 3> resolution, const float cell_size, const float vorticity_confinement, const boundary::PackedFlowBoundary& boundary) : stream{stream}, resolution{resolution}, cell_size{cell_size}, confinement{vorticity_confinement}, flow_boundary{boundary} {
        if (stream == nullptr) throw std::runtime_error{"Vorticity stream must not be null"};
        require_resolution(resolution);
        if (!std::isfinite(cell_size) || cell_size <= 0.0f) throw std::runtime_error{"Vorticity cell_size must be positive"};
        if (!std::isfinite(vorticity_confinement)) throw std::runtime_error{"Vorticity confinement must be finite"};
        this->vorticity.resize(resolution);
        this->vorticity_magnitude.resize(resolution);
    }

    void Vorticity::operator()(field::CenteredVectorField3D& force, const field::CenteredVectorField3D& centered_velocity, const std::uint8_t* occupancy) {
        if (force.resolution != this->resolution || centered_velocity.resolution != this->resolution) throw std::runtime_error{"Vorticity field resolution mismatch"};
        require_centered_field(force, "force");
        require_centered_field(centered_velocity, "centered_velocity");
        if (occupancy == nullptr) throw std::runtime_error{"Vorticity occupancy must not be null"};

        const int nx                      = this->resolution[0];
        const int ny                      = this->resolution[1];
        const int nz                      = this->resolution[2];
        const std::uint32_t* flow_types   = this->flow_boundary.types.data();
        const float* flow_velocity        = this->flow_boundary.velocity.data();
        cuda::operators::vorticity::compute_vorticity(this->stream, this->vorticity.data[0], this->vorticity.data[1], this->vorticity.data[2], this->vorticity_magnitude.data, centered_velocity.data[0], centered_velocity.data[1], centered_velocity.data[2], occupancy, nx, ny, nz, this->cell_size, flow_types, flow_velocity);
        cuda::operators::vorticity::add_confinement(this->stream, force.data[0], force.data[1], force.data[2], this->vorticity.data[0], this->vorticity.data[1], this->vorticity.data[2], this->vorticity_magnitude.data, occupancy, nx, ny, nz, this->cell_size, this->confinement, flow_types);
    }
} // namespace kfs::operators
