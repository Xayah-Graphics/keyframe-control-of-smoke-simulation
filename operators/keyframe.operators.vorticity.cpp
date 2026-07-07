module;
#include "keyframe.operators.vorticity.h"

module keyframe.operators.vorticity;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    Vorticity::Vorticity(const cudaStream_t stream, const std::array<std::int32_t, 3> resolution, const float cell_size, const float vorticity_confinement, const boundary::PackedFlowBoundary& boundary) : stream{stream}, resolution{resolution}, cell_size{cell_size}, confinement{vorticity_confinement}, flow_boundary{boundary} {
        this->vorticity.resize(resolution);
        this->vorticity_magnitude.resize(resolution);
    }

    void Vorticity::operator()(field::CenteredVectorField3D& destination, const field::CenteredVectorField3D& source, const field::IndexedField3D& cell_indices) {
        const int nx                    = this->resolution[0];
        const int ny                    = this->resolution[1];
        const int nz                    = this->resolution[2];
        const std::uint32_t* flow_types = this->flow_boundary.types.data();
        const float* flow_velocity      = this->flow_boundary.velocity.data();
        cuda::operators::vorticity::compute_vorticity(this->stream, this->vorticity.data[0], this->vorticity.data[1], this->vorticity.data[2], this->vorticity_magnitude.data, source.data[0], source.data[1], source.data[2], cell_indices.data, nx, ny, nz, this->cell_size, flow_types, flow_velocity);
        cuda::operators::vorticity::add_confinement(this->stream, destination.data[0], destination.data[1], destination.data[2], this->vorticity.data[0], this->vorticity.data[1], this->vorticity.data[2], this->vorticity_magnitude.data, cell_indices.data, nx, ny, nz, this->cell_size, this->confinement, flow_types);
    }
} // namespace kfs::operators
