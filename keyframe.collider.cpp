module;
#include "keyframe.collider.h"
#include <cuda_runtime.h>

module keyframe.collider;
import std;
import keyframe.field;
import keyframe.geometry;

namespace kfs::collider {
    void ColliderSet::rasterize(
            cudaStream_t stream,
            field::IndexedField3D& cell_indices,
            field::CenteredVectorField3D& constraint_velocity,
            field::ScalarField3D& constraint_scalar,
            const float cell_size
    ) const {
        field::fill(stream, cell_indices, 0u);
        field::fill(stream, constraint_velocity, 0.0f);
        field::fill(stream, constraint_scalar, 0.0f);

        const int nx = cell_indices.resolution[0];
        const int ny = cell_indices.resolution[1];
        const int nz = cell_indices.resolution[2];
        for (std::size_t item_index = 0u; item_index < this->items.size(); ++item_index) {
            const Collider& item = this->items[item_index];
            const std::uint32_t tag = static_cast<std::uint32_t>(item_index + 1u);
            const geometry::Ellipsoid* const ellipsoid = std::get_if<geometry::Ellipsoid>(&item.shape);
            if (ellipsoid != nullptr) {
                cuda::collider::rasterize_ellipsoid(
                        stream,
                        cell_indices.data,
                        constraint_velocity.data[0],
                        constraint_velocity.data[1],
                        constraint_velocity.data[2],
                        constraint_scalar.data,
                        nx,
                        ny,
                        nz,
                        cell_size,
                        tag,
                        ellipsoid->center,
                        ellipsoid->radius,
                        item.velocity,
                        item.constraint_scalar
                );
                continue;
            }
            const geometry::Box* const box = std::get_if<geometry::Box>(&item.shape);
            if (box != nullptr) {
                cuda::collider::rasterize_box(
                        stream,
                        cell_indices.data,
                        constraint_velocity.data[0],
                        constraint_velocity.data[1],
                        constraint_velocity.data[2],
                        constraint_scalar.data,
                        nx,
                        ny,
                        nz,
                        cell_size,
                        tag,
                        box->center,
                        box->half_extent,
                        item.velocity,
                        item.constraint_scalar
                );
            }
        }
    }
} // namespace kfs::collider
