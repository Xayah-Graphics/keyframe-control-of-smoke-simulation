module;
#include <cuda_runtime.h>

export module keyframe.collider;
import std;
import keyframe.field;
export import keyframe.geometry;

export namespace kfs::collider {
    struct Collider final {
        std::variant<geometry::Ellipsoid, geometry::Box> shape{geometry::Ellipsoid{}};
        std::array<float, 3> velocity{};
        float constraint_scalar{0.0f};
    };

    struct ColliderSet final {
        std::vector<Collider> items{};

        void rasterize(cudaStream_t stream, field::IndexedField3D& cell_indices, field::CenteredVectorField3D& constraint_velocity, field::ScalarField3D& constraint_scalar, float cell_size) const;
    };
} // namespace kfs::collider
