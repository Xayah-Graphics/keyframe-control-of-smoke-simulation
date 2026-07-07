module;
#include <cuda_runtime.h>

export module xayah.core.collider;
import std;
import xayah.core.field;
export import xayah.core.geometry;

export namespace xayah::core::collider {
    struct Collider final {
        std::variant<geometry::Ellipsoid, geometry::Box> shape{geometry::Ellipsoid{}};
        std::array<float, 3> velocity{};
        float constraint_scalar{0.0f};
    };

    struct ColliderSet final {
        std::vector<Collider> items{};

        void rasterize(cudaStream_t stream, field::IndexedField3D& cell_indices, field::CenteredVectorField3D& constraint_velocity, field::ScalarField3D& constraint_scalar, float cell_size) const;
    };
} // namespace xayah::core::collider
