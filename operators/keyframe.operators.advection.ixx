module;
#include <cuda_runtime.h>

export module keyframe.operators.advection;
import std;
import keyframe.field;
import keyframe.boundary;

export namespace kfs::operators {
    struct Advection final {
        enum class Scheme : std::uint32_t {
            linear          = 0,
            monotonic_cubic = 1,
        };

        Advection(cudaStream_t stream, float cell_size, Scheme scheme);

        Scheme scheme{Scheme::monotonic_cubic};

        void operator()(field::StaggeredVectorField3D& destination, std::uint32_t axis, const field::StaggeredVectorField3D& source, const field::StaggeredVectorField3D& vector_field, const field::IndexedField3D& cell_indices, float delta_seconds, const boundary::PackedVectorBoundary3D& boundary) const;
        void operator()(field::ScalarField3D& destination, const field::ScalarField3D& source, const field::StaggeredVectorField3D& vector_field, const field::IndexedField3D& cell_indices, float delta_seconds, const boundary::PackedScalarBoundary3D& scalar_boundary, const boundary::PackedVectorBoundary3D& vector_boundary) const;

    private:
        CUstream_st* const stream{nullptr};
        const float cell_size{0.0f};
    };
} // namespace kfs::operators
