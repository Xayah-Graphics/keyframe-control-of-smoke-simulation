module;
#include <cuda_runtime.h>

export module xayah.operators.advection;
import std;
import xayah.core.field;
import xayah.core.boundary;

export namespace xayah::operators {
    struct Advection final {
        enum class Scheme : std::uint32_t {
            linear          = 0,
            monotonic_cubic = 1,
        };

        Advection(cudaStream_t stream, float cell_size, Scheme scheme);

        Scheme scheme{Scheme::monotonic_cubic};

        void operator()(core::field::StaggeredVectorField3D& destination, std::uint32_t axis, const core::field::StaggeredVectorField3D& source, const core::field::StaggeredVectorField3D& vector_field, const core::field::IndexedField3D& cell_indices, float delta_seconds, const core::boundary::PackedVectorBoundary3D& boundary) const;
        void operator()(core::field::ScalarField3D& destination, const core::field::ScalarField3D& source, const core::field::StaggeredVectorField3D& vector_field, const core::field::IndexedField3D& cell_indices, float delta_seconds, const core::boundary::PackedScalarBoundary3D& scalar_boundary, const core::boundary::PackedVectorBoundary3D& vector_boundary) const;

    private:
        CUstream_st* const stream{nullptr};
        const float cell_size{0.0f};
    };
} // namespace xayah::operators
