module;
#include <cuda_runtime.h>

export module xayah.operators.vorticity;
import std;
import xayah.core.field;
import xayah.core.boundary;

export namespace xayah::operators {
    struct Vorticity final {
        Vorticity(cudaStream_t stream, std::array<std::int32_t, 3> resolution, float cell_size, float vorticity_confinement);
        Vorticity(const Vorticity&)                = delete;
        Vorticity& operator=(const Vorticity&)     = delete;
        Vorticity(Vorticity&&) noexcept            = delete;
        Vorticity& operator=(Vorticity&&) noexcept = delete;

        float confinement{0.0f};

        void operator()(core::field::CenteredVectorField3D& destination, const core::field::CenteredVectorField3D& source, const core::field::IndexedField3D& cell_indices, const core::boundary::PackedVectorBoundary3D& vector_boundary);

    private:
        CUstream_st* const stream{nullptr};
        const std::array<std::int32_t, 3> resolution{0, 0, 0};
        const float cell_size{0.0f};
        core::field::CenteredVectorField3D vorticity{{0, 0, 0}};
        core::field::ScalarField3D vorticity_magnitude{{0, 0, 0}};
    };
} // namespace xayah::operators
