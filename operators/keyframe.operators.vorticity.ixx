module;
#include <cuda_runtime.h>

export module keyframe.operators.vorticity;
import std;
import keyframe.field;
import keyframe.boundary;

export namespace kfs::operators {
    struct Vorticity final {
        Vorticity(cudaStream_t stream, std::array<std::int32_t, 3> resolution, float cell_size, float vorticity_confinement, const boundary::PackedFlowBoundary& boundary);
        Vorticity(const Vorticity&)                = delete;
        Vorticity& operator=(const Vorticity&)     = delete;
        Vorticity(Vorticity&&) noexcept            = delete;
        Vorticity& operator=(Vorticity&&) noexcept = delete;

        void operator()(field::CenteredVectorField3D& destination, const field::CenteredVectorField3D& source, const std::uint8_t* cell_mask);

    private:
        cudaStream_t stream{nullptr};
        std::array<std::int32_t, 3> resolution{0, 0, 0};
        float cell_size{0.0f};
        float confinement{0.0f};
        boundary::PackedFlowBoundary flow_boundary{};
        field::CenteredVectorField3D vorticity{{0, 0, 0}};
        field::ScalarField3D vorticity_magnitude{{0, 0, 0}};
    };
} // namespace kfs::operators
