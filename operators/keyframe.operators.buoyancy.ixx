module;
#include <cuda_runtime.h>

export module keyframe.operators.buoyancy;
import std;
import keyframe.field;
import keyframe.boundary;

export namespace kfs::operators {
    struct Buoyancy final {
        Buoyancy(cudaStream_t stream, float ambient_temperature, float density_factor, float temperature_factor, const boundary::PackedFlowBoundary& boundary);
        Buoyancy(const Buoyancy&)                = delete;
        Buoyancy& operator=(const Buoyancy&)     = delete;
        Buoyancy(Buoyancy&&) noexcept            = delete;
        Buoyancy& operator=(Buoyancy&&) noexcept = delete;

        void operator()(field::CenteredVectorField3D& force, const field::ScalarField3D& density, const field::ScalarField3D& temperature, const std::uint8_t* occupancy) const;

    private:
        cudaStream_t stream{nullptr};
        float ambient_temperature{0.0f};
        float density_factor{0.0f};
        float temperature_factor{0.0f};
        boundary::PackedFlowBoundary flow_boundary{};
    };
} // namespace kfs::operators
