module;
#include <cuda_runtime.h>

export module keyframe.operators.scalar_force;
import std;
import keyframe.field;
import keyframe.boundary;

export namespace kfs::operators {
    struct ScalarForce final {
        ScalarForce(cudaStream_t stream, const boundary::PackedFlowBoundary& boundary);
        ScalarForce(const ScalarForce&)                = delete;
        ScalarForce& operator=(const ScalarForce&)     = delete;
        ScalarForce(ScalarForce&&) noexcept            = delete;
        ScalarForce& operator=(ScalarForce&&) noexcept = delete;

        void operator()(field::CenteredVectorField3D& destination, std::uint32_t axis, const field::ScalarField3D& source, float scale, float bias, const std::uint8_t* cell_mask) const;

    private:
        cudaStream_t stream{nullptr};
        boundary::PackedFlowBoundary flow_boundary{};
    };
} // namespace kfs::operators
