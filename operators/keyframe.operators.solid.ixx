module;
#include <cuda_runtime.h>

export module keyframe.operators.solid;
import std;
import keyframe.field;

export namespace kfs::operators {
    struct Solid final {
        explicit Solid(cudaStream_t stream);
        Solid(const Solid&)                = delete;
        Solid& operator=(const Solid&)     = delete;
        Solid(Solid&&) noexcept            = delete;
        Solid& operator=(Solid&&) noexcept = delete;

        void apply_scalar(field::ScalarField3D& scalar, const field::ScalarField3D& solid_scalar, const std::uint8_t* occupancy) const;

    private:
        cudaStream_t stream{nullptr};
    };
} // namespace kfs::operators
