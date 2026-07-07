module;
#include <cuda_runtime.h>

export module keyframe.operators.masked_scalar_assignment;
import std;
import keyframe.field;

export namespace kfs::operators {
    struct MaskedScalarAssignment final {
        explicit MaskedScalarAssignment(cudaStream_t stream);
        MaskedScalarAssignment(const MaskedScalarAssignment&)                = delete;
        MaskedScalarAssignment& operator=(const MaskedScalarAssignment&)     = delete;
        MaskedScalarAssignment(MaskedScalarAssignment&&) noexcept            = delete;
        MaskedScalarAssignment& operator=(MaskedScalarAssignment&&) noexcept = delete;

        void operator()(field::ScalarField3D& destination, const field::ScalarField3D& source, const std::uint8_t* mask) const;

    private:
        cudaStream_t stream{nullptr};
    };
} // namespace kfs::operators
