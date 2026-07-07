module;
#include <cuda_runtime.h>

export module keyframe.operators.emitter;
import std;
import keyframe.field;
export import keyframe.geometry;

export namespace kfs::operators {
    struct Emitter final {
        struct Source final {
            geometry::Ellipsoid region{
                .center = {0.6f, 0.216f, 0.6f},
                .radius = {0.156f, 0.144f, 0.156f},
            };
            float falloff{2.2f};
        };

        Emitter(cudaStream_t stream, std::array<std::int32_t, 3> resolution, float cell_size, const Source& source);
        Emitter(const Emitter&)                = delete;
        Emitter& operator=(const Emitter&)     = delete;
        Emitter(Emitter&&) noexcept            = delete;
        Emitter& operator=(Emitter&&) noexcept = delete;

        void operator()(field::ScalarField3D& destination, const field::ScalarField3D& current, float rate, float delta_seconds) const;

    private:
        cudaStream_t stream{nullptr};
        std::array<std::int32_t, 3> resolution{0, 0, 0};
        float cell_size{0.0f};
        Source source{};
    };
} // namespace kfs::operators
