module;
#include <cuda_runtime.h>

export module xayah.operators.emitter;
import std;
import xayah.core.field;
export import xayah.core.geometry;

export namespace xayah::operators {
    struct Emitter final {
        struct Source final {
            core::geometry::Ellipsoid region{
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

        Source source{};

        void operator()(core::field::ScalarField3D& destination, const core::field::ScalarField3D& current, float rate, float delta_seconds) const;

    private:
        CUstream_st* const stream{nullptr};
        const std::array<std::int32_t, 3> resolution{0, 0, 0};
        const float cell_size{0.0f};
    };
} // namespace xayah::operators
