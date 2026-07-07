#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_COLLIDER_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_COLLIDER_H

#include <array>
#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda::collider {
    void rasterize_ellipsoid(
            cudaStream_t stream,
            std::uint32_t* cell_indices,
            float* velocity_x,
            float* velocity_y,
            float* velocity_z,
            float* scalar,
            int nx,
            int ny,
            int nz,
            float cell_size,
            std::uint32_t tag,
            std::array<float, 3> center,
            std::array<float, 3> radius,
            std::array<float, 3> velocity,
            float scalar_value
    );
    void rasterize_box(
            cudaStream_t stream,
            std::uint32_t* cell_indices,
            float* velocity_x,
            float* velocity_y,
            float* velocity_z,
            float* scalar,
            int nx,
            int ny,
            int nz,
            float cell_size,
            std::uint32_t tag,
            std::array<float, 3> center,
            std::array<float, 3> half_extent,
            std::array<float, 3> velocity,
            float scalar_value
    );
} // namespace kfs::cuda::collider

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_COLLIDER_H
