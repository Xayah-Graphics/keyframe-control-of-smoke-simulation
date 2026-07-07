#include "keyframe.operators.scalar_force.h"
#include "../keyframe.boundary.h"
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::scalar_force {
    unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    __global__ void add_scalar_force_kernel(float* destination, const float* source, const std::uint8_t* cell_mask, const int nx, const int ny, const int nz, const float scale, const float bias, const boundary::FlowBoundary boundary_config) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (boundary::cell_is_masked(cell_mask, x, y, z, nx, ny, nz, boundary_config)) return;
        const auto index = boundary::index_3d(x, y, z, nx, ny);
        destination[index] += scale * (source[index] + bias);
    }

    void add_scalar_force(cudaStream_t stream, float* destination, const float* source, const std::uint8_t* cell_mask, const int nx, const int ny, const int nz, const float scale, const float bias, const std::uint32_t* flow_types) {
        if (destination == nullptr || source == nullptr || cell_mask == nullptr || flow_types == nullptr) throw std::runtime_error{"ScalarForce arrays must not be null"};
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error{"ScalarForce launch resolution must be positive"};
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid{ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z)};
        const boundary::FlowBoundary boundary_config = boundary::make_flow_type_boundary(flow_types);
        add_scalar_force_kernel<<<grid, block, 0, stream>>>(destination, source, cell_mask, nx, ny, nz, scale, bias, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_scalar_force_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::scalar_force
