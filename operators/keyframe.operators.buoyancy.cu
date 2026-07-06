#include "keyframe.operators.buoyancy.h"
#include "../keyframe.boundary.h"
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::buoyancy {
    unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    __global__ void add_buoyancy_kernel(float* force_y, const float* density, const float* temperature, const std::uint8_t* occupancy, const int nx, const int ny, const int nz, const float ambient_temperature, const float density_factor, const float temperature_factor, const boundary::FlowBoundary boundary_config) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (boundary::load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary_config)) return;
        const auto index = boundary::index_3d(x, y, z, nx, ny);
        force_y[index] += -density_factor * density[index] + temperature_factor * (temperature[index] - ambient_temperature);
    }

    void add_buoyancy(cudaStream_t stream, float* force_y, const float* density, const float* temperature, const std::uint8_t* occupancy, const int nx, const int ny, const int nz, const float ambient_temperature, const float density_factor, const float temperature_factor, const std::uint32_t* flow_types) {
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error{"Buoyancy launch resolution must be positive"};
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid{ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z)};
        const boundary::FlowBoundary boundary_config = boundary::make_flow_type_boundary(flow_types);
        add_buoyancy_kernel<<<grid, block, 0, stream>>>(force_y, density, temperature, occupancy, nx, ny, nz, ambient_temperature, density_factor, temperature_factor, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_buoyancy_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::buoyancy
