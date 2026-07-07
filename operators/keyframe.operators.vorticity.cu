#include "../keyframe.boundary.h"
#include "keyframe.operators.vorticity.h"
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::vorticity {
    unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    __global__ void compute_vorticity_kernel(float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = boundary::index_3d(x, y, z, nx, ny);
        if (boundary::cell_is_marked(cell_indices, x, y, z, nx, ny, nz, boundary_config)) {
            omega_x[index]         = 0.0f;
            omega_y[index]         = 0.0f;
            omega_z[index]         = 0.0f;
            omega_magnitude[index] = 0.0f;
            return;
        }

        const float dvz_dy = 0.5f * (boundary::load_center_velocity_component(velocity_z, 2, x, y + 1, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(velocity_z, 2, x, y - 1, z, nx, ny, nz, boundary_config)) / h;
        const float dvy_dz = 0.5f * (boundary::load_center_velocity_component(velocity_y, 1, x, y, z + 1, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(velocity_y, 1, x, y, z - 1, nx, ny, nz, boundary_config)) / h;
        const float dvx_dz = 0.5f * (boundary::load_center_velocity_component(velocity_x, 0, x, y, z + 1, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(velocity_x, 0, x, y, z - 1, nx, ny, nz, boundary_config)) / h;
        const float dvz_dx = 0.5f * (boundary::load_center_velocity_component(velocity_z, 2, x + 1, y, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(velocity_z, 2, x - 1, y, z, nx, ny, nz, boundary_config)) / h;
        const float dvy_dx = 0.5f * (boundary::load_center_velocity_component(velocity_y, 1, x + 1, y, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(velocity_y, 1, x - 1, y, z, nx, ny, nz, boundary_config)) / h;
        const float dvx_dy = 0.5f * (boundary::load_center_velocity_component(velocity_x, 0, x, y + 1, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(velocity_x, 0, x, y - 1, z, nx, ny, nz, boundary_config)) / h;

        const float wx = dvz_dy - dvy_dz;
        const float wy = dvx_dz - dvz_dx;
        const float wz = dvy_dx - dvx_dy;

        omega_x[index]         = wx;
        omega_y[index]         = wy;
        omega_z[index]         = wz;
        omega_magnitude[index] = sqrtf(wx * wx + wy * wy + wz * wz);
    }

    __global__ void add_confinement_kernel(float* destination_x, float* destination_y, float* destination_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float h, const float epsilon, const boundary::FlowBoundary boundary_config) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (boundary::cell_is_marked(cell_indices, x, y, z, nx, ny, nz, boundary_config)) return;

        const float grad_x   = 0.5f * (boundary::load_flow_cell(omega_magnitude, x + 1, y, z, nx, ny, nz, boundary_config) - boundary::load_flow_cell(omega_magnitude, x - 1, y, z, nx, ny, nz, boundary_config)) / h;
        const float grad_y   = 0.5f * (boundary::load_flow_cell(omega_magnitude, x, y + 1, z, nx, ny, nz, boundary_config) - boundary::load_flow_cell(omega_magnitude, x, y - 1, z, nx, ny, nz, boundary_config)) / h;
        const float grad_z   = 0.5f * (boundary::load_flow_cell(omega_magnitude, x, y, z + 1, nx, ny, nz, boundary_config) - boundary::load_flow_cell(omega_magnitude, x, y, z - 1, nx, ny, nz, boundary_config)) / h;
        const float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
        if (grad_mag < 1.0e-6f) return;

        const float inv_grad      = 1.0f / grad_mag;
        const float normal_x      = grad_x * inv_grad;
        const float normal_y      = grad_y * inv_grad;
        const float normal_z      = grad_z * inv_grad;
        const auto index          = boundary::index_3d(x, y, z, nx, ny);
        const float wx            = omega_x[index];
        const float wy            = omega_y[index];
        const float wz            = omega_z[index];
        const float confinement_x = epsilon * h * (normal_y * wz - normal_z * wy);
        const float confinement_y = epsilon * h * (normal_z * wx - normal_x * wz);
        const float confinement_z = epsilon * h * (normal_x * wy - normal_y * wx);

        destination_x[index] += confinement_x;
        destination_y[index] += confinement_y;
        destination_z[index] += confinement_z;
    }

    void compute_vorticity(cudaStream_t stream, float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float cell_size, const std::uint32_t* flow_types, const float* flow_velocity) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid{ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z)};
        const boundary::FlowBoundary boundary_config = boundary::make_flow_velocity_boundary(flow_types, flow_velocity);
        compute_vorticity_kernel<<<grid, block, 0, stream>>>(omega_x, omega_y, omega_z, omega_magnitude, velocity_x, velocity_y, velocity_z, cell_indices, nx, ny, nz, cell_size, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_vorticity_kernel: "} + cudaGetErrorString(status)};
    }

    void add_confinement(cudaStream_t stream, float* destination_x, float* destination_y, float* destination_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float cell_size, const float confinement, const std::uint32_t* flow_types) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid{ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z)};
        const boundary::FlowBoundary boundary_config = boundary::make_flow_type_boundary(flow_types);
        add_confinement_kernel<<<grid, block, 0, stream>>>(destination_x, destination_y, destination_z, omega_x, omega_y, omega_z, omega_magnitude, cell_indices, nx, ny, nz, cell_size, confinement, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_confinement_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::vorticity
