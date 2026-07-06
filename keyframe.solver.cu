#include "keyframe.solver.h"

#include "keyframe.boundary.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace kfs::cuda {
    unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    dim3 cell_block() {
        return dim3{8u, 8u, 4u};
    }

    unsigned linear_block() {
        return 256u;
    }

    void require_count(const std::uint64_t count) {
        if (count == 0u) throw std::runtime_error{"Keyframe smoke launch count must be positive"};
    }

    void require_resolution(const int nx, const int ny, const int nz) {
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error{"Keyframe smoke launch resolution must be positive"};
    }

    unsigned linear_launch_grid(const std::uint64_t count, const unsigned block) {
        require_count(count);
        return ceil_div_u32(count, block);
    }

    dim3 scalar_launch_grid(const int nx, const int ny, const int nz, const dim3& block) {
        require_resolution(nx, ny, nz);
        return dim3{ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z)};
    }

    __global__ void apply_solid_temperature_kernel(float* temperature, const std::uint8_t* occupancy, const float* solid_temperature, const int nx, const int ny, const int nz, const float ambient_temperature) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        const auto count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (index >= count) return;
        if (occupancy == nullptr || occupancy[index] == 0) return;
        temperature[index] = solid_temperature != nullptr ? solid_temperature[index] : ambient_temperature;
    }

    __global__ void compute_vorticity_kernel(float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* cell_x, const float* cell_y, const float* cell_z, const std::uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = boundary::index_3d(x, y, z, nx, ny);
        if (boundary::load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary_config)) {
            omega_x[index]         = 0.0f;
            omega_y[index]         = 0.0f;
            omega_z[index]         = 0.0f;
            omega_magnitude[index] = 0.0f;
            return;
        }

        const float dvz_dy = 0.5f * (boundary::load_center_velocity_component(cell_z, 2, x, y + 1, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(cell_z, 2, x, y - 1, z, nx, ny, nz, boundary_config)) / h;
        const float dvy_dz = 0.5f * (boundary::load_center_velocity_component(cell_y, 1, x, y, z + 1, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(cell_y, 1, x, y, z - 1, nx, ny, nz, boundary_config)) / h;
        const float dvx_dz = 0.5f * (boundary::load_center_velocity_component(cell_x, 0, x, y, z + 1, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(cell_x, 0, x, y, z - 1, nx, ny, nz, boundary_config)) / h;
        const float dvz_dx = 0.5f * (boundary::load_center_velocity_component(cell_z, 2, x + 1, y, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(cell_z, 2, x - 1, y, z, nx, ny, nz, boundary_config)) / h;
        const float dvy_dx = 0.5f * (boundary::load_center_velocity_component(cell_y, 1, x + 1, y, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(cell_y, 1, x - 1, y, z, nx, ny, nz, boundary_config)) / h;
        const float dvx_dy = 0.5f * (boundary::load_center_velocity_component(cell_x, 0, x, y + 1, z, nx, ny, nz, boundary_config) - boundary::load_center_velocity_component(cell_x, 0, x, y - 1, z, nx, ny, nz, boundary_config)) / h;

        const float wx = dvz_dy - dvy_dz;
        const float wy = dvx_dz - dvz_dx;
        const float wz = dvy_dx - dvx_dy;

        omega_x[index]         = wx;
        omega_y[index]         = wy;
        omega_z[index]         = wz;
        omega_magnitude[index] = sqrtf(wx * wx + wy * wy + wz * wz);
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

    __global__ void add_confinement_kernel(float* force_x, float* force_y, float* force_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const std::uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float epsilon, const boundary::FlowBoundary boundary_config) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (boundary::load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary_config)) return;

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

        force_x[index] += confinement_x;
        force_y[index] += confinement_y;
        force_z[index] += confinement_z;
    }

    void apply_solid_scalar(cudaStream_t stream, float* scalar, const std::uint8_t* occupancy, const float* solid_scalar, const int nx, const int ny, const int nz, const float default_value) {
        require_resolution(nx, ny, nz);
        const unsigned block = linear_block();
        const unsigned grid  = linear_launch_grid(static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz), block);
        apply_solid_temperature_kernel<<<grid, block, 0, stream>>>(scalar, occupancy, solid_scalar, nx, ny, nz, default_value);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"apply_solid_temperature_kernel: "} + cudaGetErrorString(status)};
    }

    void compute_vorticity(cudaStream_t stream, float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* cell_x, const float* cell_y, const float* cell_z, const std::uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const std::uint32_t* flow_types, const float* flow_velocity) {
        const dim3 block = cell_block();
        const dim3 grid  = scalar_launch_grid(nx, ny, nz, block);
        const boundary::FlowBoundary boundary_config = boundary::make_flow_velocity_boundary(flow_types, flow_velocity);
        compute_vorticity_kernel<<<grid, block, 0, stream>>>(omega_x, omega_y, omega_z, omega_magnitude, cell_x, cell_y, cell_z, occupancy, nx, ny, nz, h, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_vorticity_kernel: "} + cudaGetErrorString(status)};
    }

    void add_buoyancy(cudaStream_t stream, float* force_y, const float* density, const float* temperature, const std::uint8_t* occupancy, const int nx, const int ny, const int nz, const float ambient_temperature, const float density_factor, const float temperature_factor, const std::uint32_t* flow_types) {
        const dim3 block = cell_block();
        const dim3 grid  = scalar_launch_grid(nx, ny, nz, block);
        const boundary::FlowBoundary boundary_config = boundary::make_flow_type_boundary(flow_types);
        add_buoyancy_kernel<<<grid, block, 0, stream>>>(force_y, density, temperature, occupancy, nx, ny, nz, ambient_temperature, density_factor, temperature_factor, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_buoyancy_kernel: "} + cudaGetErrorString(status)};
    }

    void add_vorticity_confinement(cudaStream_t stream, float* force_x, float* force_y, float* force_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const std::uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float epsilon, const std::uint32_t* flow_types) {
        const dim3 block = cell_block();
        const dim3 grid  = scalar_launch_grid(nx, ny, nz, block);
        const boundary::FlowBoundary boundary_config = boundary::make_flow_type_boundary(flow_types);
        add_confinement_kernel<<<grid, block, 0, stream>>>(force_x, force_y, force_z, omega_x, omega_y, omega_z, omega_magnitude, occupancy, nx, ny, nz, h, epsilon, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_confinement_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda
