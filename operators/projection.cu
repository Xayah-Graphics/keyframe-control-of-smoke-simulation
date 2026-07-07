#include "../core/boundary.cuh"
#include "../core/field.cuh"
#include "projection.h"
#include <stdexcept>
#include <string>

namespace xayah::operators::projection::cuda {
    __global__ void reset_pressure_anchor_kernel(int* pressure_anchor, const int value) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        *pressure_anchor = value;
    }

    __global__ void find_pressure_anchor_kernel(int* pressure_anchor, const std::uint32_t* cell_indices, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        if (cell_indices[index] != 0u) return;
        atomicMin(pressure_anchor, static_cast<int>(index));
    }

    __global__ void compute_pressure_rhs_kernel(float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint32_t* cell_indices, const int* pressure_anchor, const int nx, const int ny, const int nz, const float h, const float dt, const core::boundary::cuda::ScalarBoundary3D pressure_boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = core::field::cuda::index(x, y, z, nx, ny);
        const int anchor = *pressure_anchor;
        if (static_cast<int>(index) == anchor) {
            rhs[index] = 0.0f;
            return;
        }
        if (cell_indices[index] != 0u) {
            rhs[index] = 0.0f;
            return;
        }
        const float divergence = (velocity_x[core::field::cuda::index(0u, x + 1, y, z, nx, ny)] - velocity_x[core::field::cuda::index(0u, x, y, z, nx, ny)] + velocity_y[core::field::cuda::index(1u, x, y + 1, z, nx, ny)] - velocity_y[core::field::cuda::index(1u, x, y, z, nx, ny)] + velocity_z[core::field::cuda::index(2u, x, y, z + 1, nx, ny)] - velocity_z[core::field::cuda::index(2u, x, y, z, nx, ny)]) / h;
        float boundary_sum     = 0.0f;
        if (x == 0 && pressure_boundary.x_min.mode == core::boundary::cuda::scalar_boundary_fixed_value) boundary_sum += pressure_boundary.x_min.value;
        if (x == nx - 1 && pressure_boundary.x_max.mode == core::boundary::cuda::scalar_boundary_fixed_value) boundary_sum += pressure_boundary.x_max.value;
        if (y == 0 && pressure_boundary.y_min.mode == core::boundary::cuda::scalar_boundary_fixed_value) boundary_sum += pressure_boundary.y_min.value;
        if (y == ny - 1 && pressure_boundary.y_max.mode == core::boundary::cuda::scalar_boundary_fixed_value) boundary_sum += pressure_boundary.y_max.value;
        if (z == 0 && pressure_boundary.z_min.mode == core::boundary::cuda::scalar_boundary_fixed_value) boundary_sum += pressure_boundary.z_min.value;
        if (z == nz - 1 && pressure_boundary.z_max.mode == core::boundary::cuda::scalar_boundary_fixed_value) boundary_sum += pressure_boundary.z_max.value;
        rhs[index] = -(h * h / dt) * divergence + boundary_sum;
    }

    __device__ void accumulate_pressure_neighbor(int* active_neighbors, int& active_neighbor_count, float& diagonal, int next_x, int next_y, int next_z, const core::boundary::cuda::ScalarBoundaryFace3D min_face, const core::boundary::cuda::ScalarBoundaryFace3D max_face, const bool periodic_axis, const std::uint32_t* cell_indices, const int anchor, const int nx, const int ny, const int nz) {
        if (next_x < 0 || next_x >= nx || next_y < 0 || next_y >= ny || next_z < 0 || next_z >= nz) {
            if (periodic_axis) {
                if (next_x < 0 || next_x >= nx) next_x = core::boundary::cuda::wrap_index(next_x, nx);
                if (next_y < 0 || next_y >= ny) next_y = core::boundary::cuda::wrap_index(next_y, ny);
                if (next_z < 0 || next_z >= nz) next_z = core::boundary::cuda::wrap_index(next_z, nz);
            } else {
                const core::boundary::cuda::ScalarBoundaryFace3D face = next_x < 0 || next_y < 0 || next_z < 0 ? min_face : max_face;
                if (face.mode == core::boundary::cuda::scalar_boundary_fixed_value) diagonal += 1.0f;
                return;
            }
        }
        const int neighbor = static_cast<int>(core::field::cuda::index(next_x, next_y, next_z, nx, ny));
        if (cell_indices[static_cast<std::uint64_t>(neighbor)] != 0u) return;
        diagonal += 1.0f;
        if (neighbor == anchor) return;
        for (int index = 0; index < active_neighbor_count; ++index) {
            if (active_neighbors[index] == neighbor) return;
        }
        active_neighbors[active_neighbor_count] = neighbor;
        ++active_neighbor_count;
    }

    __global__ void build_pressure_matrix_kernel(float* values, const int* row_offsets, const int* column_indices, const std::uint32_t* cell_indices, const int* pressure_anchor, const int nx, const int ny, const int nz, const core::boundary::cuda::ScalarBoundary3D pressure_boundary) {
        const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (row >= nx * ny * nz) return;

        const int anchor      = *pressure_anchor;
        const int x           = row % nx;
        const int yz          = row / nx;
        const int y           = yz % ny;
        const int z           = yz / ny;
        const bool marked     = cell_indices[static_cast<std::uint64_t>(row)] != 0u;
        const bool special    = marked || row == anchor;
        const bool periodic_x = core::boundary::cuda::scalar_periodic_pair(pressure_boundary, 0u);
        const bool periodic_y = core::boundary::cuda::scalar_periodic_pair(pressure_boundary, 1u);
        const bool periodic_z = core::boundary::cuda::scalar_periodic_pair(pressure_boundary, 2u);

        int active_neighbors[6]{};
        int active_neighbor_count = 0;
        float diagonal            = 0.0f;

        if (!special) {
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x - 1, y, z, pressure_boundary.x_min, pressure_boundary.x_max, periodic_x, cell_indices, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x + 1, y, z, pressure_boundary.x_min, pressure_boundary.x_max, periodic_x, cell_indices, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y - 1, z, pressure_boundary.y_min, pressure_boundary.y_max, periodic_y, cell_indices, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y + 1, z, pressure_boundary.y_min, pressure_boundary.y_max, periodic_y, cell_indices, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y, z - 1, pressure_boundary.z_min, pressure_boundary.z_max, periodic_z, cell_indices, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y, z + 1, pressure_boundary.z_min, pressure_boundary.z_max, periodic_z, cell_indices, anchor, nx, ny, nz);
            if (diagonal <= 0.0f) diagonal = 1.0f;
        }

        for (int entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
            const int column = column_indices[entry];
            float value      = 0.0f;
            if (special) {
                value = column == row ? 1.0f : 0.0f;
            } else if (column == row) {
                value = diagonal;
            } else {
                for (int index = 0; index < active_neighbor_count; ++index) {
                    if (active_neighbors[index] == column) {
                        value = -1.0f;
                        break;
                    }
                }
            }
            values[entry] = value;
        }
    }

    __global__ void compute_ratio_kernel(float* destination, const float* numerator, const float* denominator) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        const float value = fabsf(*denominator) > 1.0e-20f ? *numerator / *denominator : 0.0f;
        *destination      = value;
    }

    __global__ void negate_scalar_kernel(float* destination, const float* source) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        *destination = -*source;
    }

    __global__ void project_velocity_x_kernel(float* velocity_x, const float* pressure, const std::uint32_t* cell_indices, const float* constraint_velocity_x, const int nx, const int ny, const int nz, const float h, const float dt, const core::boundary::cuda::VectorBoundary3D boundary_config) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face              = velocity_x[core::field::cuda::index(0u, i, j, k, nx, ny)];
        int left_x              = i - 1;
        int left_y              = j;
        int left_z              = k;
        int right_x             = i;
        int right_y             = j;
        int right_z             = k;
        const bool has_left     = core::boundary::cuda::resolve_cell_coordinates(left_x, left_y, left_z, nx, ny, nz, boundary_config);
        const bool has_right    = core::boundary::cuda::resolve_cell_coordinates(right_x, right_y, right_z, nx, ny, nz, boundary_config);
        const bool left_marked  = has_left && cell_indices[core::field::cuda::index(left_x, left_y, left_z, nx, ny)] != 0u;
        const bool right_marked = has_right && cell_indices[core::field::cuda::index(right_x, right_y, right_z, nx, ny)] != 0u;
        if (left_marked || right_marked) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (left_marked) {
                value += core::boundary::cuda::constraint_value(constraint_velocity_x, cell_indices, left_x, left_y, left_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            if (right_marked) {
                value += core::boundary::cuda::constraint_value(constraint_velocity_x, cell_indices, right_x, right_y, right_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_left && has_right) {
            const float pressure_right = pressure[core::field::cuda::index(right_x, right_y, right_z, nx, ny)];
            const float pressure_left  = pressure[core::field::cuda::index(left_x, left_y, left_z, nx, ny)];
            face -= dt * (pressure_right - pressure_left) / h;
        }
    }

    __global__ void project_velocity_y_kernel(float* velocity_y, const float* pressure, const std::uint32_t* cell_indices, const float* constraint_velocity_y, const int nx, const int ny, const int nz, const float h, const float dt, const core::boundary::cuda::VectorBoundary3D boundary_config) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face             = velocity_y[core::field::cuda::index(1u, i, j, k, nx, ny)];
        int down_x             = i;
        int down_y             = j - 1;
        int down_z             = k;
        int up_x               = i;
        int up_y               = j;
        int up_z               = k;
        const bool has_down    = core::boundary::cuda::resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary_config);
        const bool has_up      = core::boundary::cuda::resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary_config);
        const bool down_marked = has_down && cell_indices[core::field::cuda::index(down_x, down_y, down_z, nx, ny)] != 0u;
        const bool up_marked   = has_up && cell_indices[core::field::cuda::index(up_x, up_y, up_z, nx, ny)] != 0u;
        if (down_marked || up_marked) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (down_marked) {
                value += core::boundary::cuda::constraint_value(constraint_velocity_y, cell_indices, down_x, down_y, down_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            if (up_marked) {
                value += core::boundary::cuda::constraint_value(constraint_velocity_y, cell_indices, up_x, up_y, up_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_down && has_up) {
            const float pressure_up   = pressure[core::field::cuda::index(up_x, up_y, up_z, nx, ny)];
            const float pressure_down = pressure[core::field::cuda::index(down_x, down_y, down_z, nx, ny)];
            face -= dt * (pressure_up - pressure_down) / h;
        }
    }

    __global__ void project_velocity_z_kernel(float* velocity_z, const float* pressure, const std::uint32_t* cell_indices, const float* constraint_velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const core::boundary::cuda::VectorBoundary3D boundary_config) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face              = velocity_z[core::field::cuda::index(2u, i, j, k, nx, ny)];
        int back_x              = i;
        int back_y              = j;
        int back_z              = k - 1;
        int front_x             = i;
        int front_y             = j;
        int front_z             = k;
        const bool has_back     = core::boundary::cuda::resolve_cell_coordinates(back_x, back_y, back_z, nx, ny, nz, boundary_config);
        const bool has_front    = core::boundary::cuda::resolve_cell_coordinates(front_x, front_y, front_z, nx, ny, nz, boundary_config);
        const bool back_marked  = has_back && cell_indices[core::field::cuda::index(back_x, back_y, back_z, nx, ny)] != 0u;
        const bool front_marked = has_front && cell_indices[core::field::cuda::index(front_x, front_y, front_z, nx, ny)] != 0u;
        if (back_marked || front_marked) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (back_marked) {
                value += core::boundary::cuda::constraint_value(constraint_velocity_z, cell_indices, back_x, back_y, back_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            if (front_marked) {
                value += core::boundary::cuda::constraint_value(constraint_velocity_z, cell_indices, front_x, front_y, front_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_back && has_front) {
            const float pressure_front = pressure[core::field::cuda::index(front_x, front_y, front_z, nx, ny)];
            const float pressure_back  = pressure[core::field::cuda::index(back_x, back_y, back_z, nx, ny)];
            face -= dt * (pressure_front - pressure_back) / h;
        }
    }

    void reset_pressure_anchor(cudaStream_t stream, int* pressure_anchor, const int value) {
        reset_pressure_anchor_kernel<<<1, 1, 0, stream>>>(pressure_anchor, value);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"reset_pressure_anchor_kernel: "} + cudaGetErrorString(status)};
    }

    void find_pressure_anchor(cudaStream_t stream, int* pressure_anchor, const std::uint32_t* cell_indices, const std::uint64_t count) {
        constexpr unsigned block = 256u;
        const unsigned grid      = core::field::cuda::ceil_div_u32(count, block);
        find_pressure_anchor_kernel<<<grid, block, 0, stream>>>(pressure_anchor, cell_indices, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"find_pressure_anchor_kernel: "} + cudaGetErrorString(status)};
    }

    void compute_pressure_rhs(cudaStream_t stream, float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint32_t* cell_indices, const int* pressure_anchor, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t* pressure_boundary_modes, const float* pressure_boundary_values) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid                                    = core::field::cuda::centered_grid(nx, ny, nz, block);
        const core::boundary::cuda::ScalarBoundary3D pressure_boundary = core::boundary::cuda::make_scalar_boundary(pressure_boundary_modes, pressure_boundary_values);
        compute_pressure_rhs_kernel<<<grid, block, 0, stream>>>(rhs, velocity_x, velocity_y, velocity_z, cell_indices, pressure_anchor, nx, ny, nz, h, dt, pressure_boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_pressure_rhs_kernel: "} + cudaGetErrorString(status)};
    }

    void build_pressure_matrix(cudaStream_t stream, float* values, const int* row_offsets, const int* column_indices, const std::uint32_t* cell_indices, const int* pressure_anchor, const int nx, const int ny, const int nz, const std::uint32_t* pressure_boundary_modes) {
        const auto count         = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        constexpr unsigned block = 256u;
        const unsigned grid      = core::field::cuda::ceil_div_u32(count, block);
        constexpr float zero_values[6]{};
        const core::boundary::cuda::ScalarBoundary3D pressure_boundary = core::boundary::cuda::make_scalar_boundary(pressure_boundary_modes, zero_values);
        build_pressure_matrix_kernel<<<grid, block, 0, stream>>>(values, row_offsets, column_indices, cell_indices, pressure_anchor, nx, ny, nz, pressure_boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"build_pressure_matrix_kernel: "} + cudaGetErrorString(status)};
    }

    void compute_ratio(cudaStream_t stream, float* destination, const float* numerator, const float* denominator) {
        compute_ratio_kernel<<<1, 1, 0, stream>>>(destination, numerator, denominator);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_ratio_kernel: "} + cudaGetErrorString(status)};
    }

    void negate_scalar(cudaStream_t stream, float* destination, const float* source) {
        negate_scalar_kernel<<<1, 1, 0, stream>>>(destination, source);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"negate_scalar_kernel: "} + cudaGetErrorString(status)};
    }

    void project_staggered_component(cudaStream_t stream, const std::uint32_t axis, float* velocity_component, const float* pressure, const std::uint32_t* cell_indices, const float* constraint_velocity_component, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t* velocity_boundary_modes, const float* velocity_boundary_values) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid                                  = core::field::cuda::staggered_grid(axis, nx, ny, nz, block);
        const core::boundary::cuda::VectorBoundary3D boundary_config = core::boundary::cuda::make_vector_boundary(velocity_boundary_modes, velocity_boundary_values);
        if (axis == 0u) project_velocity_x_kernel<<<grid, block, 0, stream>>>(velocity_component, pressure, cell_indices, constraint_velocity_component, nx, ny, nz, h, dt, boundary_config);
        if (axis == 1u) project_velocity_y_kernel<<<grid, block, 0, stream>>>(velocity_component, pressure, cell_indices, constraint_velocity_component, nx, ny, nz, h, dt, boundary_config);
        if (axis == 2u) project_velocity_z_kernel<<<grid, block, 0, stream>>>(velocity_component, pressure, cell_indices, constraint_velocity_component, nx, ny, nz, h, dt, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"project_staggered_component_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace xayah::operators::projection::cuda
