#include "keyframe.operators.projection.h"
#include "../keyframe.boundary.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::projection {
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
        if (count == 0u) throw std::runtime_error{"Projection launch count must be positive"};
    }

    void require_resolution(const int nx, const int ny, const int nz) {
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error{"Projection launch resolution must be positive"};
    }

    unsigned linear_launch_grid(const std::uint64_t count, const unsigned block) {
        require_count(count);
        return ceil_div_u32(count, block);
    }

    dim3 scalar_launch_grid(const int nx, const int ny, const int nz, const dim3& block) {
        require_resolution(nx, ny, nz);
        return dim3(ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z));
    }

    dim3 staggered_launch_grid(const std::uint32_t axis, const int nx, const int ny, const int nz, const dim3& block) {
        if (axis >= 3u) throw std::runtime_error{"Projection staggered launch axis must be 0, 1, or 2"};
        require_resolution(nx, ny, nz);
        const auto nx64 = static_cast<std::uint64_t>(nx);
        const auto ny64 = static_cast<std::uint64_t>(ny);
        const auto nz64 = static_cast<std::uint64_t>(nz);
        if (axis == 0u) return dim3(ceil_div_u32(nx64 + 1u, block.x), ceil_div_u32(ny64, block.y), ceil_div_u32(nz64, block.z));
        if (axis == 1u) return dim3(ceil_div_u32(nx64, block.x), ceil_div_u32(ny64 + 1u, block.y), ceil_div_u32(nz64, block.z));
        return dim3(ceil_div_u32(nx64, block.x), ceil_div_u32(ny64, block.y), ceil_div_u32(nz64 + 1u, block.z));
    }

    __global__ void reset_pressure_anchor_kernel(int* pressure_anchor, const int value) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        *pressure_anchor = value;
    }

    __global__ void find_pressure_anchor_kernel(int* pressure_anchor, const uint8_t* occupancy, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        if (occupancy != nullptr && occupancy[index] != 0) return;
        atomicMin(pressure_anchor, static_cast<int>(index));
    }

    __global__ void compute_pressure_rhs_kernel(float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int* pressure_anchor, const int nx, const int ny, const int nz, const float h, const float dt, const boundary::FlowBoundary boundary_config) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = boundary::index_3d(x, y, z, nx, ny);
        const int anchor = *pressure_anchor;
        if (static_cast<int>(index) == anchor) {
            rhs[index] = 0.0f;
            return;
        }
        if (occupancy != nullptr && occupancy[index] != 0) {
            rhs[index] = 0.0f;
            return;
        }
        const float divergence = (velocity_x[boundary::index_velocity_x(x + 1, y, z, nx, ny)] - velocity_x[boundary::index_velocity_x(x, y, z, nx, ny)] + velocity_y[boundary::index_velocity_y(x, y + 1, z, nx, ny)] - velocity_y[boundary::index_velocity_y(x, y, z, nx, ny)] + velocity_z[boundary::index_velocity_z(x, y, z + 1, nx, ny)] - velocity_z[boundary::index_velocity_z(x, y, z, nx, ny)]) / h;
        float boundary_sum     = 0.0f;
        if (x == 0 && boundary_config.x_minus.type == boundary::flow_boundary_outflow) boundary_sum += boundary_config.x_minus.pressure;
        if (x == nx - 1 && boundary_config.x_plus.type == boundary::flow_boundary_outflow) boundary_sum += boundary_config.x_plus.pressure;
        if (y == 0 && boundary_config.y_minus.type == boundary::flow_boundary_outflow) boundary_sum += boundary_config.y_minus.pressure;
        if (y == ny - 1 && boundary_config.y_plus.type == boundary::flow_boundary_outflow) boundary_sum += boundary_config.y_plus.pressure;
        if (z == 0 && boundary_config.z_minus.type == boundary::flow_boundary_outflow) boundary_sum += boundary_config.z_minus.pressure;
        if (z == nz - 1 && boundary_config.z_plus.type == boundary::flow_boundary_outflow) boundary_sum += boundary_config.z_plus.pressure;
        rhs[index] = -(h * h / dt) * divergence + boundary_sum;
    }

    __device__ void accumulate_pressure_neighbor(int* active_neighbors, int& active_neighbor_count, float& diagonal, int next_x, int next_y, int next_z, const boundary::FlowBoundaryFace minus_face, const boundary::FlowBoundaryFace plus_face, const bool periodic_axis, const uint8_t* occupancy, const int anchor, const int nx, const int ny, const int nz) {
        if (next_x < 0 || next_x >= nx || next_y < 0 || next_y >= ny || next_z < 0 || next_z >= nz) {
            if (periodic_axis) {
                if (next_x < 0 || next_x >= nx) next_x = boundary::wrap_index(next_x, nx);
                if (next_y < 0 || next_y >= ny) next_y = boundary::wrap_index(next_y, ny);
                if (next_z < 0 || next_z >= nz) next_z = boundary::wrap_index(next_z, nz);
            } else {
                const auto face = next_x < 0 || next_y < 0 || next_z < 0 ? minus_face : plus_face;
                if (face.type == boundary::flow_boundary_outflow) diagonal += 1.0f;
                return;
            }
        }
        const int neighbor = static_cast<int>(boundary::index_3d(next_x, next_y, next_z, nx, ny));
        if (occupancy != nullptr && occupancy[static_cast<std::uint64_t>(neighbor)] != 0) return;
        diagonal += 1.0f;
        if (neighbor == anchor) return;
        for (int index = 0; index < active_neighbor_count; ++index) {
            if (active_neighbors[index] == neighbor) return;
        }
        active_neighbors[active_neighbor_count] = neighbor;
        ++active_neighbor_count;
    }

    __global__ void build_pressure_matrix_kernel(float* values, const int* row_offsets, const int* column_indices, const uint8_t* occupancy, const int* pressure_anchor, const int nx, const int ny, const int nz, const boundary::FlowBoundary boundary_config) {
        const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (row >= nx * ny * nz) return;

        const int anchor      = *pressure_anchor;
        const int x           = row % nx;
        const int yz          = row / nx;
        const int y           = yz % ny;
        const int z           = yz / ny;
        const bool occupied   = occupancy != nullptr && occupancy[static_cast<std::uint64_t>(row)] != 0;
        const bool special    = occupied || row == anchor;
        const bool periodic_x = boundary_config.x_minus.type == boundary::flow_boundary_periodic && boundary_config.x_plus.type == boundary::flow_boundary_periodic;
        const bool periodic_y = boundary_config.y_minus.type == boundary::flow_boundary_periodic && boundary_config.y_plus.type == boundary::flow_boundary_periodic;
        const bool periodic_z = boundary_config.z_minus.type == boundary::flow_boundary_periodic && boundary_config.z_plus.type == boundary::flow_boundary_periodic;

        int active_neighbors[6]{};
        int active_neighbor_count = 0;
        float diagonal            = 0.0f;

        if (!special) {
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x - 1, y, z, boundary_config.x_minus, boundary_config.x_plus, periodic_x, occupancy, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x + 1, y, z, boundary_config.x_minus, boundary_config.x_plus, periodic_x, occupancy, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y - 1, z, boundary_config.y_minus, boundary_config.y_plus, periodic_y, occupancy, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y + 1, z, boundary_config.y_minus, boundary_config.y_plus, periodic_y, occupancy, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y, z - 1, boundary_config.z_minus, boundary_config.z_plus, periodic_z, occupancy, anchor, nx, ny, nz);
            accumulate_pressure_neighbor(active_neighbors, active_neighbor_count, diagonal, x, y, z + 1, boundary_config.z_minus, boundary_config.z_plus, periodic_z, occupancy, anchor, nx, ny, nz);
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

    __global__ void project_velocity_x_kernel(float* velocity_x, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_x, const int nx, const int ny, const int nz, const float h, const float dt, const boundary::FlowBoundary boundary_config) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = velocity_x[boundary::index_velocity_x(i, j, k, nx, ny)];
        if (i == 0) {
            const auto domain_face = boundary_config.x_minus;
            if (domain_face.type != boundary::flow_boundary_periodic) {
                if (domain_face.type == boundary::flow_boundary_outflow && nx > 0)
                    face = velocity_x[boundary::index_velocity_x(1, j, k, nx, ny)];
                else
                    face = domain_face.velocity_x;
                return;
            }
        }
        if (i == nx) {
            const auto domain_face = boundary_config.x_plus;
            if (domain_face.type != boundary::flow_boundary_periodic) {
                if (domain_face.type == boundary::flow_boundary_outflow && nx > 0)
                    face = velocity_x[boundary::index_velocity_x(nx - 1, j, k, nx, ny)];
                else
                    face = domain_face.velocity_x;
                return;
            }
        }

        int left_x                = i - 1;
        int left_y                = j;
        int left_z                = k;
        int right_x               = i;
        int right_y               = j;
        int right_z               = k;
        const bool has_left       = boundary::resolve_cell_coordinates(left_x, left_y, left_z, nx, ny, nz, boundary_config);
        const bool has_right      = boundary::resolve_cell_coordinates(right_x, right_y, right_z, nx, ny, nz, boundary_config);
        const bool left_occupied  = has_left && occupancy != nullptr && occupancy[boundary::index_3d(left_x, left_y, left_z, nx, ny)] != 0;
        const bool right_occupied = has_right && occupancy != nullptr && occupancy[boundary::index_3d(right_x, right_y, right_z, nx, ny)] != 0;
        if (left_occupied || right_occupied) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (left_occupied) {
                value += boundary::solid_velocity_value(solid_velocity_x, occupancy, left_x, left_y, left_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            if (right_occupied) {
                value += boundary::solid_velocity_value(solid_velocity_x, occupancy, right_x, right_y, right_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_left && has_right) {
            const float pressure_right = pressure[boundary::index_3d(right_x, right_y, right_z, nx, ny)];
            const float pressure_left  = pressure[boundary::index_3d(left_x, left_y, left_z, nx, ny)];
            face -= dt * (pressure_right - pressure_left) / h;
        }
    }

    __global__ void project_velocity_y_kernel(float* velocity_y, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_y, const int nx, const int ny, const int nz, const float h, const float dt, const boundary::FlowBoundary boundary_config) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = velocity_y[boundary::index_velocity_y(i, j, k, nx, ny)];
        if (j == 0) {
            const auto domain_face = boundary_config.y_minus;
            if (domain_face.type != boundary::flow_boundary_periodic) {
                if (domain_face.type == boundary::flow_boundary_outflow && ny > 0)
                    face = velocity_y[boundary::index_velocity_y(i, 1, k, nx, ny)];
                else
                    face = domain_face.velocity_y;
                return;
            }
        }
        if (j == ny) {
            const auto domain_face = boundary_config.y_plus;
            if (domain_face.type != boundary::flow_boundary_periodic) {
                if (domain_face.type == boundary::flow_boundary_outflow && ny > 0)
                    face = velocity_y[boundary::index_velocity_y(i, ny - 1, k, nx, ny)];
                else
                    face = domain_face.velocity_y;
                return;
            }
        }

        int down_x               = i;
        int down_y               = j - 1;
        int down_z               = k;
        int up_x                 = i;
        int up_y                 = j;
        int up_z                 = k;
        const bool has_down      = boundary::resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary_config);
        const bool has_up        = boundary::resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary_config);
        const bool down_occupied = has_down && occupancy != nullptr && occupancy[boundary::index_3d(down_x, down_y, down_z, nx, ny)] != 0;
        const bool up_occupied   = has_up && occupancy != nullptr && occupancy[boundary::index_3d(up_x, up_y, up_z, nx, ny)] != 0;
        if (down_occupied || up_occupied) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (down_occupied) {
                value += boundary::solid_velocity_value(solid_velocity_y, occupancy, down_x, down_y, down_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            if (up_occupied) {
                value += boundary::solid_velocity_value(solid_velocity_y, occupancy, up_x, up_y, up_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_down && has_up) {
            const float pressure_up   = pressure[boundary::index_3d(up_x, up_y, up_z, nx, ny)];
            const float pressure_down = pressure[boundary::index_3d(down_x, down_y, down_z, nx, ny)];
            face -= dt * (pressure_up - pressure_down) / h;
        }
    }

    __global__ void project_velocity_z_kernel(float* velocity_z, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const boundary::FlowBoundary boundary_config) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = velocity_z[boundary::index_velocity_z(i, j, k, nx, ny)];
        if (k == 0) {
            const auto domain_face = boundary_config.z_minus;
            if (domain_face.type != boundary::flow_boundary_periodic) {
                if (domain_face.type == boundary::flow_boundary_outflow && nz > 0)
                    face = velocity_z[boundary::index_velocity_z(i, j, 1, nx, ny)];
                else
                    face = domain_face.velocity_z;
                return;
            }
        }
        if (k == nz) {
            const auto domain_face = boundary_config.z_plus;
            if (domain_face.type != boundary::flow_boundary_periodic) {
                if (domain_face.type == boundary::flow_boundary_outflow && nz > 0)
                    face = velocity_z[boundary::index_velocity_z(i, j, nz - 1, nx, ny)];
                else
                    face = domain_face.velocity_z;
                return;
            }
        }

        int back_x                = i;
        int back_y                = j;
        int back_z                = k - 1;
        int front_x               = i;
        int front_y               = j;
        int front_z               = k;
        const bool has_back       = boundary::resolve_cell_coordinates(back_x, back_y, back_z, nx, ny, nz, boundary_config);
        const bool has_front      = boundary::resolve_cell_coordinates(front_x, front_y, front_z, nx, ny, nz, boundary_config);
        const bool back_occupied  = has_back && occupancy != nullptr && occupancy[boundary::index_3d(back_x, back_y, back_z, nx, ny)] != 0;
        const bool front_occupied = has_front && occupancy != nullptr && occupancy[boundary::index_3d(front_x, front_y, front_z, nx, ny)] != 0;
        if (back_occupied || front_occupied) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (back_occupied) {
                value += boundary::solid_velocity_value(solid_velocity_z, occupancy, back_x, back_y, back_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            if (front_occupied) {
                value += boundary::solid_velocity_value(solid_velocity_z, occupancy, front_x, front_y, front_z, nx, ny, nz, boundary_config);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_back && has_front) {
            const float pressure_front = pressure[boundary::index_3d(front_x, front_y, front_z, nx, ny)];
            const float pressure_back  = pressure[boundary::index_3d(back_x, back_y, back_z, nx, ny)];
            face -= dt * (pressure_front - pressure_back) / h;
        }
    }

    void reset_pressure_anchor(cudaStream_t stream, int* pressure_anchor, const int value) {
        reset_pressure_anchor_kernel<<<1, 1, 0, stream>>>(pressure_anchor, value);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"reset_pressure_anchor_kernel: "} + cudaGetErrorString(status)};
    }

    void find_pressure_anchor(cudaStream_t stream, int* pressure_anchor, const std::uint8_t* occupancy, const std::uint64_t count) {
        const unsigned block = linear_block();
        const unsigned grid  = linear_launch_grid(count, block);
        find_pressure_anchor_kernel<<<grid, block, 0, stream>>>(pressure_anchor, occupancy, count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"find_pressure_anchor_kernel: "} + cudaGetErrorString(status)};
    }

    void compute_pressure_rhs(cudaStream_t stream, float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint8_t* occupancy, const int* pressure_anchor, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t* flow_types, const float* flow_pressure) {
        const dim3 block = cell_block();
        const dim3 grid  = scalar_launch_grid(nx, ny, nz, block);
        const boundary::FlowBoundary boundary_config = boundary::make_flow_pressure_boundary(flow_types, flow_pressure);
        compute_pressure_rhs_kernel<<<grid, block, 0, stream>>>(rhs, velocity_x, velocity_y, velocity_z, occupancy, pressure_anchor, nx, ny, nz, h, dt, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_pressure_rhs_kernel: "} + cudaGetErrorString(status)};
    }

    void build_pressure_matrix(cudaStream_t stream, float* values, const int* row_offsets, const int* column_indices, const std::uint8_t* occupancy, const int* pressure_anchor, const int nx, const int ny, const int nz, const std::uint32_t* flow_types) {
        require_resolution(nx, ny, nz);
        const auto count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        const unsigned block = linear_block();
        const unsigned grid  = linear_launch_grid(count, block);
        const boundary::FlowBoundary boundary_config = boundary::make_flow_type_boundary(flow_types);
        build_pressure_matrix_kernel<<<grid, block, 0, stream>>>(values, row_offsets, column_indices, occupancy, pressure_anchor, nx, ny, nz, boundary_config);
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

    void project_staggered_component(cudaStream_t stream, const std::uint32_t axis, float* velocity_component, const float* pressure, const std::uint8_t* occupancy, const float* solid_velocity_component, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t* flow_types, const float* flow_velocity) {
        if (axis >= 3u) throw std::runtime_error{"project_staggered_component: axis must be 0, 1, or 2"};
        const dim3 block = cell_block();
        const dim3 grid  = staggered_launch_grid(axis, nx, ny, nz, block);
        const boundary::FlowBoundary boundary_config = boundary::make_flow_velocity_boundary(flow_types, flow_velocity);
        if (axis == 0u) project_velocity_x_kernel<<<grid, block, 0, stream>>>(velocity_component, pressure, occupancy, solid_velocity_component, nx, ny, nz, h, dt, boundary_config);
        if (axis == 1u) project_velocity_y_kernel<<<grid, block, 0, stream>>>(velocity_component, pressure, occupancy, solid_velocity_component, nx, ny, nz, h, dt, boundary_config);
        if (axis == 2u) project_velocity_z_kernel<<<grid, block, 0, stream>>>(velocity_component, pressure, occupancy, solid_velocity_component, nx, ny, nz, h, dt, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"project_staggered_component_kernel: "} + cudaGetErrorString(status)};
    }

} // namespace kfs::cuda::operators::projection
