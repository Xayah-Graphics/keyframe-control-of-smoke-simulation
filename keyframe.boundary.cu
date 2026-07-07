#include "keyframe.boundary.h"
#include "keyframe.boundary.cuh"
#include <stdexcept>
#include <string>

namespace kfs::cuda::boundary {
    __global__ void enforce_velocity_x_boundaries_kernel(float* velocity_x, const std::uint32_t* cell_indices, const float* constraint_velocity_x, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = velocity_x[field::index(0u, i, j, k, nx, ny)];
        if (i == 0) {
            if (const auto domain_face = boundary.x_minus; domain_face.type != flow_boundary_periodic) {
                if (domain_face.type == flow_boundary_outflow && nx > 0)
                    face = velocity_x[field::index(0u, 1, j, k, nx, ny)];
                else
                    face = domain_face.velocity_x;
                return;
            }
        }
        if (i == nx) {
            if (const auto domain_face = boundary.x_plus; domain_face.type != flow_boundary_periodic) {
                if (domain_face.type == flow_boundary_outflow && nx > 0)
                    face = velocity_x[field::index(0u, nx - 1, j, k, nx, ny)];
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
        const bool has_left       = resolve_cell_coordinates(left_x, left_y, left_z, nx, ny, nz, boundary);
        const bool has_right      = resolve_cell_coordinates(right_x, right_y, right_z, nx, ny, nz, boundary);
        const bool left_occupied  = has_left && cell_indices[field::index(left_x, left_y, left_z, nx, ny)] != 0u;
        const bool right_occupied = has_right && cell_indices[field::index(right_x, right_y, right_z, nx, ny)] != 0u;
        if (!left_occupied && !right_occupied) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (left_occupied) {
            value += constraint_velocity_value(constraint_velocity_x, cell_indices, left_x, left_y, left_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (right_occupied) {
            value += constraint_velocity_value(constraint_velocity_x, cell_indices, right_x, right_y, right_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void enforce_velocity_y_boundaries_kernel(float* velocity_y, const std::uint32_t* cell_indices, const float* constraint_velocity_y, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = velocity_y[field::index(1u, i, j, k, nx, ny)];
        if (j == 0) {
            if (const auto domain_face = boundary.y_minus; domain_face.type != flow_boundary_periodic) {
                if (domain_face.type == flow_boundary_outflow && ny > 0)
                    face = velocity_y[field::index(1u, i, 1, k, nx, ny)];
                else
                    face = domain_face.velocity_y;
                return;
            }
        }
        if (j == ny) {
            if (const auto domain_face = boundary.y_plus; domain_face.type != flow_boundary_periodic) {
                if (domain_face.type == flow_boundary_outflow && ny > 0)
                    face = velocity_y[field::index(1u, i, ny - 1, k, nx, ny)];
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
        const bool has_down      = resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary);
        const bool has_up        = resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary);
        const bool down_occupied = has_down && cell_indices[field::index(down_x, down_y, down_z, nx, ny)] != 0u;
        const bool up_occupied   = has_up && cell_indices[field::index(up_x, up_y, up_z, nx, ny)] != 0u;
        if (!down_occupied && !up_occupied) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (down_occupied) {
            value += constraint_velocity_value(constraint_velocity_y, cell_indices, down_x, down_y, down_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (up_occupied) {
            value += constraint_velocity_value(constraint_velocity_y, cell_indices, up_x, up_y, up_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void enforce_velocity_z_boundaries_kernel(float* velocity_z, const std::uint32_t* cell_indices, const float* constraint_velocity_z, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = velocity_z[field::index(2u, i, j, k, nx, ny)];
        if (k == 0) {
            if (const auto domain_face = boundary.z_minus; domain_face.type != flow_boundary_periodic) {
                if (domain_face.type == flow_boundary_outflow && nz > 0)
                    face = velocity_z[field::index(2u, i, j, 1, nx, ny)];
                else
                    face = domain_face.velocity_z;
                return;
            }
        }
        if (k == nz) {
            if (const auto domain_face = boundary.z_plus; domain_face.type != flow_boundary_periodic) {
                if (domain_face.type == flow_boundary_outflow && nz > 0)
                    face = velocity_z[field::index(2u, i, j, nz - 1, nx, ny)];
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
        const bool has_back       = resolve_cell_coordinates(back_x, back_y, back_z, nx, ny, nz, boundary);
        const bool has_front      = resolve_cell_coordinates(front_x, front_y, front_z, nx, ny, nz, boundary);
        const bool back_occupied  = has_back && cell_indices[field::index(back_x, back_y, back_z, nx, ny)] != 0u;
        const bool front_occupied = has_front && cell_indices[field::index(front_x, front_y, front_z, nx, ny)] != 0u;
        if (!back_occupied && !front_occupied) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (back_occupied) {
            value += constraint_velocity_value(constraint_velocity_z, cell_indices, back_x, back_y, back_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (front_occupied) {
            value += constraint_velocity_value(constraint_velocity_z, cell_indices, front_x, front_y, front_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void sync_periodic_velocity_x_kernel(float* velocity_x, const int nx, const int ny, const int nz) {
        const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (j >= ny || k >= nz) return;
        velocity_x[field::index(0u, nx, j, k, nx, ny)] = velocity_x[field::index(0u, 0, j, k, nx, ny)];
    }

    __global__ void sync_periodic_velocity_y_kernel(float* velocity_y, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || k >= nz) return;
        velocity_y[field::index(1u, i, ny, k, nx, ny)] = velocity_y[field::index(1u, i, 0, k, nx, ny)];
    }

    __global__ void sync_periodic_velocity_z_kernel(float* velocity_z, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || j >= ny) return;
        velocity_z[field::index(2u, i, j, nz, nx, ny)] = velocity_z[field::index(2u, i, j, 0, nx, ny)];
    }

    __global__ void boundary_fill_centered_scalar_kernel(float* destination, const float* source, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const ScalarBoundary boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = field::index(x, y, z, nx, ny);
        if (cell_indices[index] == 0u) {
            destination[index] = source[index];
            return;
        }

        int max_radius = nx;
        if (ny > max_radius) max_radius = ny;
        if (nz > max_radius) max_radius = nz;
        for (int radius = 1; radius <= max_radius; ++radius) {
            bool found         = false;
            float best_value   = 0.0f;
            int best_distance2 = 0;
            for (int dz = -radius; dz <= radius; ++dz) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int shell_radius = abs(dx);
                        if (abs(dy) > shell_radius) shell_radius = abs(dy);
                        if (abs(dz) > shell_radius) shell_radius = abs(dz);
                        if (shell_radius != radius) continue;
                        int next_x = x + dx;
                        int next_y = y + dy;
                        int next_z = z + dz;
                        if (!resolve_scalar_cell_coordinates(next_x, next_y, next_z, nx, ny, nz, boundary)) continue;
                        const auto neighbor_index = field::index(next_x, next_y, next_z, nx, ny);
                        if (cell_indices[neighbor_index] != 0u) continue;
                        const int distance2 = dx * dx + dy * dy + dz * dz;
                        if (!found || distance2 < best_distance2) {
                            found          = true;
                            best_distance2 = distance2;
                            best_value     = source[neighbor_index];
                        }
                    }
                }
            }
            if (found) {
                destination[index] = best_value;
                return;
            }
        }
        destination[index] = 0.0f;
    }

    void enforce_staggered_boundary(cudaStream_t stream, const std::uint32_t axis, float* velocity_component, const std::uint32_t* cell_indices, const float* constraint_velocity_component, const int nx, const int ny, const int nz, const std::uint32_t* flow_types, const float* flow_velocity) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid             = field::staggered_grid(axis, nx, ny, nz, block);
        const FlowBoundary boundary = make_flow_velocity_boundary(flow_types, flow_velocity);
        if (axis == 0u) enforce_velocity_x_boundaries_kernel<<<grid, block, 0, stream>>>(velocity_component, cell_indices, constraint_velocity_component, nx, ny, nz, boundary);
        if (axis == 1u) enforce_velocity_y_boundaries_kernel<<<grid, block, 0, stream>>>(velocity_component, cell_indices, constraint_velocity_component, nx, ny, nz, boundary);
        if (axis == 2u) enforce_velocity_z_boundaries_kernel<<<grid, block, 0, stream>>>(velocity_component, cell_indices, constraint_velocity_component, nx, ny, nz, boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"enforce_staggered_boundary_kernel: "} + cudaGetErrorString(status)};
    }

    void sync_periodic_staggered_component(cudaStream_t stream, const std::uint32_t axis, float* velocity_component, const int nx, const int ny, const int nz) {
        constexpr dim3 block{8u, 8u, 1u};
        const dim3 grid = axis == 0u ? dim3{field::ceil_div_u32(static_cast<std::uint64_t>(ny), block.x), field::ceil_div_u32(static_cast<std::uint64_t>(nz), block.y), 1u} : axis == 1u ? dim3{field::ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), field::ceil_div_u32(static_cast<std::uint64_t>(nz), block.y), 1u} : dim3{field::ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), field::ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), 1u};
        if (axis == 0u) sync_periodic_velocity_x_kernel<<<grid, block, 0, stream>>>(velocity_component, nx, ny, nz);
        if (axis == 1u) sync_periodic_velocity_y_kernel<<<grid, block, 0, stream>>>(velocity_component, nx, ny, nz);
        if (axis == 2u) sync_periodic_velocity_z_kernel<<<grid, block, 0, stream>>>(velocity_component, nx, ny, nz);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"sync_periodic_staggered_component_kernel: "} + cudaGetErrorString(status)};
    }

    void boundary_fill_centered_scalar(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const std::uint32_t* scalar_boundary_types, const float* scalar_boundary_values) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid = field::centered_grid(nx, ny, nz, block);
        const ScalarBoundary boundary = make_scalar_boundary(scalar_boundary_types, scalar_boundary_values);
        boundary_fill_centered_scalar_kernel<<<grid, block, 0, stream>>>(destination, source, cell_indices, nx, ny, nz, boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"boundary_fill_centered_scalar_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::boundary
