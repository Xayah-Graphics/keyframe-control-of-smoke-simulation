#include "keyframe.boundary.h"
#include "keyframe.boundary.cuh"
#include <stdexcept>
#include <string>

namespace kfs::cuda::boundary {
    __global__ void enforce_x_kernel(float* component, const std::uint32_t* cell_indices, const float* constraint_component, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face_value = component[field::index(0u, i, j, k, nx, ny)];
        if (i == 0) {
            if (const VectorBoundaryFace3D face = boundary.x_min; face.mode != vector_boundary_periodic) {
                if (face.mode == vector_boundary_zero_gradient && nx > 0)
                    face_value = component[field::index(0u, 1, j, k, nx, ny)];
                else
                    face_value = vector_face_value(face, 0u);
                return;
            }
        }
        if (i == nx) {
            if (const VectorBoundaryFace3D face = boundary.x_max; face.mode != vector_boundary_periodic) {
                if (face.mode == vector_boundary_zero_gradient && nx > 0)
                    face_value = component[field::index(0u, nx - 1, j, k, nx, ny)];
                else
                    face_value = vector_face_value(face, 0u);
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
        const bool left_marked    = has_left && cell_indices[field::index(left_x, left_y, left_z, nx, ny)] != 0u;
        const bool right_marked   = has_right && cell_indices[field::index(right_x, right_y, right_z, nx, ny)] != 0u;
        if (!left_marked && !right_marked) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (left_marked) {
            value += constraint_value(constraint_component, cell_indices, left_x, left_y, left_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (right_marked) {
            value += constraint_value(constraint_component, cell_indices, right_x, right_y, right_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face_value = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void enforce_y_kernel(float* component, const std::uint32_t* cell_indices, const float* constraint_component, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face_value = component[field::index(1u, i, j, k, nx, ny)];
        if (j == 0) {
            if (const VectorBoundaryFace3D face = boundary.y_min; face.mode != vector_boundary_periodic) {
                if (face.mode == vector_boundary_zero_gradient && ny > 0)
                    face_value = component[field::index(1u, i, 1, k, nx, ny)];
                else
                    face_value = vector_face_value(face, 1u);
                return;
            }
        }
        if (j == ny) {
            if (const VectorBoundaryFace3D face = boundary.y_max; face.mode != vector_boundary_periodic) {
                if (face.mode == vector_boundary_zero_gradient && ny > 0)
                    face_value = component[field::index(1u, i, ny - 1, k, nx, ny)];
                else
                    face_value = vector_face_value(face, 1u);
                return;
            }
        }
        int min_x              = i;
        int min_y              = j - 1;
        int min_z              = k;
        int max_x              = i;
        int max_y              = j;
        int max_z              = k;
        const bool has_min     = resolve_cell_coordinates(min_x, min_y, min_z, nx, ny, nz, boundary);
        const bool has_max     = resolve_cell_coordinates(max_x, max_y, max_z, nx, ny, nz, boundary);
        const bool min_marked  = has_min && cell_indices[field::index(min_x, min_y, min_z, nx, ny)] != 0u;
        const bool max_marked  = has_max && cell_indices[field::index(max_x, max_y, max_z, nx, ny)] != 0u;
        if (!min_marked && !max_marked) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (min_marked) {
            value += constraint_value(constraint_component, cell_indices, min_x, min_y, min_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (max_marked) {
            value += constraint_value(constraint_component, cell_indices, max_x, max_y, max_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face_value = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void enforce_z_kernel(float* component, const std::uint32_t* cell_indices, const float* constraint_component, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face_value = component[field::index(2u, i, j, k, nx, ny)];
        if (k == 0) {
            if (const VectorBoundaryFace3D face = boundary.z_min; face.mode != vector_boundary_periodic) {
                if (face.mode == vector_boundary_zero_gradient && nz > 0)
                    face_value = component[field::index(2u, i, j, 1, nx, ny)];
                else
                    face_value = vector_face_value(face, 2u);
                return;
            }
        }
        if (k == nz) {
            if (const VectorBoundaryFace3D face = boundary.z_max; face.mode != vector_boundary_periodic) {
                if (face.mode == vector_boundary_zero_gradient && nz > 0)
                    face_value = component[field::index(2u, i, j, nz - 1, nx, ny)];
                else
                    face_value = vector_face_value(face, 2u);
                return;
            }
        }
        int min_x              = i;
        int min_y              = j;
        int min_z              = k - 1;
        int max_x              = i;
        int max_y              = j;
        int max_z              = k;
        const bool has_min     = resolve_cell_coordinates(min_x, min_y, min_z, nx, ny, nz, boundary);
        const bool has_max     = resolve_cell_coordinates(max_x, max_y, max_z, nx, ny, nz, boundary);
        const bool min_marked  = has_min && cell_indices[field::index(min_x, min_y, min_z, nx, ny)] != 0u;
        const bool max_marked  = has_max && cell_indices[field::index(max_x, max_y, max_z, nx, ny)] != 0u;
        if (!min_marked && !max_marked) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (min_marked) {
            value += constraint_value(constraint_component, cell_indices, min_x, min_y, min_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (max_marked) {
            value += constraint_value(constraint_component, cell_indices, max_x, max_y, max_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face_value = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void synchronize_x_kernel(float* component, const int nx, const int ny, const int nz) {
        const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (j >= ny || k >= nz) return;
        component[field::index(0u, nx, j, k, nx, ny)] = component[field::index(0u, 0, j, k, nx, ny)];
    }

    __global__ void synchronize_y_kernel(float* component, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || k >= nz) return;
        component[field::index(1u, i, ny, k, nx, ny)] = component[field::index(1u, i, 0, k, nx, ny)];
    }

    __global__ void synchronize_z_kernel(float* component, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || j >= ny) return;
        component[field::index(2u, i, j, nz, nx, ny)] = component[field::index(2u, i, j, 0, nx, ny)];
    }

    __global__ void extrapolate_kernel(float* destination, const float* source, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const ScalarBoundary3D boundary) {
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
                        if (!resolve_cell_coordinates(next_x, next_y, next_z, nx, ny, nz, boundary)) continue;
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

    void enforce(cudaStream_t stream, const std::uint32_t axis, float* component, const std::uint32_t* cell_indices, const float* constraint_component, const int nx, const int ny, const int nz, const std::uint32_t* boundary_modes, const float* boundary_values) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid               = field::staggered_grid(axis, nx, ny, nz, block);
        const VectorBoundary3D boundary = make_vector_boundary(boundary_modes, boundary_values);
        if (axis == 0u) enforce_x_kernel<<<grid, block, 0, stream>>>(component, cell_indices, constraint_component, nx, ny, nz, boundary);
        if (axis == 1u) enforce_y_kernel<<<grid, block, 0, stream>>>(component, cell_indices, constraint_component, nx, ny, nz, boundary);
        if (axis == 2u) enforce_z_kernel<<<grid, block, 0, stream>>>(component, cell_indices, constraint_component, nx, ny, nz, boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"boundary enforce kernel: "} + cudaGetErrorString(status)};
    }

    void synchronize(cudaStream_t stream, const std::uint32_t axis, float* component, const int nx, const int ny, const int nz) {
        constexpr dim3 block{8u, 8u, 1u};
        const dim3 grid = axis == 0u ? dim3{field::ceil_div_u32(static_cast<std::uint64_t>(ny), block.x), field::ceil_div_u32(static_cast<std::uint64_t>(nz), block.y), 1u} : axis == 1u ? dim3{field::ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), field::ceil_div_u32(static_cast<std::uint64_t>(nz), block.y), 1u} : dim3{field::ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), field::ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), 1u};
        if (axis == 0u) synchronize_x_kernel<<<grid, block, 0, stream>>>(component, nx, ny, nz);
        if (axis == 1u) synchronize_y_kernel<<<grid, block, 0, stream>>>(component, nx, ny, nz);
        if (axis == 2u) synchronize_z_kernel<<<grid, block, 0, stream>>>(component, nx, ny, nz);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"boundary synchronize kernel: "} + cudaGetErrorString(status)};
    }

    void extrapolate(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const std::uint32_t* boundary_modes, const float* boundary_values) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid                 = field::centered_grid(nx, ny, nz, block);
        const ScalarBoundary3D boundary = make_scalar_boundary(boundary_modes, boundary_values);
        extrapolate_kernel<<<grid, block, 0, stream>>>(destination, source, cell_indices, nx, ny, nz, boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"boundary extrapolate kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::boundary
