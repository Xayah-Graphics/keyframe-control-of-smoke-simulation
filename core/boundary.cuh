#ifndef XAYAH_CORE_BOUNDARY_CUH
#define XAYAH_CORE_BOUNDARY_CUH

#include "field.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace xayah::core::boundary::cuda {
    constexpr std::uint32_t scalar_boundary_fixed_value   = 0u;
    constexpr std::uint32_t scalar_boundary_zero_gradient = 1u;
    constexpr std::uint32_t scalar_boundary_periodic      = 2u;

    constexpr std::uint32_t vector_boundary_fixed_value                        = 0u;
    constexpr std::uint32_t vector_boundary_zero_gradient                      = 1u;
    constexpr std::uint32_t vector_boundary_normal_fixed_tangent_zero_gradient = 2u;
    constexpr std::uint32_t vector_boundary_periodic                           = 3u;

    struct ScalarBoundaryFace3D final {
        std::uint32_t mode{scalar_boundary_zero_gradient};
        float value{0.0f};
    };

    struct VectorBoundaryFace3D final {
        std::uint32_t mode{vector_boundary_zero_gradient};
        float value_x{0.0f};
        float value_y{0.0f};
        float value_z{0.0f};
    };

    struct ScalarBoundary3D final {
        ScalarBoundaryFace3D x_min{};
        ScalarBoundaryFace3D x_max{};
        ScalarBoundaryFace3D y_min{};
        ScalarBoundaryFace3D y_max{};
        ScalarBoundaryFace3D z_min{};
        ScalarBoundaryFace3D z_max{};
    };

    struct VectorBoundary3D final {
        VectorBoundaryFace3D x_min{};
        VectorBoundaryFace3D x_max{};
        VectorBoundaryFace3D y_min{};
        VectorBoundaryFace3D y_max{};
        VectorBoundaryFace3D z_min{};
        VectorBoundaryFace3D z_max{};
    };

    inline ScalarBoundary3D make_scalar_boundary(const std::uint32_t* modes, const float* values) {
        return ScalarBoundary3D{
            {modes[0], values[0]},
            {modes[1], values[1]},
            {modes[2], values[2]},
            {modes[3], values[3]},
            {modes[4], values[4]},
            {modes[5], values[5]},
        };
    }

    inline VectorBoundary3D make_vector_boundary(const std::uint32_t* modes, const float* values) {
        return VectorBoundary3D{
            {modes[0], values[0], values[1], values[2]},
            {modes[1], values[3], values[4], values[5]},
            {modes[2], values[6], values[7], values[8]},
            {modes[3], values[9], values[10], values[11]},
            {modes[4], values[12], values[13], values[14]},
            {modes[5], values[15], values[16], values[17]},
        };
    }

    __device__ inline int wrap_index(int value, const int size) {
        if (size <= 0) return 0;
        value %= size;
        if (value < 0) value += size;
        return value;
    }

    __device__ inline int clamp_int(const int value, const int low, const int high) {
        return value < low ? low : (value > high ? high : value);
    }

    __device__ inline bool scalar_periodic_pair(const ScalarBoundary3D boundary, const std::uint32_t dimension) {
        if (dimension == 0u) return boundary.x_min.mode == scalar_boundary_periodic && boundary.x_max.mode == scalar_boundary_periodic;
        if (dimension == 1u) return boundary.y_min.mode == scalar_boundary_periodic && boundary.y_max.mode == scalar_boundary_periodic;
        return boundary.z_min.mode == scalar_boundary_periodic && boundary.z_max.mode == scalar_boundary_periodic;
    }

    __device__ inline bool vector_periodic_pair(const VectorBoundary3D boundary, const std::uint32_t dimension) {
        if (dimension == 0u) return boundary.x_min.mode == vector_boundary_periodic && boundary.x_max.mode == vector_boundary_periodic;
        if (dimension == 1u) return boundary.y_min.mode == vector_boundary_periodic && boundary.y_max.mode == vector_boundary_periodic;
        return boundary.z_min.mode == vector_boundary_periodic && boundary.z_max.mode == vector_boundary_periodic;
    }

    __device__ inline float wrap_periodic_position(float value, const int cells, const float h, const bool periodic) {
        if (!periodic || cells <= 0) return value;
        const float extent = static_cast<float>(cells) * h;
        value              = fmodf(value, extent);
        if (value < 0.0f) value += extent;
        return value;
    }

    __device__ inline float3 wrap_scalar_position(float3 position, const int nx, const int ny, const int nz, const float h, const ScalarBoundary3D boundary) {
        position.x = wrap_periodic_position(position.x, nx, h, scalar_periodic_pair(boundary, 0u));
        position.y = wrap_periodic_position(position.y, ny, h, scalar_periodic_pair(boundary, 1u));
        position.z = wrap_periodic_position(position.z, nz, h, scalar_periodic_pair(boundary, 2u));
        return position;
    }

    __device__ inline float3 wrap_vector_position(float3 position, const int nx, const int ny, const int nz, const float h, const VectorBoundary3D boundary) {
        position.x = wrap_periodic_position(position.x, nx, h, vector_periodic_pair(boundary, 0u));
        position.y = wrap_periodic_position(position.y, ny, h, vector_periodic_pair(boundary, 1u));
        position.z = wrap_periodic_position(position.z, nz, h, vector_periodic_pair(boundary, 2u));
        return position;
    }

    __device__ inline bool resolve_cell_coordinates(int& x, int& y, int& z, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        if (vector_periodic_pair(boundary, 0u) && nx > 0) x = wrap_index(x, nx);
        if (vector_periodic_pair(boundary, 1u) && ny > 0) y = wrap_index(y, ny);
        if (vector_periodic_pair(boundary, 2u) && nz > 0) z = wrap_index(z, nz);
        return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
    }

    __device__ inline bool resolve_cell_coordinates(int& x, int& y, int& z, const int nx, const int ny, const int nz, const ScalarBoundary3D boundary) {
        if (scalar_periodic_pair(boundary, 0u) && nx > 0) x = wrap_index(x, nx);
        if (scalar_periodic_pair(boundary, 1u) && ny > 0) y = wrap_index(y, ny);
        if (scalar_periodic_pair(boundary, 2u) && nz > 0) z = wrap_index(z, nz);
        return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
    }

    __device__ inline bool cell_is_marked(const std::uint32_t* cell_indices, int x, int y, int z, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return true;
        return cell_indices[field::cuda::index(x, y, z, nx, ny)] != 0u;
    }

    __device__ inline float load_scalar(const float* values, int x, int y, int z, const int nx, const int ny, const int nz, const ScalarBoundary3D boundary) {
        if (x < 0 || x >= nx) {
            const ScalarBoundaryFace3D face = x < 0 ? boundary.x_min : boundary.x_max;
            if (scalar_periodic_pair(boundary, 0u) && nx > 0)
                x = wrap_index(x, nx);
            else if (face.mode == scalar_boundary_zero_gradient && nx > 0)
                x = x < 0 ? 0 : nx - 1;
            else
                return face.value;
        }
        if (y < 0 || y >= ny) {
            const ScalarBoundaryFace3D face = y < 0 ? boundary.y_min : boundary.y_max;
            if (scalar_periodic_pair(boundary, 1u) && ny > 0)
                y = wrap_index(y, ny);
            else if (face.mode == scalar_boundary_zero_gradient && ny > 0)
                y = y < 0 ? 0 : ny - 1;
            else
                return face.value;
        }
        if (z < 0 || z >= nz) {
            const ScalarBoundaryFace3D face = z < 0 ? boundary.z_min : boundary.z_max;
            if (scalar_periodic_pair(boundary, 2u) && nz > 0)
                z = wrap_index(z, nz);
            else if (face.mode == scalar_boundary_zero_gradient && nz > 0)
                z = z < 0 ? 0 : nz - 1;
            else
                return face.value;
        }
        return values[field::cuda::index(x, y, z, nx, ny)];
    }

    __device__ inline float load_staggered_component_boundary(const float* values, const std::uint32_t axis, const std::uint32_t boundary_dimension, const bool lower, int i, int j, int k, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        VectorBoundaryFace3D face = lower ? boundary.z_min : boundary.z_max;
        if (boundary_dimension == 0u) face = lower ? boundary.x_min : boundary.x_max;
        if (boundary_dimension == 1u) face = lower ? boundary.y_min : boundary.y_max;
        i = clamp_int(i, 0, field::cuda::extent(axis, 0u, nx, ny, nz) - 1);
        j = clamp_int(j, 0, field::cuda::extent(axis, 1u, nx, ny, nz) - 1);
        k = clamp_int(k, 0, field::cuda::extent(axis, 2u, nx, ny, nz) - 1);
        if (boundary_dimension == 0u) i = lower ? 0 : field::cuda::extent(axis, 0u, nx, ny, nz) - 1;
        if (boundary_dimension == 1u) j = lower ? 0 : field::cuda::extent(axis, 1u, nx, ny, nz) - 1;
        if (boundary_dimension == 2u) k = lower ? 0 : field::cuda::extent(axis, 2u, nx, ny, nz) - 1;
        const float interior = values[field::cuda::index(axis, i, j, k, nx, ny)];
        if (face.mode == vector_boundary_zero_gradient || (face.mode == vector_boundary_normal_fixed_tangent_zero_gradient && axis != boundary_dimension)) return interior;
        const float boundary_value = axis == 0u ? face.value_x : axis == 1u ? face.value_y : face.value_z;
        return 2.0f * boundary_value - interior;
    }

    __device__ inline float load_staggered_component(const float* values, const std::uint32_t axis, int i, int j, int k, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        if (i < 0 || i >= field::cuda::extent(axis, 0u, nx, ny, nz)) {
            if (vector_periodic_pair(boundary, 0u) && nx > 0)
                i = wrap_index(i, nx);
            else
                return load_staggered_component_boundary(values, axis, 0u, i < 0, i, j, k, nx, ny, nz, boundary);
        }
        if (j < 0 || j >= field::cuda::extent(axis, 1u, nx, ny, nz)) {
            if (vector_periodic_pair(boundary, 1u) && ny > 0)
                j = wrap_index(j, ny);
            else
                return load_staggered_component_boundary(values, axis, 1u, j < 0, i, j, k, nx, ny, nz, boundary);
        }
        if (k < 0 || k >= field::cuda::extent(axis, 2u, nx, ny, nz)) {
            if (vector_periodic_pair(boundary, 2u) && nz > 0)
                k = wrap_index(k, nz);
            else
                return load_staggered_component_boundary(values, axis, 2u, k < 0, i, j, k, nx, ny, nz, boundary);
        }
        return values[field::cuda::index(axis, i, j, k, nx, ny)];
    }

    __device__ inline float load_centered_scalar(const float* values, int x, int y, int z, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        if (x < 0 || x >= nx) {
            if (vector_periodic_pair(boundary, 0u) && nx > 0)
                x = wrap_index(x, nx);
            else
                x = x < 0 ? 0 : nx - 1;
        }
        if (y < 0 || y >= ny) {
            if (vector_periodic_pair(boundary, 1u) && ny > 0)
                y = wrap_index(y, ny);
            else
                y = y < 0 ? 0 : ny - 1;
        }
        if (z < 0 || z >= nz) {
            if (vector_periodic_pair(boundary, 2u) && nz > 0)
                z = wrap_index(z, nz);
            else
                z = z < 0 ? 0 : nz - 1;
        }
        return values[field::cuda::index(x, y, z, nx, ny)];
    }

    __device__ inline float load_centered_component(const float* values, const std::uint32_t axis, int x, int y, int z, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        if (x < 0 || x >= nx) {
            const VectorBoundaryFace3D face = x < 0 ? boundary.x_min : boundary.x_max;
            if (vector_periodic_pair(boundary, 0u) && nx > 0) {
                x = wrap_index(x, nx);
            } else {
                const float interior = values[field::cuda::index(x < 0 ? 0 : nx - 1, clamp_int(y, 0, ny - 1), clamp_int(z, 0, nz - 1), nx, ny)];
                if (face.mode == vector_boundary_zero_gradient || (face.mode == vector_boundary_normal_fixed_tangent_zero_gradient && axis != 0u)) return interior;
                const float boundary_value = axis == 0u ? face.value_x : axis == 1u ? face.value_y : face.value_z;
                return 2.0f * boundary_value - interior;
            }
        }
        if (y < 0 || y >= ny) {
            const VectorBoundaryFace3D face = y < 0 ? boundary.y_min : boundary.y_max;
            if (vector_periodic_pair(boundary, 1u) && ny > 0) {
                y = wrap_index(y, ny);
            } else {
                const float interior = values[field::cuda::index(clamp_int(x, 0, nx - 1), y < 0 ? 0 : ny - 1, clamp_int(z, 0, nz - 1), nx, ny)];
                if (face.mode == vector_boundary_zero_gradient || (face.mode == vector_boundary_normal_fixed_tangent_zero_gradient && axis != 1u)) return interior;
                const float boundary_value = axis == 0u ? face.value_x : axis == 1u ? face.value_y : face.value_z;
                return 2.0f * boundary_value - interior;
            }
        }
        if (z < 0 || z >= nz) {
            const VectorBoundaryFace3D face = z < 0 ? boundary.z_min : boundary.z_max;
            if (vector_periodic_pair(boundary, 2u) && nz > 0) {
                z = wrap_index(z, nz);
            } else {
                const float interior = values[field::cuda::index(clamp_int(x, 0, nx - 1), clamp_int(y, 0, ny - 1), z < 0 ? 0 : nz - 1, nx, ny)];
                if (face.mode == vector_boundary_zero_gradient || (face.mode == vector_boundary_normal_fixed_tangent_zero_gradient && axis != 2u)) return interior;
                const float boundary_value = axis == 0u ? face.value_x : axis == 1u ? face.value_y : face.value_z;
                return 2.0f * boundary_value - interior;
            }
        }
        return values[field::cuda::index(x, y, z, nx, ny)];
    }

    __device__ inline float constraint_value(const float* constraint_values, const std::uint32_t* cell_indices, int x, int y, int z, const int nx, const int ny, const int nz, const VectorBoundary3D boundary) {
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return 0.0f;
        if (cell_indices[field::cuda::index(x, y, z, nx, ny)] == 0u) return 0.0f;
        return constraint_values[field::cuda::index(x, y, z, nx, ny)];
    }
} // namespace xayah::core::boundary::cuda

#endif // XAYAH_CORE_BOUNDARY_CUH
