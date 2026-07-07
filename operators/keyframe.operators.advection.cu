#include "../keyframe.boundary.cuh"
#include "../keyframe.field.cuh"
#include "keyframe.operators.advection.h"
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::advection {
    constexpr std::uint32_t scheme_monotonic_cubic = 1u;

    __device__ float wrap_periodic_position(float value, const int cells, const float h, const bool periodic) {
        if (!periodic || cells <= 0) return value;
        const float extent = static_cast<float>(cells) * h;
        value              = fmodf(value, extent);
        if (value < 0.0f) value += extent;
        return value;
    }

    __device__ float3 wrap_vector_position(float3 position, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        position.x = wrap_periodic_position(position.x, nx, h, boundary::flow_periodic_pair(boundary_config, 0u));
        position.y = wrap_periodic_position(position.y, ny, h, boundary::flow_periodic_pair(boundary_config, 1u));
        position.z = wrap_periodic_position(position.z, nz, h, boundary::flow_periodic_pair(boundary_config, 2u));
        return position;
    }

    __device__ float3 wrap_scalar_position(float3 position, const int nx, const int ny, const int nz, const float h, const boundary::ScalarBoundary boundary_config) {
        position.x = wrap_periodic_position(position.x, nx, h, boundary::scalar_periodic_pair(boundary_config, 0u));
        position.y = wrap_periodic_position(position.y, ny, h, boundary::scalar_periodic_pair(boundary_config, 1u));
        position.z = wrap_periodic_position(position.z, nz, h, boundary::scalar_periodic_pair(boundary_config, 2u));
        return position;
    }

    __device__ bool resolve_index_cell(int& x, int& y, int& z, const int nx, const int ny, const int nz, const boundary::FlowBoundary boundary_config) {
        if (boundary::flow_periodic_pair(boundary_config, 0u) && nx > 0) x = boundary::wrap_index(x, nx);
        if (boundary::flow_periodic_pair(boundary_config, 1u) && ny > 0) y = boundary::wrap_index(y, ny);
        if (boundary::flow_periodic_pair(boundary_config, 2u) && nz > 0) z = boundary::wrap_index(z, nz);
        return boundary::cell_in_bounds(x, y, z, nx, ny, nz);
    }

    __device__ bool marked_cell(const std::uint32_t* cell_indices, int x, int y, int z, const int nx, const int ny, const int nz, const boundary::FlowBoundary boundary_config) {
        if (!resolve_index_cell(x, y, z, nx, ny, nz, boundary_config)) return true;
        return cell_indices[field::index(x, y, z, nx, ny)] != 0u;
    }

    __device__ float linear_interpolate(const float c000, const float c100, const float c010, const float c110, const float c001, const float c101, const float c011, const float c111, const float tx, const float ty, const float tz) {
        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float monotonic_cubic_1d(const float p0, const float p1, const float p2, const float p3, const float t) {
        const float delta = p2 - p1;
        float m1          = 0.5f * (p2 - p0);
        float m2          = 0.5f * (p3 - p1);
        if (fabsf(delta) < 1.0e-6f) {
            m1 = 0.0f;
            m2 = 0.0f;
        } else {
            if (m1 * delta <= 0.0f) m1 = 0.0f;
            if (m2 * delta <= 0.0f) m2 = 0.0f;
        }
        const float t2 = t * t;
        const float t3 = t2 * t;
        return (2.0f * t3 - 3.0f * t2 + 1.0f) * p1 + (t3 - 2.0f * t2 + t) * m1 + (-2.0f * t3 + 3.0f * t2) * p2 + (t3 - t2) * m2;
    }

    __device__ float load_scalar(const float* values, int x, int y, int z, const int nx, const int ny, const int nz, const boundary::ScalarBoundary boundary_config) {
        if (x < 0 || x >= nx) {
            const boundary::ScalarBoundaryFace face = x < 0 ? boundary_config.x_minus : boundary_config.x_plus;
            if (boundary::scalar_periodic_pair(boundary_config, 0u) && nx > 0)
                x = boundary::wrap_index(x, nx);
            else if (face.type == boundary::scalar_boundary_zero_flux && nx > 0)
                x = x < 0 ? 0 : nx - 1;
            else
                return face.value;
        }
        if (y < 0 || y >= ny) {
            const boundary::ScalarBoundaryFace face = y < 0 ? boundary_config.y_minus : boundary_config.y_plus;
            if (boundary::scalar_periodic_pair(boundary_config, 1u) && ny > 0)
                y = boundary::wrap_index(y, ny);
            else if (face.type == boundary::scalar_boundary_zero_flux && ny > 0)
                y = y < 0 ? 0 : ny - 1;
            else
                return face.value;
        }
        if (z < 0 || z >= nz) {
            const boundary::ScalarBoundaryFace face = z < 0 ? boundary_config.z_minus : boundary_config.z_plus;
            if (boundary::scalar_periodic_pair(boundary_config, 2u) && nz > 0)
                z = boundary::wrap_index(z, nz);
            else if (face.type == boundary::scalar_boundary_zero_flux && nz > 0)
                z = z < 0 ? 0 : nz - 1;
            else
                return face.value;
        }
        return values[field::index(x, y, z, nx, ny)];
    }

    __device__ float load_staggered_component_boundary(const float* values, const std::uint32_t axis, const std::uint32_t boundary_dimension, const bool lower, int i, int j, int k, const int nx, const int ny, const int nz, const boundary::FlowBoundary boundary_config) {
        const boundary::FlowBoundaryFace face = boundary::flow_face(boundary_config, boundary_dimension, lower);
        i                                     = boundary::clamp_int(i, 0, field::extent(axis, 0u, nx, ny, nz) - 1);
        j                                     = boundary::clamp_int(j, 0, field::extent(axis, 1u, nx, ny, nz) - 1);
        k                                     = boundary::clamp_int(k, 0, field::extent(axis, 2u, nx, ny, nz) - 1);
        if (boundary_dimension == 0u) i = lower ? 0 : field::extent(axis, 0u, nx, ny, nz) - 1;
        if (boundary_dimension == 1u) j = lower ? 0 : field::extent(axis, 1u, nx, ny, nz) - 1;
        if (boundary_dimension == 2u) k = lower ? 0 : field::extent(axis, 2u, nx, ny, nz) - 1;
        const float interior = values[field::index(axis, i, j, k, nx, ny)];
        if (face.type == boundary::flow_boundary_outflow || (face.type == boundary::flow_boundary_free_slip_wall && axis != boundary_dimension)) return interior;
        return 2.0f * boundary::flow_face_velocity(face, axis) - interior;
    }

    __device__ float load_staggered_component(const float* values, const std::uint32_t axis, int i, int j, int k, const int nx, const int ny, const int nz, const boundary::FlowBoundary boundary_config) {
        if (i < 0 || i >= field::extent(axis, 0u, nx, ny, nz)) {
            if (boundary::flow_periodic_pair(boundary_config, 0u) && nx > 0)
                i = boundary::wrap_index(i, nx);
            else
                return load_staggered_component_boundary(values, axis, 0u, i < 0, i, j, k, nx, ny, nz, boundary_config);
        }
        if (j < 0 || j >= field::extent(axis, 1u, nx, ny, nz)) {
            if (boundary::flow_periodic_pair(boundary_config, 1u) && ny > 0)
                j = boundary::wrap_index(j, ny);
            else
                return load_staggered_component_boundary(values, axis, 1u, j < 0, i, j, k, nx, ny, nz, boundary_config);
        }
        if (k < 0 || k >= field::extent(axis, 2u, nx, ny, nz)) {
            if (boundary::flow_periodic_pair(boundary_config, 2u) && nz > 0)
                k = boundary::wrap_index(k, nz);
            else
                return load_staggered_component_boundary(values, axis, 2u, k < 0, i, j, k, nx, ny, nz, boundary_config);
        }
        return values[field::index(axis, i, j, k, nx, ny)];
    }

    __device__ float sample_scalar_linear(const float* field, float3 position, const int nx, const int ny, const int nz, const float h, const boundary::ScalarBoundary boundary_config) {
        position       = wrap_scalar_position(position, nx, ny, nz, h, boundary_config);
        const float gx = position.x / h - 0.5f;
        const float gy = position.y / h - 0.5f;
        const float gz = position.z / h - 0.5f;
        const int x0   = static_cast<int>(floorf(gx));
        const int y0   = static_cast<int>(floorf(gy));
        const int z0   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        return linear_interpolate(load_scalar(field, x0, y0, z0, nx, ny, nz, boundary_config), load_scalar(field, x0 + 1, y0, z0, nx, ny, nz, boundary_config), load_scalar(field, x0, y0 + 1, z0, nx, ny, nz, boundary_config), load_scalar(field, x0 + 1, y0 + 1, z0, nx, ny, nz, boundary_config), load_scalar(field, x0, y0, z0 + 1, nx, ny, nz, boundary_config), load_scalar(field, x0 + 1, y0, z0 + 1, nx, ny, nz, boundary_config), load_scalar(field, x0, y0 + 1, z0 + 1, nx, ny, nz, boundary_config), load_scalar(field, x0 + 1, y0 + 1, z0 + 1, nx, ny, nz, boundary_config), tx, ty, tz);
    }

    __device__ float sample_scalar_cubic(const float* field, float3 position, const int nx, const int ny, const int nz, const float h, const boundary::ScalarBoundary boundary_config) {
        position       = wrap_scalar_position(position, nx, ny, nz, h, boundary_config);
        const float gx = position.x / h - 0.5f;
        const float gy = position.y / h - 0.5f;
        const float gz = position.z / h - 0.5f;
        const int x1   = static_cast<int>(floorf(gx));
        const int y1   = static_cast<int>(floorf(gy));
        const int z1   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(x1);
        const float ty = gy - static_cast<float>(y1);
        const float tz = gz - static_cast<float>(z1);
        float z_samples[4];
        for (int dz = 0; dz < 4; ++dz) {
            float y_samples[4];
            for (int dy = 0; dy < 4; ++dy) {
                const int yy   = y1 + dy - 1;
                const int zz   = z1 + dz - 1;
                const float p0 = load_scalar(field, x1 - 1, yy, zz, nx, ny, nz, boundary_config);
                const float p1 = load_scalar(field, x1, yy, zz, nx, ny, nz, boundary_config);
                const float p2 = load_scalar(field, x1 + 1, yy, zz, nx, ny, nz, boundary_config);
                const float p3 = load_scalar(field, x1 + 2, yy, zz, nx, ny, nz, boundary_config);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float sample_staggered_component_linear(const float* field, const std::uint32_t axis, float3 position, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        position       = wrap_vector_position(position, nx, ny, nz, h, boundary_config);
        const float gx = position.x / h - (axis == 0u ? 0.0f : 0.5f);
        const float gy = position.y / h - (axis == 1u ? 0.0f : 0.5f);
        const float gz = position.z / h - (axis == 2u ? 0.0f : 0.5f);
        const int i0   = static_cast<int>(floorf(gx));
        const int j0   = static_cast<int>(floorf(gy));
        const int k0   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(i0);
        const float ty = gy - static_cast<float>(j0);
        const float tz = gz - static_cast<float>(k0);
        return linear_interpolate(load_staggered_component(field, axis, i0, j0, k0, nx, ny, nz, boundary_config), load_staggered_component(field, axis, i0 + 1, j0, k0, nx, ny, nz, boundary_config), load_staggered_component(field, axis, i0, j0 + 1, k0, nx, ny, nz, boundary_config), load_staggered_component(field, axis, i0 + 1, j0 + 1, k0, nx, ny, nz, boundary_config), load_staggered_component(field, axis, i0, j0, k0 + 1, nx, ny, nz, boundary_config), load_staggered_component(field, axis, i0 + 1, j0, k0 + 1, nx, ny, nz, boundary_config), load_staggered_component(field, axis, i0, j0 + 1, k0 + 1, nx, ny, nz, boundary_config), load_staggered_component(field, axis, i0 + 1, j0 + 1, k0 + 1, nx, ny, nz, boundary_config), tx, ty, tz);
    }

    __device__ float sample_staggered_component_cubic(const float* field, const std::uint32_t axis, float3 position, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        position       = wrap_vector_position(position, nx, ny, nz, h, boundary_config);
        const float gx = position.x / h - (axis == 0u ? 0.0f : 0.5f);
        const float gy = position.y / h - (axis == 1u ? 0.0f : 0.5f);
        const float gz = position.z / h - (axis == 2u ? 0.0f : 0.5f);
        const int i1   = static_cast<int>(floorf(gx));
        const int j1   = static_cast<int>(floorf(gy));
        const int k1   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(i1);
        const float ty = gy - static_cast<float>(j1);
        const float tz = gz - static_cast<float>(k1);
        float z_samples[4];
        for (int dz = 0; dz < 4; ++dz) {
            float y_samples[4];
            for (int dy = 0; dy < 4; ++dy) {
                const int jj   = j1 + dy - 1;
                const int kk   = k1 + dz - 1;
                const float p0 = load_staggered_component(field, axis, i1 - 1, jj, kk, nx, ny, nz, boundary_config);
                const float p1 = load_staggered_component(field, axis, i1, jj, kk, nx, ny, nz, boundary_config);
                const float p2 = load_staggered_component(field, axis, i1 + 1, jj, kk, nx, ny, nz, boundary_config);
                const float p3 = load_staggered_component(field, axis, i1 + 2, jj, kk, nx, ny, nz, boundary_config);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float3 sample_staggered_vector(const float* x_component, const float* y_component, const float* z_component, const float3 position, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        return make_float3(sample_staggered_component_linear(x_component, 0u, position, nx, ny, nz, h, boundary_config), sample_staggered_component_linear(y_component, 1u, position, nx, ny, nz, h, boundary_config), sample_staggered_component_linear(z_component, 2u, position, nx, ny, nz, h, boundary_config));
    }

    __device__ bool position_hits_marked_cell(const float3 position, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        const float3 wrapped = wrap_vector_position(position, nx, ny, nz, h, boundary_config);
        if (wrapped.x < 0.0f || wrapped.x > static_cast<float>(nx) * h || wrapped.y < 0.0f || wrapped.y > static_cast<float>(ny) * h || wrapped.z < 0.0f || wrapped.z > static_cast<float>(nz) * h) return true;
        int cell_x = static_cast<int>(floorf(wrapped.x / h));
        int cell_y = static_cast<int>(floorf(wrapped.y / h));
        int cell_z = static_cast<int>(floorf(wrapped.z / h));
        if (cell_x == nx) cell_x = nx - 1;
        if (cell_y == ny) cell_y = ny - 1;
        if (cell_z == nz) cell_z = nz - 1;
        return !boundary::cell_in_bounds(cell_x, cell_y, cell_z, nx, ny, nz) || cell_indices[field::index(cell_x, cell_y, cell_z, nx, ny)] != 0u;
    }

    __device__ float3 trace_rk2(const float3 start, const float* x_component, const float* y_component, const float* z_component, const std::uint32_t* cell_indices, const float dt, const int nx, const int ny, const int nz, const float h, const boundary::FlowBoundary boundary_config) {
        const float3 value0 = sample_staggered_vector(x_component, y_component, z_component, start, nx, ny, nz, h, boundary_config);
        const float3 mid    = make_float3(start.x - 0.5f * dt * value0.x, start.y - 0.5f * dt * value0.y, start.z - 0.5f * dt * value0.z);
        const float3 value1 = sample_staggered_vector(x_component, y_component, z_component, mid, nx, ny, nz, h, boundary_config);
        const float3 traced = make_float3(start.x - dt * value1.x, start.y - dt * value1.y, start.z - dt * value1.z);
        if (!position_hits_marked_cell(traced, cell_indices, nx, ny, nz, h, boundary_config)) return traced;

        float lo = 0.0f;
        float hi = 1.0f;
        for (int iteration = 0; iteration < 10; ++iteration) {
            const float t     = 0.5f * (lo + hi);
            const float3 test = make_float3(start.x + (traced.x - start.x) * t, start.y + (traced.y - start.y) * t, start.z + (traced.z - start.z) * t);
            if (position_hits_marked_cell(test, cell_indices, nx, ny, nz, h, boundary_config))
                hi = t;
            else
                lo = t;
        }
        return make_float3(start.x + (traced.x - start.x) * lo, start.y + (traced.y - start.y) * lo, start.z + (traced.z - start.z) * lo);
    }

    __device__ float3 staggered_sample_position(const std::uint32_t axis, const int i, const int j, const int k, const float h) {
        return make_float3((static_cast<float>(i) + (axis == 0u ? 0.0f : 0.5f)) * h, (static_cast<float>(j) + (axis == 1u ? 0.0f : 0.5f)) * h, (static_cast<float>(k) + (axis == 2u ? 0.0f : 0.5f)) * h);
    }

    __global__ void advect_staggered_component_kernel(const std::uint32_t axis, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t scheme, const boundary::FlowBoundary boundary_config) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= field::extent(axis, 0u, nx, ny, nz) || j >= field::extent(axis, 1u, nx, ny, nz) || k >= field::extent(axis, 2u, nx, ny, nz)) return;
        const float3 start                                  = staggered_sample_position(axis, i, j, k, h);
        const float3 position                               = trace_rk2(start, vector_x, vector_y, vector_z, cell_indices, dt, nx, ny, nz, h, boundary_config);
        destination[field::index(axis, i, j, k, nx, ny)] = scheme == scheme_monotonic_cubic ? sample_staggered_component_cubic(source, axis, position, nx, ny, nz, h, boundary_config) : sample_staggered_component_linear(source, axis, position, nx, ny, nz, h, boundary_config);
    }

    __global__ void advect_centered_scalar_kernel(float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t scheme, const boundary::ScalarBoundary scalar_boundary, const boundary::FlowBoundary vector_boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (marked_cell(cell_indices, x, y, z, nx, ny, nz, vector_boundary)) {
            destination[field::index(x, y, z, nx, ny)] = 0.0f;
            return;
        }
        const float3 start                               = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
        const float3 position                            = trace_rk2(start, vector_x, vector_y, vector_z, cell_indices, dt, nx, ny, nz, h, vector_boundary);
        destination[field::index(x, y, z, nx, ny)] = scheme == scheme_monotonic_cubic ? sample_scalar_cubic(source, position, nx, ny, nz, h, scalar_boundary) : sample_scalar_linear(source, position, nx, ny, nz, h, scalar_boundary);
    }

    void advect_staggered_component(cudaStream_t stream, const std::uint32_t axis, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t advection_mode, const std::uint32_t* boundary_types, const float* boundary_values) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid                              = field::staggered_grid(axis, nx, ny, nz, block);
        const boundary::FlowBoundary boundary_config = boundary::make_flow_velocity_boundary(boundary_types, boundary_values);
        advect_staggered_component_kernel<<<grid, block, 0, stream>>>(axis, destination, source, vector_x, vector_y, vector_z, cell_indices, nx, ny, nz, h, dt, advection_mode, boundary_config);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"advect_staggered_component_kernel: "} + cudaGetErrorString(status)};
    }

    void advect_centered_scalar(cudaStream_t stream, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint32_t* cell_indices, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t advection_mode, const std::uint32_t* scalar_boundary_types, const float* scalar_boundary_values, const std::uint32_t* vector_boundary_types, const float* vector_boundary_values) {
        constexpr dim3 block{8u, 8u, 4u};
        const dim3 grid                                = field::centered_grid(nx, ny, nz, block);
        const boundary::ScalarBoundary scalar_boundary = boundary::make_scalar_boundary(scalar_boundary_types, scalar_boundary_values);
        const boundary::FlowBoundary vector_boundary   = boundary::make_flow_velocity_boundary(vector_boundary_types, vector_boundary_values);
        advect_centered_scalar_kernel<<<grid, block, 0, stream>>>(destination, source, vector_x, vector_y, vector_z, cell_indices, nx, ny, nz, h, dt, advection_mode, scalar_boundary, vector_boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"advect_centered_scalar_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::advection
