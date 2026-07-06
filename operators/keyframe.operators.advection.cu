#include "keyframe.operators.advection.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace kfs::cuda::operators::advection {
    constexpr std::uint32_t vector_boundary_no_slip_wall   = 0u;
    constexpr std::uint32_t vector_boundary_free_slip_wall = 1u;
    constexpr std::uint32_t vector_boundary_outflow        = 2u;
    constexpr std::uint32_t vector_boundary_periodic       = 3u;

    constexpr std::uint32_t scalar_boundary_fixed_value = 0u;
    constexpr std::uint32_t scalar_boundary_zero_flux   = 1u;
    constexpr std::uint32_t scalar_boundary_periodic    = 2u;

    constexpr std::uint32_t scheme_monotonic_cubic = 1u;

    struct VectorBoundaryFace final {
        std::uint32_t type{vector_boundary_no_slip_wall};
        float x{0.0f};
        float y{0.0f};
        float z{0.0f};
    };

    struct VectorBoundary final {
        VectorBoundaryFace x_minus{};
        VectorBoundaryFace x_plus{};
        VectorBoundaryFace y_minus{};
        VectorBoundaryFace y_plus{};
        VectorBoundaryFace z_minus{};
        VectorBoundaryFace z_plus{};
    };

    struct ScalarBoundaryFace final {
        std::uint32_t type{scalar_boundary_fixed_value};
        float value{0.0f};
    };

    struct ScalarBoundary final {
        ScalarBoundaryFace x_minus{};
        ScalarBoundaryFace x_plus{};
        ScalarBoundaryFace y_minus{};
        ScalarBoundaryFace y_plus{};
        ScalarBoundaryFace z_minus{};
        ScalarBoundaryFace z_plus{};
    };

    VectorBoundary make_vector_boundary(const std::uint32_t* types, const float* values) {
        if (types == nullptr || values == nullptr) throw std::runtime_error{"Advection vector boundary arrays must not be null"};
        return VectorBoundary{
            {types[0], values[0], values[1], values[2]},
            {types[1], values[3], values[4], values[5]},
            {types[2], values[6], values[7], values[8]},
            {types[3], values[9], values[10], values[11]},
            {types[4], values[12], values[13], values[14]},
            {types[5], values[15], values[16], values[17]},
        };
    }

    ScalarBoundary make_scalar_boundary(const std::uint32_t* types, const float* values) {
        if (types == nullptr || values == nullptr) throw std::runtime_error{"Advection scalar boundary arrays must not be null"};
        return ScalarBoundary{
            {types[0], values[0]},
            {types[1], values[1]},
            {types[2], values[2]},
            {types[3], values[3]},
            {types[4], values[4]},
            {types[5], values[5]},
        };
    }

    unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
        return static_cast<unsigned>((value + divisor - 1u) / divisor);
    }

    dim3 advection_block() {
        return dim3{8u, 8u, 4u};
    }

    void require_resolution(const int nx, const int ny, const int nz) {
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error{"Advection resolution must be positive"};
    }

    dim3 centered_scalar_grid(const int nx, const int ny, const int nz, const dim3& block) {
        require_resolution(nx, ny, nz);
        return dim3{ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), ceil_div_u32(static_cast<std::uint64_t>(nz), block.z)};
    }

    dim3 staggered_component_grid(const std::uint32_t axis, const int nx, const int ny, const int nz, const dim3& block) {
        if (axis >= 3u) throw std::runtime_error{"advect_staggered_component: axis must be 0, 1, or 2"};
        require_resolution(nx, ny, nz);
        const auto sx = static_cast<std::uint64_t>(nx) + (axis == 0u ? 1u : 0u);
        const auto sy = static_cast<std::uint64_t>(ny) + (axis == 1u ? 1u : 0u);
        const auto sz = static_cast<std::uint64_t>(nz) + (axis == 2u ? 1u : 0u);
        return dim3{ceil_div_u32(sx, block.x), ceil_div_u32(sy, block.y), ceil_div_u32(sz, block.z)};
    }

    __device__ std::uint64_t index_centered(const int x, const int y, const int z, const int nx, const int ny) {
        return static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(nx) * (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(z));
    }

    __device__ int wrap_index(int value, const int size) {
        if (size <= 0) return 0;
        value %= size;
        if (value < 0) value += size;
        return value;
    }

    __device__ int clamp_int(const int value, const int low, const int high) {
        return value < low ? low : (value > high ? high : value);
    }

    __device__ int axis_cells(const std::uint32_t dimension, const int nx, const int ny, const int nz) {
        if (dimension == 0u) return nx;
        if (dimension == 1u) return ny;
        return nz;
    }

    __device__ int staggered_extent(const std::uint32_t axis, const std::uint32_t dimension, const int nx, const int ny, const int nz) {
        return axis_cells(dimension, nx, ny, nz) + (axis == dimension ? 1 : 0);
    }

    __device__ std::uint64_t index_staggered(const std::uint32_t axis, const int i, const int j, const int k, const int nx, const int ny) {
        const auto nx64 = static_cast<std::uint64_t>(nx);
        const auto ny64 = static_cast<std::uint64_t>(ny);
        if (axis == 0u) return static_cast<std::uint64_t>(i) + (nx64 + 1u) * (static_cast<std::uint64_t>(j) + ny64 * static_cast<std::uint64_t>(k));
        if (axis == 1u) return static_cast<std::uint64_t>(i) + nx64 * (static_cast<std::uint64_t>(j) + (ny64 + 1u) * static_cast<std::uint64_t>(k));
        return static_cast<std::uint64_t>(i) + nx64 * (static_cast<std::uint64_t>(j) + ny64 * static_cast<std::uint64_t>(k));
    }

    __device__ bool cell_in_bounds(const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
    }

    __device__ VectorBoundaryFace vector_face(const VectorBoundary boundary, const std::uint32_t dimension, const bool lower) {
        if (dimension == 0u) return lower ? boundary.x_minus : boundary.x_plus;
        if (dimension == 1u) return lower ? boundary.y_minus : boundary.y_plus;
        return lower ? boundary.z_minus : boundary.z_plus;
    }

    __device__ bool vector_periodic_pair(const VectorBoundary boundary, const std::uint32_t dimension) {
        if (dimension == 0u) return boundary.x_minus.type == vector_boundary_periodic && boundary.x_plus.type == vector_boundary_periodic;
        if (dimension == 1u) return boundary.y_minus.type == vector_boundary_periodic && boundary.y_plus.type == vector_boundary_periodic;
        return boundary.z_minus.type == vector_boundary_periodic && boundary.z_plus.type == vector_boundary_periodic;
    }

    __device__ bool scalar_periodic_pair(const ScalarBoundary boundary, const std::uint32_t dimension) {
        if (dimension == 0u) return boundary.x_minus.type == scalar_boundary_periodic && boundary.x_plus.type == scalar_boundary_periodic;
        if (dimension == 1u) return boundary.y_minus.type == scalar_boundary_periodic && boundary.y_plus.type == scalar_boundary_periodic;
        return boundary.z_minus.type == scalar_boundary_periodic && boundary.z_plus.type == scalar_boundary_periodic;
    }

    __device__ float vector_face_value(const VectorBoundaryFace face, const std::uint32_t axis) {
        if (axis == 0u) return face.x;
        if (axis == 1u) return face.y;
        return face.z;
    }

    __device__ float wrap_periodic_position(float value, const int cells, const float h, const bool periodic) {
        if (!periodic || cells <= 0) return value;
        const float extent = static_cast<float>(cells) * h;
        value              = fmodf(value, extent);
        if (value < 0.0f) value += extent;
        return value;
    }

    __device__ float3 wrap_vector_position(float3 position, const int nx, const int ny, const int nz, const float h, const VectorBoundary boundary) {
        position.x = wrap_periodic_position(position.x, nx, h, vector_periodic_pair(boundary, 0u));
        position.y = wrap_periodic_position(position.y, ny, h, vector_periodic_pair(boundary, 1u));
        position.z = wrap_periodic_position(position.z, nz, h, vector_periodic_pair(boundary, 2u));
        return position;
    }

    __device__ float3 wrap_scalar_position(float3 position, const int nx, const int ny, const int nz, const float h, const ScalarBoundary boundary) {
        position.x = wrap_periodic_position(position.x, nx, h, scalar_periodic_pair(boundary, 0u));
        position.y = wrap_periodic_position(position.y, ny, h, scalar_periodic_pair(boundary, 1u));
        position.z = wrap_periodic_position(position.z, nz, h, scalar_periodic_pair(boundary, 2u));
        return position;
    }

    __device__ bool resolve_mask_cell(int& x, int& y, int& z, const int nx, const int ny, const int nz, const VectorBoundary boundary) {
        if (vector_periodic_pair(boundary, 0u) && nx > 0) x = wrap_index(x, nx);
        if (vector_periodic_pair(boundary, 1u) && ny > 0) y = wrap_index(y, ny);
        if (vector_periodic_pair(boundary, 2u) && nz > 0) z = wrap_index(z, nz);
        return cell_in_bounds(x, y, z, nx, ny, nz);
    }

    __device__ bool masked_cell(const std::uint8_t* cell_mask, int x, int y, int z, const int nx, const int ny, const int nz, const VectorBoundary boundary) {
        if (cell_mask == nullptr) return false;
        if (!resolve_mask_cell(x, y, z, nx, ny, nz, boundary)) return true;
        return cell_mask[index_centered(x, y, z, nx, ny)] != 0;
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

    __device__ float load_scalar(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const ScalarBoundary boundary) {
        if (x < 0 || x >= nx) {
            const ScalarBoundaryFace face = x < 0 ? boundary.x_minus : boundary.x_plus;
            if (scalar_periodic_pair(boundary, 0u) && nx > 0) x = wrap_index(x, nx);
            else if (face.type == scalar_boundary_zero_flux && nx > 0) x = x < 0 ? 0 : nx - 1;
            else return face.value;
        }
        if (y < 0 || y >= ny) {
            const ScalarBoundaryFace face = y < 0 ? boundary.y_minus : boundary.y_plus;
            if (scalar_periodic_pair(boundary, 1u) && ny > 0) y = wrap_index(y, ny);
            else if (face.type == scalar_boundary_zero_flux && ny > 0) y = y < 0 ? 0 : ny - 1;
            else return face.value;
        }
        if (z < 0 || z >= nz) {
            const ScalarBoundaryFace face = z < 0 ? boundary.z_minus : boundary.z_plus;
            if (scalar_periodic_pair(boundary, 2u) && nz > 0) z = wrap_index(z, nz);
            else if (face.type == scalar_boundary_zero_flux && nz > 0) z = z < 0 ? 0 : nz - 1;
            else return face.value;
        }
        return field[index_centered(x, y, z, nx, ny)];
    }

    __device__ float load_staggered_component_boundary(const float* field, const std::uint32_t axis, const std::uint32_t boundary_dimension, const bool lower, int i, int j, int k, const int nx, const int ny, const int nz, const VectorBoundary boundary) {
        const VectorBoundaryFace face = vector_face(boundary, boundary_dimension, lower);
        i                             = clamp_int(i, 0, staggered_extent(axis, 0u, nx, ny, nz) - 1);
        j                             = clamp_int(j, 0, staggered_extent(axis, 1u, nx, ny, nz) - 1);
        k                             = clamp_int(k, 0, staggered_extent(axis, 2u, nx, ny, nz) - 1);
        if (boundary_dimension == 0u) i = lower ? 0 : staggered_extent(axis, 0u, nx, ny, nz) - 1;
        if (boundary_dimension == 1u) j = lower ? 0 : staggered_extent(axis, 1u, nx, ny, nz) - 1;
        if (boundary_dimension == 2u) k = lower ? 0 : staggered_extent(axis, 2u, nx, ny, nz) - 1;
        const float interior = field[index_staggered(axis, i, j, k, nx, ny)];
        if (face.type == vector_boundary_outflow || (face.type == vector_boundary_free_slip_wall && axis != boundary_dimension)) return interior;
        return 2.0f * vector_face_value(face, axis) - interior;
    }

    __device__ float load_staggered_component(const float* field, const std::uint32_t axis, int i, int j, int k, const int nx, const int ny, const int nz, const VectorBoundary boundary) {
        if (i < 0 || i >= staggered_extent(axis, 0u, nx, ny, nz)) {
            if (vector_periodic_pair(boundary, 0u) && nx > 0) i = wrap_index(i, nx);
            else return load_staggered_component_boundary(field, axis, 0u, i < 0, i, j, k, nx, ny, nz, boundary);
        }
        if (j < 0 || j >= staggered_extent(axis, 1u, nx, ny, nz)) {
            if (vector_periodic_pair(boundary, 1u) && ny > 0) j = wrap_index(j, ny);
            else return load_staggered_component_boundary(field, axis, 1u, j < 0, i, j, k, nx, ny, nz, boundary);
        }
        if (k < 0 || k >= staggered_extent(axis, 2u, nx, ny, nz)) {
            if (vector_periodic_pair(boundary, 2u) && nz > 0) k = wrap_index(k, nz);
            else return load_staggered_component_boundary(field, axis, 2u, k < 0, i, j, k, nx, ny, nz, boundary);
        }
        return field[index_staggered(axis, i, j, k, nx, ny)];
    }

    __device__ float sample_scalar_linear(const float* field, float3 position, const int nx, const int ny, const int nz, const float h, const ScalarBoundary boundary) {
        position       = wrap_scalar_position(position, nx, ny, nz, h, boundary);
        const float gx = position.x / h - 0.5f;
        const float gy = position.y / h - 0.5f;
        const float gz = position.z / h - 0.5f;
        const int x0   = static_cast<int>(floorf(gx));
        const int y0   = static_cast<int>(floorf(gy));
        const int z0   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        return linear_interpolate(
            load_scalar(field, x0, y0, z0, nx, ny, nz, boundary),
            load_scalar(field, x0 + 1, y0, z0, nx, ny, nz, boundary),
            load_scalar(field, x0, y0 + 1, z0, nx, ny, nz, boundary),
            load_scalar(field, x0 + 1, y0 + 1, z0, nx, ny, nz, boundary),
            load_scalar(field, x0, y0, z0 + 1, nx, ny, nz, boundary),
            load_scalar(field, x0 + 1, y0, z0 + 1, nx, ny, nz, boundary),
            load_scalar(field, x0, y0 + 1, z0 + 1, nx, ny, nz, boundary),
            load_scalar(field, x0 + 1, y0 + 1, z0 + 1, nx, ny, nz, boundary),
            tx,
            ty,
            tz);
    }

    __device__ float sample_scalar_cubic(const float* field, float3 position, const int nx, const int ny, const int nz, const float h, const ScalarBoundary boundary) {
        position       = wrap_scalar_position(position, nx, ny, nz, h, boundary);
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
                const float p0 = load_scalar(field, x1 - 1, yy, zz, nx, ny, nz, boundary);
                const float p1 = load_scalar(field, x1, yy, zz, nx, ny, nz, boundary);
                const float p2 = load_scalar(field, x1 + 1, yy, zz, nx, ny, nz, boundary);
                const float p3 = load_scalar(field, x1 + 2, yy, zz, nx, ny, nz, boundary);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float sample_staggered_component_linear(const float* field, const std::uint32_t axis, float3 position, const int nx, const int ny, const int nz, const float h, const VectorBoundary boundary) {
        position       = wrap_vector_position(position, nx, ny, nz, h, boundary);
        const float gx = position.x / h - (axis == 0u ? 0.0f : 0.5f);
        const float gy = position.y / h - (axis == 1u ? 0.0f : 0.5f);
        const float gz = position.z / h - (axis == 2u ? 0.0f : 0.5f);
        const int i0   = static_cast<int>(floorf(gx));
        const int j0   = static_cast<int>(floorf(gy));
        const int k0   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(i0);
        const float ty = gy - static_cast<float>(j0);
        const float tz = gz - static_cast<float>(k0);
        return linear_interpolate(
            load_staggered_component(field, axis, i0, j0, k0, nx, ny, nz, boundary),
            load_staggered_component(field, axis, i0 + 1, j0, k0, nx, ny, nz, boundary),
            load_staggered_component(field, axis, i0, j0 + 1, k0, nx, ny, nz, boundary),
            load_staggered_component(field, axis, i0 + 1, j0 + 1, k0, nx, ny, nz, boundary),
            load_staggered_component(field, axis, i0, j0, k0 + 1, nx, ny, nz, boundary),
            load_staggered_component(field, axis, i0 + 1, j0, k0 + 1, nx, ny, nz, boundary),
            load_staggered_component(field, axis, i0, j0 + 1, k0 + 1, nx, ny, nz, boundary),
            load_staggered_component(field, axis, i0 + 1, j0 + 1, k0 + 1, nx, ny, nz, boundary),
            tx,
            ty,
            tz);
    }

    __device__ float sample_staggered_component_cubic(const float* field, const std::uint32_t axis, float3 position, const int nx, const int ny, const int nz, const float h, const VectorBoundary boundary) {
        position       = wrap_vector_position(position, nx, ny, nz, h, boundary);
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
                const float p0 = load_staggered_component(field, axis, i1 - 1, jj, kk, nx, ny, nz, boundary);
                const float p1 = load_staggered_component(field, axis, i1, jj, kk, nx, ny, nz, boundary);
                const float p2 = load_staggered_component(field, axis, i1 + 1, jj, kk, nx, ny, nz, boundary);
                const float p3 = load_staggered_component(field, axis, i1 + 2, jj, kk, nx, ny, nz, boundary);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float3 sample_staggered_vector(const float* x_component, const float* y_component, const float* z_component, const float3 position, const int nx, const int ny, const int nz, const float h, const VectorBoundary boundary) {
        return make_float3(
            sample_staggered_component_linear(x_component, 0u, position, nx, ny, nz, h, boundary),
            sample_staggered_component_linear(y_component, 1u, position, nx, ny, nz, h, boundary),
            sample_staggered_component_linear(z_component, 2u, position, nx, ny, nz, h, boundary));
    }

    __device__ bool position_hits_mask(const float3 position, const std::uint8_t* cell_mask, const int nx, const int ny, const int nz, const float h, const VectorBoundary boundary) {
        const float3 wrapped = wrap_vector_position(position, nx, ny, nz, h, boundary);
        if (wrapped.x < 0.0f || wrapped.x > static_cast<float>(nx) * h || wrapped.y < 0.0f || wrapped.y > static_cast<float>(ny) * h || wrapped.z < 0.0f || wrapped.z > static_cast<float>(nz) * h) return true;
        if (cell_mask == nullptr) return false;
        int cell_x = static_cast<int>(floorf(wrapped.x / h));
        int cell_y = static_cast<int>(floorf(wrapped.y / h));
        int cell_z = static_cast<int>(floorf(wrapped.z / h));
        if (cell_x == nx) cell_x = nx - 1;
        if (cell_y == ny) cell_y = ny - 1;
        if (cell_z == nz) cell_z = nz - 1;
        return !cell_in_bounds(cell_x, cell_y, cell_z, nx, ny, nz) || cell_mask[index_centered(cell_x, cell_y, cell_z, nx, ny)] != 0;
    }

    __device__ float3 trace_rk2(const float3 start, const float* x_component, const float* y_component, const float* z_component, const std::uint8_t* cell_mask, const float dt, const int nx, const int ny, const int nz, const float h, const VectorBoundary boundary) {
        const float3 value0 = sample_staggered_vector(x_component, y_component, z_component, start, nx, ny, nz, h, boundary);
        const float3 mid    = make_float3(start.x - 0.5f * dt * value0.x, start.y - 0.5f * dt * value0.y, start.z - 0.5f * dt * value0.z);
        const float3 value1 = sample_staggered_vector(x_component, y_component, z_component, mid, nx, ny, nz, h, boundary);
        const float3 traced = make_float3(start.x - dt * value1.x, start.y - dt * value1.y, start.z - dt * value1.z);
        if (!position_hits_mask(traced, cell_mask, nx, ny, nz, h, boundary)) return traced;

        float lo = 0.0f;
        float hi = 1.0f;
        for (int iteration = 0; iteration < 10; ++iteration) {
            const float t     = 0.5f * (lo + hi);
            const float3 test = make_float3(start.x + (traced.x - start.x) * t, start.y + (traced.y - start.y) * t, start.z + (traced.z - start.z) * t);
            if (position_hits_mask(test, cell_mask, nx, ny, nz, h, boundary)) hi = t;
            else lo = t;
        }
        return make_float3(start.x + (traced.x - start.x) * lo, start.y + (traced.y - start.y) * lo, start.z + (traced.z - start.z) * lo);
    }

    __device__ float3 staggered_sample_position(const std::uint32_t axis, const int i, const int j, const int k, const float h) {
        return make_float3((static_cast<float>(i) + (axis == 0u ? 0.0f : 0.5f)) * h, (static_cast<float>(j) + (axis == 1u ? 0.0f : 0.5f)) * h, (static_cast<float>(k) + (axis == 2u ? 0.0f : 0.5f)) * h);
    }

    __global__ void advect_staggered_component_kernel(const std::uint32_t axis, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint8_t* cell_mask, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t scheme, const VectorBoundary boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= staggered_extent(axis, 0u, nx, ny, nz) || j >= staggered_extent(axis, 1u, nx, ny, nz) || k >= staggered_extent(axis, 2u, nx, ny, nz)) return;
        const float3 start    = staggered_sample_position(axis, i, j, k, h);
        const float3 position = trace_rk2(start, vector_x, vector_y, vector_z, cell_mask, dt, nx, ny, nz, h, boundary);
        destination[index_staggered(axis, i, j, k, nx, ny)] = scheme == scheme_monotonic_cubic ? sample_staggered_component_cubic(source, axis, position, nx, ny, nz, h, boundary) : sample_staggered_component_linear(source, axis, position, nx, ny, nz, h, boundary);
    }

    __global__ void advect_centered_scalar_kernel(float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint8_t* cell_mask, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t scheme, const ScalarBoundary scalar_boundary, const VectorBoundary vector_boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (masked_cell(cell_mask, x, y, z, nx, ny, nz, vector_boundary)) {
            destination[index_centered(x, y, z, nx, ny)] = 0.0f;
            return;
        }
        const float3 start    = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
        const float3 position = trace_rk2(start, vector_x, vector_y, vector_z, cell_mask, dt, nx, ny, nz, h, vector_boundary);
        destination[index_centered(x, y, z, nx, ny)] = scheme == scheme_monotonic_cubic ? sample_scalar_cubic(source, position, nx, ny, nz, h, scalar_boundary) : sample_scalar_linear(source, position, nx, ny, nz, h, scalar_boundary);
    }

    void advect_staggered_component(cudaStream_t stream, const std::uint32_t axis, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint8_t* cell_mask, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t advection_mode, const std::uint32_t* boundary_types, const float* boundary_values) {
        const dim3 block             = advection_block();
        const dim3 grid              = staggered_component_grid(axis, nx, ny, nz, block);
        const VectorBoundary boundary = make_vector_boundary(boundary_types, boundary_values);
        advect_staggered_component_kernel<<<grid, block, 0, stream>>>(axis, destination, source, vector_x, vector_y, vector_z, cell_mask, nx, ny, nz, h, dt, advection_mode, boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"advect_staggered_component_kernel: "} + cudaGetErrorString(status)};
    }

    void advect_centered_scalar(cudaStream_t stream, float* destination, const float* source, const float* vector_x, const float* vector_y, const float* vector_z, const std::uint8_t* cell_mask, const int nx, const int ny, const int nz, const float h, const float dt, const std::uint32_t advection_mode, const std::uint32_t* scalar_boundary_types, const float* scalar_boundary_values, const std::uint32_t* vector_boundary_types, const float* vector_boundary_values) {
        const dim3 block                      = advection_block();
        const dim3 grid                       = centered_scalar_grid(nx, ny, nz, block);
        const ScalarBoundary scalar_boundary  = make_scalar_boundary(scalar_boundary_types, scalar_boundary_values);
        const VectorBoundary vector_boundary = make_vector_boundary(vector_boundary_types, vector_boundary_values);
        advect_centered_scalar_kernel<<<grid, block, 0, stream>>>(destination, source, vector_x, vector_y, vector_z, cell_mask, nx, ny, nz, h, dt, advection_mode, scalar_boundary, vector_boundary);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"advect_centered_scalar_kernel: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::cuda::operators::advection
