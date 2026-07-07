#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "keyframe.field.cuh"

namespace kfs::cuda::boundary {
    constexpr std::uint32_t flow_boundary_no_slip_wall   = 0u;
    constexpr std::uint32_t flow_boundary_free_slip_wall = 1u;
    constexpr std::uint32_t flow_boundary_outflow        = 2u;
    constexpr std::uint32_t flow_boundary_periodic       = 3u;

    constexpr std::uint32_t scalar_boundary_fixed_value = 0u;
    constexpr std::uint32_t scalar_boundary_zero_flux   = 1u;
    constexpr std::uint32_t scalar_boundary_periodic    = 2u;

    struct FlowBoundaryFace final {
        std::uint32_t type{flow_boundary_no_slip_wall};
        float velocity_x{0.0f};
        float velocity_y{0.0f};
        float velocity_z{0.0f};
        float pressure{0.0f};
    };

    struct FlowBoundary final {
        FlowBoundaryFace x_minus{};
        FlowBoundaryFace x_plus{};
        FlowBoundaryFace y_minus{};
        FlowBoundaryFace y_plus{};
        FlowBoundaryFace z_minus{};
        FlowBoundaryFace z_plus{};
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

    inline FlowBoundary make_flow_type_boundary(const std::uint32_t* types) {
        if (types == nullptr) throw std::runtime_error{"Flow boundary type array must not be null"};
        return FlowBoundary{
            {types[0]},
            {types[1]},
            {types[2]},
            {types[3]},
            {types[4]},
            {types[5]},
        };
    }

    inline FlowBoundary make_flow_velocity_boundary(const std::uint32_t* types, const float* velocity) {
        if (types == nullptr || velocity == nullptr) throw std::runtime_error{"Flow boundary velocity arrays must not be null"};
        return FlowBoundary{
            {types[0], velocity[0], velocity[1], velocity[2]},
            {types[1], velocity[3], velocity[4], velocity[5]},
            {types[2], velocity[6], velocity[7], velocity[8]},
            {types[3], velocity[9], velocity[10], velocity[11]},
            {types[4], velocity[12], velocity[13], velocity[14]},
            {types[5], velocity[15], velocity[16], velocity[17]},
        };
    }

    inline FlowBoundary make_flow_pressure_boundary(const std::uint32_t* types, const float* pressure) {
        if (types == nullptr || pressure == nullptr) throw std::runtime_error{"Flow boundary pressure arrays must not be null"};
        return FlowBoundary{
            {types[0], 0.0f, 0.0f, 0.0f, pressure[0]},
            {types[1], 0.0f, 0.0f, 0.0f, pressure[1]},
            {types[2], 0.0f, 0.0f, 0.0f, pressure[2]},
            {types[3], 0.0f, 0.0f, 0.0f, pressure[3]},
            {types[4], 0.0f, 0.0f, 0.0f, pressure[4]},
            {types[5], 0.0f, 0.0f, 0.0f, pressure[5]},
        };
    }

    inline ScalarBoundary make_scalar_boundary(const std::uint32_t* types, const float* values) {
        if (types == nullptr || values == nullptr) throw std::runtime_error{"Scalar boundary arrays must not be null"};
        return ScalarBoundary{
            {types[0], values[0]},
            {types[1], values[1]},
            {types[2], values[2]},
            {types[3], values[3]},
            {types[4], values[4]},
            {types[5], values[5]},
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

    __device__ inline bool cell_in_bounds(const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
    }

    __device__ inline FlowBoundaryFace flow_face(const FlowBoundary boundary, const std::uint32_t dimension, const bool lower) {
        if (dimension == 0u) return lower ? boundary.x_minus : boundary.x_plus;
        if (dimension == 1u) return lower ? boundary.y_minus : boundary.y_plus;
        return lower ? boundary.z_minus : boundary.z_plus;
    }

    __device__ inline bool flow_periodic_pair(const FlowBoundary boundary, const std::uint32_t dimension) {
        if (dimension == 0u) return boundary.x_minus.type == flow_boundary_periodic && boundary.x_plus.type == flow_boundary_periodic;
        if (dimension == 1u) return boundary.y_minus.type == flow_boundary_periodic && boundary.y_plus.type == flow_boundary_periodic;
        return boundary.z_minus.type == flow_boundary_periodic && boundary.z_plus.type == flow_boundary_periodic;
    }

    __device__ inline bool scalar_periodic_pair(const ScalarBoundary boundary, const std::uint32_t dimension) {
        if (dimension == 0u) return boundary.x_minus.type == scalar_boundary_periodic && boundary.x_plus.type == scalar_boundary_periodic;
        if (dimension == 1u) return boundary.y_minus.type == scalar_boundary_periodic && boundary.y_plus.type == scalar_boundary_periodic;
        return boundary.z_minus.type == scalar_boundary_periodic && boundary.z_plus.type == scalar_boundary_periodic;
    }

    __device__ inline float flow_face_velocity(const FlowBoundaryFace face, const std::uint32_t axis) {
        if (axis == 0u) return face.velocity_x;
        if (axis == 1u) return face.velocity_y;
        return face.velocity_z;
    }

    __device__ inline bool resolve_cell_coordinates(int& x, int& y, int& z, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        if (flow_periodic_pair(boundary, 0u) && nx > 0) x = wrap_index(x, nx);
        if (flow_periodic_pair(boundary, 1u) && ny > 0) y = wrap_index(y, ny);
        if (flow_periodic_pair(boundary, 2u) && nz > 0) z = wrap_index(z, nz);
        return cell_in_bounds(x, y, z, nx, ny, nz);
    }

    __device__ inline bool resolve_scalar_cell_coordinates(int& x, int& y, int& z, const int nx, const int ny, const int nz, const ScalarBoundary boundary) {
        if (scalar_periodic_pair(boundary, 0u) && nx > 0) x = wrap_index(x, nx);
        if (scalar_periodic_pair(boundary, 1u) && ny > 0) y = wrap_index(y, ny);
        if (scalar_periodic_pair(boundary, 2u) && nz > 0) z = wrap_index(z, nz);
        return cell_in_bounds(x, y, z, nx, ny, nz);
    }

    __device__ inline bool cell_is_marked(const std::uint32_t* cell_indices, int x, int y, int z, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return true;
        return cell_indices[field::index(x, y, z, nx, ny)] != 0u;
    }

    __device__ inline float load_flow_cell(const float* values, int x, int y, int z, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        if (x < 0 || x >= nx) {
            if (flow_periodic_pair(boundary, 0u) && nx > 0)
                x = wrap_index(x, nx);
            else
                x = x < 0 ? 0 : nx - 1;
        }
        if (y < 0 || y >= ny) {
            if (flow_periodic_pair(boundary, 1u) && ny > 0)
                y = wrap_index(y, ny);
            else
                y = y < 0 ? 0 : ny - 1;
        }
        if (z < 0 || z >= nz) {
            if (flow_periodic_pair(boundary, 2u) && nz > 0)
                z = wrap_index(z, nz);
            else
                z = z < 0 ? 0 : nz - 1;
        }
        return values[field::index(x, y, z, nx, ny)];
    }

    __device__ inline float load_center_velocity_component(const float* values, const int component_axis, int x, int y, int z, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        if (x < 0 || x >= nx) {
            const auto face = x < 0 ? boundary.x_minus : boundary.x_plus;
            if (flow_periodic_pair(boundary, 0u) && nx > 0) {
                x = wrap_index(x, nx);
            } else {
                const float interior = values[field::index(x < 0 ? 0 : nx - 1, clamp_int(y, 0, ny - 1), clamp_int(z, 0, nz - 1), nx, ny)];
                float prescribed     = 0.0f;
                if (component_axis == 0) prescribed = face.velocity_x;
                if (component_axis == 1) prescribed = face.velocity_y;
                if (component_axis == 2) prescribed = face.velocity_z;
                if (face.type == flow_boundary_outflow) return interior;
                if (face.type == flow_boundary_free_slip_wall && component_axis != 0) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        if (y < 0 || y >= ny) {
            const auto face = y < 0 ? boundary.y_minus : boundary.y_plus;
            if (flow_periodic_pair(boundary, 1u) && ny > 0) {
                y = wrap_index(y, ny);
            } else {
                const float interior = values[field::index(clamp_int(x, 0, nx - 1), y < 0 ? 0 : ny - 1, clamp_int(z, 0, nz - 1), nx, ny)];
                float prescribed     = 0.0f;
                if (component_axis == 0) prescribed = face.velocity_x;
                if (component_axis == 1) prescribed = face.velocity_y;
                if (component_axis == 2) prescribed = face.velocity_z;
                if (face.type == flow_boundary_outflow) return interior;
                if (face.type == flow_boundary_free_slip_wall && component_axis != 1) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        if (z < 0 || z >= nz) {
            const auto face = z < 0 ? boundary.z_minus : boundary.z_plus;
            if (flow_periodic_pair(boundary, 2u) && nz > 0) {
                z = wrap_index(z, nz);
            } else {
                const float interior = values[field::index(clamp_int(x, 0, nx - 1), clamp_int(y, 0, ny - 1), z < 0 ? 0 : nz - 1, nx, ny)];
                float prescribed     = 0.0f;
                if (component_axis == 0) prescribed = face.velocity_x;
                if (component_axis == 1) prescribed = face.velocity_y;
                if (component_axis == 2) prescribed = face.velocity_z;
                if (face.type == flow_boundary_outflow) return interior;
                if (face.type == flow_boundary_free_slip_wall && component_axis != 2) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        return values[field::index(x, y, z, nx, ny)];
    }

    __device__ inline float constraint_velocity_value(const float* constraint_velocity, const std::uint32_t* cell_indices, int x, int y, int z, const int nx, const int ny, const int nz, const FlowBoundary boundary) {
        if (constraint_velocity == nullptr) return 0.0f;
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return 0.0f;
        if (cell_indices[field::index(x, y, z, nx, ny)] == 0u) return 0.0f;
        return constraint_velocity[field::index(x, y, z, nx, ny)];
    }

    void enforce_staggered_boundary(cudaStream_t stream, std::uint32_t axis, float* velocity_component, const std::uint32_t* cell_indices, const float* solid_velocity_component, int nx, int ny, int nz, const std::uint32_t* flow_types, const float* flow_velocity);
    void sync_periodic_staggered_component(cudaStream_t stream, std::uint32_t axis, float* velocity_component, int nx, int ny, int nz);
    void boundary_fill_centered_scalar(cudaStream_t stream, float* destination, const float* source, const std::uint32_t* cell_indices, int nx, int ny, int nz, const std::uint32_t* scalar_boundary_types, const float* scalar_boundary_values);
} // namespace kfs::cuda::boundary

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_BOUNDARY_H
