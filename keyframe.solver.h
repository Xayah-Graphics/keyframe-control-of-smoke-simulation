#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_SOLVER_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_SOLVER_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda {
    void apply_solid_scalar(cudaStream_t stream, float* scalar, const std::uint8_t* occupancy, const float* solid_scalar, int nx, int ny, int nz, float default_value);

    void compute_vorticity(cudaStream_t stream, float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* cell_x, const float* cell_y, const float* cell_z, const std::uint8_t* occupancy, int nx, int ny, int nz, float h, const std::uint32_t* flow_boundary_types, const float* flow_boundary_velocity);
    void add_buoyancy(cudaStream_t stream, float* force_y, const float* density, const float* temperature, const std::uint8_t* occupancy, int nx, int ny, int nz, float ambient_temperature, float density_factor, float temperature_factor, const std::uint32_t* flow_boundary_types);
    void add_vorticity_confinement(cudaStream_t stream, float* force_x, float* force_y, float* force_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const std::uint8_t* occupancy, int nx, int ny, int nz, float h, float epsilon, const std::uint32_t* flow_boundary_types);

    void enforce_staggered_boundary(cudaStream_t stream, std::uint32_t axis, float* velocity_component, const std::uint8_t* occupancy, const float* solid_velocity_component, int nx, int ny, int nz, const std::uint32_t* flow_boundary_types, const float* flow_boundary_velocity);
    void sync_periodic_staggered_component(cudaStream_t stream, std::uint32_t axis, float* velocity_component, int nx, int ny, int nz);

    void fill_int(cudaStream_t stream, int* field, int value, std::uint64_t count);
    void find_pressure_anchor(cudaStream_t stream, int* pressure_anchor, const std::uint8_t* occupancy, std::uint64_t count);
    void compute_projection_rhs(cudaStream_t stream, float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint8_t* occupancy, const int* pressure_anchor, int nx, int ny, int nz, float h, float dt, const std::uint32_t* flow_boundary_types, const float* flow_boundary_pressure);
    void build_projection_matrix(cudaStream_t stream, float* values, const int* row_offsets, const int* column_indices, const std::uint8_t* occupancy, const int* pressure_anchor, int nx, int ny, int nz, const std::uint32_t* flow_boundary_types);
    void compute_ratio(cudaStream_t stream, float* destination, const float* numerator, const float* denominator);
    void negate_scalar(cudaStream_t stream, float* destination, const float* source);
    void project_staggered_component(cudaStream_t stream, std::uint32_t axis, float* velocity_component, const float* pressure, const std::uint8_t* occupancy, const float* solid_velocity_component, int nx, int ny, int nz, float h, float dt, const std::uint32_t* flow_boundary_types, const float* flow_boundary_velocity);

    void boundary_fill_centered_scalar(cudaStream_t stream, float* destination, const float* source, const std::uint8_t* occupancy, int nx, int ny, int nz, const std::uint32_t* scalar_boundary_types, const float* scalar_boundary_values);
} // namespace kfs::cuda

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_SOLVER_H
