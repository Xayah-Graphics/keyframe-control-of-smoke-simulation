#ifndef KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_SOLVER_H
#define KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_SOLVER_H

#include <cstdint>
#include <cuda_runtime.h>

namespace kfs::cuda {
    void apply_solid_scalar(cudaStream_t stream, float* scalar, const std::uint8_t* occupancy, const float* solid_scalar, int nx, int ny, int nz, float default_value);

    void add_buoyancy(cudaStream_t stream, float* force_y, const float* density, const float* temperature, const std::uint8_t* occupancy, int nx, int ny, int nz, float ambient_temperature, float density_factor, float temperature_factor, const std::uint32_t* flow_types);
} // namespace kfs::cuda

#endif // KEYFRAME_CONTROL_OF_SMOKE_SIMULATION_SOLVER_H
