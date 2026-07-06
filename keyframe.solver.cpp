module;
#include "keyframe.solver.h"

#include "keyframe.field.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

module keyframe.solver;
import std;
import keyframe.field;

namespace kfs::solver {
    namespace {
        constexpr std::uint32_t flow_boundary_periodic = 3u;

        void write_flow_face(const std::size_t index, const FlowBoundaryFace& face, std::array<std::uint32_t, 6>& types, std::array<float, 18>& velocity, std::array<float, 6>& pressure) {
            types[index]              = static_cast<std::uint32_t>(face.type);
            velocity[index * 3u + 0u] = face.velocity_x;
            velocity[index * 3u + 1u] = face.velocity_y;
            velocity[index * 3u + 2u] = face.velocity_z;
            pressure[index]           = face.pressure;
        }

        void write_scalar_face(const std::size_t index, const ScalarBoundaryFace& face, std::array<std::uint32_t, 6>& types, std::array<float, 6>& values) {
            types[index]  = static_cast<std::uint32_t>(face.type);
            values[index] = face.value;
        }

        void write_flow_boundary(const FlowBoundary& boundary, std::array<std::uint32_t, 6>& types, std::array<float, 18>& velocity, std::array<float, 6>& pressure) {
            write_flow_face(0u, boundary.x_minus, types, velocity, pressure);
            write_flow_face(1u, boundary.x_plus, types, velocity, pressure);
            write_flow_face(2u, boundary.y_minus, types, velocity, pressure);
            write_flow_face(3u, boundary.y_plus, types, velocity, pressure);
            write_flow_face(4u, boundary.z_minus, types, velocity, pressure);
            write_flow_face(5u, boundary.z_plus, types, velocity, pressure);
        }

        void write_scalar_boundary(const ScalarBoundary& boundary, std::array<std::uint32_t, 6>& types, std::array<float, 6>& values) {
            write_scalar_face(0u, boundary.x_minus, types, values);
            write_scalar_face(1u, boundary.x_plus, types, values);
            write_scalar_face(2u, boundary.y_minus, types, values);
            write_scalar_face(3u, boundary.y_plus, types, values);
            write_scalar_face(4u, boundary.z_minus, types, values);
            write_scalar_face(5u, boundary.z_plus, types, values);
        }

        bool paired_periodic(const FlowBoundaryFace& minus_face, const FlowBoundaryFace& plus_face) {
            return (minus_face.type == FlowBoundaryType::periodic) == (plus_face.type == FlowBoundaryType::periodic);
        }

        bool paired_periodic(const ScalarBoundaryFace& minus_face, const ScalarBoundaryFace& plus_face) {
            return (minus_face.type == ScalarBoundaryType::periodic) == (plus_face.type == ScalarBoundaryType::periodic);
        }

        void validate_config(const Config& config) {
            if (config.resolution[0] == 0 || config.resolution[1] == 0 || config.resolution[2] == 0) throw std::runtime_error("Keyframe smoke resolution must be positive");
            if (config.resolution[0] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) || config.resolution[1] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) || config.resolution[2] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("Keyframe smoke resolution exceeds CUDA solver int range");
            if (config.cell_size <= 0.0f) throw std::runtime_error("Keyframe smoke cell_size must be positive");
            if (config.pressure_iterations <= 0) throw std::runtime_error{"Keyframe smoke pressure_iterations must be positive"};
            if (!paired_periodic(config.flow_boundary.x_minus, config.flow_boundary.x_plus) || !paired_periodic(config.flow_boundary.y_minus, config.flow_boundary.y_plus) || !paired_periodic(config.flow_boundary.z_minus, config.flow_boundary.z_plus)) throw std::runtime_error("Keyframe smoke flow periodic boundaries must be paired");
            if (!paired_periodic(config.density_boundary.x_minus, config.density_boundary.x_plus) || !paired_periodic(config.density_boundary.y_minus, config.density_boundary.y_plus) || !paired_periodic(config.density_boundary.z_minus, config.density_boundary.z_plus)) throw std::runtime_error("Keyframe smoke density periodic boundaries must be paired");
            if (!paired_periodic(config.temperature_boundary.x_minus, config.temperature_boundary.x_plus) || !paired_periodic(config.temperature_boundary.y_minus, config.temperature_boundary.y_plus) || !paired_periodic(config.temperature_boundary.z_minus, config.temperature_boundary.z_plus)) throw std::runtime_error("Keyframe smoke temperature periodic boundaries must be paired");

            const std::uint64_t cell_count = static_cast<std::uint64_t>(config.resolution[0]) * static_cast<std::uint64_t>(config.resolution[1]) * static_cast<std::uint64_t>(config.resolution[2]);
            if (cell_count == 0 || cell_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("Keyframe smoke cell count exceeds pressure solver int range");
        }

        void validate_source(const PlumeSource& source) {
            if (source.radius[0] <= 0.0f || source.radius[1] <= 0.0f || source.radius[2] <= 0.0f) throw std::runtime_error{"Keyframe smoke plume source radius must be positive"};
            if (source.density < 0.0f) throw std::runtime_error{"Keyframe smoke plume source density must be non-negative"};
            if (source.temperature < 0.0f) throw std::runtime_error{"Keyframe smoke plume source temperature must be non-negative"};
            if (source.falloff <= 0.0f) throw std::runtime_error{"Keyframe smoke plume source falloff must be positive"};
        }

        unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
            return static_cast<unsigned>((value + divisor - 1u) / divisor);
        }

        unsigned linear_launch_grid(const std::uint64_t count) {
            if (count == 0u) throw std::runtime_error{"Keyframe smoke launch count must be positive"};
            return ceil_div_u32(count, 256u);
        }

        dim3 scalar_launch_grid(const field::ScalarField3D& field) {
            if (field.count() == 0u) throw std::runtime_error{"Keyframe smoke scalar launch resolution is empty"};
            const dim3 block{8u, 8u, 4u};
            const auto& resolution = field.resolution;
            return dim3(ceil_div_u32(static_cast<std::uint64_t>(resolution[0]), block.x), ceil_div_u32(static_cast<std::uint64_t>(resolution[1]), block.y), ceil_div_u32(static_cast<std::uint64_t>(resolution[2]), block.z));
        }

        dim3 staggered_launch_grid(const field::StaggeredVectorField3D& field, const std::uint32_t axis) {
            if (field.count(axis) == 0u) throw std::runtime_error{"Keyframe smoke staggered launch resolution is empty"};
            const dim3 block{8u, 8u, 4u};
            const auto& resolution = field.resolution;
            const auto nx          = static_cast<std::uint64_t>(resolution[0]);
            const auto ny          = static_cast<std::uint64_t>(resolution[1]);
            const auto nz          = static_cast<std::uint64_t>(resolution[2]);
            if (axis == 0u) return dim3(ceil_div_u32(nx + 1u, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz, block.z));
            if (axis == 1u) return dim3(ceil_div_u32(nx, block.x), ceil_div_u32(ny + 1u, block.y), ceil_div_u32(nz, block.z));
            return dim3(ceil_div_u32(nx, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz + 1u, block.z));
        }

        int wrap_index(const int value, const int size) {
            const int remainder = value % size;
            return remainder < 0 ? remainder + size : remainder;
        }

        std::uint64_t index_3d(const int x, const int y, const int z, const int nx, const int ny) {
            return static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(nx) * (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(z));
        }

        std::array<dim3, 3> make_sync_velocity_grid(const std::int32_t nx, const std::int32_t ny, const std::int32_t nz, const dim3& block) {
            return {
                dim3(ceil_div_u32(static_cast<std::uint64_t>(ny), block.x), ceil_div_u32(static_cast<std::uint64_t>(nz), block.y), 1u),
                dim3(ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(nz), block.y), 1u),
                dim3(ceil_div_u32(static_cast<std::uint64_t>(nx), block.x), ceil_div_u32(static_cast<std::uint64_t>(ny), block.y), 1u),
            };
        }

        void add_pressure_column(std::array<int, 7>& row_columns, int& row_entry_count, const int column) {
            for (int entry = 0; entry < row_entry_count; ++entry) {
                if (row_columns[entry] == column) return;
            }
            row_columns[row_entry_count] = column;
            ++row_entry_count;
        }

        void add_pressure_neighbor(std::array<int, 7>& row_columns, int& row_entry_count, int next_x, int next_y, int next_z, const bool periodic_axis, const int nx, const int ny, const int nz) {
            if (next_x < 0 || next_x >= nx || next_y < 0 || next_y >= ny || next_z < 0 || next_z >= nz) {
                if (!periodic_axis) return;
                if (next_x < 0 || next_x >= nx) next_x = wrap_index(next_x, nx);
                if (next_y < 0 || next_y >= ny) next_y = wrap_index(next_y, ny);
                if (next_z < 0 || next_z >= nz) next_z = wrap_index(next_z, nz);
            }
            add_pressure_column(row_columns, row_entry_count, static_cast<int>(index_3d(next_x, next_y, next_z, nx, ny)));
        }

        void initialize_host(Solver::HostData& host, const Config& config) {
            host.nx                          = static_cast<std::int32_t>(config.resolution[0]);
            host.ny                          = static_cast<std::int32_t>(config.resolution[1]);
            host.nz                          = static_cast<std::int32_t>(config.resolution[2]);
            host.cell_size                   = config.cell_size;
            host.pressure_iterations         = config.pressure_iterations;
            host.ambient_temperature         = config.ambient_temperature;
            host.buoyancy_density_factor     = config.buoyancy_density_factor;
            host.buoyancy_temperature_factor = config.buoyancy_temperature_factor;
            host.vorticity_confinement       = config.vorticity_confinement;
            host.scalar_advection_mode       = static_cast<std::uint32_t>(config.scalar_advection_mode);
            write_flow_boundary(config.flow_boundary, host.flow_boundary_types, host.flow_boundary_velocity, host.flow_boundary_pressure);
            write_scalar_boundary(config.density_boundary, host.density_boundary_types, host.density_boundary_values);
            write_scalar_boundary(config.temperature_boundary, host.temperature_boundary_types, host.temperature_boundary_values);
            const auto cell_count = static_cast<std::uint64_t>(host.nx) * static_cast<std::uint64_t>(host.ny) * static_cast<std::uint64_t>(host.nz);
            host.density_source.resize(cell_count, 0.0f);
            host.temperature_source.resize(cell_count, 0.0f);
        }

        void initialize_field_buffers(const Solver::HostData& host, Solver::DeviceData& device) {
            device.density_data.fill(host.stream, 0.0f);
            device.density_temp.fill(host.stream, 0.0f);
            device.density_source.fill(host.stream, 0.0f);
            device.temperature_data.fill(host.stream, host.ambient_temperature);
            device.temperature_temp.fill(host.stream, host.ambient_temperature);
            device.temperature_source.fill(host.stream, 0.0f);
            device.force.fill(host.stream, 0.0f);
            device.solid_velocity.fill(host.stream, 0.0f);
            device.velocity.fill(host.stream, 0.0f);
            device.temp_velocity.fill(host.stream, 0.0f);
            device.centered_velocity.fill(host.stream, 0.0f);
            device.vorticity.fill(host.stream, 0.0f);
            device.vorticity_magnitude.fill(host.stream, 0.0f);
            device.solid_temperature.fill(host.stream, host.ambient_temperature);
            if (const cudaError_t status = cudaMemsetAsync(device.occupancy, 0, device.density_data.count() * sizeof(std::uint8_t), host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync occupancy: "} + cudaGetErrorString(status)};
        }

        void destroy_device(Solver& smoke) noexcept {
            try {
                if (smoke.host.stream != nullptr) cudaStreamSynchronize(smoke.host.stream);
                if (smoke.host.pressure_matrix != nullptr) cusparseDestroySpMat(smoke.host.pressure_matrix);
                if (smoke.host.pressure_vec_p != nullptr) cusparseDestroyDnVec(smoke.host.pressure_vec_p);
                if (smoke.host.pressure_vec_ap != nullptr) cusparseDestroyDnVec(smoke.host.pressure_vec_ap);
                if (smoke.host.cublas != nullptr) cublasDestroy(smoke.host.cublas);
                if (smoke.host.cusparse != nullptr) cusparseDestroy(smoke.host.cusparse);
                cuda::free_device_buffers(smoke.device.occupancy, smoke.device.pressure, smoke.device.pressure_rhs, smoke.device.pressure_anchor, smoke.device.pressure_row_offsets, smoke.device.pressure_column_indices, smoke.device.pressure_values, smoke.device.pcg_r, smoke.device.pcg_p, smoke.device.pcg_ap, smoke.device.pressure_dot_rz, smoke.device.pressure_dot_pap, smoke.device.pressure_dot_rr, smoke.device.pressure_alpha, smoke.device.pressure_negative_alpha, smoke.device.pressure_beta, smoke.device.pressure_one, smoke.device.spmv_buffer);
                if (smoke.host.stream != nullptr) cudaStreamDestroy(smoke.host.stream);
            } catch (...) {
            }
            smoke.host.stream           = nullptr;
            smoke.host.cublas           = nullptr;
            smoke.host.cusparse         = nullptr;
            smoke.host.pressure_matrix  = nullptr;
            smoke.host.pressure_vec_p   = nullptr;
            smoke.host.pressure_vec_ap  = nullptr;
            smoke.host.spmv_buffer_size = 0;
            smoke.device                = {};
        }

        void initialize_pressure_system(Solver& smoke) {
            const auto& host      = smoke.host;
            const auto cell_count = smoke.device.density_data.count();
            const auto cell_bytes = smoke.device.density_data.bytes();
            const int cells       = static_cast<int>(cell_count);
            const bool periodic_x = host.flow_boundary_types[0] == flow_boundary_periodic && host.flow_boundary_types[1] == flow_boundary_periodic;
            const bool periodic_y = host.flow_boundary_types[2] == flow_boundary_periodic && host.flow_boundary_types[3] == flow_boundary_periodic;
            const bool periodic_z = host.flow_boundary_types[4] == flow_boundary_periodic && host.flow_boundary_types[5] == flow_boundary_periodic;
            std::vector<int> host_row_offsets(static_cast<std::size_t>(cells) + 1u, 0);
            std::vector<int> host_column_indices{};
            host_column_indices.reserve(static_cast<std::size_t>(cells) * 7u);

            for (int row = 0; row < cells; ++row) {
                host_row_offsets[static_cast<std::size_t>(row)] = static_cast<int>(host_column_indices.size());
                const int x                                     = row % host.nx;
                const int yz                                    = row / host.nx;
                const int y                                     = yz % host.ny;
                const int z                                     = yz / host.ny;
                std::array<int, 7> row_columns{};
                int row_entry_count = 0;
                add_pressure_neighbor(row_columns, row_entry_count, x - 1, y, z, periodic_x, host.nx, host.ny, host.nz);
                add_pressure_neighbor(row_columns, row_entry_count, x + 1, y, z, periodic_x, host.nx, host.ny, host.nz);
                add_pressure_neighbor(row_columns, row_entry_count, x, y - 1, z, periodic_y, host.nx, host.ny, host.nz);
                add_pressure_neighbor(row_columns, row_entry_count, x, y + 1, z, periodic_y, host.nx, host.ny, host.nz);
                add_pressure_neighbor(row_columns, row_entry_count, x, y, z - 1, periodic_z, host.nx, host.ny, host.nz);
                add_pressure_neighbor(row_columns, row_entry_count, x, y, z + 1, periodic_z, host.nx, host.ny, host.nz);
                add_pressure_column(row_columns, row_entry_count, row);
                for (int left = 0; left < row_entry_count; ++left) {
                    for (int right = left + 1; right < row_entry_count; ++right) {
                        if (row_columns[right] < row_columns[left]) {
                            const int swapped_column = row_columns[left];
                            row_columns[left]        = row_columns[right];
                            row_columns[right]       = swapped_column;
                        }
                    }
                }
                for (int entry = 0; entry < row_entry_count; ++entry) {
                    host_column_indices.push_back(row_columns[entry]);
                }
            }

            host_row_offsets[static_cast<std::size_t>(cells)] = static_cast<int>(host_column_indices.size());
            const int pressure_nnz                            = static_cast<int>(host_column_indices.size());
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_anchor), sizeof(int)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_anchor: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_row_offsets), static_cast<std::size_t>(cells + 1) * sizeof(int)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_row_offsets: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_column_indices), static_cast<std::size_t>(pressure_nnz) * sizeof(int)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_column_indices: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_values), static_cast<std::size_t>(pressure_nnz) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_values: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemcpyAsync(smoke.device.pressure_row_offsets, host_row_offsets.data(), static_cast<std::size_t>(cells + 1) * sizeof(int), cudaMemcpyHostToDevice, smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync pressure_row_offsets: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemcpyAsync(smoke.device.pressure_column_indices, host_column_indices.data(), static_cast<std::size_t>(pressure_nnz) * sizeof(int), cudaMemcpyHostToDevice, smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync pressure_column_indices: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemsetAsync(smoke.device.pressure_values, 0, static_cast<std::size_t>(pressure_nnz) * sizeof(float), smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync pressure_values: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pcg_r), cell_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pcg_r: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pcg_p), cell_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pcg_p: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pcg_ap), cell_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pcg_ap: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_dot_rz), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_dot_rz: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_dot_pap), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_dot_pap: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_dot_rr), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_dot_rr: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_alpha), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_alpha: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_negative_alpha), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_negative_alpha: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_beta), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_beta: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&smoke.device.pressure_one), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_one: "} + cudaGetErrorString(status)};
            constexpr float one = 1.0f;
            if (const cudaError_t status = cudaMemcpyAsync(smoke.device.pressure_one, &one, sizeof(float), cudaMemcpyHostToDevice, smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync pressure_one: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemsetAsync(smoke.device.pcg_r, 0, cell_bytes, smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync pcg_r: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemsetAsync(smoke.device.pcg_p, 0, cell_bytes, smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync pcg_p: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaMemsetAsync(smoke.device.pcg_ap, 0, cell_bytes, smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync pcg_ap: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaStreamSynchronize(smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize pressure_system_upload: "} + cudaGetErrorString(status)};
            if (const cusparseStatus_t status = cusparseCreateCsr(&smoke.host.pressure_matrix, cells, cells, pressure_nnz, smoke.device.pressure_row_offsets, smoke.device.pressure_column_indices, smoke.device.pressure_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreateCsr matrix"};
            if (const cusparseStatus_t status = cusparseCreateDnVec(&smoke.host.pressure_vec_p, cells, smoke.device.pcg_p, CUDA_R_32F); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreateDnVec vec_p"};
            if (const cusparseStatus_t status = cusparseCreateDnVec(&smoke.host.pressure_vec_ap, cells, smoke.device.pcg_ap, CUDA_R_32F); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreateDnVec vec_ap"};
            constexpr float spmv_alpha = 1.0f;
            constexpr float spmv_beta  = 0.0f;
            if (const cusparseStatus_t status = cusparseSpMV_bufferSize(smoke.host.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, smoke.host.pressure_matrix, smoke.host.pressure_vec_p, &spmv_beta, smoke.host.pressure_vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &smoke.host.spmv_buffer_size); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSpMV_bufferSize"};
            if (smoke.host.spmv_buffer_size > 0) {
                if (const cudaError_t status = cudaMalloc(&smoke.device.spmv_buffer, smoke.host.spmv_buffer_size); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc spmv_buffer: "} + cudaGetErrorString(status)};
            }
            if (const cusparseStatus_t status = cusparseSpMV_preprocess(smoke.host.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, smoke.host.pressure_matrix, smoke.host.pressure_vec_p, &spmv_beta, smoke.host.pressure_vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, smoke.device.spmv_buffer); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSpMV_preprocess"};
        }

        void create_device(Solver& smoke) {
            if (smoke.host.stream != nullptr) throw std::runtime_error{"Keyframe smoke device is already initialized"};
            auto& host   = smoke.host;
            auto& device = smoke.device;

            try {
                if (const cudaError_t status = cudaStreamCreateWithFlags(&host.stream, cudaStreamNonBlocking); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamCreateWithFlags: "} + cudaGetErrorString(status)};
                if (const cublasStatus_t status = cublasCreate(&host.cublas); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasCreate"};
                if (const cublasStatus_t status = cublasSetStream(host.cublas, host.stream); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSetStream"};
                if (const cublasStatus_t status = cublasSetPointerMode(host.cublas, CUBLAS_POINTER_MODE_DEVICE); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSetPointerMode"};
                if (const cusparseStatus_t status = cusparseCreate(&host.cusparse); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreate"};
                if (const cusparseStatus_t status = cusparseSetStream(host.cusparse, host.stream); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSetStream"};

                device = Solver::DeviceData{};
                const std::array resolution{host.nx, host.ny, host.nz};
                device.density_data.resize(resolution);
                device.density_temp.resize(resolution);
                device.density_source.resize(resolution);
                device.temperature_data.resize(resolution);
                device.temperature_temp.resize(resolution);
                device.temperature_source.resize(resolution);
                device.force.resize(resolution);
                device.solid_velocity.resize(resolution);
                device.velocity.resize(resolution);
                device.temp_velocity.resize(resolution);
                device.centered_velocity.resize(resolution);
                device.vorticity.resize(resolution);
                device.vorticity_magnitude.resize(resolution);
                device.solid_temperature.resize(resolution);
                const auto cell_count = device.density_data.count();
                const auto cell_bytes = device.density_data.bytes();
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device.occupancy), static_cast<std::size_t>(cell_count) * sizeof(std::uint8_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc occupancy: "} + cudaGetErrorString(status)};
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device.pressure), cell_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure: "} + cudaGetErrorString(status)};
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device.pressure_rhs), cell_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pressure_rhs: "} + cudaGetErrorString(status)};

                initialize_field_buffers(host, device);
                if (const cudaError_t status = cudaMemsetAsync(device.pressure, 0, cell_bytes, host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync pressure: "} + cudaGetErrorString(status)};
                if (const cudaError_t status = cudaMemsetAsync(device.pressure_rhs, 0, cell_bytes, host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync pressure_rhs: "} + cudaGetErrorString(status)};
                initialize_pressure_system(smoke);
            } catch (...) {
                destroy_device(smoke);
                throw;
            }
        }

        void solve_pressure(Solver& smoke, const float delta_seconds) {
            const auto& host                = smoke.host;
            const auto& device              = smoke.device;
            const auto cell_count           = device.density_data.count();
            const auto cell_bytes           = device.density_data.bytes();
            const int cell_count_int        = static_cast<int>(cell_count);
            const std::uint32_t* flow_types = host.flow_boundary_types.data();
            const float* flow_velocity      = host.flow_boundary_velocity.data();
            const float* flow_pressure      = host.flow_boundary_pressure.data();
            const dim3 block{8u, 8u, 4u};
            cuda::fill_int(smoke.host.stream, 1u, 1u, smoke.device.pressure_anchor, cell_count_int, 1u);
            cuda::find_pressure_anchor(smoke.host.stream, linear_launch_grid(cell_count), 256u, smoke.device.pressure_anchor, device.occupancy, cell_count);
            cuda::compute_projection_rhs(smoke.host.stream, scalar_launch_grid(device.density_data), block, smoke.device.pressure_rhs, device.temp_velocity.data[0], device.temp_velocity.data[1], device.temp_velocity.data[2], device.occupancy, smoke.device.pressure_anchor, host.nx, host.ny, host.nz, host.cell_size, delta_seconds, flow_types, flow_velocity, flow_pressure);
            cuda::build_projection_matrix(smoke.host.stream, linear_launch_grid(cell_count), 256u, smoke.device.pressure_values, smoke.device.pressure_row_offsets, smoke.device.pressure_column_indices, device.occupancy, smoke.device.pressure_anchor, host.nx, host.ny, host.nz, flow_types, flow_velocity, flow_pressure);
            if (const cudaError_t status = cudaMemsetAsync(smoke.device.pressure, 0, cell_bytes, smoke.host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync pressure: "} + cudaGetErrorString(status)};
            if (const cublasStatus_t status = cublasScopy(smoke.host.cublas, cell_count_int, smoke.device.pressure_rhs, 1, smoke.device.pcg_r, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasScopy rhs"};
            if (const cublasStatus_t status = cublasScopy(smoke.host.cublas, cell_count_int, smoke.device.pcg_r, 1, smoke.device.pcg_p, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasScopy pcg_p"};
            if (const cublasStatus_t status = cublasSdot(smoke.host.cublas, cell_count_int, smoke.device.pcg_r, 1, smoke.device.pcg_r, 1, smoke.device.pressure_dot_rz); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSdot pressure_dot_rz"};
            constexpr float one  = 1.0f;
            constexpr float zero = 0.0f;
            for (int iteration = 0; iteration < smoke.host.pressure_iterations; ++iteration) {
                if (const cusparseStatus_t status = cusparseSpMV(smoke.host.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smoke.host.pressure_matrix, smoke.host.pressure_vec_p, &zero, smoke.host.pressure_vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, smoke.device.spmv_buffer); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSpMV"};
                if (const cublasStatus_t status = cublasSdot(smoke.host.cublas, cell_count_int, smoke.device.pcg_p, 1, smoke.device.pcg_ap, 1, smoke.device.pressure_dot_pap); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSdot pressure_dot_pap"};
                cuda::compute_ratio(smoke.host.stream, smoke.device.pressure_alpha, smoke.device.pressure_dot_rz, smoke.device.pressure_dot_pap);
                if (const cublasStatus_t status = cublasSaxpy(smoke.host.cublas, cell_count_int, smoke.device.pressure_alpha, smoke.device.pcg_p, 1, smoke.device.pressure, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSaxpy pressure"};
                cuda::negate_scalar(smoke.host.stream, smoke.device.pressure_negative_alpha, smoke.device.pressure_alpha);
                if (const cublasStatus_t status = cublasSaxpy(smoke.host.cublas, cell_count_int, smoke.device.pressure_negative_alpha, smoke.device.pcg_ap, 1, smoke.device.pcg_r, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSaxpy pcg_r"};
                if (const cublasStatus_t status = cublasSdot(smoke.host.cublas, cell_count_int, smoke.device.pcg_r, 1, smoke.device.pcg_r, 1, smoke.device.pressure_dot_rr); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSdot rho_new"};
                cuda::compute_ratio(smoke.host.stream, smoke.device.pressure_beta, smoke.device.pressure_dot_rr, smoke.device.pressure_dot_rz);
                if (const cublasStatus_t status = cublasSscal(smoke.host.cublas, cell_count_int, smoke.device.pressure_beta, smoke.device.pcg_p, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSscal pcg_p"};
                if (const cublasStatus_t status = cublasSaxpy(smoke.host.cublas, cell_count_int, smoke.device.pressure_one, smoke.device.pcg_r, 1, smoke.device.pcg_p, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSaxpy pcg_p"};
                if (const cublasStatus_t status = cublasScopy(smoke.host.cublas, 1, smoke.device.pressure_dot_rr, 1, smoke.device.pressure_dot_rz, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasScopy rho"};
            }
        }
    } // namespace

    Solver::Solver(const Config& config) {
        try {
            validate_config(config);
            this->host   = {};
            this->device = {};
            initialize_host(this->host, config);
            create_device(*this);
            this->set_plume_source(this->host.plume_source);
        } catch (...) {
            destroy_device(*this);
            throw;
        }
    }

    Solver::~Solver() noexcept {
        destroy_device(*this);
    }

    void Solver::set_plume_source(const PlumeSource& source) {
        validate_source(source);
        this->host.plume_source = source;
        std::ranges::fill(this->host.density_source, 0.0f);
        std::ranges::fill(this->host.temperature_source, 0.0f);

        const auto nx = static_cast<std::uint32_t>(this->host.nx);
        const auto ny = static_cast<std::uint32_t>(this->host.ny);
        const auto nz = static_cast<std::uint32_t>(this->host.nz);
        const std::array extent{
            static_cast<float>(nx) * this->host.cell_size,
            static_cast<float>(ny) * this->host.cell_size,
            static_cast<float>(nz) * this->host.cell_size,
        };
        const std::array center{
            source.center[0] * extent[0],
            source.center[1] * extent[1],
            source.center[2] * extent[2],
        };
        const std::array radius{
            source.radius[0] * extent[0],
            source.radius[1] * extent[1],
            source.radius[2] * extent[2],
        };

        for (std::uint32_t z = 0; z < nz; ++z) {
            for (std::uint32_t y = 0; y < ny; ++y) {
                for (std::uint32_t x = 0; x < nx; ++x) {
                    const std::size_t index = static_cast<std::size_t>(x) + static_cast<std::size_t>(nx) * (static_cast<std::size_t>(y) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(z));
                    const float px          = (static_cast<float>(x) + 0.5f) * this->host.cell_size;
                    const float py          = (static_cast<float>(y) + 0.5f) * this->host.cell_size;
                    const float pz          = (static_cast<float>(z) + 0.5f) * this->host.cell_size;
                    const float dx          = (px - center[0]) / radius[0];
                    const float dy          = (py - center[1]) / radius[1];
                    const float dz          = (pz - center[2]) / radius[2];
                    const float r2          = dx * dx + dy * dy + dz * dz;
                    if (r2 > 1.0f) continue;
                    const float plume                    = std::exp(-source.falloff * r2);
                    this->host.density_source[index]     = source.density * plume;
                    this->host.temperature_source[index] = source.temperature * plume;
                }
            }
        }

        const auto source_bytes = this->device.density_source.bytes();
        if (const cudaError_t status = cudaMemcpyAsync(this->device.density_source.data, this->host.density_source.data(), source_bytes, cudaMemcpyHostToDevice, this->host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync density_source: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpyAsync(this->device.temperature_source.data, this->host.temperature_source.data(), source_bytes, cudaMemcpyHostToDevice, this->host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync temperature_source: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaStreamSynchronize(this->host.stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize plume source: "} + cudaGetErrorString(status)};
    }

    std::expected<StepStats, std::string> Solver::step(const StepRequest& request) {
        try {
            if (!std::isfinite(request.delta_seconds) || request.delta_seconds < 0.0f) throw std::runtime_error{"Keyframe smoke delta_seconds must be finite and non-negative"};
            if (request.iterations < 1) throw std::runtime_error{"Keyframe smoke step iterations must be positive"};
            auto& host             = this->host;
            auto& device           = this->device;
            const auto cell_count  = device.density_data.count();
            const auto linear_grid = linear_launch_grid(cell_count);
            const dim3 block{8u, 8u, 4u};
            const auto cell_grid = scalar_launch_grid(device.density_data);
            const dim3 sync_block{block.x, block.y, 1u};
            const auto sync_velocity_grid   = make_sync_velocity_grid(host.nx, host.ny, host.nz, block);
            const std::uint32_t* flow_types = host.flow_boundary_types.data();
            const float* flow_velocity      = host.flow_boundary_velocity.data();
            const float* flow_pressure      = host.flow_boundary_pressure.data();
            const std::array<bool, 3> periodic{
                host.flow_boundary_types[0] == flow_boundary_periodic && host.flow_boundary_types[1] == flow_boundary_periodic,
                host.flow_boundary_types[2] == flow_boundary_periodic && host.flow_boundary_types[3] == flow_boundary_periodic,
                host.flow_boundary_types[4] == flow_boundary_periodic && host.flow_boundary_types[5] == flow_boundary_periodic,
            };
            const auto step_start     = std::chrono::steady_clock::now();
            const float delta_seconds = request.delta_seconds;
            if (delta_seconds > 0.0f) {
                for (std::int32_t iteration = 0; iteration < request.iterations; ++iteration) {
                    cuda::apply_solid_scalar(host.stream, linear_grid, 256u, device.temperature_data.data, device.occupancy, device.solid_temperature.data, host.nx, host.ny, host.nz, host.ambient_temperature);
                    field::center_staggered(host.stream, device.centered_velocity, device.velocity);
                    cuda::compute_vorticity(host.stream, cell_grid, block, device.vorticity.data[0], device.vorticity.data[1], device.vorticity.data[2], device.vorticity_magnitude.data, device.centered_velocity.data[0], device.centered_velocity.data[1], device.centered_velocity.data[2], device.occupancy, host.nx, host.ny, host.nz, host.cell_size, flow_types, flow_velocity, flow_pressure);
                    device.force.fill(host.stream, 0.0f);
                    cuda::add_buoyancy(host.stream, cell_grid, block, device.force.data[1], device.density_data.data, device.temperature_data.data, device.occupancy, host.nx, host.ny, host.nz, host.ambient_temperature, host.buoyancy_density_factor, host.buoyancy_temperature_factor, flow_types, flow_velocity, flow_pressure);
                    cuda::add_vorticity_confinement(host.stream, cell_grid, block, device.force.data[0], device.force.data[1], device.force.data[2], device.vorticity.data[0], device.vorticity.data[1], device.vorticity.data[2], device.vorticity_magnitude.data, device.occupancy, host.nx, host.ny, host.nz, host.cell_size, host.vorticity_confinement, flow_types, flow_velocity, flow_pressure);
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        const dim3 velocity_grid = staggered_launch_grid(device.velocity, axis);
                        field::add_centered_to_staggered(host.stream, device.velocity, axis, device.force, delta_seconds);
                        cuda::enforce_staggered_boundary(host.stream, velocity_grid, block, axis, device.velocity.data[axis], device.occupancy, device.solid_velocity.data[axis], host.nx, host.ny, host.nz, flow_types, flow_velocity, flow_pressure);
                        if (periodic[axis]) cuda::sync_periodic_staggered_component(host.stream, sync_velocity_grid[axis], sync_block, axis, device.velocity.data[axis], host.nx, host.ny, host.nz);
                    }
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        const dim3 velocity_grid = staggered_launch_grid(device.velocity, axis);
                        cuda::advect_staggered_component(host.stream, velocity_grid, block, axis, device.temp_velocity.data[axis], device.velocity.data[axis], device.velocity.data[0], device.velocity.data[1], device.velocity.data[2], device.occupancy, host.nx, host.ny, host.nz, host.cell_size, delta_seconds, host.scalar_advection_mode, flow_types, flow_velocity, flow_pressure);
                        cuda::enforce_staggered_boundary(host.stream, velocity_grid, block, axis, device.temp_velocity.data[axis], device.occupancy, device.solid_velocity.data[axis], host.nx, host.ny, host.nz, flow_types, flow_velocity, flow_pressure);
                        if (periodic[axis]) cuda::sync_periodic_staggered_component(host.stream, sync_velocity_grid[axis], sync_block, axis, device.temp_velocity.data[axis], host.nx, host.ny, host.nz);
                    }
                    solve_pressure(*this, delta_seconds);
                    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                        const dim3 velocity_grid = staggered_launch_grid(device.temp_velocity, axis);
                        cuda::project_staggered_component(host.stream, velocity_grid, block, axis, device.temp_velocity.data[axis], device.pressure, device.occupancy, device.solid_velocity.data[axis], host.nx, host.ny, host.nz, host.cell_size, delta_seconds, flow_types, flow_velocity, flow_pressure);
                        cuda::enforce_staggered_boundary(host.stream, velocity_grid, block, axis, device.temp_velocity.data[axis], device.occupancy, device.solid_velocity.data[axis], host.nx, host.ny, host.nz, flow_types, flow_velocity, flow_pressure);
                        if (periodic[axis]) cuda::sync_periodic_staggered_component(host.stream, sync_velocity_grid[axis], sync_block, axis, device.temp_velocity.data[axis], host.nx, host.ny, host.nz);
                        field::copy_staggered_component(host.stream, device.velocity, axis, device.temp_velocity);
                    }
                    field::add_scaled(host.stream, device.temperature_temp, device.temperature_data, device.temperature_source, delta_seconds);
                    cuda::advect_centered_scalar(host.stream, cell_grid, block, device.temperature_data.data, device.temperature_temp.data, device.velocity.data[0], device.velocity.data[1], device.velocity.data[2], device.occupancy, host.nx, host.ny, host.nz, host.cell_size, delta_seconds, host.scalar_advection_mode, host.temperature_boundary_types.data(), host.temperature_boundary_values.data(), flow_types, flow_velocity, flow_pressure);
                    cuda::apply_solid_scalar(host.stream, linear_grid, 256u, device.temperature_data.data, device.occupancy, device.solid_temperature.data, host.nx, host.ny, host.nz, host.ambient_temperature);
                    field::add_scaled(host.stream, device.density_temp, device.density_data, device.density_source, delta_seconds);
                    cuda::advect_centered_scalar(host.stream, cell_grid, block, device.density_data.data, device.density_temp.data, device.velocity.data[0], device.velocity.data[1], device.velocity.data[2], device.occupancy, host.nx, host.ny, host.nz, host.cell_size, delta_seconds, host.scalar_advection_mode, host.density_boundary_types.data(), host.density_boundary_values.data(), flow_types, flow_velocity, flow_pressure);
                    cuda::boundary_fill_centered_scalar(host.stream, cell_grid, block, device.density_temp.data, device.density_data.data, device.occupancy, host.nx, host.ny, host.nz, host.density_boundary_types.data(), host.density_boundary_values.data());
                    field::copy(host.stream, device.density_data, device.density_temp);
                    ++this->host.current_step;
                }
            }
            const auto step_stop = std::chrono::steady_clock::now();
            return StepStats{
                .step       = this->host.current_step,
                .elapsed_ms = std::chrono::duration<float, std::milli>{step_stop - step_start}.count(),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace kfs::solver
