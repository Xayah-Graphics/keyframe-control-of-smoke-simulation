module;
#include "keyframe.operators.projection.h"

#include "../keyframe.field.h"
#include <cublas_v2.h>
#include <cusparse.h>

module keyframe.operators.projection;
import std;
import keyframe.field;
import keyframe.boundary;

namespace kfs::operators {
    namespace {
        std::uint64_t cell_count(const std::array<std::int32_t, 3> resolution) {
            return static_cast<std::uint64_t>(resolution[0]) * static_cast<std::uint64_t>(resolution[1]) * static_cast<std::uint64_t>(resolution[2]);
        }

        std::size_t cell_bytes(const std::array<std::int32_t, 3> resolution) {
            return static_cast<std::size_t>(cell_count(resolution) * sizeof(float));
        }

        int wrap_index(const int value, const int size) {
            const int remainder = value % size;
            return remainder < 0 ? remainder + size : remainder;
        }

        std::uint64_t index_3d(const int x, const int y, const int z, const int nx, const int ny) {
            return static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(nx) * (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(z));
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

    } // namespace

    Projection::Projection(const cudaStream_t stream, const std::array<std::int32_t, 3> resolution, const float cell_size, const std::int32_t pressure_iterations, const boundary::PackedFlowBoundary& boundary) : stream{stream}, resolution{resolution}, cell_size{cell_size}, pressure_iterations{pressure_iterations}, flow_boundary{boundary} {
        if (stream == nullptr) throw std::runtime_error{"Projection stream must not be null"};
        if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"Projection resolution must be positive"};
        const std::uint64_t cells64 = cell_count(resolution);
        if (cells64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) throw std::runtime_error{"Projection cell count exceeds cuBLAS int range"};
        if (cells64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 7)) throw std::runtime_error{"Projection pressure matrix may exceed cuSPARSE int range"};
        if (!std::isfinite(cell_size) || cell_size <= 0.0f) throw std::runtime_error{"Projection cell_size must be positive"};
        if (pressure_iterations <= 0) throw std::runtime_error{"Projection pressure_iterations must be positive"};

        try {
            this->initialize();
        } catch (...) {
            this->release();
            throw;
        }
    }

    Projection::~Projection() noexcept {
        this->release();
    }

    void Projection::initialize() {
        const std::uint64_t cells64 = cell_count(this->resolution);
        const int cells             = static_cast<int>(cells64);
        const std::size_t bytes     = cell_bytes(this->resolution);
        const int nx                = this->resolution[0];
        const int ny                = this->resolution[1];
        const int nz                = this->resolution[2];

        std::vector host_row_offsets(static_cast<std::size_t>(cells) + 1u, 0);
        std::vector<int> host_column_indices{};
        host_column_indices.reserve(static_cast<std::size_t>(cells) * 7u);

        for (int row = 0; row < cells; ++row) {
            host_row_offsets[static_cast<std::size_t>(row)] = static_cast<int>(host_column_indices.size());
            const int x                                     = row % nx;
            const int yz                                    = row / nx;
            const int y                                     = yz % ny;
            const int z                                     = yz / ny;
            std::array<int, 7> row_columns{};
            int row_entry_count = 0;
            add_pressure_neighbor(row_columns, row_entry_count, x - 1, y, z, this->flow_boundary.periodic[0], nx, ny, nz);
            add_pressure_neighbor(row_columns, row_entry_count, x + 1, y, z, this->flow_boundary.periodic[0], nx, ny, nz);
            add_pressure_neighbor(row_columns, row_entry_count, x, y - 1, z, this->flow_boundary.periodic[1], nx, ny, nz);
            add_pressure_neighbor(row_columns, row_entry_count, x, y + 1, z, this->flow_boundary.periodic[1], nx, ny, nz);
            add_pressure_neighbor(row_columns, row_entry_count, x, y, z - 1, this->flow_boundary.periodic[2], nx, ny, nz);
            add_pressure_neighbor(row_columns, row_entry_count, x, y, z + 1, this->flow_boundary.periodic[2], nx, ny, nz);
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
            for (int entry = 0; entry < row_entry_count; ++entry) host_column_indices.push_back(row_columns[entry]);
        }

        if (host_column_indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::runtime_error{"Projection pressure matrix nnz exceeds cuSPARSE int range"};
        host_row_offsets[static_cast<std::size_t>(cells)] = static_cast<int>(host_column_indices.size());
        const int pressure_nnz                            = static_cast<int>(host_column_indices.size());
        if (const cublasStatus_t status = cublasCreate(&this->cublas); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasCreate projection"};
        if (const cublasStatus_t status = cublasSetStream(this->cublas, this->stream); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSetStream projection"};
        if (const cublasStatus_t status = cublasSetPointerMode(this->cublas, CUBLAS_POINTER_MODE_DEVICE); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSetPointerMode projection"};
        if (const cusparseStatus_t status = cusparseCreate(&this->cusparse); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreate projection"};
        if (const cusparseStatus_t status = cusparseSetStream(this->cusparse, this->stream); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSetStream projection"};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure), bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_rhs), bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_rhs: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_anchor), sizeof(int)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_anchor: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_row_offsets), static_cast<std::size_t>(cells + 1) * sizeof(int)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_row_offsets: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_column_indices), static_cast<std::size_t>(pressure_nnz) * sizeof(int)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_column_indices: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_values), static_cast<std::size_t>(pressure_nnz) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_values: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pcg_r), bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pcg_r: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pcg_p), bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pcg_p: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pcg_ap), bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pcg_ap: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_dot_rz), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_dot_rz: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_dot_pap), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_dot_pap: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_dot_rr), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_dot_rr: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_alpha), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_alpha: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_negative_alpha), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_negative_alpha: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_beta), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_beta: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&this->pressure_one), sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection pressure_one: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpyAsync(this->pressure_row_offsets, host_row_offsets.data(), static_cast<std::size_t>(cells + 1) * sizeof(int), cudaMemcpyHostToDevice, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync projection pressure_row_offsets: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpyAsync(this->pressure_column_indices, host_column_indices.data(), static_cast<std::size_t>(pressure_nnz) * sizeof(int), cudaMemcpyHostToDevice, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync projection pressure_column_indices: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemsetAsync(this->pressure_values, 0, static_cast<std::size_t>(pressure_nnz) * sizeof(float), this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync projection pressure_values: "} + cudaGetErrorString(status)};
        constexpr float one = 1.0f;
        if (const cudaError_t status = cudaMemcpyAsync(this->pressure_one, &one, sizeof(float), cudaMemcpyHostToDevice, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync projection pressure_one: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemsetAsync(this->pressure, 0, bytes, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync projection pressure: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemsetAsync(this->pressure_rhs, 0, bytes, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync projection pressure_rhs: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemsetAsync(this->pcg_r, 0, bytes, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync projection pcg_r: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemsetAsync(this->pcg_p, 0, bytes, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync projection pcg_p: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemsetAsync(this->pcg_ap, 0, bytes, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync projection pcg_ap: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaStreamSynchronize(this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize projection initialize: "} + cudaGetErrorString(status)};
        if (const cusparseStatus_t status = cusparseCreateCsr(&this->pressure_matrix, cells, cells, pressure_nnz, this->pressure_row_offsets, this->pressure_column_indices, this->pressure_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreateCsr projection matrix"};
        if (const cusparseStatus_t status = cusparseCreateDnVec(&this->pressure_vec_p, cells, this->pcg_p, CUDA_R_32F); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreateDnVec projection vec_p"};
        if (const cusparseStatus_t status = cusparseCreateDnVec(&this->pressure_vec_ap, cells, this->pcg_ap, CUDA_R_32F); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseCreateDnVec projection vec_ap"};
        constexpr float spmv_alpha = 1.0f;
        constexpr float spmv_beta  = 0.0f;
        if (const cusparseStatus_t status = cusparseSpMV_bufferSize(this->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, this->pressure_matrix, this->pressure_vec_p, &spmv_beta, this->pressure_vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &this->spmv_buffer_size); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSpMV_bufferSize projection"};
        if (this->spmv_buffer_size > 0) {
            if (const cudaError_t status = cudaMalloc(&this->spmv_buffer, this->spmv_buffer_size); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc projection spmv_buffer: "} + cudaGetErrorString(status)};
        }
        if (const cusparseStatus_t status = cusparseSpMV_preprocess(this->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, this->pressure_matrix, this->pressure_vec_p, &spmv_beta, this->pressure_vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, this->spmv_buffer); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSpMV_preprocess projection"};
    }

    void Projection::release() noexcept {
        try {
            if (this->stream != nullptr) cudaStreamSynchronize(this->stream);
            if (this->pressure_matrix != nullptr) cusparseDestroySpMat(this->pressure_matrix);
            if (this->pressure_vec_p != nullptr) cusparseDestroyDnVec(this->pressure_vec_p);
            if (this->pressure_vec_ap != nullptr) cusparseDestroyDnVec(this->pressure_vec_ap);
            if (this->cublas != nullptr) cublasDestroy(this->cublas);
            if (this->cusparse != nullptr) cusparseDestroy(this->cusparse);
            cuda::free_device_buffers(this->pressure, this->pressure_rhs, this->pressure_anchor, this->pressure_row_offsets, this->pressure_column_indices, this->pressure_values, this->pcg_r, this->pcg_p, this->pcg_ap, this->pressure_dot_rz, this->pressure_dot_pap, this->pressure_dot_rr, this->pressure_alpha, this->pressure_negative_alpha, this->pressure_beta, this->pressure_one, this->spmv_buffer);
        } catch (...) {
        }
        this->cublas           = nullptr;
        this->cusparse         = nullptr;
        this->pressure_matrix  = nullptr;
        this->pressure_vec_p   = nullptr;
        this->pressure_vec_ap  = nullptr;
        this->spmv_buffer_size = 0;
    }

    void Projection::operator()(field::StaggeredVectorField3D& destination, field::StaggeredVectorField3D& working, const field::CenteredVectorField3D& constraint_velocity, const std::uint8_t* cell_mask, const float delta_seconds) {
        if (destination.resolution != this->resolution || working.resolution != this->resolution || constraint_velocity.resolution != this->resolution) throw std::runtime_error{"Projection field resolution mismatch"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
            if (working.count(axis) == 0u || working.data[axis] == nullptr) throw std::runtime_error{"working field component is empty"};
        }
        if (constraint_velocity.count() == 0u) throw std::runtime_error{"Projection constraint_velocity field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (constraint_velocity.data[axis] == nullptr) throw std::runtime_error{"Projection constraint_velocity field component is empty"};
        if (cell_mask == nullptr) throw std::runtime_error{"Projection cell_mask must not be null"};
        if (!std::isfinite(delta_seconds) || delta_seconds <= 0.0f) throw std::runtime_error{"Projection delta_seconds must be positive"};

        const std::uint64_t cells64     = cell_count(this->resolution);
        const int cells                 = static_cast<int>(cells64);
        const std::size_t bytes         = cell_bytes(this->resolution);
        const int nx                    = this->resolution[0];
        const int ny                    = this->resolution[1];
        const int nz                    = this->resolution[2];
        const std::uint32_t* flow_types = this->flow_boundary.types.data();
        const float* flow_velocity      = this->flow_boundary.velocity.data();
        const float* flow_pressure      = this->flow_boundary.pressure.data();
        cuda::operators::projection::reset_pressure_anchor(this->stream, this->pressure_anchor, cells);
        cuda::operators::projection::find_pressure_anchor(this->stream, this->pressure_anchor, cell_mask, cells64);
        cuda::operators::projection::compute_pressure_rhs(this->stream, this->pressure_rhs, working.data[0], working.data[1], working.data[2], cell_mask, this->pressure_anchor, nx, ny, nz, this->cell_size, delta_seconds, flow_types, flow_pressure);
        cuda::operators::projection::build_pressure_matrix(this->stream, this->pressure_values, this->pressure_row_offsets, this->pressure_column_indices, cell_mask, this->pressure_anchor, nx, ny, nz, flow_types);
        if (const cudaError_t status = cudaMemsetAsync(this->pressure, 0, bytes, this->stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync projection pressure: "} + cudaGetErrorString(status)};
        if (const cublasStatus_t status = cublasScopy(this->cublas, cells, this->pressure_rhs, 1, this->pcg_r, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasScopy projection rhs"};
        if (const cublasStatus_t status = cublasScopy(this->cublas, cells, this->pcg_r, 1, this->pcg_p, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasScopy projection pcg_p"};
        if (const cublasStatus_t status = cublasSdot(this->cublas, cells, this->pcg_r, 1, this->pcg_r, 1, this->pressure_dot_rz); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSdot projection pressure_dot_rz"};
        constexpr float one  = 1.0f;
        constexpr float zero = 0.0f;
        for (int iteration = 0; iteration < this->pressure_iterations; ++iteration) {
            if (const cusparseStatus_t status = cusparseSpMV(this->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, this->pressure_matrix, this->pressure_vec_p, &zero, this->pressure_vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, this->spmv_buffer); status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error{"cusparseSpMV projection"};
            if (const cublasStatus_t status = cublasSdot(this->cublas, cells, this->pcg_p, 1, this->pcg_ap, 1, this->pressure_dot_pap); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSdot projection pressure_dot_pap"};
            cuda::operators::projection::compute_ratio(this->stream, this->pressure_alpha, this->pressure_dot_rz, this->pressure_dot_pap);
            if (const cublasStatus_t status = cublasSaxpy(this->cublas, cells, this->pressure_alpha, this->pcg_p, 1, this->pressure, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSaxpy projection pressure"};
            cuda::operators::projection::negate_scalar(this->stream, this->pressure_negative_alpha, this->pressure_alpha);
            if (const cublasStatus_t status = cublasSaxpy(this->cublas, cells, this->pressure_negative_alpha, this->pcg_ap, 1, this->pcg_r, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSaxpy projection pcg_r"};
            if (const cublasStatus_t status = cublasSdot(this->cublas, cells, this->pcg_r, 1, this->pcg_r, 1, this->pressure_dot_rr); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSdot projection rho_new"};
            cuda::operators::projection::compute_ratio(this->stream, this->pressure_beta, this->pressure_dot_rr, this->pressure_dot_rz);
            if (const cublasStatus_t status = cublasSscal(this->cublas, cells, this->pressure_beta, this->pcg_p, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSscal projection pcg_p"};
            if (const cublasStatus_t status = cublasSaxpy(this->cublas, cells, this->pressure_one, this->pcg_r, 1, this->pcg_p, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasSaxpy projection pcg_p"};
            if (const cublasStatus_t status = cublasScopy(this->cublas, 1, this->pressure_dot_rr, 1, this->pressure_dot_rz, 1); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasScopy projection rho"};
        }

        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            cuda::operators::projection::project_staggered_component(this->stream, axis, working.data[axis], this->pressure, cell_mask, constraint_velocity.data[axis], nx, ny, nz, this->cell_size, delta_seconds, flow_types, flow_velocity);
            boundary::enforce_staggered_boundary(this->stream, axis, working, cell_mask, constraint_velocity, this->flow_boundary);
            if (this->flow_boundary.periodic[axis]) boundary::sync_periodic_staggered_component(this->stream, axis, working);
            field::copy_component(this->stream, destination, axis, working);
        }
    }
} // namespace kfs::operators
