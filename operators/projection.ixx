module;
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

export module xayah.operators.projection;
import std;
import xayah.core.field;
import xayah.core.boundary;

export namespace xayah::operators {
    struct Projection final {
        Projection(cudaStream_t stream, std::array<std::int32_t, 3> resolution, float cell_size, const core::boundary::PackedScalarBoundary3D& pressure_boundary, std::int32_t pressure_iterations);
        ~Projection() noexcept;
        Projection(const Projection&)                = delete;
        Projection& operator=(const Projection&)     = delete;
        Projection(Projection&&) noexcept            = delete;
        Projection& operator=(Projection&&) noexcept = delete;

        std::int32_t pressure_iterations{0};

        void operator()(core::field::StaggeredVectorField3D& destination, core::field::StaggeredVectorField3D& working, const core::field::CenteredVectorField3D& constraint_velocity, const core::field::IndexedField3D& cell_indices, const core::boundary::PackedVectorBoundary3D& velocity_boundary, float delta_seconds);

    private:
        void initialize();
        void release() noexcept;

        CUstream_st* const stream{nullptr};
        const std::array<std::int32_t, 3> resolution{0, 0, 0};
        const float cell_size{0.0f};
        const core::boundary::PackedScalarBoundary3D pressure_boundary{};

        cublasHandle_t cublas{nullptr};
        cusparseHandle_t cusparse{nullptr};
        cusparseSpMatDescr_t pressure_matrix{nullptr};
        cusparseDnVecDescr_t pressure_vec_p{nullptr};
        cusparseDnVecDescr_t pressure_vec_ap{nullptr};
        std::size_t spmv_buffer_size{0};

        float* pressure{nullptr};
        float* pressure_rhs{nullptr};
        int* pressure_anchor{nullptr};
        int* pressure_row_offsets{nullptr};
        int* pressure_column_indices{nullptr};
        float* pressure_values{nullptr};
        float* pcg_r{nullptr};
        float* pcg_p{nullptr};
        float* pcg_ap{nullptr};
        float* pressure_dot_rz{nullptr};
        float* pressure_dot_pap{nullptr};
        float* pressure_dot_rr{nullptr};
        float* pressure_alpha{nullptr};
        float* pressure_negative_alpha{nullptr};
        float* pressure_beta{nullptr};
        float* pressure_one{nullptr};
        void* spmv_buffer{nullptr};
    };
} // namespace xayah::operators
