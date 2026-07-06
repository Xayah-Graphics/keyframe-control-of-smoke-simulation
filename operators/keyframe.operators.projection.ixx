module;
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

export module keyframe.operators.projection;
import std;
import keyframe.field;

export namespace kfs::operators {
    struct Projection final {
        Projection(cudaStream_t stream, std::array<std::int32_t, 3> resolution, float cell_size, std::int32_t pressure_iterations, const std::uint32_t* flow_boundary_types, const float* flow_boundary_velocity, const float* flow_boundary_pressure);
        ~Projection() noexcept;
        Projection(const Projection&)                = delete;
        Projection& operator=(const Projection&)     = delete;
        Projection(Projection&&) noexcept            = delete;
        Projection& operator=(Projection&&) noexcept = delete;

        void operator()(field::StaggeredVectorField3D& destination, field::StaggeredVectorField3D& working, const field::CenteredVectorField3D& solid_velocity, const std::uint8_t* occupancy, float delta_seconds);

    private:
        void initialize();
        void release() noexcept;

        cudaStream_t stream{nullptr};
        std::array<std::int32_t, 3> resolution{0, 0, 0};
        float cell_size{0.0f};
        std::int32_t pressure_iterations{0};
        std::array<std::uint32_t, 6> flow_boundary_types{};
        std::array<float, 18> flow_boundary_velocity{};
        std::array<float, 6> flow_boundary_pressure{};
        std::array<bool, 3> periodic{};

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
} // namespace kfs::operators
