module;
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

export module keyframe.solver;
import std;
export import keyframe.field;
export import keyframe.operators.advection;

namespace kfs::solver {
    export enum class FlowBoundaryType : std::uint32_t {
        no_slip_wall   = 0,
        free_slip_wall = 1,
        outflow        = 2,
        periodic       = 3,
    };

    export enum class ScalarBoundaryType : std::uint32_t {
        fixed_value = 0,
        zero_flux   = 1,
        periodic    = 2,
    };

    export struct FlowBoundaryFace final {
        FlowBoundaryType type{FlowBoundaryType::no_slip_wall};
        float velocity_x{0.0f};
        float velocity_y{0.0f};
        float velocity_z{0.0f};
        float pressure{0.0f};
    };

    export struct FlowBoundary final {
        FlowBoundaryFace x_minus{FlowBoundaryType::periodic};
        FlowBoundaryFace x_plus{FlowBoundaryType::periodic};
        FlowBoundaryFace y_minus{FlowBoundaryType::no_slip_wall};
        FlowBoundaryFace y_plus{FlowBoundaryType::outflow};
        FlowBoundaryFace z_minus{FlowBoundaryType::periodic};
        FlowBoundaryFace z_plus{FlowBoundaryType::periodic};
    };

    export struct ScalarBoundaryFace final {
        ScalarBoundaryType type{ScalarBoundaryType::fixed_value};
        float value{0.0f};
    };

    export struct ScalarBoundary final {
        ScalarBoundaryFace x_minus{ScalarBoundaryType::periodic, 0.0f};
        ScalarBoundaryFace x_plus{ScalarBoundaryType::periodic, 0.0f};
        ScalarBoundaryFace y_minus{ScalarBoundaryType::fixed_value, 0.0f};
        ScalarBoundaryFace y_plus{ScalarBoundaryType::fixed_value, 0.0f};
        ScalarBoundaryFace z_minus{ScalarBoundaryType::periodic, 0.0f};
        ScalarBoundaryFace z_plus{ScalarBoundaryType::periodic, 0.0f};
    };

    export struct Config final {
        std::array<std::uint32_t, 3> resolution{64, 96, 64};
        float cell_size{0.01875f};
        std::int32_t pressure_iterations{64};
        float ambient_temperature{0.0f};
        float buoyancy_density_factor{0.15f};
        float buoyancy_temperature_factor{1.2f};
        float vorticity_confinement{0.22f};
        operators::Advection::Scheme advection_scheme{operators::Advection::Scheme::monotonic_cubic};
        FlowBoundary flow_boundary{};
        ScalarBoundary density_boundary{};
        ScalarBoundary temperature_boundary{};
    };

    export struct PlumeSource final {
        std::array<float, 3> center{0.5f, 0.12f, 0.5f};
        std::array<float, 3> radius{0.13f, 0.08f, 0.13f};
        float density{18.0f};
        float temperature{36.0f};
        float falloff{2.2f};
    };

    export struct StepRequest final {
        float delta_seconds{1.0f / 60.0f};
        std::int32_t iterations{1};
    };

    export struct StepStats final {
        std::uint32_t step{0u};
        float elapsed_ms{0.0f};
    };

    export struct Solver final {
        explicit Solver(const Config& config = {});
        ~Solver() noexcept;
        Solver(const Solver& other)                = delete;
        Solver& operator=(const Solver& other)     = delete;
        Solver(Solver&& other) noexcept            = delete;
        Solver& operator=(Solver&& other) noexcept = delete;

        void set_plume_source(const PlumeSource& source);
        [[nodiscard]] std::expected<StepStats, std::string> step(const StepRequest& request);

        struct HostData final {
            std::int32_t nx{0};
            std::int32_t ny{0};
            std::int32_t nz{0};
            float cell_size{0.0f};
            std::int32_t pressure_iterations{0};
            float ambient_temperature{0.0f};
            float buoyancy_density_factor{0.0f};
            float buoyancy_temperature_factor{0.0f};
            float vorticity_confinement{0.0f};
            std::array<std::uint32_t, 6> flow_boundary_types{};
            std::array<float, 18> flow_boundary_velocity{};
            std::array<float, 6> flow_boundary_pressure{};
            std::array<std::uint32_t, 6> density_boundary_types{};
            std::array<float, 6> density_boundary_values{};
            std::array<std::uint32_t, 6> temperature_boundary_types{};
            std::array<float, 6> temperature_boundary_values{};
            cudaStream_t stream{nullptr};
            cublasHandle_t cublas{nullptr};
            cusparseHandle_t cusparse{nullptr};
            cusparseSpMatDescr_t pressure_matrix{nullptr};
            cusparseDnVecDescr_t pressure_vec_p{nullptr};
            cusparseDnVecDescr_t pressure_vec_ap{nullptr};
            std::size_t spmv_buffer_size{0};
            PlumeSource plume_source{};
            std::vector<float> density_source{};
            std::vector<float> temperature_source{};
            std::uint32_t current_step{0u};
        } host;

        struct DeviceData final {
            field::ScalarField3D density_data{{0, 0, 0}};
            field::ScalarField3D density_temp{{0, 0, 0}};
            field::ScalarField3D density_source{{0, 0, 0}};
            field::ScalarField3D temperature_data{{0, 0, 0}};
            field::ScalarField3D temperature_temp{{0, 0, 0}};
            field::ScalarField3D temperature_source{{0, 0, 0}};
            field::CenteredVectorField3D force{{0, 0, 0}};
            field::CenteredVectorField3D solid_velocity{{0, 0, 0}};
            field::StaggeredVectorField3D velocity{{0, 0, 0}};
            field::StaggeredVectorField3D temp_velocity{{0, 0, 0}};
            field::CenteredVectorField3D centered_velocity{{0, 0, 0}};
            field::CenteredVectorField3D vorticity{{0, 0, 0}};
            field::ScalarField3D vorticity_magnitude{{0, 0, 0}};
            field::ScalarField3D solid_temperature{{0, 0, 0}};
            std::uint8_t* occupancy{nullptr};
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
        } device;

    private:
        std::optional<operators::Advection> advection{};
    };
} // namespace kfs::solver
