module;
#include <cuda_runtime.h>

export module keyframe.solver;
import std;
export import keyframe.field;
export import keyframe.boundary;
export import keyframe.operators.advection;
import keyframe.operators.projection;

namespace kfs::solver {
    export struct Config final {
        std::array<std::uint32_t, 3> resolution{64, 96, 64};
        float cell_size{0.01875f};
        std::int32_t pressure_iterations{64};
        float ambient_temperature{0.0f};
        float buoyancy_density_factor{0.15f};
        float buoyancy_temperature_factor{1.2f};
        float vorticity_confinement{0.22f};
        operators::Advection::Scheme advection_scheme{operators::Advection::Scheme::monotonic_cubic};
        boundary::DomainBoundary boundary{};
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
            float ambient_temperature{0.0f};
            float buoyancy_density_factor{0.0f};
            float buoyancy_temperature_factor{0.0f};
            float vorticity_confinement{0.0f};
            boundary::PackedDomainBoundary boundary{};
            cudaStream_t stream{nullptr};
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
        } device;

    private:
        std::optional<operators::Advection> advection{};
        std::optional<operators::Projection> projection{};
    };
} // namespace kfs::solver
