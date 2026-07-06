module;
#include <cuda_runtime.h>

export module keyframe.solver;
import std;
export import keyframe.field;
export import keyframe.boundary;
export import keyframe.operators.advection;
export import keyframe.operators.emitter;
import keyframe.operators.buoyancy;
import keyframe.operators.projection;
import keyframe.operators.solid;
import keyframe.operators.vorticity;

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
        operators::Emitter::Source emitter{};
        boundary::DomainBoundary boundary{};
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

        [[nodiscard]] std::expected<StepStats, std::string> step(const StepRequest& request);

        struct HostData final {
            std::int32_t nx{0};
            std::int32_t ny{0};
            std::int32_t nz{0};
            float cell_size{0.0f};
            float ambient_temperature{0.0f};
            boundary::PackedDomainBoundary boundary{};
            cudaStream_t stream{nullptr};
            std::uint32_t current_step{0u};
        } host;

        struct DeviceData final {
            field::ScalarField3D density_data{{0, 0, 0}};
            field::ScalarField3D density_temp{{0, 0, 0}};
            field::ScalarField3D temperature_data{{0, 0, 0}};
            field::ScalarField3D temperature_temp{{0, 0, 0}};
            field::CenteredVectorField3D force{{0, 0, 0}};
            field::CenteredVectorField3D solid_velocity{{0, 0, 0}};
            field::StaggeredVectorField3D velocity{{0, 0, 0}};
            field::StaggeredVectorField3D temp_velocity{{0, 0, 0}};
            field::CenteredVectorField3D centered_velocity{{0, 0, 0}};
            field::ScalarField3D solid_temperature{{0, 0, 0}};
            std::uint8_t* occupancy{nullptr};
        } device;

    private:
        std::optional<operators::Advection> advection{};
        std::optional<operators::Emitter> emitter{};
        std::optional<operators::Buoyancy> buoyancy{};
        std::optional<operators::Projection> projection{};
        std::optional<operators::Solid> solid{};
        std::optional<operators::Vorticity> vorticity{};
    };
} // namespace kfs::solver
