module;
#include <cuda_runtime.h>

export module keyframe.solver;
import std;
export import xayah.core.field;
export import xayah.core.boundary;
export import xayah.core.collider;
export import xayah.operators.advection;
export import xayah.operators.emitter;
export import xayah.operators.projection;
export import xayah.operators.vorticity;

namespace kfs::solver {
    export struct SmokeBoundary final {
        xayah::core::boundary::VectorBoundary3D velocity{
            .x_min = {.mode = xayah::core::boundary::VectorBoundaryMode::periodic},
            .x_max = {.mode = xayah::core::boundary::VectorBoundaryMode::periodic},
            .y_min = {.mode = xayah::core::boundary::VectorBoundaryMode::fixed_value, .value = {0.0f, 0.0f, 0.0f}},
            .y_max = {},
            .z_min = {.mode = xayah::core::boundary::VectorBoundaryMode::periodic},
            .z_max = {.mode = xayah::core::boundary::VectorBoundaryMode::periodic},
        };
        xayah::core::boundary::ScalarBoundary3D pressure{
            .x_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .x_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .y_min = {},
            .y_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::fixed_value, .value = 0.0f},
            .z_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .z_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
        };
        xayah::core::boundary::ScalarBoundary3D density{
            .x_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .x_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .y_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::fixed_value, .value = 0.0f},
            .y_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::fixed_value, .value = 0.0f},
            .z_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .z_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
        };
        xayah::core::boundary::ScalarBoundary3D temperature{
            .x_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .x_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .y_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::fixed_value, .value = 0.0f},
            .y_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::fixed_value, .value = 0.0f},
            .z_min = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
            .z_max = {.mode = xayah::core::boundary::ScalarBoundaryMode::periodic},
        };
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
        explicit Solver(std::array<std::uint32_t, 3> resolution = {64, 96, 64}, float cell_size = 0.01875f, SmokeBoundary boundaries = {}, cudaStream_t execution_stream = nullptr);
        ~Solver() noexcept                         = default;
        Solver(const Solver& other)                = delete;
        Solver& operator=(const Solver& other)     = delete;
        Solver(Solver&& other) noexcept            = delete;
        Solver& operator=(Solver&& other) noexcept = delete;

        const std::array<std::int32_t, 3> resolution{0, 0, 0};
        const float cell_size{0.0f};
        const xayah::core::boundary::ScalarBoundary3D pressure_boundary{};
        CUstream_st* const stream{nullptr};

        float ambient_temperature{0.0f};
        float buoyancy_density_factor{0.15f};
        float buoyancy_temperature_factor{1.2f};
        float density_emission_rate{18.0f};
        float temperature_emission_rate{36.0f};
        xayah::core::boundary::VectorBoundary3D velocity_boundary{};
        xayah::core::boundary::ScalarBoundary3D density_boundary{};
        xayah::core::boundary::ScalarBoundary3D temperature_boundary{};
        xayah::core::collider::ColliderSet colliders{};
        std::uint32_t current_step{0u};
        struct DeviceData final {
            xayah::core::field::ScalarField3D density_data{{0, 0, 0}};
            xayah::core::field::ScalarField3D density_temp{{0, 0, 0}};
            xayah::core::field::ScalarField3D temperature_data{{0, 0, 0}};
            xayah::core::field::ScalarField3D temperature_temp{{0, 0, 0}};
            xayah::core::field::CenteredVectorField3D force{{0, 0, 0}};
            xayah::core::field::CenteredVectorField3D constraint_velocity{{0, 0, 0}};
            xayah::core::field::StaggeredVectorField3D velocity{{0, 0, 0}};
            xayah::core::field::StaggeredVectorField3D temp_velocity{{0, 0, 0}};
            xayah::core::field::CenteredVectorField3D centered_velocity{{0, 0, 0}};
            xayah::core::field::ScalarField3D constraint_scalar{{0, 0, 0}};
            xayah::core::field::IndexedField3D cell_indices{{0, 0, 0}};
        } device{};
        xayah::operators::Advection advection;
        xayah::operators::Emitter emitter;
        xayah::operators::Projection projection;
        xayah::operators::Vorticity vorticity;

        [[nodiscard]] std::expected<StepStats, std::string> step(const StepRequest& request);
    };
} // namespace kfs::solver
