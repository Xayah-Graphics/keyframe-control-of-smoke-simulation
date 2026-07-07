module;
#include <cuda_runtime.h>

export module keyframe.solver;
import std;
export import keyframe.field;
export import keyframe.boundary;
export import keyframe.collider;
export import keyframe.operators.advection;
export import keyframe.operators.emitter;
export import keyframe.operators.projection;
export import keyframe.operators.vorticity;

namespace kfs::solver {
    export struct SmokeBoundary final {
        boundary::VectorBoundary3D velocity{
            .x_min = boundary::periodic_vector(),
            .x_max = boundary::periodic_vector(),
            .y_min = boundary::no_slip(),
            .y_max = boundary::outflow(),
            .z_min = boundary::periodic_vector(),
            .z_max = boundary::periodic_vector(),
        };
        boundary::ScalarBoundary3D pressure{
            .x_min = boundary::periodic_scalar(),
            .x_max = boundary::periodic_scalar(),
            .y_min = boundary::zero_gradient(),
            .y_max = boundary::fixed_value(0.0f),
            .z_min = boundary::periodic_scalar(),
            .z_max = boundary::periodic_scalar(),
        };
        boundary::ScalarBoundary3D density{
            .x_min = boundary::periodic_scalar(),
            .x_max = boundary::periodic_scalar(),
            .y_min = boundary::fixed_value(0.0f),
            .y_max = boundary::fixed_value(0.0f),
            .z_min = boundary::periodic_scalar(),
            .z_max = boundary::periodic_scalar(),
        };
        boundary::ScalarBoundary3D temperature{
            .x_min = boundary::periodic_scalar(),
            .x_max = boundary::periodic_scalar(),
            .y_min = boundary::fixed_value(0.0f),
            .y_max = boundary::fixed_value(0.0f),
            .z_min = boundary::periodic_scalar(),
            .z_max = boundary::periodic_scalar(),
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
        const boundary::ScalarBoundary3D pressure_boundary{};
        CUstream_st* const stream{nullptr};

        float ambient_temperature{0.0f};
        float buoyancy_density_factor{0.15f};
        float buoyancy_temperature_factor{1.2f};
        float density_emission_rate{18.0f};
        float temperature_emission_rate{36.0f};
        boundary::VectorBoundary3D velocity_boundary{};
        boundary::ScalarBoundary3D density_boundary{};
        boundary::ScalarBoundary3D temperature_boundary{};
        collider::ColliderSet colliders{};
        std::uint32_t current_step{0u};
        struct DeviceData final {
            field::ScalarField3D density_data{{0, 0, 0}};
            field::ScalarField3D density_temp{{0, 0, 0}};
            field::ScalarField3D temperature_data{{0, 0, 0}};
            field::ScalarField3D temperature_temp{{0, 0, 0}};
            field::CenteredVectorField3D force{{0, 0, 0}};
            field::CenteredVectorField3D constraint_velocity{{0, 0, 0}};
            field::StaggeredVectorField3D velocity{{0, 0, 0}};
            field::StaggeredVectorField3D temp_velocity{{0, 0, 0}};
            field::CenteredVectorField3D centered_velocity{{0, 0, 0}};
            field::ScalarField3D constraint_scalar{{0, 0, 0}};
            field::IndexedField3D cell_indices{{0, 0, 0}};
        } device{};
        operators::Advection advection;
        operators::Emitter emitter;
        operators::Projection projection;
        operators::Vorticity vorticity;

        [[nodiscard]] std::expected<StepStats, std::string> step(const StepRequest& request);
    };
} // namespace kfs::solver
