export module keyframe.inspector;
import std;
import keyframe.solver;

namespace kfs::inspector {
    export struct FrameSnapshot final {
        int frame_index{0};
        std::array<std::uint32_t, 3> resolution{0, 0, 0};
        float cell_size{0.0f};
        std::vector<float> density{};
        std::vector<float> temperature{};
        std::vector<float> velocity_x{};
        std::vector<float> velocity_y{};
        std::vector<float> velocity_z{};
    };

    export struct ScalarFieldStats final {
        float min{0.0f};
        float max{0.0f};
        double sum{0.0};
        float mean{0.0f};
        std::uint64_t nonzero_count{0u};
    };

    export struct FrameStats final {
        int frame_index{0};
        std::array<std::uint32_t, 3> resolution{0, 0, 0};
        float cell_size{0.0f};
        ScalarFieldStats density{};
        ScalarFieldStats temperature{};
    };

    export struct SolverDeviceView final {
        std::array<std::uint32_t, 3> resolution{0, 0, 0};
        std::uint64_t cell_count{0u};
        std::array<std::uint64_t, 3> velocity_count{};
        const float* density{};
        const float* temperature{};
        std::array<const float*, 3> velocity{};
        bool initialized{false};
    };

    export struct Inspector final {
        explicit Inspector(const solver::KeyframeSmoke& smoke);

        [[nodiscard]] SolverDeviceView device_view() const;
        [[nodiscard]] FrameSnapshot read_frame(int frame_index) const;
        [[nodiscard]] FrameStats frame_stats(int frame_index) const;

        const solver::KeyframeSmoke* smoke = nullptr;
    };
} // namespace kfs::inspector
