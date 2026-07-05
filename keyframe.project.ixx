export module keyframe.project;

import keyframe.plugin;
import std;

export namespace kfs::project {
    class Project final {
    public:
        struct State;

        Project(const Project& other) = delete;
        Project(Project&& other) noexcept;
        Project& operator=(const Project& other) = delete;
        Project& operator=(Project&& other) noexcept;
        ~Project() noexcept;

        [[nodiscard]] static const plugin::PluginDefinition<Project>& plugin();
        [[nodiscard]] static Project open(plugin::OpenContext context);

        void update(const plugin::UpdateInfo& update);
        [[nodiscard]] std::uint64_t revision() const;
        void write_document(plugin::SceneBuilder& scene) const;
        void write_frame(plugin::SceneBuilder& scene, plugin::FrameInfo frame);
        void write_controls(plugin::ControlBuilder& controls) const;

        void set_show_volume(bool value);
        void set_density_scale(float value);
        void set_show_domain(bool value);

    private:
        explicit Project(std::unique_ptr<State> state);

        std::unique_ptr<State> state;
    };
} // namespace kfs::project
