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

        void reset_simulation();
        void set_scene(std::string_view value);
        void set_delta_seconds(float value);
        void set_steps_per_update(std::uint64_t value);
        void set_vorticity_confinement(float value);
        void set_ambient_temperature(float value);
        void set_buoyancy_density_factor(float value);
        void set_buoyancy_temperature_factor(float value);
        void set_density_emission_rate(float value);
        void set_temperature_emission_rate(float value);
        void set_emitter_center_x(float value);
        void set_emitter_center_y(float value);
        void set_emitter_center_z(float value);
        void set_emitter_radius_x(float value);
        void set_emitter_radius_y(float value);
        void set_emitter_radius_z(float value);
        void set_emitter_falloff(float value);
        void set_show_emitter(bool value);
        void set_show_colliders(bool value);
        void set_collider_scale(float value);
        void set_moving_gate_amplitude(float value);
        void set_moving_gate_speed(float value);
        void set_constraint_scalar(float value);
        void set_show_density(bool value);
        void set_density_scale(float value);
        void set_show_temperature(bool value);
        void set_temperature_scale(float value);
        void set_show_cell_indices(bool value);
        void set_cell_index_cell_scale(float value);
        void set_show_domain(bool value);
        void set_camera_yaw_degrees(float value);
        void set_camera_pitch_degrees(float value);
        void set_camera_distance(float value);
        void set_camera_fov_degrees(float value);

    private:
        explicit Project(std::unique_ptr<State> state);

        std::unique_ptr<State> state;
    };
} // namespace kfs::project
