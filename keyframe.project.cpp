module;

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#else
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>

#if defined(_WIN32)
#define SPECTRA_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_SCENE_EXPORT __attribute__((visibility("default")))
#endif

module keyframe.project;

import xayah.core.field;
import keyframe.plugin;
import keyframe.solver;
import xayah.core.collider;
import xayah.core.geometry;
import std;

namespace kfs::project {
    namespace {
        constexpr char scene_free_plume[]                       = "free_plume";
        constexpr char scene_obstacle_gallery[]                 = "obstacle_gallery";
        constexpr char scene_moving_gate[]                      = "moving_gate";
        constexpr char section_scene_id[]                       = "scene";
        constexpr char section_simulation_id[]                  = "simulation";
        constexpr char section_emitter_id[]                     = "emitter";
        constexpr char section_display_id[]                     = "display";
        constexpr char section_statistics_id[]                  = "statistics";
        constexpr char action_reset_simulation_key[]            = "reset_simulation";
        constexpr char option_scene_key[]                       = "scene";
        constexpr char setting_scene_key[]                      = "scene";
        constexpr char setting_delta_seconds_key[]              = "delta_seconds";
        constexpr char setting_steps_per_update_key[]           = "steps_per_update";
        constexpr char setting_vorticity_confinement_key[]      = "vorticity_confinement";
        constexpr char setting_ambient_temperature_key[]        = "ambient_temperature";
        constexpr char setting_buoyancy_density_factor_key[]    = "buoyancy_density_factor";
        constexpr char setting_buoyancy_temperature_factor_key[] = "buoyancy_temperature_factor";
        constexpr char setting_density_emission_rate_key[]      = "density_emission_rate";
        constexpr char setting_temperature_emission_rate_key[]  = "temperature_emission_rate";
        constexpr char setting_emitter_center_x_key[]           = "emitter_center_x";
        constexpr char setting_emitter_center_y_key[]           = "emitter_center_y";
        constexpr char setting_emitter_center_z_key[]           = "emitter_center_z";
        constexpr char setting_emitter_radius_x_key[]           = "emitter_radius_x";
        constexpr char setting_emitter_radius_y_key[]           = "emitter_radius_y";
        constexpr char setting_emitter_radius_z_key[]           = "emitter_radius_z";
        constexpr char setting_emitter_falloff_key[]            = "emitter_falloff";
        constexpr char setting_show_emitter_key[]               = "show_emitter";
        constexpr char setting_show_colliders_key[]             = "show_colliders";
        constexpr char setting_collider_scale_key[]             = "collider_scale";
        constexpr char setting_moving_gate_amplitude_key[]      = "moving_gate_amplitude";
        constexpr char setting_moving_gate_speed_key[]          = "moving_gate_speed";
        constexpr char setting_constraint_scalar_key[]          = "constraint_scalar";
        constexpr char setting_show_density_key[]               = "show_density";
        constexpr char setting_density_scale_key[]              = "density_scale";
        constexpr char setting_show_temperature_key[]           = "show_temperature";
        constexpr char setting_temperature_scale_key[]          = "temperature_scale";
        constexpr char setting_show_cell_indices_key[]          = "show_cell_indices";
        constexpr char setting_cell_index_cell_scale_key[]      = "cell_index_cell_scale";
        constexpr char setting_show_domain_key[]                = "show_domain";
        constexpr char setting_camera_yaw_degrees_key[]         = "camera_yaw_degrees";
        constexpr char setting_camera_pitch_degrees_key[]       = "camera_pitch_degrees";
        constexpr char setting_camera_distance_key[]            = "camera_distance";
        constexpr char setting_camera_fov_degrees_key[]         = "camera_fov_degrees";
        constexpr char volume_name[]                            = "Keyframe Smoke Volume";
        constexpr char volume_material_name[]                   = "Keyframe Smoke Volume Material";
        constexpr char emitter_material_name[]                  = "Keyframe Emitter Material";
        constexpr char collider_box_material_name[]             = "Keyframe Collider Box Material";
        constexpr char collider_sphere_material_name[]          = "Keyframe Collider Sphere Material";
        constexpr char key_light_name[]                         = "Keyframe Smoke Key Light";
        constexpr char domain_segments_name[]                   = "Keyframe Smoke Domain";
        constexpr char cell_indices_grid_name[]                 = "Keyframe Cell Indices";
        constexpr char emitter_entity_name[]                    = "Keyframe Emitter";
        constexpr char camera_name[]                            = "Overview";
        constexpr float default_density_scale                   = 4.0f;
        constexpr float default_temperature_scale               = 0.18f;

        enum class ScenePreset : std::uint32_t {
            free_plume,
            obstacle_gallery,
            moving_gate,
        };

        struct SimulationSettings final {
            float delta_seconds{1.0f / 60.0f};
            std::int32_t steps_per_update{1};
            float ambient_temperature{0.0f};
            float buoyancy_density_factor{0.15f};
            float buoyancy_temperature_factor{1.2f};
            float density_emission_rate{18.0f};
            float temperature_emission_rate{36.0f};
            float vorticity_confinement{0.22f};
        };

        struct EmitterSettings final {
            std::array<float, 3> center{0.6f, 0.216f, 0.6f};
            std::array<float, 3> radius{0.156f, 0.144f, 0.156f};
            float falloff{2.2f};
        };

        struct SceneSettings final {
            ScenePreset preset{ScenePreset::obstacle_gallery};
            float collider_scale{1.0f};
            float moving_gate_amplitude{0.24f};
            float moving_gate_speed{1.4f};
            float constraint_scalar{0.0f};
        };

        struct DisplaySettings final {
            bool show_density{true};
            bool show_temperature{true};
            bool show_domain{true};
            bool show_emitter{true};
            bool show_colliders{true};
            bool show_cell_indices{true};
            float density_scale{default_density_scale};
            float temperature_scale{default_temperature_scale};
            float cell_index_cell_scale{0.78f};
            float camera_yaw_degrees{26.0f};
            float camera_pitch_degrees{16.0f};
            float camera_distance{2.75f};
            float camera_fov_degrees{45.0f};
        };

        struct OpenOptions final {
            std::array<std::uint32_t, 3> resolution{64, 96, 64};
            SimulationSettings simulation{};
            SceneSettings scene{};
            DisplaySettings display{};
        };

        class ExternalGpuBuffer final {
        public:
            ExternalGpuBuffer()                                    = default;
            ExternalGpuBuffer(const ExternalGpuBuffer&)            = delete;
            ExternalGpuBuffer(ExternalGpuBuffer&&)                 = delete;
            ExternalGpuBuffer& operator=(const ExternalGpuBuffer&) = delete;
            ExternalGpuBuffer& operator=(ExternalGpuBuffer&&)      = delete;
            ~ExternalGpuBuffer() noexcept;

            void ensure(std::shared_ptr<plugin::HostServices> host_services, std::uint32_t kind, std::uint64_t byte_size, std::string_view debug_name, std::string_view label);
            void reset() noexcept;

            [[nodiscard]] bool has_capacity(std::uint64_t requested_byte_size) const noexcept;
            [[nodiscard]] std::uint64_t resource_id() const noexcept;

            template <typename Value>
            [[nodiscard]] Value* mapped_as() const noexcept {
                return static_cast<Value*>(this->mapped_buffer);
            }

        private:
            std::shared_ptr<plugin::HostServices> host_services{};
            plugin::GpuBufferAllocation allocation{};
            cudaExternalMemory_t external_memory{};
            void* mapped_buffer{};
            std::uint64_t byte_size{};
        };

        struct GpuExportState final {
            ExternalGpuBuffer density_buffer{};
            ExternalGpuBuffer temperature_buffer{};
            ExternalGpuBuffer cell_indices_buffer{};
            std::optional<plugin::VolumeGrid> volume{};
            std::optional<plugin::ViewportVoxelGrid> cell_indices{};
            std::uint64_t volume_revision{};
            std::uint64_t volume_byte_size{};
            std::uint32_t volume_channel_mask{};
            std::uint64_t cell_indices_revision{};
            std::uint64_t cell_indices_byte_size{};

            void reset() noexcept {
                this->density_buffer.reset();
                this->temperature_buffer.reset();
                this->cell_indices_buffer.reset();
                this->volume.reset();
                this->cell_indices.reset();
                this->volume_revision          = 0u;
                this->volume_byte_size         = 0u;
                this->volume_channel_mask      = 0u;
                this->cell_indices_revision    = 0u;
                this->cell_indices_byte_size    = 0u;
            }
        };

        struct SmokeStats final {
            xayah::core::field::ScalarFieldStats density{};
            xayah::core::field::ScalarFieldStats temperature{};
        };

        [[nodiscard]] float parse_float_option(const std::string& text, const std::string_view name) {
            float value{};
            const char* const begin             = text.data();
            const char* const end               = text.data() + text.size();
            const std::from_chars_result result = std::from_chars(begin, end, value);
            if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(value)) throw std::runtime_error{std::format("{} must be a finite float.", name)};
            return value;
        }

        [[nodiscard]] std::uint32_t parse_u32_option(const std::string& text, const std::string_view name) {
            std::uint64_t value{};
            const char* const begin             = text.data();
            const char* const end               = text.data() + text.size();
            const std::from_chars_result result = std::from_chars(begin, end, value);
            if (result.ec != std::errc{} || result.ptr != end || value > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{std::format("{} must be an unsigned integer in uint32 range.", name)};
            return static_cast<std::uint32_t>(value);
        }

        [[nodiscard]] ScenePreset parse_scene_preset(const std::string_view value) {
            if (value == scene_free_plume) return ScenePreset::free_plume;
            if (value == scene_obstacle_gallery) return ScenePreset::obstacle_gallery;
            if (value == scene_moving_gate) return ScenePreset::moving_gate;
            throw std::runtime_error{std::format("scene must be one of {}, {}, {}.", scene_free_plume, scene_obstacle_gallery, scene_moving_gate)};
        }

        [[nodiscard]] std::string_view scene_preset_name(const ScenePreset preset) {
            switch (preset) {
            case ScenePreset::free_plume: return scene_free_plume;
            case ScenePreset::obstacle_gallery: return scene_obstacle_gallery;
            case ScenePreset::moving_gate: return scene_moving_gate;
            }
            throw std::runtime_error{"scene preset is invalid."};
        }

        [[nodiscard]] OpenOptions parse_open_options(const std::span<const plugin::Option> options) {
            OpenOptions parsed{};
            std::set<std::string> seen_options;
            for (const plugin::Option& option : options) {
                if (!seen_options.insert(option.key).second) throw std::runtime_error{std::format("Keyframe smoke open option '{}' is duplicated.", option.key)};
                if (option.key == option_scene_key)
                    parsed.scene.preset = parse_scene_preset(option.value);
                else if (option.key == "resolution_x")
                    parsed.resolution[0] = parse_u32_option(option.value, "resolution_x");
                else if (option.key == "resolution_y")
                    parsed.resolution[1] = parse_u32_option(option.value, "resolution_y");
                else if (option.key == "resolution_z")
                    parsed.resolution[2] = parse_u32_option(option.value, "resolution_z");
                else if (option.key == "dt")
                    parsed.simulation.delta_seconds = parse_float_option(option.value, "dt");
                else if (option.key == "steps_per_update") {
                    const std::uint32_t steps = parse_u32_option(option.value, "steps_per_update");
                    if (steps > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"steps_per_update must fit int32."};
                    parsed.simulation.steps_per_update = static_cast<std::int32_t>(steps);
                } else if (option.key == "density_scale")
                    parsed.display.density_scale = parse_float_option(option.value, "density_scale");
                else if (option.key == "temperature_scale")
                    parsed.display.temperature_scale = parse_float_option(option.value, "temperature_scale");
                else if (option.key == "vorticity_confinement")
                    parsed.simulation.vorticity_confinement = parse_float_option(option.value, "vorticity_confinement");
                else
                    throw std::runtime_error{std::format("unknown Keyframe smoke open option '{}'.", option.key)};
            }
            if (parsed.resolution[0] == 0u || parsed.resolution[1] == 0u || parsed.resolution[2] == 0u) throw std::runtime_error{"resolution dimensions must be positive."};
            if (parsed.simulation.delta_seconds <= 0.0f) throw std::runtime_error{"dt must be positive."};
            if (parsed.simulation.steps_per_update < 1) throw std::runtime_error{"steps_per_update must be positive."};
            if (parsed.display.density_scale <= 0.0f) throw std::runtime_error{"density_scale must be positive."};
            if (parsed.display.temperature_scale <= 0.0f) throw std::runtime_error{"temperature_scale must be positive."};
            if (parsed.simulation.vorticity_confinement < 0.0f) throw std::runtime_error{"vorticity_confinement must be non-negative."};
            return parsed;
        }

        [[nodiscard]] bool has_nonzero_bytes(const std::span<const std::uint8_t> bytes) {
            return std::ranges::any_of(bytes, [](const std::uint8_t value) { return value != 0u; });
        }

        void close_imported_handle(plugin::GpuBufferAllocation& allocation) noexcept {
#if defined(_WIN32)
            if (allocation.handle_kind == plugin::GpuResourceHandleKind::OpaqueWin32 && allocation.handle != 0u) static_cast<void>(CloseHandle(reinterpret_cast<HANDLE>(allocation.handle)));
#else
            if (allocation.handle_kind == plugin::GpuResourceHandleKind::OpaqueFileDescriptor && allocation.handle != 0u) static_cast<void>(close(static_cast<int>(allocation.handle)));
#endif
            allocation.handle = 0u;
        }

        void validate_cuda_device_identity(const plugin::GpuDeviceIdentity& identity) {
            int cuda_device = -1;
            if (const cudaError_t status = cudaGetDevice(&cuda_device); status != cudaSuccess) throw std::runtime_error{std::string{"cudaGetDevice failed: "} + cudaGetErrorString(status)};
            cudaDeviceProp device_properties{};
            if (const cudaError_t status = cudaGetDeviceProperties(&device_properties, cuda_device); status != cudaSuccess) throw std::runtime_error{std::string{"cudaGetDeviceProperties failed: "} + cudaGetErrorString(status)};

            const bool has_vulkan_uuid = has_nonzero_bytes(std::span{identity.device_uuid.data(), identity.device_uuid.size()});
            const bool has_vulkan_luid = has_nonzero_bytes(std::span{identity.device_luid.data(), identity.device_luid.size()});
            if (!has_vulkan_uuid && !has_vulkan_luid) throw std::runtime_error{"Spectra GPU resource device identity did not include UUID or LUID."};
            if (has_vulkan_uuid) {
                const std::uint8_t* const cuda_uuid = reinterpret_cast<const std::uint8_t*>(device_properties.uuid.bytes);
                for (std::size_t index = 0u; index < identity.device_uuid.size(); ++index)
                    if (identity.device_uuid[index] != cuda_uuid[index]) throw std::runtime_error{"Spectra Vulkan device UUID does not match the active CUDA device."};
            }
#if defined(_WIN32)
            if (has_vulkan_luid) {
                const std::uint8_t* const cuda_luid = reinterpret_cast<const std::uint8_t*>(device_properties.luid);
                for (std::size_t index = 0u; index < identity.device_luid.size(); ++index)
                    if (identity.device_luid[index] != cuda_luid[index]) throw std::runtime_error{"Spectra Vulkan device LUID does not match the active CUDA device."};
                if (identity.device_node_mask != device_properties.luidDeviceNodeMask) throw std::runtime_error{"Spectra Vulkan device node mask does not match the active CUDA device."};
            }
#endif
        }

        [[nodiscard]] std::array<float, 3u> domain_size(const solver::Solver& smoke) {
            const auto& resolution = smoke.device.density_data.resolution;
            const float cell_size  = smoke.cell_size;
            return {
                static_cast<float>(resolution[0]) * cell_size,
                static_cast<float>(resolution[1]) * cell_size,
                static_cast<float>(resolution[2]) * cell_size,
            };
        }

        [[nodiscard]] plugin::Camera orbit_camera(const solver::Solver& smoke, const DisplaySettings& display) {
            const std::array size        = domain_size(smoke);
            const std::array focus       = std::array{size[0] * 0.5f, size[1] * 0.52f, size[2] * 0.5f};
            constexpr float radians      = 0.017453292519943295769f;
            const float yaw              = display.camera_yaw_degrees * radians;
            const float pitch            = display.camera_pitch_degrees * radians;
            const float pitch_cosine     = std::cos(pitch);
            const float distance         = std::max({size[0], size[1], size[2]}) * display.camera_distance;
            const std::array forward     = std::array{-std::sin(yaw) * pitch_cosine, -std::sin(pitch), -std::cos(yaw) * pitch_cosine};
            const std::array position    = std::array{focus[0] - forward[0] * distance, focus[1] - forward[1] * distance, focus[2] - forward[2] * distance};
            const std::array world_up    = std::array{0.0f, 1.0f, 0.0f};
            std::array right             = std::array{forward[1] * world_up[2] - forward[2] * world_up[1], forward[2] * world_up[0] - forward[0] * world_up[2], forward[0] * world_up[1] - forward[1] * world_up[0]};
            const float right_length     = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
            right                        = {right[0] / right_length, right[1] / right_length, right[2] / right_length};
            const std::array down        = std::array{forward[1] * right[2] - forward[2] * right[1], forward[2] * right[0] - forward[0] * right[2], forward[0] * right[1] - forward[1] * right[0]};
            return plugin::Camera{
                .name                 = camera_name,
                .position             = position,
                .right                = right,
                .down                 = down,
                .forward              = forward,
                .projection           = plugin::CameraProjection::Perspective,
                .vertical_fov_degrees = display.camera_fov_degrees,
                .near_plane           = 0.01f,
                .far_plane            = std::max({size[0], size[1], size[2]}) * std::max(display.camera_distance * 4.0f, 8.0f),
            };
        }

        [[nodiscard]] plugin::ViewportSegmentSet domain_box(const solver::Solver& smoke) {
            const std::array minimum{0.0f, 0.0f, 0.0f};
            const std::array maximum = domain_size(smoke);
            const std::array corners = {
                std::array{minimum[0u], minimum[1u], minimum[2u]},
                std::array{maximum[0u], minimum[1u], minimum[2u]},
                std::array{maximum[0u], maximum[1u], minimum[2u]},
                std::array{minimum[0u], maximum[1u], minimum[2u]},
                std::array{minimum[0u], minimum[1u], maximum[2u]},
                std::array{maximum[0u], minimum[1u], maximum[2u]},
                std::array{maximum[0u], maximum[1u], maximum[2u]},
                std::array{minimum[0u], maximum[1u], maximum[2u]},
            };
            constexpr std::array color{0.20f, 0.58f, 1.0f, 0.88f};
            return plugin::ViewportSegmentSet{
                .name  = domain_segments_name,
                .owner = plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::Camera, .name = camera_name},
                .segments =
                    {
                        plugin::ViewportSegment{.start = corners[0u], .end = corners[1u]},
                        plugin::ViewportSegment{.start = corners[1u], .end = corners[2u]},
                        plugin::ViewportSegment{.start = corners[2u], .end = corners[3u]},
                        plugin::ViewportSegment{.start = corners[3u], .end = corners[0u]},
                        plugin::ViewportSegment{.start = corners[4u], .end = corners[5u]},
                        plugin::ViewportSegment{.start = corners[5u], .end = corners[6u]},
                        plugin::ViewportSegment{.start = corners[6u], .end = corners[7u]},
                        plugin::ViewportSegment{.start = corners[7u], .end = corners[4u]},
                        plugin::ViewportSegment{.start = corners[0u], .end = corners[4u]},
                        plugin::ViewportSegment{.start = corners[1u], .end = corners[5u]},
                        plugin::ViewportSegment{.start = corners[2u], .end = corners[6u]},
                        plugin::ViewportSegment{.start = corners[3u], .end = corners[7u]},
                    },
                .colors =
                    {
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                        plugin::Color{.value = color},
                    },
                .source_kind = plugin::ViewportSegmentSourceKind::Values,
                .width       = 1.5f,
                .width_mode  = plugin::ViewportSegmentWidthMode::Screen,
                .depth_mode  = plugin::ViewportSegmentDepthMode::Overlay,
            };
        }

        [[nodiscard]] plugin::Sphere ellipsoid_entity(const std::string& name, const xayah::core::geometry::Ellipsoid& ellipsoid, const std::string& material_name) {
            return plugin::Sphere{
                .name          = name,
                .radius        = 1.0f,
                .material_name = material_name,
                .transform =
                    plugin::Transform{
                        .position = ellipsoid.center,
                        .scale    = ellipsoid.radius,
                    },
            };
        }

        [[nodiscard]] plugin::Mesh box_entity(const std::string& name, const xayah::core::geometry::Box& box, const std::string& material_name) {
            return plugin::Mesh{
                .name = name,
                .vertices =
                    {
                        plugin::MeshVertex{.position = {-1.0f, -1.0f, -1.0f}, .normal = {0.0f, 0.0f, -1.0f}},
                        plugin::MeshVertex{.position = {1.0f, -1.0f, -1.0f}, .normal = {0.0f, 0.0f, -1.0f}},
                        plugin::MeshVertex{.position = {1.0f, 1.0f, -1.0f}, .normal = {0.0f, 0.0f, -1.0f}},
                        plugin::MeshVertex{.position = {-1.0f, 1.0f, -1.0f}, .normal = {0.0f, 0.0f, -1.0f}},
                        plugin::MeshVertex{.position = {-1.0f, -1.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f}},
                        plugin::MeshVertex{.position = {1.0f, -1.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f}},
                        plugin::MeshVertex{.position = {1.0f, 1.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f}},
                        plugin::MeshVertex{.position = {-1.0f, 1.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f}},
                        plugin::MeshVertex{.position = {-1.0f, -1.0f, -1.0f}, .normal = {-1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {-1.0f, 1.0f, -1.0f}, .normal = {-1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {-1.0f, 1.0f, 1.0f}, .normal = {-1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {-1.0f, -1.0f, 1.0f}, .normal = {-1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, -1.0f, -1.0f}, .normal = {1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, 1.0f, -1.0f}, .normal = {1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, 1.0f, 1.0f}, .normal = {1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, -1.0f, 1.0f}, .normal = {1.0f, 0.0f, 0.0f}},
                        plugin::MeshVertex{.position = {-1.0f, -1.0f, -1.0f}, .normal = {0.0f, -1.0f, 0.0f}},
                        plugin::MeshVertex{.position = {-1.0f, -1.0f, 1.0f}, .normal = {0.0f, -1.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, -1.0f, 1.0f}, .normal = {0.0f, -1.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, -1.0f, -1.0f}, .normal = {0.0f, -1.0f, 0.0f}},
                        plugin::MeshVertex{.position = {-1.0f, 1.0f, -1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
                        plugin::MeshVertex{.position = {-1.0f, 1.0f, 1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, 1.0f, 1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
                        plugin::MeshVertex{.position = {1.0f, 1.0f, -1.0f}, .normal = {0.0f, 1.0f, 0.0f}},
                    },
                .indices =
                    {
                        0u, 1u, 2u, 0u, 2u, 3u,
                        4u, 6u, 5u, 4u, 7u, 6u,
                        8u, 9u, 10u, 8u, 10u, 11u,
                        12u, 14u, 13u, 12u, 15u, 14u,
                        16u, 17u, 18u, 16u, 18u, 19u,
                        20u, 22u, 21u, 20u, 23u, 22u,
                    },
                .material_name = material_name,
                .transform =
                    plugin::Transform{
                        .position = box.center,
                        .scale    = box.half_extent,
                    },
            };
        }
    } // namespace

    struct Project::State final {
        OpenOptions open{};
        SimulationSettings simulation{};
        EmitterSettings emitter{};
        SceneSettings scene{};
        DisplaySettings display{};
        std::shared_ptr<plugin::HostServices> host_services{};
        std::unique_ptr<solver::Solver> smoke{};
        GpuExportState exports{};
        std::optional<solver::StepStats> latest_step_stats{};
        std::optional<SmokeStats> latest_smoke_stats{};
        std::uint64_t scene_revision{1u};
        double simulation_time_seconds{};
        bool host_update_running{};
    };

    namespace {
        void apply_settings_to_solver(Project::State& state) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            state.smoke->ambient_temperature          = state.simulation.ambient_temperature;
            state.smoke->buoyancy_density_factor     = state.simulation.buoyancy_density_factor;
            state.smoke->buoyancy_temperature_factor = state.simulation.buoyancy_temperature_factor;
            state.smoke->density_emission_rate       = state.simulation.density_emission_rate;
            state.smoke->temperature_emission_rate   = state.simulation.temperature_emission_rate;
            state.smoke->vorticity.confinement       = state.simulation.vorticity_confinement;
            state.smoke->emitter.source.region.center = state.emitter.center;
            state.smoke->emitter.source.region.radius = state.emitter.radius;
            state.smoke->emitter.source.falloff       = state.emitter.falloff;
        }

        void update_scene_colliders(Project::State& state, const double simulation_time_seconds) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            std::vector<xayah::core::collider::Collider>& colliders = state.smoke->colliders.items;
            colliders.clear();
            const std::array size  = domain_size(*state.smoke);
            const float scale      = state.scene.collider_scale;
            const float constraint = state.scene.constraint_scalar;
            switch (state.scene.preset) {
            case ScenePreset::free_plume: return;
            case ScenePreset::obstacle_gallery:
                colliders.push_back(xayah::core::collider::Collider{
                    .shape =
                        xayah::core::geometry::Box{
                            .center      = {size[0] * 0.50f, size[1] * 0.40f, size[2] * 0.50f},
                            .half_extent = {size[0] * 0.075f * scale, size[1] * 0.18f * scale, size[2] * 0.34f * scale},
                        },
                    .constraint_scalar = constraint,
                });
                colliders.push_back(xayah::core::collider::Collider{
                    .shape =
                        xayah::core::geometry::Ellipsoid{
                            .center = {size[0] * 0.68f, size[1] * 0.34f, size[2] * 0.52f},
                            .radius = {size[0] * 0.10f * scale, size[1] * 0.08f * scale, size[2] * 0.18f * scale},
                        },
                    .constraint_scalar = constraint,
                });
                return;
            case ScenePreset::moving_gate:
                {
                    const float phase     = static_cast<float>(simulation_time_seconds) * state.scene.moving_gate_speed;
                    const float amplitude = state.scene.moving_gate_amplitude;
                    colliders.push_back(xayah::core::collider::Collider{
                        .shape =
                            xayah::core::geometry::Box{
                                .center =
                                    {
                                        size[0] * (0.50f + amplitude * std::sin(phase)),
                                        size[1] * 0.38f,
                                        size[2] * 0.50f,
                                    },
                                .half_extent = {size[0] * 0.07f * scale, size[1] * 0.18f * scale, size[2] * 0.34f * scale},
                            },
                        .velocity          = {size[0] * amplitude * state.scene.moving_gate_speed * std::cos(phase), 0.0f, 0.0f},
                        .constraint_scalar = constraint,
                    });
                    return;
                }
            }
            throw std::runtime_error{"scene preset is invalid."};
        }

        void rasterize_current_colliders(Project::State& state) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            state.smoke->colliders.rasterize(state.smoke->stream, state.smoke->device.cell_indices, state.smoke->device.constraint_velocity, state.smoke->device.constraint_scalar, state.smoke->cell_size);
            state.exports.cell_indices.reset();
            state.exports.cell_indices_revision  = 0u;
            state.exports.cell_indices_byte_size = 0u;
        }

        void reset_solver(Project::State& state) {
            state.smoke                   = std::make_unique<solver::Solver>(state.open.resolution);
            state.simulation_time_seconds = 0.0;
            state.latest_step_stats.reset();
            apply_settings_to_solver(state);
            xayah::core::field::fill(state.smoke->stream, state.smoke->device.temperature_data, state.simulation.ambient_temperature);
            xayah::core::field::fill(state.smoke->stream, state.smoke->device.temperature_temp, state.simulation.ambient_temperature);
            update_scene_colliders(state, state.simulation_time_seconds);
            rasterize_current_colliders(state);
            state.latest_smoke_stats = SmokeStats{
                .density     = xayah::core::field::stats(state.smoke->stream, state.smoke->device.density_data),
                .temperature = xayah::core::field::stats(state.smoke->stream, state.smoke->device.temperature_data),
            };
            state.exports.reset();
        }

        void publish_volume_channels(Project::State& state) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            const bool needs_volume_shell = state.display.show_density || state.display.show_temperature || state.display.show_cell_indices;
            if (!needs_volume_shell) {
                state.exports.volume.reset();
                state.exports.volume_revision     = 0u;
                state.exports.volume_byte_size    = 0u;
                state.exports.volume_channel_mask = 0u;
                return;
            }

            const auto& density             = state.smoke->device.density_data;
            const auto& temperature         = state.smoke->device.temperature_data;
            const std::array dimensions     = std::array{static_cast<std::uint32_t>(density.resolution[0]), static_cast<std::uint32_t>(density.resolution[1]), static_cast<std::uint32_t>(density.resolution[2])};
            const std::uint64_t revision    = static_cast<std::uint64_t>(state.smoke->current_step) + 1u;
            const bool publish_density_channel = state.display.show_density || state.display.show_temperature || state.display.show_cell_indices;
            const bool publish_temperature_channel = state.display.show_temperature;
            const std::uint32_t channel_mask = (publish_density_channel ? 1u : 0u) | (publish_temperature_channel ? 2u : 0u);
            const std::uint64_t density_bytes = publish_density_channel ? static_cast<std::uint64_t>(density.bytes()) : 0u;
            const std::uint64_t temperature_bytes = publish_temperature_channel ? static_cast<std::uint64_t>(temperature.bytes()) : 0u;
            const std::uint64_t total_bytes = density_bytes + temperature_bytes;
            if (state.exports.volume.has_value() && state.exports.volume_revision == revision && state.exports.volume_byte_size == total_bytes && state.exports.volume_channel_mask == channel_mask) return;

            std::vector<plugin::VolumeChannel> channels;
            channels.reserve(2u);
            const cudaStream_t stream = state.smoke->stream;
            if (publish_density_channel) {
                state.exports.density_buffer.ensure(state.host_services, plugin::GpuBufferKindVolumeChannel, density_bytes, "keyframe smoke density volume", "density volume");
                float* const density_values = state.exports.density_buffer.mapped_as<float>();
                if (density_values == nullptr) throw std::runtime_error{"Keyframe smoke density external buffer was not mapped."};
                if (const cudaError_t status = cudaMemcpyAsync(density_values, density.data, density_bytes, cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync Keyframe smoke density volume failed: "} + cudaGetErrorString(status)};
                channels.push_back(plugin::VolumeChannel{
                    .name                    = "density",
                    .dimensions              = dimensions,
                    .format                  = plugin::VolumeChannelFormat::Float32,
                    .source_kind             = plugin::VolumeChannelSourceKind::ExternalGpuBuffer,
                    .index_encoding          = plugin::VolumeChannelIndexEncoding::Linear,
                    .buffer_id               = state.exports.density_buffer.resource_id(),
                    .external_device_pointer = reinterpret_cast<std::uintptr_t>(density_values),
                    .source_byte_size        = density_bytes,
                    .revision                = revision,
                });
            }
            if (publish_temperature_channel) {
                state.exports.temperature_buffer.ensure(state.host_services, plugin::GpuBufferKindVolumeChannel, temperature_bytes, "keyframe smoke temperature volume", "temperature volume");
                float* const temperature_values = state.exports.temperature_buffer.mapped_as<float>();
                if (temperature_values == nullptr) throw std::runtime_error{"Keyframe smoke temperature external buffer was not mapped."};
                if (const cudaError_t status = cudaMemcpyAsync(temperature_values, temperature.data, temperature_bytes, cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync Keyframe smoke temperature volume failed: "} + cudaGetErrorString(status)};
                channels.push_back(plugin::VolumeChannel{
                    .name                    = "temperature",
                    .dimensions              = dimensions,
                    .format                  = plugin::VolumeChannelFormat::Float32,
                    .source_kind             = plugin::VolumeChannelSourceKind::ExternalGpuBuffer,
                    .index_encoding          = plugin::VolumeChannelIndexEncoding::Linear,
                    .buffer_id               = state.exports.temperature_buffer.resource_id(),
                    .external_device_pointer = reinterpret_cast<std::uintptr_t>(temperature_values),
                    .source_byte_size        = temperature_bytes,
                    .revision                = revision,
                });
            }
            if (channel_mask != 0u)
                if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize Keyframe smoke volume failed: "} + cudaGetErrorString(status)};

            state.exports.volume = plugin::VolumeGrid{
                .name       = volume_name,
                .dimensions = dimensions,
                .origin     = {0.0f, 0.0f, 0.0f},
                .voxel_size = {state.smoke->cell_size, state.smoke->cell_size, state.smoke->cell_size},
                .channels   = std::move(channels),
                .material_name = volume_material_name,
            };
            state.exports.volume_revision     = revision;
            state.exports.volume_byte_size    = total_bytes;
            state.exports.volume_channel_mask = channel_mask;
        }

        void publish_cell_indices(Project::State& state) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            if (!state.display.show_cell_indices) {
                state.exports.cell_indices.reset();
                state.exports.cell_indices_revision  = 0u;
                state.exports.cell_indices_byte_size = 0u;
                return;
            }
            if (!state.exports.volume.has_value()) throw std::runtime_error{"Keyframe cell index visualization requires a volume owner."};
            const auto& indices                  = state.smoke->device.cell_indices;
            const std::uint64_t word_count       = xayah::core::field::BitField3D::word_count(indices.count());
            const std::uint64_t byte_size        = word_count * sizeof(std::uint32_t);
            const std::uint64_t revision         = static_cast<std::uint64_t>(state.smoke->current_step) + 1u;
            const std::array dimensions          = std::array{static_cast<std::uint32_t>(indices.resolution[0]), static_cast<std::uint32_t>(indices.resolution[1]), static_cast<std::uint32_t>(indices.resolution[2])};
            if (state.exports.cell_indices.has_value() && state.exports.cell_indices_revision == revision && state.exports.cell_indices_byte_size == byte_size && state.exports.cell_indices->cell_scale == state.display.cell_index_cell_scale) return;

            state.exports.cell_indices_buffer.ensure(state.host_services, plugin::GpuBufferKindViewportVoxelGrid, byte_size, "keyframe cell indices bitfield", "cell indices");
            std::uint32_t* const bitfield_words = state.exports.cell_indices_buffer.mapped_as<std::uint32_t>();
            if (bitfield_words == nullptr) throw std::runtime_error{"Keyframe cell index external buffer was not mapped."};
            const cudaStream_t stream = state.smoke->stream;
            xayah::core::field::pack(stream, bitfield_words, indices);
            if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize Keyframe cell index bitfield failed: "} + cudaGetErrorString(status)};

            state.exports.cell_indices = plugin::ViewportVoxelGrid{
                .name           = cell_indices_grid_name,
                .owner          = plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::VolumeGrid, .name = volume_name},
                .dimensions     = dimensions,
                .origin         = {0.0f, 0.0f, 0.0f},
                .voxel_size     = {state.smoke->cell_size, state.smoke->cell_size, state.smoke->cell_size},
                .color          = {0.08f, 0.78f, 1.0f, 0.36f},
                .cell_scale     = state.display.cell_index_cell_scale,
                .depth_mode     = static_cast<std::uint32_t>(plugin::ViewportSegmentDepthMode::Overlay),
                .source_kind    = plugin::ViewportVoxelGridSourceKind::Bitfield,
                .index_encoding = plugin::ViewportVoxelGridIndexEncoding::Linear,
                .buffer_id      = state.exports.cell_indices_buffer.resource_id(),
                .source_byte_size = byte_size,
                .index_count      = 0u,
                .revision         = revision,
            };
            state.exports.cell_indices_revision  = revision;
            state.exports.cell_indices_byte_size = byte_size;
        }

        [[nodiscard]] std::vector<plugin::Material> scene_materials(const DisplaySettings& display) {
            return {
                plugin::Material{
                    .name                     = volume_material_name,
                    .model                    = "volume",
                    .alpha_mode               = "blend",
                    .base_color               = {1.0f, 1.0f, 1.0f, 1.0f},
                    .roughness                = 0.35f,
                    .volume_density_scale     = display.show_density || display.show_temperature ? display.density_scale : 0.0f,
                    .volume_temperature_scale = display.show_temperature ? display.temperature_scale : 0.0f,
                },
                plugin::Material{
                    .name       = emitter_material_name,
                    .model      = "unlit_surface",
                    .alpha_mode = "blend",
                    .base_color = {1.0f, 0.48f, 0.18f, 0.34f},
                    .roughness  = 0.25f,
                },
                plugin::Material{
                    .name       = collider_box_material_name,
                    .model      = "lit_surface",
                    .alpha_mode = "blend",
                    .base_color = {0.10f, 0.72f, 1.0f, 0.42f},
                    .roughness  = 0.48f,
                },
                plugin::Material{
                    .name       = collider_sphere_material_name,
                    .model      = "lit_surface",
                    .alpha_mode = "blend",
                    .base_color = {0.72f, 0.30f, 1.0f, 0.42f},
                    .roughness  = 0.48f,
                },
            };
        }

        void append_scene_geometry(plugin::Document& document, const Project::State& state) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            if (state.display.show_emitter) document.spheres.push_back(ellipsoid_entity(emitter_entity_name, state.smoke->emitter.source.region, emitter_material_name));
            if (!state.display.show_colliders) return;
            for (std::size_t index = 0u; index < state.smoke->colliders.items.size(); ++index) {
                const xayah::core::collider::Collider& item = state.smoke->colliders.items[index];
                if (const xayah::core::geometry::Ellipsoid* const ellipsoid = std::get_if<xayah::core::geometry::Ellipsoid>(&item.shape); ellipsoid != nullptr) {
                    document.spheres.push_back(ellipsoid_entity(std::format("Keyframe Collider Ellipsoid {}", index + 1u), *ellipsoid, collider_sphere_material_name));
                    continue;
                }
                if (const xayah::core::geometry::Box* const box = std::get_if<xayah::core::geometry::Box>(&item.shape); box != nullptr) document.meshes.push_back(box_entity(std::format("Keyframe Collider Box {}", index + 1u), *box, collider_box_material_name));
            }
        }
    } // namespace

    ExternalGpuBuffer::~ExternalGpuBuffer() noexcept {
        this->reset();
    }

    void ExternalGpuBuffer::reset() noexcept {
        this->mapped_buffer = nullptr;
        if (this->external_memory != nullptr) {
            static_cast<void>(cudaDestroyExternalMemory(this->external_memory));
            this->external_memory = nullptr;
        }
        if (this->allocation.resource_id != 0u && this->host_services != nullptr) {
            try {
                this->host_services->release_gpu_buffer(this->allocation.resource_id);
            } catch (...) {
            }
        }
        this->allocation = plugin::GpuBufferAllocation{};
        this->byte_size  = 0u;
        this->host_services.reset();
    }

    bool ExternalGpuBuffer::has_capacity(const std::uint64_t requested_byte_size) const noexcept {
        return this->allocation.resource_id != 0u && this->byte_size >= requested_byte_size;
    }

    std::uint64_t ExternalGpuBuffer::resource_id() const noexcept {
        return this->allocation.resource_id;
    }

    void ExternalGpuBuffer::ensure(std::shared_ptr<plugin::HostServices> next_host_services, const std::uint32_t kind, const std::uint64_t requested_byte_size, const std::string_view debug_name, const std::string_view label) {
        if (this->has_capacity(requested_byte_size)) return;
        this->reset();
        if (next_host_services == nullptr) throw std::runtime_error{std::format("Spectra host services are required for {} visualization.", label)};
        if (requested_byte_size == 0u) throw std::runtime_error{std::format("{} byte size is invalid.", label)};
        if (!next_host_services->request_gpu_buffer) throw std::runtime_error{"Spectra host services request_gpu_buffer callback is not configured."};
        if (!next_host_services->release_gpu_buffer) throw std::runtime_error{"Spectra host services release_gpu_buffer callback is not configured."};
        plugin::GpuBufferAllocation next_allocation = next_host_services->request_gpu_buffer(kind, requested_byte_size, debug_name);
        if (next_allocation.resource_id == 0u) throw std::runtime_error{std::format("Spectra returned an invalid {} resource id.", label)};
        if (next_allocation.kind != kind) throw std::runtime_error{std::format("Spectra returned an unexpected GPU buffer kind for {}.", label)};
        if (next_allocation.byte_size < requested_byte_size) {
            close_imported_handle(next_allocation);
            next_host_services->release_gpu_buffer(next_allocation.resource_id);
            throw std::runtime_error{std::format("Spectra returned a {} buffer smaller than requested.", label)};
        }
        if (next_allocation.handle == 0u) {
            next_host_services->release_gpu_buffer(next_allocation.resource_id);
            throw std::runtime_error{std::format("Spectra returned an empty {} external memory handle.", label)};
        }

        try {
            validate_cuda_device_identity(next_allocation.device_identity);
            cudaExternalMemoryHandleDesc memory_desc{};
            memory_desc.size = next_allocation.byte_size;
            switch (next_allocation.handle_kind) {
#if defined(_WIN32)
            case plugin::GpuResourceHandleKind::OpaqueWin32:
                memory_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
                memory_desc.handle.win32.handle = reinterpret_cast<void*>(next_allocation.handle);
                break;
#else
            case plugin::GpuResourceHandleKind::OpaqueFileDescriptor:
                memory_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
                memory_desc.handle.fd = static_cast<int>(next_allocation.handle);
                break;
#endif
            default: throw std::runtime_error{std::format("Spectra returned an unsupported {} external memory handle kind.", label)};
            }

            cudaExternalMemory_t imported_memory{};
            const cudaError_t import_status = cudaImportExternalMemory(&imported_memory, &memory_desc);
            close_imported_handle(next_allocation);
            if (import_status != cudaSuccess) throw std::runtime_error{std::format("cudaImportExternalMemory for {} failed: {}", label, cudaGetErrorString(import_status))};

            cudaExternalMemoryBufferDesc buffer_desc{};
            buffer_desc.size         = next_allocation.byte_size;
            void* next_mapped_buffer = nullptr;
            if (const cudaError_t status = cudaExternalMemoryGetMappedBuffer(&next_mapped_buffer, imported_memory, &buffer_desc); status != cudaSuccess) {
                static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                throw std::runtime_error{std::format("cudaExternalMemoryGetMappedBuffer for {} failed: {}", label, cudaGetErrorString(status))};
            }
            if (next_mapped_buffer == nullptr) {
                static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                throw std::runtime_error{std::format("cudaExternalMemoryGetMappedBuffer returned null for {}.", label)};
            }
            this->host_services   = std::move(next_host_services);
            this->allocation      = next_allocation;
            this->external_memory = imported_memory;
            this->mapped_buffer   = next_mapped_buffer;
            this->byte_size       = requested_byte_size;
        } catch (...) {
            if (next_allocation.handle != 0u) close_imported_handle(next_allocation);
            next_host_services->release_gpu_buffer(next_allocation.resource_id);
            throw;
        }
    }

    Project::Project(std::unique_ptr<State> state) : state(std::move(state)) {}
    Project::Project(Project&& other) noexcept            = default;
    Project& Project::operator=(Project&& other) noexcept = default;
    Project::~Project() noexcept                          = default;

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition{
            .id                = "keyframe.smoke",
            .title             = "Keyframe Smoke",
            .open_action_label = "Open Smoke Solver",
            .sections =
                {
                    plugin::section(section_scene_id, "Scene"),
                    plugin::section(section_simulation_id, "Simulation"),
                    plugin::section(section_emitter_id, "Emitter"),
                    plugin::section(section_display_id, "Display"),
                    plugin::section(section_statistics_id, "Statistics"),
                },
            .open_options =
                {
                    plugin::choice(option_scene_key, "Scene", {scene_free_plume, scene_obstacle_gallery, scene_moving_gate}).defaulted(scene_obstacle_gallery).section(section_scene_id),
                    plugin::unsigned_integer("resolution_x", "Resolution X", 64u).section(section_simulation_id),
                    plugin::unsigned_integer("resolution_y", "Resolution Y", 96u).section(section_simulation_id),
                    plugin::unsigned_integer("resolution_z", "Resolution Z", 64u).section(section_simulation_id),
                    plugin::float_option("dt", "Delta Seconds", 1.0f / 60.0f).section(section_simulation_id),
                    plugin::unsigned_integer("steps_per_update", "Steps Per Update", 1u).section(section_simulation_id),
                    plugin::float_option("vorticity_confinement", "Vorticity Confinement", 0.22f).section(section_simulation_id),
                    plugin::float_option("density_scale", "Density Scale", default_density_scale).section(section_display_id),
                    plugin::float_option("temperature_scale", "Temperature Scale", default_temperature_scale).section(section_display_id),
                },
            .actions =
                {
                    plugin::action(action_reset_simulation_key, "Reset Simulation", &Project::reset_simulation).description("Recreate solver state with the current runtime settings.").section(section_simulation_id),
                },
            .settings =
                {
                    plugin::choice_value(setting_scene_key, "Scene", {scene_free_plume, scene_obstacle_gallery, scene_moving_gate}, scene_obstacle_gallery, &Project::set_scene).section(section_scene_id),
                    plugin::float_value(setting_collider_scale_key, "Collider Scale", 1.0f, &Project::set_collider_scale).section(section_scene_id).slider(0.05f, 2.0f, 0.01f),
                    plugin::float_value(setting_moving_gate_amplitude_key, "Moving Gate Amplitude", 0.24f, &Project::set_moving_gate_amplitude).section(section_scene_id).slider(0.0f, 0.45f, 0.001f),
                    plugin::float_value(setting_moving_gate_speed_key, "Moving Gate Speed", 1.4f, &Project::set_moving_gate_speed).section(section_scene_id).slider(0.0f, 8.0f, 0.001f),
                    plugin::float_value(setting_constraint_scalar_key, "Constraint Scalar", 0.0f, &Project::set_constraint_scalar).section(section_scene_id).slider(-20.0f, 80.0f, 0.01f),
                    plugin::float_value(setting_delta_seconds_key, "Delta Seconds", 1.0f / 60.0f, &Project::set_delta_seconds).section(section_simulation_id).slider(0.0001f, 0.08f, 0.0001f),
                    plugin::unsigned_integer_value(setting_steps_per_update_key, "Steps Per Update", 1u, &Project::set_steps_per_update).section(section_simulation_id),
                    plugin::float_value(setting_vorticity_confinement_key, "Vorticity Confinement", 0.22f, &Project::set_vorticity_confinement).section(section_simulation_id).slider(0.0f, 2.0f, 0.001f),
                    plugin::float_value(setting_ambient_temperature_key, "Ambient Temperature", 0.0f, &Project::set_ambient_temperature).section(section_simulation_id).slider(-20.0f, 40.0f, 0.01f),
                    plugin::float_value(setting_buoyancy_density_factor_key, "Density Buoyancy", 0.15f, &Project::set_buoyancy_density_factor).section(section_simulation_id).slider(-4.0f, 4.0f, 0.001f),
                    plugin::float_value(setting_buoyancy_temperature_factor_key, "Temperature Buoyancy", 1.2f, &Project::set_buoyancy_temperature_factor).section(section_simulation_id).slider(-4.0f, 8.0f, 0.001f),
                    plugin::float_value(setting_density_emission_rate_key, "Density Emission Rate", 18.0f, &Project::set_density_emission_rate).section(section_simulation_id).slider(0.0f, 120.0f, 0.001f),
                    plugin::float_value(setting_temperature_emission_rate_key, "Temperature Emission Rate", 36.0f, &Project::set_temperature_emission_rate).section(section_simulation_id).slider(0.0f, 160.0f, 0.001f),
                    plugin::float_value(setting_emitter_center_x_key, "Emitter Center X", 0.6f, &Project::set_emitter_center_x).section(section_emitter_id).slider(0.0f, 2.0f, 0.001f),
                    plugin::float_value(setting_emitter_center_y_key, "Emitter Center Y", 0.216f, &Project::set_emitter_center_y).section(section_emitter_id).slider(0.0f, 2.0f, 0.001f),
                    plugin::float_value(setting_emitter_center_z_key, "Emitter Center Z", 0.6f, &Project::set_emitter_center_z).section(section_emitter_id).slider(0.0f, 2.0f, 0.001f),
                    plugin::float_value(setting_emitter_radius_x_key, "Emitter Radius X", 0.156f, &Project::set_emitter_radius_x).section(section_emitter_id).slider(0.001f, 1.0f, 0.001f),
                    plugin::float_value(setting_emitter_radius_y_key, "Emitter Radius Y", 0.144f, &Project::set_emitter_radius_y).section(section_emitter_id).slider(0.001f, 1.0f, 0.001f),
                    plugin::float_value(setting_emitter_radius_z_key, "Emitter Radius Z", 0.156f, &Project::set_emitter_radius_z).section(section_emitter_id).slider(0.001f, 1.0f, 0.001f),
                    plugin::float_value(setting_emitter_falloff_key, "Emitter Falloff", 2.2f, &Project::set_emitter_falloff).section(section_emitter_id).slider(0.001f, 10.0f, 0.001f),
                    plugin::toggle(setting_show_emitter_key, "Show Emitter", true, &Project::set_show_emitter).section(section_display_id),
                    plugin::toggle(setting_show_colliders_key, "Show Colliders", true, &Project::set_show_colliders).section(section_display_id),
                    plugin::toggle(setting_show_density_key, "Show Density", true, &Project::set_show_density).section(section_display_id),
                    plugin::float_value(setting_density_scale_key, "Density Scale", default_density_scale, &Project::set_density_scale).section(section_display_id).slider(0.001f, 16.0f, 0.001f),
                    plugin::toggle(setting_show_temperature_key, "Show Temperature", true, &Project::set_show_temperature).section(section_display_id),
                    plugin::float_value(setting_temperature_scale_key, "Temperature Scale", default_temperature_scale, &Project::set_temperature_scale).section(section_display_id).slider(0.001f, 8.0f, 0.001f),
                    plugin::toggle(setting_show_cell_indices_key, "Show Cell Indices", true, &Project::set_show_cell_indices).section(section_display_id),
                    plugin::float_value(setting_cell_index_cell_scale_key, "Cell Index Cell Scale", 0.78f, &Project::set_cell_index_cell_scale).section(section_display_id).slider(0.05f, 1.0f, 0.01f),
                    plugin::toggle(setting_show_domain_key, "Show Domain", true, &Project::set_show_domain).section(section_display_id),
                    plugin::float_value(setting_camera_yaw_degrees_key, "Camera Yaw", 26.0f, &Project::set_camera_yaw_degrees).section(section_display_id).slider(-180.0f, 180.0f, 0.1f),
                    plugin::float_value(setting_camera_pitch_degrees_key, "Camera Pitch", 16.0f, &Project::set_camera_pitch_degrees).section(section_display_id).slider(-80.0f, 80.0f, 0.1f),
                    plugin::float_value(setting_camera_distance_key, "Camera Distance", 2.75f, &Project::set_camera_distance).section(section_display_id).slider(0.5f, 8.0f, 0.01f),
                    plugin::float_value(setting_camera_fov_degrees_key, "Camera FOV", 45.0f, &Project::set_camera_fov_degrees).section(section_display_id).slider(10.0f, 90.0f, 0.1f),
                },
        };
        return definition;
    }

    Project Project::open(plugin::OpenContext context) {
        OpenOptions options          = parse_open_options(std::span<const plugin::Option>{context.options});
        auto state                   = std::make_unique<State>();
        state->open                  = options;
        state->simulation            = options.simulation;
        state->scene                 = options.scene;
        state->display               = options.display;
        state->host_services         = std::move(context.host_services);
        reset_solver(*state);
        return Project{std::move(state)};
    }

    void Project::update(const plugin::UpdateInfo& update) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(update.wall_delta_seconds) || update.wall_delta_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project wall delta time is invalid."};
        if (!std::isfinite(update.update_delta_seconds) || update.update_delta_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project update delta time is invalid."};
        this->state->host_update_running = update.update_running;
        if (update.update_delta_seconds == 0.0) return;

        apply_settings_to_solver(*this->state);
        const double step_seconds                 = static_cast<double>(this->state->simulation.delta_seconds) * static_cast<double>(this->state->simulation.steps_per_update);
        const double next_simulation_time_seconds = this->state->simulation_time_seconds + step_seconds;
        update_scene_colliders(*this->state, next_simulation_time_seconds);
        const std::expected<solver::StepStats, std::string> stats = this->state->smoke->step(solver::StepRequest{
            .delta_seconds = this->state->simulation.delta_seconds,
            .iterations    = this->state->simulation.steps_per_update,
        });
        if (!stats) throw std::runtime_error{stats.error()};
        this->state->simulation_time_seconds = next_simulation_time_seconds;
        this->state->latest_step_stats       = *stats;
        this->state->latest_smoke_stats      = SmokeStats{
                 .density     = xayah::core::field::stats(this->state->smoke->stream, this->state->smoke->device.density_data),
                 .temperature = xayah::core::field::stats(this->state->smoke->stream, this->state->smoke->device.temperature_data),
        };
        this->state->exports.volume.reset();
        this->state->exports.cell_indices.reset();
        this->state->exports.volume_revision        = 0u;
        this->state->exports.volume_byte_size       = 0u;
        this->state->exports.volume_channel_mask    = 0u;
        this->state->exports.cell_indices_revision  = 0u;
        this->state->exports.cell_indices_byte_size = 0u;
        ++this->state->scene_revision;
    }

    std::uint64_t Project::revision() const {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        return this->state->scene_revision;
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        const std::array size = domain_size(*this->state->smoke);
        scene.set_document(plugin::Document{
            .timeline =
                plugin::TimelineDescriptor{
                    .kind = plugin::TimelineKind::Static,
                },
            .update =
                plugin::UpdateDescriptor{
                    .enabled            = true,
                    .initial_running    = false,
                    .step_delta_seconds = static_cast<double>(this->state->simulation.delta_seconds),
                },
            .navigation_target =
                plugin::ViewportNavigationTarget{
                    .revision       = 1u,
                    .focus          = {size[0] * 0.5f, size[1] * 0.5f, size[2] * 0.5f},
                    .bounds_minimum = {0.0f, 0.0f, 0.0f},
                    .bounds_maximum = size,
                    .navigation_up  = {0.0f, 1.0f, 0.0f},
                },
            .active_camera_name = camera_name,
            .cameras            = {orbit_camera(*this->state->smoke, this->state->display)},
            .materials          = scene_materials(this->state->display),
            .lights =
                {
                    plugin::Light{
                        .name      = key_light_name,
                        .kind      = "directional",
                        .color     = {1.0f, 1.0f, 1.0f},
                        .intensity = 3.0f,
                    },
                },
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo frame) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(frame.delta_seconds) || frame.delta_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project frame delta time is invalid."};
        if (!std::isfinite(frame.time_seconds) || frame.time_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project frame time is invalid."};

        publish_volume_channels(*this->state);
        publish_cell_indices(*this->state);

        plugin::Document document{
            .cameras = {orbit_camera(*this->state->smoke, this->state->display)},
        };
        if (this->state->exports.volume.has_value()) document.volumes.push_back(*this->state->exports.volume);
        if (this->state->display.show_domain) document.debug_attachments.viewport_segment_sets.push_back(domain_box(*this->state->smoke));
        if (this->state->exports.cell_indices.has_value()) document.debug_attachments.viewport_voxel_grids.push_back(*this->state->exports.cell_indices);
        append_scene_geometry(document, *this->state);
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        controls.phase(this->state->host_update_running ? "Running" : "Paused").headline(this->state->latest_step_stats.has_value() ? "Smoke simulation running" : "Smoke solver loaded").message(this->state->latest_step_stats.has_value() ? "Keyframe smoke advances on Spectra update ticks." : "Update clock is paused before the first simulation step.");
        controls.enable(action_reset_simulation_key);
        const auto& resolution = this->state->smoke->device.density_data.resolution;
        controls.metric("scene", "Scene", scene_preset_name(this->state->scene.preset)).section(section_scene_id).display_primary().color({1.0f, 0.58f, 0.20f, 1.0f});
        controls.metric("colliders", "Colliders", this->state->smoke->colliders.items.size()).section(section_scene_id);
        controls.metric("simulation_time", "Simulation Time", std::format("{:.3f}", this->state->simulation_time_seconds)).section(section_scene_id);
        controls.metric("resolution", "Resolution", std::format("{} x {} x {}", resolution[0], resolution[1], resolution[2])).section(section_simulation_id).display_primary().color({0.55f, 0.85f, 1.0f, 1.0f});
        controls.metric("cell_size", "Cell Size", std::format("{:.6f}", this->state->smoke->cell_size)).section(section_simulation_id);
        controls.metric("dt", "Delta Seconds", std::format("{:.6f}", this->state->simulation.delta_seconds)).section(section_simulation_id);
        controls.metric("steps_per_update", "Steps Per Update", this->state->simulation.steps_per_update).section(section_simulation_id);
        controls.metric("vorticity_confinement", "Vorticity", std::format("{:.6f}", this->state->smoke->vorticity.confinement)).section(section_simulation_id);
        controls.metric("emitter_center", "Emitter Center", std::format("{:.3f}, {:.3f}, {:.3f}", this->state->emitter.center[0], this->state->emitter.center[1], this->state->emitter.center[2])).section(section_emitter_id);
        controls.metric("emitter_radius", "Emitter Radius", std::format("{:.3f}, {:.3f}, {:.3f}", this->state->emitter.radius[0], this->state->emitter.radius[1], this->state->emitter.radius[2])).section(section_emitter_id);
        controls.metric("volume", "Volume", this->state->exports.volume.has_value() ? "visible" : "hidden").section(section_display_id);
        controls.metric("density", "Density", this->state->display.show_density ? "visible" : "hidden").section(section_display_id);
        controls.metric("temperature", "Temperature", this->state->display.show_temperature ? "visible" : "hidden").section(section_display_id);
        controls.metric("cell_indices", "Cell Indices", this->state->display.show_cell_indices && this->state->exports.cell_indices.has_value() ? "visible" : this->state->display.show_cell_indices ? "pending" : "hidden").section(section_display_id);
        controls.metric("domain", "Domain", this->state->display.show_domain ? "visible" : "hidden").section(section_display_id);
        controls.metric("volume_mib", "Volume MiB", static_cast<double>(this->state->exports.volume_byte_size) / 1048576.0).section(section_display_id);
        controls.metric("cell_index_mib", "Cell Index MiB", static_cast<double>(this->state->exports.cell_indices_byte_size) / 1048576.0).section(section_display_id);
        controls.metric("volume_revision", "Volume Rev", this->state->exports.volume_revision).section(section_statistics_id);
        controls.metric("cell_index_revision", "Cell Index Rev", this->state->exports.cell_indices_revision).section(section_statistics_id);
        if (this->state->latest_step_stats.has_value()) {
            controls.metric("step", "Step", this->state->latest_step_stats->step).section(section_statistics_id).display_primary().color({0.16f, 0.86f, 0.55f, 1.0f});
            controls.metric("step_elapsed_ms", "Step ms", std::format("{:.3f}", this->state->latest_step_stats->elapsed_ms)).section(section_statistics_id);
        }
        if (this->state->latest_smoke_stats.has_value()) {
            const SmokeStats& stats = *this->state->latest_smoke_stats;
            controls.metric("density_sum", "Density Sum", std::format("{:.6g}", stats.density.sum)).section(section_statistics_id);
            controls.metric("density_max", "Density Max", std::format("{:.6g}", stats.density.max)).section(section_statistics_id);
            controls.metric("density_nonzero", "Density Nonzero", stats.density.nonzero_count).section(section_statistics_id);
            controls.metric("temperature_sum", "Temperature Sum", std::format("{:.6g}", stats.temperature.sum)).section(section_statistics_id);
            controls.metric("temperature_max", "Temperature Max", std::format("{:.6g}", stats.temperature.max)).section(section_statistics_id);
        }
    }

    void Project::reset_simulation() {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        reset_solver(*this->state);
        ++this->state->scene_revision;
    }

    void Project::set_scene(const std::string_view value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        const ScenePreset preset = parse_scene_preset(value);
        if (this->state->scene.preset == preset) return;
        this->state->scene.preset = preset;
        update_scene_colliders(*this->state, this->state->simulation_time_seconds);
        rasterize_current_colliders(*this->state);
        ++this->state->scene_revision;
    }

    void Project::set_delta_seconds(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke delta seconds must be finite and positive."};
        if (this->state->simulation.delta_seconds == value) return;
        this->state->simulation.delta_seconds = value;
        ++this->state->scene_revision;
    }

    void Project::set_steps_per_update(const std::uint64_t value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (value == 0u || value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"Keyframe smoke steps per update must fit positive int32."};
        if (this->state->simulation.steps_per_update == static_cast<std::int32_t>(value)) return;
        this->state->simulation.steps_per_update = static_cast<std::int32_t>(value);
        ++this->state->scene_revision;
    }

    void Project::set_vorticity_confinement(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value < 0.0f) throw std::runtime_error{"Keyframe smoke vorticity confinement must be finite and non-negative."};
        if (this->state->simulation.vorticity_confinement == value) return;
        this->state->simulation.vorticity_confinement = value;
        this->state->smoke->vorticity.confinement     = value;
        ++this->state->scene_revision;
    }

    void Project::set_ambient_temperature(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke ambient temperature must be finite."};
        if (this->state->simulation.ambient_temperature == value) return;
        this->state->simulation.ambient_temperature = value;
        this->state->smoke->ambient_temperature     = value;
        ++this->state->scene_revision;
    }

    void Project::set_buoyancy_density_factor(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke density buoyancy factor must be finite."};
        if (this->state->simulation.buoyancy_density_factor == value) return;
        this->state->simulation.buoyancy_density_factor = value;
        this->state->smoke->buoyancy_density_factor     = value;
        ++this->state->scene_revision;
    }

    void Project::set_buoyancy_temperature_factor(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke temperature buoyancy factor must be finite."};
        if (this->state->simulation.buoyancy_temperature_factor == value) return;
        this->state->simulation.buoyancy_temperature_factor = value;
        this->state->smoke->buoyancy_temperature_factor     = value;
        ++this->state->scene_revision;
    }

    void Project::set_density_emission_rate(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value < 0.0f) throw std::runtime_error{"Keyframe smoke density emission rate must be finite and non-negative."};
        if (this->state->simulation.density_emission_rate == value) return;
        this->state->simulation.density_emission_rate = value;
        this->state->smoke->density_emission_rate     = value;
        ++this->state->scene_revision;
    }

    void Project::set_temperature_emission_rate(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value < 0.0f) throw std::runtime_error{"Keyframe smoke temperature emission rate must be finite and non-negative."};
        if (this->state->simulation.temperature_emission_rate == value) return;
        this->state->simulation.temperature_emission_rate = value;
        this->state->smoke->temperature_emission_rate     = value;
        ++this->state->scene_revision;
    }

    void Project::set_emitter_center_x(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke emitter center x must be finite."};
        if (this->state->emitter.center[0] == value) return;
        this->state->emitter.center[0] = value;
        this->state->smoke->emitter.source.region.center = this->state->emitter.center;
        ++this->state->scene_revision;
    }

    void Project::set_emitter_center_y(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke emitter center y must be finite."};
        if (this->state->emitter.center[1] == value) return;
        this->state->emitter.center[1] = value;
        this->state->smoke->emitter.source.region.center = this->state->emitter.center;
        ++this->state->scene_revision;
    }

    void Project::set_emitter_center_z(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke emitter center z must be finite."};
        if (this->state->emitter.center[2] == value) return;
        this->state->emitter.center[2] = value;
        this->state->smoke->emitter.source.region.center = this->state->emitter.center;
        ++this->state->scene_revision;
    }

    void Project::set_emitter_radius_x(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke emitter radius x must be finite and positive."};
        if (this->state->emitter.radius[0] == value) return;
        this->state->emitter.radius[0] = value;
        this->state->smoke->emitter.source.region.radius = this->state->emitter.radius;
        ++this->state->scene_revision;
    }

    void Project::set_emitter_radius_y(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke emitter radius y must be finite and positive."};
        if (this->state->emitter.radius[1] == value) return;
        this->state->emitter.radius[1] = value;
        this->state->smoke->emitter.source.region.radius = this->state->emitter.radius;
        ++this->state->scene_revision;
    }

    void Project::set_emitter_radius_z(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke emitter radius z must be finite and positive."};
        if (this->state->emitter.radius[2] == value) return;
        this->state->emitter.radius[2] = value;
        this->state->smoke->emitter.source.region.radius = this->state->emitter.radius;
        ++this->state->scene_revision;
    }

    void Project::set_emitter_falloff(const float value) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke emitter falloff must be finite and positive."};
        if (this->state->emitter.falloff == value) return;
        this->state->emitter.falloff = value;
        this->state->smoke->emitter.source.falloff = value;
        ++this->state->scene_revision;
    }

    void Project::set_show_emitter(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->display.show_emitter == value) return;
        this->state->display.show_emitter = value;
        ++this->state->scene_revision;
    }

    void Project::set_show_colliders(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->display.show_colliders == value) return;
        this->state->display.show_colliders = value;
        ++this->state->scene_revision;
    }

    void Project::set_collider_scale(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke collider scale must be finite and positive."};
        if (this->state->scene.collider_scale == value) return;
        this->state->scene.collider_scale = value;
        update_scene_colliders(*this->state, this->state->simulation_time_seconds);
        rasterize_current_colliders(*this->state);
        ++this->state->scene_revision;
    }

    void Project::set_moving_gate_amplitude(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value < 0.0f) throw std::runtime_error{"Keyframe smoke moving gate amplitude must be finite and non-negative."};
        if (this->state->scene.moving_gate_amplitude == value) return;
        this->state->scene.moving_gate_amplitude = value;
        update_scene_colliders(*this->state, this->state->simulation_time_seconds);
        rasterize_current_colliders(*this->state);
        ++this->state->scene_revision;
    }

    void Project::set_moving_gate_speed(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value < 0.0f) throw std::runtime_error{"Keyframe smoke moving gate speed must be finite and non-negative."};
        if (this->state->scene.moving_gate_speed == value) return;
        this->state->scene.moving_gate_speed = value;
        update_scene_colliders(*this->state, this->state->simulation_time_seconds);
        rasterize_current_colliders(*this->state);
        ++this->state->scene_revision;
    }

    void Project::set_constraint_scalar(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke constraint scalar must be finite."};
        if (this->state->scene.constraint_scalar == value) return;
        this->state->scene.constraint_scalar = value;
        update_scene_colliders(*this->state, this->state->simulation_time_seconds);
        rasterize_current_colliders(*this->state);
        ++this->state->scene_revision;
    }

    void Project::set_show_density(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->display.show_density == value) return;
        this->state->display.show_density = value;
        this->state->exports.volume.reset();
        this->state->exports.volume_revision     = 0u;
        this->state->exports.volume_byte_size    = 0u;
        this->state->exports.volume_channel_mask = 0u;
        ++this->state->scene_revision;
    }

    void Project::set_density_scale(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke density scale must be finite and positive."};
        if (this->state->display.density_scale == value) return;
        this->state->display.density_scale = value;
        ++this->state->scene_revision;
    }

    void Project::set_show_temperature(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->display.show_temperature == value) return;
        this->state->display.show_temperature = value;
        this->state->exports.volume.reset();
        this->state->exports.volume_revision     = 0u;
        this->state->exports.volume_byte_size    = 0u;
        this->state->exports.volume_channel_mask = 0u;
        ++this->state->scene_revision;
    }

    void Project::set_temperature_scale(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke temperature scale must be finite and positive."};
        if (this->state->display.temperature_scale == value) return;
        this->state->display.temperature_scale = value;
        ++this->state->scene_revision;
    }

    void Project::set_show_cell_indices(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->display.show_cell_indices == value) return;
        this->state->display.show_cell_indices = value;
        this->state->exports.volume.reset();
        this->state->exports.cell_indices.reset();
        this->state->exports.volume_revision        = 0u;
        this->state->exports.volume_byte_size       = 0u;
        this->state->exports.volume_channel_mask    = 0u;
        this->state->exports.cell_indices_revision  = 0u;
        this->state->exports.cell_indices_byte_size = 0u;
        ++this->state->scene_revision;
    }

    void Project::set_cell_index_cell_scale(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f || value > 1.0f) throw std::runtime_error{"Keyframe smoke cell index cell scale must be in (0, 1]."};
        if (this->state->display.cell_index_cell_scale == value) return;
        this->state->display.cell_index_cell_scale = value;
        this->state->exports.cell_indices.reset();
        this->state->exports.cell_indices_revision  = 0u;
        this->state->exports.cell_indices_byte_size = 0u;
        ++this->state->scene_revision;
    }

    void Project::set_show_domain(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->display.show_domain == value) return;
        this->state->display.show_domain = value;
        ++this->state->scene_revision;
    }

    void Project::set_camera_yaw_degrees(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value)) throw std::runtime_error{"Keyframe smoke camera yaw must be finite."};
        if (this->state->display.camera_yaw_degrees == value) return;
        this->state->display.camera_yaw_degrees = value;
        ++this->state->scene_revision;
    }

    void Project::set_camera_pitch_degrees(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= -89.0f || value >= 89.0f) throw std::runtime_error{"Keyframe smoke camera pitch must be finite and inside (-89, 89)."};
        if (this->state->display.camera_pitch_degrees == value) return;
        this->state->display.camera_pitch_degrees = value;
        ++this->state->scene_revision;
    }

    void Project::set_camera_distance(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke camera distance must be finite and positive."};
        if (this->state->display.camera_distance == value) return;
        this->state->display.camera_distance = value;
        ++this->state->scene_revision;
    }

    void Project::set_camera_fov_degrees(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f || value >= 180.0f) throw std::runtime_error{"Keyframe smoke camera FOV must be finite and inside (0, 180)."};
        if (this->state->display.camera_fov_degrees == value) return;
        this->state->display.camera_fov_degrees = value;
        ++this->state->scene_revision;
    }
} // namespace kfs::project

extern "C" SPECTRA_SCENE_EXPORT auto spectra_scene_plugin_v17(void) -> decltype(kfs::plugin::export_plugin<kfs::project::Project>()) {
    return kfs::plugin::export_plugin<kfs::project::Project>();
}
