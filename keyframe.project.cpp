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

import keyframe.field;
import keyframe.plugin;
import keyframe.solver;
import keyframe.collider;
import keyframe.geometry;
import std;

namespace kfs::project {
    namespace {
        constexpr char scene_free_plume[]                    = "free_plume";
        constexpr char scene_obstacle_gallery[]              = "obstacle_gallery";
        constexpr char scene_moving_gate[]                   = "moving_gate";
        constexpr char section_scene_id[]                    = "scene";
        constexpr char section_simulation_id[]               = "simulation";
        constexpr char section_view_id[]                     = "view";
        constexpr char section_statistics_id[]               = "statistics";
        constexpr char option_scene_key[]                    = "scene";
        constexpr char setting_show_volume_key[]             = "show_volume";
        constexpr char setting_density_scale_key[]           = "density_scale";
        constexpr char setting_show_domain_key[]             = "show_domain";
        constexpr char setting_show_scene_geometry_key[]     = "show_scene_geometry";
        constexpr char density_volume_name[]                 = "Keyframe Smoke Density";
        constexpr char density_material_name[]               = "Keyframe Smoke Density Material";
        constexpr char emitter_material_name[]               = "Keyframe Emitter Material";
        constexpr char collider_box_material_name[]          = "Keyframe Collider Box Material";
        constexpr char collider_sphere_material_name[]       = "Keyframe Collider Sphere Material";
        constexpr char density_light_name[]                  = "Keyframe Smoke Key Light";
        constexpr char domain_segments_name[]                = "Keyframe Smoke Domain";
        constexpr char emitter_entity_name[]                 = "Keyframe Emitter";
        constexpr float default_density_scale                = 4.0f;

        enum class ScenePreset : std::uint32_t {
            free_plume,
            obstacle_gallery,
            moving_gate,
        };

        struct DebugOptions final {
            bool show_volume{true};
            bool show_domain{true};
            bool show_scene_geometry{true};
            float density_scale{default_density_scale};
        };

        struct OpenOptions final {
            ScenePreset preset{ScenePreset::obstacle_gallery};
            std::array<std::uint32_t, 3> resolution{64, 96, 64};
            float vorticity_confinement{0.22f};
            float delta_seconds{1.0f / 60.0f};
            std::int32_t steps_per_update{1};
            float density_scale{default_density_scale};
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
                    parsed.preset = parse_scene_preset(option.value);
                else if (option.key == "resolution_x")
                    parsed.resolution[0] = parse_u32_option(option.value, "resolution_x");
                else if (option.key == "resolution_y")
                    parsed.resolution[1] = parse_u32_option(option.value, "resolution_y");
                else if (option.key == "resolution_z")
                    parsed.resolution[2] = parse_u32_option(option.value, "resolution_z");
                else if (option.key == "dt")
                    parsed.delta_seconds = parse_float_option(option.value, "dt");
                else if (option.key == "steps_per_update") {
                    const std::uint32_t steps = parse_u32_option(option.value, "steps_per_update");
                    if (steps > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"steps_per_update must fit int32."};
                    parsed.steps_per_update = static_cast<std::int32_t>(steps);
                } else if (option.key == "density_scale")
                    parsed.density_scale = parse_float_option(option.value, "density_scale");
                else if (option.key == "vorticity_confinement")
                    parsed.vorticity_confinement = parse_float_option(option.value, "vorticity_confinement");
                else
                    throw std::runtime_error{std::format("unknown Keyframe smoke open option '{}'.", option.key)};
            }
            if (parsed.delta_seconds <= 0.0f) throw std::runtime_error{"dt must be positive."};
            if (parsed.steps_per_update < 1) throw std::runtime_error{"steps_per_update must be positive."};
            if (parsed.density_scale <= 0.0f) throw std::runtime_error{"density_scale must be positive."};
            if (!std::isfinite(parsed.vorticity_confinement) || parsed.vorticity_confinement < 0.0f) throw std::runtime_error{"vorticity_confinement must be finite and non-negative."};
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

        [[nodiscard]] plugin::Camera overview_camera(const solver::Solver& smoke) {
            const std::array size = domain_size(smoke);
            return plugin::Camera{
                .name                 = "Overview",
                .position             = {size[0] * 0.5f, size[1] * 0.58f, std::max(size[2] * 2.6f, 1.0f)},
                .right                = {1.0f, 0.0f, 0.0f},
                .down                 = {0.0f, -1.0f, 0.0f},
                .forward              = {0.0f, 0.0f, -1.0f},
                .projection           = plugin::CameraProjection::Perspective,
                .vertical_fov_degrees = 45.0f,
                .near_plane           = 0.01f,
                .far_plane            = std::max({size[0], size[1], size[2]}) * 8.0f,
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
                .owner = plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::Camera, .name = "Overview"},
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

        [[nodiscard]] plugin::Sphere ellipsoid_entity(const std::string& name, const geometry::Ellipsoid& ellipsoid, const std::string& material_name) {
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

        [[nodiscard]] plugin::Mesh box_entity(const std::string& name, const geometry::Box& box, const std::string& material_name) {
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

        struct SmokeStats final {
            field::ScalarFieldStats density{};
            field::ScalarFieldStats temperature{};
        };
    } // namespace

    struct Project::State final {
        OpenOptions open{};
        DebugOptions debug{};
        ScenePreset preset{ScenePreset::obstacle_gallery};
        std::shared_ptr<plugin::HostServices> host_services{};
        ExternalGpuBuffer density_buffer{};
        std::unique_ptr<solver::Solver> smoke{};
        std::optional<plugin::VolumeGrid> density_volume{};
        std::optional<plugin::ViewportSegmentSet> domain_segments{};
        plugin::DebugAttachmentSet debug_attachments{};
        std::optional<solver::StepStats> latest_step_stats{};
        std::optional<SmokeStats> latest_smoke_stats{};
        std::uint64_t scene_revision{1u};
        std::uint64_t exported_density_revision{};
        std::uint64_t exported_density_byte_size{};
        double simulation_time_seconds{};
        bool density_external_ready{};
        bool host_update_running{};
    };

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

    namespace {
        void apply_scene_preset(Project::State& state, const double simulation_time_seconds) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            std::vector<collider::Collider>& colliders = state.smoke->colliders.items;
            colliders.clear();
            const std::array size = domain_size(*state.smoke);
            switch (state.preset) {
            case ScenePreset::free_plume:
                return;
            case ScenePreset::obstacle_gallery:
                colliders.push_back(collider::Collider{
                    .shape = geometry::Box{
                        .center      = {size[0] * 0.50f, size[1] * 0.40f, size[2] * 0.50f},
                        .half_extent = {size[0] * 0.075f, size[1] * 0.18f, size[2] * 0.34f},
                    },
                    .constraint_scalar = 0.0f,
                });
                colliders.push_back(collider::Collider{
                    .shape = geometry::Ellipsoid{
                        .center = {size[0] * 0.68f, size[1] * 0.34f, size[2] * 0.52f},
                        .radius = {size[0] * 0.10f, size[1] * 0.08f, size[2] * 0.18f},
                    },
                    .constraint_scalar = 0.0f,
                });
                return;
            case ScenePreset::moving_gate: {
                constexpr float angular_speed = 1.4f;
                const float phase             = static_cast<float>(simulation_time_seconds) * angular_speed;
                colliders.push_back(collider::Collider{
                    .shape = geometry::Box{
                        .center =
                            {
                                size[0] * (0.50f + 0.24f * std::sin(phase)),
                                size[1] * 0.38f,
                                size[2] * 0.50f,
                            },
                        .half_extent = {size[0] * 0.07f, size[1] * 0.18f, size[2] * 0.34f},
                    },
                    .velocity          = {size[0] * 0.24f * angular_speed * std::cos(phase), 0.0f, 0.0f},
                    .constraint_scalar = 0.0f,
                });
                return;
            }
            }
            throw std::runtime_error{"scene preset is invalid."};
        }

        void rebuild_debug_attachments(Project::State& state) {
            state.debug_attachments.viewport_segment_sets.clear();
            if (state.domain_segments.has_value()) state.debug_attachments.viewport_segment_sets.push_back(*state.domain_segments);
        }

        void publish_domain_if_ready(Project::State& state) {
            state.domain_segments.reset();
            if (state.debug.show_domain && state.smoke != nullptr) state.domain_segments = domain_box(*state.smoke);
            rebuild_debug_attachments(state);
        }

        void prepare_density_external_buffer(Project::State& state) {
            if (state.smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
            const auto& density          = state.smoke->device.density_data;
            const std::uint64_t byte_size = static_cast<std::uint64_t>(density.bytes());
            state.density_buffer.ensure(state.host_services, plugin::GpuBufferKindVolumeChannel, byte_size, "keyframe smoke density volume", "density volume");
            float* const density_values = state.density_buffer.mapped_as<float>();
            if (density_values == nullptr) throw std::runtime_error{"Keyframe smoke density external buffer was not mapped."};

            if (state.density_external_ready) return;
            const cudaStream_t stream = state.smoke->stream;
            if (const cudaError_t status = cudaMemsetAsync(density_values, 0, byte_size, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemsetAsync Keyframe smoke density external buffer failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize Keyframe smoke density external buffer failed: "} + cudaGetErrorString(status)};
            state.density_external_ready     = true;
            state.exported_density_revision  = 0u;
            state.exported_density_byte_size = 0u;
            state.density_volume.reset();
        }

        [[nodiscard]] bool publish_density_if_ready(Project::State& state) {
            if (!state.debug.show_volume || state.smoke == nullptr || !state.latest_step_stats.has_value()) {
                state.density_volume.reset();
                state.exported_density_revision  = 0u;
                state.exported_density_byte_size = 0u;
                return true;
            }

            if (!state.density_external_ready) throw std::runtime_error{"Keyframe smoke density external buffer is not ready."};
            const auto& density           = state.smoke->device.density_data;
            const std::uint64_t byte_size = static_cast<std::uint64_t>(density.bytes());
            const std::uint64_t revision  = static_cast<std::uint64_t>(state.smoke->current_step) + 1u;
            if (state.density_volume.has_value() && state.exported_density_revision == revision && state.exported_density_byte_size == byte_size) return false;

            float* const density_values = state.density_buffer.mapped_as<float>();
            if (density_values == nullptr) throw std::runtime_error{"Keyframe smoke density external buffer was not mapped."};
            const cudaStream_t stream = state.smoke->stream;
            if (const cudaError_t status = cudaMemcpyAsync(density_values, density.data, byte_size, cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync Keyframe smoke density volume failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaStreamSynchronize Keyframe smoke density volume failed: "} + cudaGetErrorString(status)};
            const std::array dimensions{
                static_cast<std::uint32_t>(density.resolution[0]),
                static_cast<std::uint32_t>(density.resolution[1]),
                static_cast<std::uint32_t>(density.resolution[2]),
            };

            state.density_volume = plugin::VolumeGrid{
                .name       = density_volume_name,
                .dimensions = dimensions,
                .origin     = {0.0f, 0.0f, 0.0f},
                .voxel_size = {state.smoke->cell_size, state.smoke->cell_size, state.smoke->cell_size},
                .channels =
                    {
                        plugin::VolumeChannel{
                            .name                    = "density",
                            .dimensions              = dimensions,
                            .format                  = plugin::VolumeChannelFormat::Float32,
                            .source_kind             = plugin::VolumeChannelSourceKind::ExternalGpuBuffer,
                            .index_encoding          = plugin::VolumeChannelIndexEncoding::Linear,
                            .buffer_id               = state.density_buffer.resource_id(),
                            .external_device_pointer = reinterpret_cast<std::uintptr_t>(density_values),
                            .source_byte_size        = byte_size,
                            .revision                = revision,
                        },
                    },
                .material_name = density_material_name,
            };
            state.exported_density_revision  = revision;
            state.exported_density_byte_size = byte_size;
            return true;
        }

        void append_scene_geometry(plugin::Document& document, const solver::Solver& smoke) {
            document.spheres.push_back(ellipsoid_entity(emitter_entity_name, smoke.emitter.source.region, emitter_material_name));
            for (std::size_t index = 0u; index < smoke.colliders.items.size(); ++index) {
                const collider::Collider& item = smoke.colliders.items[index];
                if (const geometry::Ellipsoid* const ellipsoid = std::get_if<geometry::Ellipsoid>(&item.shape); ellipsoid != nullptr) {
                    document.spheres.push_back(ellipsoid_entity(std::format("Keyframe Collider Ellipsoid {}", index + 1u), *ellipsoid, collider_sphere_material_name));
                    continue;
                }
                if (const geometry::Box* const box = std::get_if<geometry::Box>(&item.shape); box != nullptr) {
                    document.meshes.push_back(box_entity(std::format("Keyframe Collider Box {}", index + 1u), *box, collider_box_material_name));
                }
            }
        }
    } // namespace

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
                    plugin::section(section_view_id, "View"),
                    plugin::section(section_statistics_id, "Statistics"),
                },
            .open_options =
                {
                    plugin::choice(option_scene_key, "Scene", {scene_free_plume, scene_obstacle_gallery, scene_moving_gate})
                            .defaulted(scene_obstacle_gallery)
                            .section(section_scene_id),
                    plugin::unsigned_integer("resolution_x", "Resolution X", 64u).section(section_simulation_id),
                    plugin::unsigned_integer("resolution_y", "Resolution Y", 96u).section(section_simulation_id),
                    plugin::unsigned_integer("resolution_z", "Resolution Z", 64u).section(section_simulation_id),
                    plugin::float_option("dt", "Delta Seconds", 1.0f / 60.0f).section(section_simulation_id),
                    plugin::unsigned_integer("steps_per_update", "Steps Per Update", 1u).section(section_simulation_id),
                    plugin::float_option("density_scale", "Density Scale", default_density_scale).section(section_view_id),
                    plugin::float_option("vorticity_confinement", "Vorticity Confinement", 0.22f).section(section_simulation_id),
                },
            .settings =
                {
                    plugin::toggle(setting_show_volume_key, "Show Volume", true, &Project::set_show_volume).section(section_view_id),
                    plugin::float_value(setting_density_scale_key, "Density Scale", default_density_scale, &Project::set_density_scale).section(section_view_id).slider(0.001f, 16.0f, 0.001f),
                    plugin::toggle(setting_show_domain_key, "Show Domain", true, &Project::set_show_domain).section(section_view_id),
                    plugin::toggle(setting_show_scene_geometry_key, "Show Scene Geometry", true, &Project::set_show_scene_geometry).section(section_view_id),
                },
        };
        return definition;
    }

    Project Project::open(plugin::OpenContext context) {
        OpenOptions options  = parse_open_options(std::span<const plugin::Option>{context.options});
        auto state           = std::make_unique<State>();
        state->open          = options;
        state->preset        = options.preset;
        state->debug         = DebugOptions{.density_scale = options.density_scale};
        state->host_services = std::move(context.host_services);
        state->smoke         = std::make_unique<solver::Solver>(options.resolution);
        state->smoke->vorticity.confinement = options.vorticity_confinement;
        apply_scene_preset(*state, state->simulation_time_seconds);
        publish_domain_if_ready(*state);
        state->latest_smoke_stats = SmokeStats{
            .density     = field::stats(state->smoke->stream, state->smoke->device.density_data),
            .temperature = field::stats(state->smoke->stream, state->smoke->device.temperature_data),
        };
        return Project{std::move(state)};
    }

    void Project::update(const plugin::UpdateInfo& update) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(update.wall_delta_seconds) || update.wall_delta_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project wall delta time is invalid."};
        if (!std::isfinite(update.update_delta_seconds) || update.update_delta_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project update delta time is invalid."};
        this->state->host_update_running = update.update_running;
        if (update.update_delta_seconds == 0.0) return;

        prepare_density_external_buffer(*this->state);
        const double step_seconds = static_cast<double>(this->state->open.delta_seconds) * static_cast<double>(this->state->open.steps_per_update);
        const double next_simulation_time_seconds = this->state->simulation_time_seconds + step_seconds;
        apply_scene_preset(*this->state, next_simulation_time_seconds);
        const std::expected<solver::StepStats, std::string> stats = this->state->smoke->step(solver::StepRequest{
            .delta_seconds = this->state->open.delta_seconds,
            .iterations    = this->state->open.steps_per_update,
        });
        if (!stats) throw std::runtime_error{stats.error()};
        this->state->simulation_time_seconds = next_simulation_time_seconds;
        this->state->latest_step_stats  = *stats;
        this->state->latest_smoke_stats = SmokeStats{
            .density     = field::stats(this->state->smoke->stream, this->state->smoke->device.density_data),
            .temperature = field::stats(this->state->smoke->stream, this->state->smoke->device.temperature_data),
        };
        publish_domain_if_ready(*this->state);
        static_cast<void>(publish_density_if_ready(*this->state));
        ++this->state->scene_revision;
    }

    std::uint64_t Project::revision() const {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        return this->state->scene_revision;
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        std::vector<plugin::Material> materials{
            plugin::Material{
                .name                 = density_material_name,
                .model                = "volume",
                .alpha_mode           = "blend",
                .base_color           = {1.0f, 1.0f, 1.0f, 1.0f},
                .roughness            = 0.35f,
                .volume_density_scale = this->state->debug.density_scale,
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
        std::vector<plugin::Light> lights{
            plugin::Light{
                .name      = density_light_name,
                .kind      = "directional",
                .color     = {1.0f, 1.0f, 1.0f},
                .intensity = 3.0f,
            },
        };

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
                    .step_delta_seconds = static_cast<double>(this->state->open.delta_seconds),
                },
            .navigation_target =
                plugin::ViewportNavigationTarget{
                    .revision       = 1u,
                    .focus          = {size[0] * 0.5f, size[1] * 0.5f, size[2] * 0.5f},
                    .bounds_minimum = {0.0f, 0.0f, 0.0f},
                    .bounds_maximum = size,
                    .navigation_up  = {0.0f, 1.0f, 0.0f},
                },
            .active_camera_name = "Overview",
            .materials          = std::move(materials),
            .lights             = std::move(lights),
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo frame) {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(frame.delta_seconds) || frame.delta_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project frame delta time is invalid."};
        if (!std::isfinite(frame.time_seconds) || frame.time_seconds < 0.0) throw std::runtime_error{"Keyframe smoke project frame time is invalid."};
        publish_domain_if_ready(*this->state);
        static_cast<void>(publish_density_if_ready(*this->state));

        plugin::Document document{
            .cameras           = {overview_camera(*this->state->smoke)},
            .debug_attachments = this->state->debug_attachments,
        };
        if (this->state->debug.show_volume && this->state->density_volume.has_value()) document.volumes.push_back(*this->state->density_volume);
        if (this->state->debug.show_scene_geometry) append_scene_geometry(document, *this->state->smoke);
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        if (this->state == nullptr || this->state->smoke == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        controls.phase(this->state->host_update_running ? "Running" : "Paused").headline(this->state->latest_step_stats.has_value() ? "Smoke simulation running" : "Smoke solver loaded").message(this->state->latest_step_stats.has_value() ? "Keyframe smoke advances on Spectra update ticks." : "Update clock is paused before the first simulation step.");
        const auto& resolution = this->state->smoke->device.density_data.resolution;
        controls.metric("scene", "Scene", scene_preset_name(this->state->preset)).section(section_scene_id).display_primary().color({1.0f, 0.58f, 0.20f, 1.0f});
        controls.metric("colliders", "Colliders", this->state->smoke->colliders.items.size()).section(section_scene_id);
        controls.metric("scene_geometry", "Scene Geometry", this->state->debug.show_scene_geometry ? "visible" : "hidden").section(section_scene_id);
        controls.metric("simulation_time", "Simulation Time", std::format("{:.3f}", this->state->simulation_time_seconds)).section(section_scene_id);
        controls.metric("resolution", "Resolution", std::format("{} x {} x {}", resolution[0], resolution[1], resolution[2])).section(section_simulation_id).display_primary().color({0.55f, 0.85f, 1.0f, 1.0f});
        controls.metric("cell_size", "Cell Size", std::format("{:.6f}", this->state->smoke->cell_size)).section(section_simulation_id);
        controls.metric("dt", "Delta Seconds", std::format("{:.6f}", this->state->open.delta_seconds)).section(section_simulation_id);
        controls.metric("steps_per_update", "Steps Per Update", this->state->open.steps_per_update).section(section_simulation_id);
        controls.metric("vorticity_confinement", "Vorticity", std::format("{:.6f}", this->state->smoke->vorticity.confinement)).section(section_simulation_id);
        controls.metric("volume", "Volume", this->state->debug.show_volume && this->state->density_volume.has_value() ? "visible" : this->state->debug.show_volume ? "pending" : "hidden").section(section_view_id);
        controls.metric("density_scale", "Density Scale", std::format("{:.3f}x", this->state->debug.density_scale)).section(section_view_id);
        controls.metric("domain", "Domain", this->state->domain_segments.has_value() ? "visible" : "hidden").section(section_view_id);
        controls.metric("density_volume_bytes", "Volume MiB", static_cast<double>(this->state->exported_density_byte_size) / 1048576.0).section(section_view_id);
        controls.metric("density_revision", "Density Rev", this->state->exported_density_revision).section(section_statistics_id);
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

    void Project::set_show_volume(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->debug.show_volume == value) return;
        this->state->debug.show_volume = value;
        this->state->density_volume.reset();
        this->state->exported_density_revision  = 0u;
        this->state->exported_density_byte_size = 0u;
        ++this->state->scene_revision;
    }

    void Project::set_density_scale(const float value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error{"Keyframe smoke density scale must be finite and positive."};
        if (this->state->debug.density_scale == value) return;
        this->state->debug.density_scale = value;
        ++this->state->scene_revision;
    }

    void Project::set_show_domain(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->debug.show_domain == value) return;
        this->state->debug.show_domain = value;
        publish_domain_if_ready(*this->state);
        ++this->state->scene_revision;
    }

    void Project::set_show_scene_geometry(const bool value) {
        if (this->state == nullptr) throw std::runtime_error{"Keyframe smoke project is not open."};
        if (this->state->debug.show_scene_geometry == value) return;
        this->state->debug.show_scene_geometry = value;
        ++this->state->scene_revision;
    }
} // namespace kfs::project

extern "C" SPECTRA_SCENE_EXPORT auto spectra_scene_plugin_v17(void) -> decltype(kfs::plugin::export_plugin<kfs::project::Project>()) {
    return kfs::plugin::export_plugin<kfs::project::Project>();
}
