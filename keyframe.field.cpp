module;
#include "keyframe.field.h"

#include <cuda_runtime.h>

module keyframe.field;
import std;

namespace kfs::field {
    namespace {
        std::uint64_t cell_element_count(const std::array<std::int32_t, 3>& resolution) {
            if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) return 0u;
            if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"invalid field resolution"};
            const auto count = static_cast<std::uint64_t>(resolution[0]) * static_cast<std::uint64_t>(resolution[1]) * static_cast<std::uint64_t>(resolution[2]);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"field element count exceeds int range"};
            return count;
        }

        std::uint64_t face_element_count(const std::array<std::int32_t, 3>& resolution, const std::uint32_t axis) {
            if (axis >= 3u) throw std::runtime_error{"invalid axis"};
            if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) return 0u;
            if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"invalid field resolution"};
            const auto nx    = static_cast<std::uint64_t>(resolution[0]);
            const auto ny    = static_cast<std::uint64_t>(resolution[1]);
            const auto nz    = static_cast<std::uint64_t>(resolution[2]);
            const auto count = axis == 0u ? (nx + 1u) * ny * nz : axis == 1u ? nx * (ny + 1u) * nz : nx * ny * (nz + 1u);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"field element count exceeds int range"};
            return count;
        }

        void copy_device_buffer(const cudaStream_t stream, float* const destination, const float* const source, const std::size_t bytes, const char* name) {
            if (const cudaError_t status = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync "} + name + ": " + cudaGetErrorString(status)};
        }

        void upload_device_buffer(const cudaStream_t stream, float* const destination, const std::span<const float> source, const std::size_t bytes, const char* name) {
            if (const cudaError_t status = cudaMemcpyAsync(destination, source.data(), bytes, cudaMemcpyHostToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync "} + name + ": " + cudaGetErrorString(status)};
        }
    } // namespace

    ScalarField3D::ScalarField3D(const std::array<std::int32_t, 3> resolution) : resolution{} {
        this->resize(resolution);
    }

    ScalarField3D::~ScalarField3D() noexcept {
        if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
        this->resolution   = {};
        this->data         = nullptr;
        this->storage_kind = ScalarFieldStorageKind::owned;
    }

    ScalarField3D::ScalarField3D(ScalarField3D&& other) noexcept : resolution{std::exchange(other.resolution, std::array<std::int32_t, 3>{})}, data{std::exchange(other.data, nullptr)}, storage_kind{std::exchange(other.storage_kind, ScalarFieldStorageKind::owned)} {}

    ScalarField3D& ScalarField3D::operator=(ScalarField3D&& other) noexcept {
        if (this == &other) return *this;
        if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
        this->resolution   = std::exchange(other.resolution, std::array<std::int32_t, 3>{});
        this->data         = std::exchange(other.data, nullptr);
        this->storage_kind = std::exchange(other.storage_kind, ScalarFieldStorageKind::owned);
        return *this;
    }

    [[nodiscard]] std::uint64_t ScalarField3D::count() const {
        return cell_element_count(this->resolution);
    }

    [[nodiscard]] std::size_t ScalarField3D::bytes() const {
        return cell_element_count(this->resolution) * sizeof(float);
    }

    void ScalarField3D::resize(const std::array<std::int32_t, 3> resolution) {
        if (this->resolution == resolution && this->storage_kind == ScalarFieldStorageKind::owned) return;
        if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) {
            if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
            this->resolution   = {};
            this->data         = nullptr;
            this->storage_kind = ScalarFieldStorageKind::owned;
            return;
        }

        float* next = nullptr;
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next), cell_element_count(resolution) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc scalar field: "} + cudaGetErrorString(status)};
        if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
        this->data         = next;
        this->resolution   = resolution;
        this->storage_kind = ScalarFieldStorageKind::owned;
    }

    void ScalarField3D::bind_external(const std::array<std::int32_t, 3> resolution, float* const data) {
        if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) throw std::runtime_error{"external field resolution is empty"};
        static_cast<void>(cell_element_count(resolution));
        if (data == nullptr) throw std::runtime_error{"external field data is null"};
        if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
        this->resolution   = resolution;
        this->data         = data;
        this->storage_kind = ScalarFieldStorageKind::external;
    }

    void copy(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count() == 0u || destination.data == nullptr) throw std::runtime_error{"destination field is empty"};
        if (source.count() == 0u || source.data == nullptr) throw std::runtime_error{"source field is empty"};
        copy_device_buffer(stream, destination.data, source.data, destination.bytes(), "scalar copy");
    }

    void add_scaled(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& current, const ScalarField3D& source, const float scale) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.resolution != current.resolution || destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count() == 0u || destination.data == nullptr) throw std::runtime_error{"destination field is empty"};
        if (current.count() == 0u || current.data == nullptr) throw std::runtime_error{"current field is empty"};
        if (source.count() == 0u || source.data == nullptr) throw std::runtime_error{"source field is empty"};
        cuda::field::add_scaled(stream, destination.data, current.data, source.data, destination.count(), scale);
    }

    void fill(cudaStream_t stream, ScalarField3D& values, const float value) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (values.count() == 0u || values.data == nullptr) throw std::runtime_error{"values field is empty"};
        cuda::field::fill(stream, values.data, values.count(), value);
    }

    void upload(cudaStream_t stream, ScalarField3D& destination, const std::span<const float> source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.count() == 0u || destination.data == nullptr) throw std::runtime_error{"destination field is empty"};
        if (static_cast<std::uint64_t>(source.size()) != destination.count()) throw std::runtime_error{"source span size mismatch"};
        if (destination.count() > 0u && source.data() == nullptr) throw std::runtime_error{"source span data is null"};
        upload_device_buffer(stream, destination.data, source, destination.bytes(), "scalar upload");
    }

    CenteredVectorField3D::CenteredVectorField3D(const std::array<std::int32_t, 3> resolution) : resolution{} {
        this->resize(resolution);
    }

    CenteredVectorField3D::~CenteredVectorField3D() noexcept {
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = {};
    }

    CenteredVectorField3D::CenteredVectorField3D(CenteredVectorField3D&& other) noexcept : resolution{std::exchange(other.resolution, std::array<std::int32_t, 3>{})}, data{std::exchange(other.data, std::array<float*, 3>{})} {}

    CenteredVectorField3D& CenteredVectorField3D::operator=(CenteredVectorField3D&& other) noexcept {
        if (this == &other) return *this;
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = std::exchange(other.resolution, std::array<std::int32_t, 3>{});
        this->data       = std::exchange(other.data, std::array<float*, 3>{});
        return *this;
    }

    [[nodiscard]] std::uint64_t CenteredVectorField3D::count() const {
        return cell_element_count(this->resolution);
    }

    [[nodiscard]] std::size_t CenteredVectorField3D::bytes() const {
        return cell_element_count(this->resolution) * sizeof(float);
    }

    void CenteredVectorField3D::resize(const std::array<std::int32_t, 3> resolution) {
        if (this->resolution == resolution) return;
        if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) {
            cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->resolution = {};
            return;
        }

        std::array<float*, 3> next{};
        try {
            const std::size_t bytes = cell_element_count(resolution) * sizeof(float);
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next[axis]), bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc centered vector field: "} + cudaGetErrorString(status)};
            }
        } catch (...) {
            cuda::free_device_buffers(next[0], next[1], next[2]);
            throw;
        }
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->data       = next;
        this->resolution = resolution;
    }

    void fill(cudaStream_t stream, CenteredVectorField3D& values, const float value) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (values.count() == 0u) throw std::runtime_error{"values field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (values.data[axis] == nullptr) throw std::runtime_error{"values field component is empty"};
        const std::uint64_t count = values.count();
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::field::fill(stream, values.data[axis], count, value);
    }

    void copy(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count() == 0u) throw std::runtime_error{"destination field is empty"};
        if (source.count() == 0u) throw std::runtime_error{"source field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
            if (source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        }
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) copy_device_buffer(stream, destination.data[axis], source.data[axis], destination.bytes(), "centered vector copy");
    }

    void copy_component(cudaStream_t stream, CenteredVectorField3D& destination, const std::uint32_t axis, const CenteredVectorField3D& source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count() == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        if (source.count() == 0u || source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        copy_device_buffer(stream, destination.data[axis], source.data[axis], destination.bytes(), "centered vector component copy");
    }

    void add_scaled(cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& current, const CenteredVectorField3D& source, const float scale) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.resolution != current.resolution || destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count() == 0u) throw std::runtime_error{"destination field is empty"};
        if (current.count() == 0u) throw std::runtime_error{"current field is empty"};
        if (source.count() == 0u) throw std::runtime_error{"source field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
            if (current.data[axis] == nullptr) throw std::runtime_error{"current field component is empty"};
            if (source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        }
        const std::uint64_t count = destination.count();
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::field::add_scaled(stream, destination.data[axis], current.data[axis], source.data[axis], count, scale);
    }

    void upload(cudaStream_t stream, CenteredVectorField3D& destination, const std::array<std::span<const float>, 3> source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.count() == 0u) throw std::runtime_error{"destination field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) upload_component(stream, destination, axis, source[axis]);
    }

    void upload_component(cudaStream_t stream, CenteredVectorField3D& destination, const std::uint32_t axis, const std::span<const float> source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        if (destination.count() == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        if (static_cast<std::uint64_t>(source.size()) != destination.count()) throw std::runtime_error{"source span size mismatch"};
        if (destination.count() > 0u && source.data() == nullptr) throw std::runtime_error{"source span data is null"};
        upload_device_buffer(stream, destination.data[axis], source, destination.bytes(), "centered vector component upload");
    }

    void center_staggered(cudaStream_t stream, CenteredVectorField3D& destination, const StaggeredVectorField3D& source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count() == 0u) throw std::runtime_error{"destination field is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
            if (source.count(axis) == 0u || source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        }
        cuda::field::center_staggered(stream, destination.data[0], destination.data[1], destination.data[2], source.data[0], source.data[1], source.data[2], destination.resolution);
    }

    StaggeredVectorField3D::StaggeredVectorField3D(const std::array<std::int32_t, 3> resolution) : resolution{} {
        this->resize(resolution);
    }

    StaggeredVectorField3D::~StaggeredVectorField3D() noexcept {
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = {};
    }

    StaggeredVectorField3D::StaggeredVectorField3D(StaggeredVectorField3D&& other) noexcept : resolution{std::exchange(other.resolution, std::array<std::int32_t, 3>{})}, data{std::exchange(other.data, std::array<float*, 3>{})} {}

    StaggeredVectorField3D& StaggeredVectorField3D::operator=(StaggeredVectorField3D&& other) noexcept {
        if (this == &other) return *this;
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = std::exchange(other.resolution, std::array<std::int32_t, 3>{});
        this->data       = std::exchange(other.data, std::array<float*, 3>{});
        return *this;
    }

    [[nodiscard]] std::uint64_t StaggeredVectorField3D::count(const std::uint32_t axis) const {
        return face_element_count(this->resolution, axis);
    }

    [[nodiscard]] std::size_t StaggeredVectorField3D::bytes(const std::uint32_t axis) const {
        return face_element_count(this->resolution, axis) * sizeof(float);
    }

    void StaggeredVectorField3D::resize(const std::array<std::int32_t, 3> resolution) {
        if (this->resolution == resolution) return;
        if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) {
            cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->resolution = {};
            return;
        }

        std::array<float*, 3> next{};
        try {
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next[axis]), face_element_count(resolution, axis) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc staggered vector field: "} + cudaGetErrorString(status)};
            }
        } catch (...) {
            cuda::free_device_buffers(next[0], next[1], next[2]);
            throw;
        }
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->data       = next;
        this->resolution = resolution;
    }

    void fill(cudaStream_t stream, StaggeredVectorField3D& values, const float value) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (values.count(axis) == 0u || values.data[axis] == nullptr) throw std::runtime_error{"values field component is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::field::fill(stream, values.data[axis], values.count(axis), value);
    }

    void copy(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
            if (source.count(axis) == 0u || source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        }
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) copy_device_buffer(stream, destination.data[axis], source.data[axis], destination.bytes(axis), "staggered vector copy");
    }

    void copy_component(cudaStream_t stream, StaggeredVectorField3D& destination, const std::uint32_t axis, const StaggeredVectorField3D& source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        if (source.count(axis) == 0u || source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        copy_device_buffer(stream, destination.data[axis], source.data[axis], destination.bytes(axis), "staggered vector component copy");
    }

    void add_scaled(cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& current, const StaggeredVectorField3D& source, const float scale) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (destination.resolution != current.resolution || destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
            if (current.count(axis) == 0u || current.data[axis] == nullptr) throw std::runtime_error{"current field component is empty"};
            if (source.count(axis) == 0u || source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        }
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::field::add_scaled(stream, destination.data[axis], current.data[axis], source.data[axis], destination.count(axis), scale);
    }

    void upload(cudaStream_t stream, StaggeredVectorField3D& destination, const std::array<std::span<const float>, 3> source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) upload_component(stream, destination, axis, source[axis]);
    }

    void upload_component(cudaStream_t stream, StaggeredVectorField3D& destination, const std::uint32_t axis, const std::span<const float> source) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        if (static_cast<std::uint64_t>(source.size()) != destination.count(axis)) throw std::runtime_error{"source span size mismatch"};
        if (destination.count(axis) > 0u && source.data() == nullptr) throw std::runtime_error{"source span data is null"};
        upload_device_buffer(stream, destination.data[axis], source, destination.bytes(axis), "staggered vector component upload");
    }

    void add_centered_to_staggered(cudaStream_t stream, StaggeredVectorField3D& destination, const std::uint32_t axis, const CenteredVectorField3D& source, const float scale) {
        if (stream == nullptr) throw std::runtime_error{"field stream must not be null"};
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.count(axis) == 0u || destination.data[axis] == nullptr) throw std::runtime_error{"destination field component is empty"};
        if (source.count() == 0u || source.data[axis] == nullptr) throw std::runtime_error{"source field component is empty"};
        cuda::field::add_centered_to_staggered(stream, destination.data[axis], source.data[axis], axis, destination.resolution, scale);
    }
} // namespace kfs::field
