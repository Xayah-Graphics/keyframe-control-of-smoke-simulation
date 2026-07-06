module;
#include "keyframe.field.h"
#include <cuda_runtime.h>

module keyframe.field;
import std;

namespace kfs::cuda {
    void free_device_buffers(void** const pointers, const std::size_t count) noexcept {
        if (pointers == nullptr) return;
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] == nullptr) continue;
            cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }
} // namespace kfs::cuda

namespace kfs::field {
    namespace {
        constexpr std::array<std::int32_t, 3> empty_resolution{0, 0, 0};

        std::uint64_t cell_element_count(const std::array<std::int32_t, 3>& resolution) {
            if (resolution == empty_resolution) return 0u;
            if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"invalid field resolution"};
            const auto count = static_cast<std::uint64_t>(resolution[0]) * static_cast<std::uint64_t>(resolution[1]) * static_cast<std::uint64_t>(resolution[2]);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"field element count exceeds int range"};
            return count;
        }

        std::uint64_t face_element_count(const std::array<std::int32_t, 3>& resolution, const std::uint32_t axis) {
            if (axis >= 3u) throw std::runtime_error{"invalid axis"};
            if (resolution == empty_resolution) return 0u;
            if (resolution[0] <= 0 || resolution[1] <= 0 || resolution[2] <= 0) throw std::runtime_error{"invalid field resolution"};
            const auto nx    = static_cast<std::uint64_t>(resolution[0]);
            const auto ny    = static_cast<std::uint64_t>(resolution[1]);
            const auto nz    = static_cast<std::uint64_t>(resolution[2]);
            const auto count = axis == 0u ? (nx + 1u) * ny * nz : axis == 1u ? nx * (ny + 1u) * nz : nx * ny * (nz + 1u);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"field element count exceeds int range"};
            return count;
        }
    } // namespace

    ScalarField3D::ScalarField3D(const std::array<std::int32_t, 3> resolution) : resolution{empty_resolution} {
        this->resize(resolution);
    }

    ScalarField3D::~ScalarField3D() noexcept {
        if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
        this->resolution   = empty_resolution;
        this->data         = nullptr;
        this->storage_kind = ScalarFieldStorageKind::owned;
    }

    ScalarField3D::ScalarField3D(ScalarField3D&& other) noexcept : resolution{std::exchange(other.resolution, empty_resolution)}, data{std::exchange(other.data, nullptr)}, storage_kind{std::exchange(other.storage_kind, ScalarFieldStorageKind::owned)} {}

    ScalarField3D& ScalarField3D::operator=(ScalarField3D&& other) noexcept {
        if (this == &other) return *this;
        if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
        this->resolution   = std::exchange(other.resolution, empty_resolution);
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
        if (resolution == empty_resolution) {
            if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
            this->resolution   = empty_resolution;
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
        if (resolution == empty_resolution) throw std::runtime_error{"external field resolution is empty"};
        static_cast<void>(cell_element_count(resolution));
        if (data == nullptr) throw std::runtime_error{"external field data is null"};
        if (this->storage_kind == ScalarFieldStorageKind::owned) cuda::free_device_buffers(this->data);
        this->resolution   = resolution;
        this->data         = data;
        this->storage_kind = ScalarFieldStorageKind::external;
    }

    void ScalarField3D::fill(cudaStream_t stream, const float value) {
        if (this->resolution == empty_resolution || this->data == nullptr) throw std::runtime_error{"field is empty"};
        cuda::field::fill(stream, this->data, this->count(), value);
    }

    void copy(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source) {
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.resolution == empty_resolution || destination.data == nullptr) throw std::runtime_error{"field is empty"};
        if (source.resolution == empty_resolution || source.data == nullptr) throw std::runtime_error{"field is empty"};
        if (const cudaError_t status = cudaMemcpyAsync(destination.data, source.data, destination.bytes(), cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync scalar copy: "} + cudaGetErrorString(status)};
    }

    void add_scaled(cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& current, const ScalarField3D& source, const float scale) {
        if (destination.resolution != current.resolution || destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.resolution == empty_resolution || destination.data == nullptr) throw std::runtime_error{"field is empty"};
        if (current.resolution == empty_resolution || current.data == nullptr) throw std::runtime_error{"field is empty"};
        if (source.resolution == empty_resolution || source.data == nullptr) throw std::runtime_error{"field is empty"};
        cuda::field::add_scaled(stream, destination.data, current.data, source.data, destination.count(), scale);
    }

    CenteredVectorField3D::CenteredVectorField3D(const std::array<std::int32_t, 3> resolution) : resolution{empty_resolution} {
        this->resize(resolution);
    }

    CenteredVectorField3D::~CenteredVectorField3D() noexcept {
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = empty_resolution;
    }

    CenteredVectorField3D::CenteredVectorField3D(CenteredVectorField3D&& other) noexcept : resolution{std::exchange(other.resolution, empty_resolution)}, data{std::exchange(other.data, std::array<float*, 3>{})} {}

    CenteredVectorField3D& CenteredVectorField3D::operator=(CenteredVectorField3D&& other) noexcept {
        if (this == &other) return *this;
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = std::exchange(other.resolution, empty_resolution);
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
        if (resolution == empty_resolution) {
            cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->resolution = empty_resolution;
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

    void CenteredVectorField3D::fill(cudaStream_t stream, const float value) {
        const std::uint64_t count = this->count();
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (this->resolution == empty_resolution || this->data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            cuda::field::fill(stream, this->data[axis], count, value);
        }
    }

    void center_staggered(cudaStream_t stream, CenteredVectorField3D& destination, const StaggeredVectorField3D& source) {
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (destination.resolution == empty_resolution || destination.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            if (source.resolution == empty_resolution || source.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
        }
        cuda::field::center_staggered(stream, destination.data[0], destination.data[1], destination.data[2], source.data[0], source.data[1], source.data[2], destination.resolution);
    }

    StaggeredVectorField3D::StaggeredVectorField3D(const std::array<std::int32_t, 3> resolution) : resolution{empty_resolution} {
        this->resize(resolution);
    }

    StaggeredVectorField3D::~StaggeredVectorField3D() noexcept {
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = empty_resolution;
    }

    StaggeredVectorField3D::StaggeredVectorField3D(StaggeredVectorField3D&& other) noexcept : resolution{std::exchange(other.resolution, empty_resolution)}, data{std::exchange(other.data, std::array<float*, 3>{})} {}

    StaggeredVectorField3D& StaggeredVectorField3D::operator=(StaggeredVectorField3D&& other) noexcept {
        if (this == &other) return *this;
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->resolution = std::exchange(other.resolution, empty_resolution);
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
        if (resolution == empty_resolution) {
            cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->resolution = empty_resolution;
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

    void StaggeredVectorField3D::fill(cudaStream_t stream, const float value) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (this->resolution == empty_resolution || this->data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            cuda::field::fill(stream, this->data[axis], this->count(axis), value);
        }
    }

    void add_centered_to_staggered(cudaStream_t stream, StaggeredVectorField3D& destination, const std::uint32_t axis, const CenteredVectorField3D& source, const float scale) {
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.resolution == empty_resolution || destination.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
        if (source.resolution == empty_resolution || source.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
        cuda::field::add_centered_to_staggered(stream, destination.data[axis], source.data[axis], axis, destination.resolution, scale);
    }

    void copy_staggered_component(cudaStream_t stream, StaggeredVectorField3D& destination, const std::uint32_t axis, const StaggeredVectorField3D& source) {
        if (axis >= 3u) throw std::runtime_error{"invalid axis"};
        if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
        if (destination.resolution == empty_resolution || destination.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
        if (source.resolution == empty_resolution || source.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
        if (const cudaError_t status = cudaMemcpyAsync(destination.data[axis], source.data[axis], destination.bytes(axis), cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync staggered component copy: "} + cudaGetErrorString(status)};
    }
} // namespace kfs::field
