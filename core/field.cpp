module;
#include "field.h"

#include <cuda_runtime.h>

module xayah.core.field;
import std;

namespace xayah::core::field {
    namespace {
        std::uint64_t cell_element_count(const std::array<std::int32_t, 3>& resolution) {
            if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) return 0u;
            const auto count = static_cast<std::uint64_t>(resolution[0]) * static_cast<std::uint64_t>(resolution[1]) * static_cast<std::uint64_t>(resolution[2]);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"field element count exceeds int range"};
            return count;
        }

        std::uint64_t face_element_count(const std::array<std::int32_t, 3>& resolution, const std::uint32_t axis) {
            const auto nx    = static_cast<std::uint64_t>(resolution[0]);
            const auto ny    = static_cast<std::uint64_t>(resolution[1]);
            const auto nz    = static_cast<std::uint64_t>(resolution[2]);
            const auto count = axis == 0u ? (nx + 1u) * ny * nz : axis == 1u ? nx * (ny + 1u) * nz : nx * ny * (nz + 1u);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error{"field element count exceeds int range"};
            return count;
        }
    } // namespace

    ScalarField3D::ScalarField3D(const std::array<std::int32_t, 3> resolution) : resolution{} {
        this->resize(resolution);
    }

    ScalarField3D::~ScalarField3D() noexcept {
        cuda::free_device_buffers(this->data);
        this->resolution = {};
    }

    ScalarField3D::ScalarField3D(ScalarField3D&& other) noexcept : resolution{std::exchange(other.resolution, std::array<std::int32_t, 3>{})}, data{std::exchange(other.data, nullptr)} {}

    ScalarField3D& ScalarField3D::operator=(ScalarField3D&& other) noexcept {
        if (this == &other) return *this;
        cuda::free_device_buffers(this->data);
        this->resolution = std::exchange(other.resolution, std::array<std::int32_t, 3>{});
        this->data       = std::exchange(other.data, nullptr);
        return *this;
    }

    [[nodiscard]] std::uint64_t ScalarField3D::count() const {
        return cell_element_count(this->resolution);
    }

    [[nodiscard]] std::size_t ScalarField3D::bytes() const {
        return cell_element_count(this->resolution) * sizeof(float);
    }

    void ScalarField3D::resize(const std::array<std::int32_t, 3> resolution) {
        if (this->resolution == resolution) return;
        if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) {
            cuda::free_device_buffers(this->data);
            this->resolution = {};
            return;
        }

        float* next = nullptr;
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next), cell_element_count(resolution) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc scalar field: "} + cudaGetErrorString(status)};
        cuda::free_device_buffers(this->data);
        this->data       = next;
        this->resolution = resolution;
    }

    IndexedField3D::IndexedField3D(const std::array<std::int32_t, 3> resolution) : resolution{} {
        this->resize(resolution);
    }

    IndexedField3D::~IndexedField3D() noexcept {
        cuda::free_device_buffers(this->data);
        this->resolution = {};
    }

    IndexedField3D::IndexedField3D(IndexedField3D&& other) noexcept : resolution{std::exchange(other.resolution, std::array<std::int32_t, 3>{})}, data{std::exchange(other.data, nullptr)} {}

    IndexedField3D& IndexedField3D::operator=(IndexedField3D&& other) noexcept {
        if (this == &other) return *this;
        cuda::free_device_buffers(this->data);
        this->resolution = std::exchange(other.resolution, std::array<std::int32_t, 3>{});
        this->data       = std::exchange(other.data, nullptr);
        return *this;
    }

    [[nodiscard]] std::uint64_t IndexedField3D::count() const {
        return cell_element_count(this->resolution);
    }

    [[nodiscard]] std::size_t IndexedField3D::bytes() const {
        return cell_element_count(this->resolution) * sizeof(std::uint32_t);
    }

    void IndexedField3D::resize(const std::array<std::int32_t, 3> resolution) {
        if (this->resolution == resolution) return;
        if (resolution[0] == 0 && resolution[1] == 0 && resolution[2] == 0) {
            cuda::free_device_buffers(this->data);
            this->resolution = {};
            return;
        }

        std::uint32_t* next = nullptr;
        if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next), cell_element_count(resolution) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc indexed field: "} + cudaGetErrorString(status)};
        cuda::free_device_buffers(this->data);
        this->data       = next;
        this->resolution = resolution;
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
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next[axis]), bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc centered vector field: "} + cudaGetErrorString(status)};
        } catch (...) {
            cuda::free_device_buffers(next[0], next[1], next[2]);
            throw;
        }
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->data       = next;
        this->resolution = resolution;
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
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next[axis]), face_element_count(resolution, axis) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc staggered vector field: "} + cudaGetErrorString(status)};
        } catch (...) {
            cuda::free_device_buffers(next[0], next[1], next[2]);
            throw;
        }
        cuda::free_device_buffers(this->data[0], this->data[1], this->data[2]);
        this->data       = next;
        this->resolution = resolution;
    }

    void fill(const cudaStream_t stream, ScalarField3D& values, const float value) {
        cuda::fill(stream, values.data, values.count(), value);
    }

    void fill(const cudaStream_t stream, IndexedField3D& values, const std::uint32_t value) {
        cuda::fill(stream, values.data, values.count(), value);
    }

    void fill(const cudaStream_t stream, CenteredVectorField3D& values, const float value) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::fill(stream, values.data[axis], values.count(), value);
    }

    void fill(const cudaStream_t stream, StaggeredVectorField3D& values, const float value) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::fill(stream, values.data[axis], values.count(axis), value);
    }

    void copy(const cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source) {
        if (const cudaError_t status = cudaMemcpyAsync(destination.data, source.data, destination.bytes(), cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync scalar copy: "} + cudaGetErrorString(status)};
    }

    void copy(const cudaStream_t stream, IndexedField3D& destination, const IndexedField3D& source) {
        if (const cudaError_t status = cudaMemcpyAsync(destination.data, source.data, destination.bytes(), cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync indexed copy: "} + cudaGetErrorString(status)};
    }

    void copy(const cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source, const IndexedField3D& indices, const IndexSelection selection) {
        cuda::copy(stream, destination.data, source.data, indices.data, destination.count(), static_cast<std::uint32_t>(selection));
    }

    void copy(const cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& source) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (const cudaError_t status = cudaMemcpyAsync(destination.data[axis], source.data[axis], destination.bytes(), cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync centered vector copy: "} + cudaGetErrorString(status)};
    }

    void copy(const cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& source) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (const cudaError_t status = cudaMemcpyAsync(destination.data[axis], source.data[axis], destination.bytes(axis), cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync staggered vector copy: "} + cudaGetErrorString(status)};
    }

    void upload(const cudaStream_t stream, ScalarField3D& destination, const std::span<const float> source) {
        if (const cudaError_t status = cudaMemcpyAsync(destination.data, source.data(), destination.bytes(), cudaMemcpyHostToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync scalar upload: "} + cudaGetErrorString(status)};
    }

    void upload(const cudaStream_t stream, IndexedField3D& destination, const std::span<const std::uint32_t> source) {
        if (const cudaError_t status = cudaMemcpyAsync(destination.data, source.data(), destination.bytes(), cudaMemcpyHostToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync indexed upload: "} + cudaGetErrorString(status)};
    }

    void upload(const cudaStream_t stream, CenteredVectorField3D& destination, const std::array<std::span<const float>, 3> source) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (const cudaError_t status = cudaMemcpyAsync(destination.data[axis], source[axis].data(), destination.bytes(), cudaMemcpyHostToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync centered vector upload: "} + cudaGetErrorString(status)};
    }

    void upload(const cudaStream_t stream, StaggeredVectorField3D& destination, const std::array<std::span<const float>, 3> source) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (const cudaError_t status = cudaMemcpyAsync(destination.data[axis], source[axis].data(), destination.bytes(axis), cudaMemcpyHostToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync staggered vector upload: "} + cudaGetErrorString(status)};
    }

    void download(const cudaStream_t stream, const std::span<float> destination, const ScalarField3D& source) {
        if (const cudaError_t status = cudaMemcpyAsync(destination.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync scalar download: "} + cudaGetErrorString(status)};
    }

    void download(const cudaStream_t stream, const std::span<std::uint32_t> destination, const IndexedField3D& source) {
        if (const cudaError_t status = cudaMemcpyAsync(destination.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync indexed download: "} + cudaGetErrorString(status)};
    }

    void download(const cudaStream_t stream, const std::array<std::span<float>, 3> destination, const CenteredVectorField3D& source) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (const cudaError_t status = cudaMemcpyAsync(destination[axis].data(), source.data[axis], source.bytes(), cudaMemcpyDeviceToHost, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync centered vector download: "} + cudaGetErrorString(status)};
    }

    void download(const cudaStream_t stream, const std::array<std::span<float>, 3> destination, const StaggeredVectorField3D& source) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            if (const cudaError_t status = cudaMemcpyAsync(destination[axis].data(), source.data[axis], source.bytes(axis), cudaMemcpyDeviceToHost, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync staggered vector download: "} + cudaGetErrorString(status)};
    }

    void add(const cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& left, const ScalarField3D& right) {
        cuda::add(stream, destination.data, left.data, right.data, destination.count());
    }

    void add(const cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& current, const ScalarField3D& source, const float scale) {
        cuda::add(stream, destination.data, current.data, source.data, destination.count(), scale);
    }

    void add(const cudaStream_t stream, CenteredVectorField3D& destination, const std::uint32_t axis, const ScalarField3D& source, const float scale, const float bias, const IndexedField3D& indices, const IndexSelection selection) {
        cuda::add(stream, destination.data[axis], source.data, indices.data, destination.count(), scale, bias, static_cast<std::uint32_t>(selection));
    }

    void add(const cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& left, const CenteredVectorField3D& right) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::add(stream, destination.data[axis], left.data[axis], right.data[axis], destination.count());
    }

    void add(const cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& current, const CenteredVectorField3D& source, const float scale) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::add(stream, destination.data[axis], current.data[axis], source.data[axis], destination.count(), scale);
    }

    void add(const cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& left, const StaggeredVectorField3D& right) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::add(stream, destination.data[axis], left.data[axis], right.data[axis], destination.count(axis));
    }

    void add(const cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& current, const StaggeredVectorField3D& source, const float scale) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::add(stream, destination.data[axis], current.data[axis], source.data[axis], destination.count(axis), scale);
    }

    void add(const cudaStream_t stream, StaggeredVectorField3D& destination, const CenteredVectorField3D& source, const float scale) {
        cuda::add(stream, destination.data[0], destination.data[1], destination.data[2], source.data[0], source.data[1], source.data[2], destination.resolution, scale);
    }

    void subtract(const cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& left, const ScalarField3D& right) {
        cuda::subtract(stream, destination.data, left.data, right.data, destination.count());
    }

    void subtract(const cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& left, const CenteredVectorField3D& right) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::subtract(stream, destination.data[axis], left.data[axis], right.data[axis], destination.count());
    }

    void subtract(const cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& left, const StaggeredVectorField3D& right) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::subtract(stream, destination.data[axis], left.data[axis], right.data[axis], destination.count(axis));
    }

    void multiply(const cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& left, const ScalarField3D& right) {
        cuda::multiply(stream, destination.data, left.data, right.data, destination.count());
    }

    void multiply(const cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& left, const CenteredVectorField3D& right) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::multiply(stream, destination.data[axis], left.data[axis], right.data[axis], destination.count());
    }

    void multiply(const cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& left, const StaggeredVectorField3D& right) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::multiply(stream, destination.data[axis], left.data[axis], right.data[axis], destination.count(axis));
    }

    void scale(const cudaStream_t stream, ScalarField3D& destination, const ScalarField3D& source, const float factor) {
        cuda::scale(stream, destination.data, source.data, destination.count(), factor);
    }

    void scale(const cudaStream_t stream, CenteredVectorField3D& destination, const CenteredVectorField3D& source, const float factor) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::scale(stream, destination.data[axis], source.data[axis], destination.count(), factor);
    }

    void scale(const cudaStream_t stream, StaggeredVectorField3D& destination, const StaggeredVectorField3D& source, const float factor) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) cuda::scale(stream, destination.data[axis], source.data[axis], destination.count(axis), factor);
    }

    void sample(const cudaStream_t stream, CenteredVectorField3D& destination, const StaggeredVectorField3D& source) {
        cuda::sample(stream, destination.data[0], destination.data[1], destination.data[2], source.data[0], source.data[1], source.data[2], destination.resolution);
    }

    ScalarFieldStats stats(const cudaStream_t stream, const ScalarField3D& source) {
        cuda::ScalarStats raw{};
        cuda::stats(stream, source.data, source.count(), raw);
        return ScalarFieldStats{
            .min           = raw.min,
            .max           = raw.max,
            .sum           = raw.sum,
            .mean          = raw.mean,
            .nonzero_count = raw.nonzero_count,
        };
    }
} // namespace xayah::core::field
