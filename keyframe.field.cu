#include "keyframe.field.h"
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfs::cuda {
    void free_device_buffers(void** const pointers, const std::size_t count) noexcept {
        if (pointers == nullptr) return;
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] == nullptr) continue;
            cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }

    namespace field {
        namespace {
            constexpr std::array<std::int32_t, 3> empty_resolution{0, 0, 0};

            unsigned ceil_div_u32(const std::uint64_t value, const std::uint64_t divisor) {
                return static_cast<unsigned>((value + divisor - 1u) / divisor);
            }

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

            __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
                return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
            }

            __device__ std::uint64_t index_staggered_x(const int i, const int j, const int k, const int nx, const int ny) {
                const auto nx64 = static_cast<std::uint64_t>(nx);
                const auto ny64 = static_cast<std::uint64_t>(ny);
                return static_cast<std::uint64_t>(k) * (nx64 + 1u) * ny64 + static_cast<std::uint64_t>(j) * (nx64 + 1u) + static_cast<std::uint64_t>(i);
            }

            __device__ std::uint64_t index_staggered_y(const int i, const int j, const int k, const int nx, const int ny) {
                const auto nx64 = static_cast<std::uint64_t>(nx);
                const auto ny64 = static_cast<std::uint64_t>(ny);
                return static_cast<std::uint64_t>(k) * nx64 * (ny64 + 1u) + static_cast<std::uint64_t>(j) * nx64 + static_cast<std::uint64_t>(i);
            }

            __device__ std::uint64_t index_staggered_z(const int i, const int j, const int k, const int nx, const int ny) {
                return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(i);
            }

            __global__ void fill_kernel(float* values, const float value, const std::uint64_t count) {
                const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
                if (index >= count) return;
                values[index] = value;
            }

            __global__ void add_scaled_kernel(float* destination, const float* current, const float* source, const float scale, const std::uint64_t count) {
                const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
                if (index >= count) return;
                destination[index] = current[index] + scale * source[index];
            }

            __global__ void center_staggered_kernel(float* cx, float* cy, float* cz, const float* sx, const float* sy, const float* sz, const int nx, const int ny, const int nz) {
                const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
                const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
                if (x >= nx || y >= ny || z >= nz) return;
                const auto index = index_3d(x, y, z, nx, ny);
                cx[index]        = 0.5f * (sx[index_staggered_x(x, y, z, nx, ny)] + sx[index_staggered_x(x + 1, y, z, nx, ny)]);
                cy[index]        = 0.5f * (sy[index_staggered_y(x, y, z, nx, ny)] + sy[index_staggered_y(x, y + 1, z, nx, ny)]);
                cz[index]        = 0.5f * (sz[index_staggered_z(x, y, z, nx, ny)] + sz[index_staggered_z(x, y, z + 1, nx, ny)]);
            }

            __global__ void add_centered_to_staggered_x_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
                const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
                const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
                if (i > nx || j >= ny || k >= nz) return;

                float sum    = 0.0f;
                float weight = 0.0f;
                if (i > 0) {
                    sum += source[index_3d(i - 1, j, k, nx, ny)];
                    weight += 1.0f;
                }
                if (i < nx) {
                    sum += source[index_3d(i, j, k, nx, ny)];
                    weight += 1.0f;
                }
                if (weight > 0.0f) destination[index_staggered_x(i, j, k, nx, ny)] += scale * (sum / weight);
            }

            __global__ void add_centered_to_staggered_y_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
                const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
                const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
                if (i >= nx || j > ny || k >= nz) return;

                float sum    = 0.0f;
                float weight = 0.0f;
                if (j > 0) {
                    sum += source[index_3d(i, j - 1, k, nx, ny)];
                    weight += 1.0f;
                }
                if (j < ny) {
                    sum += source[index_3d(i, j, k, nx, ny)];
                    weight += 1.0f;
                }
                if (weight > 0.0f) destination[index_staggered_y(i, j, k, nx, ny)] += scale * (sum / weight);
            }

            __global__ void add_centered_to_staggered_z_kernel(float* destination, const float* source, const int nx, const int ny, const int nz, const float scale) {
                const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
                const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
                if (i >= nx || j >= ny || k > nz) return;

                float sum    = 0.0f;
                float weight = 0.0f;
                if (k > 0) {
                    sum += source[index_3d(i, j, k - 1, nx, ny)];
                    weight += 1.0f;
                }
                if (k < nz) {
                    sum += source[index_3d(i, j, k, nx, ny)];
                    weight += 1.0f;
                }
                if (weight > 0.0f) destination[index_staggered_z(i, j, k, nx, ny)] += scale * (sum / weight);
            }

        } // namespace

        ScalarField3D::ScalarField3D(const std::array<std::int32_t, 3> resolution) : resolution{empty_resolution} {
            this->resize(resolution);
        }

        ScalarField3D::~ScalarField3D() noexcept {
            free_device_buffers(this->data);
            this->resolution = empty_resolution;
        }

        ScalarField3D::ScalarField3D(ScalarField3D&& other) noexcept : resolution{std::exchange(other.resolution, empty_resolution)}, data{std::exchange(other.data, nullptr)} {}

        ScalarField3D& ScalarField3D::operator=(ScalarField3D&& other) noexcept {
            if (this == &other) return *this;
            free_device_buffers(this->data);
            this->resolution = std::exchange(other.resolution, empty_resolution);
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
            if (resolution == empty_resolution) {
                free_device_buffers(this->data);
                this->resolution = empty_resolution;
                return;
            }

            float* next = nullptr;
            if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next), cell_element_count(resolution) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc scalar field: "} + cudaGetErrorString(status)};
            free_device_buffers(this->data);
            this->data       = next;
            this->resolution = resolution;
        }

        void ScalarField3D::fill(cudaStream_t stream, const float value) {
            if (this->resolution == empty_resolution || this->data == nullptr) throw std::runtime_error{"field is empty"};
            const std::uint64_t count = this->count();
            fill_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(this->data, value, count);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_kernel: "} + cudaGetErrorString(status)};
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
            const std::uint64_t count = destination.count();
            add_scaled_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(destination.data, current.data, source.data, scale, count);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_scaled_kernel: "} + cudaGetErrorString(status)};
        }

        CenteredVectorField3D::CenteredVectorField3D(const std::array<std::int32_t, 3> resolution) : resolution{empty_resolution} {
            this->resize(resolution);
        }

        CenteredVectorField3D::~CenteredVectorField3D() noexcept {
            free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->resolution = empty_resolution;
        }

        CenteredVectorField3D::CenteredVectorField3D(CenteredVectorField3D&& other) noexcept : resolution{std::exchange(other.resolution, empty_resolution)}, data{std::exchange(other.data, std::array<float*, 3>{})} {}

        CenteredVectorField3D& CenteredVectorField3D::operator=(CenteredVectorField3D&& other) noexcept {
            if (this == &other) return *this;
            free_device_buffers(this->data[0], this->data[1], this->data[2]);
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
                free_device_buffers(this->data[0], this->data[1], this->data[2]);
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
                free_device_buffers(next[0], next[1], next[2]);
                throw;
            }
            free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->data       = next;
            this->resolution = resolution;
        }

        void CenteredVectorField3D::fill(cudaStream_t stream, const float value) {
            const std::uint64_t count = this->count();
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                if (this->resolution == empty_resolution || this->data[axis] == nullptr) throw std::runtime_error{"field is empty"};
                fill_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(this->data[axis], value, count);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_kernel: "} + cudaGetErrorString(status)};
            }
        }

        void center_staggered(cudaStream_t stream, CenteredVectorField3D& destination, const StaggeredVectorField3D& source) {
            if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                if (destination.resolution == empty_resolution || destination.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
                if (source.resolution == empty_resolution || source.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            }
            constexpr dim3 block{8u, 8u, 4u};
            const dim3 grid{
                ceil_div_u32(static_cast<std::uint64_t>(destination.resolution[0]), block.x),
                ceil_div_u32(static_cast<std::uint64_t>(destination.resolution[1]), block.y),
                ceil_div_u32(static_cast<std::uint64_t>(destination.resolution[2]), block.z),
            };
            center_staggered_kernel<<<grid, block, 0, stream>>>(destination.data[0], destination.data[1], destination.data[2], source.data[0], source.data[1], source.data[2], destination.resolution[0], destination.resolution[1], destination.resolution[2]);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"center_staggered_kernel: "} + cudaGetErrorString(status)};
        }

        StaggeredVectorField3D::StaggeredVectorField3D(const std::array<std::int32_t, 3> resolution) : resolution{empty_resolution} {
            this->resize(resolution);
        }

        StaggeredVectorField3D::~StaggeredVectorField3D() noexcept {
            free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->resolution = empty_resolution;
        }

        StaggeredVectorField3D::StaggeredVectorField3D(StaggeredVectorField3D&& other) noexcept : resolution{std::exchange(other.resolution, empty_resolution)}, data{std::exchange(other.data, std::array<float*, 3>{})} {}

        StaggeredVectorField3D& StaggeredVectorField3D::operator=(StaggeredVectorField3D&& other) noexcept {
            if (this == &other) return *this;
            free_device_buffers(this->data[0], this->data[1], this->data[2]);
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
                free_device_buffers(this->data[0], this->data[1], this->data[2]);
                this->resolution = empty_resolution;
                return;
            }

            std::array<float*, 3> next{};
            try {
                for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                    if (const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&next[axis]), face_element_count(resolution, axis) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc staggered vector field: "} + cudaGetErrorString(status)};
                }
            } catch (...) {
                free_device_buffers(next[0], next[1], next[2]);
                throw;
            }
            free_device_buffers(this->data[0], this->data[1], this->data[2]);
            this->data       = next;
            this->resolution = resolution;
        }

        void StaggeredVectorField3D::fill(cudaStream_t stream, const float value) {
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                if (this->resolution == empty_resolution || this->data[axis] == nullptr) throw std::runtime_error{"field is empty"};
                const std::uint64_t count = this->count(axis);
                fill_kernel<<<ceil_div_u32(count, 256u), 256u, 0, stream>>>(this->data[axis], value, count);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_kernel: "} + cudaGetErrorString(status)};
            }
        }

        void add_centered_to_staggered(cudaStream_t stream, StaggeredVectorField3D& destination, const std::uint32_t axis, const CenteredVectorField3D& source, const float scale) {
            if (axis >= 3u) throw std::runtime_error{"invalid axis"};
            if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
            if (destination.resolution == empty_resolution || destination.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            if (source.resolution == empty_resolution || source.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            constexpr dim3 block{8u, 8u, 4u};
            const auto nx   = static_cast<std::uint64_t>(destination.resolution[0]);
            const auto ny   = static_cast<std::uint64_t>(destination.resolution[1]);
            const auto nz   = static_cast<std::uint64_t>(destination.resolution[2]);
            const dim3 grid = axis == 0u ? dim3(ceil_div_u32(nx + 1u, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz, block.z)) : axis == 1u ? dim3(ceil_div_u32(nx, block.x), ceil_div_u32(ny + 1u, block.y), ceil_div_u32(nz, block.z)) : dim3(ceil_div_u32(nx, block.x), ceil_div_u32(ny, block.y), ceil_div_u32(nz + 1u, block.z));
            if (axis == 0u) add_centered_to_staggered_x_kernel<<<grid, block, 0, stream>>>(destination.data[axis], source.data[axis], destination.resolution[0], destination.resolution[1], destination.resolution[2], scale);
            if (axis == 1u) add_centered_to_staggered_y_kernel<<<grid, block, 0, stream>>>(destination.data[axis], source.data[axis], destination.resolution[0], destination.resolution[1], destination.resolution[2], scale);
            if (axis == 2u) add_centered_to_staggered_z_kernel<<<grid, block, 0, stream>>>(destination.data[axis], source.data[axis], destination.resolution[0], destination.resolution[1], destination.resolution[2], scale);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_centered_to_staggered_kernel: "} + cudaGetErrorString(status)};
        }

        void copy_staggered_component(cudaStream_t stream, StaggeredVectorField3D& destination, const std::uint32_t axis, const StaggeredVectorField3D& source) {
            if (axis >= 3u) throw std::runtime_error{"invalid axis"};
            if (destination.resolution != source.resolution) throw std::runtime_error{"field resolution mismatch"};
            if (destination.resolution == empty_resolution || destination.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            if (source.resolution == empty_resolution || source.data[axis] == nullptr) throw std::runtime_error{"field is empty"};
            if (const cudaError_t status = cudaMemcpyAsync(destination.data[axis], source.data[axis], destination.bytes(axis), cudaMemcpyDeviceToDevice, stream); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpyAsync staggered component copy: "} + cudaGetErrorString(status)};
        }
    } // namespace field
} // namespace kfs::cuda
