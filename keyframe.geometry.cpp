module keyframe.geometry;
import std;

namespace kfs::geometry {
    void validate(const Ellipsoid& value) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) if (!std::isfinite(value.center[axis])) throw std::runtime_error{"Geometry ellipsoid center must be finite"};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) if (!std::isfinite(value.radius[axis]) || value.radius[axis] <= 0.0f) throw std::runtime_error{"Geometry ellipsoid radius must be positive"};
    }
} // namespace kfs::geometry
