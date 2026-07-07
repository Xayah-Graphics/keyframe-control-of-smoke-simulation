export module keyframe.geometry;
import std;

export namespace kfs::geometry {
    struct Ellipsoid final {
        std::array<float, 3> center{0.0f, 0.0f, 0.0f};
        std::array<float, 3> radius{1.0f, 1.0f, 1.0f};
    };

    struct Box final {
        std::array<float, 3> center{0.0f, 0.0f, 0.0f};
        std::array<float, 3> half_extent{1.0f, 1.0f, 1.0f};
    };
} // namespace kfs::geometry
