export module xayah.core.geometry;
import std;

export namespace xayah::core::geometry {
    struct Ellipsoid final {
        std::array<float, 3> center{0.0f, 0.0f, 0.0f};
        std::array<float, 3> radius{1.0f, 1.0f, 1.0f};
    };

    struct Box final {
        std::array<float, 3> center{0.0f, 0.0f, 0.0f};
        std::array<float, 3> half_extent{1.0f, 1.0f, 1.0f};
    };
} // namespace xayah::core::geometry
