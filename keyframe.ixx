export module keyframe;

namespace kfs {
    export class KeyframeSmoke final {
    public:
        explicit KeyframeSmoke();
        ~KeyframeSmoke() noexcept;
        KeyframeSmoke(const KeyframeSmoke& other)                = delete;
        KeyframeSmoke(KeyframeSmoke&& other) noexcept            = default;
        KeyframeSmoke& operator=(const KeyframeSmoke& other)     = delete;
        KeyframeSmoke& operator=(KeyframeSmoke&& other) noexcept = default;

    private:
        struct HostData {
        } host;
        struct DeviceData {
        } device;
    };
} // namespace kfs
