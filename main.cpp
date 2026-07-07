#include <cstdio>

import std;
import xayah.util.xcli;
import keyframe.field;
import keyframe.solver;

int main(const int argc, const char* const* const argv) {
    try {
        const std::span arguments{argv, static_cast<std::size_t>(argc)};
        std::uint32_t frames = 120u;
        float delta_seconds  = 1.0f / 60.0f;
        bool quiet           = false;

        xayah::util::Command command = xayah::util::Command{"Forward keyframe smoke simulation statistics runner."}
                                     | xayah::util::option(
                                         xayah::util::OptionSpec{
                                             .long_name    = "frames",
                                             .value_name   = "n",
                                             .description  = "Number of smoke simulation frames to run.",
                                             .default_text = "120",
                                         },
                                         frames, xayah::util::NumericRule{.minimum = 1.0, .maximum = static_cast<double>(std::numeric_limits<int>::max())})
                                     | xayah::util::option(
                                         xayah::util::OptionSpec{
                                             .long_name    = "dt",
                                             .value_name   = "seconds",
                                             .description  = "Smoke simulation time step in seconds.",
                                             .default_text = "0.0166667",
                                         },
                                         delta_seconds, xayah::util::NumericRule{.minimum = 0.0, .minimum_is_exclusive = true})
                                     | xayah::util::option(
                                         xayah::util::OptionSpec{
                                             .long_name   = "quiet",
                                             .description = "Suppress per-frame progress output.",
                                         },
                                         quiet)
                                     | xayah::util::example("--frames 1 --quiet") | xayah::util::example("--frames 10 --dt 0.0166667 --quiet");

        const std::string usage = command.help(arguments);
        const auto parsed       = command.parse(arguments);
        if (!parsed) {
            std::print(stderr, "keyframe: {}\n\n{}", parsed.error(), usage);
            return 1;
        }
        if (parsed->help_requested) {
            std::print("{}", usage);
            return 0;
        }
        const auto validation = command.validate();
        if (!validation) {
            std::print(stderr, "keyframe: {}\n\n{}", validation.error(), usage);
            return 1;
        }

        kfs::solver::Solver smoke{};
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            const std::expected<kfs::solver::StepStats, std::string> stats = smoke.step(kfs::solver::StepRequest{
                .delta_seconds = delta_seconds,
                .iterations    = 1,
            });
            if (!stats.has_value()) throw std::runtime_error{stats.error()};
            if (!quiet) std::println("frame {}/{}", stats->step, frames);
        }

        const kfs::field::ScalarFieldStats density_stats     = kfs::field::stats(smoke.stream, smoke.device.density_data);
        const kfs::field::ScalarFieldStats temperature_stats = kfs::field::stats(smoke.stream, smoke.device.temperature_data);
        const auto& resolution                               = smoke.device.density_data.resolution;

        std::println("keyframe frames={} dt={:.6f} resolution=({}, {}, {}) density_sum={:.6f} temperature_sum={:.6f}", frames, delta_seconds, resolution[0], resolution[1], resolution[2], density_stats.sum, temperature_stats.sum);
        return 0;
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
}
