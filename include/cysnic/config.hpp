#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <spdlog/spdlog.h>
#include <optional>

namespace cysnic {

struct TrackerConfig {
    double psrThreshold = 10.0;
    double maxOcclusionTime = 2.0;
    double driftBaseline = 0.2;
    double varianceCap = 1000.0;
    double maxRotationShift = 5.0;

    static std::optional<TrackerConfig> load(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            spdlog::warn("Config file {} not found. Using default parameters.", path);
            return std::nullopt;
        }

        try {
            nlohmann::json j;
            file >> j;
            TrackerConfig config;
            if (j.contains("psrThreshold")) config.psrThreshold = j["psrThreshold"];
            if (j.contains("maxOcclusionTime")) config.maxOcclusionTime = j["maxOcclusionTime"];
            if (j.contains("driftBaseline")) config.driftBaseline = j["driftBaseline"];
            if (j.contains("varianceCap")) config.varianceCap = j["varianceCap"];
            if (j.contains("maxRotationShift")) config.maxRotationShift = j["maxRotationShift"];
            
            spdlog::info("Successfully loaded tracker configuration from {}", path);
            return config;
        } catch (const std::exception& e) {
            spdlog::error("Error parsing config file {}: {}. Using default parameters.", path, e.what());
            return std::nullopt;
        }
    }
};

} // namespace cysnic
