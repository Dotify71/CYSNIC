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
            if (j.contains("psrThreshold")) {
                if (j["psrThreshold"] > 0) config.psrThreshold = j["psrThreshold"];
                else spdlog::warn("Invalid psrThreshold in config.json. Using default {}.", config.psrThreshold);
            }
            if (j.contains("maxOcclusionTime")) {
                if (j["maxOcclusionTime"] > 0) config.maxOcclusionTime = j["maxOcclusionTime"];
                else spdlog::warn("Invalid maxOcclusionTime in config.json. Using default {}.", config.maxOcclusionTime);
            }
            if (j.contains("driftBaseline")) {
                if (j["driftBaseline"] > 0) config.driftBaseline = j["driftBaseline"];
                else spdlog::warn("Invalid driftBaseline in config.json. Using default {}.", config.driftBaseline);
            }
            if (j.contains("varianceCap")) {
                if (j["varianceCap"] > 0) config.varianceCap = j["varianceCap"];
                else spdlog::warn("Invalid varianceCap in config.json. Using default {}.", config.varianceCap);
            }
            if (j.contains("maxRotationShift")) {
                if (j["maxRotationShift"] > 0) config.maxRotationShift = j["maxRotationShift"];
                else spdlog::warn("Invalid maxRotationShift in config.json. Using default {}.", config.maxRotationShift);
            }
            
            spdlog::info("Successfully loaded tracker configuration from {}", path);
            return config;
        } catch (const std::exception& e) {
            spdlog::error("Error parsing config file {}: {}. Using default parameters.", path, e.what());
            return std::nullopt;
        }
    }
};

} // namespace cysnic
