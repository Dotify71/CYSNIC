#include "cysnic/tracker.hpp"
#include "cysnic/config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <spdlog/spdlog.h>

void drawTacticalHUD(cv::Mat& frame, const cv::Rect2d& box, int id, bool occluded, double dx = 0, double dy = 0) {
    // Dynamic contrast color based on background brightness
    cv::Rect safeBox(cvRound(box.x), cvRound(box.y), cvRound(box.width), cvRound(box.height));
    safeBox.x = std::max(0, safeBox.x);
    safeBox.y = std::max(0, safeBox.y);
    safeBox.width = std::min(frame.cols - safeBox.x, safeBox.width);
    safeBox.height = std::min(frame.rows - safeBox.y, safeBox.height);
    
    cv::Scalar hudColor(0, 0, 0); // Default Black
    if (safeBox.width > 0 && safeBox.height > 0) {
        cv::Scalar meanColor = cv::mean(frame(safeBox));
        double brightness = 0.299 * meanColor[2] + 0.587 * meanColor[1] + 0.114 * meanColor[0];
        if (brightness < 128) {
            hudColor = cv::Scalar(255, 255, 255); // Switch to White on dark background
        }
    }
    
    int lineType = cv::LINE_AA;
    
    // Draw corners
    int cornerLen = 20;
    int t = 2; // thickness
    
    if (occluded) {
        // Draw X-RAY dashed representation (or just standard corners with crosshair)
        cv::Point center(cvRound(box.x + box.width / 2.0), cvRound(box.y + box.height / 2.0));
        cv::drawMarker(frame, center, hudColor, cv::MARKER_CROSS, 20, 1, lineType);
        
        // Dashed box for X-Ray
        cv::Rect drawBox(cvRound(box.x), cvRound(box.y), cvRound(box.width), cvRound(box.height));
        cv::rectangle(frame, drawBox, hudColor, 1, lineType);
        
        std::string status = "STATUS: X-RAY / COASTING";
        cv::putText(frame, status, cv::Point(cvRound(box.x), cvRound(box.y) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, hudColor, 1, lineType);
    } else {
        int bx = cvRound(box.x), by = cvRound(box.y), bw = cvRound(box.width), bh = cvRound(box.height);
        // Top left
        cv::line(frame, cv::Point(bx, by), cv::Point(bx + cornerLen, by), hudColor, t, lineType);
        cv::line(frame, cv::Point(bx, by), cv::Point(bx, by + cornerLen), hudColor, t, lineType);
        // Top right
        cv::line(frame, cv::Point(bx + bw, by), cv::Point(bx + bw - cornerLen, by), hudColor, t, lineType);
        cv::line(frame, cv::Point(bx + bw, by), cv::Point(bx + bw, by + cornerLen), hudColor, t, lineType);
        // Bottom left
        cv::line(frame, cv::Point(bx, by + bh), cv::Point(bx + cornerLen, by + bh), hudColor, t, lineType);
        cv::line(frame, cv::Point(bx, by + bh), cv::Point(bx, by + bh - cornerLen), hudColor, t, lineType);
        // Bottom right
        cv::line(frame, cv::Point(bx + bw, by + bh), cv::Point(bx + bw - cornerLen, by + bh), hudColor, t, lineType);
        cv::line(frame, cv::Point(bx + bw, by + bh), cv::Point(bx + bw, by + bh - cornerLen), hudColor, t, lineType);
        
        // Status Text
        std::string status = "STATUS: TRACK LOCKED";
        cv::putText(frame, status, cv::Point(bx, by - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, hudColor, 1, lineType);
    }
    
    // Calculations / Telemetry placed strictly outside the box
    std::string telemetry = fmt::format("ID: {:02d} | POS: [{:.1f}, {:.1f}] | VEL: [{:.1f}, {:.1f}]", id, box.x, box.y, dx, dy);
    cv::putText(frame, telemetry, cv::Point(cvRound(box.x), cvRound(box.y + box.height) + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, hudColor, 1, lineType);
}
int main() {
    spdlog::set_level(spdlog::level::info);
    
    // Load external configuration
    auto config = cysnic::TrackerConfig::load("config.json").value_or(cysnic::TrackerConfig());

    cv::VideoCapture cap(0);
    spdlog::info("CYSNIC (Surveillance-Inspired UI Tracker) starting...");

    if (!cap.isOpened()) {
        spdlog::error("Failed to open camera 0");
        return -1;
    }

    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
        spdlog::error("Failed to read first frame.");
        return -1;
    }

    std::vector<std::unique_ptr<cysnic::TargetTracker>> trackers;
    int targetId = 1;

    while (true) {
        cv::Rect2d roi = cv::selectROI("CYSNIC Tracker - Select ROIs (Space to finish)", frame, false, false);
        if (roi.width == 0 || roi.height == 0) {
            break; // Stop selecting if empty ROI is chosen or space is pressed
        }
        
        cv::Mat grayFrame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        } else {
            grayFrame = frame;
        }
        
        auto tracker = std::make_unique<cysnic::TargetTracker>(config);
        if (tracker->init(frame, roi, targetId++)) {
            trackers.push_back(std::move(tracker));
        } else {
            spdlog::warn("Failed to initialize tracker for ROI. Discarding.");
            continue;
        }
        
        // Draw the ROI on frame to show what has been selected
        cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);
    }

    if (trackers.empty()) {
        spdlog::error("No ROIs selected. Exiting.");
        return 1;
    }
    
    cv::destroyWindow("CYSNIC Tracker - Select ROIs (Space to finish)");
    spdlog::info("Tracking {} targets...", trackers.size());

    auto prevTime = std::chrono::high_resolution_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        auto currTime = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(currTime - prevTime).count() / 1e9;
        if (dt <= 0.0) dt = 0.033; // Fallback to 30fps if timing is weird
        double fps = 1.0 / dt;
        prevTime = currTime;

        cv::Mat grayFrame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        } else {
            grayFrame = frame;
        }

        // Process all trackers in parallel using extracted free function
        std::vector<std::optional<cv::Rect2d>> results = processTrackers(trackers, grayFrame, dt);

        // Gather results and draw HUD
        for (size_t i = 0; i < results.size(); ++i) {
            auto trackedBox = results[i];

            if (trackedBox.has_value()) {
                cv::Rect2d box = trackedBox.value();
                bool occluded = trackers[i]->getOcclusionState();
                auto vel = trackers[i]->getVelocity();
                
                // Draw HUD with actual KF velocity
                drawTacticalHUD(frame, box, i+1, occluded, vel.first, vel.second);
            } else {
                cv::putText(frame, "T" + std::to_string(i+1) + " LOST", cv::Point(20, 50 + i * 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }
        
        // Global Telemetry
        cv::Scalar globalHudColor(0, 0, 0);
        cv::Scalar meanGlobal = cv::mean(frame(cv::Rect(0, 0, std::min(300, frame.cols), std::min(50, frame.rows))));
        if ((0.299 * meanGlobal[2] + 0.587 * meanGlobal[1] + 0.114 * meanGlobal[0]) < 128) {
            globalHudColor = cv::Scalar(255, 255, 255);
        }
        
        std::string globalTelemetry = fmt::format("SYS: CYSNIC v1.0 | FPS: {:.1f} | TARGETS: {} | RES: {}x{}", fps, trackers.size(), frame.cols, frame.rows);
        cv::putText(frame, globalTelemetry, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, globalHudColor, 1, cv::LINE_AA);

        cv::imshow("CYSNIC Tracker", frame);

        if (cv::waitKey(1) == 27) { // ESC to exit
            break;
        }
    }

    return 0;
}
