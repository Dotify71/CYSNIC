#include "cysnic/tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <future>
#include <chrono>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <spdlog/spdlog.h>

void drawTacticalHUD(cv::Mat& frame, const cv::Rect2d& box, int id, bool occluded, double dx = 0, double dy = 0) {
    cv::Scalar hudColor(0, 0, 0); // Black for everything
    int lineType = cv::LINE_AA;
    
    // Draw corners
    int cornerLen = 20;
    int t = 2; // thickness
    
    if (occluded) {
        // Draw X-RAY dashed representation (or just standard corners with crosshair)
        cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
        cv::drawMarker(frame, center, hudColor, cv::MARKER_CROSS, 20, 1, lineType);
        
        // Dashed box for X-Ray
        cv::rectangle(frame, box, hudColor, 1, lineType);
        
        std::string status = "STATUS: X-RAY / COASTING";
        cv::putText(frame, status, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, hudColor, 1, lineType);
    } else {
        // Top left
        cv::line(frame, cv::Point(box.x, box.y), cv::Point(box.x + cornerLen, box.y), hudColor, t, lineType);
        cv::line(frame, cv::Point(box.x, box.y), cv::Point(box.x, box.y + cornerLen), hudColor, t, lineType);
        // Top right
        cv::line(frame, cv::Point(box.x + box.width, box.y), cv::Point(box.x + box.width - cornerLen, box.y), hudColor, t, lineType);
        cv::line(frame, cv::Point(box.x + box.width, box.y), cv::Point(box.x + box.width, box.y + cornerLen), hudColor, t, lineType);
        // Bottom left
        cv::line(frame, cv::Point(box.x, box.y + box.height), cv::Point(box.x + cornerLen, box.y + box.height), hudColor, t, lineType);
        cv::line(frame, cv::Point(box.x, box.y + box.height), cv::Point(box.x, box.y + box.height - cornerLen), hudColor, t, lineType);
        // Bottom right
        cv::line(frame, cv::Point(box.x + box.width, box.y + box.height), cv::Point(box.x + box.width - cornerLen, box.y + box.height), hudColor, t, lineType);
        cv::line(frame, cv::Point(box.x + box.width, box.y + box.height), cv::Point(box.x + box.width, box.y + box.height - cornerLen), hudColor, t, lineType);
        
        // Status Text
        std::string status = "STATUS: TRACK LOCKED";
        cv::putText(frame, status, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, hudColor, 1, lineType);
    }
    
    // Calculations / Telemetry placed strictly outside the box
    char telemetry[128];
    sprintf(telemetry, "ID: %02d | POS: [%.1f, %.1f] | VEL: [%.1f, %.1f]", id, box.x, box.y, dx, dy);
    cv::putText(frame, telemetry, cv::Point(box.x, box.y + box.height + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, hudColor, 1, lineType);
}
int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::debug); // Set global log level
    spdlog::info("CYSNIC Tactical Tracker starting...");

    if (argc < 2) {
        spdlog::error("Usage: {} <video_path_or_camera_id>", argv[0]);
        return -1;
    }

    std::string source = argv[1];
    cv::VideoCapture cap;

    if (source.length() == 1 && isdigit(source[0])) {
        cap.open(std::stoi(source));
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        spdlog::error("Failed to open video source: {}", source);
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
        
        auto tracker = std::make_unique<cysnic::TargetTracker>();
        tracker->init(grayFrame, roi, targetId++);
        trackers.push_back(std::move(tracker));
        
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
        double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(currTime - prevTime).count();
        prevTime = currTime;

        cv::Mat grayFrame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        } else {
            grayFrame = frame;
        }

        // Process all trackers in parallel using Intel TBB
        std::vector<std::optional<cv::Rect2d>> results(trackers.size());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, trackers.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    results[i] = trackers[i]->update(grayFrame);
                }
            });

        // Gather results and draw HUD
        for (size_t i = 0; i < results.size(); ++i) {
            auto trackedBox = results[i];

            if (trackedBox.has_value()) {
                cv::Rect2d box = trackedBox.value();
                bool occluded = trackers[i]->getOcclusionState();
                auto vel = trackers[i]->getVelocity();
                
                // Draw HUD with actual EKF velocity
                drawTacticalHUD(frame, box, i+1, occluded, vel.first, vel.second);
            } else {
                cv::putText(frame, "T" + std::to_string(i+1) + " LOST", cv::Point(20, 50 + i * 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }
        
        // Global Telemetry
        char globalTelemetry[128];
        sprintf(globalTelemetry, "SYS: CYSNIC v1.0 | FPS: %.1f | TARGETS: %zu | RES: %dx%d", fps, trackers.size(), frame.cols, frame.rows);
        cv::putText(frame, globalTelemetry, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        cv::imshow("CYSNIC Tracker", frame);

        if (cv::waitKey(1) == 27) { // ESC to exit
            break;
        }
    }

    return 0;
}
