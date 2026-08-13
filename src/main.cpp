#include "cysnic/tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <future>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path_or_camera_id>" << std::endl;
        return 1;
    }

    cv::VideoCapture cap;
    if (std::string(argv[1]).length() == 1 && isdigit(argv[1][0])) {
        cap.open(std::stoi(argv[1]));
    } else {
        cap.open(argv[1]);
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source " << argv[1] << std::endl;
        return 1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Empty first frame" << std::endl;
        return 1;
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
        std::cerr << "Error: No ROIs selected. Exiting." << std::endl;
        return 1;
    }
    
    cv::destroyWindow("CYSNIC Tracker - Select ROIs (Space to finish)");

    while (cap.read(frame)) {
        if (frame.empty()) break;

        cv::Mat grayFrame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        } else {
            grayFrame = frame;
        }

        // Process all trackers in parallel
        std::vector<std::future<std::optional<cv::Rect2d>>> futures;
        for (auto& tracker : trackers) {
            futures.push_back(std::async(std::launch::async, [&tracker, grayFrame]() {
                return tracker->update(grayFrame);
            }));
        }

        // Gather results and draw
        for (size_t i = 0; i < futures.size(); ++i) {
            auto trackedBox = futures[i].get();
            cv::Scalar color((i * 50) % 255, (255 - i * 80) % 255, (i * 120) % 255); // Different color for each

            if (trackedBox.has_value()) {
                cv::rectangle(frame, trackedBox.value(), color, 2);
                cv::putText(frame, "T" + std::to_string(i+1), cv::Point(trackedBox.value().x, trackedBox.value().y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            } else {
                cv::putText(frame, "T" + std::to_string(i+1) + " Lost", cv::Point(20, 50 + i * 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }
        }

        cv::imshow("CYSNIC Tracker", frame);

        if (cv::waitKey(1) == 27) { // ESC to exit
            break;
        }
    }

    return 0;
}
