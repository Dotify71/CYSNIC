#include "cysnic/tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

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

    // Select ROI for the first target
    cv::Rect2d roi = cv::selectROI("CYSNIC Tracker", frame, false, false);
    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "Error: Empty ROI selected. Exiting." << std::endl;
        return 1;
    }

    cysnic::TargetTracker tracker;
    tracker.init(frame, roi);

    while (cap.read(frame)) {
        if (frame.empty()) break;

        auto trackedBox = tracker.update(frame);

        if (trackedBox.has_value()) {
            cv::rectangle(frame, trackedBox.value(), cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "Target Lost", cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("CYSNIC Tracker", frame);

        if (cv::waitKey(1) == 27) { // ESC to exit
            break;
        }
    }

    return 0;
}
