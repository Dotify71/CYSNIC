#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/tracking.hpp>
#include <memory>
#include <optional>

namespace cysnic {

struct TrackTarget {
    int id;
    cv::Rect2d bounding_box;
    cv::KalmanFilter kalman;
    double confidence; // Peak-to-Sidelobe Ratio (PSR)
    
    // Original template for rotation recovery
    cv::Mat initial_frame; 
};

class TargetTracker {
public:
    TargetTracker();
    ~TargetTracker();

    // Initialize the tracker with a specific bounding box for a target
    bool init(const cv::Mat& frame, const cv::Rect2d& boundingBox, int targetId = 1);
    
    // Update the tracker state with a new frame
    std::optional<cv::Rect2d> update(const cv::Mat& frame, int targetId = 1);
    
    // Reset tracker
    void reset();

private:
    cv::Ptr<cv::Tracker> cvTracker;
    TrackTarget currentTarget;
    
    bool isOccluded = false;
    double psrThreshold = 0.5; // Example threshold
    
    // Physics gating
    bool checkPhysicsGating(const cv::Rect2d& newBox);
    void setupKalmanFilter();
};

} // namespace cysnic
