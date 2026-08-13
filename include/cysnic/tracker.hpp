#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include <fftw3.h>
#include <spdlog/spdlog.h>
#include <memory>
#include <optional>

namespace cysnic {

struct TrackTarget {
    int id;
    cv::Rect2d boundingBox; // Ensure this matches implementation
    
    // Eigen-based EKF State (x, y, dx, dy)
    Eigen::Vector4f state;
    Eigen::Matrix4f P; // Estimate covariance
    Eigen::Matrix4f Q; // Process noise covariance
    Eigen::Matrix2f R; // Measurement noise covariance
    
    double confidence; // Peak-to-Sidelobe Ratio (PSR)
    
    // FFTW pre-allocated plans and arrays for Zero-Allocation loop
    fftw_complex *in1, *in2, *out1, *out2, *cross, *spatial;
    fftw_plan p1, p2, p3;
    bool fftw_initialized = false;
    
    // Original template for rotation recovery
    cv::Mat initial_frame; 
    cv::Mat logPolar_initial;
};

class TargetTracker {
public:
    TargetTracker();
    ~TargetTracker();

    // Rule of Five to safely manage FFTW raw pointers
    TargetTracker(const TargetTracker&) = delete;
    TargetTracker& operator=(const TargetTracker&) = delete;
    TargetTracker(TargetTracker&&) noexcept = delete;
    TargetTracker& operator=(TargetTracker&&) noexcept = delete;

    // Initialize the tracker with the first frame and a bounding box
    bool init(const cv::Mat& frame, const cv::Rect2d& boundingBox, int targetId = 1);
    
    // Update the tracker state with a new frame
    std::optional<cv::Rect2d> update(const cv::Mat& frame, int targetId = 1);
    
    // Reset tracker
    void reset();

    // X-Ray / Occlusion state access
    bool getOcclusionState() const { return isOccluded; }
    cv::Mat getInitialFrame() const { return currentTarget.initial_frame; }
    std::pair<double, double> getVelocity() const { 
        return {currentTarget.state(2), currentTarget.state(3)}; 
    }

private:
    cv::Ptr<cv::Tracker> cvTracker;
    TrackTarget currentTarget;
    
    bool isOccluded = false;
    double psrThreshold = 0.5; // Example threshold
    
    // Physics gating
    void setupKalmanFilter();
    void predictKF();
    void correctKF(const Eigen::Vector2f& measurement);
    bool checkPhysicsGating(const cv::Rect2d& newBox);
    
    // Rotation Recovery
    double recoverRotation(const cv::Mat& currentFrame, const cv::Rect2d& box);
};

} // namespace cysnic
