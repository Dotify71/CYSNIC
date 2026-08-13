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

struct FftwState {
    fftw_complex *in1 = nullptr, *in2 = nullptr, *out1 = nullptr, *out2 = nullptr, *cross = nullptr, *spatial = nullptr;
    fftw_plan p1 = nullptr, p2 = nullptr, p3 = nullptr;
    bool initialized = false;

    FftwState() = default;
    ~FftwState() {
        if (p1) fftw_destroy_plan(p1);
        if (p2) fftw_destroy_plan(p2);
        if (p3) fftw_destroy_plan(p3);
        if (in1) fftw_free(in1);
        if (in2) fftw_free(in2);
        if (out1) fftw_free(out1);
        if (out2) fftw_free(out2);
        if (cross) fftw_free(cross);
        if (spatial) fftw_free(spatial);
    }
    
    // Disable copying
    FftwState(const FftwState&) = delete;
    FftwState& operator=(const FftwState&) = delete;
    
    // Enable moving
    FftwState(FftwState&& other) noexcept {
        *this = std::move(other);
    }
    FftwState& operator=(FftwState&& other) noexcept {
        if (this != &other) {
            std::swap(in1, other.in1); std::swap(in2, other.in2);
            std::swap(out1, other.out1); std::swap(out2, other.out2);
            std::swap(cross, other.cross); std::swap(spatial, other.spatial);
            std::swap(p1, other.p1); std::swap(p2, other.p2); std::swap(p3, other.p3);
            std::swap(initialized, other.initialized);
        }
        return *this;
    }
};

struct TrackTarget {
    int id;
    cv::Rect2d boundingBox; // Ensure this matches implementation
    
    // Eigen-based Linear KF State (x, y, dx, dy)
    Eigen::Vector4f state;
    Eigen::Matrix4f P; // Estimate covariance
    Eigen::Matrix4f Q; // Process noise covariance
    Eigen::Matrix2f R; // Measurement noise covariance
    
    double confidence; // Peak-to-Sidelobe Ratio (PSR)
    
    // FFTW specific state using RAII
    FftwState fftw;
    
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
    cv::Rect2d getBoundingBox() const { return currentTarget.boundingBox; }
private:
    cv::Ptr<cv::Tracker> cvTracker;
    TrackTarget currentTarget;
    
    bool isOccluded = false;
    double psrThreshold = 0.5; // Example threshold
    
    // Internal KF math and physics gating
    void setupKalmanFilter();
    void predictKF();
    void correctKF(const Eigen::Vector2f& measurement);
    bool checkPhysicsGating(const cv::Rect2d& newBox);
    
    // Rotation Recovery
    double recoverRotation(const cv::Mat& currentFrame, const cv::Rect2d& box);
};

} // namespace cysnic
