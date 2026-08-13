#include "cysnic/tracker.hpp"
#include <iostream>
#include <cmath>

namespace cysnic {

TargetTracker::TargetTracker() {
    // Attempting to create KCF Tracker as baseline (fast correlation filter)
    cvTracker = cv::TrackerKCF::create();
}

TargetTracker::~TargetTracker() {}

void TargetTracker::setupKalmanFilter() {
    // 4 state variables (x, y, dx, dy), 2 measurements (x, y)
    currentTarget.kalman.init(4, 2, 0);
    
    // Transition matrix (A)
    // [1 0 1 0]
    // [0 1 0 1]
    // [0 0 1 0]
    // [0 0 0 1]
    currentTarget.kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    
    // Measurement matrix (H)
    cv::setIdentity(currentTarget.kalman.measurementMatrix);
    
    // Process noise covariance (Q)
    cv::setIdentity(currentTarget.kalman.processNoiseCov, cv::Scalar::all(1e-4));
    
    // Measurement noise covariance (R)
    cv::setIdentity(currentTarget.kalman.measurementNoiseCov, cv::Scalar::all(1e-1));
    
    // Error covariance (P)
    cv::setIdentity(currentTarget.kalman.errorCovPost, cv::Scalar::all(.1));
}

bool TargetTracker::init(const cv::Mat& frame, const cv::Rect2d& boundingBox, int targetId) {
    currentTarget.id = targetId;
    currentTarget.boundingBox = boundingBox;
    currentTarget.initial_frame = frame(boundingBox).clone();
    
    // Compute log-polar transform of the initial frame for rotation recovery
    cv::Mat grayInitial;
    if (currentTarget.initial_frame.channels() == 3) {
        cv::cvtColor(currentTarget.initial_frame, grayInitial, cv::COLOR_BGR2GRAY);
    } else {
        grayInitial = currentTarget.initial_frame.clone();
    }
    
    cv::Point2f center(grayInitial.cols / 2.0f, grayInitial.rows / 2.0f);
    double M = grayInitial.cols / std::log(grayInitial.cols / 2.0);
    cv::logPolar(grayInitial, currentTarget.logPolar_initial, center, M, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
    
    setupKalmanFilter();
    
    // Initialize Kalman state with initial bounding box center
    currentTarget.kalman.statePost.at<float>(0) = boundingBox.x + boundingBox.width / 2.0f;
    currentTarget.kalman.statePost.at<float>(1) = boundingBox.y + boundingBox.height / 2.0f;
    currentTarget.kalman.statePost.at<float>(2) = 0.0f;
    currentTarget.kalman.statePost.at<float>(3) = 0.0f;
    
    cvTracker->init(frame, boundingBox);
    return true;
}

bool TargetTracker::checkPhysicsGating(const cv::Rect2d& newBox) {
    // Physics Rule 1: Bounding box scale shouldn't change drastically (e.g., limit 10% per frame)
    double scaleChangeX = std::abs(newBox.width - currentTarget.boundingBox.width) / currentTarget.boundingBox.width;
    double scaleChangeY = std::abs(newBox.height - currentTarget.boundingBox.height) / currentTarget.boundingBox.height;
    
    if (scaleChangeX > 0.10 || scaleChangeY > 0.10) {
        // std::cout << "Gating triggered: Target scale changed by >10%.\n";
        return false;
    }
    
    // Physics Rule 2: Reject implausible velocity jumps 
    // Example: Target center should not jump more than half its width in one frame
    cv::Point2d currentCenter(currentTarget.boundingBox.x + currentTarget.boundingBox.width/2, 
                              currentTarget.boundingBox.y + currentTarget.boundingBox.height/2);
    cv::Point2d newCenter(newBox.x + newBox.width/2, newBox.y + newBox.height/2);
    
    double dist = std::sqrt(std::pow(newCenter.x - currentCenter.x, 2) + std::pow(newCenter.y - currentCenter.y, 2));
    if (dist > currentTarget.boundingBox.width * 0.5) {
        // std::cout << "Gating triggered: Implausible velocity jump.\n";
        return false;
    }
    
    return true;
}

double TargetTracker::recoverRotation(const cv::Mat& currentFrame, const cv::Rect2d& box) {
    if (box.width <= 0 || box.height <= 0 || 
        box.x < 0 || box.y < 0 || 
        box.x + box.width > currentFrame.cols || 
        box.y + box.height > currentFrame.rows) {
        return 0.0;
    }

    cv::Mat currentCrop = currentFrame(box).clone();
    cv::Mat grayCurrent;
    if (currentCrop.channels() == 3) {
        cv::cvtColor(currentCrop, grayCurrent, cv::COLOR_BGR2GRAY);
    } else {
        grayCurrent = currentCrop.clone();
    }

    // Resize to match initial frame size if needed
    if (grayCurrent.size() != currentTarget.logPolar_initial.size()) {
        cv::resize(grayCurrent, grayCurrent, currentTarget.logPolar_initial.size());
    }

    cv::Point2f center(grayCurrent.cols / 2.0f, grayCurrent.rows / 2.0f);
    double M = grayCurrent.cols / std::log(grayCurrent.cols / 2.0);
    cv::Mat logPolar_current;
    cv::logPolar(grayCurrent, logPolar_current, center, M, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);

    // Ensure they are CV_64F for phaseCorrelate
    cv::Mat initial64f, current64f;
    currentTarget.logPolar_initial.convertTo(initial64f, CV_64F);
    logPolar_current.convertTo(current64f, CV_64F);

    cv::Point2d shift = cv::phaseCorrelate(initial64f, current64f);
    
    // The Y shift in log-polar corresponds to rotation in degrees (scaled by M)
    double angle = shift.y * 360.0 / logPolar_current.rows;
    return angle;
}

std::optional<cv::Rect2d> TargetTracker::update(const cv::Mat& frame, int /*targetId*/) {
    // 1. Predict state using Kalman Filter
    cv::Mat prediction = currentTarget.kalman.predict();
    cv::Point2f predictedCenter(prediction.at<float>(0), prediction.at<float>(1));
    
    // 2. Update with underlying correlation filter backend
    cv::Rect2d newBox;
    bool found = cvTracker->update(frame, newBox);
    
    // 3. Occlusion & Confidence Management
    if (!found) {
        isOccluded = true;
        // If occluded, use Kalman prediction as the box location
        newBox.x = predictedCenter.x - currentTarget.boundingBox.width / 2.0;
        newBox.y = predictedCenter.y - currentTarget.boundingBox.height / 2.0;
        newBox.width = currentTarget.boundingBox.width;
        newBox.height = currentTarget.boundingBox.height;
        currentTarget.boundingBox = newBox;
        return newBox; // Return estimated box without updating measurement
    }
    
    // 4. Physics Gating
    if (isOccluded || !checkPhysicsGating(newBox)) {
        // Target is either recovering from occlusion or failing physics check.
        // We could implement rotation recovery (Log-Polar + FFT) here before trusting the box.
        
        // Attempt rotation recovery
        double angleShift = recoverRotation(frame, currentTarget.boundingBox);
        // std::cout << "Rotation offset computed: " << angleShift << " degrees." << std::endl;
        // In a full implementation, if PSR improves after rotating the image patch back, we accept it.
        // For baseline, we just continue relying on Kalman predictions or reset track.
        isOccluded = true;
        return currentTarget.boundingBox; 
    }
    
    // Target is valid and passes physics check
    isOccluded = false;
    currentTarget.boundingBox = newBox;
    
    // Correct Kalman Filter with new measurement
    cv::Mat_<float> measurement(2, 1);
    measurement(0) = newBox.x + newBox.width / 2.0f;
    measurement(1) = newBox.y + newBox.height / 2.0f;
    currentTarget.kalman.correct(measurement);
    
    return newBox;
}

void TargetTracker::reset() {
    isOccluded = false;
}

} // namespace cysnic
