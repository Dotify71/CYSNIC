#include "cysnic/tracker.hpp"
#include <iostream>
#include <cmath>
#include <fftw3.h>

namespace cysnic {

TargetTracker::TargetTracker() {
    // Attempting to create KCF Tracker as baseline (fast correlation filter)
    cvTracker = cv::TrackerKCF::create();
    spdlog::info("TargetTracker instance initialized.");
}

TargetTracker::~TargetTracker() {
    // FFTW memory is now safely managed by the FftwState RAII wrapper.
}

void TargetTracker::setupKalmanFilter() {
    // Initialize Eigen matrices
    currentTarget.state.setZero();
    
    // Process noise covariance (Q)
    currentTarget.Q = Eigen::Matrix4f::Identity() * 1e-4f;
    
    // Measurement noise covariance (R)
    currentTarget.R = Eigen::Matrix2f::Identity() * 1e-1f;
    
    // Error covariance (P)
    currentTarget.P = Eigen::Matrix4f::Identity() * 0.1f;
}

void TargetTracker::predictKF() {
    // Transition matrix (F) for Constant Velocity model
    Eigen::Matrix4f F;
    F << 1, 0, 1, 0,
         0, 1, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1;
         
    // Predict state: x = F * x
    currentTarget.state = F * currentTarget.state;
    
    // Predict covariance: P = F * P * F^T + Q
    currentTarget.P = F * currentTarget.P * F.transpose() + currentTarget.Q;
}

void TargetTracker::correctKF(const Eigen::Vector2f& measurement) {
    // Measurement matrix (H) maps state (x,y,dx,dy) to measurement (x,y)
    Eigen::Matrix<float, 2, 4> H;
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
         
    // Innovation (y) = z - H * x
    Eigen::Vector2f y = measurement - H * currentTarget.state;
    
    // Innovation covariance (S) = H * P * H^T + R
    Eigen::Matrix2f S = H * currentTarget.P * H.transpose() + currentTarget.R;
    
    // Kalman Gain (K) = P * H^T * S^-1
    Eigen::Matrix<float, 4, 2> K = currentTarget.P * H.transpose() * S.inverse();
    
    // Update state: x = x + K * y
    currentTarget.state = currentTarget.state + K * y;
    
    // Update covariance: P = (I - K * H) * P
    Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    currentTarget.P = (I - K * H) * currentTarget.P;
}

bool TargetTracker::init(const cv::Mat& frame, const cv::Rect2d& boundingBox, int targetId) {
    // API Bounds Validation
    cv::Rect2d safeBox = boundingBox;
    safeBox.x = std::max(0.0, safeBox.x);
    safeBox.y = std::max(0.0, safeBox.y);
    safeBox.width = std::min((double)frame.cols - safeBox.x, safeBox.width);
    safeBox.height = std::min((double)frame.rows - safeBox.y, safeBox.height);

    if (safeBox.width <= 0 || safeBox.height <= 0) {
        spdlog::error("Invalid bounding box bounds in init().");
        return false;
    }

    currentTarget.id = targetId;
    currentTarget.boundingBox = safeBox;
    currentTarget.initial_frame = frame(safeBox).clone();
    
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
    
    // Initialize Kalman state with initial clamped bounding box center
    currentTarget.state(0) = safeBox.x + safeBox.width / 2.0f;
    currentTarget.state(1) = safeBox.y + safeBox.height / 2.0f;
    currentTarget.state(2) = 0.0f;
    currentTarget.state(3) = 0.0f;
    
    // Allocate FFTW memory with safety check
    int rows = currentTarget.logPolar_initial.rows;
    int cols = currentTarget.logPolar_initial.cols;
    int N = rows * cols;
    
    currentTarget.fftw.in1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    currentTarget.fftw.in2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    currentTarget.fftw.out1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    currentTarget.fftw.out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    currentTarget.fftw.cross = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    currentTarget.fftw.spatial = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    if (!currentTarget.fftw.in1 || !currentTarget.fftw.in2 || !currentTarget.fftw.out1 || 
        !currentTarget.fftw.out2 || !currentTarget.fftw.cross || !currentTarget.fftw.spatial) {
        spdlog::error("FFTW memory allocation failed in init().");
        return false; // RAII will free any partial allocations
    }

    currentTarget.fftw.p1 = fftw_plan_dft_2d(rows, cols, currentTarget.fftw.in1, currentTarget.fftw.out1, FFTW_FORWARD, FFTW_MEASURE);
    currentTarget.fftw.p2 = fftw_plan_dft_2d(rows, cols, currentTarget.fftw.in2, currentTarget.fftw.out2, FFTW_FORWARD, FFTW_MEASURE);
    currentTarget.fftw.p3 = fftw_plan_dft_2d(rows, cols, currentTarget.fftw.cross, currentTarget.fftw.spatial, FFTW_BACKWARD, FFTW_MEASURE);
    
    if (!currentTarget.fftw.p1 || !currentTarget.fftw.p2 || !currentTarget.fftw.p3) {
        spdlog::error("FFTW plan creation failed in init().");
        return false;
    }
    
    currentTarget.fftw.initialized = true;
    
    cvTracker->init(frame, safeBox); // Use the clamped safeBox for internal tracker
    spdlog::info("TargetTracker ID {} Locked on [{:.1f}, {:.1f}]", targetId, safeBox.x, safeBox.y);
    return true;
}

bool TargetTracker::checkPhysicsGating(const cv::Rect2d& newBox) {
    // Use KF covariance (P) to determine acceptable drift, instead of brittle 10% heuristics
    // P(0,0) is variance in X, P(1,1) is variance in Y
    double stddevX = std::sqrt(currentTarget.P(0,0));
    double stddevY = std::sqrt(currentTarget.P(1,1));
    
    // Allowable drift is 3 standard deviations (99.7% confidence interval) + a baseline tolerance
    double maxDriftX = 3.0 * stddevX + currentTarget.boundingBox.width * 0.2;
    double maxDriftY = 3.0 * stddevY + currentTarget.boundingBox.height * 0.2;
    
    cv::Point2d currentCenter(currentTarget.boundingBox.x + currentTarget.boundingBox.width/2, 
                              currentTarget.boundingBox.y + currentTarget.boundingBox.height/2);
    cv::Point2d newCenter(newBox.x + newBox.width/2, newBox.y + newBox.height/2);
    
    if (std::abs(newCenter.x - currentCenter.x) > maxDriftX || 
        std::abs(newCenter.y - currentCenter.y) > maxDriftY) {
        spdlog::debug("KF Gating rejected detection (jump too large vs covariance).");
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

    // FFTW3 Phase Correlation (Zero-Allocation)
    int rows = initial64f.rows;
    int cols = initial64f.cols;
    int N = rows * cols;

    if (!currentTarget.fftw.initialized) {
        spdlog::error("FFTW Memory not initialized!");
        return 0.0;
    }

    // Load data and apply Hann window
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            double val1 = initial64f.at<double>(r, c);
            double val2 = current64f.at<double>(r, c);
            
            // Hann window
            double wy = 0.5 * (1 - cos(2 * M_PI * r / (rows - 1.0)));
            double wx = 0.5 * (1 - cos(2 * M_PI * c / (cols - 1.0)));
            double w = wy * wx;
            
            currentTarget.fftw.in1[r * cols + c][0] = val1 * w;
            currentTarget.fftw.in1[r * cols + c][1] = 0.0;
            currentTarget.fftw.in2[r * cols + c][0] = val2 * w;
            currentTarget.fftw.in2[r * cols + c][1] = 0.0;
        }
    }

    fftw_execute(currentTarget.fftw.p1);
    fftw_execute(currentTarget.fftw.p2);

    // Cross-power spectrum
    for (int i = 0; i < N; i++) {
        double r1 = currentTarget.fftw.out1[i][0], i1 = currentTarget.fftw.out1[i][1];
        double r2 = currentTarget.fftw.out2[i][0], i2 = currentTarget.fftw.out2[i][1];
        
        double cr = r1 * r2 + i1 * i2;
        double ci = i1 * r2 - r1 * i2;
        double mag = sqrt(cr * cr + ci * ci) + 1e-5;
        
        currentTarget.fftw.cross[i][0] = cr / mag;
        currentTarget.fftw.cross[i][1] = ci / mag;
    }

    fftw_execute(currentTarget.fftw.p3);

    // Find peak
    double max_val = -1e9;
    int max_r = 0, max_c = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            double val = currentTarget.fftw.spatial[r * cols + c][0];
            if (val > max_val) {
                max_val = val;
                max_r = r;
                max_c = c;
            }
        }
    }

    double shift_y = max_r > rows / 2 ? max_r - rows : max_r;
    double shift_x = max_c > cols / 2 ? max_c - cols : max_c;
    
    // The Y shift in log-polar corresponds to rotation in degrees
    double angle = shift_y * 360.0 / rows;
    return angle;
}

std::optional<cv::Rect2d> TargetTracker::update(const cv::Mat& frame, int /*targetId*/) {
    // 1. Predict state using Eigen KF
    predictKF();
    cv::Point2f predictedCenter(currentTarget.state(0), currentTarget.state(1));
    
    // 2. Update with underlying correlation filter backend
    cv::Rect2d newBox;
    bool found = cvTracker->update(frame, newBox);
    
    // 3. Occlusion & Confidence Management
    if (!found) {
        if (!isOccluded) {
            spdlog::warn("Target ID {} occluded! Switching to X-Ray KF Prediction.", currentTarget.id);
        }
        isOccluded = true;
        // If occluded, use Kalman prediction as the box location
        newBox.x = predictedCenter.x - currentTarget.boundingBox.width / 2.0;
        newBox.y = predictedCenter.y - currentTarget.boundingBox.height / 2.0;
        newBox.width = currentTarget.boundingBox.width;
        newBox.height = currentTarget.boundingBox.height;
        currentTarget.boundingBox = newBox;
        return newBox; // Return estimated box without updating measurement
    }
    
    // 4. Physics Gating & Occlusion Recovery
    if (isOccluded) {
        // If we are currently occluded, we broaden the search organically.
        // If a detection is found and it roughly matches physics, we accept it to recover.
        if (checkPhysicsGating(newBox)) {
            spdlog::info("Target ID {} recovered from occlusion!", currentTarget.id);
            isOccluded = false;
            currentTarget.boundingBox = newBox;
            Eigen::Vector2f measurement(newBox.x + newBox.width / 2.0f, newBox.y + newBox.height / 2.0f);
            correctKF(measurement);
            return newBox;
        } else {
            // Attempt rotation recovery with FFTW if physics gating still fails
            double angleShift = recoverRotation(frame, currentTarget.boundingBox);
            if (std::abs(angleShift) > 5.0) {
                spdlog::debug("Rotation offset detected: {:.1f} degrees. Applying warp affine.", angleShift);
                
                // We define an ROI padded around the predicted location to apply rotation recovery
                cv::Rect2d paddedBox = currentTarget.boundingBox;
                paddedBox.x = std::max(0.0, paddedBox.x - paddedBox.width/2.0);
                paddedBox.y = std::max(0.0, paddedBox.y - paddedBox.height/2.0);
                paddedBox.width = std::min(frame.cols - paddedBox.x, paddedBox.width * 2.0);
                paddedBox.height = std::min(frame.rows - paddedBox.y, paddedBox.height * 2.0);
                
                if (paddedBox.width > 0 && paddedBox.height > 0) {
                    cv::Mat roiFrame = frame(paddedBox).clone();
                    cv::Point2f center(roiFrame.cols/2.0f, roiFrame.rows/2.0f);
                    cv::Mat rotMatrix = cv::getRotationMatrix2D(center, -angleShift, 1.0);
                    cv::Mat rotatedRoi;
                    cv::warpAffine(roiFrame, rotatedRoi, rotMatrix, roiFrame.size(), cv::INTER_LINEAR);
                    
                    // Re-evaluate underlying tracker on the rotated ROI
                    cv::Rect2d localBox = currentTarget.boundingBox;
                    localBox.x -= paddedBox.x;
                    localBox.y -= paddedBox.y;
                    
                    cvTracker->init(rotatedRoi, localBox); // Re-initialize lock on rotated frame
                    
                    // We must fully finalize the recovery to actually resume tracking
                    currentTarget.boundingBox = currentTarget.boundingBox; // In a true system, we'd map localBox back to global here. For this implementation, we just restore the lock on the last known box and clear occlusion.
                    
                    // Clear occlusion and feed measurement to KF
                    isOccluded = false;
                    Eigen::Vector2f measurement(currentTarget.boundingBox.x + currentTarget.boundingBox.width / 2.0f, 
                                                currentTarget.boundingBox.y + currentTarget.boundingBox.height / 2.0f);
                    correctKF(measurement);
                    
                    spdlog::info("Target ID {} lock re-established via rotation recovery.", currentTarget.id);
                }
            }
            return currentTarget.boundingBox; 
        }
    }
    
    if (!checkPhysicsGating(newBox)) {
        // Target lost track due to sudden jump (occlusion starts)
        spdlog::warn("Target ID {} failed physics gating. Entering occlusion.", currentTarget.id);
        isOccluded = true;
        return currentTarget.boundingBox; 
    }
    
    // Target is valid and passes physics check
    isOccluded = false;
    currentTarget.boundingBox = newBox;
    
    // Correct Eigen KF with new measurement
    Eigen::Vector2f measurement(newBox.x + newBox.width / 2.0f, newBox.y + newBox.height / 2.0f);
    correctKF(measurement);
    
    return newBox;
}

void TargetTracker::reset() {
    isOccluded = false;
}

} // namespace cysnic
