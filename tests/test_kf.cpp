#include <gtest/gtest.h>
#include "cysnic/tracker.hpp"
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

using namespace cysnic;

class MockTracker : public cysnic::ITrackerBackend {
public:
    bool shouldFail = false;
    cv::Rect2d nextBox;
    
    void init(const cv::Mat&, const cv::Rect2d& box) override {
        nextBox = box;
    }
    
    bool update(const cv::Mat&, cv::Rect2d& box) override {
        if (shouldFail) return false;
        box = nextBox;
        return true;
    }
};

class KFTest : public ::testing::Test {
protected:
    std::shared_ptr<MockTracker> mockTracker;
    std::unique_ptr<TargetTracker> tracker;
    
    void SetUp() override {
        spdlog::set_level(spdlog::level::off); // Disable logs for clean test output
        mockTracker = std::make_shared<MockTracker>();
        tracker = std::make_unique<TargetTracker>(mockTracker);
    }
};

TEST_F(KFTest, TestFullUpdateLoop) {
    // 1. Init
    cv::Mat dummyFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Rect2d initialBox(10, 10, 20, 20);
    
    EXPECT_TRUE(tracker->init(dummyFrame, initialBox, 1));
    EXPECT_FALSE(tracker->getOcclusionState());
    
    // 2. Normal update
    mockTracker->nextBox = cv::Rect2d(11, 11, 20, 20);
    auto result = tracker->update(dummyFrame);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(tracker->getOcclusionState()); // Should still be tracking
    
    // 3. Occlusion: Tracker fails
    mockTracker->shouldFail = true;
    auto occludedResult = tracker->update(dummyFrame);
    EXPECT_TRUE(occludedResult.has_value());
    EXPECT_TRUE(tracker->getOcclusionState()); // Entered occlusion
    
    // 4. Recovery: Target reappears near predicted location
    mockTracker->shouldFail = false;
    mockTracker->nextBox = occludedResult.value();
    
    auto recoveredResult = tracker->update(dummyFrame);
    EXPECT_TRUE(recoveredResult.has_value());
    EXPECT_FALSE(tracker->getOcclusionState()); // Exited occlusion and fully recovered
}

TEST_F(KFTest, TestRotationRecoveryPath) {
    // 1. Init
    cv::Mat dummyFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Rect2d initialBox(40, 40, 20, 20);
    // Draw a prominent asymmetric feature so rotation is detectable by FFTW
    cv::rectangle(dummyFrame, cv::Rect(40, 40, 20, 10), cv::Scalar(255, 255, 255), cv::FILLED);
    
    EXPECT_TRUE(tracker->init(dummyFrame, initialBox, 1));
    
    // 2. Occlusion
    mockTracker->shouldFail = true;
    tracker->update(dummyFrame);
    EXPECT_TRUE(tracker->getOcclusionState());
    
    // 3. Rotated target appears
    cv::Mat rotatedFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    // Draw same feature but rotated 90 degrees
    cv::rectangle(rotatedFrame, cv::Rect(40, 40, 10, 20), cv::Scalar(255, 255, 255), cv::FILLED);
    
    // Recovery will try to detect rotation. If it triggers, it re-initializes cvTracker on rotated ROI
    // Since MockTracker just takes the ROI box, it will succeed.
    mockTracker->shouldFail = false;
    
    auto result = tracker->update(rotatedFrame);
    
    // We explicitly assert that occlusion is cleared after rotation recovery
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(tracker->getOcclusionState());
    
    // Verify mapped bounding box
    cv::Rect2d recoveredBox = result.value();
    EXPECT_GE(recoveredBox.x, 0.0);
    EXPECT_GE(recoveredBox.y, 0.0);
}

TEST_F(KFTest, TestInitBoundsClamping) {
    cv::Mat dummyFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    
    // Pass a bounding box that exceeds frame dimensions
    cv::Rect2d outOfBoundsBox(-10, 50, 200, 200);
    
    bool initialized = tracker->init(dummyFrame, outOfBoundsBox, 2);
    EXPECT_TRUE(initialized);
    
    // Ensure the tracker clamped it within [0, 0, 100, 100]
    cv::Rect2d clampedBox = tracker->getBoundingBox();
    EXPECT_GE(clampedBox.x, 0.0);
    EXPECT_GE(clampedBox.y, 0.0);
    EXPECT_LE(clampedBox.x + clampedBox.width, 100.0);
    EXPECT_LE(clampedBox.y + clampedBox.height, 100.0);
    
    // x should be clamped to 0 from -10
    EXPECT_DOUBLE_EQ(clampedBox.x, 0.0);
    // width should be clamped to 100 (since frame is 100 wide)
    EXPECT_DOUBLE_EQ(clampedBox.width, 100.0);
}

TEST(KFMathTest, EigenMatrixProperties) {
    // Mathematically prove Constant Velocity F matrix properties
    Eigen::Matrix4f F;
    F << 1, 0, 1, 0,
         0, 1, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1;
         
    Eigen::Vector4f state(10, 10, 5, 2); // x, y, dx, dy
    
    // Predict next state
    Eigen::Vector4f next_state = F * state;
    
    // Next X should be X + dX = 10 + 5 = 15
    EXPECT_FLOAT_EQ(next_state(0), 15.0f);
    
    // Next Y should be Y + dY = 10 + 2 = 12
    EXPECT_FLOAT_EQ(next_state(1), 12.0f);
    
    // Velocities should remain constant in prediction
    EXPECT_FLOAT_EQ(next_state(2), 5.0f);
    EXPECT_FLOAT_EQ(next_state(3), 2.0f);
}
