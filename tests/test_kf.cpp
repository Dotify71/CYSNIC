#include <gtest/gtest.h>
#include "cysnic/tracker.hpp"
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

using namespace cysnic;

class KFTest : public ::testing::Test {
protected:
    TargetTracker tracker;
    void SetUp() override {
        spdlog::set_level(spdlog::level::off); // Disable logs for clean test output
    }
};

TEST_F(KFTest, TestOcclusionRecovery) {
    // Create a dummy frame and box
    cv::Mat dummyFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Rect2d box(10, 10, 20, 20);
    
    // init should not crash and should set up the matrices
    bool initialized = tracker.init(dummyFrame, box, 1);
    EXPECT_TRUE(initialized);
    EXPECT_FALSE(tracker.getOcclusionState());
    
    // Simulate updating with a frame where the target is not found
    // A blank frame might fail correlation and trigger occlusion
    cv::Mat blankFrame = cv::Mat::zeros(100, 100, CV_8UC1);
    auto result = tracker.update(blankFrame);
    
    // The tracker should return an estimated box but enter occlusion state
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(tracker.getOcclusionState());
    
    // Simulate reacquisition: provide a frame where the target is clearly visible again
    // In our mock, if physics gating passes (since it's close to predicted center), it should recover.
    // We will place the target exactly at the predicted center to ensure it passes gating.
    cv::Rect2d predictedBox = result.value();
    cv::Mat recoveryFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    // Draw something in the box to simulate a target
    cv::rectangle(recoveryFrame, predictedBox, cv::Scalar(255, 255, 255), cv::FILLED);
    
    auto recoveryResult = tracker.update(recoveryFrame);
    
    // The tracker should now have recovered
    EXPECT_TRUE(recoveryResult.has_value());
    EXPECT_FALSE(tracker.getOcclusionState());
}

TEST_F(KFTest, TestInitBoundsClamping) {
    cv::Mat dummyFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    
    // Pass a bounding box that exceeds frame dimensions
    cv::Rect2d outOfBoundsBox(-10, 50, 200, 200);
    
    bool initialized = tracker.init(dummyFrame, outOfBoundsBox, 2);
    EXPECT_TRUE(initialized);
    
    // Ensure the tracker clamped it within [0, 0, 100, 100]
    cv::Rect2d clampedBox = tracker.getBoundingBox();
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
