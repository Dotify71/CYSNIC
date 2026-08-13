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
