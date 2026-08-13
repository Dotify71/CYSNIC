#include <gtest/gtest.h>
#include "cysnic/tracker.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <optional>

class MockMainBackend : public cysnic::ITrackerBackend {
public:
    void init(const cv::Mat&, const cv::Rect2d&) override {}
    bool update(const cv::Mat&, cv::Rect2d&) override {
        // Mock successful tracking by leaving box unchanged
        return true;
    }
};

TEST(MainLoopTest, ProcessTrackersParallel) {
    cv::Mat dummyFrame = cv::Mat::zeros(100, 100, CV_8UC3);
    
    std::vector<std::unique_ptr<cysnic::TargetTracker>> trackers;
    for (int i = 0; i < 4; ++i) {
        auto t = std::make_unique<cysnic::TargetTracker>(std::make_shared<MockMainBackend>());
        cv::Rect2d box(10 + i * 10, 10 + i * 10, 20, 20);
        t->init(dummyFrame, box, i + 1);
        trackers.push_back(std::move(t));
    }
    
    // Process them in parallel
    double dt = 0.033;
    auto results = cysnic::processTrackers(trackers, dummyFrame, dt);
    
    ASSERT_EQ(results.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(results[i].has_value());
        EXPECT_EQ(results[i]->x, 10 + i * 10);
        EXPECT_EQ(results[i]->y, 10 + i * 10);
    }
}
