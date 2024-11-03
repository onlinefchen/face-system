#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "net.h"
#include <vector>

struct FaceInfo {
    cv::Rect rect;            // 人脸框
    float score;              // 置信度
    std::vector<cv::Point2f> landmarks;  // 关键点
};

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    bool loadModel(const std::string& param_path, const std::string& bin_path);
    std::vector<FaceInfo> detect(const cv::Mat& img, float threshold = 0.5f);
    cv::Mat align(const cv::Mat& img, const FaceInfo& face);

private:
    ncnn::Net net;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/128.0f, 1/128.0f, 1/128.0f};

    void preprocess(const cv::Mat& img, ncnn::Mat& in);
    std::vector<FaceInfo> postprocess(const std::vector<ncnn::Mat>& outs, float threshold);
};

#endif // FACE_DETECTOR_H
