#ifndef FACE_SYSTEM_H
#define FACE_SYSTEM_H

#include <opencv2/opencv.hpp>
#include "net.h"
#include "face_detector.h"
#include <sqlite3.h>
#include <vector>
#include <string>

class FaceSystem {
public:
    struct Recognition {
        std::string id;
        float confidence;
    };

    FaceSystem();
    ~FaceSystem();

    bool init(const std::string& model_dir);
    bool registerFace(const cv::Mat& img, const std::string& id);
    Recognition recognize(const cv::Mat& img, float threshold = 0.6f);
    bool clear();

private:
    FaceDetector detector;
    ncnn::Net recognizer;
    sqlite3* db;

    bool initDatabase();
    bool loadModels(const std::string& model_dir);
    std::vector<float> extractFeature(const cv::Mat& face);
    bool saveFaceFeature(const std::string& id, const std::vector<float>& feature);
    float calculateSimilarity(const std::vector<float>& f1, const std::vector<float>& f2);
};

#endif // FACE_SYSTEM_H
