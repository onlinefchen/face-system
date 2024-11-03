#include "face_detector.h"
#include <algorithm>

FaceDetector::FaceDetector() {}

FaceDetector::~FaceDetector() {
    net.clear();
}

bool FaceDetector::loadModel(const std::string& param_path, const std::string& bin_path) {
    return net.load_param(param_path.c_str()) == 0 && 
           net.load_model(bin_path.c_str()) == 0;
}

void FaceDetector::preprocess(const cv::Mat& img, ncnn::Mat& in) {
    in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    in.substract_mean_normalize(mean_vals, norm_vals);
}

std::vector<FaceInfo> FaceDetector::detect(const cv::Mat& img, float threshold) {
    ncnn::Mat in;
    preprocess(img, in);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ex.input("input.1", in);

    std::vector<ncnn::Mat> outs;
    ncnn::Mat score_blob, bbox_blob, landmark_blob;
    ex.extract("score_8", score_blob);
    ex.extract("bbox_8", bbox_blob);
    ex.extract("landmark_8", landmark_blob);

    outs.push_back(score_blob);
    outs.push_back(bbox_blob);
    outs.push_back(landmark_blob);

    return postprocess(outs, threshold);
}

cv::Mat FaceDetector::align(const cv::Mat& img, const FaceInfo& face) {
    // 简单实现：直接裁剪和缩放到112x112
    cv::Mat aligned;
    cv::resize(img(face.rect), aligned, cv::Size(112, 112));
    return aligned;
}

std::vector<FaceInfo> FaceDetector::postprocess(const std::vector<ncnn::Mat>& outs, float threshold) {
    std::vector<FaceInfo> faces;
    const ncnn::Mat& score_blob = outs[0];
    const ncnn::Mat& bbox_blob = outs[1];
    const ncnn::Mat& landmark_blob = outs[2];

    for (int i = 0; i < score_blob.h; i++) {
        float score = score_blob[i];
        if (score < threshold)
            continue;

        // 解析bbox
        const float* bbox = bbox_blob.row(i);
        FaceInfo face;
        face.rect.x = bbox[0];
        face.rect.y = bbox[1];
        face.rect.width = bbox[2] - bbox[0];
        face.rect.height = bbox[3] - bbox[1];
        face.score = score;

        // 解析关键点
        const float* landmark = landmark_blob.row(i);
        for (int j = 0; j < 5; j++) {
            face.landmarks.push_back(cv::Point2f(landmark[j*2], landmark[j*2+1]));
        }

        faces.push_back(face);
    }

    return faces;
}
