#include "face_system.h"
#include <cmath>

FaceSystem::FaceSystem() : db(nullptr) {}

FaceSystem::~FaceSystem() {
    if (db) {
        sqlite3_close(db);
    }
    recognizer.clear();
}

bool FaceSystem::init(const std::string& model_dir) {
    return initDatabase() && loadModels(model_dir);
}

bool FaceSystem::initDatabase() {
    int rc = sqlite3_open("faces.db", &db);
    if (rc) {
        return false;
    }

    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            feature BLOB NOT NULL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    )";
    char* errMsg = nullptr;
    rc = sqlite3_exec(db, sql, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errMsg);
        return false;
    }
    return true;
}

bool FaceSystem::loadModels(const std::string& model_dir) {
    std::string det_param = model_dir + "/scrfd_500m.param";
    std::string det_bin = model_dir + "/scrfd_500m.bin";
    std::string rec_param = model_dir + "/mobilefacenet.param";
    std::string rec_bin = model_dir + "/mobilefacenet.bin";

    if (!detector.loadModel(det_param, det_bin)) {
        return false;
    }

    if (recognizer.load_param(rec_param.c_str()) != 0 ||
        recognizer.load_model(rec_bin.c_str()) != 0) {
        return false;
    }

    return true;
}

bool FaceSystem::registerFace(const cv::Mat& img, const std::string& id) {
    std::vector<FaceInfo> faces = detector.detect(img);
    if (faces.empty()) {
        return false;
    }

    // 使用最大的人脸
    auto largest_face = std::max_element(faces.begin(), faces.end(),
        [](const FaceInfo& a, const FaceInfo& b) {
            return a.rect.area() < b.rect.area();
        });

    cv::Mat aligned = detector.align(img, *largest_face);
    std::vector<float> feature = extractFeature(aligned);

    return saveFaceFeature(id, feature);
}

std::vector<float> FaceSystem::extractFeature(const cv::Mat& face) {
    std::vector<float> feature;
    feature.resize(128);

    ncnn::Mat in = ncnn::Mat::from_pixels(face.data, ncnn::Mat::PIXEL_BGR, 112, 112);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/128.0f, 1/128.0f, 1/128.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = recognizer.create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);

    memcpy(feature.data(), out.data, out.total() * sizeof(float));
    return feature;
}

bool FaceSystem::saveFaceFeature(const std::string& id, const std::vector<float>& feature) {
    sqlite3_stmt* stmt;
    const char* sql = "INSERT OR REPLACE INTO faces (id, feature) VALUES (?, ?)";

    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_blob(stmt, 2, feature.data(), feature.size() * sizeof(float), SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_DONE;
}

FaceSystem::Recognition FaceSystem::recognize(const cv::Mat& img, float threshold) {
    std::vector<FaceInfo> faces = detector.detect(img);
    if (faces.empty()) {
        return {"unknown", 0.0f};
    }

    auto largest_face = std::max_element(faces.begin(), faces.end(),
        [](const FaceInfo& a, const FaceInfo& b) {
            return a.rect.area() < b.rect.area();
        });

    cv::Mat aligned = detector.align(img, *largest_face);
    std::vector<float> feature = extractFeature(aligned);

    Recognition best_match = {"unknown", 0.0f};
    sqlite3_stmt* stmt;
    const char* sql = "SELECT id, feature FROM faces";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return best_match;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string id = (const char*)sqlite3_column_text(stmt, 0);
        const float* db_feature = (const float*)sqlite3_column_blob(stmt, 1);
        int size = sqlite3_column_bytes(stmt, 1) / sizeof(float);

        std::vector<float> db_feature_vec(db_feature, db_feature + size);
        float similarity = calculateSimilarity(feature, db_feature_vec);

        if (similarity > threshold && similarity > best_match.confidence) {
            best_match.id = id;
            best_match.confidence = similarity;
        }
    }

    sqlite3_finalize(stmt);
    return best_match;
}

float FaceSystem::calculateSimilarity(const std::vector<float>& f1, const std::vector<float>& f2) {
    float dot = 0;
    float norm1 = 0;
    float norm2 = 0;

    for (size_t i = 0; i < f1.size(); i++) {
        dot += f1[i] * f2[i];
        norm1 += f1[i] * f1[i];
        norm2 += f2[i] * f2[i];
    }

    return dot / (sqrt(norm1) * sqrt(norm2));
}

bool FaceSystem::clear() {
    const char* sql = "DELETE FROM faces";
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errMsg);
        return false;
    }
    return true;
}
