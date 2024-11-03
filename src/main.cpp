#include "face_system.h"
#include <iostream>
#include <chrono>

void printUsage(const char* progName) {
    std::cout << "Usage:\n"
              << progName << " register <image_path> <person_id>\n"
              << progName << " recognize <image_path>\n"
              << progName << " clear\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    FaceSystem system;
    if (!system.init("models")) {
        std::cerr << "Failed to initialize face system\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "register" && argc == 4) {
        cv::Mat img = cv::imread(argv[2]);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << argv[2] << std::endl;
            return 1;
        }

        auto start = std::chrono::steady_clock::now();
        if (system.registerFace(img, argv[3])) {
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Successfully registered face for " << argv[3] 
                      << " (time: " << duration.count() << "ms)" << std::endl;
        } else {
            std::cerr << "Failed to register face\n";
            return 1;
        }
    }
    else if (mode == "recognize" && argc == 3) {
        cv::Mat img = cv::imread(argv[2]);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << argv[2] << std::endl;
            return 1;
        }

        auto start = std::chrono::steady_clock::now();
        auto result = system.recognize(img);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Recognized as: " << result.id 
                  << " (confidence: " << result.confidence 
                  << ", time: " << duration.count() << "ms)" << std::endl;
    }
    else if (mode == "clear") {
        if (system.clear()) {
            std::cout << "Successfully cleared all registered faces\n";
        } else {
            std::cerr << "Failed to clear database\n";
            return 1;
        }
    }
    else {
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
