#include <iostream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <unistd.h>
#include <GenerateDepthImg.h>

using namespace std;

int main() {
    const char* imagePath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/demo01.jpg";
    cv::Mat img = cv::imread(imagePath);

    cv::resize(img, img, cv::Size(1932, 1092));
    int imgH = img.rows;
    int imgW = img.cols;
    int imgC = img.channels();

    uint8_t *depthData = new uint8_t[1080 * 1920];
    uint8_t *outputData = new uint8_t[1080 * 1920 * 3];

    GenerateDepthImg* depth = new GenerateDepthImg(1920, 1080, 3);

    auto time1 = std::chrono::high_resolution_clock::now();
    for (int i =0; i<10;++i) {
        depth->Execute(img.data, outputData, depthData);
    }

    auto time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timeCost1 = time2-time1;
    std::cout << "run time: " << timeCost1.count() << " ms\n";

    delete depthData;
    delete outputData;
    delete depth;

    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl; 
    return 0;
}