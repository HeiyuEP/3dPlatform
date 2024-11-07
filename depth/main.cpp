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
/*
int test() {
    // 模型加载
    const char* modelPath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/vits_1080P.pt";
    torch::DeviceType deviceType;
    deviceType = torch::kCUDA;
    //deviceType = torch::kCPU;
    //int gpuID = 1;
    torch::Device device(deviceType);
    torch::jit::script::Module module = torch::jit::load(modelPath, deviceType);

    // 加载图片
    const char* imagePath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/demo01.jpg";
    cv::Mat image = cv::imread(imagePath);
    int imgH = image.rows;
    int imgW = image.cols;
    int imgC = image.channels();

    int h = 1092;
    int w = 1932;
    cv::resize(image, image, cv::Size(w,h));
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    auto imageTensor = torch::from_blob(image.data, {1, h, w, imgC}); //转成tensor

    // 转成NCHW，构建输入
    imageTensor = imageTensor.permute({0, 3, 1, 2});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(imageTensor.to(deviceType));
    for (int i =0;i<10;i++) {
        std::cout << "==> traceof it is:\n" << imageTensor[0][0][0][i] << std::endl;
    }
    // sleep(8000);
    // 执行推理，保存输出
    torch::Tensor out;
    auto time1 = std::chrono::high_resolution_clock::now();
    {
        torch::NoGradGuard no_grad;
        out = module.forward(inputs).toTensor();
    }
    auto time2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> timeCost1 = time2-time1;
    std::cout << "run time: " << timeCost1.count() << " ms\n";
    // 处理结果
    out = out[0].detach();
    out = (out - out.min()) / (out.max() - out.min());
    out = out.to(torch::kCPU);

    // 保存图片
    cv::Mat outputImage(h,w,CV_32F,out.data_ptr());
    outputImage.convertTo(outputImage,CV_8UC1,255);
    cv::imwrite("/home/zhike/公共的/x30032372/Depth-Anything-V2/1111.png",outputImage);

    return 0;
}
*/
int main() {
    //test();
    const char* imagePath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/demo01.jpg";
    cv::Mat img = cv::imread(imagePath);
    printf("1111\n");
    // cv::Mat image; // 创建一个4通道的空白图像
    // cv::cvtColor(img, image, cv::COLOR_RGB2RGBA); // 如果是RGB

    // // 在alpha通道填充255，表示不透明
    // cv::Mat alpha = cv::Mat::ones(img.size(), CV_8UC1) * 255;
    // cv::merge(std::vector<cv::Mat>{image, alpha}, image);
    // // 保存或处理转换后的4通道图像
    //cv::imwrite("/home/zhike/公共的/x30032372/Depth-Anything-V2/demo01-4C.jpg", image);

    cv::resize(img, img, cv::Size(1932, 1092));
    int imgH = img.rows;
    int imgW = img.cols;
    int imgC = img.channels();

    uint8_t *depthData = new uint8_t[1080 * 1920];
    uint8_t *outputData = new uint8_t[1080 * 1920 * 3];

    GenerateDepthImg* depth = new GenerateDepthImg(1920, 1080, 3);
    printf("2222\n");
    auto time1 = std::chrono::high_resolution_clock::now();
    for (int i =0; i<100;++i) {
        depth->Execute(img.data, outputData, depthData);
    }
    //depth->Execute(img.data, outputData, depthData);
    auto time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timeCost1 = time2-time1;
    std::cout << "run time: " << timeCost1.count() << " ms\n";

    delete depthData;
    delete outputData;
    delete depth;

    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl; 
    return 0;
}