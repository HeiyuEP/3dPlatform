#include <iostream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

int test() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_test");
    Ort::SessionOptions session_options;

    OrtCUDAProviderOptions cuda_options;

    session_options.AppendExecutionProvider_CUDA(cuda_options);
    
    const char* modelPath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/vits.onnx";
    Ort::Session session(env,modelPath,session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    //2、read pic
    const std::string imagePath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/re.png";
    cv::Mat image = cv::imread(imagePath);
    int height = 1092;
    int width = 1932;
    cv::resize(image,image,cv::Size(width,height));
    image.convertTo(image, CV_32F, 1.0/255.0);
    size_t inputTensorSize = 1 * 3 * width * height;
    std::vector<float> inputTensorValue(inputTensorSize);
    for (int h = 0;h<image.rows;++h) {
        for (int w = 0;w<image.cols;++w) {
            for (int c = 0;c<image.channels();++c) {
                inputTensorValue[0*3*width*height + c*width*height + h*width + w] = image.at<cv::Vec3f>(h,w)[c];
            }
        }
    }

    size_t numInputNodes = session.GetInputCount();
    // auto inputName = session.GetInput();
    // std::cout << "Input names: " << std::endl;
    // for (auto& input_name : input_names) {
    //     std::cout << input_name.Name() << std::endl;
    // }
    std::vector<const char*> inputName = {"x.1"};
    std::vector<const char*> outputName = {"1271"};
    std::vector<int64_t> inputNodeDims = {1,3,1092,1932};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,inputTensorValue.data(),
                                                            inputTensorSize,inputNodeDims.data(),4);
    std::vector<Ort::Value> ortInputs;
    ortInputs.push_back(std::move(inputTensor));

    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputName.data(), ortInputs.data(), ortInputs.size(), outputName.data(), 1);
    float* floatarr = outputTensors[0].GetTensorMutableData<float>();
    std::vector<float> depthVector(floatarr, floatarr+width*height);

    float minValue =  *std::min_element(depthVector.begin(),depthVector.end());
    float maxValue =  *std::max_element(depthVector.begin(),depthVector.end());

    for (int i = 0; i<width*height;++i) {
        depthVector[i] = (depthVector[i] - minValue) / (maxValue - minValue);
    }

    cv::Mat outputImage(height,width,CV_32F,depthVector.data());
    outputImage.convertTo(outputImage,CV_8UC1,255);
    cv::imwrite("/home/zhike/公共的/x30032372/Depth-Anything-V2/c.png",outputImage);
    return 0;
}

int main() {
    test();
    return 0;
}