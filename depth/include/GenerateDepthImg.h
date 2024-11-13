#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

struct Img
{
    uint8_t *data;
    int32_t width;
    int32_t height;
    int32_t channel;
};

class GenerateDepthImg
{
public:
    GenerateDepthImg(int32_t width, int32_t height, int32_t channel);
    ~GenerateDepthImg();

    void Init();
    void Execute(uint8_t *inputData, uint8_t *outputData, uint8_t *depthData);
    void DestroyResource();

private:
    int32_t PreProcess(uint8_t *inputData, uint8_t *outputData);
    int32_t MainProcess();
    int32_t PostProcess(uint8_t *outputData);

    void getAlignSize();

    torch::jit::script::Module m_module;
    torch::DeviceType m_deviceType = torch::kCUDA;
    int m_gpuID = 0;

    std::vector<torch::jit::IValue> m_inputs;
    torch::jit::IValue m_outputs_;
    torch::Tensor m_outputs;

    Img m_alignImg;

    Img m_srcImg;
};