#include <GenerateDepthImg.h>

GenerateDepthImg::GenerateDepthImg(int32_t width, int32_t height, int32_t channel)
{
    m_deviceType = torch::kCUDA;
    m_srcImg.width = width;
    m_srcImg.height = height;
    m_srcImg.channel = channel;
    m_gpuID = 2;
    Init();
}

GenerateDepthImg::~GenerateDepthImg()
{
    DestroyResource();
}

void GenerateDepthImg::Init()
{
    const char* modelPath;
    switch (m_srcImg.height)
    {
        case 540:
            m_alignImg.height = 546;
            m_alignImg.width = 966;
            modelPath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/vits_540P.pt";
            break;
        case 720:
            m_alignImg.height = 728;
            m_alignImg.width = 1288;
            modelPath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/vits_720P.pt";
            break;
        case 1080:
            m_alignImg.height = 1092;
            m_alignImg.width = 1932;
            modelPath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/vits_1080P.pt";
            break;
        case 1440:
            m_alignImg.height = 1442;
            m_alignImg.width = 2562;
            modelPath = "/home/zhike/公共的/x30032372/Depth-Anything-V2/vits_2K.pt";
            break;
        default:
            printf("error image\n");
            break;
    }


    //m_device = device;
    torch::Device device(m_deviceType, m_gpuID);
    m_module = torch::jit::load(modelPath, device);
    m_alignImg.channel = 3;
    //m_alignImg.data = new uint8_t[m_alignImg.height * m_alignImg.width * m_alignImg.channel];
    printf("torch ok\n");
}

int32_t GenerateDepthImg::PreProcess(uint8_t *inputData, uint8_t *outputData)
{
    //cv::Mat srcImg(m_srcImg.height, m_srcImg.width, CV_8UC3, inputData);
    int imageType;
    if(m_srcImg.channel == 3) {
        imageType = CV_8UC3;
    } else if(m_srcImg.channel == 4) {
        imageType = CV_8UC4;
    }
    // get alignImg
    cv::Mat srcImg(m_srcImg.height, m_srcImg.width, imageType, inputData);
    if(m_srcImg.channel == 4) {
        cv::cvtColor(srcImg, srcImg, cv::COLOR_RGBA2RGB);
    }
    memcpy(outputData, srcImg.data, m_srcImg.width*m_srcImg.height*3);
    cv::resize(srcImg, srcImg, cv::Size(m_alignImg.width, m_alignImg.height));
    srcImg.convertTo(srcImg, CV_32F, 1.0f / 255.0f); // 归一化

    // get model inputs
    auto imageTensor = torch::from_blob(srcImg.data, {1, m_alignImg.height, m_alignImg.width, m_alignImg.channel});

    // 转NCHW
    torch::Device device(m_deviceType, m_gpuID);
    imageTensor = imageTensor.permute({0,3,1,2});
    m_inputs.push_back(imageTensor.to(device));
    return 0;
}

int32_t GenerateDepthImg::MainProcess()
{
    torch::NoGradGuard no_grad;
    m_outputs_ = m_module.forward(m_inputs);
    m_outputs = m_outputs_.toTensor();
    m_inputs.pop_back();
    return 0;
}

int32_t GenerateDepthImg::PostProcess(uint8_t *depthData)
{
    m_outputs = m_outputs[0].detach();
    m_outputs = (m_outputs - m_outputs.min()) / (m_outputs.max() - m_outputs.min());
    m_outputs = m_outputs.to(torch::kCPU);
    int height = m_outputs.size(0);
    int width = m_outputs.size(1);

    cv::Mat resultImg(height, width , CV_32F, m_outputs.data_ptr());
    resultImg.convertTo(resultImg, CV_8UC1, 255); // 反归一化
    cv::resize(resultImg,resultImg,cv::Size(m_srcImg.width,m_srcImg.height));
    memcpy(depthData, resultImg.data, m_srcImg.width * m_srcImg.height);

    return 0;
}

void GenerateDepthImg::Execute(uint8_t *inputData, uint8_t *outputData, uint8_t *depthData)
{

    if (inputData == nullptr || outputData == nullptr || depthData == nullptr) {
        printf("address is nullptr!\n");
        return;
    }

    auto time1 = std::chrono::high_resolution_clock::now();
    PreProcess(inputData, outputData);
    auto time2 = std::chrono::high_resolution_clock::now();
    MainProcess();
    auto time3 = std::chrono::high_resolution_clock::now();
    PostProcess(depthData);
    auto time4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timeCost1 = time2-time1;
    std::chrono::duration<double, std::milli> timeCost2 = time3-time2;
    std::chrono::duration<double, std::milli> timeCost3 = time4-time3;
    std::cout << "pre time: " << timeCost1.count() << " ms\n";
    std::cout << "main time: " << timeCost2.count() << " ms\n";
    std::cout << "post time: " << timeCost3.count() << " ms\n";
}

void GenerateDepthImg::DestroyResource()
{
    //delete m_alignImg.data;
    //m_alignImg.data = nullptr;
}