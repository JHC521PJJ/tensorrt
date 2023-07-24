/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-24 21:59:01 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-07-24 22:01:38
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * This header file is an implementation of the TensorRT inference class, 
 * It draws on examples from official reference documents
 */


#ifndef __TRT_RUNNER_H__
#define __TRT_RUNNER_H__

#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include "imagePreprocess.cuh"
#include "resultTransformate.cuh"
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

static const char* IN_NAME = "input"; 
static char* OUT_NAME = "output"; 
static constexpr int BATCH_SIZE = 1; 

class TrtInferenceRunner {
private:
    samplesCommon::OnnxSampleParams mParams; 
    nvinfer1::Dims mInputDims;  
    nvinfer1::Dims mOutputDims; 
    int mNumber{0};             

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; 

    const char* m_onnx_name; 
    std::vector<std::string> m_onnx_dirs;

private:
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

public:
    TrtInferenceRunner(): mRuntime(nullptr), mEngine(nullptr) {}
    TrtInferenceRunner(const TrtInferenceRunner& other) = delete;
    TrtInferenceRunner(TrtInferenceRunner&& other) = delete;
    TrtInferenceRunner& operator=(const TrtInferenceRunner& other) = delete;
    TrtInferenceRunner& operator=(TrtInferenceRunner&& other) = delete;
    ~TrtInferenceRunner() {}

    void loadOnnxModel(const char* onnx_name, const char* onnx_dirs);
    bool build();
    bool infer_v2(float* d_preprocess_output, float* d_output);
};

void TrtInferenceRunner::loadOnnxModel(const char* onnx_name, const char* onnx_dirs) {
    m_onnx_name = onnx_name;
    m_onnx_dirs.push_back(onnx_dirs);
}

bool TrtInferenceRunner::build() {
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) { return false;}

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {return false;}

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {return false;}

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) { return false;}

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) { return false; }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) { return false;}
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) { return false; }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime) { return false;}

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine) { return false;}

    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();
    sample::gLogInfo << "mInputDims: " << mInputDims.d[0]<<" "<< mInputDims.d[1]<<" " << mInputDims.d[2]<<" " << mInputDims.d[3] << std::endl;
    sample::gLogInfo << "mOutputDims: " << mOutputDims.d[0]<<" " << mOutputDims.d[1]<<" " << mOutputDims.d[2]<<" " << mOutputDims.d[3] << std::endl;

    return true;
}

bool TrtInferenceRunner::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser) {

    auto parsed = parser->parseFromFile(locateFile(m_onnx_name, m_onnx_dirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) { return false;}

    samplesCommon::enableDLA(builder.get(), config.get(), -1);
    
    return true;
}

bool TrtInferenceRunner::infer_v2(float* d_preprocess_output, float* d_output) {
    void* buffers[2]; 
    const int32_t inputIndex = mEngine->getBindingIndex(IN_NAME); 
    const int32_t outputIndex = mEngine->getBindingIndex(OUT_NAME); 

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    cudaMalloc(&buffers[inputIndex], mInputDims.d[0] * mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3] * sizeof(float)); 
    cudaMalloc(&buffers[outputIndex], mOutputDims.d[0] * mOutputDims.d[1] * mOutputDims.d[2] * mOutputDims.d[3] * sizeof(float)); 
    cudaMemcpy(buffers[inputIndex], d_preprocess_output, mInputDims.d[0] * mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3] * sizeof(float), cudaMemcpyDeviceToDevice); 
    context->executeV2(buffers); 
    cudaMemcpy(d_output, buffers[outputIndex], mOutputDims.d[0] * mOutputDims.d[1] * mOutputDims.d[2] * mOutputDims.d[3] * sizeof(float), cudaMemcpyDeviceToDevice);  

    return true;
}

#endif


