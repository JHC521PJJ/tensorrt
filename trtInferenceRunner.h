/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-24 21:59:01 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-07-30 21:08:05
 * 
 * https://github.com/JHC521PJJ/tensorrt
 * 
 * This header file is an implementation of the TensorRT inference class, 
 * It draws on examples from official reference documents
 */


#ifndef __TRT_RUNNER_H__
#define __TRT_RUNNER_H__

// Define TRT entrypoints used in common code
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
#include <vector>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

inline const char* IN_NAME = "input"; 
inline char* OUT_NAME = "output"; 
inline constexpr int BATCH_SIZE = 1; 

class TrtInferenceRunner {
private:
    nvinfer1::Dims m_input_dims;                        // The dimensions of the input to the network
    nvinfer1::Dims m_output_dims;                       // The dimensions of the output to the network. 
    std::shared_ptr<nvinfer1::IRuntime> m_runtime;      // The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;    // The TensorRT engine used to run the network

    const char* m_onnx_name;                // The name of onnx model
    std::vector<std::string> m_onnx_dirs;   // The path of onnx model

private:
    // Parses an ONNX model and creates a TensorRT network
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

public:
    TrtInferenceRunner(): m_runtime(nullptr), m_engine(nullptr), m_onnx_dirs{} {}
    ~TrtInferenceRunner() {}
    TrtInferenceRunner(const TrtInferenceRunner& other) = delete;
    TrtInferenceRunner(TrtInferenceRunner&& other) = delete;
    TrtInferenceRunner& operator=(const TrtInferenceRunner& other) = delete;
    TrtInferenceRunner& operator=(TrtInferenceRunner&& other) = delete;
    
    // Import the onnx model from disk
    void loadOnnxModel(const char* onnx_name, const char* onnx_dirs);
    // Function builds the network engine
    bool build();
    // Runs the TensorRT inference engine by a synchronous manner
    bool infer(float* d_preprocess_output, float* d_output);
    bool infer_v2(float* d_preprocess_output, float* d_output);
};

void TrtInferenceRunner::loadOnnxModel(const char* onnx_name, const char* onnx_dirs) {
    m_onnx_name = onnx_name;
    m_onnx_dirs.push_back(onnx_dirs);
}

// This function creates the Onnx network by parsing the Onnx model and builds the engine 
// return true if the engine was created successfully and false otherwise
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

    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!m_runtime) { return false;}

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!m_engine) { return false;}

    m_input_dims = network->getInput(0)->getDimensions();
    m_output_dims = network->getOutput(0)->getDimensions();
    // Print the dimensions of the input and output
    // sample::gLogInfo << "Input Dims: " << m_input_dims.d[0]<<" "<< m_input_dims.d[1]<<" " << m_input_dims.d[2]<<" " << m_input_dims.d[3] << std::endl;
    // sample::gLogInfo << "Output Dims: " << m_output_dims.d[0]<<" " << m_output_dims.d[1]<<" " << m_output_dims.d[2]<<" " << m_output_dims.d[3] << std::endl;

    return true;
}

// Uses a ONNX parser to create the Onnx Network and marks the output layers
// return true if the network was created successfully and false otherwise
bool TrtInferenceRunner::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser) {

    auto parsed = parser->parseFromFile(locateFile(m_onnx_name, m_onnx_dirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) { return false;}

    samplesCommon::enableDLA(builder.get(), config.get(), -1);
    
    return true;
}

// Allocates the buffer, sets inputs and executes the engine
// return true if infering successfully and false otherwise
bool TrtInferenceRunner::infer(float* d_preprocess_output, float* d_output) {
    void* buffers[2]; 
    const int32_t inputIndex = m_engine->getBindingIndex(IN_NAME); 
    const int32_t outputIndex = m_engine->getBindingIndex(OUT_NAME); 

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!context) { return false;}

    // Memcpy from device input array to device input buffers
    cudaMalloc(&buffers[inputIndex], m_input_dims.d[0] * m_input_dims.d[1] * m_input_dims.d[2] * m_input_dims.d[3] * sizeof(float)); 
    cudaMalloc(&buffers[outputIndex], m_output_dims.d[0] * m_output_dims.d[1] * m_output_dims.d[2] * m_output_dims.d[3] * sizeof(float)); 
    cudaMemcpy(buffers[inputIndex], d_preprocess_output, m_input_dims.d[0] * m_input_dims.d[1] * m_input_dims.d[2] * m_input_dims.d[3] * sizeof(float), cudaMemcpyDeviceToDevice); 
    bool status = context->executeV2(buffers); 
    if (!status) { return false;}
    cudaMemcpy(d_output, buffers[outputIndex], m_output_dims.d[0] * m_output_dims.d[1] * m_output_dims.d[2] * m_output_dims.d[3] * sizeof(float), cudaMemcpyDeviceToDevice);  

    return true;
}

#endif


