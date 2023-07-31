#include "onnxLog.h"
#include "onnxInferenceRun.h"
#include <iostream>
#include <numeric>
#include <cuda_provider_factory.h>


using VecFloat = std::vector<float>;
using VecInt64 = std::vector<int64_t>;

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for(size_t i = 0; i < vec.size(); ++i) {
        os << vec[i] << " ";
    }
    os << "]";
    return os;
}

static std::string log_id_{};
Ort::Env OnnxInferenceRunner::env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, log_id_.c_str());

OnnxInferenceRunner::OnnxInferenceRunner() {
    session_options_.SetIntraOpNumThreads(4); 
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

OnnxInferenceRunner::~OnnxInferenceRunner(){}

void OnnxInferenceRunner::loadModel(const std::string& model_path) {
    session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), session_options_);

    if(model_path == "") {
        ocr::log_error << "Onnx model path is error" << std::endl;
    }
    else if (env_ == nullptr) {
        ocr::log_error << "Onnx env not initialized" << std::endl;
    }
    else if (session_ == nullptr) {
        ocr::log_error << "Onnx model not initialized" << std::endl;
    }
    else {
        ocr::log_info << "Onnx load model success" << std::endl;
    }
}

void OnnxInferenceRunner::setSessionNumThreads(const int num) {
    session_options_.SetIntraOpNumThreads(1);
}

void OnnxInferenceRunner::setSessionCUDA(const int device_id) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, device_id);
    ocr::log_info << "Onnx model has loaded in cuda: " << device_id << std::endl;
}

size_t OnnxInferenceRunner::getSessionInputCount() {
    return session_->GetInputCount();
}

size_t OnnxInferenceRunner::getSessionOutputCount() {
    return session_->GetOutputCount();
}

VecInt64 OnnxInferenceRunner::getSessionInputNodeDims(size_t index) {
    return session_->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

VecInt64 OnnxInferenceRunner::getSessionOutputNodeDims(size_t index) {
    return session_->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

const char* OnnxInferenceRunner::getSessionInputName(size_t index) {
    return session_->GetInputName(index, allocator_);
}

const char* OnnxInferenceRunner::getSessionOutputName(size_t index) {
    return session_->GetOutputName(index, allocator_);
}

void OnnxInferenceRunner::printModelInfo() {
    ocr::log_info << "OnnxInferenceRunner with parameters:" << std::endl;
    ocr::log_info << "Number of Input Nodes: " << getSessionInputCount() << std::endl;
    for (size_t i = 0; i < getSessionInputCount(); ++i) {
        ocr::log_info << "Input " << i << ": ";
        ocr::log_info << "Name = " << getSessionInputName(i) << ", ";
        ocr::log_info << "Shape = " << getSessionInputNodeDims(i) << std::endl;
    }

    ocr::log_info << "Number of Output Nodes: " << getSessionOutputCount() << std::endl;
    for (size_t i = 0; i < getSessionOutputCount(); ++i) {
        ocr::log_info << "Output " << i << ": ";
        ocr::log_info << "Name = " << getSessionOutputName(i) << ", ";
        ocr::log_info << "Shape = " << getSessionOutputNodeDims(i) << std::endl;
    }
}

VecFloat OnnxInferenceRunner::infer(VecFloat& input_vector) {
    std::vector<const char*> input_name = {getSessionInputName(0)};
    std::vector<const char*> output_name = {getSessionOutputName(0)};
    
    VecInt64 input_dim = getSessionInputNodeDims(0);
    VecInt64 output_dim = getSessionOutputNodeDims(0);
    
    size_t input_size_count = std::accumulate(input_dim.begin(), input_dim.end(), 1, std::multiplies<int64_t>());
    size_t output_size_count = std::accumulate(output_dim.begin(), output_dim.end(), 1, std::multiplies<int64_t>());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value onnx_input = Ort::Value::CreateTensor<float>(memory_info, input_vector.data(), input_size_count, input_dim.data(), input_dim.size());   
    
    auto onnx_output = session_->Run(Ort::RunOptions{ nullptr }, input_name.data(), &onnx_input, input_name.size(),output_name.data(), output_name.size());
    float* p_onnx_output = onnx_output[0].GetTensorMutableData<float>();

    if(p_onnx_output != nullptr) {
        ocr::log_info << "Onnx model inference successe" << std::endl;
        return VecFloat(p_onnx_output, p_onnx_output + output_size_count);
    }
    else {
        ocr::log_error << "Onnx model inference false" << std::endl;
        return {};
    }
}

