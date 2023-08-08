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

static std::string log_id{};
Ort::Env OnnxInferenceRunner::m_env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, log_id.c_str());

OnnxInferenceRunner::OnnxInferenceRunner() {
    m_session_options.SetIntraOpNumThreads(4); 
    m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

OnnxInferenceRunner::~OnnxInferenceRunner(){}

void OnnxInferenceRunner::loadModel(const std::string& model_path) {
    m_session = std::make_shared<Ort::Session>(m_env, model_path.c_str(), m_session_options);

    if(model_path == "") {
        ocr::log_error << "Onnx model path is error" << std::endl;
    }
    else if (m_env == nullptr) {
        ocr::log_error << "Onnx env not initialized" << std::endl;
    }
    else if (m_session == nullptr) {
        ocr::log_error << "Onnx model not initialized" << std::endl;
    }
    else {
        ocr::log_info << "Onnx load model success" << std::endl;
    }
}

void OnnxInferenceRunner::setSessionNumThreads(const int num) {
    m_session_options.SetIntraOpNumThreads(1);
}

void OnnxInferenceRunner::setSessionCUDA(const int device_id) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(m_session_options, device_id);
    ocr::log_info << "Onnx model has loaded in cuda: " << device_id << std::endl;
}

size_t OnnxInferenceRunner::getSessionInputCount() {
    return m_session->GetInputCount();
}

size_t OnnxInferenceRunner::getSessionOutputCount() {
    return m_session->GetOutputCount();
}

VecInt64 OnnxInferenceRunner::getSessionInputNodeDims(size_t index) {
    return m_session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

VecInt64 OnnxInferenceRunner::getSessionOutputNodeDims(size_t index) {
    return m_session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

const char* OnnxInferenceRunner::getSessionInputName(size_t index) {
    return m_session->GetInputName(index, m_allocator);
}

const char* OnnxInferenceRunner::getSessionOutputName(size_t index) {
    return m_session->GetOutputName(index, m_allocator);
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
    
    auto onnx_output = m_session->Run(Ort::RunOptions{ nullptr }, input_name.data(), &onnx_input, input_name.size(),output_name.data(), output_name.size());
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

