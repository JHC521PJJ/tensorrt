/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-31 21:44:11 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-07-31 21:44:11 
 * 
 * https://github.com/JHC521PJJ/tensorrt
 */

#ifndef __ONNXRUN_H__
#define __ONNXRUN_H__

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

class OnnxInferenceRunner {
public:
    using VecFloat = std::vector<float>;
    using VecInt64 = std::vector<int64_t>;

private:
    static Ort::Env m_env;
    std::shared_ptr<Ort::Session> m_session;
    Ort::SessionOptions m_session_options;
    Ort::AllocatorWithDefaultOptions m_allocator;

private:
    size_t getSessionInputCount();
    size_t getSessionOutputCount();
    VecInt64 getSessionInputNodeDims(size_t index);
    VecInt64 getSessionOutputNodeDims(size_t index);
    const char* getSessionInputName(size_t index);
    const char* getSessionOutputName(size_t index);
    
public:
    OnnxInferenceRunner();
    ~OnnxInferenceRunner();
    OnnxInferenceRunner(const OnnxInferenceRunner& other) = delete;
    OnnxInferenceRunner(OnnxInferenceRunner&& other) = delete;
    OnnxInferenceRunner& operator=(const OnnxInferenceRunner& other) = delete;
    OnnxInferenceRunner& operator=(OnnxInferenceRunner&& other) = delete;

    void loadModel(const std::string& model_path);
    void setSessionNumThreads(const int num);
    void setSessionCUDA(const int device_id);
    void printModelInfo();
    VecFloat infer(VecFloat& input_vector);
};

#endif