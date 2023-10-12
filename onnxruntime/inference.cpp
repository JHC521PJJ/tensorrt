/*
 * @Author: JHC521PJJ 
 * @Date: 2023-08-15 12:27:11 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-08-15 12:34:25
 */

#include "inference.h"
#include "imagePreprocess.h"
#include "npyToVector.h"
#include "resultTransformate.h"
#include "onnxLog.h"
#include <iostream>
#include <algorithm>


Inference::Inference(): m_batch_size(1), m_input_channel(3), m_output_channel(384),
    m_input_size(256), m_output_size(56) {
    init();
}

Inference::Inference(int batch_size, int input_channel, int output_channel, int input_size, int output_size)
    : m_batch_size(batch_size), m_input_channel(input_channel), m_output_channel(output_channel),
      m_input_size(input_size), m_output_size(output_size) {
    init();
}

Inference::~Inference() {}

// Initialize member variables, read the onnx model, and allocate memory. 
void Inference::init() {
    const std::string t_model_path = "/home/pjj/pythoncode/EfficientAD-main/onnx/chip4/teacher_final.onnx";
    const std::string s_model_path = "/home/pjj/pythoncode/EfficientAD-main/onnx/chip4/student_final.onnx";
    const std::string ae_model_path = "/home/pjj/pythoncode/EfficientAD-main/onnx/chip4/autoencoder_final.onnx";

    m_teacher_infer.setSessionCUDA(8);
    m_student_infer.setSessionCUDA(8);
    m_ae_infer.setSessionCUDA(8);
    m_teacher_infer.loadModel(t_model_path);
    m_student_infer.loadModel(s_model_path);
    m_ae_infer.loadModel(ae_model_path);

    m_teacher_mean = npyToVector("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip4/t_mean_quantiles.npy");
    m_teacher_std = npyToVector("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip4/t_std_quantiles.npy");
    m_q_st_start_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip4/q_st_start_quantiles.npy");
    m_q_st_end_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip4/q_st_end_quantiles.npy");
    m_q_ae_start_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip4/q_ae_start_quantiles.npy");
    m_q_ae_end_quantiles = npyToValue("/home/pjj/pythoncode/EfficientAD-main/output/4/trainings/mvtec_loco/chip4/q_ae_end_quantiles.npy");
}

// Image preprocessing on the CPU
void Inference::processInput(cv::Mat& image) {
    m_preprocess_output = imagePreprocessing(image);
}

// TRT inference on the GPU
void Inference::otrInfer() {
    m_tinfer_output = m_teacher_infer.infer(m_preprocess_output);
    m_sinfer_output = m_student_infer.infer(m_preprocess_output);
    m_aeinfer_output = m_ae_infer.infer(m_preprocess_output);
}

// Processing the output on the CPU
void Inference::processOutput() {
    for(int c = 0; c < 384; ++c) {
        for(int i = 0; i < 56 * 56; ++i) {
            m_tinfer_output[i + c * 56 * 56] = (m_tinfer_output[i + c * 56 * 56] - m_teacher_mean[c]) / m_teacher_std[c]; 
        }
    }
    
    auto vec_mean = meanOperation(m_tinfer_output, m_sinfer_output, m_aeinfer_output);
    std::vector<float> vec_mean_st = vec_mean[0];
    std::vector<float> vec_mean_ae = vec_mean[1];
    std::vector<float> vec_combined = combineOperation(vec_mean_st, vec_mean_ae, m_q_st_start_quantiles, m_q_st_end_quantiles, m_q_ae_start_quantiles, m_q_ae_end_quantiles);

    auto it_ad_score = std::max_element(vec_combined.begin(), vec_combined.end());
    ocr::log_info << "Score: " << *it_ad_score << " ";
    if(*it_ad_score > 1.0f) {
        ocr::log_info << "[Defect]" << std::endl;
    }
    else {
        ocr::log_info << "[Normal]" << std::endl;
    }
}

// Inference
void Inference::infer(cv::Mat& image) {
    processInput(image);
    otrInfer();
    processOutput();
}


