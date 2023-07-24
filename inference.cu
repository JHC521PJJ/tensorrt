#include "inference.cuh"
#include "imagePreprocess.cuh"
#include "time_count.h"
#include "npyToVector.h"
#include "resultTransformate.cuh"
#include <iostream>
#include <algorithm>



Inference::Inference(): m_batch_size(1), m_channel(384), m_out_size(56) {
    init();
}

void Inference::init() {
    std::vector<float> vec_teacher_mean = npyToVector("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/t_mean_quantiles.npy");
    std::vector<float> vec_teacher_std = npyToVector("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/t_std_quantiles.npy");
    float q_st_start_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_st_start_quantiles.npy");
    float q_st_end_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_st_end_quantiles.npy");
    float q_ae_start_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_ae_start_quantiles.npy");
    float q_ae_end_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_ae_end_quantiles.npy");

    m_d_st_start_quantiles = q_st_start_quantiles;
    m_d_st_end_quantiles = q_st_end_quantiles;
    m_d_ae_start_quantiles = q_ae_start_quantiles;
    m_d_ae_end_quantiles = q_ae_end_quantiles;
    cudaMalloc((void **) &m_d_teacher_mean, sizeof(float) * m_channel);
    cudaMalloc((void **) &m_d_teacher_std, sizeof(float) * m_channel);
    cudaMemcpy(m_d_teacher_mean, vec_teacher_mean.data(), sizeof(float) * m_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_teacher_std, vec_teacher_std.data(), sizeof(float) * m_channel, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&m_d_preprocess_output, 256 * 256 * 3 * sizeof(float));
    cudaMalloc((void**)&m_t_infer_output, 384 * 56 * 56 * sizeof(float));
    cudaMalloc((void**)&m_s_infer_output, 768 * 56 * 56 * sizeof(float));
    cudaMalloc((void**)&m_ae_infer_output, 384 * 56 * 56 * sizeof(float));

    const char* t_onnx = "teacher_final_simp.onnx";
    const char* s_onnx = "student_final_simp.onnx";
    const char* ae_onnx = "autoencoder_final_simp.onnx";
    const char* onnx_dirs = "/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/onnx/chip6/";
    m_teacher_infer.loadOnnxModel(t_onnx, onnx_dirs);
    m_student_infer.loadOnnxModel(s_onnx, onnx_dirs);
    m_ae_infer.loadOnnxModel(ae_onnx, onnx_dirs);
    
    m_teacher_infer.build();
    m_student_infer.build();
    m_ae_infer.build();
}

void Inference::processInput(cv::Mat& image) {
    imagePreprocessingGpu(image, m_d_preprocess_output);
}

void Inference::trtInference() {
    m_teacher_infer.infer_v2(m_d_preprocess_output, m_t_infer_output);
    m_student_infer.infer_v2(m_d_preprocess_output, m_s_infer_output);
    m_ae_infer.infer_v2(m_d_preprocess_output, m_ae_infer_output);
}

void Inference::processOutput() {
    std::vector<float> vec_combined(56 * 56);
    resultTransformate_v2(
        m_t_infer_output, m_s_infer_output, m_ae_infer_output, 
        m_d_teacher_mean, m_d_teacher_std,
        m_d_st_start_quantiles,
        m_d_st_end_quantiles,
        m_d_ae_start_quantiles,
        m_d_ae_end_quantiles,
        0,
        vec_combined);
    auto it_ad_score = std::max_element(vec_combined.begin(), vec_combined.end());
    std::cout<< "Score: " << *it_ad_score << std::endl;
}

void Inference::infer(cv::Mat& image) {
    processInput(image);
    trtInference();
    processOutput();
}


