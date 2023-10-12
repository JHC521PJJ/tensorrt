/*
 * @Author: JHC521PJJ 
 * @Date: 2023-08-15 12:26:21 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-08-15 12:28:31
 */

#include "inference.cuh"
#include "imagePreprocess.cuh"
#include "npyToVector.h"
#include "resultTransformate.cuh"
#include <iostream>
#include <algorithm>
#include <omp.h>


Inference::Inference(): 
    m_batch_size(1), m_input_channel(3), m_output_channel(384),
    m_input_size(256), m_output_size(56) {
    init();
}

Inference::Inference(int batch_size, int input_channel, int output_channel, int input_size, int output_size)
    : m_batch_size(batch_size), m_input_channel(input_channel), m_output_channel(output_channel),
      m_input_size(input_size), m_output_size(output_size) {
    init();
}

// Release the memory on the device
Inference::~Inference() {
    cudaFree(m_d_teacher_mean); 
    cudaFree(m_d_teacher_std);
    cudaFree(m_d_preprocess_output);
    cudaFree(m_d_tinfer_output);
    cudaFree(m_d_sinfer_output);
    cudaFree(m_d_aeinfer_output);
    cudaFree(m_d_map_st);
    cudaFree(m_d_map_ae);
    cudaFree(m_d_combine);
}

// Initialize member variables, read the onnx model, and allocate memory on the device 
void Inference::init() noexcept {
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
    cudaMalloc((void**)&m_d_teacher_mean, sizeof(float) * m_output_channel);
    cudaMalloc((void**)&m_d_teacher_std, sizeof(float) * m_output_channel);
    cudaMemcpy(m_d_teacher_mean, vec_teacher_mean.data(), sizeof(float) * m_output_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_teacher_std, vec_teacher_std.data(), sizeof(float) * m_output_channel, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&m_d_preprocess_output, m_input_size * m_input_size * m_input_channel * sizeof(float));
    cudaMalloc((void**)&m_d_tinfer_output, m_output_channel * m_output_size * m_output_size * sizeof(float));
    cudaMalloc((void**)&m_d_sinfer_output, m_output_channel * 2 * m_output_size * m_output_size * sizeof(float));
    cudaMalloc((void**)&m_d_aeinfer_output, m_output_channel * m_output_size * m_output_size * sizeof(float));
    cudaMalloc((void**)&m_d_map_st, m_output_channel * m_output_size * m_output_size * sizeof(float));
    cudaMalloc((void**)&m_d_map_ae, m_output_channel * m_output_size * m_output_size * sizeof(float));
    cudaMalloc((void**)&m_d_combine, m_output_size * m_output_size * sizeof(float));
    
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

// Image preprocessing on the GPU
void Inference::processInput(cv::Mat& image) noexcept {
    imagePreprocessGpu(image, m_d_preprocess_output);
}

// TRT inference on the GPU
void Inference::trtInfer() noexcept {
    m_teacher_infer.infer(m_d_preprocess_output, m_d_tinfer_output);
    m_student_infer.infer(m_d_preprocess_output, m_d_sinfer_output);
    m_ae_infer.infer(m_d_preprocess_output, m_d_aeinfer_output);
}

// Asynchronous infering by openMP
void Inference::trtInferAsyn() noexcept {
    omp_set_num_threads(3);
    #pragma omp sections
    {
        #pragma omp section 
        {
            m_teacher_infer.infer(m_d_preprocess_output, m_d_tinfer_output);
        }
        #pragma omp section 
        {
            m_student_infer.infer(m_d_preprocess_output, m_d_sinfer_output);
        }
        #pragma omp section 
        {
            m_ae_infer.infer(m_d_preprocess_output, m_d_aeinfer_output);
        }
    }
}

// Processing the output on the GPU
void Inference::processOutput() noexcept {
    float max_element = 0.0f;
    resultTransformateGpu(
        m_d_tinfer_output, m_d_sinfer_output, m_d_aeinfer_output, 
        m_d_teacher_mean, m_d_teacher_std,
        m_d_map_st, 
        m_d_map_ae, 
        m_d_combine, 
        m_d_st_start_quantiles,
        m_d_st_end_quantiles,
        m_d_ae_start_quantiles,
        m_d_ae_end_quantiles,
        max_element);
    
    sample::gLogInfo << "Anomaly score: " << max_element << " ";
    if(max_element > 1.0) {
        sample::gLogInfo << "[Defect]" << std::endl;
    }
    else {
        sample::gLogInfo << "[Normal]" << std::endl;
    }
}

// Inference
void Inference::infer(cv::Mat& image) noexcept {
    processInput(image);
    // trtInfer();
    trtInferAsyn();
    processOutput();
}


