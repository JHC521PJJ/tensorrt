/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-24 21:40:18 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-07-30 20:06:27
 * 
 * https://github.com/JHC521PJJ/tensorrt
 * 
 * This header file is an implementation of Infernece class
 */


#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include "trtInferenceRunner.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>


class Inference {
private:
    int m_batch_size;                   // The batch size of input image 
    int m_input_channel;                // The channels of input image
    int m_output_channel;               // The channels of output image
    int m_input_size;                   // The size of input image
    int m_output_size;                  // The size of output image
    
    TrtInferenceRunner m_teacher_infer; // The TRT inference class of teacher networks
    TrtInferenceRunner m_student_infer; // The TRT inference class of student network
    TrtInferenceRunner m_ae_infer;      // The TRT inference class of auto encorder network

    // Initializes temporary variables, allocates their memory on the device
    float* m_d_teacher_mean; 
    float* m_d_teacher_std;
    float* m_d_preprocess_output;
    float* m_d_tinfer_output;
    float* m_d_sinfer_output;
    float* m_d_aeinfer_output;
    float* m_d_map_st; 
    float* m_d_map_ae;
    float* m_d_combine;
    __device__ float m_d_st_start_quantiles;
    __device__ float m_d_st_end_quantiles;
    __device__ float m_d_ae_start_quantiles;
    __device__ float m_d_ae_end_quantiles; 

private:
    void init();
    void processInput(cv::Mat& image);
    void processOutput();
    void trtInfer();
    void trtInferAsyn();

public:
    Inference();
    Inference(int batch_size, int input_channel, int output_channel, int input_size, int output_size);
    ~Inference();
    Inference(const Inference& other) = delete;
    Inference(Inference&& other) = delete;
    Inference& operator=(const Inference& other) = delete;
    Inference& operator=(Inference&& other) = delete;

    void infer(cv::Mat& image);
};

#endif


