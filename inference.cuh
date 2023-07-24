/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-24 21:40:18 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-07-24 21:56:00
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
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
    int m_channel = 384;
    int m_batch_size = 1; 
    int m_out_size = 56;
    TrtInferenceRunner m_teacher_infer;
    TrtInferenceRunner m_student_infer;
    TrtInferenceRunner m_ae_infer;

    float* m_d_teacher_mean; 
    float* m_d_teacher_std;
    float* m_d_preprocess_output;
    float* m_t_infer_output;
    float* m_s_infer_output;
    float* m_ae_infer_output;
    __device__ float m_d_st_start_quantiles;
    __device__ float m_d_st_end_quantiles;
    __device__ float m_d_ae_start_quantiles;
    __device__ float m_d_ae_end_quantiles; 

public:
    Inference();
    Inference(const Inference& other) = delete;
    Inference(Inference&& other) = delete;
    Inference& operator=(const Inference& other) = delete;
    Inference& operator=(Inference&& other) = delete;
    ~Inference() {}

    void init();
    void processInput(cv::Mat& image);
    void trtInference();
    void processOutput();
    void infer(cv::Mat& image);
};

#endif


