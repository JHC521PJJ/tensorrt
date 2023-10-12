/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-31 21:43:44 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-08-15 12:30:38
 * 
 * https://github.com/JHC521PJJ/tensorrt
 */

#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include "onnxInferenceRun.h"
#include <opencv2/opencv.hpp>


class Inference {
public:
    using VecFloat = std::vector<float>;

private:
    int m_batch_size;                       // The batch size of input image 
    int m_input_channel;                    // The channels of input image
    int m_output_channel;                   // The channels of output image
    int m_input_size;                       // The size of input image
    int m_output_size;                      // The size of output image
    
    OnnxInferenceRunner m_teacher_infer;    // The Otr inference class of teacher networks
    OnnxInferenceRunner m_student_infer;    // The Otr inference class of student network
    OnnxInferenceRunner m_ae_infer;         // The Otr inference class of auto encorder network

    // Initializes temporary variables
    VecFloat m_teacher_mean; 
    VecFloat m_teacher_std;
    VecFloat m_preprocess_output;
    VecFloat m_tinfer_output;
    VecFloat m_sinfer_output;
    VecFloat m_aeinfer_output;
    VecFloat m_d_map_st; 
    VecFloat m_d_map_ae;
    VecFloat m_d_combine;
    float m_q_st_start_quantiles;
    float m_q_st_end_quantiles;
    float m_q_ae_start_quantiles;
    float m_q_ae_end_quantiles; 

private:
    void init();
    void processInput(cv::Mat& image);
    void processOutput();
    void otrInfer();

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