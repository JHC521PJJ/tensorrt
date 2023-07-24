#include "trtInferenceRunner.h"
#include "imagePreprocess.h"
#include "imagePreprocess.cuh"
#include "time_count.h"
#include "npyToVector.h"
#include "resultTransformate.cuh"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

static constexpr int batch_size = 1; 
static constexpr int channel = 384;
static constexpr int out_size = 56;
__device__ float d_st_start_quantiles;
__device__ float d_st_end_quantiles;
__device__ float d_ae_start_quantiles;
__device__ float d_ae_end_quantiles; 

void printHelpInfo() {
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv) {
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK) {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help) {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    const char* t_onnx = "teacher_final_simp.onnx";
    const char* s_onnx = "student_final_simp.onnx";
    const char* ae_onnx = "autoencoder_final_simp.onnx";
    TrtInferenceRunner t_sample(initializeSampleParams(args, t_onnx));
    TrtInferenceRunner s_sample(initializeSampleParams(args, s_onnx));
    TrtInferenceRunner ae_sample(initializeSampleParams(args, ae_onnx));

    std::vector<float> vec_teacher_mean = npyToVector("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/t_mean_quantiles.npy");
    std::vector<float> vec_teacher_std = npyToVector("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/t_std_quantiles.npy");
    float q_st_start_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_st_start_quantiles.npy");
    float q_st_end_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_st_end_quantiles.npy");
    float q_ae_start_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_ae_start_quantiles.npy");
    float q_ae_end_quantiles = npyToValue("/mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/output/2/trainings/mvtec_loco/chip6/q_ae_end_quantiles.npy");

    float* d_teacher_mean; 
    float* d_teacher_std;
    cudaMalloc((void **) &d_teacher_mean, sizeof(float) * channel);
    cudaMalloc((void **) &d_teacher_std, sizeof(float) * channel);
    cudaMemcpy(d_teacher_mean, vec_teacher_mean.data(), sizeof(float) * channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_teacher_std, vec_teacher_std.data(), sizeof(float) * channel, cudaMemcpyHostToDevice);
    d_st_start_quantiles = q_st_start_quantiles;
    d_st_end_quantiles = q_st_end_quantiles;
    d_ae_start_quantiles = q_ae_start_quantiles;
    d_ae_end_quantiles = q_ae_end_quantiles;
    float* t_output;
    float* s_output;
    float* ae_output;
    cudaMalloc((void**)&t_output, 384 * 56 * 56 * sizeof(float));
    cudaMalloc((void**)&s_output, 768 * 56 * 56 * sizeof(float));
    cudaMalloc((void**)&ae_output, 384 * 56 * 56 * sizeof(float));
    
    sample::gLogInfo << "Building and running a GPU inference engine for Onnx Model" << std::endl;
    t_sample.build();
    s_sample.build();
    ae_sample.build();
    sample::gLogInfo<<"Build engine success"<<std::endl;

    const std::string file_path = "/mnt/DataDisk01/Data/chip6/test/broken/";
    std::string img_path = file_path + "*.bmp";
    std::vector<cv::String> vec_file{};
    cv::glob(img_path, vec_file);
    std::cout<<"File size: "<< vec_file.size() << "\n";

    std::vector<double> vec_time_avg{};
    for(int i = 0; i < vec_file.size(); ++i) {
        cv::Mat image = cv::imread(vec_file[i]);
        sample::gLogInfo << vec_file[i] << std::endl;
        TimeCount::instance().start();

        t_sample.processInput(image);
        s_sample.processInput(image);
        ae_sample.processInput(image);

        t_sample.infer_v2(t_output);
        s_sample.infer_v2(s_output);
        ae_sample.infer_v2(ae_output);
        auto infer_time_count = TimeCount::instance().getTime();
        sample::gLogInfo<< "Inference takes time: " << infer_time_count << "ms" << std::endl;
        sample::gLogInfo<<"Inference success"<<std::endl;
        cudaDeviceSynchronize();
        std::vector<float> vec_combined(56 * 56);
        resultTransformate_v2(t_output, s_output, ae_output, d_teacher_mean, d_teacher_std,
                                d_st_start_quantiles,
                                d_st_end_quantiles,
                                d_ae_start_quantiles,
                                d_ae_end_quantiles,
                                0,
                                vec_combined);
        auto result_transformate = TimeCount::instance().getTime();
        sample::gLogInfo<< "Result Transformate takes time: " << result_transformate - infer_time_count<< "ms" << std::endl;

        auto it_ad_score = std::max_element(vec_combined.begin(), vec_combined.end());
        sample::gLogInfo<< "Score: " << *it_ad_score << " ";
        if(*it_ad_score < 1.0f) {
            sample::gLogInfo<< "[Normal]" << std::endl;
        }
        auto all_time_count = TimeCount::instance().getTime();
        sample::gLogInfo<< "All takes time: " << all_time_count << "ms" << std::endl;
    }
    

    return 0;
}