/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-24 21:40:18 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-07-26 18:06:28
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * This file is a demo program
 */


#include "time_count.h"
#include "inference.cuh"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Create an inference object 
    Inference infer_runner;
    
    // Get the path of all images in the folder
    const std::string file_path = "/mnt/DataDisk01/Data/chip6/test/good/";
    std::string img_path = file_path + "*.bmp";
    std::vector<cv::String> vec_file{};
    cv::glob(img_path, vec_file);
    std::cout<<"File size: "<< vec_file.size() << "\n";

    // Inference time record
    std::vector<double> time_record{};
    for(int i = 0; i < vec_file.size(); ++i) {
        // Read imag
        cv::Mat image = cv::imread(vec_file[i]);
        std::cout << vec_file[i] << std::endl;

        TimeCount::instance().start();
        // begin infer
        infer_runner.infer(image);
        
        auto infer_time_count = TimeCount::instance().getTime();
        sample::gLogInfo<< "Inference takes time: " << infer_time_count << "ms" << std::endl;
        time_record.push_back(infer_time_count);
    }

    auto time_sum = 0.0;
    for(auto& v: time_record) {
        time_sum += v;
    }
    auto time_avg = time_sum / time_record.size();
    sample::gLogInfo<< "Average time: " << time_avg << "ms" << std::endl;

    return 0;
}