/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-31 21:44:01 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-07-31 21:44:01 
 */

#include "inference.h"
#include "time_count.h"
#include "onnxLog.h"
#include "resultTransformate.h"
#include <iostream>



int main(){
    const std::string file_path = "/data2/mvtec_loco/chip2/test/structural_anomalies/";
    std::string img_path = file_path + "*.bmp";
    std::vector<cv::String> vec_file{};
    cv::glob(img_path, vec_file);
    ocr::log_info<<"File size: "<< vec_file.size() << std::endl;

    Inference infer_run;

    std::vector<double> vec_time_avg{};
    for(int i = 0; i < vec_file.size(); ++i) {
        cv::Mat image = cv::imread(vec_file[i]);
        ocr::log_info << vec_file[i] << std::endl;

        TimeCount::instance().start();
        infer_run.infer(image);
        auto inference_time_count = TimeCount::instance().getTime();
        ocr::log_info << "All takes time: " << inference_time_count << "ms" << std::endl;
        vec_time_avg.emplace_back(inference_time_count);
    }

    double time_avg = vectorAverage(vec_time_avg);
    ocr::log_info << "Avg time: " << time_avg << std::endl;
    
    return 0;
}
