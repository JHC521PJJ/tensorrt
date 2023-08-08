/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-31 21:43:34 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-07-31 21:43:34 
 * 
 * https://github.com/JHC521PJJ/tensorrt
 * 
 * This header file is an implementation of image preprocessing by OpenCV
 */

#ifndef __IMG_PREPROCESS_H__
#define __IMG_PREPROCESS_H__

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <iostream>
#include <vector>

inline constexpr int input_size = 256;

// Normalized three channels
inline cv::Mat normalizePerChannel(const cv::Mat& imageData, std::vector<float> means, std::vector<float> std) {
    cv::Mat normalizedImage;
    cv::Mat channels[3];
    cv::split(imageData, channels);

    for (int i=0; i < 3; ++i) {
        channels[i] = (channels[i] - means[i]) / std[i];
    }
    cv::merge(channels, 3, normalizedImage);
    return normalizedImage;
}

// Image to vector 
inline std::vector<float> imgToVector(cv::Mat& img){
    img = img.reshape(1, 1);
    std::vector<float> vec{};
    img.convertTo(vec, CV_32FC1);

    std::vector<float> out_vec{};
    for(int ch = 0; ch < 3; ch++){
        for(int i = ch; i < vec.size(); i += 3){
            out_vec.emplace_back(vec[i]);
        }
    }
    return out_vec;
}

inline std::vector<float> imagePreprocessing(cv::Mat& image) {
    cv::resize(image, image, cv::Size(input_size, input_size), cv::InterpolationFlags::INTER_LINEAR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1.0 / 255);
    cv::Mat process_img = normalizePerChannel(image, std::vector<float>{0.485f, 0.456f, 0.406f}, std::vector<float>{0.229f, 0.224f, 0.225f});
    return imgToVector(process_img);
}

template<typename T>
inline void printVector(const std::vector<T>& vec) {
    for(int i = 0; i < 10; ++i) {
        std::cout<<vec[i]<<" "; 
    }
    std::cout<<"\n";
}

#endif