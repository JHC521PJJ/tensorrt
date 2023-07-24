/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-24 21:40:18 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-07-24 21:55:46
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * This header file is an implementation of image preprocessing on the GPU
 */


#ifndef __IMG_PREPROCESSGPU_H__
#define __IMG_PREPROCESSGPU_H__

#include <opencv2/opencv.hpp>
#include <vector>

void imagePreprocessingGpu(cv::Mat& image, float* d_preprocess_output);

#endif