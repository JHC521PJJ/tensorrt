/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-24 21:40:18 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-07-30 15:32:12
 * 
 * https://github.com/JHC521PJJ/tensorrt
 * 
 * This header file is an implementation of post-processing the result of tensorrt inference on the GPU
 */

#ifndef __RESULTTRANSFORMATE_H__
#define __RESULTTRANSFORMATE_H__


void resultTransformateGpu(float*  d_t_output, float*  d_s_output, float*  d_ae_output,
    float* d_teacher_mean,
    float* d_teacher_std,
    float* d_map_st,
    float* d_map_ae,
    float* d_combine,
    float& d_st_start_quantiles,
    float& d_st_end_quantiles,
    float& d_ae_start_quantiles,
    float& d_ae_end_quantiles,
    float& h_max_element);


#endif
