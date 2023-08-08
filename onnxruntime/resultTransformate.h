/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-31 21:44:27 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-07-31 21:44:27 
 * 
 * https://github.com/JHC521PJJ/tensorrt
 */

#ifndef __RESULT_TRANS_H__
#define __RESULT_TRANS_H__

#include <vector>
#include <fstream>

// Calculate the mean of dim=1 after the subtraction of the tensor
inline std::vector<std::vector<float>> meanOperation(const std::vector<float>& t_onnxrun_output, 
    const std::vector<float>& s_onnxrun_output,
    const std::vector<float>& ae_onnxrun_output) {

    std::vector<float> vec_map_st(384 * 56 * 56);
    std::vector<float> vec_map_ae(384 * 56 * 56);
    std::vector<float> vec_mean_st(56 * 56);
    std::vector<float> vec_mean_ae(56 * 56);

    for(int i = 0; i < 384 * 56 * 56; ++i) {
        vec_map_st[i] = t_onnxrun_output[i] - s_onnxrun_output[i];
        vec_map_ae[i] = ae_onnxrun_output[i] - s_onnxrun_output[384 * 56 * 56 + i];
        vec_map_st[i] *= vec_map_st[i]; 
        vec_map_ae[i] *= vec_map_ae[i]; 
    }

    for(int i = 0; i < 56 * 56; ++i) {
        float temp_st = 0.0f;
        float temp_ae = 0.0f;
        for(int c = 0; c < 384; ++c) {
            temp_st += vec_map_st[i + c * 56 * 56];
            temp_ae += vec_map_ae[i + c * 56 * 56];
        }
        vec_mean_st[i] = temp_st / 384;
        vec_mean_ae[i] = temp_ae / 384;
    }
    
    return {vec_mean_st, vec_mean_ae};
}

// Tensor combining
inline std::vector<float> combineOperation(std::vector<float>& vec_mean_st, std::vector<float>& vec_mean_ae,
    const float q_st_start_quantiles,
    const float q_st_end_quantiles,
    const float q_ae_start_quantiles,
    const float q_ae_end_quantiles) {

    std::vector<float> vec_combined(56 * 56);
    for(int i = 0; i < 56 * 56; ++i) {
        vec_mean_st[i] = 0.1f * (vec_mean_st[i] - q_st_start_quantiles) / (q_st_end_quantiles - q_st_start_quantiles);
        vec_mean_ae[i] = 0.1f * (vec_mean_ae[i] - q_ae_start_quantiles) / (q_ae_end_quantiles - q_ae_start_quantiles);
        vec_combined[i] = 0.5f * vec_mean_st[i] + 0.5f * vec_mean_ae[i];
    }

    return vec_combined;
}

// Calculate the mean of the vector
template<typename T>
inline T vectorAverage(const std::vector<T>& vec) {
    T avg = T{};
    for(auto& v: vec) {
        avg += v;
    }
    avg = avg / vec.size();
    return avg;
}

template<typename T>
inline void saveVector(const std::vector<T>& vec, const char* save_path) {
    std::ofstream outfile(save_path);
    if (outfile.is_open()) {
        for (const auto& v : vec) {
            outfile << v << " ";
        }
        outfile.close();
    } 
    else {
        std::cout << "Unable to open file for writing." << std::endl;
    }
}

inline std::vector<float> readVectorFromTxt(const char* save_path) {
    std::ifstream infile(save_path);
    std::vector<float> vec{};
    float num;
    while (infile >> num) {
        vec.push_back(num);
    }
    infile.close();
    return vec;
}

#endif

