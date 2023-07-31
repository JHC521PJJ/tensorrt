/*
 * @Author: JHC521PJJ 
 * @Date: 2023-07-31 21:44:27 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-07-31 21:44:27 
 */

#ifndef __RESULT_TRANS_H__
#define __RESULT_TRANS_H__

#include <vector>
#include <fstream>
// #include <pmmintrin.h>
// #include <omp.h>



// inline std::vector<std::vector<float>> meanOperation(const std::vector<float>& t_onnxrun_output, 
//     const std::vector<float>& s_onnxrun_output,
//     const std::vector<float>& ae_onnxrun_output) {

//     std::vector<float> vec_map_st(384 * 56 * 56);
//     std::vector<float> vec_map_ae(384 * 56 * 56);
//     std::vector<float> vec_mean_st(56 * 56);
//     std::vector<float> vec_mean_ae(56 * 56);

//     for(int i = 0; i < 384 * 56 * 56; ++i) {
//         vec_map_st[i] = t_onnxrun_output[i] - s_onnxrun_output[i];
//         vec_map_st[i] *= vec_map_st[i]; 
//     }
//     for(int i = 0; i < 56 * 56; ++i) {
//         float temp = 0.0f;
//         for(int c = 0; c < 384; ++c) {
//             temp += vec_map_st[i + c * 56 * 56];
//         }
//         vec_mean_st[i] = temp / 384;
//     }
//     for(int i = 0; i < 384 * 56 * 56; ++i) {
//         vec_map_ae[i] = ae_onnxrun_output[i] - s_onnxrun_output[384 * 56 * 56 + i];
//         vec_map_ae[i] *= vec_map_ae[i]; 
//     }
//     for(int i = 0; i < 56 * 56; ++i) {
//         float temp = 0.0f;
//         for(int c = 0; c < 384; ++c) {
//             temp += vec_map_ae[i + c * 56 * 56];
//         }
//         vec_mean_ae[i] = temp / 384;
//     }
    
//     return {vec_mean_st, vec_mean_ae};
// }

// inline std::vector<std::vector<float>> meanOperation(const std::vector<float>& t_onnxrun_output, 
//     const std::vector<float>& s_onnxrun_output,
//     const std::vector<float>& ae_onnxrun_output) {

//     std::vector<float> vec_mean_st(56 * 56);
//     std::vector<float> vec_mean_ae(56 * 56);
//     float* ptr_map_st = new float[384 * 56 * 56];
//     float* ptr_map_ae = new float[384 * 56 * 56];
//     float** ptr_ptr_st = new float*[384];
//     float** ptr_ptr_ae = new float*[384];

//     for(int i = 0; i < 384 * 56 * 56 / 8; ++i) {
//         __m256 mm_t = _mm256_loadu_ps(&t_onnxrun_output[i * 8]);
//         __m256 mm_s1 = _mm256_loadu_ps(&s_onnxrun_output[i * 8]);
//         __m256 mm_s2 = _mm256_loadu_ps(&s_onnxrun_output[384 * 56 * 56 + i * 8]);
//         __m256 mm_ae = _mm256_loadu_ps(&ae_onnxrun_output[i * 8]);

//         __m256 mm_sub_st = _mm256_sub_ps(mm_t, mm_s1);
//         __m256 mm_sub_ae = _mm256_sub_ps(mm_ae, mm_s2);

//         __m256 mm_map_st = _mm256_mul_ps(mm_sub_st, mm_sub_st);
//         __m256 mm_map_ae = _mm256_mul_ps(mm_sub_ae, mm_sub_ae);

//         _mm256_storeu_ps(ptr_map_st + i * 8, mm_map_st);
//         _mm256_storeu_ps(ptr_map_ae + i * 8, mm_map_ae);
//     }

//     for(int i = 0; i < 384; ++i) {
//         float* ptr_channel = ptr_map_st + i * 56 * 56;
//         ptr_ptr_st[i] = ptr_channel;
//     }
//     for(int i = 0; i < 384; ++i) {
//         float* ptr_channel = ptr_map_ae + i * 56 * 56;
//         ptr_ptr_ae[i] = ptr_channel;
//     }

//     for(int i = 0; i < 56 * 56; ++i) {
//         float temp = 0.0f;
//         for(int c = 0; c < 384 / 8; ++c) {
//             temp += *(ptr_ptr_st[c] + i);

//             // __m256 mm_st = _mm256_loadu_ps(ptr_ptr_st[c * 8] + i);
//             // __m256 mm_ae = _mm256_loadu_ps(ptr_ptr_ae[c * 8] + i);


//         }
//         vec_mean_st[i] = temp / 384;
//     }

//     for(int i = 0; i < 56 * 56; ++i) {
//         float temp = 0.0f;
//         for(int c = 0; c < 384; ++c) {
//             temp += *(ptr_ptr_ae[c] + i);
//         }
//         vec_mean_ae[i] = temp / 384;
//     }
    
//     return {vec_mean_st, vec_mean_ae};
// }

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

