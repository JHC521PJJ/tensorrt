#ifndef __NPY_VEC_H__
#define __NPY_VEC_H__

#include <vector>
#include "cnpy.h"

/*
struct NpyArray {
    std::vector<size_t> shape;
    size_t word_size;
    template<typename T> T* data();
};
*/

inline std::vector<float> npyToVector(const char* npy_path) {
    cnpy::NpyArray npy_arr = cnpy::npy_load(npy_path);    
    return std::vector<float>(npy_arr.data<float>(), npy_arr.data<float>() + npy_arr.shape[1]);
}

inline float npyToValue(const char* npy_path) {
    cnpy::NpyArray npy_value = cnpy::npy_load(npy_path);
    return *npy_value.data<float>();
}

#endif
