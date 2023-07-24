#include "resultTransformate.cuh"
#include <stdio.h>
#include <cuda_runtime_api.h>

static constexpr int channel = 384;
static constexpr int out_size = 56;
__device__ float d_st_start_quantiles;
__device__ float d_st_end_quantiles;
__device__ float d_ae_start_quantiles;
__device__ float d_ae_end_quantiles; 


__global__ void squareDifferenceKernel(float* d_teacher, float* d_student, float* d_autoencoder, 
    const float* d_mean, const float* d_std, 
    float* d_map_st, float* d_map_ae, const int size) {

    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < size) {
        int c = idx / (out_size * out_size);
        d_teacher[idx] = (d_teacher[idx] - d_mean[c]) / d_std[c];
        d_map_st[idx] = d_teacher[idx] - d_student[idx];
        d_map_ae[idx] = d_autoencoder[idx] - d_student[channel * 56 * 56 + idx];
        d_map_st[idx] *= d_map_st[idx]; 
        d_map_ae[idx] *= d_map_ae[idx]; 
    }
}

__global__ void combineKernel(float* d_map_st, float* d_map_ae, float* d_combine,
    float d_st_start_quantiles,
    float d_st_end_quantiles,
    float d_ae_start_quantiles,
    float d_ae_end_quantiles,
    const int size) {
        
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float temp_st = 0.0f;
    float temp_ae = 0.0f;
    float d_mean_st = 0.0f;
    float d_mean_ae = 0.0f;

    if(idx < size) {
        for(int i = 0; i < channel; ++i) {
            temp_st += d_map_st[idx + i * 56 * 56];
            temp_ae += d_map_ae[idx + i * 56 * 56];
        }
        d_mean_st = temp_st / channel;
        d_mean_ae = temp_ae / channel;

        d_mean_st = 0.1f * (d_mean_st - d_st_start_quantiles) / (d_st_end_quantiles - d_st_start_quantiles);
        d_mean_ae = 0.1f * (d_mean_ae - d_ae_start_quantiles) / (d_ae_end_quantiles - d_ae_start_quantiles);
        d_combine[idx] = 0.5f * d_mean_st + 0.5f * d_mean_ae;
    }
}

void resultTransformate_v2(float*  d_t_output, float*  d_s_output, float*  d_ae_output,
    float* d_teacher_mean,
    float* d_teacher_std,
    float d_st_start_quantiles,
    float d_st_end_quantiles,
    float d_ae_start_quantiles,
    float d_ae_end_quantiles,
    const int device_id,
    std::vector<float>& vec_combine) {

    cudaSetDevice(device_id);
    float* d_map_st; float* d_map_ae;
    float* d_combine;
    cudaMalloc((void **) &d_map_st, sizeof(float) * channel * out_size * out_size);
    cudaMalloc((void **) &d_map_ae, sizeof(float) * channel * out_size * out_size);
    cudaMalloc((void **) &d_combine, sizeof(float) * out_size * out_size);

    int size = channel * out_size * out_size;
    unsigned int block_size = 16 * 16;
    unsigned int grid_size = (size + block_size - 1) / block_size;
    dim3 grid_dim(grid_size);
    dim3 block_dim(block_size);

    squareDifferenceKernel<<<grid_dim, block_dim>>>(d_t_output, d_s_output, d_ae_output, d_teacher_mean, d_teacher_std, d_map_st, d_map_ae, size);
    cudaDeviceSynchronize();

    size = out_size * out_size;
    grid_size = (size + block_size - 1) / block_size;
    combineKernel<<<grid_size, block_dim>>>(d_map_st, d_map_ae, d_combine, d_st_start_quantiles, d_st_end_quantiles, d_ae_start_quantiles, d_ae_end_quantiles, size);
    cudaMemcpy(vec_combine.data(), d_combine, sizeof(float) * out_size * out_size, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}