#include "resultTransformate.cuh"
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

static constexpr int channel = 384;
static constexpr int out_size = 56;

// teacher_output - student_output[:, :out_channels])**2
// autoencoder_output - student_output[:, out_channels:])**2
// Calculate the squared variance of the two arrays
__global__ void squareDifferenceKernel(float* d_teacher, float* d_student, float* d_autoencoder, 
    const float* d_mean, const float* d_std, 
    float* d_map_st, float* d_map_ae, const int size) {

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < size) {
        int c = idx / (out_size * out_size);
        d_teacher[idx] = (d_teacher[idx] - d_mean[c]) / d_std[c];
        d_map_st[idx] = d_teacher[idx] - d_student[idx];
        d_map_ae[idx] = d_autoencoder[idx] - d_student[channel * out_size * out_size + idx];
        d_map_st[idx] *= d_map_st[idx]; 
        d_map_ae[idx] *= d_map_ae[idx]; 
    }
}

// torch.mean(d_map_st, dim=1)
// torch.mean(d_map_ae, dim=1)
// Calculate the mean of the two arrays in the dim=1 dimension and add them
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
    float& h_max_element) {

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
    
    // Use the thrust library to calculate array maximums
    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(d_combine);
    thrust::device_ptr<float> iter = thrust::max_element(d_ptr, d_ptr + out_size * out_size);
    h_max_element = *iter;
}


// void resultTransformateGpu(float*  d_t_output, float*  d_s_output, float*  d_ae_output,
//     float* d_teacher_mean,
//     float* d_teacher_std,
//     float d_st_start_quantiles,
//     float d_st_end_quantiles,
//     float d_ae_start_quantiles,
//     float d_ae_end_quantiles,
//     std::vector<float>& vec_combine) {

//     float* d_map_st; 
//     float* d_map_ae;
//     float* d_combine;
//     cudaMalloc((void **) &d_map_st, sizeof(float) * channel * out_size * out_size);
//     cudaMalloc((void **) &d_map_ae, sizeof(float) * channel * out_size * out_size);
//     cudaMalloc((void **) &d_combine, sizeof(float) * out_size * out_size);

//     int size = channel * out_size * out_size;
//     unsigned int block_size = 16 * 16;
//     unsigned int grid_size = (size + block_size - 1) / block_size;
//     dim3 grid_dim(grid_size);
//     dim3 block_dim(block_size);

//     squareDifferenceKernel<<<grid_dim, block_dim>>>(d_t_output, d_s_output, d_ae_output, d_teacher_mean, d_teacher_std, d_map_st, d_map_ae, size);
//     cudaDeviceSynchronize();

//     size = out_size * out_size;
//     grid_size = (size + block_size - 1) / block_size;
//     combineKernel<<<grid_size, block_dim>>>(d_map_st, d_map_ae, d_combine, d_st_start_quantiles, d_st_end_quantiles, d_ae_start_quantiles, d_ae_end_quantiles, size);
//     // cudaMemcpy(vec_combine.data(), d_combine, sizeof(float) * out_size * out_size, cudaMemcpyDeviceToHost);
    
//     thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(d_combine);
//     thrust::device_ptr<float> iter = thrust::max_element(d_ptr, d_ptr + out_size * out_size);

//     cudaFree(d_map_st);
//     cudaFree(d_map_ae);
//     cudaFree(d_combine);
// }