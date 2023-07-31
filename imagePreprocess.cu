#include "imagePreprocess.cuh"
#include <cuda_runtime_api.h>

// Img resize
__global__ void resizeKernel(const unsigned char* d_input, unsigned char* d_output, 
    const int input_w, const int input_h, const int output_w, const int output_h) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < output_h && col < output_w) {
        int old_row = row * input_h / output_h;
        int old_col = col * input_w / output_w;
        
        d_output[(row * output_w + col) * 3]     = d_input[(old_row * input_w + old_col) * 3];
        d_output[(row * output_w + col) * 3 + 1] = d_input[(old_row * input_w + old_col) * 3 + 1];
        d_output[(row * output_w + col) * 3 + 2] = d_input[(old_row * input_w + old_col) * 3 + 2];
    }
}

// BGR to RGB
__global__ void toRGBKernel(unsigned char* d_input, const int width, const int height) {
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int index = (row * width + col) * 3;

    if (col < width && row < height) {
        auto temp = d_input[index];
        d_input[index] = d_input[index + 2];
        d_input[index + 2] = temp;
    }
}

// // The uchar image is converted to float
// __global__ void convertToFloatKernel(unsigned char* d_input, float* d_output, 
//     const int width, const int height) {
//     const int col = threadIdx.x + blockIdx.x * blockDim.x;
//     const int row = threadIdx.y + blockIdx.y * blockDim.y;
//     const int index = (row * width + col) * 3;

//     if (col < width && row < height) {
//         d_output[index]     = (float)d_input[index] / 255.0f;
//         d_output[index + 1] = (float)d_input[index + 1] / 255.0f;
//         d_output[index + 2] = (float)d_input[index + 2] / 255.0f;
//     }
// }

// // Normalize
// __global__ void normalizeKernel(float* d_input, const int width, const int height) {
//     const int col = threadIdx.x + blockIdx.x * blockDim.x;
//     const int row = threadIdx.y + blockIdx.y * blockDim.y;
//     const int index = (row * width + col) * 3;

//     if (col < width && row < height) {
//         d_input[index]     = (d_input[index] - 0.485f) / 0.229f;
//         d_input[index + 1] = (d_input[index + 1] - 0.456f) / 0.224f;
//         d_input[index + 2] = (d_input[index + 2] - 0.406f) / 0.225f;
//     }
// }

// Normalize
__global__ void normalizeKernel(unsigned char* d_input, float* d_output, 
    const int width, const int height) {
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int index = (row * width + col) * 3;

    if (col < width && row < height) {
        // The uchar image is converted to float
        d_output[index]     = (float)d_input[index] / 255.0f;
        d_output[index + 1] = (float)d_input[index + 1] / 255.0f;
        d_output[index + 2] = (float)d_input[index + 2] / 255.0f;

        // Normalize
        d_output[index]     = (d_output[index] - 0.485f) / 0.229f;
        d_output[index + 1] = (d_output[index + 1] - 0.456f) / 0.224f;
        d_output[index + 2] = (d_output[index + 2] - 0.406f) / 0.225f;
    }
}

// In the opencv, HWC format to CHW
__global__ void toVectorKernel(const float* d_input, float* d_output, 
    const int width, const int height) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < width * height) {
        d_output[index] = d_input[index * 3];
        d_output[index + width * height] = d_input[index * 3 + 1];
        d_output[index + width * height * 2] = d_input[index * 3 + 2];
    }
}

void imagePreprocessingGpu(cv::Mat& image, float* d_preprocess_output) {
    int old_width = image.cols;
    int old_height = image.rows;
    constexpr int new_width = 256;
    constexpr int new_height = 256;
    unsigned char* d_input;
    unsigned char* d_output_resize;
    float* d_outout_float;

    constexpr int block_size = 16;
    cudaMalloc((void**)&d_input, old_width * old_height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_output_resize, new_width * new_height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_outout_float, new_width * new_height * 3 * sizeof(float));
    cudaMemcpy(d_input, image.data, old_width * old_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned int grid_rows = (new_height + block_size - 1) / block_size;
    unsigned int grid_cols = (new_width + block_size - 1) / block_size;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(grid_rows, grid_cols);
    
    resizeKernel<<<dim_grid, dim_block>>>(d_input, d_output_resize, old_width, old_height, new_width, new_height);
    cudaDeviceSynchronize();

    toRGBKernel<<<dim_grid, dim_block>>>(d_output_resize, new_width, new_height);
    cudaDeviceSynchronize();

    // convertToFloatKernel<<<dim_grid, dim_block>>>(d_output_resize, d_outout_float, new_width, new_height);
    // cudaDeviceSynchronize();

    // normalizeKernel<<<dim_grid, dim_block>>>(d_outout_float, new_width, new_height);
    normalizeKernel<<<dim_grid, dim_block>>>(d_output_resize, d_outout_float, new_width, new_height);
    cudaDeviceSynchronize();

    dim3 dim_block1(256);
    dim3 dim_grid1((new_height * new_width + 256 - 1) / 256);
    toVectorKernel<<<dim_grid1, dim_block1>>>(d_outout_float, d_preprocess_output, new_width, new_height);
    
    cudaFree(d_input);
    cudaFree(d_output_resize);
    cudaFree(d_outout_float);
}