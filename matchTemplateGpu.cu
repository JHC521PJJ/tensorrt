#include "matchTemplateGpu.cuh"
#include "time_count.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>


using uchar = unsigned char;

constexpr int block_size = 16;
__constant__ uchar d_templ[221 * 221];

// __global__ void matchTemplate(const cv::cudev::PtrStepSz<uchar> img, 
//     const cv::cudev::PtrStepSz<uchar> templ, 
//     cv::cudev::PtrStepSz<float> result) {

//     const int x = blockDim.x * blockIdx.x + threadIdx.x;
//     const int y = blockDim.y * blockIdx.y + threadIdx.y;

//     if((x < result.cols) && (y < result.rows)){
//         long sum = 0;
//         for(int yy = 0; yy < templ.rows; yy++){
//             for(int xx = 0; xx < templ.cols; xx++){
//                 int diff = (img.ptr((y + yy))[x + xx] - templ.ptr(yy)[xx]);
//                 sum += (diff*diff);
//             }
//         }
//         result.ptr(y)[x] = sum;
//     }
// }

__global__ void matchTemplateKernel(const uchar* d_img, const uchar* d_templ, int* d_result,
    const int img_row, const int img_col,
    const int templ_row, const int templ_col, 
    const int result_row, const int result_col) {

    const int row = blockDim.y * blockIdx.y + threadIdx.y; 
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < result_row && col < result_col) {
        long sum = 0;
        for(int block_row = 0; block_row < templ_row; ++block_row) {
            for(int block_col = 0; block_col < templ_col; ++block_col) {
                int img_idx = (row + block_row) * img_col + col + block_col;
                int templ_idx = block_row * templ_col + block_col;
                
                int diff = d_img[img_idx] - d_templ[templ_idx];
                sum += (diff * diff);
            }
        }
        d_result[row * result_col + col] = sum;
    }
}

__global__ void matchTemplateConstKernel(const uchar* d_img, int* d_result,
    const int img_row, const int img_col,
    const int templ_row, const int templ_col, 
    const int result_row, const int result_col) {

    const int row = blockDim.y * blockIdx.y + threadIdx.y; 
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < result_row && col < result_col) {
        int sum = 0;
        #pragma unroll 8
        for(int block_row = 0; block_row < templ_row; ++block_row) {
            #pragma unroll 8
            for(int block_col = 0; block_col < templ_col; ++block_col) {
                int img_idx = (row + block_row) * img_col + col + block_col;
                int templ_idx = block_row * templ_col + block_col;
                
                int diff = d_img[img_idx] - d_templ[templ_idx];
                sum += (diff * diff);
            }
        }
        d_result[row * result_col + col] = sum;
    }
}

void matchTemplateGpu(const cv::Mat& h_img, const cv::Mat& h_templ, std::vector<int>& diff) {
    const int img_col = h_img.cols;
    const int img_row = h_img.rows;
	const int templ_col = h_templ.cols;
    const int templ_row = h_templ.rows;
    const int diff_col = h_img.cols - h_templ.cols + 1;
    const int diff_row = h_img.rows - h_templ.rows + 1;
	unsigned char* d_img;
    // unsigned char* d_templ;
    int* d_diff;

    size_t img_size = img_col * img_row * sizeof(unsigned char);
	size_t templ_size = templ_col * templ_row * sizeof(unsigned char);
    size_t diff_size = diff_col * diff_row * sizeof(int);

    cudaMalloc((void**)&d_img, img_size);
    // cudaMalloc((void**)&d_templ, templ_size);
    cudaMalloc((void**)&d_diff, diff_size);
    cudaMemcpy(d_img, h_img.data, img_size, cudaMemcpyHostToDevice); 
    //cudaMemcpy(d_templ, h_templ.data, templ_size, cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(d_templ, h_templ.data, templ_size);
    cudaMemcpy(d_diff, diff.data(), diff_size, cudaMemcpyHostToDevice); 

    unsigned int grid_rows = (diff_col + block_size - 1) / block_size;
    unsigned int grid_cols = (diff_row + block_size - 1) / block_size;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(grid_rows, grid_cols);
    TimeCount::instance().start(); 
    // matchTemplateKernel<<<dim_grid, dim_block>>>(d_img, d_templ, d_diff, img_row, img_col, templ_row, templ_col, diff_row, diff_col);
    matchTemplateConstKernel<<<dim_grid, dim_block>>>(d_img, d_diff, img_row, img_col, templ_row, templ_col, diff_row, diff_col);
    
    thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_diff);
    thrust::device_ptr<int> iter = thrust::min_element(d_ptr, d_ptr + diff_col * diff_row);
    std::cout<<"idx: " << iter - d_ptr <<"\n";

    cudaMemcpy(diff.data(), d_diff, diff_size, cudaMemcpyDeviceToHost); 
    TimeCount::instance().printTime();

    cudaFree(d_img);
    cudaFree(d_templ);
    cudaFree(d_diff);
}

void matchTemplateGpu(const cv::Mat& h_img, const cv::Mat& h_templ, int& min_index) {
    const int img_col = h_img.cols;
    const int img_row = h_img.rows;
	const int templ_col = h_templ.cols;
    const int templ_row = h_templ.rows;
    const int diff_col = h_img.cols - h_templ.cols + 1;
    const int diff_row = h_img.rows - h_templ.rows + 1;
	unsigned char* d_img;
    // unsigned char* d_templ;
    int* d_diff;

    size_t img_size = img_col * img_row * sizeof(unsigned char);
	size_t templ_size = templ_col * templ_row * sizeof(unsigned char);
    size_t diff_size = diff_col * diff_row * sizeof(int);

    cudaMalloc((void**)&d_img, img_size);
    cudaMalloc((void**)&d_diff, diff_size);
    cudaMemcpy(d_img, h_img.data, img_size, cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(d_templ, h_templ.data, templ_size);

    unsigned int grid_rows = (diff_col + block_size - 1) / block_size;
    unsigned int grid_cols = (diff_row + block_size - 1) / block_size;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(grid_rows, grid_cols);
    TimeCount::instance().start(); 
    matchTemplateConstKernel<<<dim_grid, dim_block>>>(d_img, d_diff, img_row, img_col, templ_row, templ_col, diff_row, diff_col);
    
    thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_diff);
    thrust::device_ptr<int> iter = thrust::min_element(d_ptr, d_ptr + diff_col * diff_row);
    min_index = iter - d_ptr;
    std::cout<<"idx: " << min_index <<"\n";

    TimeCount::instance().printTime();

    cudaFree(d_img);
    cudaFree(d_templ);
    cudaFree(d_diff);
}

/*
// use shared memory
__global__ void matchTemplateGpu_opt
(
    const cv::cudev::PtrStepSz<uchar> img,
    const cv::cudev::PtrStepSz<uchar> templ,
    cv::cudev::PtrStepSz<float> result
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ uchar temp[];

    if(threadIdx.x == 0){
        for(int yy = 0; yy < templ.rows; yy++){
            for(int xx = 0; xx < templ.cols; xx++){
                temp[yy*templ.cols+xx] = templ.ptr(yy)[xx];
            }
        }
    }
    __syncthreads();

    if((x < result.cols) && (y < result.rows)){
        long sum = 0;
        for(int yy = 0; yy < templ.rows; yy++){
            for(int xx = 0; xx < templ.cols; xx++){
                int diff = (img.ptr((y+yy))[x+xx] - temp[yy*templ.cols+xx]);
                sum += (diff*diff);
            }
        }
        result.ptr(y)[x] = sum;
    }
}

// use shared memory
__global__ void matchTemplateGpu_opt2
(
    const cv::cudev::PtrStepSz<uchar> img,
    const cv::cudev::PtrStepSz<uchar> templ,
    cv::cudev::PtrStepSz<float> result
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ uchar temp[];

    if(threadIdx.x == 0){
        for(int yy = 0; yy < templ.rows; yy++){
            const uchar* ptempl = templ.ptr(yy);
            for(int xx = 0; xx < templ.cols; xx++){
                temp[yy*templ.cols+xx]  = __ldg(&ptempl[xx]);
            }
        }
    }
    __syncthreads();

    if((x < result.cols) && (y < result.rows)){
        long sum = 0;
        for(int yy = 0; yy < templ.rows; yy++){
            const uchar* pimg = img.ptr((y+yy)) + x;
            for(int xx = 0; xx < templ.cols; xx++){
                int diff = (__ldg(&pimg[xx]) - temp[yy*templ.cols+xx]);
                sum += (diff*diff);
            }
        }
        result.ptr(y)[x] = sum;
    }
}

void launchMatchTemplateGpu
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result
)
{
    const dim3 block(64, 2);
    const dim3 grid(cv::cudev::divUp(result.cols, block.x), cv::cudev::divUp(result.rows, block.y));

    matchTemplateGpu<<<grid, block>>>(img, templ, result);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

// use shared memory
void launchMatchTemplateGpu_opt
(
    cv::cuda::GpuMat& img,
    cv::cuda::GpuMat& templ,
    cv::cuda::GpuMat& result
)
{
    const dim3 block(64, 2);
    const dim3 grid(cv::cudev::divUp(result.cols, block.x), cv::cudev::divUp(result.rows, block.y));
    const size_t shared_mem_size = templ.cols*templ.rows*sizeof(uchar);

    matchTemplateGpu_opt<<<grid, block, shared_mem_size>>>(img, templ, result);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

// use shared memory
void launchMatchTemplateGpu_opt2
(
    cv::cuda::GpuMat& img,
    cv::cuda::GpuMat& templ,
    cv::cuda::GpuMat& result
)
{
    const dim3 block(64, 2);
    const dim3 grid(cv::cudev::divUp(result.cols, block.x), cv::cudev::divUp(result.rows, block.y));
    const size_t shared_mem_size = templ.cols*templ.rows*sizeof(uchar);

    matchTemplateGpu_opt2<<<grid, block, shared_mem_size>>>(img, templ, result);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

double launchMatchTemplateGpu
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launchMatchTemplateGpu(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}

// use shared memory
double launchMatchTemplateGpu_opt
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launchMatchTemplateGpu_opt(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}

// use shared memory + __ldg
double launchMatchTemplateGpu_opt2
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launchMatchTemplateGpu_opt2(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}
*/
