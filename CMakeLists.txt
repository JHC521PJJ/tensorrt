cmake_minimum_required(VERSION 3.5)
project(test_infer)


include_directories(/mnt/DataDisk02/home/pjj/tensorrt/TensorRT-8.6.1.6/include)
include_directories(/mnt/DataDisk02/home/pjj/tensorrt/TensorRT-8.6.1.6/samples/common)
set(TENSORRT_LIB_PATH "/mnt/DataDisk02/home/pjj/tensorrt/TensorRT-8.6.1.6/lib")
set(CNPY_DIR /mnt/DataDisk02/home/pjj/anomaly_detection/EfficientAD-main/deployment/cnpy/build/libcnpy.so)
set(SAMPLES_COMMON_SOURCES ${SAMPLES_DIR}/mnt/DataDisk02/home/pjj/tensorrt/TensorRT-8.6.1.6/samples/common/logger.cpp )
set(CUDA_MODULE_LOADING=LAZY)
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")
file(GLOB_RECURSE CUDA_HEADERS  *.cuh)
file(GLOB CUDA_SOURCES *.cu)

# add CUDA
find_package(CUDA 12.0 REQUIRED)
find_package(OpenCV REQUIRED)
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 14)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14 -O2)

cuda_add_library(process_gpu SHARED ${CUDA_HEADERS} ${CUDA_SOURCES})
target_link_libraries(process_gpu ${OpenCV_LIBS})
target_link_libraries(process_gpu ${CNPY_DIR})
set (CUDAEXTRA_LIBS ${CUDAEXTRA_LIBS} process_gpu)

add_executable(test_infer main.cpp ${SAMPLES_COMMON_SOURCES})


include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(test_infer ${LIBS} ${CUDA_LIBRARIES})
target_link_libraries(test_infer ${OpenCV_LIBS})
target_link_libraries(test_infer ${CNPY_DIR})
target_link_libraries(test_infer ${CUDAEXTRA_LIBS})
