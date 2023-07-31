cmake_minimum_required(VERSION 3.5)
project(test_infer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(SOURCE_FILES test.cpp)
set(INC_DIR /home/pjj/onnxruntime/onnxruntime-linux-x64-gpu-1.4.0/include)
set(LINK_DIR /home/pjj/onnxruntime/onnxruntime-linux-x64-gpu-1.4.0/lib/libonnxruntime.so)
set(CNPY_DIR /home/pjj/pythoncode/EfficientAD-main/deployment/cnpy/build/libcnpy.so)
set(ZLIB_DIR /usr/lib/x86_64-linux-gnu/libz.so)
aux_source_directory(. DIR_SRCS)

find_package(CUDA 10.1 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
endif()

message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message(STATUS "    OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


add_compile_options(-O2 -mavx -mfma -lz)
# add_executable(test_infer main.cpp)
add_executable(test_infer ${DIR_SRCS})
# target_sources(test_infer PUBLIC onnxInferenceRun.cpp onnxLog.cpp)

include_directories(${INC_DIR})     # 用${}引用变量
target_link_libraries(test_infer ${LINK_DIR})
target_link_libraries(test_infer ${CNPY_DIR})
target_link_libraries(test_infer ${ZLIB_DIR})
target_link_libraries(test_infer ${OpenCV_LIBS})
target_link_libraries(test_infer ${CUDAEXTRA_LIBS})
# set_property(TARGET test_infer PROPERTY CXX_STANDARD 11)





