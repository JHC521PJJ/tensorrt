# Background
1. This project is based on a paper of anomaly detection from CVPR2023：[EfficientAD](https://arxiv.org/abs/2303.14535).
2. Once the model has been trained using pytorch, the model was exported to onnx format.
3. ONNXRuntime-gpu was usred for model inference, and then used c++ to rewrite the image pre-processing and post-inference.
4. TensorRT was used for model inference, and all image preprocessing and subsequent processing after inference were rewritten using CUDA C++ and transplanted to GPU for accelerated processing.
# Environment
* OS: Ubuntu22.04
* CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
* GPU: Nvidia GeForce RTX 3090
* CUDA: 12.0
* cudnn: 8.8.0
* OpenCV: 4.6.0
* Pytorch: 2.0.1
* TensorRT: 8.6.1
* ONNXRuntime-gpu: 1.15.0
# Description to the main document
* main.cpp:               Use this project to detect the demo of the test image
* imagePreprocess.cuh:    The implementation of image preprocessing on the GPU
* resultTransformate.cuh: The implementation of post-processing the result of tensorrt inference on the GPU
* trtInferenceRunner.h:   The implementation of the TensorRT inference class
* inference.cuh:          The implementation of Infernece class
* onnxrumtime:            The implementation of ONNXRuntime-gpu
# Running the sample
```
cd tensorrt-master
mkdir build
cd build
cmake ..
make
./test_infer
```
# Comparison of results
Results on 500 images test set
| | CPU  | ONNXRuntime-gpu  | TensorRT
----| ---- | ---- | ----  
 Inference time/ms  | 2245 | 35 |8
 Acceleration ratio  | 1 | 64 |281
 
# Maintainers
[@JHC521PJJ](https://github.com/JHC521PJJ).
# License
[MIT](LICENSE) © Richard Littauer
