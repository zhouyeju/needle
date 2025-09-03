#include "tensor.h"
#include <cuda.h>
#include <cuda_runtime.h>

void free_tensor(void* x) {
    cudaFree(x);
}

void* copy_tensor_d2h(void* x, int nelements, int element_size) {
    float* y = (float*)malloc(nelements * element_size);
    cudaMemcpy(y, x, nelements * element_size, cudaMemcpyDeviceToHost);
    return (void*)y;
}

void* copy_tensor_h2d(void* x, int nelements, int element_size) {
    float* y;
    cudaMalloc((void**)&y, nelements * element_size);
    cudaMemcpy(y, x, nelements * element_size, cudaMemcpyHostToDevice);
    return (void*)y;
}