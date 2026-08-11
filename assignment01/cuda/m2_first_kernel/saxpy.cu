#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err_));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void saxpy(float *x, float *y, float a, int n) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n ; idx += blockDim.x * gridDim.x  ){
        y[idx] = a * x[idx] + y[idx];
    }
}

int main(int argc, char **argv){
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    long parsed = strtol(argv[1], nullptr, 10);
    if (parsed < 0) {
        fprintf(stderr, "n must be non-negative\n");
        return 1;
    }

    int n = (int)parsed;
    if (!n) {
        printf("SUM=0\n");
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);
    float a = 2.0;

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    for (int i = 0; i < n ;i ++){
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512; 
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // ================= 计时窗口开始 =================
    CUDA_CHECK(cudaEventRecord(start));

    saxpy<<<blocks, threads>>>(d_x, d_y, a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // ================= 计时窗口结束 =================

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double SUM = 0;
    for (int i = 0; i < n; i++) SUM += (double)h_y[i];

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    free(h_x);
    free(h_y);
    printf("SUM=%.0f n=%d time=%.2fms\n",SUM,n,kernel_ms);
    return 0;
}