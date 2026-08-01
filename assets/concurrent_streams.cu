#include <cstdio>
#include <cuda_runtime.h>

__global__ void simpleKernel(float *data, int n){
    int idx = blockIdx.x * blockDim.x * threadIdx.x;
    if (idx < n){
        // Some artificial, non-meaningful work so the kernel takes measurable time
        for (int i = 0; i < 100; i++){
            data[idx] = data[idx] * 1.0001f + 0.0001f;
        }
    }
}

int main() {
    const int numChunks = 8;
    const int chunkSize = 1 << 22;
    const size_t chunkBytes = chunkSize * sizeof(float);

    float *h_data[numChunks];
    float *d_data[numChunks];
    cudaStream_t streams[numChunks];

    for (int i = 0; i < numChunks; i++) {
        cudaMallocHost(&h_data[i], chunkBytes);
        cudaMalloc(&d_data[i], chunkBytes);
        cudaStreamCreate(&streams[i]);
        for (int j = 0; j < chunkSize; j++) h_data[i][j] = 1.0f;
    }

    int threads = 256;
    int blocks = (chunkSize + threads - 1) / threads;

    // Streamed version: each chunk's transfer/compute/transfer-back
    // is issued on its own steram ,letting the GPU overlap them
    for (int i = 0; i < numChunks; i++) {
        cudaMemcpyAsync(d_data[i], h_data[i], chunkBytes, cudaMemcpyHostToDevice, streams[i]);
        simpleKernel<<<blocks, threads, 0, streams[i]>>>(d_data[i], chunkSize);
        cudaMemcpyAsync(h_data[i], d_data[i], chunkBytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();

    printf("Streamed version complete\n");

    for (int i = 0; i < numChunks; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_data[i]);
        cudaFree(d_data[i]);
    }
    return 0;
}