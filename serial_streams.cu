#include <cstdio>
#include <cuda_runtime.h>

__global__ void simpleKernel(float *data, int n){
    int idx = blockIdx.x * blockDim.x * threadIdx.x;
    if (idx < n){
        // Putting in some artificial work so the kernel takes measurable time
        for (int i = 0; i < 100; i++){
            data[idx] = data[idx] * 1.0001f + 0.0001f;
        }
    }
}

int main() {
    const int numChunks = 8;
    const int chunkSize = 1 << 22; // ~4M floats per chunk
    const size_t chunkBytes = chunkSize * sizeof(float);

    float *h_data[numChunks];
    float *d_data[numChunks];

    for (int i = 0; i < numChunks; i++){
        cudaMallocHost(&h_data[i], chunkBytes); // pinned host memory
        cudaMalloc(&d_data[i], chunkBytes);
    }

    int threads = 256;
    int blocks = (chunkSize + threads - 1) / threads;

    // Serial Version: copy, compute, copy back, one chunk fully at a time
    for (int i = 0; i < numChunks; i++){
        cudaMemcpy(d_data[i], h_data[i], chunkBytes, cudaMemcpyHostToDevice);
        simpleKernel<<<blocks, threads>>>(d_data[i], chunkSize);
        cudaMemcpy(h_data[i], d_data[i], chunkBytes, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    printf("Serial version complete\n");

    for (int i = 0; i < numChunks; i++){
        cudaFreeHost(h_data[i]);
        cudaFree(d_data[i]);
    }
    return 0;
}