
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define imin(a, b) ((a < b) ? a : b)

const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

void printDevProp()
{
    cudaDeviceProp devProp;
    int count;
    cudaGetDeviceCount(&count);
    cudaGetDeviceProperties(&devProp, count - 1);

    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %u\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %u\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %u\n", devProp.totalConstMem);
    printf("Texture alignment:             %u\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));

    return;
}

//can be called only from GPU function
__device__ void generated()
{
    printf("generated\n");
}

//vector add function
__global__ void addKernel(int *a, int *b, int* c)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N)
    {
        c[i] = a[i] + b[i];
        i += blockDim.x * gridDim.x;
    }
}

//can be called from GPU and CPU
__global__ void vectorGenerateKernel(int* a, int* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N)
    {
        a[i] = i;
        b[i] = i * 2;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void dotKernel(int* a, int* b, int* c)
{
    
    __shared__ int cache[threadsPerBlock];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    int temp = 0;
    while (i < N)
    {
        temp += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;

    __syncthreads();

    int j = blockDim.x / 2;
    while (j != 0)
    {
        if (cacheIndex < j) cache[cacheIndex] += cache[cacheIndex + 1];
        __syncthreads();
        j /= 2;
    }

    if (cacheIndex == 0) c[blockIdx.x] = cache[0];
}

int main()
{
    printDevProp();
    int c, *partial_c;
    int *dev_a, *dev_b, *dev_partial_c;

    c = 0;
    partial_c = (int *)malloc(blocksPerGrid * sizeof(int));

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(int));

    vectorGenerateKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b);

    printf("generated");
    //addKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_c);
    dotKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < blocksPerGrid; ++i)
    {
        c += partial_c[i];
    }

    printf("dot %d", c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    delete[] partial_c;
    return 0;
}
