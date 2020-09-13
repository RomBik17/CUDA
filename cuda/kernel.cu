
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10

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
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

//can be called from GPU and CPU
__global__ void vectorGenerateKernel(int* a, int* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = -i;
        b[i] = i * i;
    }
}

int main()
{
    printDevProp();
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    vectorGenerateKernel << <N, 1 >> > (dev_a, dev_b);

    addKernel << <N, 1 >> > (dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; ++i)
    {
        printf("%d\n", c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
