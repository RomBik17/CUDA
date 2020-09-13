
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel(int a, int b, int* c)
{
    *c = a + b;
}

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

int main()
{
    printDevProp();
    int c;
    int* dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int));
    kernel << <1, 1 >> > (2, 7, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2+7=%d\n", c);
    cudaFree(dev_c);
    return 0;
}