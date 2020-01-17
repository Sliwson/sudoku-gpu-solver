#include "solver.cuh"

__global__ void testKernel()
{
    const unsigned int tid = threadIdx.x;
    const unsigned int num_threads = blockDim.x;
}

void runKernel()
{
	// setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(1, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads >>>();
}
