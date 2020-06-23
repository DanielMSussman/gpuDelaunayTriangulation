#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "noiseSource.cuh"
#include "std_include.h"

#define nThreads 256

/** \file noiseSource.cu
    * Defines kernel callers and kernels for GPU random number generation
*/

/*!
    \addtogroup utilityKernels
    @{
*/

/*!
  Each thread -- most likely corresponding to each cell -- is initialized with a different sequence
  of the same seed of a cudaRNG
*/
__global__ void initialize_RNG_array_kernel(curandState *state, int N,int Timestep,int GlobalSeed)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    curand_init(GlobalSeed,idx,Timestep,&state[idx]);
    return;
    };

//!Call the kernel to initialize a different RNG for each particle
bool gpu_initialize_RNG_array(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed)
    {
    unsigned int block_size = nThreads;
    if (N < nThreads) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    initialize_RNG_array_kernel<<<nblocks,block_size>>>(states,N,Timestep,GlobalSeed);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
