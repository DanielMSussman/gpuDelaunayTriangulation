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

__global__ void fill_double2_array_uniform_kernel(curandState *state,
                                                  double2 *array,
                                                  double min,double max,
                                                  int N)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    curandState randState;
    randState = state[idx];

    double rng = curand_uniform(&randState);
    double rng2 = curand_uniform(&randState);
    array[idx].x= rng*(max-min)+min;
    array[idx].y= rng2*(max-min)+min;
    state[idx ] =randState;
    return;
    }
//!Call kernel to fill a gpuarray with uniform random values
bool gpu_fill_double2_array_uniform(curandState *state,
                                    double2 *array,
                                    double min,
                                    double max,
                                    int N)
    {
    unsigned int block_size = nThreads;
    if (N < nThreads) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    fill_double2_array_uniform_kernel<<<nblocks,block_size>>>(state,array,min,max,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

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
