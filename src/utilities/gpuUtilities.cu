#include "gpuUtilities.cuh"
#include <cuda_runtime.h>

/*! \file gpuUtilities.cu
  defines kernel callers and kernels for some simple GPU array calculations

 \addtogroup utilityKernels
 @{
 */

/*!
  A function of convenience... set an array on the device
  */
template <typename T>
__global__ void gpu_set_array_kernel(T *arr,T value, int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    arr[idx] = value;
    return;
    };

template<typename T>
bool gpu_set_array(T *array, T value, int N,int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    gpu_set_array_kernel<<<nblocks, block_size>>>(array,value,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

template <typename T>
__global__ void gpu_copy_gpuarray_kernel(T *copyInto,T *copyFrom, int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    copyInto[idx] = copyFrom[idx];
    return;
    };

template<typename T>
bool gpu_copy_gpuarray(GPUArray<T> &copyInto,GPUArray<T> &copyFrom,int numberOfElementsToCopy,int maxBlockSize)
    {
    int N = copyFrom.getNumElements();
    if(numberOfElementsToCopy >0)
        N = numberOfElementsToCopy;
    if(copyInto.getNumElements() < N)
        copyInto.resize(N);
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = (N)/block_size + 1;
    ArrayHandle<T> ci(copyInto,access_location::device,access_mode::overwrite);
    ArrayHandle<T> cf(copyFrom,access_location::device,access_mode::read);
    gpu_copy_gpuarray_kernel<<<nblocks,block_size>>>(ci.data,cf.data,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

//Declare templates used...cuda is annoying sometimes
template bool gpu_copy_gpuarray<double>(GPUArray<double> &copyInto,GPUArray<double> &copyFrom,int n, int maxBlockSize);
template bool gpu_copy_gpuarray<double2>(GPUArray<double2> &copyInto,GPUArray<double2> &copyFrom,int n, int maxBlockSize);
template bool gpu_copy_gpuarray<int>(GPUArray<int> &copyInto,GPUArray<int> &copyFrom,int n, int maxBlockSize);
template bool gpu_copy_gpuarray<int3>(GPUArray<int3> &copyInto,GPUArray<int3> &copyFrom,int n, int maxBlockSize);

template bool gpu_set_array<int>(int *,int, int, int);
template bool gpu_set_array<unsigned int>(unsigned int *,unsigned int, int, int);
template bool gpu_set_array<int2>(int2 *,int2, int, int);
template bool gpu_set_array<int3>(int3 *,int3, int, int);
template bool gpu_set_array<double>(double *,double, int, int);
template bool gpu_set_array<double2>(double2 *,double2, int, int);
/** @} */ //end of group declaration
