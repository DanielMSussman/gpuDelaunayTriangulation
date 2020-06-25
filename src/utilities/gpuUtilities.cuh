#ifndef gpuutilities_CUH__
#define gpuutilities_CUH__

//#include <cuda_runtime.h>
//#include "std_include.h"
//#include "gpuarray.h"

/*!
 \file gpuUtilities.cuh
A file providing an interface to the relevant cuda calls for some simple GPU array manipulations
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for the utilities base
 */

//!set every element of an array to the specified value
template<typename T>
bool gpu_set_array(T *arr,
                   T value,
                   int N,
                   int maxBlockSize=512);

//!copy data into target on the device
template<typename T>
bool gpu_copy_gpuarray(GPUArray<T> &copyInto,GPUArray<T> &copyFrom,int block_size=512);
#endif
