#ifndef STDINCLUDE_H
#define STDINCLUDE_H

/*! \file std_include.h
a file to be included all the time... carries with it things DMS often uses.
It includes some handy debugging / testing functions, and includes too many
standard library headers
*/

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE __attribute__((always_inline)) inline
#endif

#define THRESHOLD 1e-18
#define EPSILON 1e-18

#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>
#include <string.h>
#include <stdexcept>
#include <cassert>
#include <vector>
#include <omp.h>
#include <thread>
#include <cpuid.h>

using namespace std;

#include <cuda_runtime.h>
#include "vector_types.h"
#include "vector_functions.h"
#include "nvToolsExt.h"
#include "vectorTypeOperations.h"
//double precision value of pi
#define PI 3.141592653589793115997963468544185161590576171875

#define cur_norm curand_normal_double

//texture load templating correctly for old cuda devices and host functions
template<typename T>
HOSTDEVICE T ldgHD(const T* ptr)
    {
    #if __CUDA_ARCH__ >=350
        return __ldg(ptr);
    #else
        return *ptr;
    #endif
    }

/*!
omp Template: loop over the function with omp or not
the syntax requires that the first argument of the function is the "index" of whatever the function acts on.
So, if the function is f(int, double, double,Index2D,...) then this template function should be called by:
ompFunctionLoop(ompThreadNum,maxIdx, f, double, double,Index2D,...).
*/
template< typename... Args>
void ompFunctionLoop(int nThreads, int maxIdx, void (*fPointer)(int, Args...), Args... args)
    {
    if(nThreads == 1)
        {
        for(int idx = 0; idx < maxIdx; ++idx)
            fPointer(idx,args...);
        }
    else
        {
	    #pragma omp parallel for num_threads(nThreads)
        for(int idx = 0; idx < maxIdx; ++idx)
            fPointer(idx,args...);
        }
    };

/*

template<typename... Args>
void ompFunctionLoop(int nThreads, int maxIdx)
    {
    if(nThreads == 1)
        {
        for(int idx = 0; idx < maxIdx; ++idx)
            T(idx,args...);
        }
    else
        {
	    #pragma omp parallel for num_threads(ompThreadNum)
        for(int idx = 0; idx < maxIdx; ++idx)
            T(idx,args...);
        }
    }
*/
//!Handle errors in kernel calls...returns file and line numbers if cudaSuccess doesn't pan out
static void HandleError(cudaError_t err, const char *file, int line)
    {
    //as an additional debugging check, synchronize cuda threads after every kernel call
    #ifdef DEBUGFLAGUP
    cudaDeviceSynchronize();
    #endif
    if (err != cudaSuccess)
        {
        printf("\nError: %s in file %s at line %d\n",cudaGetErrorString(err),file,line);
        throw std::exception();
        }
    }

//!Report somewhere that code needs to be written
static void unwrittenCode(const char *message, const char *file, int line)
    {
    printf("\nCode unwritten (file %s; line %d)\nMessage: %s\n",file,line,message);
    throw std::exception();
    }

//!A utility function for checking if a file exists
inline bool fileExists(const std::string& name)
    {
    ifstream f(name.c_str());
    return f.good();
    }

__host__ inline bool chooseCPU(int gpuSwitch,bool verbose = false)
    {
	char CPUBrandString[0x40];
	unsigned int CPUInfo[4] = {0,0,0,0};
	__cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
	unsigned int nExIds = CPUInfo[0];

	memset(CPUBrandString, 0, sizeof(CPUBrandString));

	for (unsigned int i = 0x80000000; i <= nExIds; ++i)
		{
    		__cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);

    		if (i == 0x80000002)
	        	memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000003)
			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000004)
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
		}

    if(verbose)
        cout << "using "<<CPUBrandString <<"     Available threads: "<< std::thread::hardware_concurrency() <<"     Threads requested: "<<abs(gpuSwitch) <<"\n"<< endl;
    else
        cout << "Running on the CPU with " << abs(gpuSwitch) << " openMP-based threads" << endl;
    return false;
    }

//!Get basic stats about the chosen GPU (if it exists)
__host__ inline bool chooseGPU(int USE_GPU,bool verbose = false)
    {
    if(USE_GPU < 0)
        {
        return chooseCPU(abs(USE_GPU),true);
        }
    int nDev;
    cudaGetDeviceCount(&nDev);
    if (USE_GPU >= nDev)
        {
        cout << "Requested GPU (device " << USE_GPU<<") does not exist. Stopping triangulation" << endl;
        return false;
        };
    if (USE_GPU <nDev)
        cudaSetDevice(USE_GPU);
    if(verbose)    cout << "Device # \t\t Device Name \t\t MemClock \t\t MemBusWidth" << endl;
    for (int ii=0; ii < nDev; ++ii)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,ii);
        if (verbose)
            {
            if (ii == USE_GPU) cout << "********************************" << endl;
            if (ii == USE_GPU) cout << "****Using the following gpu ****" << endl;
            cout << ii <<"\t\t\t" << prop.name << "\t\t" << prop.memoryClockRate << "\t\t" << prop.memoryBusWidth << endl;
            if (ii == USE_GPU) cout << "*******************************" << endl;
            };
        };
    if (!verbose)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,USE_GPU);
        cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << endl << endl;
        };
    return true;
    };

//A macro to wrap cuda calls
#define HANDLE_ERROR(err) (HandleError( err, __FILE__,__LINE__ ))
//A macro to say code needs to be written
#define UNWRITTENCODE(message) (unwrittenCode(message,__FILE__,__LINE__))
//spot-checking of code for debugging
#define DEBUGCODEHELPER printf("\nReached: file %s at line %d\n",__FILE__,__LINE__);

#undef HOSTDEVICE
#endif
