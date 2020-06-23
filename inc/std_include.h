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
#define HOSTDEVICE inline __attribute__((always_inline))
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

using namespace std;

#include <cuda_runtime.h>
#include "vector_types.h"
#include "vector_functions.h"
#include "nvToolsExt.h"
#include "vectorTypeOperations.h"
//double precision value of pi
#define PI 3.141592653589793115997963468544185161590576171875

#define cur_norm curand_normal_double

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
//!Get basic stats about the chosen GPU (if it exists)
__host__ inline bool chooseGPU(int USE_GPU,bool verbose = false)
    {
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
