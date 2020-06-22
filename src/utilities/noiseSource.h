#ifndef noiseSource_H
#define noiseSource_H

#include <random>
#include "curand.h"
#include "curand_kernel.h"
#include "std_include.h"
#include "gpuarray.h"
#include "noiseSource.cuh"

/*! \file noiseSource.h */
//!A class that gives access to a RNG on the cpu and gpu
/*!
Provides features to some psuedo-rng functions. On the CPU side, one can call for a random integer
(in a specified range), a random real with a uniform distribution, or a random real from a normal
distribution. On the GPU side, provides access to a GPUArray of curandState objects, and functionality to initialize them.
*/
class noiseSource
    {
    public:
        //!base constructor
        noiseSource(bool rep = false)
            {
            Reproducible = rep;
            mt19937 Gener(13377);
        #ifndef DEBUGFLAGUP
            mt19937 GenerRd(rd());
        #else
            mt19937 GenerRd(13377);
        #endif
            gen = Gener;
            genrd=GenerRd;
            }

        //!Get a reproducible integer
        int getInt(int minimum, int maximum);
        //!Get a real from uniform distribution
        double getRealUniform(double minimum =0., double maximum =1.);
        //!Get a real from normal distribution with zero mean and unit var
        double getRealNormalStandardized();
        //!Get a real from normal distribution
        double getRealNormal(double mean =0., double std =1.);

        //!Set the array size of the cuda rngs
        void initialize(int _N)
            {
            N=_N;
            RNGs.resize(N);
            };
        //!set reproducibility
        void setReproducible(bool _rep){Reproducible = _rep;};
        //!set the seed on a reproducible RNG run
        void setReproducibleSeed(int _seed);
        //!should the dynamics be reproducible?
        bool Reproducible;
        //!number of entries for the cuda RNG
        int N;
        //!The seed used by the random number generator, when non-reproducible dynamics have been set
        int RNGSeed;
        //!an initializer for non-reproducible random number generation on the cpu
        random_device rd;
        //!A reproducible Mersenne Twister
        mt19937 gen;
        //!A non-reproducible Mersenne Twister
        mt19937 genrd;
        //!A flag to determine whether the CUDA RNGs should be initialized or not (so that the program will run on systems with no GPU by setting this to false
        bool initializeGPURNG;

        //!allow for whatever GPU RNG initialization is needed
        void initializeGPURNGs(int globalSeed=1337, int tempSeed=0);

        //!An array random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> RNGs;

        //!a normal_distribution with mean zero and unit variance
        normal_distribution<> unitNormal;
    };

#endif
