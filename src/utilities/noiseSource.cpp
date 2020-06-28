#define NVCC

#include "noiseSource.h"

/*! \file noiseSource.cpp */

int noiseSource::getInt(int minimum, int maximum)
    {
    int answer;
    uniform_int_distribution<int> uniIntRand(minimum,maximum);
    if (Reproducible)
        answer = uniIntRand(gen);
    else
        answer = uniIntRand(genrd);
    return answer;
    };

double noiseSource::getRealUniform(double minimum, double maximum)
    {
    double answer;
    uniform_real_distribution<double> uniRealRand(minimum,maximum);
    if (Reproducible)
        answer = uniRealRand(gen);
    else
        answer = uniRealRand(genrd);
    return answer;
    };

double noiseSource::getRealNormalStandardized()
    {
    double answer;
    if (Reproducible)
        answer = unitNormal(gen);
    else
        answer = unitNormal(genrd);
    return answer;
    };

double noiseSource::getRealNormal(double mean, double std)
    {
    double answer;
    normal_distribution<> normal(mean,std);
    if (Reproducible)
        answer = normal(gen);
    else
        answer = normal(genrd);
    return answer;
    };

void noiseSource::fillArray(GPUArray<double2> &array, double min, double max)
    {
    ArrayHandle<double2> dat(array,access_location::device,access_mode::overwrite);
    ArrayHandle<curandState> d_curandRNGs(RNGs,access_location::device,access_mode::readwrite);

    gpu_fill_double2_array_uniform(d_curandRNGs.data,dat.data,min,max,array.getNumElements());
    }

/*!
\param globalSeed the global seed to use
\param offset the value of the offset that should be sent to the cuda RNG...
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void noiseSource::initializeGPURNGs(int globalSeed,int tempSeed)
    {
    if(RNGs.getNumElements() != N)
        RNGs.resize(N);
    ArrayHandle<curandState> d_curandRNGs(RNGs,access_location::device,access_mode::overwrite);
    int globalseed = globalSeed;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        RNGSeed = globalseed;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_RNG_array(d_curandRNGs.data,N,tempSeed,globalseed);
    };

void noiseSource::setReproducibleSeed(int _seed)
    {
    RNGSeed = _seed;
    Reproducible = true;
    mt19937 Gener(RNGSeed);
    gen = Gener;
#ifdef DEBUGFLAGUP
    mt19937 GenerRd(RNGSeed);
    genrd=GenerRd;
#endif
    };

