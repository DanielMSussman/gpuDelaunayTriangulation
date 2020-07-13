#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it, because I'm lazy

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>

#include "noiseSource.h"
#include "DelaunayCGAL.h"
#include "DelaunayGPU.h"
#include "periodicBoundaries.h"
#include "multiProfiler.h"
#include "profiler.h"
#include "indexer.h"
#include "cuda_profiler_api.h"
#include "gpuUtilities.cuh"

using namespace std;
using namespace TCLAP;


//! simple output of int vecs
void printVec(vector<int> &a)
    {
    for (int ii = 0; ii < a.size();++ii)
        cout<< a[ii] << "\t";
    cout << endl;
    }

//! Compare neighbors found by different algorithsm. vector<vector< int> > a la CGAL, vs GPUArray<int> together with another GPUArray<int> for number of neighbors each cell has
void compareTriangulation(vector<vector<int> > &neighs, GPUArray<int> &gNeighs, GPUArray<int> &gNeighborNum)
    {
    int N = neighs.size();
    int neighMax = gNeighs.getNumElements() / N;
    Index2D nIdx(neighMax,N);
    ArrayHandle<int> gnn(gNeighborNum);
    ArrayHandle<int> gn(gNeighs);
    //test every neighbor list
    for(int ii = 0; ii < N; ++ii)
        {
        vector<int> gpuNeighbors(gnn.data[ii]);
        vector<int> cpuNeighbors = neighs[ii];

        for (int jj = 0; jj < gnn.data[ii];++jj)
            gpuNeighbors[jj] = gn.data[nIdx(jj,ii)];

        //prepare to compare vector of neighbors by rotating the gpuNeighborlist as appropriate
        bool neighborsMatch = true;
        vector<int>::iterator it;
        it = std::find(gpuNeighbors.begin(),gpuNeighbors.end(),cpuNeighbors[0]);
        if (it == gpuNeighbors.end() || gpuNeighbors.size() != cpuNeighbors.size())
            {
            neighborsMatch = false;
            }
        else
            {
            int rotatePos = it - gpuNeighbors.begin();
            std::rotate(gpuNeighbors.begin(),gpuNeighbors.begin()+rotatePos, gpuNeighbors.end());
            for (int jj = 0; jj < gpuNeighbors.size(); ++jj)
                {
                if (gpuNeighbors[jj] != cpuNeighbors[jj])
                    neighborsMatch = false;
                }
            }

        if (neighborsMatch == false)
            {
            cout << " cell " << ii << " failed!";
            printVec(gpuNeighbors);
            printVec(neighs[ii]);
            }
        }
    };
/*!
Testing shell for gpuDelaunayTriangulation
*/
int main(int argc, char*argv[])
{
    // wrap tclap in a try block
    try
    {
    //First, we set up a basic command line parser...
    //      cmd("command description message", delimiter, version string)
    CmdLine cmd("basic testing of gpuDelaunayTriangulation", ' ', "V0.1");

    //define the various command line strings that can be passed in...
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,"value type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","USEGPU","an integer controlling which gpu to use... g < 0 uses the cpu",false,-1,"int",cmd);
    ValueArg<int> nSwitchArg("n","Number","number of particles in the simulation",false,100,"int",cmd);
    ValueArg<int> maxIterationsSwitchArg("i","iterations","number of timestep iterations",false,1,"int",cmd);
    ValueArg<int> maxNeighSwitchArg("m","maxNeighsDefault","default maximum neighbor number for gpu triangulation routine",false,16,"int",cmd);
    ValueArg<int> fileIdxSwitch("f","file","file Index",false,-1,"int",cmd);
    ValueArg<int> safetyModeSwitch("s","safetyMode","0 is false, anything else is true", false,0,"int",cmd);
    ValueArg<double> localTestSwitch("l","localFraction","fraction of points to repair", false,0.05,"int",cmd);
    ValueArg<double> cellListSizeSwitch("c","cellSize","linear dimension of cell list", false,1.0,"int",cmd);

    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int fIdx = fileIdxSwitch.getValue();
    int N = nSwitchArg.getValue();
    int maximumIterations = maxIterationsSwitchArg.getValue();
    int maxNeighs = maxNeighSwitchArg.getValue();
    
    bool safetyMode  = safetyModeSwitch.getValue() ==0 ? false : true;

    double localFraction = localTestSwitch.getValue();
    double cellSize = cellListSizeSwitch.getValue();

    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    bool reproducible = true;

    multiProfiler mProf;

    mProf.start("noise initialize");
    noiseSource noise(reproducible);
    noise.initialize(N);
    noise.initializeGPURNGs();
    mProf.end("noise initialize");

    vector<double2> initialPositions(N);
    double L = sqrt(N);
    PeriodicBoxPtr domain = make_shared<periodicBoundaries>(L,L);

    vector<pair<Point,int> > pts(N);
    GPUArray<double2> gpuPts((unsigned int) N);

if(programSwitch >=0) //global tests
    {
    profiler prof("triangulation");
    profiler prof2("total");
    //for timing tests, iteratate a random triangulation maximumIterations number of times
    cout << "iterating over " << maximumIterations << " random triangulations of " << N << " points randomly (uniformly) distributed in a square domain"  << endl;
    mProf.addName("triangulation comparison");
    mProf.addName("delGPU total timing");
    mProf.addName("delGPU triangulation");
    mProf.addName("delGPU cellList");
    mProf.addName("CGAL triangulation");
    mProf.addName("generate points");
    mProf.addName("delGPU initialization");

    mProf.start("delGPU initialization");
    DelaunayGPU delGPU(N, maxNeighs, cellSize, domain);
    GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs)*N);
    GPUArray<int> cellNeighborNumber((unsigned int) N);
    {
    ArrayHandleAsync<double2> gps(gpuPts,access_location::device,access_mode::readwrite);
    ArrayHandleAsync<int> gt(gpuTriangulation,access_location::device,access_mode::readwrite);
    ArrayHandleAsync<int> cnn(cellNeighborNumber,access_location::device,access_mode::readwrite);
    }
    /*
        If true, will successfully rescue triangulation even if maxNeighs is too small.
        this is currently very slow (can be improved a lot), and will be once 
        other optimizations are done
    */
    delGPU.setSafetyMode(safetyMode);
    mProf.end("delGPU initialization");
    DelaunayCGAL cgalTriangulation;

    //When built in non-debug mode, the first iteration is much slower than subsequent ones... just time later ones
    for (int iteration = 0; iteration<maximumIterations; ++iteration)
        {
        mProf.start("generate points");
        noise.fillArray(gpuPts,0.,L);
        mProf.end("generate points");

        if(programSwitch == 0)
            {
            ArrayHandle<double2> gps(gpuPts,access_location::host,access_mode::read);
            for (int ii = 0; ii < N; ++ii)
                {
                pts[ii]=make_pair(Point(gps.data[ii].x,gps.data[ii].y),ii);
                }
                mProf.start("CGAL triangulation");
            cgalTriangulation.PeriodicTriangulation(pts,L,0,0,L);
                mProf.end("CGAL triangulation");
            }//end CGAL test

        cudaProfilerStart();
        prof2.start();
        mProf.start("delGPU total timing");
        mProf.start("delGPU cellList");
        delGPU.updateList(gpuPts);
        mProf.end("delGPU cellList");
        mProf.start("delGPU triangulation");
        prof.start();

        delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
        
        prof.end();
        cudaProfilerStop();
        mProf.end("delGPU triangulation");
        prof2.end();
        mProf.end("delGPU total timing");
        if(programSwitch ==0)
            {
            cout << "testing quality of triangulation..." << endl;
            mProf.start("triangulation comparison");
            compareTriangulation(cgalTriangulation.allneighs, gpuTriangulation,cellNeighborNumber);
            mProf.end("triangulation comparison");
            cout << "... testing done!" << endl;
            }
        }//end loop over iterations
    cout << endl;
    mProf.print();
    prof.printVec();
    prof2.printVec();
    }//end global tests
else if (programSwitch == -1)
    {
    vector<int> repairList(N);
    for(int ii = 0; ii < N; ++ii)
        repairList[ii]=ii;
    int localNumber = floor(localFraction*N);
    noise.fillArray(gpuPts,0,L);
    DelaunayGPU delGPU(N, maxNeighs, cellSize, domain);
    delGPU.setSafetyMode(safetyMode);
    GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs)*N);
    GPUArray<int> cellNeighborNumber((unsigned int) N);
    {
    ArrayHandle<double2> p(gpuPts,access_location::device,access_mode::readwrite);
    ArrayHandle<int> t(gpuTriangulation,access_location::device,access_mode::readwrite);
    ArrayHandle<int> c(cellNeighborNumber,access_location::device,access_mode::readwrite);
    }
    /*
    mProf.start("initial global triangulation");
    delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
    mProf.end("initial global triangulation");
    mProf.start("second global triangulation");
    delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
    mProf.end("second global triangulation");
    mProf.start("third global triangulation");
    delGPU.cListUpdated=false;
    delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
    mProf.end("third global triangulation");
    printf("Initial global triangulation done\n");
    */
    printf("randomly repairing %i points (out of %i) %i times\n",localNumber,N,maximumIterations);
    for (int iteration = 0; iteration<maximumIterations; ++iteration)
        {
        std::random_shuffle(repairList.begin(),repairList.end());
        //GPUArray<int> setRep((unsigned int) N+1);
        GPUArray<int> setRep((unsigned int) N);
        {
        ArrayHandle<int> sr(setRep);
        for (int ii =0; ii < setRep.getNumElements(); ++ii)
            {
            sr.data[ii]=-1;
            }
        for (int ii =0; ii < localNumber; ++ii)
            {
            //sr.data[ii]=repairList[ii];
            sr.data[repairList[ii]]=repairList[ii];
            }
        }
        {
        ArrayHandle<int> sr(setRep,access_location::device,access_mode::readwrite);
        }
        mProf.start("delGPU local triangulation");
        cudaProfilerStart();
            mProf.start("delGPU cellList");
            delGPU.updateList(gpuPts);
            mProf.end("delGPU cellList");
        delGPU.locallyRepairDelaunayTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber,setRep,localNumber);
        cudaProfilerStop();
        mProf.end("delGPU local triangulation");
        }
    cout << endl;
    mProf.print();
    }
else
    {
    vector<int> repairList(N);
    for(int ii = 0; ii < N; ++ii)
        repairList[ii]=ii;
    int localNumber = floor(localFraction*N);
    printf("randomly repairing %i points (out of %i) %i times\n",localNumber,N,maximumIterations);
    noise.fillArray(gpuPts,0,L);
    DelaunayGPU delGPU(N, maxNeighs, cellSize, domain);
    delGPU.setSafetyMode(safetyMode);
    GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs)*N);
    GPUArray<int> cellNeighborNumber((unsigned int) N);
    delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
    printf("Initial global triangulation done\n");

    for (int iteration = 0; iteration<=maximumIterations; ++iteration)
        {
    
        GPUArray<double2> randomDisplacement((unsigned int) N);
        noise.fillArray(randomDisplacement,-localFraction,localFraction);
        gpu_add_gpuarray<double2>(gpuPts,randomDisplacement,N);

        {
        ArrayHandle<double2> p(gpuPts);
        for(int ii = 0; ii < N; ++ii)
            {
            delGPU.Box->putInBoxReal(p.data[ii]);
            };
        }
        {
        ArrayHandle<double2> p(gpuPts,access_location::device,access_mode::readwrite);
        }



        mProf.start("delGPU local triangulation");
        cudaProfilerStart();
        delGPU.getCircumcenters(gpuTriangulation,cellNeighborNumber);
        delGPU.GPU_LocalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
        cudaProfilerStop();
        mProf.end("delGPU local triangulation");
        }
    cout << endl;
    mProf.print();
    }

//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

    return 0;
};
