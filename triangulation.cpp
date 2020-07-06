#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it, because I'm lazy

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>

#include "noiseSource.h"
#include "DelaunayCGAL.h"
#include "DelaunayGPU.h"
#include "periodicBoundaries.h"
#include "multiProfiler.h"
#include "indexer.h"
#include "cuda_profiler_api.h"

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
    ValueArg<int> maxNeighSwitchArg("m","maxNeighsDefault","default maximum neighbor number for gpu triangulation routine",false,32,"int",cmd);
    ValueArg<int> fileIdxSwitch("f","file","file Index",false,-1,"int",cmd);

    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int fIdx = fileIdxSwitch.getValue();
    int N = nSwitchArg.getValue();
    int maximumIterations = maxIterationsSwitchArg.getValue();
    int maxNeighs = maxNeighSwitchArg.getValue();

    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    bool reproducible = true;

    multiProfiler mProf;

    mProf.start("noise");
    noiseSource noise(reproducible);
    noise.initialize(N);
    noise.initializeGPURNGs();
    mProf.end("noise");

    vector<double2> initialPositions(N);
    double L = sqrt(N);


    //for timing tests, iteratate a random triangulation maximumIterations number of times
    cout << "iterating over " << maximumIterations << " random triangulations of " << N << " points randomly (uniformly) distributed in a square domain"  << endl;
    for (int iteration = 0; iteration<maximumIterations; ++iteration)
        {

        PeriodicBoxPtr domain = make_shared<periodicBoundaries>(L,L);

        vector<pair<Point,int> > pts(N);
        GPUArray<double2> gpuPts((unsigned int) N);
        noise.fillArray(gpuPts,0,L);


        DelaunayCGAL cgalTriangulation;
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
            maxNeighs=0;
            for (int ii = 0; ii < cgalTriangulation.allneighs.size();++ii)
                if(cgalTriangulation.allneighs[ii].size() > maxNeighs)
                    maxNeighs = cgalTriangulation.allneighs[ii].size();
            }//end array handle scope
            {
            ArrayHandle<double2> gps(gpuPts,access_location::device,access_mode::read);
            }


        double cellSize=1.0;
        mProf.start("delGPU total timing");
        DelaunayGPU delGPU(N, maxNeighs+2, cellSize, domain);
        GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs+2)*N);
        GPUArray<int> cellNeighborNumber((unsigned int) N);
        {
        ArrayHandle<double2> gps(gpuPts,access_location::device,access_mode::read);
        ArrayHandle<int> gt(gpuTriangulation,access_location::device,access_mode::read);
        ArrayHandle<int> cnn(cellNeighborNumber,access_location::device,access_mode::read);
        }

        mProf.start("delGPU triangulation");
        cudaProfilerStart();
        delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
        cudaProfilerStop();
        mProf.end("delGPU triangulation");
        mProf.end("delGPU total timing");
        if(programSwitch ==0)
            {
            cout << "testing quality of triangulation..." << endl;
            mProf.start("triangulation comparison");
            compareTriangulation(cgalTriangulation.allneighs, gpuTriangulation,cellNeighborNumber);
            mProf.end("triangulation comparison");
            cout << "... testing done!" << endl;
            };

    }
    cout << endl;
    mProf.print();

//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

    return 0;
};
