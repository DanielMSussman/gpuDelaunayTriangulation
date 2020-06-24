#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it, because I'm lazy

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>

#include "noiseSource.h"
#include "DelaunayCGAL.h"
#include "DelaunayGPU.h"
#include "periodicBoundaries.h"
#include "profiler.h"
#include "indexer.h"
#include <ctime>

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
    noiseSource noise(reproducible);



    vector<int> ns;
    ns.push_back(100);
    ns.push_back(200);
    ns.push_back(400);
    ns.push_back(800);
    ns.push_back(1600);
    ns.push_back(3200);
    ns.push_back(6400);
    ns.push_back(12800);
    ns.push_back(25000);
    ns.push_back(50000);
    ns.push_back(100000);
    ns.push_back(200000);
    ns.push_back(400000);
    time_t now = time(0);
    tm *ltm = localtime(&now);
    char fname[256];
    sprintf(fname,"timing_%i_%i_%i.txt",ltm->tm_mon,ltm->tm_mday,ltm->tm_hour);

    ofstream outfile;
    outfile.open(fname);
    for(int nn = 0; nn < ns.size();++nn)
    {
    N=ns[nn];

    vector<double2> initialPositions(N);
    double L = sqrt(N);

    profiler delGPUtotalTiming("DelGPUtotal");
    profiler delGPUTiming("DelGPU");
    profiler cgalTiming("CGAL");

    //for timing tests, iteratate a random triangulation maximumIterations number of times
    cout << "iterating over " << maximumIterations << " random triangulations of " << N << " points randomly (uniformly) distributed in a square domain"  << endl;
    for (int iteration = 0; iteration<maximumIterations; ++iteration)
        {

        PeriodicBoxPtr domain = make_shared<periodicBoundaries>(L,L);
        for (int ii = 0; ii < N; ++ii)
            {
            initialPositions[ii].x=noise.getRealUniform()*L;
            initialPositions[ii].y=noise.getRealUniform()*L;
            }

        DelaunayCGAL cgalTriangulation;
        vector<pair<Point,int> > pts(N);
        GPUArray<double2> gpuPts((unsigned int) N);
        {
        ArrayHandle<double2> gps(gpuPts);
        for (int ii = 0; ii < N; ++ii)
            {
            pts[ii]=make_pair(Point(initialPositions[ii].x,initialPositions[ii].y),ii);
            gps.data[ii] = initialPositions[ii];
            }
        }//end array handle scope

        if(programSwitch ==0)
            {
            cgalTiming.start();
            cgalTriangulation.PeriodicTriangulation(pts,L,0,0,L);
            cgalTiming.end();
            maxNeighs=0;
            for (int ii = 0; ii < cgalTriangulation.allneighs.size();++ii)
                if(cgalTriangulation.allneighs[ii].size() > maxNeighs)
                    maxNeighs = cgalTriangulation.allneighs[ii].size();
            }

        double cellSize=1.0;
        delGPUtotalTiming.start();//include initialization and data transfer times
        DelaunayGPU delGPU(N, maxNeighs+2, cellSize, domain);
        GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs+2)*N);
        GPUArray<int> cellNeighborNumber((unsigned int) N);
        {
        ArrayHandle<double2> gps(gpuPts,access_location::device,access_mode::read);
        ArrayHandle<int> gt(gpuTriangulation,access_location::device,access_mode::read);
        ArrayHandle<int> cnn(cellNeighborNumber,access_location::device,access_mode::read);
        }

        delGPUTiming.start();//profile just the triangulation routine
        delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
        delGPUTiming.end();
        delGPUtotalTiming.end();
        if(programSwitch ==0)
            {
            cout << "testing quality of triangulation..." << endl;
            compareTriangulation(cgalTriangulation.allneighs, gpuTriangulation,cellNeighborNumber);
            cout << "... testing done!" << endl;
            };

    
    }
    cout << endl;
    delGPUTiming.print();
    delGPUtotalTiming.print();
    if(programSwitch==0)
        {
        cgalTiming.print();
        cout <<endl;
        cout <<endl << "ratio = " << cgalTiming.timeTaken*delGPUTiming.functionCalls / (cgalTiming.functionCalls * delGPUTiming.timeTaken) << endl;
        }

    outfile << N << "\t" << delGPUTiming.timeTaken/delGPUTiming.functionCalls << "\n";
    }
    outfile.close();

//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

    return 0;
};
