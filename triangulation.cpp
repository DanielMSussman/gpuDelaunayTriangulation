#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it, because I'm lazy

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>

#include "noiseSource.h"
#include "DelaunayCGAL.h"
#include "DelaunayGPU.h"
#include "periodicBoundaries.h"
#include "profiler.h"

using namespace std;
using namespace TCLAP;

/*!
core of spherical vertexmodel
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
    ValueArg<int> maxIterationsSwitchArg("i","iterations","number of timestep iterations",false,100000,"int",cmd);
    ValueArg<int> fileIdxSwitch("f","file","file Index",false,-1,"int",cmd);
    ValueArg<double> forceToleranceSwitchArg("t","fTarget","target minimization threshold for norm of residual forces",false,0.000000000000001,"double",cmd);
    ValueArg<double> lengthSwitchArg("l","sideLength","size of simulation domain",false,10.0,"double",cmd);
    ValueArg<double> krSwitchArg("k","springRatio","kA divided by kP",false,1.0,"double",cmd);

    //allow setting of system size by either volume fraction or density (assuming N has been set)
    ValueArg<double> p0SwitchArg("p","p0","preferred perimeter",false,3.78,"double",cmd);
    ValueArg<double> a0SwitchArg("a","a0","preferred area per cell",false,1.0,"double",cmd);
    ValueArg<double> rhoSwitchArg("r","rho","density",false,-1.0,"double",cmd);
    ValueArg<double> dtSwitchArg("e","dt","timestep",false,0.001,"double",cmd);
    ValueArg<double> v0SwitchArg("v","v0","v0",false,0.5,"double",cmd);
    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int fIdx = fileIdxSwitch.getValue();
    int N = nSwitchArg.getValue();
    int maximumIterations = maxIterationsSwitchArg.getValue();
    double L = lengthSwitchArg.getValue();
    double rho = rhoSwitchArg.getValue();
    double dt = dtSwitchArg.getValue();
    double v0 = v0SwitchArg.getValue();
    double p0 = p0SwitchArg.getValue();
    double a0 = a0SwitchArg.getValue();
    double kr = krSwitchArg.getValue();
    double forceCutoff = forceToleranceSwitchArg.getValue();

    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    bool reproducible = true;
    noiseSource noise(reproducible);

    vector<double2> initialPositions(N);
    L = sqrt(N);

    PeriodicBoxPtr domain = make_shared<periodicBoundaries>(L,L);
    for (int ii = 0; ii < N; ++ii)
        {
        initialPositions[ii].x=noise.getRealUniform()*L;
        initialPositions[ii].y=noise.getRealUniform()*L;
        }
    cout << N << " points randomly (uniformly) initialized" << endl;

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

    profiler cgalTiming("CGAL");
    cgalTiming.start();
    cgalTriangulation.PeriodicTriangulation(pts,L,0,0,L);
    cgalTiming.end();

    int maxNeighs = 0;
    for (int ii = 0; ii < cgalTriangulation.allneighs.size();++ii)
        if(cgalTriangulation.allneighs[ii].size() > maxNeighs)
            maxNeighs = cgalTriangulation.allneighs[ii].size();
    cout << "max neighbors found by CGAL = " << maxNeighs << endl;



    DelaunayGPU delGPU;
    GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs+2)*N);
    GPUArray<int> cellNeighborNumber((unsigned int) N);
    delGPU.initialize(gpuPts,1.0,N,maxNeighs+2,domain);

    profiler delGPUTiming("DelGPU");
    
    delGPUTiming.start();
    delGPU.GPU_GlobalDelTriangulation(gpuTriangulation,cellNeighborNumber);
    delGPUTiming.end();

    {
    ArrayHandle<int> cnn(cellNeighborNumber);
    int maxNeighs = 0;
    for (int ii = 0; ii < N;++ii)
        if(cnn.data[ii] > maxNeighs)
            maxNeighs = cnn.data[ii];
    cout << "max neighbors found by delGPU = " << maxNeighs << endl;
    }

    cout << endl;
    cgalTiming.print();cout <<endl;
    delGPUTiming.print();

    cout <<endl << "ratio = " << cgalTiming.timeTaken*delGPUTiming.functionCalls / (cgalTiming.functionCalls * delGPUTiming.timeTaken) << endl;
//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

    return 0;
};

