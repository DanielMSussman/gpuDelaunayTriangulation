#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it, because I'm lazy

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>

#include "noiseSource.h"
#include "DelaunayCGAL.h"
#include "DelaunayGPU.h"
#include "periodicBoundaries.h"
#include "profiler.h"
#include "multiProfiler.h"
#include "indexer.h"
#include <ctime>

#include <algorithm>
#include "hilbert_curve.hpp"

using namespace std;
using namespace TCLAP;


//! simple output of int vecs
void printVec(vector<int> &a)
    {
    for (int ii = 0; ii < a.size();++ii)
        cout<< a[ii] << "\t";
    cout << endl;
    }

void squareLattice(GPUArray<double2> &A, noiseSource &noise, int N)
    {
    cout << "initializing in generic square lattice..." << endl;
    ArrayHandle<double2> a(A);
    int xb = ceil(sqrt((double) N));
    double Lscale = sqrt((double) N)/(double) xb;
    double phase = 0.25;
    int ii =0;
    for (int x = 0; x < xb; ++x)
        for (int y = 0; y < xb; ++y)
            {
            double p1 = noise.getRealUniform(Lscale*(x+.5-phase),Lscale*(x+.5+phase));
            double p2 = noise.getRealUniform(Lscale*(y+.5-phase),Lscale*(y+.5+phase));
            if(ii < N)
                a.data[ii] = make_double2(p1,p2);
            ii +=1;
            }
    {
    ArrayHandle<double2> a(A,access_location::device,access_mode::readwrite);
    }
    cout <<  ii << " ...done" << endl;
    }


//! sort points along a hilbert curve
void hilbertSortArray(GPUArray<double2> &A)
    {
    cout << "hilbert sorting...";
    int N = A.getNumElements();
    {
    ArrayHandle<double2> a(A);
    vector<pair<int,double2> > p(N);
    for(int ii =0; ii < N; ++ii)
        {
        double x = a.data[ii].x;
        double y = a.data[ii].y;
        int hilbertM = 30;
        int hilbertIndex =xy2d(hilbertM,x,y);
        p[ii] = make_pair(hilbertIndex,a.data[ii]);
        }
    std::sort(p.begin(),p.end());
    for(int ii =0; ii < N; ++ii)
        {
        a.data[ii] = p[ii].second;
        }
    }
    {
    ArrayHandle<double2> a(A,access_location::device,access_mode::readwrite);
    }
    cout << "...done" << endl;
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
    ValueArg<int> gpuSwitchArg("g","USEGPU","an integer controlling which gpu to use... g < 0 uses the cpu",false,0,"int",cmd);
    ValueArg<int> nSwitchArg("n","Number","number of particles in the simulation",false,100,"int",cmd);
    ValueArg<int> maxIterationsSwitchArg("i","iterations","number of timestep iterations",false,4,"int",cmd);
    ValueArg<int> maxNeighSwitchArg("m","maxNeighsDefault","default maximum neighbor number for gpu triangulation routine",false,16,"int",cmd);
    ValueArg<int> fileIdxSwitch("f","file","file Index",false,-1,"int",cmd);
    ValueArg<int> safetyModeSwitch("s","safetyMode","0 is false, anything else is true", false,0,"int",cmd);
    ValueArg<double> cellListSizeSwitch("c","cellSize","linear dimension of cell list", false,1.0,"int",cmd);

    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int fIdx = fileIdxSwitch.getValue();
    int N = nSwitchArg.getValue();
    int maximumIterations = maxIterationsSwitchArg.getValue();
    int maxNeighs = maxNeighSwitchArg.getValue();

    double cellSize = cellListSizeSwitch.getValue();
    bool safetyMode  = safetyModeSwitch.getValue() ==0 ? false : true;

    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    bool reproducible = true;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuSwitch);


    vector<int> ns;
    for (int p2 = 7; p2 < 22; ++p2)
        ns.push_back(pow(2,p2));
    time_t now = time(0);
    tm *ltm = localtime(&now);
    char fname[256];
    sprintf(fname,"timing_gpu%i_dateTime_%i_%i_%i.txt",gpuSwitch,ltm->tm_mon+1,ltm->tm_mday,ltm->tm_hour);

    for(int nn = 0; nn < ns.size();++nn)
    {
    multiProfiler mProf;

    mProf.addName("delGPU total timing");
    mProf.addName("delGPU triangulation");

    N=ns[nn];
    if(N > 500000) maxNeighs = 23;
    noiseSource noise(reproducible);
    noise.initialize(N);
    noise.initializeGPURNGs();

    vector<double2> initialPositions(N);
    double L = sqrt(N);
    PeriodicBoxPtr domain = make_shared<periodicBoundaries>(L,L);

    profiler delGPUtotalTiming("DelGPUtotal");
    profiler delGPUTiming("DelGPU");
    profiler cgalTiming("CGAL");

    //for timing tests, iteratate a random triangulation maximumIterations number of times
    cout << "iterating over " << maximumIterations << " random triangulations of " << N << " points randomly (uniformly) distributed in a square domain"  << endl;
    vector<pair<Point,int> > pts(N);
    GPUArray<double2> gpuPts((unsigned int) N);
    GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs)*N);
    GPUArray<int> cellNeighborNumber((unsigned int) N);
    DelaunayGPU delGPU(N, maxNeighs, cellSize, domain);
    /*
       If true, will successfully rescue triangulation even if maxNeighs is too small.
       this is currently very slow (can be improved a lot), and will be once 
       other optimizations are done
     */
    {
    ArrayHandleAsync<double2> gps(gpuPts,access_location::device,access_mode::readwrite);
    ArrayHandleAsync<int> gt(gpuTriangulation,access_location::device,access_mode::readwrite);
    ArrayHandleAsync<int> cnn(cellNeighborNumber,access_location::device,access_mode::readwrite);
    }
    delGPU.setSafetyMode(safetyMode);
    DelaunayCGAL cgalTriangulation;

    for (int iteration = 0; iteration<maximumIterations; ++iteration)
        {
        cout << "iteration "<< iteration << endl << std::flush;

        cout << "\tcreating points..." << std::flush;
        if(programSwitch <=1)
            noise.fillArray(gpuPts,0.,L);
        else if (programSwitch <=3)
            {
            noise.fillArray(gpuPts,0.,L);
            hilbertSortArray(gpuPts);
            }
        else
            {
            squareLattice(gpuPts,noise,N);
            hilbertSortArray(gpuPts);
            }
        cout << "...done" << endl<< std::flush;
cudaDeviceSynchronize();
        if(programSwitch%2 ==0)
            {
            cout << "\ttriangulating via cgal ..." << std::flush;
            ArrayHandle<double2> gps(gpuPts,access_location::host,access_mode::read);
            for (int ii = 0; ii < N; ++ii)
            {
                pts[ii]=make_pair(Point(gps.data[ii].x,gps.data[ii].y),ii);
            }
            cgalTiming.start();
            cgalTriangulation.PeriodicTriangulation(pts,L,0,0,L);
            cgalTiming.end();
            cout << "...done " <<endl << std::flush;
            }

        cout << "\ttriangulating via delGPU..." << std::flush;
        mProf.start("delGPU total timing");
        delGPUtotalTiming.start();

        delGPU.updateList(gpuPts);
        delGPUTiming.start();//profile just the triangulation routine
        mProf.start("delGPU triangulation");
        delGPU.GPU_GlobalDelTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
        cudaDeviceSynchronize();
        mProf.end("delGPU triangulation");
        mProf.end("delGPU total timing");
        delGPUTiming.end();
        delGPUtotalTiming.end();
        cout << "...done " <<endl << std::flush;
        if(programSwitch%2 ==0)
            {
            cout << "\ttesting quality of triangulation..." << std::flush;
            compareTriangulation(cgalTriangulation.allneighs, gpuTriangulation,cellNeighborNumber);
            cout << "\t... testing done!" << endl << std::flush;
            };

    }
    cout << endl;
    delGPUTiming.print();
    delGPUtotalTiming.print();
    mProf.print();
    double ratio = 0;
    if(programSwitch%2 ==0)
        {
        ratio = cgalTiming.timeTaken*delGPUTiming.functionCalls / (cgalTiming.functionCalls * delGPUTiming.timeTaken);
        cgalTiming.print();
        cout <<endl;
        cout <<endl << "ratio = " << ratio << endl;
        }

    ofstream outfile;
    outfile.open(fname, std::ofstream::out | std::ofstream::app);
    outfile << N << "\t" 
                << delGPUTiming.timeTaken/delGPUTiming.functionCalls << "\t"
                << delGPUtotalTiming.timeTaken/delGPUtotalTiming.functionCalls << "\t"
                << ratio << "\n";
    outfile.close();
    }

//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

    return 0;
};
