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
#include "utilities.cuh"

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

//To test the triangulation speed of more regular point patterns, initialize in a randomized square lattice
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

//resort points according to their order along a hilbert curve
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

// Compare neighbors found by different algorithsm. vector<vector< int> > a la CGAL, vs GPUArray<int> together with another GPUArray<int> for number of neighbors each cell has
void compareTriangulation(vector<vector<int> > &neighs, GPUArray<int> &gNeighs, GPUArray<int> &gNeighborNum, bool verbose = true)
    {
    int failedCells = 0;
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
            if(verbose)
                {
                cout << " cell " << ii << " failed!";
                printVec(gpuNeighbors);
                printVec(neighs[ii]);
                }
            failedCells += 1;
            }
        }
    printf("total number of mismatched cells = %i\n",failedCells);
    };

/*
Testing shell for gpuDelaunayTriangulation. the -z command line flag calls various code branches:
-z 0 : compute a triangulation of a point set drawn at random (uniformly) in a square domain. Compare the result with CGAL
-z 1 : same as -z 0, but don't perform a CGAL triangulation and compare
-z 2 : same as -z 0, but perform a hilbert sort of the point set first
-z 3 : same as -z 2, but don't perform a CGAL triangulation and compare
-z 4 : same as -z 0, but choose points to be a distorted square lattice and perform a hilbert sort of the point set first
-z 5 : same as -z 4, but don't perform a CGAL triangulation and compare

-z -2 : check the testAndRepair function by (a) globally triangulating a point set, (b) randomly moving some of the points, and (c) calling the test and repair function. Compare with the results of a CGAL triangulation at each step
-z -1 : same as -z -2, except don't do any CGAL work or checking
*/
int main(int argc, char*argv[])
{
    // wrap tclap in a try block
    try
    {
    CmdLine cmd("basic testing of gpuDelaunayTriangulation", ' ', "V0.1");

    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","USEGPU","an integer controlling which gpu to use... g < 0 uses the cpu",false,0,"int",cmd);
    ValueArg<int> nSwitchArg("n","Number","number of particles in the simulation",false,100,"int",cmd);
    ValueArg<int> maxIterationsSwitchArg("i","iterations","number of timestep iterations",false,1,"int",cmd);
    ValueArg<int> maxNeighSwitchArg("m","maxNeighsDefault","default maximum neighbor number for gpu triangulation routine",false,16,"int",cmd);
    ValueArg<int> fileIdxSwitch("f","file","file Index",false,-1,"int",cmd);
    ValueArg<int> safetyModeSwitch("s","safetyMode","0 is false, anything else is true", false,0,"int",cmd);
    ValueArg<double> localTestSwitch("l","localFraction","fraction of points to repair", false,0.05,"int",cmd);
    ValueArg<double> cellListSizeSwitch("c","cellSize","linear dimension of cell list", false,1.0,"int",cmd);

    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int fIdx = fileIdxSwitch.getValue();
    int N = nSwitchArg.getValue();
    int maximumIterations = maxIterationsSwitchArg.getValue();
    int maxNeighs = maxNeighSwitchArg.getValue();
    int gpuSwitch = gpuSwitchArg.getValue();
    bool safetyMode  = safetyModeSwitch.getValue() ==0 ? false : true;
    double localFraction = localTestSwitch.getValue();
    double cellSize = cellListSizeSwitch.getValue();

    bool GPU = false;
    //chooseGPU will call chooseCPU if gpuSwitch < 0
    GPU = chooseGPU(gpuSwitch);

    bool reproducible = true;//set to false if you want the RNG to be different each time. reproducibility for testing the code, though

    multiProfiler mProf;//a few lightweigt profilers for various rough timings
    profiler prof("triangulation");
    profiler prof2("total");

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
    //for timing tests, iteratate a random triangulation maximumIterations number of times
    cout << "iterating over " << maximumIterations << " random triangulations of " << N << " points distributed in a square domain"  << endl;
    mProf.addName("triangulation comparison");
    mProf.addName("delGPU total timing");
    mProf.addName("delGPU triangulation");
    mProf.addName("delGPU cellList");
    mProf.addName("CGAL triangulation");
    mProf.addName("generate points");
    mProf.addName("delGPU initialization");

    mProf.start("delGPU initialization");
    //maxNeighs is a guess for the maximum number of neighbors any point will have; this controls the size of various data structures in the delGPU class
    DelaunayGPU delGPU(N, maxNeighs, cellSize, domain);
    GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs)*N);
    GPUArray<int> cellNeighborNumber((unsigned int) N);
    {
    ArrayHandleAsync<double2> gps(gpuPts,access_location::device,access_mode::readwrite);
    ArrayHandleAsync<int> gt(gpuTriangulation,access_location::device,access_mode::readwrite);
    ArrayHandleAsync<int> cnn(cellNeighborNumber,access_location::device,access_mode::readwrite);
    }
    /*
    If safetyMode is true, the code will check whether any point has more neighbors than maxNeighs, and,
    if so, will enlarge the data structures and recompute the triangulation.
    This comes at the expense of some (small) device-to-host transfers, so if for some reason you have a 
    guarantee that maxNeighs will not be exceeded, set to true
    */
    delGPU.setSafetyMode(safetyMode);

    if(gpuSwitch>=0)
        delGPU.setGPUcompute(true);
    else
        {	
    	delGPU.setGPUcompute(false);
	    delGPU.setOMPthreads(abs(gpuSwitch));
        }
    mProf.end("delGPU initialization");

    DelaunayCGAL cgalTriangulation;
//these cudaDeviceSynchronize() calls are not needed, and are here for more accurate timing of the various components of the triangulating process
cudaDeviceSynchronize();
    for (int iteration = 0; iteration<maximumIterations; ++iteration)
        {
        mProf.start("generate points");
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
        mProf.end("generate points");
cudaDeviceSynchronize();

        if(programSwitch%2 == 0)
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

cudaDeviceSynchronize();
        //similarly, the cudaProfiler calls are just for...profiling
        cudaProfilerStart();
        prof2.start();
        mProf.start("delGPU total timing");
cudaDeviceSynchronize();
        mProf.start("delGPU triangulation");
        prof.start();

        delGPU.globalDelaunayTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
cudaDeviceSynchronize();
        prof.end();
        cudaProfilerStop();
        mProf.end("delGPU triangulation");
        prof2.end();
        mProf.end("delGPU total timing");
        if(programSwitch %2==0)
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
else if (programSwitch == -1 || programSwitch == -2)//test and repair routines
    {
    vector<int> repairList(N);
    for(int ii = 0; ii < N; ++ii)
        repairList[ii]=ii;
    int localNumber = floor(localFraction*N);
    noise.fillArray(gpuPts,0,L);
    DelaunayGPU delGPU(N, maxNeighs, cellSize, domain);
    delGPU.setSafetyMode(safetyMode);
    if(gpuSwitch>=0)
        delGPU.setGPUcompute(true);
    else
        delGPU.setGPUcompute(false);
    DelaunayCGAL cgalTriangulation;
cudaDeviceSynchronize();
    GPUArray<int> gpuTriangulation((unsigned int) (maxNeighs)*N);
    GPUArray<int> cellNeighborNumber((unsigned int) N);
    {
    ArrayHandle<double2> p(gpuPts,access_location::device,access_mode::readwrite);
    ArrayHandle<int> t(gpuTriangulation,access_location::device,access_mode::readwrite);
    ArrayHandle<int> c(cellNeighborNumber,access_location::device,access_mode::readwrite);
    }
cudaDeviceSynchronize();
    printf("global routine to begin with a putative triangulation...");
    mProf.start("delGPU initial global repair");
    delGPU.globalDelaunayTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
cudaDeviceSynchronize();
    mProf.end("delGPU initial global repair");
    printf("...done\n");
    for (int iteration = 0; iteration<maximumIterations; ++iteration)
        {
        cout << "iteration " <<iteration << "... shuffling " << localNumber << " points"<< endl;
        double phase = 0.1;
        std::random_shuffle(repairList.begin(),repairList.end());
        {
        ArrayHandle<double2> p(gpuPts);
        for (int ii =0; ii < localNumber; ++ii)
            {
            double2 pTarget = p.data[repairList[ii]];
            pTarget.x += noise.getRealUniform(-phase,phase);
            pTarget.y += noise.getRealUniform(-phase,phase);
            delGPU.Box->putInBoxReal(pTarget);
            p.data[repairList[ii]]= pTarget;
            }
        }
        //make sure the triangulation is wrong:
    if(programSwitch == -2)
    {
        cout << " comparing with CGAL to see the old triangulation is wrong" << endl;
        {
        ArrayHandle<double2> gps(gpuPts,access_location::host,access_mode::read);
        for (int ii = 0; ii < N; ++ii)
            {
            pts[ii]=make_pair(Point(gps.data[ii].x,gps.data[ii].y),ii);
            }
        mProf.start("CGAL triangulation");
        cgalTriangulation.PeriodicTriangulation(pts,L,0,0,L);
        mProf.end("CGAL triangulation");
        }//end CGAL 
        mProf.start("triangulation comparison");
        compareTriangulation(cgalTriangulation.allneighs, gpuTriangulation,cellNeighborNumber,false);
        mProf.end("triangulation comparison");
        {
        ArrayHandle<double2> p(gpuPts,access_location::device,access_mode::readwrite);
        }
    }
        cout << "using delGPU to testAndRepair" << endl;
cudaDeviceSynchronize();
        mProf.start("delGPU testAndRepair");
        delGPU.testAndRepairDelaunayTriangulation(gpuPts,gpuTriangulation,cellNeighborNumber);
cudaDeviceSynchronize();
        mProf.end("delGPU testAndRepair");

    if(programSwitch == -2)
    {
    cout << "re-triangulating with CGAL for comparison..." << endl;
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
    mProf.start("triangulation comparison");

    printf("comparing repaired triangulation with CGAL...");
    compareTriangulation(cgalTriangulation.allneighs, gpuTriangulation,cellNeighborNumber);
    printf("...done\n");
    mProf.end("triangulation comparison");
    }

        }
#ifdef DEBUGFLAGUP
    cout << endl;
    delGPU.prof.print();
#endif
    cout << endl;
    mProf.print();
    }

//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
    return 0;
};
