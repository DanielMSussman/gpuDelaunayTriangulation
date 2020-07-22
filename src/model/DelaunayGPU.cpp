#include "DelaunayGPU.h"
#include "DelaunayGPU.cuh"
#include "cellListGPU.cuh"
#include "utilities.cuh"

DelaunayGPU::DelaunayGPU() :
	cellsize(1.10), delGPUcircumcirclesInitialized(false), cListUpdated(false), Ncells(0), NumCircumcircles(0)
{
Box = make_shared<periodicBoundaries>();
}

DelaunayGPU::DelaunayGPU(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx) :
		cellsize(cellSize), delGPUcircumcirclesInitialized(false), cListUpdated(false),
		Ncells(N), NumCircumcircles(0), MaxSize(maximumNeighborsGuess)
		{
		initialize(bx);
        if(MaxSize <4)
            MaxSize = 4;//the initial enclosing polygon is always at least 4 points...
		}

//!initialization
void DelaunayGPU::initialize(PeriodicBoxPtr bx)
		{
        prof.start("initialization");
		setBox(bx);
		sizeFixlist.resize(1);
        maxOneRingSize.resize(1);
        {
        ArrayHandle<int> ms(maxOneRingSize);
        ms.data[0] = MaxSize;
        }
        resize(MaxSize);

		neighs.resize(Ncells);
		repair.resize(Ncells);
        delGPUcircumcircles.resize(Ncells);
        {
        ArrayHandleAsync<int> ms(maxOneRingSize,access_location::device,access_mode::read);
        ArrayHandleAsync<int>  n(neighs,access_location::device,access_mode::read);
        ArrayHandleAsync<int> r(repair,access_location::device,access_mode::read);
        }
		initializeCellList();
        prof.end("initialization");
		}


//Resize the relevant array for the triangulation
void DelaunayGPU::resize(const int nmax)
    {
    MaxSize=nmax;
    GPUVoroCur.resize(nmax*Ncells);
    GPUDelNeighsPos.resize(nmax*Ncells);
    GPUVoroCurRad.resize(nmax*Ncells);
    GPUPointIndx.resize(nmax*Ncells);
    GPU_idx = Index2D(nmax,Ncells);
    {
    ArrayHandleAsync<double2> vc(GPUVoroCur,access_location::device,access_mode::read);
    ArrayHandleAsync<double2> np(GPUDelNeighsPos,access_location::device,access_mode::read);
    ArrayHandleAsync<double> vcr(GPUVoroCurRad,access_location::device,access_mode::read);
    ArrayHandleAsync<int> pi(GPUPointIndx,access_location::device,access_mode::read);
    }
    }

void DelaunayGPU::initializeCellList()
	{
	cList.setNp(Ncells);
    cList.setBox(Box);
    cList.setGridSize(cellsize);
    }

/*!
\param points a GPUArray of double2's with the new desired points
Use the GPU to copy the arrays into this class.
Might not have a performance boost but it reduces HtD memory copies
*/
void DelaunayGPU::setCircumcircles(GPUArray<int3> &circumcircles)
{
    if(delGPUcircumcircles.getNumElements()!=circumcircles.getNumElements())
    {
    NumCircumcircles=circumcircles.getNumElements();
    delGPUcircumcircles.resize(NumCircumcircles);
    }

    gpu_copy_gpuarray<int3>(delGPUcircumcircles,circumcircles);
    delGPUcircumcirclesInitialized=true;
};

/*!
\param bx a periodicBoundaries that the DelaunayLoc object should use in internal computations
*/
void DelaunayGPU::setBox(periodicBoundaries &bx)
    {
    cListUpdated=false;
    Box = make_shared<periodicBoundaries>();
    double b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    if (bx.isBoxSquare())
        Box->setSquare(b11,b22);
    else
        Box->setGeneral(b11,b12,b21,b22);
    };

//sets the bucket lists with the points that they contain to use later in the triangulation
void DelaunayGPU::setList(double csize, GPUArray<double2> &points)
{
    cListUpdated=true;
    if(points.getNumElements()!=Ncells || points.getNumElements()==0)
    {
	    printf("GPU DT: No points for cell lists\n");
            throw std::exception();
    }
    cList.computeGPU(points);
}

//if one wants to choose which points they want to repair
void DelaunayGPU::setRepair(GPUArray<int> &rep)
{
    if(repair.getNumElements()!=rep.getNumElements())
        {
	    printf("GPU DT: repair array has incorrect size. Make sure to update points array first!\n");
	    throw std::exception();
        }
    gpu_copy_gpuarray<int>(repair,rep);
}

//automatically goes thorough the process of updating the points
//and lists to get ready for the triangulation (previous initializaton required!).
void DelaunayGPU::updateList(GPUArray<double2> &points)
    {
    cList.computeGPU(points);
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("cell list computation GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };
    cListUpdated=true;
}

//call the triangulation routines on a subset of the total number of points
void DelaunayGPU::locallyRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList,int numberToRepair)
    {
    //setRepair(repairList);
    //size_fixlist = numberToRepair;

    int currentN = points.getNumElements();
    if(cListUpdated==false)
		{
        prof.start("cellList");
		cList.computeGPU(points);
        prof.end("cellList");
		cListUpdated=true;
		}
    bool recompute = true;
    while (recompute)
        {
        voronoiCalcRepairList(points, GPUTriangulation, cellNeighborNum,repairList);
        recompute = computeTriangulationRepairList(points, GPUTriangulation, cellNeighborNum,repairList);
        if(recompute)
            {
            GPUTriangulation.resize(MaxSize*currentN);
            }
        };

	delGPUcircumcirclesInitialized=false;
    }

//Main function that does the complete triangulation of all points
void DelaunayGPU::GPU_GlobalDelTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
	int currentN = points.getNumElements();
	if(currentN==0)
        {
        cout<<"No points in GPU DT"<<endl;
        return;
        }
    if(cListUpdated==false)
		{
        prof.start("cellList");
		cList.computeGPU(points);
        prof.end("cellList");
		cListUpdated=true;
		}
    if(currentN!=Ncells)
		{
		printf("GPU DT Global: Bug in GPU DT\n");
        throw std::exception();
		}
	if(GPUTriangulation.getNumElements()!=GPUVoroCur.getNumElements())
      {
      printf("GPU DT Global: Incorrect sizes in the GPUArrays\n");
      throw std::exception();
      }

    size_fixlist=Ncells;
    bool recompute = true;
    while (recompute)
	    {
        prof.start("vorocalc");
        Voronoi_Calc(points, GPUTriangulation, cellNeighborNum);
        prof.end("vorocalc");
        prof.start("get 1 ring");
        recompute = get_neighbors(points, GPUTriangulation, cellNeighborNum);
        prof.end("get 1 ring");
        if(recompute)
            {
            GPUTriangulation.resize(MaxSize*currentN);
            }
        };

	delGPUcircumcirclesInitialized=false;
}

void DelaunayGPU::voronoiCalcRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList)
    {
    ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_repair(repairList,access_location::device,access_mode::read);

    ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
    ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
    ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);

    gpu_voronoi_calc_no_sort(d_pt.data,
                        d_cell_sizes.data,
                        d_cell_idx.data,
                        d_P_idx.data,
                        d_P.data,
                        d_Q.data,
                        d_Q_rad.data,
                        d_neighnum.data,
                        Ncells,
                        cList.getXsize(),
                        cList.getYsize(),
                        cList.getBoxsize(),
                        *(Box),
                        cList.cell_indexer,
                        cList.cell_list_indexer,
                        d_repair.data,
                        GPU_idx
                        );
    };

//One of the main functions called by the triangulation.
//This creates a simple convex polygon around each point for triangulation.
//Currently the polygon is created with only four points
void DelaunayGPU::Voronoi_Calc(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::overwrite);
    ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::overwrite);

    ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::overwrite);
    ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::overwrite);
    ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::overwrite);

    gpu_voronoi_calc(d_pt.data,
                        d_cell_sizes.data,
                        d_cell_idx.data,
                        d_P_idx.data,
                        d_P.data,
                        d_Q.data,
                        d_Q_rad.data,
                        d_neighnum.data,
                        Ncells,
                        cList.getXsize(),
                        cList.getYsize(),
                        cList.getBoxsize(),
                        *(Box),
                        cList.cell_indexer,
                        cList.cell_list_indexer,
                        GPU_idx
                        );


    }

bool DelaunayGPU::computeTriangulationRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList)
    {
        bool recomputeNeighbors = false;
        int postCallMaxOneRingSize;
        int currentMaxOneRingSize = MaxSize;
        {
                ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
                ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
                ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

                ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
                ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);
                ArrayHandle<int> d_repair(repairList,access_location::device,access_mode::read);

                ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
                ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
                ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);
                ArrayHandle<int> d_ms(maxOneRingSize, access_location::device,access_mode::readwrite);


                gpu_get_neighbors_no_sort(d_pt.data,
                                d_cell_sizes.data,
                                d_cell_idx.data,
                                d_P_idx.data,
                                d_P.data,
                                d_Q.data,
                                d_Q_rad.data,
                                d_neighnum.data,
                                Ncells,
                                cList.getXsize(),
                                cList.getYsize(),
                                cList.getBoxsize(),
                                *(Box),
                                cList.cell_indexer,
                                cList.cell_list_indexer,
                                d_repair.data,
                                GPU_idx,
                                d_ms.data,
                                currentMaxOneRingSize
                                    );
        }
        if(safetyMode)
        {
                {//check initial maximum ring_size allocated
                        ArrayHandle<int> h_ms(maxOneRingSize, access_location::host,access_mode::read);
                        postCallMaxOneRingSize = h_ms.data[0];
                }
                //printf("initial and post %i %i\n", currentMaxOneRingSize,postCallMaxOneRingSize);
                if(postCallMaxOneRingSize > currentMaxOneRingSize)
                {
                        recomputeNeighbors = true;
                        printf("resizing potential neighbors from %i to %i and re-computing...\n",currentMaxOneRingSize,postCallMaxOneRingSize);
                        resize(postCallMaxOneRingSize);
                }
        };
        return recomputeNeighbors;
    }

//The final main function of the triangulation.
//This takes the previous polygon and further updates it to create the final delaunay triangulation
bool DelaunayGPU::get_neighbors(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
{
        bool recomputeNeighbors = false;
        int postCallMaxOneRingSize;
        int currentMaxOneRingSize = MaxSize;
            {
            ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
            ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
            ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

            ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);

            ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
            ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
            ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ms(maxOneRingSize, access_location::device,access_mode::readwrite);


            gpu_get_neighbors(d_pt.data,
                                d_cell_sizes.data,
                                d_cell_idx.data,
                                d_P_idx.data,
                                d_P.data,
                                d_Q.data,
                                d_Q_rad.data,
                                d_neighnum.data,
                                Ncells,
                                cList.getXsize(),
                                cList.getYsize(),
                                cList.getBoxsize(),
                                *(Box),
                                cList.cell_indexer,
                                cList.cell_list_indexer,
                                GPU_idx,
                                d_ms.data,
                                currentMaxOneRingSize
                            );
            }
        if(safetyMode)
        {
                {//check initial maximum ring_size allocated
                        ArrayHandle<int> h_ms(maxOneRingSize, access_location::host,access_mode::read);
                        postCallMaxOneRingSize = h_ms.data[0];
                }
                //printf("initial and post %i %i\n", currentMaxOneRingSize,postCallMaxOneRingSize);
                if(postCallMaxOneRingSize > currentMaxOneRingSize)
                {
                        recomputeNeighbors = true;
                        printf("resizing potential neighbors from %i to %i and re-computing...\n",currentMaxOneRingSize,postCallMaxOneRingSize);
                        resize(postCallMaxOneRingSize);
                }
        };
        return recomputeNeighbors;
}

void DelaunayGPU::testAndRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    //resize circumcircles array if needed and populate:
    if(delGPUcircumcircles.getNumElements()!= 2*points.getNumElements())
        delGPUcircumcircles.resize(2*points.getNumElements());
    prof.start("getCCS");
    getCircumcirclesGPU(GPUTriangulation,cellNeighborNum);
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    prof.end("getCCS");

    prof.start("cellList");
	cList.computeGPU(points);
    cListUpdated=true;
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    prof.end("cellList");

    prof.start("resetRepairList");
    {
    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);
    gpu_set_array(d_repair.data,-1,Ncells);
    }
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    prof.end("resetRepairList");
    prof.start("testCCS");
    testTriangulation(points);
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    prof.end("testCCS");
    
    //locally repair
    prof.start("repairPoints");
    int filloutfunction = -1;//has no effect, but exists so we can easily swap sorting vs non-sorting algorithms during testing phase
    locallyRepairDelaunayTriangulation(points,GPUTriangulation,cellNeighborNum,repair,filloutfunction);
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    prof.end("repairPoints");
    }

/*!
only intended to be used as part of the testAndRepair sequence
*/
void DelaunayGPU::testTriangulation(GPUArray<double2> &points)
    {
    //access data handles
    ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);

    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_c_idx(cList.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);

    ArrayHandle<int3> d_ccs(delGPUcircumcircles,access_location::device,access_mode::read);

    NumCircumcircles = Ncells*2;
    gpu_test_circumcircles(d_repair.data,
                           d_ccs.data,
                           NumCircumcircles,
                           d_pt.data,
                           d_cell_sizes.data,
                           d_c_idx.data,
                           Ncells,
                           cList.getXsize(),
                           cList.getYsize(),
                           cList.getBoxsize(),
                           *(Box),
                           cList.cell_indexer,
                           cList.cell_list_indexer
                           );
    };

/*!
Call the GPU to test each circumcenter to see if it is still empty (i.e., how much of the
triangulation from the last time step is still valid?). Note that because gpu_test_circumcircles
*always* copies at least a single integer back and forth (to answer the question "did any
circumcircle come back non-empty?" for the cpu) this function is always an implicit cuda
synchronization event. At least until non-default streams are added to the code.
*/
void DelaunayGPU::testTriangulation()
    {
    //access data handles
    ArrayHandle<double2> d_pt(pts,access_location::device,access_mode::read);

    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_c_idx(cList.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);

    ArrayHandle<int3> d_ccs(delGPUcircumcircles,access_location::device,access_mode::read);

    NumCircumcircles = Ncells*2;
    gpu_test_circumcircles(d_repair.data,
                           d_ccs.data,
                           NumCircumcircles,
                           d_pt.data,
                           d_cell_sizes.data,
                           d_c_idx.data,
                           Ncells,
                           cList.getXsize(),
                           cList.getYsize(),
                           cList.getBoxsize(),
                           *(Box),
                           cList.cell_indexer,
                           cList.cell_list_indexer
                           );
    };

void DelaunayGPU::getCircumcirclesGPU(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    ArrayHandle<int> assist(sizeFixlist,access_location::device,access_mode::readwrite);
    gpu_set_array(assist.data,0,1);//set the fixlist to zero, for indexing purposes
    delGPUcircumcirclesInitialized=true;
    ArrayHandle<int> neighbors(GPUTriangulation,access_location::device,access_mode::read);
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::device,access_mode::read);
    ArrayHandle<int3> ccs(delGPUcircumcircles,access_location::device,access_mode::overwrite);
    gpu_get_circumcircles(neighbors.data,
                          neighnum.data,
                          ccs.data,
                          assist.data,
                          Ncells,
                          GPU_idx);
    }

/*!
Converts the neighbor list data structure into a list of the three particle indices defining
all of the circumcircles in the triangulation. Keeping this version of the topology on the GPU
allows for fast testing of what points need to be retriangulated.
*/
void DelaunayGPU::getCircumcircles(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    delGPUcircumcirclesInitialized=true;
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> ns(GPUTriangulation,access_location::host,access_mode::read);
    ArrayHandle<int3> h_ccs(delGPUcircumcircles,access_location::host,access_mode::overwrite);

    int totaln = 0;
    int cidx = 0;
    bool fail = false;
    for (int nn = 0; nn < Ncells; ++nn)
        {
        int nmax = neighnum.data[nn];
        totaln+=nmax;
        for (int jj = 0; jj < nmax; ++jj)
            {
            if (fail) continue;

            int n1 = ns.data[GPU_idx(jj,nn)];
            int ne2 = jj + 1;
            if (jj == nmax-1)  ne2=0;
            int n2 = ns.data[GPU_idx(ne2,nn)];
            if (nn < n1 && nn < n2)
                {
                h_ccs.data[cidx].x = nn;
                h_ccs.data[cidx].y = n1;
                h_ccs.data[cidx].z = n2;
                cidx+=1;
                };
            };
        };
    NumCircumcircles = cidx;

    if((totaln != 6*Ncells || cidx != 2*Ncells))
        {
        printf("GPU step: getCCs failed, %i out of %i ccs, %i out of %i neighs \n",cidx,2*Ncells,totaln,6*Ncells);
        throw std::exception();
        };

    };
