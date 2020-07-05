#include "DelaunayGPU.h"
#include "DelaunayGPU.cuh"
#include "cellListGPU.cuh"
#include "gpuUtilities.cuh"

DelaunayGPU::DelaunayGPU() :
	cellsize(1.10), delGPUcircumcentersInitialized(false), cListUpdated(false), Ncells(0), NumCircumCenters(0)
{
Box = make_shared<periodicBoundaries>();
}

DelaunayGPU::DelaunayGPU(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx) :
		cellsize(cellSize), delGPUcircumcentersInitialized(false), cListUpdated(false),
		Ncells(N), NumCircumCenters(0), MaxSize(maximumNeighborsGuess)
		{
		initialize(bx);
		}

//!initialization
void DelaunayGPU::initialize(PeriodicBoxPtr bx)
		{
		setBox(bx);
		sizeFixlist.resize(1);
		GPUVoroCur.resize(MaxSize*Ncells);
		GPUDelNeighsPos.resize(MaxSize*Ncells);
		GPUVoroCurRad.resize(MaxSize*Ncells);
		GPUPointIndx.resize(MaxSize*Ncells);

		GPU_idx = Index2D(MaxSize,Ncells);
		neighs.resize(Ncells);
		repair.resize(Ncells+1);
		initializeCellList();
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
}

/*!
\param points a GPUArray of double2's with the new desired points
Use the GPU to copy the arrays into this class.
Might not have a performance boost but it reduces HtD memory copies
*/
void DelaunayGPU::setCircumcenters(GPUArray<int3> &circumcenters)
{
    if(delGPUcircumcenters.getNumElements()!=circumcenters.getNumElements())
    {
    NumCircumCenters=circumcenters.getNumElements();
    delGPUcircumcenters.resize(NumCircumCenters);
    }

    gpu_copy_gpuarray<int3>(delGPUcircumcenters,circumcenters);
    delGPUcircumcentersInitialized=true;
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

void DelaunayGPU::initializeCellList()
	{
	cList.setNp(Ncells);
    cList.setBox(Box);
    cList.setGridSize(cellsize);
    }

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

//One of the main triangulation routines.
//This function completly creates the triangulation fo each point in the repair array.
void DelaunayGPU::GPU_LocalDelTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
{

	if(points.getNumElements()==0){printf("No points in GPU DT\n");return;}
	if(delGPUcircumcenters.getNumElements()==0)
	{
		printf("GPU DT Local: No circuncircles for testing\n");
		throw std::exception();
	}
        if(delGPUcircumcentersInitialized==false)
	{
		printf("GPU DT Local: No circuncircles initialized\n");
                throw std::exception();
	}
        if(cListUpdated==false)
	{
		printf("GPU DT Local: Cell list not updated\n");
                throw std::exception();
	}
	if(points.getNumElements()!=Ncells)
	{
		printf("GPU DT Local: Bug in GPU DT\n");
                throw std::exception();
	}
        if(GPUTriangulation.getNumElements()!=GPUVoroCur.getNumElements())
        {
                printf("GPU DT Global: Incorrect sizes in the GPUArrays\n");
                throw std::exception();
        }


        testTriangulation();
        build_repair();
        ArrayHandle<int> sizefixlist(sizeFixlist,access_location::host,access_mode::read);
        size_fixlist=sizefixlist.data[0];

        if(size_fixlist>0)
	{
		Voronoi_Calc(points, GPUTriangulation, cellNeighborNum);
		get_neighbors(points, GPUTriangulation, cellNeighborNum);
	}

	delGPUcircumcentersInitialized=false;
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
		cList.computeGPU(points);
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
    bool callGlobal = true;
    Voronoi_Calc(points, GPUTriangulation, cellNeighborNum,callGlobal);
	get_neighbors(points, GPUTriangulation, cellNeighborNum,callGlobal);

	delGPUcircumcentersInitialized=false;
}


//Helper fucntion to organize repair array to get ready for triangulation
void DelaunayGPU::build_repair()
{
        ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);
        ArrayHandle<int> sizefixlist(sizeFixlist,access_location::device,access_mode::overwrite);

        gpu_build_repair(d_repair.data,
                     Ncells,
                     sizefixlist.data
                    );
}

//One of the main functions called by the triangulation.
//This creates a simple convex polygon around each point for triangulation.
//Currently the polygon is created with only four points
void DelaunayGPU::Voronoi_Calc(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum, bool callGlobalRoutine)
{

  ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
  ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
  ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

  ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
  ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);
  ArrayHandle<int> d_repair(repair,access_location::device,access_mode::read);

  ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
  ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
  ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);

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
                   d_repair.data,
                   size_fixlist,
                   GPU_idx,
                   callGlobalRoutine
                   );
}

//The final main function of the triangulation.
//This takes the previous polygon and further updates it to create the final delaunay triangulation
void DelaunayGPU::get_neighbors(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,bool callGlobalRoutine)
{

  ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
  ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
  ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

  ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
  ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);
  ArrayHandle<int> d_repair(repair,access_location::device,access_mode::read);

  ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
  ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
  ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);

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
                   d_repair.data,
                   size_fixlist,
                   GPU_idx,
                   callGlobalRoutine
                   );
}

//Same as above but used by the balanced section
void DelaunayGPU::Voronoi_Calc()
{

  ArrayHandle<double2> d_pt(pts,access_location::device,access_mode::read);
  ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
  ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

  ArrayHandle<int> d_P_idx(GPUPointIndx,access_location::device,access_mode::readwrite);
  ArrayHandle<int> d_neighnum(neighs,access_location::device,access_mode::readwrite);
  ArrayHandle<int> d_repair(repair,access_location::device,access_mode::read);

  ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
  ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
  ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);

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
                   d_repair.data,
                   size_fixlist,
                   GPU_idx
                   );
}

/*!
Call the GPU to test each circumcenter to see if it is still empty (i.e., how much of the
triangulation from the last time step is still valid?). Note that because gpu_test_circumcenters
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

    ArrayHandle<int3> d_ccs(delGPUcircumcenters,access_location::device,access_mode::read);

    NumCircumCenters = Ncells*2;
    gpu_test_circumcenters(d_repair.data,
                           d_ccs.data,
                           NumCircumCenters,
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
Converts the neighbor list data structure into a list of the three particle indices defining
all of the circumcenters in the triangulation. Keeping this version of the topology on the GPU
allows for fast testing of what points need to be retriangulated.
*/
void DelaunayGPU::getCircumcenters(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    delGPUcircumcentersInitialized=true;
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> ns(GPUTriangulation,access_location::host,access_mode::read);
    ArrayHandle<int3> h_ccs(delGPUcircumcenters,access_location::host,access_mode::overwrite);

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
    NumCircumCenters = cidx;

    if((totaln != 6*Ncells || cidx != 2*Ncells))
        {
        printf("GPU step: getCCs failed, %i out of %i ccs, %i out of %i neighs \n",cidx,2*Ncells,totaln,6*Ncells);
        throw std::exception();
        };

    };
