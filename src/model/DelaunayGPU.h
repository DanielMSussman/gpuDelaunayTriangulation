#ifndef DELAUNAYGPU_H
#define DELAUNAYGPU_H

#include "gpuarray.h"
#include "periodicBoundaries.h"
#include "cellListGPU.h"

using namespace std;

/*! \file DelaunayGPU.h */
 //!A GPU-based class for locally constructing the Delaunay triangulation of part of a point set
/*!
 *GPU implementation of the DT.
 *It makes use of a locallity lema described in (doi: 10.1109/ISVD.2012.9).
 *It will only make the repair of the topology in case it is necessary.
 *Steps are detailed as in paper.
 * This function operates strictly on the GPU
 */

class DelaunayGPU
    {
	    public:

		    //!Constructor
		    DelaunayGPU();
        //!Constructor + initialiation
        DelaunayGPU(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx);
		    //!Destructor
		    ~DelaunayGPU(){};

        //!primitive initialization function
        void initialize(PeriodicBoxPtr bx);
		    //!Initialize various things, based on a given cell size for the underlying grid
		    void initialize(GPUArray<double2> &points, double csize, int N, const int nmax, PeriodicBoxPtr bx);
        //!prep the cell list
        void initializeCellList();
		    void resize(const int nmax);

		    //!<Set points via a GPUarray
		    void setPoints(GPUArray<double2> &points);
		    //!<Set points that need repair via a GPUarray
		    void setRepair(GPUArray<int> &rep);
		    //!Set the circumcenters via a GPUArray
		    void setCircumcenters(GPUArray<int3> &circumcenters);
		    //!Initialize various things, based on a given cell size for the underlying grid
		    void setList(double csize, GPUArray<double2> &points);
		    //!Only update the cell list
		    void updateList(GPUArray<double2> &points);
		    //!Set the box
		    void setBox(periodicBoundaries &bx);
		    void setBox(PeriodicBoxPtr bx){Box=bx;};
		    //!Set the cell size of the underlying grid
		    void setCellSize(double cs){cellsize=cs;};

	      //!build the auxiliary data structure containing the indices of the particle circumcenters from the neighbor list
        void getCircumcenters(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

		    //!Tests the circuncircles of the DT to check if they overlap any new poin
		    void testTriangulation();

		    //!Globally and locally construct the triangulation via GPU
        //!Functions used by the GPU DT
        void GPU_LocalDelTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        void GPU_GlobalDelTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

		    //These functions rely on the fact that one only needs to calculate 2 triangles in the triangulation per point,
		    //thus reducing load balancing, but for that a new function needs to be called to do a final ordering.
		    //It is also worth mentioning that these functions eliminate the need to use the getcircumcircles function,
		    //as they are a byproduct of the triangulation algorithm
		    void GPU_BalancedLocalDelTriangulation(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        void GPU_BalancedGlobalDelTriangulation(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

	    private:

		    //!Functions used by the GPU DT
		    //!Creates the organized array of cells to triangulate
		    void build_repair();
		    void global_repair();
		    void Balanced_repair(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

		    //!Main function of this class, it performs the Delaunay triangulation
		    void Voronoi_Calc(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
		    void Voronoi_Calc();
		    void get_neighbors(GPUArray<double2> &points,GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
		    void BalancedGetNeighbors();
		    void OrganizeDelTriangulation(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

	    protected:

		    //!A helper array used for the triangulation on the GPU, before the topology is known
        GPUArray<double2> GPUVoroCur;
        GPUArray<double2> GPUDelNeighsPos;
        GPUArray<double> GPUVoroCurRad;
        GPUArray<int> GPUPointIndx;

		    GPUArray<int> neighs;
		    GPUArray<double2> pts;
		    GPUArray<int3> delGPUcircumcenters;
		    GPUArray<int>repair;

		    bool delGPUcircumcentersInitialized;

		    int Ncells;
        //! The maximum number of neighbors any point has
		    int MaxSize;
		    int NumCircumCenters;

		    //!A list to save all the cells that need fixing
		    GPUArray<int> sizeFixlist;
        int size_fixlist;

		    //!A 2dIndexer for computing where in the GPUArray to look for a given particles neighbors GPU
        Index2D GPU_idx;

		    //!< A box to calculate relative distances in a periodic domain.
		    PeriodicBoxPtr Box;
	 	    //!A cell list for speeding up the calculation of the candidate 1-ring
        cellListGPU cList;
        int cellsize;
		    bool cListUpdated;
    };
#endif
