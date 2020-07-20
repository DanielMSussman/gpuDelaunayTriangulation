#ifndef DELAUNAYGPU_H
#define DELAUNAYGPU_H

#include "gpuarray.h"
#include "periodicBoundaries.h"
#include "cellListGPU.h"
#include "multiProfiler.h"

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

		//!Constructor -- not to be used right now
		DelaunayGPU();
        //!Constructor + initialiation
        DelaunayGPU(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx);
		//!Destructor
		~DelaunayGPU(){};

        //!primitive initialization function
        void initialize(PeriodicBoxPtr bx);

        //!function call to change the maximum number of neighbors per point
        void resize(const int nmax);

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
        void GPU_GlobalDelTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        void locallyRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList, int numberToRepair);

        multiProfiler prof;

        //!Set the safetyMode flag...IF safetyMode is false and the assumptions are not satisfied, program will be wrong with (possibly) no warning
        void setSafetyMode(bool _sm){safetyMode=_sm;};

        //!< A box to calculate relative distances in a periodic domain.
        PeriodicBoxPtr Box;

        bool cListUpdated;

    private:
        //!Functions used by the GPU DT

        //!Main function of this class, it performs the Delaunay triangulation
        void Voronoi_Calc(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        bool get_neighbors(GPUArray<double2> &points,GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!testing an alternate memory pattern for local repairs
        void voronoiCalcRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //!same memory pattern, for getNeighbors
        bool computeTriangulationRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //!prep the cell list
        void initializeCellList();

        //!If false, the user is guaranteeing that the current maximum number of neighbors per point will not be exceeded
        bool safetyMode = false;

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

        //!An array that holds a single int keeping track of maximum 1-ring size
        GPUArray<int> maxOneRingSize;

        int Ncells;
        //! The maximum number of neighbors any point has
        int MaxSize;
        int NumCircumCenters;

        //!A list to save all the cells that need fixing
        GPUArray<int> sizeFixlist;
        int size_fixlist;

        //!A 2dIndexer for computing where in the GPUArray to look for a given particles neighbors GPU
        Index2D GPU_idx;

        //!A cell list for speeding up the calculation of the candidate 1-ring
        cellListGPU cList;
        double cellsize;
    };
#endif
