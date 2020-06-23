#ifndef __voronoiModelBase_CUH__

#include <cuda_runtime.h>
#include "std_include.h"
#include "indexer.h"
#include "periodicBoundaries.h"


/*!
 \file DelaunayGPU.cuh
A file providing an interface to the relevant cuda calls for the delaunay GPU class
*/

/** @defgroup DelaunayGPUKernels DelaunayGPU Kernels
 * @{
 * \brief CUDA kernels and callers for the DelaunayGPU class
 */


//Use the GPU to copy the arrays into this class.
//Might not have a performance boost but it reduces HtD memory copies
bool gpu_setPoints(double2 *hp, double2 *d_pts, int *d_repair, int Nf);
bool gpu_setCircumcenters(int3 *hp, int3 *d_ccs, int Nf);
bool gpu_global_repair(int *d_repair, int Nf);
bool gpu_setRepair(int *hp, int *d_rep, int Nf);


bool gpu_Balanced_repair(int *d_repair, 
		         int Ncells, 
			 int *Nf,
			 int *d_Tri,
			 int *d_neighnum,
			 int *P_idx,
			 int *neighs,
			 Index2D GPU_idx
			 );

//test the triangulation to see if it is still valid
bool gpu_test_circumcenters(int *d_repair,
                            int3 *d_ccs,
                            int Nccs,
                            double2 *d_pt,
                            unsigned int *d_cell_sizes,
                            int *d_idx,
                            int Np,
                            int xsize,
                            int ysize,
                            double boxsize,
                            periodicBoundaries &Box,
                            Index2D &ci,
                            Index2D &cli
                            );

//!Organize the repair array to send off to be triangulated
bool gpu_build_repair( int *d_repair,
                   int Np,
                   int *Nf
                   );

//create initial polygon around cell i to start triangulation algorithm
bool gpu_initial_poly(double2* d_pt,
                             int* P_idx,
                             double2* P,
                             int* d_neighnum,
                             int* c,
                             int Ncells,
                             periodicBoundaries Box,
                             int* d_fixlist,
                             int Nf,
                             Index2D GPU_idx
                             );

//calculate voronoi points using previous polygon
//this function is currently bg since we cannot guarantee 100% that the previous poly contains cell i
//this should be a step that could b optimized
bool gpu_voronoi_calc(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      int* d_fixlist,
                      int Nf,
                      Index2D GPU_idx
                      );

//the meat of the triangulation algorithm, calculates the actual del neighs of cell i
//this is also a bit large, but to optimize it, big algorithmical changes might be needed (I'm too lazy though...)                  
bool gpu_get_neighbors(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      int* d_fixlist,
                      int Nf,
                      Index2D GPU_idx
                      );

bool gpu_BalancedGetNeighbors(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      int* d_fixlist,
                      int Nf,
                      Index2D GPU_idx
                      );

bool gpu_OrganizeDelTriangulation(
                   int *d_neighnum,
                   int Ncells,
                   int *d_repair,
                   int size_fixlist,
                   int *d_Tri,
		   int *P_idx,
		   int *neighs,
		   Index2D GPU_idx
                   );




/** @} */ //end of group declaration

#endif
