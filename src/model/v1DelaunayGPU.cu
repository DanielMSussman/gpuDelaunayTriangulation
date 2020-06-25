#include <cuda_runtime.h>
#include "cellListGPU.cuh"
#include "indexer.h"
#include "periodicBoundaries.h"
#include "functions.h"
#include <iostream>
#include <stdio.h>
#include "DelaunayGPU.cuh"
#include <thrust/sort.h>

/*! \file DelaunayGPU.cu */
/*!
    \addtogroup DelaunayGPUBaseKernels
    @{
*/


/*!+*9
  Independently check every triangle in the Delaunay mesh to see if the cirumcircle defined by the
  vertices of that triangle is empty. Use the cell list to ensure that only checks of nearby
  particles are required.
  */
__global__ void gpu_test_circumcenters_kernel(int* __restrict__ d_repair,
                                              const int3* __restrict__ d_circumcircles,
                                              const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int Nccs,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli
                                              )
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nccs)
        return;

    //the indices of particles forming the circumcircle
    int3 i1 = d_circumcircles[idx];
    //the vertex we will take to be the origin, and its cell position
    double2 v = d_pt[i1.x];
    int ib=floor(v.x/boxsize);
    int jb=floor(v.y/boxsize);

    double2 pt1,pt2;
    Box.minDist(d_pt[i1.y],v,pt1);
    Box.minDist(d_pt[i1.z],v,pt2);


    //get the circumcircle
    double2 Q;
    double rad;
    Circumcircle(pt1,pt2,Q,rad);

    //look through cells for other particles...re-use pt1 and pt2 variables below
    bool badParticle = false;
    int wcheck = ceil(rad/boxsize)+1;

    if(wcheck > xsize/2) wcheck = xsize/2;
    rad = rad*rad;
    for (int ii = ib-wcheck; ii <= ib+wcheck; ++ii)
        {
        for (int jj = jb-wcheck; jj <= jb+wcheck; ++jj)
            {
            int cx = ii;
            if(cx < 0) cx += xsize;
            if(cx >= xsize) cx -= xsize;
            int cy = jj;
            if(cy < 0) cy += ysize;
            if(cy >= ysize) cy -= ysize;

            int bin = ci(cx,cy);

            for (int pp = 0; pp < d_cell_sizes[bin]; ++pp)
                {
                int newidx = d_cell_idx[cli(pp,bin)];

                Box.minDist(d_pt[newidx],v,pt1);
                Box.minDist(pt1,Q,pt2);

                //if it's in the circumcircle, check that its not one of the three points
                if(pt2.x*pt2.x+pt2.y*pt2.y < rad)
                    {
                    if (newidx != i1.x && newidx != i1.y && newidx !=i1.z)
                        {
                        badParticle = true;
                        d_repair[newidx] = newidx;
                        };
                    };
                };//end loop over particles in the given cell
            };
        };// end loop over cells
    if (badParticle)
        {
          d_repair[i1.x] = i1.x;
          d_repair[i1.y] = i1.y;
          d_repair[i1.z] = i1.z;
        };

    return;
    };

//Gives back the number of points that need to be re-triangulated
__global__ void gpu_size_kernel(
                                    int* __restrict__ d_repair,
                                    int Np,
                                    int* __restrict__ Nf
                                    )
    {
      unsigned int kidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (kidx >= Np)
        return;

      int val1=d_repair[kidx];
      int val2=d_repair[kidx+1];
      if(val1<Np && val2==Np)Nf[0]=kidx+1;
    }

//Not used
//The idea was that one could use the triangulation of the previous timestep as the initial trial polygon.
//Unfortunatly this has not been very successful in some test, thus further consideratons are needed
__global__ void gpu_initial_poly_kernel(const double2* __restrict__ d_pt,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              int* __restrict__ d_neighnum,
                                              int* __restrict__ c,
                                              int Ncells,
                                              periodicBoundaries Box,
                                              const int* __restrict__ d_fixlist,
                                              int Nf,
                                              Index2D GPU_idx
                                              )
    {
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;
      unsigned int kidx=d_fixlist[tidx];

      double2 v = d_pt[kidx];
      unsigned int j,i,newidx;
      double2 pt1;
      unsigned int poly_size=d_neighnum[kidx];

      i=0;
      newidx=P_idx[GPU_idx(i, kidx)];
      Box.minDist(d_pt[newidx],v,pt1);
      P[GPU_idx(i,kidx)]=pt1;

      for(i=1; i<poly_size; i++)
        {
          newidx=P_idx[GPU_idx(i, kidx)];
          Box.minDist(d_pt[newidx],v,pt1);

          P[GPU_idx(i,kidx)]=pt1;
          j=i-1;
          if ( ((P[GPU_idx(i,kidx)].y>0) != (P[GPU_idx(j,kidx)].y>0)) && (0 < (P[GPU_idx(j,kidx)].x-P[GPU_idx(i,kidx)].x) * (0-P[GPU_idx(i,kidx)].y) / (P[GPU_idx(j,kidx)].y-P[GPU_idx(i,kidx)].y) + P[GPU_idx(i,kidx)].x) ) c[kidx] = !c[kidx];
        }

      j=poly_size-1;
      i=0;
      if ( ((P[GPU_idx(i,kidx)].y>0) != (P[GPU_idx(j,kidx)].y>0)) && (0 < (P[GPU_idx(j,kidx)].x-P[GPU_idx(i,kidx)].x) * (0-P[GPU_idx(i,kidx)].y) / (P[GPU_idx(j,kidx)].y-P[GPU_idx(i,kidx)].y) + P[GPU_idx(i,kidx)].x) ) c[kidx] = !c[kidx];

      return;
    }

/*
GPU implementatio of the DT. It makes use of a locallity lema described in (doi: 10.1109/ISVD.2012.9). It will only make the repair of the topology in case it is necessary. Steps are detailed as in paper.
*/
//This kernel constructs the initial test polygon.
//Currently it only uses 4 points, one in each quadrant.
//The initial test voronoi cell needs to be valid for the algorithm to work.
//Thus if the search fails, 4 virtual points are used at maximum distance as the starting polygon
__global__ void gpu_voronoi_calc_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              const int* __restrict__ d_fixlist,
                                              int Nf,
                                              Index2D GPU_idx
                                              )
    {
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;
      unsigned int kidx=d_fixlist[tidx];

      int i,j,k;
      double2 pt1,pt2;
      double rr, z1, z2;
      unsigned int poly_size;
      double Lmax=(xsize*boxsize)/2;

      //construct new poly around i
        double2 v = d_pt[kidx];
        poly_size=4;
        d_neighnum[kidx]=4;
        int Quad[4]={0,0,0,0};
        unsigned int find=0;
        int ii=floor(v.x/boxsize);
        int jj=floor(v.y/boxsize);
        int w=0;
        int bin, cx, cy, quad_angle, numberInCell, newidx, pp, m, n, l, g;
        while(find<4)
        {
        //if(w>Lmax)printf("voro_calc: %d, %d\n", kidx, w);
        //go through the shell
	for (i = -w; i <=w; ++i)
          {
            for (j = -w; j <=w; ++j)
            {
              if(i ==-w ||i == w ||j ==-w ||j==w)
              {
                cx = (ii+j)%xsize;
                if (cx <0) cx+=xsize;
                cy = (jj+i)%ysize;
                if (cy <0) cy+=ysize;
                bin = ci(cx,cy);
                //go through the cell

                numberInCell = d_cell_sizes[bin];
                for (pp = 0; pp < numberInCell; ++pp)
                {
                  newidx = d_cell_idx[cli(pp,bin)];
                  if(newidx==kidx)continue;

                  Box.minDist(d_pt[newidx],v,pt1);
                  if(pt1.y>0)
                  {
                    if(pt1.x>0)quad_angle=0;
                    else quad_angle=1;
                  }
                  else
                  {
                    if(pt1.x>0)quad_angle=3;
                    else quad_angle=2;
                  }

                  if(Quad[quad_angle]==0)
                  {
                    //check if it is in quadrant
                    P[GPU_idx(quad_angle, kidx)]=pt1;
                    P_idx[GPU_idx(quad_angle, kidx)]=newidx;
                    Quad[quad_angle]=1;
                    find++;

                    //check if it is convex and self-intersecting
                    if(find==4)
                    {
                      for(m=0; m<poly_size; m++)
                      {
                        n=m+1;
                        if(n>=poly_size)n-=poly_size;
                        Circumcircle(P[GPU_idx(m,kidx)],P[GPU_idx(n,kidx)], pt1, rr);
                        Q[GPU_idx(m,kidx)]=pt1;
                        Q_rad[GPU_idx(m,kidx)]=rr;
                      }

                      for(m=2; m<=poly_size+1; m++)
                      {
                        l=m;
                        n=m-1;
                        k=m-2;
                        if(n>=poly_size)n-=poly_size;
                        if(l>=poly_size)l-=poly_size;
                        Box.minDist(Q[GPU_idx(k, kidx)], Q[GPU_idx(n, kidx)],pt1);
                        Box.minDist(Q[GPU_idx(n, kidx)], Q[GPU_idx(l, kidx)],pt2);
                        z1=pt1.x*pt2.y-pt1.y*pt2.x;

                        //check if it is convex
                        if(m>2 && z1/z2<0)
                        {
                          find-=1;
                          Quad[n]=0;
                          break;
                        }
                        z2=z1;
                        g=m+1;
                        if(g>=poly_size)g-=poly_size;

                        //check if it is self-intersecting
                        if(ccw(Q[GPU_idx(k, kidx)],Q[GPU_idx(l, kidx)],Q[GPU_idx(g, kidx)])!=ccw(Q[GPU_idx(n, kidx)],Q[GPU_idx(l, kidx)],Q[GPU_idx(g, kidx)]) && ccw(Q[GPU_idx(k, kidx)],Q[GPU_idx(n, kidx)],Q[GPU_idx(l, kidx)])!=ccw(Q[GPU_idx(k, kidx)],Q[GPU_idx(n, kidx)],Q[GPU_idx(g, kidx)]))
                        {
                          find-=1;
                          Quad[n]=0;
                          break;
                        }
                      }
                    }//end if find
                  }//end if Quad
                  if(find==4)break;
                }//end pt check
              }//end in if ij 
              if(find==4)break;
            }//end j
            if(find==4)break;
          }//end i
        w++;

        //if the routine was not able to find a valid polygon around the cell 
        //then create 4 virtual points at "infinity" that form a quadrilateral
        if(w>=Lmax)
        {
		double LL=Lmax/1.414213562373095-EPSILON;
                P[GPU_idx(0, kidx)].x=LL;
                P[GPU_idx(0, kidx)].y=LL;
                P[GPU_idx(1, kidx)].x=-LL;
                P[GPU_idx(1, kidx)].y=LL;
                P[GPU_idx(2, kidx)].x=-LL;
                P[GPU_idx(2, kidx)].y=-LL;
                P[GPU_idx(3, kidx)].x=LL;
                P[GPU_idx(3, kidx)].y=-LL;
                for(m=0; m<poly_size; m++)
                {
                   P_idx[GPU_idx(m, kidx)]=-1;
                   n=m+1;
                   if(n>=poly_size)n-=poly_size;
                   Circumcircle(P[GPU_idx(m,kidx)],P[GPU_idx(n,kidx)], pt1, rr);
                   Q[GPU_idx(m,kidx)]=pt1;
                   Q_rad[GPU_idx(m,kidx)]=rr;
                }
                find=4;
        }

        }//end while

      return;
    }

//This kernel updates the initial polygon into the real delaunay one.
//It goes through the same steps as in the paper, using the half plane intersection routine.
//It outputs the complete triangulation per point in CCW order
__global__ void gpu_get_neighbors_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              const int* __restrict__ d_fixlist,
                                              int Nf,
                                              Index2D GPU_idx
                                              )
    {
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;
      unsigned int kidx=d_fixlist[tidx];
      if (kidx >= Ncells)return;

      //int testpart=-506;
      //int side=5;

      //I will reuse most variables
      int Hv[32];//={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
      double2 disp, pt1, pt2, v;
      double rr, xx, yy;
      unsigned int ii, numberInCell, newidx, iii, aa, removed;
      int q, pp, m, w, j, jj, cx, cy, save_j, cc, dd, cell_rad_in, bin, cell_x, cell_y, save,ff;
      unsigned int poly_size=d_neighnum[kidx];
      //unsigned int static_poly_size=poly_size;
      v = d_pt[kidx];
      bool flag=false;
      bool again=false;
      //int cont=0;
      //if(kidx==testpart)printf("\n START ALLLLLLLLLL--------------%d, %f, %f\n", kidx, d_pt[kidx].x, d_pt[kidx].y);

      /*if(kidx==testpart)
      {
        for(jj=0; jj<poly_size; jj++)printf("EDGE--------------%d, %f, %f, %f, %f, %f\n", P_idx[GPU_idx(jj, kidx)],P[GPU_idx(jj, kidx)].x,P[GPU_idx(jj, kidx)].y,Q[GPU_idx(jj, kidx)].x,Q[GPU_idx(jj, kidx)].y,Q_rad[GPU_idx(jj, kidx)]);
      }*/


      for(jj=0; jj<poly_size; jj++)
      {
      //if(kidx==testpart)printf("START--------------%d, %d, %d\n", kidx, jj, poly_size);
      /*if(kidx==testpart)
      {
        for(int oo=0; oo<poly_size; oo++)printf("EDGE--------------%d\n", P_idx[GPU_idx(oo, kidx)]);
      }*/

      for(ff=jj; ff<jj+1; ff++)//search the edges
        {
          pt1=v+Q[GPU_idx(ff,kidx)];
          Box.putInBoxReal(pt1);
          cc = max(0,min(xsize-1,(int)floor(pt1.x/boxsize)));
          dd = max(0,min(ysize-1,(int)floor(pt1.y/boxsize)));
          q = ci(cc,dd);
          //check neighbours of Q's cell inside the circumcircle
          cc = ceil(Q_rad[GPU_idx(ff,kidx)]/boxsize)+1;
          cell_rad_in = min(cc,xsize/2);
          cell_x = q%xsize;
          cell_y = (q - cell_x)/ysize;
          //cont++;
          //if(kidx==testpart )printf("Poly: %d,  %d, %d, %d, %d, %d, %d, %d\n",P_idx[GPU_idx(ff,kidx)],cc,dd,q,cell_rad_in,cell_x,cell_y,xsize);
          //if(cont>10000)printf("START--------------%d, %d, %d, %d\n", kidx, jj, poly_size, cont);
          //cList.getCellNeighbors(cix,wcheck,cns);
          for (cc = -cell_rad_in; cc <= cell_rad_in; ++cc)//check neigh i
            {
              for (dd = -cell_rad_in; dd <=cell_rad_in; ++dd)//check neigh q
                {
                  cx = (cell_x+dd)%xsize;
                  if (cx <0) cx+=xsize;
                  cy = (cell_y+cc)%ysize;
                  if (cy <0) cy+=ysize;
                  //check if there are any points in cellsns, if so do change, otherwise go for next bin
                  bin = ci(cx,cy);
                  numberInCell = d_cell_sizes[bin];
                  //if(kidx==testpart )printf("\nBINS: %d, %d\n",cx, cy);

                  for (aa = 0; aa < numberInCell; ++aa)//check parts in cell
                    {
                      newidx = d_cell_idx[cli(aa,bin)];
                      //6-Compute the half-plane Hv defined by the bissector of v and c, containing c
                          ii=GPU_idx(jj, kidx);
                          iii=GPU_idx((jj+1)%poly_size, kidx);
                          if(newidx==P_idx[ii] || newidx==P_idx[iii] || newidx==kidx)continue;

                          //how far is the point from the circumcircle's center?
                          rr=Q_rad[ii]*Q_rad[ii];
                          Box.minDist(d_pt[newidx], v, disp);
                          Box.minDist(disp,Q[ii],pt1);
                          //if(kidx==testpart )printf("\nIS IN RADIUS?: %d, %d, %d, %f, %f, %d, %d, %f, %f, %f, %f, %f, %f, %f\n",newidx,ff,jj,pt1.x*pt1.x+pt1.y*pt1.y,rr,P_idx[ii],P_idx[iii],disp.x,disp.y,Q[ff].x,Q[ff].y, Q_rad[ff], v.x, v.y);
                          if(pt1.x*pt1.x+pt1.y*pt1.y>rr)continue;
//if(kidx==testpart)printf("\nYES: %d, %d, %d, %f, %f, %d, %d, %f, %f\n",newidx,ff,jj,pt1.x*pt1.x+pt1.y*pt1.y,rr,P_idx[ii],P_idx[iii],disp.x,disp.y);
                          //calculate half-plane bissector
                          if(abs(disp.y)<THRESHOLD)
                          {
                                  yy=disp.y/2+1;
                                  xx=disp.x/2;
                          }
                          else if(abs(disp.x)<THRESHOLD)
                          {
                                  yy=disp.y/2;
                                  xx=disp.x/2+1;
                          }
                          else
                          {
                                  yy=(disp.y*disp.y+disp.x*disp.x)/(2*disp.y);
                                  xx=0;
                          }

                          //7-Q<-Hv intersect Q
                          //8-Update P, based on Q (Algorithm 2)      
                          if((disp.x/2-xx)*(disp.y/2-0)-(disp.y/2-yy)*(disp.x/2-0)>0)cx=0; //which side is v at
                          else cx=1;
                          cy=0; //which side will Q be at
                          j=jj-1;
                          if(j<0)j+=poly_size;
                          m=jj;
                          removed=0;
                          save_j=-1;
                          //see which voronoi temp points fall within the same bisector as cell v
                          for(pp=0; pp<poly_size; pp++)
                            {
                              q=jj-pp;
                              if(q<0)q+=poly_size;

                              if((disp.x/2-xx)*(disp.y/2-Q[GPU_idx(q,kidx)].y)-(disp.y/2-yy)*(disp.x/2-Q[GPU_idx(q, kidx)].x)>0)cy=0;
                              else cy=1;

                              save=(q+1)%poly_size;
                              if(newidx==P_idx[GPU_idx(q, kidx)] || newidx==P_idx[GPU_idx(save,kidx)])cy=cx+1;

                              Hv[q]=cy;
                              if(cy==cx && save_j==-1)save_j=q;
//if(kidx==testpart)printf("\nHALF-PLANE: %d, %d, %d, %f, %f, %f, %f, %f\n",P_idx[GPU_idx(q, kidx)],cx,cy,xx,Q[GPU_idx(q,kidx)].y,yy,Q[GPU_idx(q, kidx)].x,(disp.x/2-xx)*(disp.y/2-Q[GPU_idx(q,kidx)].y)-(disp.y/2-yy)*(disp.x/2-Q[GPU_idx(q, kidx)].x));

                            }
                          if(Hv[jj]==cx)continue;

                          //Remove the voronoi test points on the opposite half sector from the cell v
                          //If more than 1 voronoi test point is removed, then also adjust the delaunay neighbors of v
                          for(w=0; w<poly_size; w++)
                            {
                              q=(save_j+w)%poly_size;
                              cy=Hv[q];
                              if(cy!=cx)
                                {
                                  switch(removed)
                                    {
                                      case 0:
                                        j=q;
                                        m=(j+1)%poly_size;
                                        removed++;
                                      break;
                                      case 1:
                                        m=(m+1)%poly_size;
                                        removed++;
                                      break;
                                      case 2:
                                      for(pp=q; pp<poly_size-1; pp++)
                                        {
                                          Q[GPU_idx(pp,kidx)]=Q[GPU_idx(pp+1,kidx)];
                                          P[GPU_idx(pp,kidx)]=P[GPU_idx(pp+1,kidx)];
                                          Q_rad[GPU_idx(pp,kidx)]=Q_rad[GPU_idx(pp+1,kidx)];
                                          P_idx[GPU_idx(pp,kidx)]=P_idx[GPU_idx(pp+1,kidx)];
                                          Hv[pp]=Hv[pp+1];
                                        }
                                      poly_size--;
                                      if(j>q)j--;
                                      if(save_j>q)save_j--;
                                      m=m%poly_size;
                                      w--;
                                      break;
                                    }
                                }
                              else if(removed>0)break;
                            }
                          if(removed==0)continue;
                          else if(removed==1 && poly_size==32){again=true;continue;}
                          else if(again==true && poly_size<32){again=false;}

//if(kidx==testpart )printf("\nMIDDLE: %d, %d, %d, %d\n",newidx,removed,j,m);
                          //Introduce new (if it exists) delaunay neighbor and new voronoi points
                          Circumcircle(P[GPU_idx(j,kidx)], disp, pt1, xx);
                          Circumcircle(disp, P[GPU_idx(m,kidx)], pt2, yy);
                          if(removed==1)
                            {
                              poly_size++;
                              for(pp=poly_size-2; pp>j; pp--)
                                {
                                  Q[GPU_idx(pp+1,kidx)]=Q[GPU_idx(pp,kidx)];
                                  P[GPU_idx(pp+1,kidx)]=P[GPU_idx(pp,kidx)];
                                  Q_rad[GPU_idx(pp+1,kidx)]=Q_rad[GPU_idx(pp,kidx)];
                                  P_idx[GPU_idx(pp+1,kidx)]=P_idx[GPU_idx(pp,kidx)];
                                }
                            }

                          m=(j+1)%poly_size;
                          Q[GPU_idx(m,kidx)]=pt2;
                          Q_rad[GPU_idx(m,kidx)]=yy;
                          P[GPU_idx(m,kidx)]=disp;
                          P_idx[GPU_idx(m,kidx)]=newidx;

                          Q[GPU_idx(j,kidx)]=pt1;
                          Q_rad[GPU_idx(j,kidx)]=xx;
                          flag=true;
                          break;
                    }//end checking all parts in cell
                  if(flag==true)break;
                }//end cell neighbor check, q
              if(flag==true)break;
            }//end cell neighbor check, i
          if(flag==true){ff=jj-1;flag=false;}
        }//end edge search
      if(again==true){jj--;again=false;printf("\nAGAIN: %d, %d\n",kidx,jj);}
      }

      d_neighnum[kidx]=poly_size;
      /*if(kidx==testpart)printf("END--------------%d\n", kidx);
            if(kidx==testpart)
      {
        for(jj=0; jj<poly_size; jj++)printf("EDGE--------------%d, %f, %f, %f, %f, %f\n", P_idx[GPU_idx(jj, kidx)],P[GPU_idx(jj, kidx)].x,P[GPU_idx(jj, kidx)].y,Q[GPU_idx(jj, kidx)].x,Q[GPU_idx(jj, kidx)].y,Q_rad[GPU_idx(jj, kidx)]);
      }*/

      return;
    }//end function


//Same as before but the balanced version.
//This means that only two edges will be completly updated.
//As in the paper, only the top and bottom triangles are completly triangulated per point
__global__ void gpu_BalancedGetNeighbors_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              const int* __restrict__ d_fixlist,
                                              int Nf,
                                              Index2D GPU_idx
                                              )
    {
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;
      unsigned int kidx=d_fixlist[tidx];
      if (kidx >= Ncells)return;

      //int testpart=-94;
      //int side=5;

      //I will reuse most variables
      int Hv[32];//={-1,-1,-1,-1};
      int saveP[4];
      double2 disp, pt1, pt2, v;
      double rr, xx, yy; 
      unsigned int ii, numberInCell, newidx, iii, aa, removed;
      int q, pp, m, w, j, jj, cx, cy, save_j, cc, dd, cell_rad_in, bin, cell_x, cell_y, save,s; 
      unsigned int poly_size=4;
      v = d_pt[kidx];
      bool flag=false;
      //int cont=0;
      //if(kidx==testpart)printf("\n START ALLLLLLLLLL--------------%d, %f, %f\n", kidx, d_pt[kidx].x, d_pt[kidx].y);

      /*if(kidx==testpart)
      {
        for(jj=0; jj<poly_size; jj++)printf("EDGE--------------%d, %f, %f, %f, %f, %f\n", P_idx[GPU_idx(jj, kidx)],P[GPU_idx(jj, kidx)].x,P[GPU_idx(jj, kidx)].y,Q[GPU_idx(jj, kidx)].x,Q[GPU_idx(jj, kidx)].y,Q_rad[GPU_idx(jj, kidx)]);
      }*/
      
      jj=-1;
      int EdgeComplete=0;
      s=1;
      while(EdgeComplete<4)//search the edges
        {
	  jj=(jj+1)%poly_size;
          ii=GPU_idx(jj, kidx);
          iii=GPU_idx((jj+1)%poly_size, kidx);
          if( !(  ( (P[ii].x>0) != (P[iii].x>0) ) && ( 0 < s*((P[iii].y-P[ii].y) * (0-P[ii].x) / (P[iii].x-P[ii].x) + P[ii].y) ) ) )continue;

      /*if(kidx==testpart)
      {
        for(int oo=0; oo<poly_size; oo++)printf("EDGE--------------%d\n", P_idx[GPU_idx(oo, kidx)]);
      }*/

          pt1=v+Q[GPU_idx(jj,kidx)];
          Box.putInBoxReal(pt1);
          cell_x = max(0,min(xsize-1,(int)floor(pt1.x/boxsize)));
          cell_y = max(0,min(ysize-1,(int)floor(pt1.y/boxsize)));
          q = ci(cell_x,cell_y);
          //check neighbours of Q's cell inside the circumcircle
          cc = ceil(Q_rad[GPU_idx(jj,kidx)]/boxsize)+1;
          cell_rad_in = min(cc,xsize/2);

          //cont++;
          //if(kidx==testpart )printf("Poly: %d,  %d, %d, %d, %d, %d, %d, %d\n",P_idx[GPU_idx(jj,kidx)],cc,dd,q,cell_rad_in,cell_x,cell_y,xsize);
          //if(cont>10000)printf("START--------------%d, %d, %d, %d\n", kidx, jj, poly_size, cont);

          //cList.getCellNeighbors(cix,wcheck,cns);
          for (cc = -cell_rad_in; cc <= cell_rad_in; ++cc)//check neigh i
            {
              for (dd = -cell_rad_in; dd <=cell_rad_in; ++dd)//check neigh q
                {
                  cx = (cell_x+dd)%xsize;
                  if (cx <0) cx+=xsize;
                  cy = (cell_y+cc)%ysize;
                  if (cy <0) cy+=ysize;
                  //check if there are any points in cellsns, if so do change, otherwise go for next bin
                  bin = ci(cx,cy);
                  numberInCell = d_cell_sizes[bin];
                  //if(kidx==testpart )printf("\nBINS: %d, %d\n",cx, cy);

                  for (aa = 0; aa < numberInCell; ++aa)//check parts in cell
                    {
                      newidx = d_cell_idx[cli(aa,bin)];
                      //6-Compute the half-plane Hv defined by the bissector of v and c, containing c
                      if(newidx==P_idx[ii] || newidx==P_idx[iii] || newidx==kidx)continue;

                      //how far is the point from the circumcircle's center?
                      rr=Q_rad[ii]*Q_rad[ii];
                      Box.minDist(d_pt[newidx], v, disp);
                      Box.minDist(disp,Q[ii],pt1);
                      //if(kidx==testpart )printf("\nIS IN RADIUS?: %d, %d, %d, %f, %f, %d, %d, %f, %f, %f, %f, %f, %f, %f\n",newidx,ff,jj,pt1.x*pt1.x+pt1.y*pt1.y,rr,P_idx[ii],P_idx[iii],disp.x,disp.y,Q[ff].x,Q[ff].y, Q_rad[ff], v.x, v.y);
                      if(pt1.x*pt1.x+pt1.y*pt1.y>rr)continue;
//if(kidx==testpart)printf("\nYES: %d, %d, %f, %f, %d, %d, %f, %f\n",newidx,jj,pt1.x*pt1.x+pt1.y*pt1.y,rr,P_idx[ii],P_idx[iii],disp.x,disp.y);

                      //calculate half-plane bissector
                      if(abs(disp.y)<THRESHOLD)
                      {
                        yy=disp.y/2+1;
                        xx=disp.x/2;
		      }
                      else if(abs(disp.x)<THRESHOLD)
                      {
                        yy=disp.y/2;
                        xx=disp.x/2+1;
		      }
                      else
		      {
                        yy=(disp.y*disp.y+disp.x*disp.x)/(2*disp.y);
                        xx=0;
                      }

                      //7-Q<-Hv intersect Q
                      //8-Update P, based on Q (Algorithm 2)      
                      if((disp.x/2-xx)*(disp.y/2-0)-(disp.y/2-yy)*(disp.x/2-0)>0)cx=0; //which side is v at
                      else cx=1;
                      cy=0; //which side will Q be at
                      j=jj-1;
                      if(j<0)j+=poly_size;
                      m=jj;
                      removed=0;
                      save_j=-1;
                      //see which voronoi temp points fall within the same bisector as cell v
                      for(pp=0; pp<poly_size; pp++)
                      {
                        q=jj-pp;
                        if(q<0)q+=poly_size;
                        if((disp.x/2-xx)*(disp.y/2-Q[GPU_idx(q,kidx)].y)-(disp.y/2-yy)*(disp.x/2-Q[GPU_idx(q, kidx)].x)>0)cy=0;
                        else cy=1;

                        save=(q+1)%poly_size;
			//if a point already in the poly falls within the CC, then force remove it and add it later
                        if(newidx==P_idx[GPU_idx(q, kidx)] || newidx==P_idx[GPU_idx(save,kidx)])cy=cx+1;
                        Hv[q]=cy;
                        if(cy==cx && save_j==-1)save_j=q;
//if(kidx==testpart)printf("\nHALF-PLANE: %d, %d, %d, %f, %f, %f, %f, %f\n",P_idx[GPU_idx(q, kidx)],cx,cy,xx,Q[GPU_idx(q,kidx)].y,yy,Q[GPU_idx(q, kidx)].x,(disp.x/2-xx)*(disp.y/2-Q[GPU_idx(q,kidx)].y)-(disp.y/2-yy)*(disp.x/2-Q[GPU_idx(q, kidx)].x));
                       }
                       if(Hv[jj]==cx)continue;

                       //Remove the voronoi test points on the opposite half sector from the cell v
                       //If more than 1 voronoi test point is removed, then also adjust the delaunay neighbors of v
                       for(w=0; w<poly_size; w++)
                       {
                         q=(save_j+w)%poly_size;
                         cy=Hv[q];
                         if(cy!=cx)
                         {
                           switch(removed)
                           {
                             case 0:
                               j=q;
                               m=(j+1)%poly_size;
                               removed++;
                               break;
                             case 1:
                               m=(m+1)%poly_size;
                               removed++;
                               break;
                             case 2:
                               for(pp=q; pp<poly_size-1; pp++)
                               {
                                 Q[GPU_idx(pp,kidx)]=Q[GPU_idx(pp+1,kidx)];
                                 P[GPU_idx(pp,kidx)]=P[GPU_idx(pp+1,kidx)];
                                 Q_rad[GPU_idx(pp,kidx)]=Q_rad[GPU_idx(pp+1,kidx)];
                                 P_idx[GPU_idx(pp,kidx)]=P_idx[GPU_idx(pp+1,kidx)];
                                 Hv[pp]=Hv[pp+1];
                               }
                               poly_size--;
                               if(j>q)j--;
                               if(save_j>q)save_j--;
                               m=m%poly_size;
                               w--;
                               break;
                             }
                           }
                           else if(removed>0)break;
                         }
                         if(removed==0)continue;

//if(kidx==testpart )printf("\nMIDDLE: %d, %d, %d, %d\n",newidx,removed,j,m);
                         //Introduce new (if it exists) delaunay neighbor and new voronoi points
                         Circumcircle(P[GPU_idx(j,kidx)], disp, pt1, xx);
                         Circumcircle(disp, P[GPU_idx(m,kidx)], pt2, yy);
                         if(removed==1)
                         {
                           poly_size++;
                           for(pp=poly_size-2; pp>j; pp--)
                             {
                               Q[GPU_idx(pp+1,kidx)]=Q[GPU_idx(pp,kidx)];
                               P[GPU_idx(pp+1,kidx)]=P[GPU_idx(pp,kidx)];
                               Q_rad[GPU_idx(pp+1,kidx)]=Q_rad[GPU_idx(pp,kidx)];
                               P_idx[GPU_idx(pp+1,kidx)]=P_idx[GPU_idx(pp,kidx)];
                             }
                         }

                         m=(j+1)%poly_size;
                         Q[GPU_idx(m,kidx)]=pt2;
                         Q_rad[GPU_idx(m,kidx)]=yy;
                         P[GPU_idx(m,kidx)]=disp;
                         P_idx[GPU_idx(m,kidx)]=newidx;

                         Q[GPU_idx(j,kidx)]=pt1;
                         Q_rad[GPU_idx(j,kidx)]=xx;
                         flag=true;

		         //stay in the triangle
                         if(EdgeComplete==0)
                         {
                           if(disp.x>0)jj=m;
                           else jj=j;
                         }
                         else if(EdgeComplete==2)
                         { 
                           if(disp.x<0)jj=m;
                           else jj=j;
                         }

                         break;
                    }//end checking all parts in cell
                  if(flag==true)break;
                }//end cell neighbor check, q
              if(flag==true)break;
            }//end cell neighbor check, i
          if(flag==true){flag=false;jj--;}
	  else
	  {
	    s*=(-1);
	    saveP[EdgeComplete]=P_idx[GPU_idx(jj,kidx)];
	    saveP[EdgeComplete+1]=P_idx[GPU_idx((jj+1)%poly_size,kidx)];
	    //if(kidx==testpart)printf("ENTRY--------------%d, %d, %d, %d, %d\n", jj, saveP[EdgeComplete], P_idx[GPU_idx(jj,kidx)], saveP[EdgeComplete+1], P_idx[GPU_idx(jj+1,kidx)]);
	    EdgeComplete+=2;
	  }
        }//end edge search

      /*if(kidx==testpart)printf("END--------------%d\n", kidx);
            if(kidx==testpart)
      {
        for(jj=0; jj<poly_size; jj++)printf("EDGE--------------%d, %d, %f, %f, %f, %f, %f\n", P_idx[GPU_idx(jj, kidx)],saveP[jj], P[GPU_idx(jj, kidx)].x,P[GPU_idx(jj, kidx)].y,Q[GPU_idx(jj, kidx)].x,Q[GPU_idx(jj, kidx)].y,Q_rad[GPU_idx(jj, kidx)]);
      }*/

      for(ii=0; ii<4; ii++)P_idx[GPU_idx(ii,kidx)]=saveP[ii];
      return;
    };//end function

//Kernel that tries to organize the output of the balanced functions into the required data structures for the cellGPU code to use.
//This means taking all the top and bottom triangles known to each point and filling out the remainder delaunay neighbors with only that information.
//The final data structure needs to have the correct number of neighbors as well as all neighbors per point ordered in CCW
//Currently it is not working properly
__global__ void gpu_OrganizeDelTriangulation_kernel(
                    int* __restrict__ d_neighnum,
                    int Ncells,
                    const int* __restrict__ d_repair,
                    int size_fixlist,
                    int* __restrict__ d_Tri,
		    const int* __restrict__ P_idx,
		    const int* __restrict__ neighs,
		    Index2D GPU_idx
		    )
{
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= size_fixlist)return;
      unsigned const int kidx=d_repair[tidx];
      if (kidx >= Ncells)return;

      int test=145;

      int ii, jj, nmax, curr, cellidx, idx, cell, kk, pp, nn, save, start;
      bool flag;
      d_neighnum[kidx]=0;
      bool followThrough=false;

      if(kidx==test)printf("START---------------------------------------------------------------------------%d\n", kidx);
      if(kidx==test){for(jj=0; jj<neighs[kidx]; jj++)printf("INITIAL POLY: %d\n", P_idx[GPU_idx(jj,kidx)]);}

      curr=0;
      nn=0;
      //organize the triangulation array
      while(nn<4)
      {
      if(kidx==test)printf("\nIN--------------------------------------------------------------------------------%d, %d\n", nn, d_neighnum[kidx]);

      flag=true;
      followThrough=true;

      switch(curr)
      {
	      case 0:
		      ii=d_neighnum[kidx];
          	      d_Tri[GPU_idx(ii,kidx)]=P_idx[GPU_idx(curr,kidx)];
          	      d_neighnum[kidx]++;
        	      ii=d_neighnum[kidx];
          	      d_Tri[GPU_idx(ii,kidx)]=P_idx[GPU_idx(curr+1,kidx)];
          	      d_neighnum[kidx]++;
		      save=2;
		      curr=1;
		      if(P_idx[GPU_idx(curr,kidx)]==P_idx[GPU_idx(curr+1,kidx)]){flag=false;followThrough=false;}
		      break;
	      case 1:
		      curr=2;
        	      ii=d_neighnum[kidx];
        	      if(P_idx[GPU_idx(curr,kidx)]!=P_idx[GPU_idx(curr-1,kidx)])
        	      {
          	        d_Tri[GPU_idx(ii,kidx)]=P_idx[GPU_idx(curr,kidx)];
                        d_neighnum[kidx]++;
                      }
        	      else {flag=false;followThrough=false;};
		      save=ii;
        	      ii=d_neighnum[kidx];
        	      if(P_idx[GPU_idx(curr+1,kidx)]!=P_idx[GPU_idx(0,kidx)])
        		{
          		  d_Tri[GPU_idx(ii,kidx)]=P_idx[GPU_idx(curr+1,kidx)];
          		  d_neighnum[kidx]++;
        		}
        	      else nn++;
		      break;
	      case 2:
		      curr=3;
		      if(P_idx[GPU_idx(curr,kidx)]==P_idx[GPU_idx(0,kidx)]){flag=false;followThrough=false;}
		      save=d_neighnum[kidx];
		      break;
	      case 3:
		      curr=0;
		      save=d_neighnum[kidx];
		      break;
      }

      idx=curr;
      cell=kidx;
      //go to the cell's neigh and see if they share another different one
      while(flag==true)
      {
        cellidx=P_idx[GPU_idx(idx, cell)];
        nmax=neighs[cellidx];
	flag=false;
	if(kidx==test)printf("NEXT---------------%d, %d, %d, %d\n", cellidx, nmax, curr, curr%2);
        if(kidx==test)printf("Poly: ");
	if(kidx==test){for(jj=0; jj<nmax; jj++)printf("%d, ", P_idx[GPU_idx(jj,cellidx)]);}
	if(kidx==test)printf("\n");

        for(jj=0; jj<nmax; jj++)
        {
	  //if they share another neigh add it to cell kidx	
          if(P_idx[GPU_idx(jj,cellidx)]==kidx)
          {
	    if(curr%2==0)
	    {
	      pp=(jj+1)%nmax;
              if(P_idx[GPU_idx(pp,cellidx)]==P_idx[GPU_idx(jj,cellidx)])pp=(jj+2)%nmax;
	    }
	    else
	    {
	    pp=jj-1;
	    if(pp<0)pp+=nmax; 
	    if(P_idx[GPU_idx(pp,cellidx)]==P_idx[GPU_idx(jj,cellidx)])pp=jj-2;
	    if(pp<0)pp+=nmax;
	    }
            if(kidx==test)printf("FOUND---------------%d, %d, %d, %d, %d, %d\n", jj, pp, curr%2, d_Tri[GPU_idx(save%d_neighnum[kidx],kidx)], d_Tri[GPU_idx(save-1,kidx)], P_idx[GPU_idx(pp,cellidx)]);
	    //unless it is already there...
	    if(curr%2==0 && P_idx[GPU_idx(pp,cellidx)]==d_Tri[GPU_idx(save-1,kidx)]){followThrough=false; break;}
	    else if(curr%2!=0 && P_idx[GPU_idx(pp,cellidx)]==d_Tri[GPU_idx(save%d_neighnum[kidx],kidx)]){followThrough=false; break;}
	    //or if it is one of the initial four points already triangulated... 
	    if(P_idx[GPU_idx(pp,cellidx)]==P_idx[GPU_idx(0,kidx)] || P_idx[GPU_idx(pp,cellidx)]==P_idx[GPU_idx(1,kidx)] || P_idx[GPU_idx(pp,cellidx)]==P_idx[GPU_idx(2,kidx)] || P_idx[GPU_idx(pp,cellidx)]==P_idx[GPU_idx(3,kidx)]){followThrough=false; break;}
	    //if all is good then add that cell to kidx
	    flag=true;
	    idx=pp;
	    cell=cellidx;
	    if(kidx==test)printf("PASSED---------------%d, %d, %d, %d\n", idx, cell, save, P_idx[GPU_idx(idx, cell)]);
            if(curr%2!=0)
	    {
	      d_Tri[GPU_idx(save,kidx)]=P_idx[GPU_idx(idx, cell)];
	      save++;
	    }
	    else
	    {
	      for(kk=d_neighnum[kidx]; kk>save; kk--)d_Tri[GPU_idx(kk,kidx)]=d_Tri[GPU_idx(kk-1,kidx)];
	      d_Tri[GPU_idx(save,kidx)]=P_idx[GPU_idx(idx, cell)];
	    }
            d_neighnum[kidx]++;
	    if(kidx==test)printf("END---------------%d, %d\n", save, curr);
            if(kidx==test)printf("Poly: ");
            if(kidx==test){for(ii=0; ii<d_neighnum[kidx]; ii++)printf("%d, ", d_Tri[GPU_idx(ii,kidx)]);}
            if(kidx==test)printf("\n\n");

            break;
          }//end if found new neigh
        }//end for in new cell
      }//end while it is finding new neighs

      if(followThrough==true)
      {
	      if(curr%2==0)
	      {
		      cell=cellidx;
		      pp=neighs[cell];
		      start=0;
		      idx=1;

	      }
	      else
	      {
		      cell=cellidx;
                      pp=-1;
		      start=neighs[cell]-1;
		      idx=-1;
	      }
      }
      else
      {
	      cell=0;
	      pp=0;
	      start=0;
	      idx=0;
      }
      if(kidx==test)printf("\nFollowThrough--------------------------------------------%d, %d, %d\n", kidx, pp, cell);
      for(ii=start; ii!=pp; ii=ii+idx) 
      {
        cellidx=P_idx[GPU_idx(ii, cell)];
        nmax=neighs[cellidx];
        if(kidx==test)printf("NEXT---------------%d, %d, %d, %d\n", cellidx, nmax, curr, curr%2);
        if(kidx==test)printf("Poly: ");
        if(kidx==test){for(jj=0; jj<nmax; jj++)printf("%d, ", P_idx[GPU_idx(jj,cellidx)]);}
        if(kidx==test)printf("\n");

        for(jj=0; jj<nmax; jj++)
        {
	  if(cellidx==kidx)break;
          if(curr%2!=0 && (cellidx==d_Tri[GPU_idx(save-2,kidx)] || cellidx==d_Tri[GPU_idx(save-3,kidx)]))break;
	  if(curr%2==0 && (cellidx==d_Tri[GPU_idx(save-1,kidx)] || cellidx==d_Tri[GPU_idx((save+1)%d_neighnum[kidx],kidx)]))break;
          if(cellidx==P_idx[GPU_idx(0,kidx)] || cellidx==P_idx[GPU_idx(1,kidx)] || cellidx==P_idx[GPU_idx(2,kidx)] || cellidx==P_idx[GPU_idx(3,kidx)])break;

          //if they share another neigh add it to cell kidx     
          if(P_idx[GPU_idx(jj,cellidx)]==kidx)
          {
            //if all is good then add that cell to kidx
            cell=cellidx;
	    //ii=-1;
            if(kidx==test)printf("PASSED---------------%d, %d, %d, %d\n", pp, cell, save, cellidx);
            if(curr%2!=0)
            {
	      ii=neighs[cell]-1-idx;
              d_Tri[GPU_idx(save,kidx)]=cellidx;
              save++;
            }
            else
            {
	      ii=0-idx;
	      pp=neighs[cell];
              for(kk=d_neighnum[kidx]; kk>save; kk--)d_Tri[GPU_idx(kk,kidx)]=d_Tri[GPU_idx(kk-1,kidx)];
              d_Tri[GPU_idx(save,kidx)]=cellidx;
            }
            d_neighnum[kidx]++;
	    if(d_neighnum[kidx]>15){ii=pp;break;}
            if(kidx==test)printf("END---------------%d, %d, %d, %d\n", save, curr, kidx, d_neighnum[kidx]);
            if(kidx==test)printf("Poly: ");
            if(kidx==test){for(int aa=0; aa<d_neighnum[kidx]; aa++)printf("%d, ", d_Tri[GPU_idx(aa,kidx)]);}
            if(kidx==test)printf("\n\n");

            break;
          }//end if found new neigh
        }//end for in new cell
      }//end while it is finding new neighs

      nn++;
      }//end while through all the P_idx neighs
      if(kidx==test){for(jj=0; jj<d_neighnum[kidx]; jj++)printf("FINAL POLY: %d\n", d_Tri[GPU_idx(jj,kidx)]);}
}


//Kernel to set points from a supplied GPUArray
__global__ void gpu_setPoints_kernel(double2 *hp,
                   double2 *d_pts,
                   int *d_repair,
                   int Nf)
{
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;

      d_pts[tidx].x=hp[tidx].x;
      d_pts[tidx].y=hp[tidx].y;
      d_repair[tidx]=Nf;
      if(tidx==1)d_repair[Nf]=Nf;
}

//Kernel to set circumcircles from a supplied GPUArray
__global__ void gpu_setCircumcenters_kernel(int3 *hp,
                   int3 *d_ccs,
                   int Nf)
{
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;

      d_ccs[tidx].x=hp[tidx].x;
      d_ccs[tidx].y=hp[tidx].y;
      d_ccs[tidx].z=hp[tidx].z;
}

//Kernel that organizes the repair array to be triangulated
__global__ void gpu_global_repair_kernel(int *d_repair,
                   int Nf)
{
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;

      d_repair[tidx]=tidx;
}

//Kernel that organizes the repair array to be triangulated
__global__ void gpu_setRepair_kernel(int *hp,
                   int *d_rep,
                   int Nf)
{
      unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (tidx >= Nf)return;

      d_rep[tidx]=hp[tidx];
}

//Kernel that organizes the repair array to be triangulated
__global__ void gpu_Balanced_repair_kernel(
                                    int* __restrict__ d_repair,
                                    int Ncells,
                                    int* __restrict__ Nf,
				    const int* __restrict__ d_Tri,
				    const int* __restrict__ d_neighnum,
				    int* __restrict__ P_idx,
				    int* __restrict__ neighs,
				    Index2D GPU_idx
                                    )
    {
      unsigned int kidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (kidx >= Ncells)return;

      int val1=d_repair[kidx];
      int val2=d_repair[kidx+1];
      if(val1<Ncells && val2==Ncells)Nf[0]=kidx+1;

      neighs[kidx]=d_neighnum[kidx];
      for(int i=0; i<d_neighnum[kidx]; i++)P_idx[GPU_idx(i,kidx)]=d_Tri[GPU_idx(i,kidx)];
    }

/////////////////////////////////////////////////////////////
//////
//////			Kernel Calls
//////
/////////////////////////////////////////////////////////////


bool gpu_Balanced_repair(int *d_repair, 
                         int Ncells, 
                         int *Nf,
                         int *d_Tri,
                         int *d_neighnum,
                         int *P_idx,
                         int *neighs,
			 Index2D GPU_idx
			 )
{
    unsigned int block_size = 128;
    if (Ncells < 128) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;

    unsigned int N=Ncells+1;
    thrust::sort(thrust::device,d_repair, d_repair + N);

    gpu_Balanced_repair_kernel<<<nblocks,block_size>>>(
                    d_repair,
                    Ncells,
                    Nf,
		    d_Tri,
		    d_neighnum,
		    P_idx,
		    neighs,
		    GPU_idx
                    );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
}

bool gpu_setRepair(int *hp,
                   int *d_rep,
                   int Nf)
{
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_setRepair_kernel<<<nblocks,block_size>>>(
                             hp,
                             d_rep,
                             Nf);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
}

bool gpu_global_repair(int *d_repair, 
		       int Nf)
{
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_global_repair_kernel<<<nblocks,block_size>>>(
                             d_repair,
                             Nf);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
}

bool gpu_setPoints(double2 *hp, 
		   double2 *d_pts, 
		   int *d_repair, 
		   int Nf)
{
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_setPoints_kernel<<<nblocks,block_size>>>(
                             hp,
                             d_pts, 
                             d_repair, 
                             Nf);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
}

bool gpu_setCircumcenters(int3 *hp,
                          int3 *d_ccs,
                          int Nf)
{
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_setCircumcenters_kernel<<<nblocks,block_size>>>(
                             hp,
                             d_ccs,
                             Nf);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
}

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
                             )
    {
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_initial_poly_kernel<<<nblocks,block_size>>>(
                             d_pt,
                             P_idx,
                             P,
                             d_neighnum,
                             c,
                             Ncells,
                             Box,
                             d_fixlist,
                             Nf,
                             GPU_idx);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


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
                      )
    {
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_voronoi_calc_kernel<<<nblocks,block_size>>>(
                      d_pt,
                      d_cell_sizes,
                      d_cell_idx,
                      P_idx,
                      P,
                      Q,
                      Q_rad,
                      d_neighnum,
                      Ncells,
                      xsize,
                      ysize,
                      boxsize,
                      Box,
                      ci,
                      cli,
                      d_fixlist,
                      Nf,
                      GPU_idx
                      );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

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
                      )
    {
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_get_neighbors_kernel<<<nblocks,block_size>>>(
                      d_pt,
                      d_cell_sizes,
                      d_cell_idx,
                      P_idx,
                      P,
                      Q,
                      Q_rad,
                      d_neighnum,
                      Ncells,
                      xsize,
                      ysize,
                      boxsize,
                      Box,
                      ci,
                      cli,
                      d_fixlist,
                      Nf,
                      GPU_idx
                      );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

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
                      )
    {
    unsigned int block_size = 128;
    if (Nf < 128) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    gpu_BalancedGetNeighbors_kernel<<<nblocks,block_size>>>(
                      d_pt,
                      d_cell_sizes,
                      d_cell_idx,
                      P_idx,
                      P,
                      Q,
                      Q_rad,
                      Ncells,
                      xsize,
                      ysize,
                      boxsize,
                      Box,
                      ci,
                      cli,
                      d_fixlist,
                      Nf,
                      GPU_idx
                      );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_OrganizeDelTriangulation(int *d_neighnum,
                   int Ncells,
                   int *d_repair,
                   int size_fixlist,
                   int *d_Tri,
		   int *P_idx,
		   int *neighs,
		   Index2D GPU_idx
                 )
    {
    unsigned int block_size = 128;
    if (Ncells < 128) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;

    gpu_OrganizeDelTriangulation_kernel<<<nblocks,block_size>>>(
		    d_neighnum,
                    Ncells,
                    d_repair,
                    size_fixlist,
                    d_Tri,
		    P_idx,
		    neighs,
		    GPU_idx
                    );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_build_repair(int* d_repair,
                 int Np,
                 int* Nf
                 )
    {
    unsigned int block_size = 128;
    if (Np < 128) block_size = 32;
    unsigned int nblocks  = Np/block_size + 1;

    unsigned int N=Np+1;
    thrust::sort(thrust::device,d_repair, d_repair + N);

    gpu_size_kernel<<<nblocks,block_size>>>(
                    d_repair,
                    Np,
                    Nf
                    );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!call the kernel to test every circumcenter to see if it's empty
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
                            )
    {
    unsigned int block_size = 128;
    if (Nccs < 128) block_size = 32;
    unsigned int nblocks  = Nccs/block_size + 1;

    gpu_test_circumcenters_kernel<<<nblocks,block_size>>>(
                            d_repair,
                            d_ccs,
                            d_pt,
                            d_cell_sizes,
                            d_idx,
                            Nccs,
                            xsize,
                            ysize,
                            boxsize,
                            Box,
                            ci,
                            cli
                            );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
